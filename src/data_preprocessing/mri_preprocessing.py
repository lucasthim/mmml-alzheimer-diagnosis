# %%
import os
from pathlib import Path
import sys
import argparse
import time
import datetime
import functools
import multiprocessing as mp

import numpy as np
import pandas as pd
import nibabel as nib
# Linux GPU fix: preload cu12 libcusolver.so.11 before TF imports (no-op off Linux).
# See src/_cuda_preload.py.
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
try:
    from _cuda_preload import preload_cusolver
    preload_cusolver()
except Exception:
    pass
# Multi-worker GPU fix: when MRI_FORCE_CPU=1 (set by the parent for spawned workers),
# hide the GPU BEFORE TensorFlow imports so each worker's TF runs on CPU. This avoids
# every worker's TF reserving the whole GPU's VRAM (N processes x ~all-of-24GB -> the
# GPU saturates and everything stalls). Skull stripping is ~0.8s on CPU anyway; the real
# cost is CPU-bound ANTs registration (~80% of per-image time), which parallelizes across
# workers. Must be set before `import tensorflow` to take effect.
if os.environ.get('MRI_FORCE_CPU') == '1':
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Import TensorFlow/deepbrain before ants (ITK): both ship an OpenMP runtime and, on macOS,
# if ITK's initializes first, TF's session.run deadlocks during skull stripping.
import tensorflow as tf
from deepbrain import Extractor
import ants


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'utils'))
from utils.utils import create_reference_table, list_available_images, create_file_name_from_path
from base_mri import check_mri_integrity, save_mri, load_mri, set_env_variables
from deepbrain_skull_strip import deep_brain_skull_stripping
from antspy_registration import register_image_with_atlas
from mri_crop import crop_mri_at_center
from mri_standardize import clip_and_normalize_mri
# from mri_label import label_image_files

def _limit_threads_in_worker(threads):
    '''
    Pool initializer: cap the per-worker thread count so N workers x threads
    stays near the physical core count instead of each worker grabbing every core.

    ITK/ANTs reads ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS per operation at runtime,
    so setting it here (after fork, before the worker's first registration) works.
    TensorFlow/deepbrain reads OMP_NUM_THREADS and the intra/inter-op env vars when
    it builds its session, which happens lazily on the worker's first ext.run().
    '''
    threads = str(int(threads))
    os.environ['ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS'] = threads
    os.environ['OMP_NUM_THREADS'] = threads
    os.environ['TF_NUM_INTRAOP_THREADS'] = threads
    os.environ['TF_NUM_INTEROP_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = threads
    os.environ['MKL_NUM_THREADS'] = threads


def _process_one_image(job, output_path, box, skip_skull_stripping, total):
    '''
    Preprocess a single MRI end-to-end and save it. Module-level (picklable) so it
    can run inside a multiprocessing Pool. `job` is (index, image_path, output_name).

    Returns (index, image_path, saved_bool) for progress accounting. Exceptions are
    caught and reported rather than killing the whole pool, so one bad scan doesn't
    abort a multi-hour batch.
    '''
    ii, image_path, output_name = job
    start_img = time.time()
    try:
        input_image = load_mri(path=image_path)
        print(f"\nProcessing image ({ii+1}/{total}): {image_path}")

        standardized_image = clip_and_normalize_mri(input_image)
        registered_image = register_image_with_atlas(standardized_image)

        if not skip_skull_stripping:
            stripped_image = deep_brain_skull_stripping(image=registered_image, probability=0.5, output_as_array=False)
        else:
            stripped_image = registered_image

        cropped_image = crop_mri_at_center(image=stripped_image, cropping_box=box)

        if check_mri_integrity(cropped_image):
            name = output_name if output_name is not None else create_file_name_from_path(image_path)
            save_mri(image=cropped_image, output_path=output_path, name=name, file_format='.nii.gz')
            saved = True
        else:
            print(f"Skipping image ({ii+1}/{total}) because skull stripping failed: {image_path}")
            saved = False
    except Exception as e:
        print(f"ERROR on image ({ii+1}/{total}) {image_path}: {type(e).__name__}: {e}")
        saved = False

    print('Process for image (%d/%d) took %.2f sec\n' % (ii+1, total, time.time() - start_img))
    return ii, image_path, saved


def execute_preprocessing(input_path = None,
                          output_path = None,
                          images_to_process = None,
                          image_names = None,
                          box = 100,
                          skip = 0,
                          limit = 0,
                          mri_reference_path = None,
                          skip_skull_stripping=False,
                          workers = 1,
                          threads_per_worker = None):

    '''
    MRI Preprocessing pipeline.

    Main steps:

    - MRI standardization

    - MRI Registration

    - MRI Skull Stripping

    - MRI Cropping at 100x100x100

    Parameters
    ----------

    input_path: path where raw MRIs are located.

    output_path: path to save preprocessed MRIs.

    images_to_process: custom list of image paths to preprocess. Each entry may be a
        .nii/.nii.gz file or a DICOM series folder (read via ants.dicom_read).

    image_names: optional list (parallel to images_to_process) of output file names,
        without extension. When given, output <i> is saved as <name>.nii.gz; otherwise
        the name is derived from the input path via create_file_name_from_path. Use this
        for DICOM series, whose input path is an I<id> folder with no ADNI subject token.

    skip: amount of files to skip when executing preprocessing. This is to be used when reprocessing a batch of files that failed during execution.

    limit: max amount of files to process when executing preprocessing. This is to be used when reprocessing a batch of files that failed during execution.
    
    Example
    ----------
    
    python mri_preprocessing.py --input "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/raw/ADNI" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/mri/preprocessed/20210402" --skip 0
        
    '''   
    
    set_env_variables()
    start = time.time()

    if images_to_process is None:
        images_to_process,_,_ = list_available_images(input_path)
    print('------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting pre-processing (Labeling + Standardizing + Registration + Skull Stripping + Cropping) for {len(images_to_process)} images. This might take a while... =)")
    print('------------------------------------------------------------------------------------------------------------------------')

    if skip > 0 and limit > 0:
        images_to_process = images_to_process[skip:limit]
        if image_names is not None: image_names = image_names[skip:limit]
        print(f"Processing from  image {skip} to image {limit}.")

    elif skip > 0:
        images_to_process = images_to_process[skip:]
        if image_names is not None: image_names = image_names[skip:]
        print(f"Processing from image {skip}.")

    elif limit > 0:
        images_to_process = images_to_process[:limit]
        if image_names is not None: image_names = image_names[:limit]
        print(f"Processing up to image {limit}.")
    
    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)

    total = len(images_to_process)
    # One job per image: (index, input_path, output_name-or-None).
    jobs = [
        (ii, image_path, (image_names[ii] if image_names is not None else None))
        for ii, image_path in enumerate(images_to_process)
    ]
    worker = functools.partial(
        _process_one_image,
        output_path=output_path,
        box=box,
        skip_skull_stripping=skip_skull_stripping,
        total=total,
    )

    if workers is None or workers <= 1:
        # Serial path (unchanged behavior).
        for job in jobs:
            worker(job)
    else:
        # Parallel path: N worker processes, each capped so N x threads ~= cores.
        if threads_per_worker is None:
            cpu = os.cpu_count() or 1
            threads_per_worker = max(1, cpu // workers)
        print(f"Running preprocessing with {workers} workers x {threads_per_worker} threads each "
              f"(~{workers * threads_per_worker} threads over {os.cpu_count()} cores). "
              f"Workers run skull-strip on CPU (GPU hidden) to avoid VRAM contention.")
        ctx = mp.get_context('spawn')  # fresh interpreter per worker: thread caps + CPU-only take effect before ants/TF init
        # Children inherit os.environ; MRI_FORCE_CPU=1 makes each worker's TF hide the GPU
        # (set at module top, before `import tensorflow`), so N workers don't fight over VRAM.
        os.environ['MRI_FORCE_CPU'] = '1'
        with ctx.Pool(processes=workers,
                      initializer=_limit_threads_in_worker,
                      initargs=(threads_per_worker,)) as pool:
            done = 0
            # imap_unordered streams results back so progress prints as images finish, in any order.
            for ii, image_path, saved in pool.imap_unordered(worker, jobs):
                done += 1
                if done % 25 == 0 or done == total:
                    print(f"[progress] {done}/{total} images finished.")

    print("Creating new reference image table for preprocessed images...")
    generate_metadata_for_preprocessed_images(output_path,mri_reference_path)
    
    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All images pre processed! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def generate_metadata_for_preprocessed_images(output_path,mri_reference_path):
    preprocessed_images,_,_ = list_available_images(output_path,file_format='.nii.gz',verbose=0)
    create_reference_table(preprocessed_images,output_path = output_path,previous_reference_file_path=mri_reference_path)
    # label_image_files(preprocessed_images,file_format='.nii.gz')

def build_image_list_from_reference(reference_csv_path, adnimerge_path=None):
    '''
    Build the list of raw-MRI (input_path, output_name) pairs to preprocess from a
    reference CSV instead of sweeping a directory.

    Expects the DOWNLOAD_RAW_MRI.csv schema (scripts/list_raw_mri.py):
    SUBJECT, IMAGE_DATA_ID, IMAGE_NAME, FORMAT, N_FILES, PATH — where PATH is the
    I<id> scan folder and IMAGE_NAME is a representative file inside it.

    Input path per row:
      - .nii scan  -> PATH/IMAGE_NAME (the single-file volume).
      - DICOM series (FORMAT == 'dcm') -> the folder PATH itself, which load_mri()
        reassembles into one 3D volume via ants.dicom_read.

    Output name per row is constructed as ADNI_<SUBJECT>_<IMAGE_DATA_ID> (e.g.
    ADNI_002_S_0413_I1221051), so every preprocessed .nii.gz carries the ADNI
    subject + '_I######' tokens that downstream metadata parsing expects. This is
    built from the SUBJECT / IMAGE_DATA_ID columns rather than IMAGE_NAME, because
    DICOM IMAGE_NAME is a per-slice filename that may not contain the image id.

    If adnimerge_path is given, the reference is inner-joined to ADNIMERGE on the
    MRI id so only images present in ADNIMERGE survive. ADNIMERGE's integer IMAGEUID
    is matched against the reference IMAGE_DATA_ID ('I' + IMAGEUID).

    Returns two parallel lists: (image_paths, output_names).
    '''
    df = pd.read_csv(reference_csv_path)

    if adnimerge_path is not None:
        print(f"Filtering reference against ADNIMERGE: {adnimerge_path}")
        df_adni = pd.read_csv(adnimerge_path, low_memory=False)
        adni_ids = df_adni['IMAGEUID'].dropna().astype(int)
        adni_image_data_ids = set('I' + adni_ids.astype(str))
        before = len(df)
        df = df[df['IMAGE_DATA_ID'].isin(adni_image_data_ids)]
        print(f"ADNIMERGE filter kept {len(df)}/{before} scans.")

    image_paths = []
    output_names = []
    for _, row in df.iterrows():
        if str(row.get('FORMAT', 'nii')).lower() == 'dcm':
            image_paths.append(row['PATH'])                       # DICOM series: the folder
        else:
            image_paths.append(os.path.join(row['PATH'], row['IMAGE_NAME']))
        output_names.append(f"ADNI_{row['SUBJECT']}_{row['IMAGE_DATA_ID']}")

    n_nii = int((df['FORMAT'].astype(str).str.lower() == 'nii').sum())
    n_dcm = int((df['FORMAT'].astype(str).str.lower() == 'dcm').sum())
    print(f"Built {len(image_paths)} image paths from reference CSV ({n_nii} .nii, {n_dcm} DICOM series).")
    return image_paths, output_names
    
# %%

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(
        description='Preprocess raw ADNI MRIs (3D pipeline: standardize -> register to atlas -> skull-strip -> crop 100^3).')

    arg_parser.add_argument('-i', '--input',
                        metavar='input_path',
                        type=str,
                        required=False,
                        default='/mnt/d/lucas/Downloads/raw/',
                        help='Folder of raw .nii files, searched recursively (run from the repo root). Default: data/mri/raw/ADNI')

    arg_parser.add_argument('-o', '--output',
                        metavar='output_path',
                        type=str,
                        required=False,
                        default='/mnt/d/lucas/Downloads/preprocessed/' + datetime.datetime.now().strftime('%Y%m%d'),
                        help='Output folder for preprocessed .nii.gz + REFERENCE.csv. Default: data/mri/preprocessed/<today>')

    arg_parser.add_argument('-b', '--box',
                        metavar='box',
                        type=int,
                        required=False,
                        default=100,
                        help='Center-crop cube size in voxels. Default: 100 (-> 100x100x100).')

    arg_parser.add_argument('-s', '--skip',
                        metavar='skip',
                        type=int,
                        required=False,
                        default=0,
                        help='Skip the first N images (to resume a failed batch).')

    arg_parser.add_argument('-l', '--limit',
                        metavar='limit',
                        type=int,
                        required=False,
                        default=0,
                        help='Process at most N images (0 = all). Use 1 to smoke-test.')

    arg_parser.add_argument('--skip-skull-stripping',
                        action='store_true',
                        help='Bypass DeepBrain skull stripping (registered image goes straight to crop).')

    arg_parser.add_argument('-r', '--mri-reference',
                        metavar='mri_reference_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Optional prior MRI metadata CSV to merge into the output REFERENCE.csv.')

    arg_parser.add_argument('-c', '--reference-csv',
                        metavar='reference_csv_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Select images to preprocess from a reference CSV (DOWNLOAD_RAW_MRI.csv schema) '
                             'instead of sweeping --input recursively. Overrides --input as the source of images.')

    arg_parser.add_argument('--adnimerge',
                        metavar='adnimerge_path',
                        type=str,
                        required=False,
                        default=None,
                        help='Optional ADNIMERGE.csv path. When given with --reference-csv, keeps only scans whose '
                             'IMAGE_DATA_ID matches an ADNIMERGE IMAGEUID before preprocessing.')

    arg_parser.add_argument('-w', '--workers',
                        metavar='workers',
                        type=int,
                        required=False,
                        default=1,
                        help='Number of parallel worker processes (1 = serial, default). Each image already '
                             'uses many cores internally, so total threads = workers x threads-per-worker is '
                             'held near the core count. On a 10-core box, 3 is a good start.')

    arg_parser.add_argument('--threads-per-worker',
                        metavar='threads_per_worker',
                        type=int,
                        required=False,
                        default=None,
                        help='Threads each worker may use for ANTs/TF (default: cpu_count // workers). '
                             'Lower this if the machine gets sluggish; raise it if cores sit idle.')

    args = arg_parser.parse_args()

    images_to_process = None
    image_names = None
    if args.reference_csv is not None:
        images_to_process, image_names = build_image_list_from_reference(args.reference_csv, adnimerge_path=args.adnimerge)
    elif args.adnimerge is not None:
        arg_parser.error('--adnimerge requires --reference-csv (it filters the reference CSV).')

    execute_preprocessing(
        input_path=args.input,
        output_path=args.output,
        images_to_process=images_to_process,
        image_names=image_names,
        box=args.box,
        skip=args.skip,
        limit=args.limit,
        mri_reference_path=args.mri_reference,
        skip_skull_stripping=args.skip_skull_stripping,
        workers=args.workers,
        threads_per_worker=args.threads_per_worker,
    )

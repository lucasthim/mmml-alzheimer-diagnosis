import os
from pathlib import Path
import argparse
import time
import zipfile

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Supresses warnings, logs, infos and errors from TF. Need to use it carefully

def execute_extraction(input_path,output_path):
 
    '''
    MRI Extraction pipeline. 
    
    Main steps:
    
    - Extract MRI from zip files

    Example:

        python mri_extract.py --input "/home/lucasthim1/alzheimer_bucket/raw/" --output "/home/lucasthim1/mmml-alzheimer-diagnosis/data/raw/"
    
    Bash commmand equivalent (for single files):
    
         unzip ~/alzheimer_bucket/raw/MRI_42_50_chunks_MPRAGE_2.zip -d ~/mmml-alzheimer-diagnosis/data/raw/
    '''    
    if input_path.endswith(".zip"):
        zips_to_process = [input_path]
    else:
        zips_to_process = list(Path(input_path).rglob("*.zip"))
        print("Looking for MRI zip files in path:",input_path,'\n')

    if not os.path.exists(output_path):
        print("Creating output path... \n")
        os.makedirs(output_path)
    
    start = time.time()
    print('------------------------------------------------------------------------------------------------------------------------')
    print(f"Starting extraction for {len(zips_to_process)} zip files. This might take a while... =)")
    print('------------------------------------------------------------------------------------------------------------------------')

    for ii,zip_path in enumerate(zips_to_process):
        
        start_img = time.time()
        print(f"Processing file ({ii+1}/{len(zips_to_process)}):",zip_path)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_path)


    total_time = (time.time() - start) / 60.
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('All zip files extracted! Process took %.2f min' % total_time)
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')
    print('-------------------------------------------------------------')

def main():
    
    execute_extraction(input_path=args.input,output_path=args.output)
    
arg_parser = argparse.ArgumentParser(description='Extract MRI from zip files')


arg_parser.add_argument('-i','--input',
                    metavar='input',
                    type=str,
                    required=True,
                    help='Input directory containing the zip files')

arg_parser.add_argument('-o','--output',
                    metavar='output',
                    type=str,
                    required=True,
                    help='Output directory to extract nifti files')

args = arg_parser.parse_args()


if __name__ == '__main__':
    main()
"""
Train CNNs for ADxCN and MCIxCN across all 3 orientations (coronal, axial, sagittal).

Set MODEL to the architecture you want (e.g. "vgg19_bn", "vgg13", "resnet34"). All output
filenames derive from MODEL, so switching models NEVER overwrites a previous run's outputs.

Config matches the best-evidenced historical run (notebooks/20211027_Run_CNN_VGG19_for_ensemble.ipynb
cell 21 / dissertation Table 5.6): lr=0.0001, batch_size=16, Adam, BCE loss, max_epochs=100,
early_stop=10, no augmentation (matches the current notebooks/2026_paper/20260710_train_cnn_adxcn_axial.ipynb
config exactly).

matplotlib.use('Agg') must be the first matplotlib-related statement in the whole process --
mri_train.py does a bare `import matplotlib.pyplot as plt` with no backend set, so this has to
run before mri_train (or anything importing it) gets imported, or plt.show() would try to open
a window.

Run in the background with output logged to a file:
    cd /home/lucasthim/projects/phd/mmml-alzheimer-diagnosis
    nohup .venv/bin/python -u src/model_training/train_all_cnns.py \
        > data/mri/experiments/train_all_cnns_$(date +%Y%m%d_%H%M).log 2>&1 &
"""
import matplotlib
matplotlib.use("Agg")

import os
import sys
from datetime import datetime

REPO_ROOT = "/home/lucasthim/projects/phd/mmml-alzheimer-diagnosis"

# mri_train.py's own internal imports (`sys.path.append("./../models")` etc.) are CWD-relative,
# not __file__-relative -- CWD must be src/model_training/ for them to resolve.
os.chdir(os.path.join(REPO_ROOT, "src/model_training"))
sys.path.insert(0, os.getcwd())

from mri_train import run_experiments_for_ensemble

MRI_REFERENCE = os.path.join(REPO_ROOT, "data/reference/PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_20260710_1413.csv")

# The one knob: switch this to re-run every experiment with a different architecture.
# All output paths below derive from it, so runs never clobber each other.
MODEL = "vgg13"
MODEL_TAG = MODEL.upper()   # used in output filenames, e.g. PREDICTIONS_AD_VGG13.csv

MRI_CONFIG_BASE = {
    "num_samples": 0,
    "num_rotations": 0,
    "sampling_range": 0,
    "mri_reference": MRI_REFERENCE,
    "output_path": os.path.join(REPO_ROOT, "data/reference/"),
}

ADDITIONAL_PARAMS = {
    "lr": 0.0001,
    "batch_size": 16,
    "optimizer": "adam",
    "max_epochs": 100,
    "early_stop": 10,
    "early_stop_metric": "auc",
    "prediction_threshold": 0.5,
    "loss": "BCE",
}

EXPERIMENTS = [
    {
        "name": "ADxCN",
        "classes": ["AD", "CN"],
        "orientation_and_slices": [("coronal", [43]), ("axial", [23]), ("sagittal", [26])],
        "save_path": os.path.join(REPO_ROOT, f"data/PREDICTIONS_AD_{MODEL_TAG}.csv"),
        "model_path": os.path.join(REPO_ROOT, f"models/{MODEL}_adxcn"),
    },
    {
        "name": "MCIxCN",
        "classes": ["MCI", "CN"],
        "orientation_and_slices": [("coronal", [70]), ("axial", [8]), ("sagittal", [50])],
        "save_path": os.path.join(REPO_ROOT, f"data/PREDICTIONS_MCI_{MODEL_TAG}.csv"),
        "model_path": os.path.join(REPO_ROOT, f"models/{MODEL}_mcixcn"),
    },
]


def main():
    # Safety net: never silently overwrite a previous run's predictions. If a target
    # PREDICTIONS_*.csv already exists, stop before training rather than clobber it.
    existing = [exp["save_path"] for exp in EXPERIMENTS if os.path.exists(exp["save_path"])]
    if existing:
        print("REFUSING TO RUN -- these output files already exist (move/rename them first):", flush=True)
        for p in existing:
            print(f"   {p}", flush=True)
        sys.exit(1)

    print(f"MODEL = {MODEL}  ->  outputs tagged {MODEL_TAG}", flush=True)
    for exp in EXPERIMENTS:
        started = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 100}", flush=True)
        print(f"STARTING EXPERIMENT: {exp['name']}  ({started})", flush=True)
        print(f"{'=' * 100}\n", flush=True)

        # fresh copy per experiment -- run_experiments_for_ensemble mutates mri_config in place
        # (adds 'orientation'/'slice' keys per iteration)
        mri_config = dict(MRI_CONFIG_BASE)

        df_predictions = run_experiments_for_ensemble(
            orientation_and_slices=exp["orientation_and_slices"],
            model=MODEL,
            classes=exp["classes"],
            mri_config=mri_config,
            additional_experiment_params=dict(ADDITIONAL_PARAMS),
            save_path=exp["save_path"],
            model_path=exp["model_path"],
        )

        finished = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'=' * 100}", flush=True)
        print(f"FINISHED EXPERIMENT: {exp['name']} -> {exp['save_path']}  shape={df_predictions.shape}  ({finished})", flush=True)
        print(f"{'=' * 100}\n", flush=True)

    print("\nALL EXPERIMENTS DONE.", flush=True)


if __name__ == "__main__":
    main()

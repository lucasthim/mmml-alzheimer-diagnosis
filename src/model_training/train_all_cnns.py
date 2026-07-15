"""
Train VGG19_BN CNNs for ADxCN and MCIxCN across all 3 orientations (coronal, axial, sagittal).

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
MODEL = "vgg19_bn"

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
        "save_path": os.path.join(REPO_ROOT, "data/PREDICTIONS_AD_VGG19_BN.csv"),
        "model_path": os.path.join(REPO_ROOT, "models/vgg19_bn_adxcn"),
    },
    {
        "name": "MCIxCN",
        "classes": ["MCI", "CN"],
        "orientation_and_slices": [("coronal", [70]), ("axial", [8]), ("sagittal", [50])],
        "save_path": os.path.join(REPO_ROOT, "data/PREDICTIONS_MCI_VGG19_BN.csv"),
        "model_path": os.path.join(REPO_ROOT, "models/vgg19_bn_mcixcn"),
    },
]


def main():
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

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
import glob
from datetime import datetime

import pandas as pd

REPO_ROOT = "/home/lucasthim/projects/phd/mmml-alzheimer-diagnosis"

# mri_train.py's own internal imports (`sys.path.append("./../models")` etc.) are CWD-relative,
# not __file__-relative -- CWD must be src/model_training/ for them to resolve.
os.chdir(os.path.join(REPO_ROOT, "src/model_training"))
sys.path.insert(0, os.getcwd())

from mri_train import run_experiments_for_ensemble

# Reuse the repo's own metric implementation (AUC/F1/precision/recall/accuracy) instead of
# re-deriving it. base_evaluation lives in src/model_evaluation and its internal
# `from de_long_evaluation import ...` is dir-relative, so that dir must be on sys.path.
sys.path.insert(0, os.path.join(REPO_ROOT, "src/model_evaluation"))
from base_evaluation import compute_metrics_binary


# Set to a specific PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_*.csv to pin one; leave None to
# auto-pick the newest slice run's master reference, so a fresh slicing run is picked up without
# hand-editing this file (avoids training against a stale reference).
MRI_REFERENCE_OVERRIDE = None


def resolve_latest_mri_reference():
    """Return the newest slice-run master reference in data/reference/.

    run_slice_preparation.py writes PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_<ts>.csv at the end
    of every run; <ts> is fixed-width YYYYMMDD_HHMM, so lexical max == most recent.
    """
    pattern = os.path.join(REPO_ROOT, "data/reference/PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_*.csv")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No master reference matching {pattern}. "
            "Run src/data_preparation/run_slice_preparation.py first."
        )
    return matches[-1]


MRI_REFERENCE = MRI_REFERENCE_OVERRIDE or resolve_latest_mri_reference()

# The one knob: switch this to re-run every experiment with a different architecture.
# All output paths below derive from it, so runs never clobber each other.
MODEL = "vgg19_bn"   # e.g. "vgg19_bn", "vgg13", "resnet34"
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
        "save_path": os.path.join(REPO_ROOT, f"data/predictions/PREDICTIONS_AD_{MODEL_TAG}.csv"),
        "model_path": os.path.join(REPO_ROOT, f"models/{MODEL}_adxcn"),
    },
    {
        "name": "MCIxCN",
        "classes": ["MCI", "CN"],
        "orientation_and_slices": [("coronal", [70]), ("axial", [8]), ("sagittal", [50])],
        "save_path": os.path.join(REPO_ROOT, f"data/predictions/PREDICTIONS_MCI_{MODEL_TAG}.csv"),
        "model_path": os.path.join(REPO_ROOT, f"models/{MODEL}_mcixcn"),
    },
]


def summarize_experiment(name, save_path):
    """Print & save per-slice, per-split metrics from a finished PREDICTIONS_*.csv.

    True label = MACRO_GROUP (0/1), predicted score = CNN_SCORE. Reuses the repo's
    compute_metrics_binary (src/model_evaluation/base_evaluation.py) at threshold 0.5.
    Writes a sibling METRICS_*.csv and returns the tidy DataFrame (or None).
    """
    if not os.path.exists(save_path):
        print(f"[metrics] {name}: no predictions at {save_path}, skipping.", flush=True)
        return None

    df = pd.read_csv(save_path)
    split_rank = {"train": 0, "validation": 1, "test": 2}
    rows = []
    for (orientation, sl, dataset), g in df.groupby(["ORIENTATION", "SLICE", "DATASET"]):
        try:
            m = compute_metrics_binary(g["MACRO_GROUP"].values, g["CNN_SCORE"].values)
        except ValueError as err:   # e.g. a split with a single class present -> AUC undefined
            print(f"[metrics] {name} {orientation} slice {sl} {dataset}: skipped ({err})", flush=True)
            continue
        rows.append({
            "EXPERIMENT": name, "ORIENTATION": orientation, "SLICE": sl, "DATASET": dataset,
            "N": len(g), "AUC": m["auc"], "F1": m["f1score"],
            "PRECISION": m["precision"], "RECALL": m["recall"], "ACCURACY": m["accuracy"],
        })
    if not rows:
        return None

    summary = pd.DataFrame(rows).sort_values(
        ["ORIENTATION", "SLICE", "DATASET"],
        key=lambda s: s.map(split_rank) if s.name == "DATASET" else s,
    ).reset_index(drop=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    metrics_path = save_path.replace("PREDICTIONS_", "METRICS_")
    summary.to_csv(metrics_path, index=False)

    print(f"\n{'-' * 100}", flush=True)
    print(f"METRICS -- {name}   (label=MACRO_GROUP, score=CNN_SCORE, threshold=0.5)", flush=True)
    with pd.option_context("display.max_rows", None, "display.width", 200):
        print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
    print(f"saved -> {metrics_path}", flush=True)
    print(f"{'-' * 100}\n", flush=True)
    return summary


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
    print(f"MRI_REFERENCE = {MRI_REFERENCE}", flush=True)

    summaries = []
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

        # summarize now (not just at the end) so metrics survive even if a later experiment crashes
        summary = summarize_experiment(exp["name"], exp["save_path"])
        if summary is not None:
            summaries.append(summary)

    if summaries:
        combined = pd.concat(summaries, ignore_index=True)
        os.makedirs(os.path.join(REPO_ROOT, "data/predictions"), exist_ok=True)
        combined_path = os.path.join(REPO_ROOT, f"data/predictions/METRICS_ALL_{MODEL_TAG}.csv")
        combined.to_csv(combined_path, index=False)
        print(f"\n{'=' * 100}", flush=True)
        print(f"COMBINED TEST-SET METRICS ({MODEL_TAG})", flush=True)
        test_only = combined[combined["DATASET"] == "test"]
        with pd.option_context("display.max_rows", None, "display.width", 200):
            print(test_only.to_string(index=False, float_format=lambda x: f"{x:.3f}"), flush=True)
        print(f"\nfull per-split metrics saved -> {combined_path}", flush=True)
        print(f"{'=' * 100}", flush=True)

    print("\nALL EXPERIMENTS DONE.", flush=True)


if __name__ == "__main__":
    main()

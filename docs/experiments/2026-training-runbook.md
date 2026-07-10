*Part of the [MMML-Alzheimer documentation](../README.md). A personalized, ordered checklist for the 2026 re-run: generate training-ready data from the reprocessed MRIs, then train the CNNs, cognitive model, and ensemble.*

# 2026 Training Runbook

This picks up from the general [running-experiments.md](running-experiments.md) runbook, but reflects
the *actual* state of this specific 2026 re-run and three corrections found by cross-checking the
historical notebooks (not just the source/docs) during this session. Read
[running-experiments.md](running-experiments.md) first for full background, the complete gotcha
catalogue, and why each step exists — this doc only carries the parts that differ or need to be made
concrete for this run.

## Where this run stands

Done:

- **Step 1** — `data/tabular/ADNIMERGE.csv` rebuilt from ADNIMERGE2.
- **Step 2** — [cognitive_tests_preprocessing.py](../../src/data_preprocessing/cognitive_tests_preprocessing.py) run → `data/tabular/COGNITIVE_DATA_PREPROCESSED.csv`.
- **Step 3a–3c** — MRIs already downloaded and unzipped under `/mnt/d/lucas/Downloads/raw/ADNI`.
- **Step 5** — [mri_preprocessing.py](../../src/data_preprocessing/mri_preprocessing.py) re-run against
  `data/reference/REPROCESS_MRI_REFERENCE.csv` (all DICOM-sourced scans, since that batch was corrupted,
  plus the handful of never-processed NIfTI scans). Output spans **three** dated folders that all matter
  for Step 5b: `/mnt/d/lucas/Downloads/preprocessed/20260707`, `20260708`, `20260709`.

**Skippable:** Step 4 (`RAW_MRI_REFERENCE.csv`) — the 2026 rewrite of `ensemble_preprocessing.py` no
longer reconciles MRI-side `GROUP` against cognitive `DX` (single diagnosis source now), so nothing
downstream reads Step 4's output anymore.

Remaining: Step 5b → 6 → 7 → 8 → 9 → 10 → 11 → 12 → 13, detailed below.

---

## Step 5b — Concat metadata + backfill labels

**Correction from the original plan:** the labels (`GROUP`/`MACRO_GROUP`/`SEX`/`AGE`) needed by
[mri_batch_preparation.py:98-100](../../src/data_preparation/mri_batch_preparation.py#L98-L100) are
**not** written by `mri_preprocessing.py` unless `-r/--mri-reference` was passed (it wasn't, for this
run). Checked three candidate sources for backfilling them against the 3310 IMAGE_DATA_IDs in the
reprocessing batch:

| Source | Missing IMAGE_DATA_IDs |
|---|---|
| `data/reference/REFERENCE_TABLE_FOR_MRI.csv` (LONI collection exports) | 75 — stale, pre-dates the latest MRI downloads |
| `data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv` | 1549 — already filtered to `classes=[0,1]` and to patients with complete cognitive data |
| `data/tabular/COGNITIVE_DATA_PREPROCESSED.csv` | **0** — ADNIMERGE-derived, full coverage |

Use `COGNITIVE_DATA_PREPROCESSED.csv`. One more subtlety: `MACRO_GROUP` must be the **string** labels
`'CN'/'AD'/'MCI'`, not the numeric `DIAGNOSIS` encoding (`0/1/2`). `MACRO_GROUP == 'MCI'` remap
happens later; if `MACRO_GROUP` is already numeric, [return_sets()](../../src/model_training/mri_train.py#L315-L334)'s remap silently no-ops and an MCIxCN run would incorrectly keep AD rows instead of MCI
rows. `GROUP` and `SEX` are confirmed (by grep) never read downstream — they only need to exist to
satisfy the hardcoded column selection.

```python
import sys; sys.path.insert(0, 'src/data_preprocessing')
from mri_metadata_preprocessing import execute_mri_metadata_preprocessing
import pandas as pd

df = execute_mri_metadata_preprocessing(
    input=['/mnt/d/lucas/Downloads/preprocessed/20260707/REFERENCE.csv',
           '/mnt/d/lucas/Downloads/preprocessed/20260708/REFERENCE.csv',
           '/mnt/d/lucas/Downloads/preprocessed/20260709/REFERENCE.csv'],
    output='data/reference/PREPROCESSED_MRI_REFERENCE.csv',
    drop_cols=['FORMAT','TYPE','UNIQUE_IMAGE_ID','MODALITY','DOWNLOADED','SUBJECT_ID'])

df_cog = pd.read_csv('data/tabular/COGNITIVE_DATA_PREPROCESSED.csv', low_memory=False)
df_cog['IMAGE_DATA_ID'] = 'I' + df_cog['IMAGEUID'].astype(int).astype(str)
label_map = {0: 'CN', 1: 'AD', 2: 'MCI'}
df_cog['MACRO_GROUP'] = df_cog['DIAGNOSIS'].map(label_map)   # string labels — return_sets() needs this, not 0/1/2
df_cog['GROUP'] = df_cog['MACRO_GROUP']                       # unused downstream, kept for schema compatibility
df_cog['SEX'] = df_cog['MALE']                                 # unused downstream

df = df.merge(df_cog[['IMAGE_DATA_ID','GROUP','MACRO_GROUP','SEX','AGE']], on='IMAGE_DATA_ID', how='left')
df.to_csv('data/reference/PREPROCESSED_MRI_REFERENCE.csv', index=False)
```

---

## Step 6 — Ensemble reference

```bash
python src/data_preprocessing/ensemble_preprocessing.py \
    --cognitive data/tabular/COGNITIVE_DATA_PREPROCESSED.csv \
    --output data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv \
    --downloaded-mri-reference data/reference/DOWNLOAD_RAW_MRI.csv \
    --classes 0 1
```

Run again with `--classes 0 1 2` if you also want one combined reference covering the MCI cohort.

## Step 7 — DATASET split (the milestone)

```python
import sys; sys.path.insert(0, 'src/data_preparation')
from ensemble_preparation import execute_ensemble_preparation
execute_ensemble_preparation('data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv',
                              'data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv', classes=[0,1])
```

This `DATASET` column (train/validation/test, subject-level, seed 151) is fixed across every model
built from here on — CNNs, cognitive model, and ensemble.

## Step 8 — 2D slices: only what's actually needed

The full 100-slice-per-orientation sweep was the historical *slice search*; the dissertation already
resolved which slices to use, so there's no need to repeat it:

- **AD (ADxCN):** coronal 43, axial 23, sagittal 26
- **MCI (MCIxCN):** coronal 70, axial 8, sagittal 50

Call once per orientation with an explicit single-key dict — this avoids the duplicate-dict-key bug in
[mri_batch_preparation.py:20-26](../../src/data_preparation/mri_batch_preparation.py#L20-L26), the same
workaround the original 2021 notebooks used:

```python
import sys; sys.path.insert(0, 'src/data_preparation')
from mri_batch_preparation import execute_mri_batch_preparation

for orientation, slice_idx in [('coronal',43), ('axial',23), ('sagittal',26),   # AD
                                ('coronal',70), ('axial',8),  ('sagittal',50)]:  # MCI
    execute_mri_batch_preparation(
        mri_reference_path='data/reference/PREPROCESSED_MRI_REFERENCE.csv',
        ensemble_reference_path='data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
        output_path='data/mri/processed/storage/',
        orientations={orientation: [slice_idx]})
```

Also fix the inverted zero-pad bug
([mri_batch_preparation.py:208](../../src/data_preparation/mri_batch_preparation.py#L208)) first, or
filenames come out as `coronal_070.npz` instead of `coronal_70.npz`. Then concat the six
`PROCESSED_MRI_REFERENCE_<timestamp>.csv` outputs into one
`PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_*.csv` (`pd.concat` + save — the same pattern the notebooks
used).

## Step 9 — CNN training

Augmentation happens here, via `generate_mri_dataset_reference`/`mri_dataset_generation.py` — not at
Step 8. Best-evidenced config, traced from the actual executed cells in
[20211027_Run_CNN_VGG19_for_ensemble.ipynb](../../notebooks/20211027_Run_CNN_VGG19_for_ensemble.ipynb)
(not just the function defaults, which were never actually used as-is):

```python
mri_config = {'num_samples': 0, 'num_rotations': 3, 'sampling_range': 3,
              'mri_reference': 'data/reference/PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_<ts>.csv',
              'output_path': 'data/mri/experiments/'}
additional_experiment_params = {'lr': 0.0001, 'batch_size': 16, 'optimizer': 'adam',
                                 'max_epochs': 100, 'early_stop': 15,
                                 'prediction_threshold': 0.5, 'loss': 'BCE'}
```

- Augmentation is **rotation-only** — `num_samples` (neighbor-slice sampling) was hardcoded to `0` in
  every real invocation found; only `num_rotations=3` (angles from `{-15,...,15}` step 2, plus the
  original) was actually used.
- `lr=0.0001` is not optional — higher learning rates collapsed this exact setup to AUC=0.5
  (all-negative predictions) for several epochs in the traced runs, for both an early VGG19 attempt and
  ResNet101.
- The dissertation text says `early_stopping=10` for ADxCN; the closest-matching traced notebook run
  actually used `early_stop=15`. Small discrepancy, not resolved — pick either.
- MCIxCN config is **not reliably traceable** from the notebooks (that exploration was messier, mostly
  around slice 95 rather than the final slice 70) — expect to tune from scratch. Historically needed a
  much lower learning rate with `FocalLoss` (down to `~0.000001`–`0.000005`).

Patch first (see [running-experiments.md](running-experiments.md#must-fix-before-you-run-anything) for
the full list):
- `additional_experiment_params` must include an explicit `'loss'` key or `mri_train.py` raises `KeyError`.
- `WeightedFocalLoss` hardcodes `.cuda()` — breaks on CPU/MPS.
- Model checkpointing must `deepcopy(model.state_dict())`, not save a live reference — otherwise "best"
  predictions keep changing after supposedly frozen checkpoints are reloaded (this was an actual bug
  the original researcher hit and fixed, per `20211107_Fix_CNN_changing_predictions.ipynb`).
- The `best_epoch == max_epochs` save-check can skip saving when early stopping never triggers — save
  `best_model_params` unconditionally.

## Step 10 — Cognitive / tabular model training

```bash
python src/model_training/cognitive_tests_train.py
```

Output has a `Score_1` column, not `COGTEST_SCORE` — the rename only ever existed as notebook glue, not
in the script, so do it yourself before Step 11:

```python
import pandas as pd
df_cog = pd.read_csv('data/PREDICTIONS_COGNITIVE_TESTS.csv').rename(
    columns={'Score_1': 'COGTEST_SCORE', 'IMAGEUID': 'IMAGE_DATA_ID'})
df_cog.to_csv('data/PREDICTIONS_COGNITIVE_TESTS_renamed.csv', index=False)
```

## Step 11 — Ensemble (fusion) training

```python
from ensemble_train import prepare_ensemble_experiment_set, get_experiment_sets, train_ensemble_models
from interpret.glassbox import ExplainableBoostingClassifier
from sklearn.linear_model import LogisticRegression

df_ensemble = prepare_ensemble_experiment_set(
    'data/PREDICTIONS_COGNITIVE_TESTS_renamed.csv', 'data/PREDICTIONS_AD_VGG19_BN.csv')
df_train, df_val, df_test = get_experiment_sets(df_ensemble, cols_to_drop=['SUBJECT','DATASET'])
models = train_ensemble_models(df_train, 'DIAGNOSIS', [ExplainableBoostingClassifier(), LogisticRegression()])
```

**Gap found in the current code:** if Step 9's predictions include rotation-augmented rows
(`ROTATION_ANGLE != 0`, which they will with the augmentation config above),
[ensemble_train.py::prepare_mri_predictions](../../src/model_training/ensemble_train.py#L18) has **no**
`ROTATION_ANGLE == 0` filter — the original 2021 notebook
([20211028_Ensemble_Results.ipynb](../../notebooks/20211028_Ensemble_Results.ipynb)) explicitly filtered
`ROTATION_ANGLE == 0` before pivoting; the current script doesn't. Its `pivot_table` won't crash — it'll
silently mean-average the original + rotated-copy CNN scores per image. Add the filter yourself if you
want validation/test rows to reflect only the true (unrotated) scan:

```python
# inside/after prepare_mri_predictions, before the pivot_table call
df_mri = df_mri.query("ROTATION_ANGLE == 0 or ROTATION_ANGLE != ROTATION_ANGLE")  # keep unrotated + NaN (no-augmentation) rows
```

## Step 12 — Evaluation

- **Modules:** [base_evaluation.py](../../src/model_evaluation/base_evaluation.py) +
  [ensemble_evaluation.py](../../src/model_evaluation/ensemble_evaluation.py) +
  [de_long_evaluation.py](../../src/model_evaluation/de_long_evaluation.py). Each score column
  (`CNN_SCORE_*`, `COGTEST_SCORE`, ensemble output) is treated as a "model"; computes ROC/AUC, an
  optimal decision threshold set on **validation** and applied to **test**
  (`set_threshold_for_test`), confidence intervals, and pairwise DeLong AUC-comparison p-values.
- **`np.float` fix already applied this session** — `de_long_evaluation.py` used the removed
  `np.float` alias (NumPy ≥1.24 raises `AttributeError`) at the midrank/fastDeLong helpers; replaced
  with `np.float64` at lines 17, 25, 61–63. Verified with a smoke test (`delong_roc_test` on a toy
  6-sample array runs without the `AttributeError`). This was the top "won't run after a few years"
  hazard for this step — it's resolved now, no further action needed before running evaluation.
- Full metric definitions, the DeLong test details, and how to call these modules from a notebook:
  **[evaluation.md](../modeling/evaluation.md)**.
- `mri_evaluation.py` is empty (0 bytes) — there is no MRI-specific evaluation module; MRI scores are
  evaluated the same way as any other "model" column via `base_evaluation.py`.

## Step 13 — Explanation (XAI)

- **Modules:** [mri_explanation.py](../../src/model_explanation/mri_explanation.py) (Captum DeepLift +
  Guided Grad-CAM on the CNNs) and
  [ensemble_explanation.py](../../src/model_explanation/ensemble_explanation.py) (EBM/LR feature
  weights, local + global).
- **Image explanations need:** a prediction reference carrying `MODEL`, `MODEL_PATH`, `IMAGE_PATH`,
  `ORIENTATION`, `SLICE`, `MACRO_GROUP`, `CNN_SCORE`, `CNN_PREDICTION`, `IMAGE_DATA_ID`.
- **Tabular explanations need:** the fitted EBM + the ensemble feature frame indexed by
  `IMAGE_DATA_ID` with a `DIAGNOSIS` target and slice-score columns (`AXIAL_23`/`CORONAL_43`/
  `SAGITTAL_26` for AD, `AXIAL_8`/`CORONAL_70`/`SAGITTAL_50` for MCI) plus a demographics table via
  `prepare_patient_data_for_explanations`.
- **Output:** matplotlib figures only (`plt.show()` / returned `fig`) — nothing is written to disk by
  these modules; save any figure you want to keep yourself.
- Full walkthrough (DeepLift/Grad-CAM configuration, SmoothGrad parameters, global vs. local EBM
  explanations): **[explainability.md](../modeling/explainability.md)**.

---

## See also

- [running-experiments.md](running-experiments.md) — the general runbook this doc specializes, with
  the full gotcha catalogue and README-vs-code discrepancy table.
- [data-preparation.md](../data/data-preparation.md) — 3D→2D slicing, augmentation mechanics, and the
  DATASET split in more detail.
- [evaluation.md](../modeling/evaluation.md) — Step 12 in full.
- [explainability.md](../modeling/explainability.md) — Step 13 in full.
- [known-issues.md](../reference/known-issues.md) — the full bug/stub/gotcha catalogue.

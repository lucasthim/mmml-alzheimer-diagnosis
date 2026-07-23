*Part of the [MMML-Alzheimer documentation](../README.md). A personalized, ordered checklist for the 2026 re-run: generate training-ready data from the reprocessed MRIs, then train the CNNs, cognitive model, and ensemble.*

# 2026 Training Runbook

This picks up from the general [running-experiments.md](running-experiments.md) runbook, but reflects
the *actual* state of this specific 2026 re-run and a growing list of corrections found by cross-checking
the historical notebooks (not just the source/docs) and by actually executing each step during this and
prior sessions. Read
[running-experiments.md](running-experiments.md) first for full background, the complete gotcha
catalogue, and why each step exists — this doc only carries the parts that differ or need to be made
concrete for this run.

## The data-generation flow (2026 re-run)

For this re-run the training-data pipeline collapses to **three** commands:

1. **MRI preprocessing** — [mri_preprocessing.py](../../src/data_preprocessing/mri_preprocessing.py) (Step 5) → atlas-registered, skull-stripped, cropped `.nii.gz` volumes.
2. **Ensemble preprocessing + preparation** — [ensemble_preprocessing.py](../../src/data_preprocessing/ensemble_preprocessing.py) (Step 6) then [ensemble_preparation.py](../../src/data_preparation/ensemble_preparation.py) (Step 7) → the `DATASET` train/val/test split. Cognitive-derived, so **atlas-independent** — not affected by the Step 5 re-run.
3. **Slice preparation** — [run_slice_preparation.py](../../src/data_preparation/run_slice_preparation.py) → folds the old Steps 5b + 8 into one runner (enrich labels, cut the 6 slices, backfill `DATASET`).

Then Steps 9 → 13 (train / evaluate / explain).

## Where this run stands

> **Atlas fix (2026-07-22).** The earlier runs used ANTsPy's MNI152 fallback, not the study's real
> `atlas_t1.nii` — registration was off and CNN AUCs suffered. The true atlas is now at
> `data/mri/atlas/atlas_t1.nii`, and Step 5 was re-run against it. This **invalidates the old 3D volumes
> and the slices derived from them** (Steps 5 and 8), so both are being regenerated. Steps 6–7 are
> cognitive-derived and unaffected.

Done:

- **Step 1** — `data/tabular/ADNIMERGE.csv` rebuilt from ADNIMERGE2.
- **Step 2** — [cognitive_tests_preprocessing.py](../../src/data_preprocessing/cognitive_tests_preprocessing.py) run → `data/tabular/COGNITIVE_DATA_PREPROCESSED.csv`.
- **Step 3a–3c** — MRIs already downloaded and unzipped under `/mnt/d/lucas/Downloads/raw/ADNI`.
- **Step 5** — [mri_preprocessing.py](../../src/data_preprocessing/mri_preprocessing.py) **re-run 2026-07-22
  with the true registration atlas** over all ADNIMERGE-matched scans (`--reference-csv DOWNLOAD_RAW_MRI.csv
  --adnimerge ADNIMERGE.csv`). Output is a **single** clean folder
  `/mnt/d/lucas/Downloads/preprocessed/20260722/` (**7279 images, 0 missing, 0 duplicate IMAGE_DATA_IDs**).
  This **supersedes** the earlier bad-atlas output (`20260707/08/09`) — the multi-folder concat + newest-first
  dedup that the old Step 5b needed no longer applies.
- **Step 6** — run with `--classes 0 1 2` (not `0 1`) → `data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv`
  covers CN/AD/MCI (911/179/672 rows). One combined reference covering all cohorts is simpler for Step 8.
  Cognitive-derived → **still valid after the atlas re-run**, not regenerated.
- **Step 7** — run with `classes=[0,1,2]` → `data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv`
  (train 861 / validation 455 / test 446, after fixing the train/test leakage bug — see below).
  Cognitive-derived → **still valid**, not regenerated.

To run (regenerating slices on the atlas-fixed volumes):

- **Steps 5b + 8** — now combined in [run_slice_preparation.py](../../src/data_preparation/run_slice_preparation.py)
  → enrich labels from cognitive data, cut the 6 slices in one call, backfill `DATASET`, write the
  `PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_<ts>.csv` training master. Supersedes the old
  `PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_20260710_1413.csv`. See the combined recipe below.

**Skippable:** Step 4 (`RAW_MRI_REFERENCE.csv`) — the 2026 rewrite of `ensemble_preprocessing.py` no
longer reconciles MRI-side `GROUP` against cognitive `DX` (single diagnosis source now), so nothing
downstream reads Step 4's output anymore.

Remaining: run Step 5b+8 ([run_slice_preparation.py](../../src/data_preparation/run_slice_preparation.py)), then Step 9 → 10 → 11 → 12 → 13, detailed below.

---

## Step 5b — Backfill labels (now inside run_slice_preparation.py)

Step 5b is no longer a standalone step — it is `build_enriched_reference()` in
[run_slice_preparation.py](../../src/data_preparation/run_slice_preparation.py), run automatically
before slicing. It exists because the labels (`SUBJECT`/`GROUP`/`MACRO_GROUP`/`SEX`/`AGE`) needed by
[mri_batch_preparation.py:98-100](../../src/data_preparation/mri_batch_preparation.py#L98-L100) are
**not** written by `mri_preprocessing.py` unless `-r/--mri-reference` is passed (it wasn't). The raw
`REFERENCE.csv` carries only `SUBJECT_IMAGE_ID, SUBJECT_ID, IMAGE_DATA_ID, IMAGE_PATH`.

**Source: `COGNITIVE_DATA_PREPROCESSED.csv`.** Of the three candidates checked, only it has full coverage:

| Source | Missing IMAGE_DATA_IDs (of 7279) |
|---|---|
| `data/reference/REFERENCE_TABLE_FOR_MRI.csv` (LONI collection exports) | 75 — stale, pre-dates the latest MRI downloads |
| `data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv` | 1549 — filtered to complete-cognitive-data patients |
| `data/tabular/COGNITIVE_DATA_PREPROCESSED.csv` | **0** — ADNIMERGE-derived, full coverage |

**Critical subtlety — `MACRO_GROUP` must be the STRING labels `'CN'/'AD'/'MCI'`**, not the numeric
`DIAGNOSIS` encoding. The `'MCI'` remap in [return_sets()](../../src/model_training/mri_train.py#L315-L334)
happens later; if `MACRO_GROUP` is already numeric, that remap silently no-ops and an MCIxCN run keeps
AD rows instead of MCI rows. `GROUP` and `SEX` are never read downstream (grep-confirmed) — they exist
only to satisfy the hardcoded column selection; the runner sets `GROUP = MACRO_GROUP`, `SEX = MALE`.

**The old multi-folder concat is obsolete.** The prior run's Step 5b concatenated three bad-atlas folders
(`20260707/08/09`) newest-first to dedup a corrupted-then-reprocessed batch, and dropped 7 genuinely-missing
files. The **atlas-fixed `20260722` run is a single clean folder — 7279 images, 0 duplicate IDs, 0 missing
files, 0 unmatched labels** (verified), so none of that machinery is needed. `run_slice_preparation.py`
reads the one folder, joins the labels, and still applies the `os.path.exists` guard defensively.

## Step 6 — Ensemble reference

Run with `--classes 0 1 2`, not `0 1` — one combined reference covering CN/AD/MCI together is simpler
for Step 8 than maintaining two separately-scoped references (see Step 8 below):

```bash
python src/data_preprocessing/ensemble_preprocessing.py \
    --cognitive data/tabular/COGNITIVE_DATA_PREPROCESSED.csv \
    --output data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv \
    --downloaded-mri-reference data/reference/DOWNLOAD_RAW_MRI.csv \
    --classes 0 1 2
```

**Why this matters beyond convenience:** `ensemble_preprocessing.py` filters to `classes` *before*
anything downstream sees the data — rows for classes not listed are dropped outright, not just
excluded from the split. If Step 6 is run with the historical default (`0 1`, AD/CN only) and Step 8's
MCI-orientation slices are then merged against that reference for their `DATASET` column, every MCI
image gets `DATASET = NaN` — silently breaking the MCIxCN train/val/test split with no error raised.

## Step 7 — DATASET split (the milestone)

Match Step 6's scope — `classes=[0,1,2]`. `train_test_split_by_subject` explicitly branches on
`len(labels) == 3` ([train_test_split.py:39](../../src/data_preparation/train_test_split.py#L39)), so
the 3-class split isn't a hack, it's a supported path — but running it with 3 classes surfaced a real
bug that a 2-class run mostly hides.

**Bug found by running this: train/test leakage from a per-label merge that isn't scoped to its own
label.** [train_test_split.py:55-56](../../src/data_preparation/train_test_split.py#L55-L56) used to
read:
```python
df_train_cl = df_classes.query("SUBJECT in @train_subjects")
df_test_cl = df_classes.query("SUBJECT in @test_subjects")
```
`df_classes` holds rows for **all** requested classes, not just the label the current loop iteration
is processing. Any subject with scans under more than one diagnosis (a CN→MCI or MCI→AD progressor —
common in ADNI's longitudinal data) gets *all* their rows pulled in at every label iteration where they
qualify. Since each label iteration shuffles/splits subjects independently, the same subject — and with
`classes=[0,1,2]` frequently the same exact `IMAGE_DATA_ID` — could land in `train` during one label's
iteration and `test` during another's. Verified on the real data: **180 images ended up with
contradictory `DATASET` values (both train and test simultaneously)**, out of 949 duplicated
`IMAGE_DATA_ID` rows in the naive output. Fixed by scoping both queries to the current label:
```python
df_train_cl = df_classes.query(label_column + " == @label and SUBJECT in @train_subjects")
df_test_cl = df_classes.query(label_column + " == @label and SUBJECT in @test_subjects")
```
This eliminates same-image duplication/contradiction entirely (verified: 0 duplicate `IMAGE_DATA_ID`
rows, 0 contradictory assignments after the fix). **Residual, smaller issue not fixed:** 51/816
subjects (~6%) still have *different* images (different visits) landing in different splits, because
the overlap-protection logic ([train_test_split.py:39-44](../../src/data_preparation/train_test_split.py#L39-L44))
only balances subjects present in **all three** classes at once, not subjects spanning exactly two of
the three. This is patient-level (not image-level) leakage across visits — smaller in scope than the
bug that's fixed, and left as a known gap rather than a silent one.

```python
import sys; sys.path.insert(0, 'src/data_preparation')
from ensemble_preparation import execute_ensemble_preparation
execute_ensemble_preparation('data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv',
                              'data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv', classes=[0,1,2])
```

Result (post-fix): train 861 / validation 455 / test 446 (1762 rows total — a clean partition of Step
6's 1762 input rows, spanning all three classes).

This `DATASET` column (train/validation/test, subject-level, seed 151) is fixed across every model
built from here on — CNNs, cognitive model, and ensemble.

## Step 8 — 2D slices ([run_slice_preparation.py](../../src/data_preparation/run_slice_preparation.py))

Steps 5b + 8 are folded into one runner. Launch it in the background (from the repo root):

```bash
cd /home/lucasthim/projects/phd/mmml-alzheimer-diagnosis
nohup uv run python -u src/data_preparation/run_slice_preparation.py \
    > data/mri/experiments/slice_prep_$(date +%Y%m%d_%H%M).log 2>&1 &
```

The full 100-slice-per-orientation sweep was the historical *slice search*; the dissertation already
resolved which slices to use, so the runner cuts only those 6:

- **AD (ADxCN):** coronal 43, axial 23, sagittal 26
- **MCI (MCIxCN):** coronal 70, axial 8, sagittal 50

**One combined call, one list of indices per orientation key.** `generate_slices`
([mri_batch_preparation.py:132](../../src/data_preparation/mri_batch_preparation.py#L132)) calls the
expensive `load_mri` once per `(image, orientation)` pair, then cuts *every* index in that orientation's
list from the same loaded volume. Combining AD's and MCI's index per orientation —
`{'coronal': [43, 70], 'axial': [23, 8], 'sagittal': [26, 50]}` — loads each volume 3 times (once per
orientation) instead of 6, halving ANTs loads (~21.9k for ~7279 images). Unique dict keys also sidestep
the duplicate-dict-key bug at [mri_batch_preparation.py:20-26](../../src/data_preparation/mri_batch_preparation.py#L20-L26)
(that bug was writing `'coronal'` twice as separate entries; one key holding a list is what it was
trying to express).

**In-code fixes already applied** (verified present in the current source, so no manual patch needed):

1. **Zero-pad** ([:208](../../src/data_preparation/mri_batch_preparation.py#L208)) — `slice['SLICE'] >= 10`
   (not `< 10`), so filenames are `coronal_70.npz`, not `coronal_070.npz`.
2. **Return value** ([:101](../../src/data_preparation/mri_batch_preparation.py#L101)) — returns
   `mri_reference_path+reference_file_name` (where the file is actually written, `data/reference/`), not
   the old `output_path+...` which pointed at a nonexistent path and raised `FileNotFoundError` when
   captured.
3. **Step 7 leakage fix** — the ensemble-reference merge ([:92](../../src/data_preparation/mri_batch_preparation.py#L92))
   fans out rows if `IMAGE_DATA_ID` isn't unique in `PROCESSED_ENSEMBLE_REFERENCE.csv`. Step 7's
   label-scoped split fix means it's now unique (0 duplicates), so no fan-out.

**DATASET backfill (`backfill_dataset_and_write_master`).** Only ~1762 of the 7279 sliced images get a
`DATASET` from the ensemble reference — the rest merge to `NaN`, because `PROCESSED_ENSEMBLE_REFERENCE.csv`
covers only subjects with *complete* cognitive data (Step 6's `dropna()`), and that skews by diagnosis
(AD patients more often have incomplete batteries). The runner sets `DATASET='train'` on every `NaN` row
that has a `MACRO_GROUP`, giving the CNN its full training cohort; `validation`/`test` are untouched, so
the CNN, cognitive, and ensemble models still share one evaluation set. (`return_sets()` already treated
`NaN` as "not validation/test" → train, so this only makes the implicit behavior explicit.) The runner
then writes `PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_<ts>.csv` — the training master; point
[train_all_cnns.py](../../src/model_training/train_all_cnns.py)'s `MRI_REFERENCE` at it.

Runtime: ~21.9k ANTs image loads over 7279 images — long; the background launch above is not optional.

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

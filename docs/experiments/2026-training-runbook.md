*Part of the [MMML-Alzheimer documentation](../README.md). A personalized, ordered checklist for the 2026 re-run: generate training-ready data from the reprocessed MRIs, then train the CNNs, cognitive model, and ensemble.*

# 2026 Training Runbook

This picks up from the general [running-experiments.md](running-experiments.md) runbook, but reflects
the *actual* state of this specific 2026 re-run and a growing list of corrections found by cross-checking
the historical notebooks (not just the source/docs) and by actually executing each step during this and
prior sessions. Read
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
- **Step 5b** — see corrected recipe below; done.
- **Step 6** — already run before this session, but with `--classes 0 1 2` (not the `0 1` shown
  originally in this doc) — `data/tabular/PREPROCESSED_ENSEMBLE_REFERENCE.csv` covers CN/AD/MCI
  (911/179/672 rows). Kept as-is rather than re-running with `0 1`, since one combined reference
  covering both cohorts is simpler for Step 8 (see below).
- **Step 7** — run with `classes=[0,1,2]` to match Step 6's scope → `data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv`
  (train 861 / validation 455 / test 446, across all three classes, after fixing the train/test leakage
  bug — see below).
- **Step 8** — done → `data/reference/PROCESSED_MRI_REFERENCE_ALL_ORIENTATIONS_20260710_1413.csv`
  (43722 rows = 2 classes × 3 orientations × 7287 images; `DATASET` = train for every image outside the
  ensemble's cognitively-complete cohort, validation/test unchanged); see corrected recipe below.

**Skippable:** Step 4 (`RAW_MRI_REFERENCE.csv`) — the 2026 rewrite of `ensemble_preprocessing.py` no
longer reconciles MRI-side `GROUP` against cognitive `DX` (single diagnosis source now), so nothing
downstream reads Step 4's output anymore.

Remaining: Step 9 → 10 → 11 → 12 → 13, detailed below.

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
satisfy the hardcoded column selection. **`SUBJECT` also needs backfilling** — it's part of the same
hardcoded column selection at [mri_batch_preparation.py:98-100](../../src/data_preparation/mri_batch_preparation.py#L98-L100)
but wasn't in the original candidate check; pull it from `COGNITIVE_DATA_PREPROCESSED.csv` alongside
the rest.

**Three more issues only surfaced by actually running this against the real files:**

1. `load_reference_table()` ([utils.py:82](../../src/utils/utils.py#L82)) unconditionally did
   `df['MACRO_GROUP'] = df['GROUP']` when `MACRO_GROUP` was missing — but the raw `REFERENCE.csv` files
   have *neither* column (`SUBJECT_IMAGE_ID, SUBJECT_ID, IMAGE_DATA_ID, IMAGE_PATH` only, since
   `mri_preprocessing.py` wasn't run with `-r/--mri-reference`), so this raised `KeyError: 'GROUP'` and
   `execute_mri_metadata_preprocessing` never got past reading the first file. Fixed to only backfill
   when `GROUP` is actually present: `if 'MACRO_GROUP' not in df.columns and 'GROUP' in df.columns:`.
2. **List the three dated folders newest-first**, not chronologically. The 20260707 batch was corrupted
   (see Step 5 above) and some of its images were silently reprocessed again on 20260709 under the
   *same* `IMAGE_DATA_ID` but a different `IMAGE_PATH` — 2676 such duplicate IDs across the three
   files. `execute_mri_metadata_preprocessing`'s dedup keeps the *first* occurrence per `IMAGE_DATA_ID`
   in input order, so feeding the folders oldest-first keeps the stale 20260707 path even where the
   physical file no longer exists on disk. Feeding them as `[20260709, 20260708, 20260707]` makes the
   corrected/latest reprocessing win.
3. Even newest-first, **7 of 7294 rows still point at files that don't exist anywhere** (verified with
   `find` — genuinely lost, not superseded). Drop rows failing `os.path.exists(IMAGE_PATH)` before
   handing the reference to Step 8, or `load_mri` crashes on them.

```python
import sys; sys.path.insert(0, 'src/data_preprocessing')
from mri_metadata_preprocessing import execute_mri_metadata_preprocessing
import pandas as pd, os

df = execute_mri_metadata_preprocessing(
    input=['/mnt/d/lucas/Downloads/preprocessed/20260709/REFERENCE.csv',   # newest first — see note 2 above
           '/mnt/d/lucas/Downloads/preprocessed/20260708/REFERENCE.csv',
           '/mnt/d/lucas/Downloads/preprocessed/20260707/REFERENCE.csv'],
    output='data/reference/PREPROCESSED_MRI_REFERENCE.csv',
    drop_cols=['FORMAT','TYPE','UNIQUE_IMAGE_ID','MODALITY','DOWNLOADED','SUBJECT_ID'])

df_cog = pd.read_csv('data/tabular/COGNITIVE_DATA_PREPROCESSED.csv', low_memory=False)
df_cog['IMAGE_DATA_ID'] = 'I' + df_cog['IMAGEUID'].astype(int).astype(str)
label_map = {0: 'CN', 1: 'AD', 2: 'MCI'}
df_cog['MACRO_GROUP'] = df_cog['DIAGNOSIS'].map(label_map)   # string labels — return_sets() needs this, not 0/1/2
df_cog['GROUP'] = df_cog['MACRO_GROUP']                       # unused downstream, kept for schema compatibility
df_cog['SEX'] = df_cog['MALE']                                 # unused downstream

df = df.merge(df_cog[['IMAGE_DATA_ID','SUBJECT','GROUP','MACRO_GROUP','SEX','AGE']], on='IMAGE_DATA_ID', how='left')
df = df[df['IMAGE_PATH'].apply(os.path.exists)].reset_index(drop=True)   # drop the 7 genuinely-missing files — note 3
df.to_csv('data/reference/PREPROCESSED_MRI_REFERENCE.csv', index=False)
```

Result: 7287 rows (7294 after concat/dedup, minus the 7 missing files), only 2 with no cognitive-data
match (`MACRO_GROUP`/`SUBJECT` null — negligible).

---

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

## Step 8 — 2D slices: only what's actually needed

The full 100-slice-per-orientation sweep was the historical *slice search*; the dissertation already
resolved which slices to use, so there's no need to repeat it:

- **AD (ADxCN):** coronal 43, axial 23, sagittal 26
- **MCI (MCIxCN):** coronal 70, axial 8, sagittal 50

Fix the inverted zero-pad bug first
([mri_batch_preparation.py:208](../../src/data_preparation/mri_batch_preparation.py#L208)) — otherwise
filenames come out as `coronal_070.npz` instead of `coronal_70.npz`. Already applied this session
(`slice['SLICE'] >= 10` instead of `< 10`).

**Best: one call total, one list of slice indices per orientation key** — not one call per orientation,
and not even one call per diagnosis class. `generate_slices` ([mri_batch_preparation.py:132](../../src/data_preparation/mri_batch_preparation.py#L132))
calls the expensive `load_mri` exactly once per `(image, orientation)` pair, then slices out *every*
index in that orientation's list from the same loaded volume. So combining AD's and MCI's index for the
same orientation into one list —
`{'coronal': [43, 70], 'axial': [23, 8], 'sagittal': [26, 50]}` — loads each 3D volume 3 times total
(once per orientation) instead of 6 (3 orientations × 2 separate class-scoped calls), halving total
ANTs image loads (~21.9k vs ~43.7k for ~7287 images). This does **not** hit the duplicate-dict-key bug
in [mri_batch_preparation.py:20-26](../../src/data_preparation/mri_batch_preparation.py#L20-L26) —
that bug is from writing the *same key* (`'coronal'`) twice as separate entries in one dict literal
(Python silently keeps only the last one); a single key holding a list of multiple indices is exactly
the pattern that bug was trying (and failing) to express in the first place:

```python
import sys; sys.path.insert(0, 'src/data_preparation')
from mri_batch_preparation import execute_mri_batch_preparation

out = execute_mri_batch_preparation(
    mri_reference_path='data/reference/PREPROCESSED_MRI_REFERENCE.csv',
    ensemble_reference_path='data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv',
    output_path='data/mri/processed/storage/',
    orientations={'coronal': [43, 70], 'axial': [23, 8], 'sagittal': [26, 50]})  # AD + MCI slices combined
```

This run actually used two calls (one dict per class, not the fully-combined version above — the
combined pattern was worked out only after this run was already partway through) — harmless but ~2x
slower than necessary; documented here so the next re-run uses the better version.

**Bug found while running this: the function's return value points at the wrong file.**
[mri_batch_preparation.py:101](../../src/data_preparation/mri_batch_preparation.py#L101) *saves* the
output CSV at `mri_reference_path`'s directory (e.g. `data/reference/`), but the function used to
*return* `output_path + reference_file_name` (e.g. `data/mri/processed/storage/...`) — a different,
nonexistent path. Fixed to return `mri_reference_path+reference_file_name`, matching where the file is
actually written. If you concat outputs by capturing the return value (as below), this bug means
`pd.read_csv(out)` raises `FileNotFoundError` — the file is one directory up, in `data/reference/`.

**Bug found while running this: the ensemble-reference merge fans out rows if `IMAGE_DATA_ID` isn't
unique in it.** [mri_batch_preparation.py:92](../../src/data_preparation/mri_batch_preparation.py#L92)
does `df_mri_processed_reference.merge(df_ensemble_reference[['IMAGE_DATA_ID','DATASET']], how='left')`
— if the Step 7 leakage bug (see Step 7 above) hasn't been fixed yet, duplicate `IMAGE_DATA_ID` rows in
the ensemble reference fan out into duplicate slice rows here too, inheriting the same contradictory
`DATASET` values. Fix Step 7 first; if you've already run Step 8 against a since-fixed ensemble
reference, you don't need to redo the (expensive) slicing — the per-image slice data doesn't depend on
the ensemble reference at all, only the `DATASET` column does. Cheaper recovery: dedupe the existing
per-class output CSVs down to one row per `(IMAGE_DATA_ID, ORIENTATION, SLICE)`, drop the (possibly
contradictory) `DATASET` column, and re-merge against the corrected `PROCESSED_ENSEMBLE_REFERENCE.csv`:

```python
import pandas as pd
key_cols = ['IMAGE_DATA_ID','ORIENTATION','SLICE']
df_ens = pd.read_csv('data/tabular/PROCESSED_ENSEMBLE_REFERENCE.csv')
df_ens['IMAGE_DATA_ID'] = 'I' + df_ens['IMAGEUID'].astype(str)

frames = []
for path in [...]:  # the per-class PROCESSED_MRI_REFERENCE_<ts>.csv outputs
    df = pd.read_csv(path)
    df = df[[c for c in df.columns if c != 'DATASET']].drop_duplicates(subset=key_cols)
    frames.append(df.merge(df_ens[['IMAGE_DATA_ID','DATASET']], on='IMAGE_DATA_ID', how='left'))

df_all = pd.concat(frames, ignore_index=True)
```

**Deliberate extension: give the CNN a larger training cohort than the ensemble uses.** Only 1762 of the
7287 sliced images have a `DATASET` label — the rest merged to `NaN` because `PROCESSED_ENSEMBLE_REFERENCE.csv`
only covers subjects with *complete* cognitive test data (Step 6's `dropna()`), and that drop skews by
diagnosis (in the unfilled data, only 13.1% of AD-labeled rows have a `DATASET` vs 30.4% of CN — AD
patients more often have incomplete cognitive batteries). `return_sets()`
([mri_train.py:329](../../src/model_training/mri_train.py#L329)) does
`df_mri_reference.query("DATASET not in ('validation','test')")` for its train split — pandas treats
`NaN` as trivially "not in" any list, so **the unlabeled rows were already silently landing in CNN
training** even without this step. Made that explicit rather than relying on the implicit pandas
behavior — set `DATASET='train'` on every row that still has no label (validation/test, tied to the
ensemble's cognitively-complete cohort, are untouched, so the CNN/cognitive/ensemble models still share
the same evaluation set):

```python
fillable = df_all['DATASET'].isna() & df_all['MACRO_GROUP'].notna()
df_all.loc[fillable, 'DATASET'] = 'train'
```

Result: train jumps from 5148 → 38322 rows (all three classes), validation (2724) and test (2664)
unchanged. The 12 rows with no `MACRO_GROUP` (2 images × 6 slices, no cognitive-data match at all) stay
`DATASET`-less — harmless, since `return_sets()` filters to `MACRO_GROUP in [0,1]` after remap regardless.

Runtime note: a single combined call still iterates all ~7287 images in the reference (the AD/MCI slice
choice doesn't filter which images get included, only which indices get cut per orientation) — expect
~21.9k ANTs image loads for the combined single-call version (~43.7k for the two-call version this run
actually used). Either way, run it in the background.

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

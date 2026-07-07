*Part of the [MMML-Alzheimer documentation](../README.md). Alphabetical glossary of the clinical, neuroimaging, ML, and project-specific terms a returning reader needs.*

# Glossary

This page defines the vocabulary that recurs across the docs and code: ADNI/clinical terms, the 3D MRI preprocessing methods, the ML and explainability stack, and the project's own ID and column conventions. Each entry is one to two lines; where a doc covers the term in depth, the term links there.

Three quick anchors a returning reader always wants:
- The diagnostic classes are `CN`, `MCI`, `AD`, encoded `CN=0, AD=1, MCI=2` for the cognitive `DIAGNOSIS` ([cognitive_tests_preprocessing.py#L100](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L100)). In 2026 `MACRO_GROUP` is a copy of `DIAGNOSIS` (single source). See [data-semantics.md](../data/data-semantics.md).
- Every model is **binary**; `2` (MCI) is excluded from AD-vs-CN and re-coded to `1` in its own MCI-vs-CN task.
- The full bug/stub/rot catalogue lives in [known-issues.md](known-issues.md).

---

## A

**AD (Alzheimer's Disease)** — Dementia class. ADNI's `DX` value `Dementia` and `DX_bl`/`GROUP` value `AD` both map to `AD`, encoded `1`. The positive class in the primary AD-vs-CN task.

**ADAS / ADAS-Cog** — Alzheimer's Disease Assessment Scale – Cognitive subscale. A cognitive-battery score where higher = worse. Three columns are kept: `ADAS11` (11-item total), `ADAS13` (13-item, adds delayed recall + digit cancellation), `ADASQ4` (Q4 delayed word-recall subscore). See [data-semantics.md](../data/data-semantics.md).

**ADNI (Alzheimer's Disease Neuroimaging Initiative)** — The multi-site longitudinal study that supplies every input: the merged tabular spreadsheet, the MRI metadata CSVs, and the raw `.nii` volumes. Data is gitignored; re-download via [data-acquisition.md](../data/data-acquisition.md).

**ADNIMERGE / `ADNIMERGE.csv`** — ADNI's master merged spreadsheet, one row per subject-visit (~115 columns; the same table TADPOLE distributed). The repo selects ~34 columns from it; everything else is dropped. Input to the tabular track. See [data-semantics.md](../data/data-semantics.md).

**Affine transform** — The registration mode used to align each MRI to the atlas: `type_of_transform='Affine'` with `grad_step=0.1` ([antspy_registration.py](../../src/data_preprocessing/antspy_registration.py)). `Similarity` and `Rigid` were also tested but Affine is the default. See [mri-preprocessing.md](../data/mri-preprocessing.md).

**ANTsPy / `antspyx`** — The Python wrapper around ANTs (Advanced Normalization Tools) used for image I/O, registration, and cropping. The only image library actually listed in `requirements.txt`. See [mri-preprocessing.md](../data/mri-preprocessing.md).

**APOE4** — Count of APOE ε4 alleles (0/1/2), ADNI's headline genetic AD-risk feature. **Not used anywhere** in this repo — `grep -rn "APOE" src/` returns nothing. Listed here so a returning reader does not assume it was modeled.

**Atlas / template (`atlas_t1.nii`)** — The fixed T1-weighted brain template every subject is registered to, at `data/mri/atlas/atlas_t1.nii`. Likely a generic MNI/ICBM T1 (inferred — the filename is the only evidence). Also anchors intensity standardization. See [data-acquisition.md](../data/data-acquisition.md), [mri-preprocessing.md](../data/mri-preprocessing.md).

**AUC** — Area under the ROC curve, the headline metric. Computed two ways: `roc_auc_score` on continuous scores in `compute_metrics_binary`, and trapezoidal `auc(fpr,tpr)` in the ROC routine ([base_evaluation.py](../../src/model_evaluation/base_evaluation.py)). See [evaluation.md](../modeling/evaluation.md).

**Axial** — The horizontal (top-down) MRI slice plane. One of the three orientations; the AD model uses slice `axial_23`, the MCI model `axial_8`. See [data-preparation.md](../data/data-preparation.md).

---

## C

**Captum** — PyTorch's model-interpretability library, used for the image-side XAI in [mri_explanation.py](../../src/model_explanation/mri_explanation.py): `DeepLift`, `NoiseTunnel`, `GuidedGradCam`, and `visualization`. See [explainability.md](../modeling/explainability.md).

**CDRSB** — Clinical Dementia Rating – Sum of Boxes (0–18), higher = worse. A strong single predictor; also used as a standalone `CDRSB` ensemble baseline. See [data-semantics.md](../data/data-semantics.md).

**CN (Cognitively Normal)** — The healthy control class, encoded `0`. ADNI's `SMC` (significant memory concern) collapses into `CN`. The negative class in both binary tasks.

**CNN (Convolutional Neural Network)** — The per-slice 2D image classifier. One CNN is trained per (orientation, slice). The factory in [neural_network.py](../../src/models/neural_network.py) builds `shallow_cnn`/`vgg*`/`resnet*` variants with `Conv2d(in_channels=1, ...)` and a single-logit head. See [models.md](../modeling/models.md).

**`CNN_SCORE`** — The sigmoid probability output of a per-slice CNN ([mri_train.py#L503](../../src/model_training/mri_train.py#L503)). Pivoted wide into ensemble columns named `CNN_SCORE_<ORIENTATION>_<SLICE>` (e.g. `CNN_SCORE_AXIAL_23`). See [data-semantics.md](../data/data-semantics.md).

**`COGTEST_SCORE`** — The tabular/cognitive model's positive-class probability, fed into the ensemble. The raw PyCaret column is `Score_1`; the rename to `COGTEST_SCORE` happens in an uncommitted notebook glue step (inferred — see [known-issues.md](known-issues.md)).

**`COGNITIVE_DATA_PREPROCESSED.csv`** — Output of the tabular track: cleaned cognitive + demographic table, one row per **visit**, with `DIAGNOSIS` encoded 0/1/2. The README mistakenly calls it `COGNITIVE_DATA_PROCESSED.csv` (see [known-issues.md](known-issues.md)). See [data-structure.md](../data/data-structure.md).

**`CONFLICT_DIAGNOSIS`** — Boolean column on the ensemble reference. **2026:** always `False`, because diagnosis has a single source (`MACRO_GROUP = DIAGNOSIS`); the column is kept only so downstream filters still run. **(pre-2026:** `True` where the cognitive `DIAGNOSIS` disagreed with the independent MRI `MACRO_GROUP`, and conflicting rows were dropped.) See [data-semantics.md](../data/data-semantics.md).

**Coronal** — The front-to-back (face-on) MRI slice plane. One of three orientations; AD uses `coronal_43`, MCI uses `coronal_70`.

**Cropping** — Step 4 of MRI preprocessing: a fixed center crop to a `100×100×100` box ([mri_crop.py](../../src/data_preprocessing/mri_crop.py), `cropping_box=100`). Valid only because every image was first affine-registered to the same grid. See [mri-preprocessing.md](../data/mri-preprocessing.md).

---

## D

**`DATASET`** — Split column with values `train`, `validation`, `test`, plus the assembly-time `train_cnn` (a CNN-training image outside the ensemble train set) and `NaN` (unassigned). Subject-level stratified split, seed `151` ([ensemble_preparation.py#L44](../../src/data_preparation/ensemble_preparation.py#L44)). Validation and test are fixed across all three modalities. See [data-preparation.md](../data/data-preparation.md).

**DeepLift** — A Captum gradient-attribution method for the CNNs, wrapped in `NoiseTunnel` (SmoothGrad) with a Gaussian-blurred baseline. Produces a local attribution map with positive and negative contributions. See [explainability.md](../modeling/explainability.md).

**DeLong test** — Statistical test of whether two ROC AUCs computed on the **same samples** differ. Because the AUCs are correlated, it estimates their covariance from mid-rank placement statistics ([de_long_evaluation.py](../../src/model_evaluation/de_long_evaluation.py), vendored from Netflix/vmaf). Note: uses `np.float`, removed in NumPy ≥ 1.24 — see [known-issues.md](known-issues.md). See [evaluation.md](../modeling/evaluation.md).

**Dementia** — ADNI's `DX` value for clinical dementia; mapped to `AD` ([cognitive_tests_preprocessing.py#L63](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L63)).

**`DIAGNOSIS`** — The cognitive/ADNIMERGE-side label (from `DX`), numeric `CN=0, AD=1, MCI=2`. The target for the tabular and ensemble models. Distinct from `DIAGNOSIS_BASELINE` and `MACRO_GROUP`. See [data-semantics.md](../data/data-semantics.md).

**`DIAGNOSIS_BASELINE`** — The baseline diagnosis (from `DX_bl`), kept as a **string** (`CN`/`MCI`/`AD`) and never numerically encoded. Do not treat it as a numeric target; it is dropped before modeling.

**`DummyModel`** — Single-column-threshold ensemble baselines (`CNNCoronal`, `CNNAxial`, `CNNSagittal`, `CNN3Slices`, `CNN3SlicesCogScore`, `CNN3SlicesDemographics`, `CDRSB`) used to benchmark the EBM/LR fusion ([ensemble_train.py](../../src/model_training/ensemble_train.py)). See [training.md](../modeling/training.md).

**`DX` / `DX_bl`** — ADNIMERGE's current (`DX`: `CN`/`MCI`/`Dementia`) and baseline (`DX_bl`: `CN`/`SMC`/`EMCI`/`LMCI`/`AD`) diagnosis fields. Renamed to `DIAGNOSIS` / `DIAGNOSIS_BASELINE`.

---

## E

**EBM (Explainable Boosting Machine)** — The glassbox ensemble classifier (`interpret.glassbox.ExplainableBoostingClassifier`) that fuses the CNN scores + cognitive/demographic features. Glassbox = inherently interpretable, exposing per-feature `names`/`scores` for local explanations and non-negative `feature_importances_` for global ones. See [models.md](../modeling/models.md), [explainability.md](../modeling/explainability.md).

**Ecog (Everyday Cognition)** — 14 ADNI questionnaire columns, self-report (`EcogPt*`) and study-partner (`EcogSP*`), higher = worse. **Dropped by default** (`exclude_ecog_tests=True`) along with `LDELTOTAL` and `DIGITSCOR`.

**EMCI (Early MCI)** — A `DX_bl`/`GROUP` value; collapses into `MCI`.

**Ensemble learning** — The project's fusion strategy: train modality-specific models (per-slice CNNs, a tabular cognitive model), then combine their probability outputs in a second-stage EBM/LR classifier. See [models.md](../modeling/models.md), [system-architecture.md](../architecture/system-architecture.md).

---

## F

**FAQ** — Functional Activities Questionnaire (0–30), a functional-impairment score where higher = worse. Kept as a model feature. See [data-semantics.md](../data/data-semantics.md).

**Focal loss** — The CNN training loss that down-weights easy examples to focus on hard/minority cases, addressing class imbalance. See [models.md](../modeling/models.md), [training.md](../modeling/training.md).

---

## G

**Global explanation** — Population/whole-model interpretation: which features matter across the dataset. Produced from EBM `feature_importances_` and LR `coef_` via `plot_global_explanations` / `show_feature_weights` ([ensemble_explanation.py](../../src/model_explanation/ensemble_explanation.py)). Image side has **no** global XAI. See [explainability.md](../modeling/explainability.md).

**Grad-CAM / Guided Grad-CAM** — A Captum saliency method targeting the CNN's last conv block (`net.features[-1]`), producing a local heatmap (positive sign only). See [explainability.md](../modeling/explainability.md).

**`GROUP` / `MACRO_GROUP`** — `GROUP` is the raw MRI-metadata diagnosis field (`CN`/`SMC`/`EMCI`/`LMCI`/`AD`/`MCI`); `MACRO_GROUP` is its 3-class collapse (`SMC→CN`, `EMCI`/`LMCI→MCI`) derived in [utils.py#L82](../../src/utils/utils.py#L82), later encoded `AD=1, CN=0, MCI=2`. The MRI-side label, analogous to the cognitive `DIAGNOSIS`. See [data-semantics.md](../data/data-semantics.md).

---

## I

**`IMAGE_DATA_ID`** — ADNI's **string** MRI id, `I` + IMAGEUID (e.g. `I261073`), used on the MRI-metadata side and as the ensemble feature-table index. Bridged to the integer `IMAGEUID` by stripping the leading `I`. See [data-semantics.md](../data/data-semantics.md).

**`IMAGEUID`** — ADNI's **integer** MRI id (e.g. `261073`), used on the cognitive side. NaN (visit with no MRI) is filled with the sentinel `999999` and cast to int ([cognitive_tests_preprocessing.py#L97](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L97)).

**Intensity standardization** — Step 1 of MRI preprocessing: clip voxels to the 0.02/99.8 percentiles, then linearly rescale onto the atlas's intensity range (hardcoded thresholds `(0.05545412003993988, 92.05744171142578)`, [mri_standardize.py](../../src/data_preprocessing/mri_standardize.py)). "Atlas-anchored" refers to intensity only, not spatial alignment. See [mri-preprocessing.md](../data/mri-preprocessing.md).

---

## L

**LDELTOTAL** — Logical Memory delayed recall (WMS-R story recall), higher = better. **Dropped by default** with the Ecog columns.

**Leakage (subject-level)** — Avoided by splitting on **subjects**, not rows: all visits/slices for one subject stay in the same fold (a subject has many visits and 3 slices per MRI). Also enforced by choosing the test threshold on validation, not test (`set_threshold_for_test`). See [data-preparation.md](../data/data-preparation.md), [evaluation.md](../modeling/evaluation.md).

**LMCI (Late MCI)** — A `DX_bl`/`GROUP` value; collapses into `MCI`.

**Local explanation** — Single-patient interpretation. Image side: `MRIExplainer` / `MRIDiagnosisExplainer` (per `image_id`, DeepLift + Guided Grad-CAM). Tabular side: `EnsembleExplainer.explain` (per `sample_id`, EBM `explain_local`). See [explainability.md](../modeling/explainability.md).

**Logistic Regression (LR)** — The linear ensemble fusion model run alongside the EBM; its `coef_` feeds the global feature-weight charts. See [models.md](../modeling/models.md).

---

## M

**`MACRO_GROUP`** — See `GROUP` / `MACRO_GROUP` above. The MRI-CNN target column (`mri_dataset.py`, `target_column='MACRO_GROUP'`).

**MCI (Mild Cognitive Impairment)** — The intermediate class, stored as `2`. **A separate task**: excluded from AD-vs-CN, and re-encoded to the positive class `1` in its own MCI-vs-CN run. See [data-semantics.md](../data/data-semantics.md).

**MMSE** — Mini-Mental State Exam (0–30), higher = better. A core cognitive feature. See [data-semantics.md](../data/data-semantics.md).

**MOCA** — Montreal Cognitive Assessment (0–30), higher = better. Kept as a model feature.

**MP-RAGE / MPRAGE** — Magnetization-Prepared Rapid Gradient-Echo, the T1-weighted MRI acquisition sequence ADNI uses; the MRI metadata arrives as `MPRAGE_REFERENCE.csv`. See [data-acquisition.md](../data/data-acquisition.md).

---

## N

**NIfTI / `.nii` / `.nii.gz`** — The neuroimaging volume format. Raw downloads are `.nii`; preprocessed volumes are gzip-compressed `.nii.gz`, named with a trailing `_I<id>` token that links the file back to its metadata. See [data-structure.md](../data/data-structure.md).

**`.npz` / `'arr_0'`** — The NumPy-archive format storing each preprocessed 2D MRI slice. Loaded with `np.load(path)['arr_0']` — the default unnamed-array key — yielding a single-channel `100×100` array. See [data-preparation.md](../data/data-preparation.md).

---

## P

**PyCaret** — The low-code AutoML wrapper used to set up, compare, and tune the tabular/cognitive model ([cognitive_tests_train.py](../../src/model_training/cognitive_tests_train.py)). Its `ignore_features` (`RID, SUBJECT, IMAGEUID, DATASET`) and dropped/categorical/numeric feature lists define the 23-feature model input. See [training.md](../modeling/training.md).

**`PTID`** — ADNI Participant ID, string form `XXX_S_XXXX` (site_S_roster); renamed to `SUBJECT`, the join key to MRI metadata. See `SUBJECT`.

---

## R

**RAVLT** — Rey Auditory Verbal Learning Test. Four columns: `RAVLT_immediate` (sum of trials 1–5, ↑ better), `RAVLT_learning` (trial 5 − trial 1, ↑ better), `RAVLT_forgetting` (↑ worse), `RAVLT_perc_forgetting` (↑ worse). See [data-semantics.md](../data/data-semantics.md).

**Registration** — Step 2 of MRI preprocessing: spatially align (warp) each volume onto the `atlas_t1.nii` template via ANTsPy, so anatomy sits at consistent coordinates and a fixed center crop is meaningful. Uses an affine transform. See [mri-preprocessing.md](../data/mri-preprocessing.md).

**ResNet** — A residual-connection CNN family available from the [neural_network.py](../../src/models/neural_network.py) factory (`resnet*`). See [models.md](../modeling/models.md).

**`RID`** — ADNI Roster ID, the integer subject key unique within ADNI. Kept in the table but an `ignore_features` in PyCaret. See [data-semantics.md](../data/data-semantics.md).

**ROC (Receiver Operating Characteristic)** — Sensitivity vs (1 − specificity) curve; the basis of AUC and the operating-point selection. The "optimal cutoff" is the ROC point closest to the top-left corner (0,1), not Youden's J. See [evaluation.md](../modeling/evaluation.md).

**`RUN_ID`** — A single (orientation, slice) CNN "model" id, `ORIENTATION_<SLICE>` (e.g. `coronal_50`), built in [ensemble_train.py#L20](../../src/model_training/ensemble_train.py#L20). Pivoted into the `CNN_SCORE_<RUN_ID>` columns. See [data-semantics.md](../data/data-semantics.md).

---

## S

**Sagittal** — The side-on (left-right profile) MRI slice plane. One of three orientations; AD uses `sagittal_26`, MCI uses `sagittal_50`.

**Skull stripping** — Step 3 of MRI preprocessing: remove non-brain voxels with the DeepBrain 3D U-Net (`Extractor`, mask threshold `probability=0.5`, [deepbrain_skull_strip.py](../../src/data_preprocessing/deepbrain_skull_strip.py)). An integrity check drops any volume the strip zeroed out entirely. See [mri-preprocessing.md](../data/mri-preprocessing.md).

**Slice** — A single 2D plane extracted from the 3D volume at a fixed index along one orientation. The unit a CNN classifies; stored as a `100×100` `.npz` array. See [data-preparation.md](../data/data-preparation.md).

**SMC (Significant Memory Concern)** — A `DX_bl`/`GROUP` value; collapses into `CN`.

**`SUBJECT`** — ADNI subject id, `XXX_S_XXXX` = site_S_roster (e.g. `002_S_4270`). From `PTID` on the cognitive side; parsed from filename tokens on the MRI side. The cross-modality join key and the unit of CV splitting. See [data-semantics.md](../data/data-semantics.md).

---

## T

**T1 / T1-weighted** — The MRI contrast type used throughout (the `atlas_t1.nii` template and the MP-RAGE acquisitions are T1-weighted). See [mri-preprocessing.md](../data/mri-preprocessing.md).

**TADPOLE** — The Alzheimer's prediction challenge that distributed the same merged ADNI spreadsheet; the column meanings here follow the TADPOLE/ADNI data dictionary. See [data-semantics.md](../data/data-semantics.md).

---

## V

**VGG** — A deep convolutional CNN family available from the [neural_network.py](../../src/models/neural_network.py) factory (`vgg*`, e.g. `vgg19_bn`); each exposes a `.features` Sequential targeted by Guided Grad-CAM. See [models.md](../modeling/models.md).

**`VISCODE`** — ADNI Visit code (`bl`, `m06`, `m12`, …) identifying which follow-up a row represents. Kept in the table but dropped before modeling. See [data-semantics.md](../data/data-semantics.md).

**Visit vs subject** — One ADNIMERGE row = one **visit**; a subject has many visits. This is why splits are subject-level (see Leakage). See [data-semantics.md](../data/data-semantics.md).

**Voxel** — A 3D pixel — the volumetric unit of a NIfTI MRI. Skull stripping zeros out non-brain voxels; the integrity check sums all voxels to detect an emptied volume.

---

## 999999

**`999999`** — The "no MRI for this visit" sentinel filled into `IMAGEUID` when ADNIMERGE leaves it blank ([cognitive_tests_preprocessing.py#L97](../../src/data_preprocessing/cognitive_tests_preprocessing.py#L97)). Filtered out wherever the tabular table is joined to images. See [data-semantics.md](../data/data-semantics.md).

---

## See also

- [data-semantics.md](../data/data-semantics.md) — full data dictionary and label scheme behind most terms here.
- [data-overview.md](../data/data-overview.md) — the ADNI data landscape and lineage.
- [mri-preprocessing.md](../data/mri-preprocessing.md) — the neuroimaging-method terms in pipeline context.
- [explainability.md](../modeling/explainability.md) — Captum, EBM, local vs global XAI in depth.
- [evaluation.md](../modeling/evaluation.md) — AUC, ROC, DeLong, and metric definitions.
- [known-issues.md](known-issues.md) — every bug, stub, and 4-year-rot hazard flagged above.

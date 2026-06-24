# CLAUDE.md

Guidance for AI agents (and humans) working in this repository.

## What this repo is

**Explainable Ensemble Learning for Alzheimer's Disease Diagnosis** — a PhD research
codebase for multi-modal machine learning on [ADNI](https://adni.loni.usc.edu) data. It
diagnoses Alzheimer's Disease (AD) by combining two modalities:

- **Tabular** — neuropsychological/cognitive tests + demographics (from `ADNIMERGE.csv`).
- **Imaging** — 3D T1 MP-RAGE brain MRI scans.

Each modality is preprocessed and modeled separately; their outputs are fused by a final
**ensemble** classifier (an Explainable Boosting Machine, EBM). Predictions are explained
both locally (per patient) and globally (population feature importance).

The classification tasks are **binary**: **AD vs. CN** (cognitively normal) and, separately,
**MCI vs. CN** (mild cognitive impairment). Labels are stored as `CN=0, AD=1, MCI=2` but
re-encoded to binary at training time.

## Documentation — read this first

Comprehensive docs live in [docs/](docs/). **Start at the hub: [docs/README.md](docs/README.md).**
Route to the right page instead of re-deriving from source:

- **Big picture:** [docs/architecture/system-architecture.md](docs/architecture/system-architecture.md)
- **Find a file/module:** [docs/architecture/repository-map.md](docs/architecture/repository-map.md)
- **Data — re-download:** [docs/data/data-acquisition.md](docs/data/data-acquisition.md)
- **Data — rebuild `ADNIMERGE.csv` (ADNIMERGE2 R package):** [docs/data/adnimerge2.md](docs/data/adnimerge2.md)
- **Data — on-disk layout & files:** [docs/data/data-structure.md](docs/data/data-structure.md)
- **Data — column/label meanings:** [docs/data/data-semantics.md](docs/data/data-semantics.md)
- **Data — MRI 3D pipeline:** [docs/data/mri-preprocessing.md](docs/data/mri-preprocessing.md)
- **Data — 3D→2D, augmentation, CV folds:** [docs/data/data-preparation.md](docs/data/data-preparation.md)
- **Models & loss:** [docs/modeling/models.md](docs/modeling/models.md)
- **Training:** [docs/modeling/training.md](docs/modeling/training.md)
- **Evaluation:** [docs/modeling/evaluation.md](docs/modeling/evaluation.md)
- **Explainability:** [docs/modeling/explainability.md](docs/modeling/explainability.md)
- **How experiments were tracked:** [docs/experiments/experiment-management.md](docs/experiments/experiment-management.md)
- **All notebooks:** [docs/experiments/notebooks-guide.md](docs/experiments/notebooks-guide.md)
- **Run an experiment end-to-end:** [docs/experiments/running-experiments.md](docs/experiments/running-experiments.md)
- **Glossary:** [docs/reference/glossary.md](docs/reference/glossary.md)
- **Bugs / stubs / gotchas:** [docs/reference/known-issues.md](docs/reference/known-issues.md)

## Repository layout

- [src/](src/) — the importable library (`mmmlalzheimer`, see [setup.py](setup.py)):
  - `data_preprocessing/` — cognitive + MRI metadata + 3D MRI preprocessing.
  - `data_preparation/` — 3D→2D slicing, augmentation, subject-level CV splits, ensemble alignment.
  - `model_training/` — MRI CNN training (offline + online), cognitive (PyCaret), ensemble (EBM).
  - `models/` — CNN architectures + focal loss.
  - `model_evaluation/` — metrics + DeLong test.
  - `model_explanation/` — Captum (MRI) + EBM (ensemble) explainers.
  - `utils/` — shared MRI/IO helpers.
  - `run/`, `experiment/` — **intended orchestration layer, but stubbed/empty.** Experiments run from notebooks, not these.
- [notebooks/](notebooks/) — ~50 notebooks: `early_mri_exploration/` (R&D), `mri_preprocessing/` (productionized), `final_studies/` (thesis results), and dated experiment-run notebooks. See [notebooks guide](docs/experiments/notebooks-guide.md).
- [scripts/](scripts/) — standalone data tools, e.g. [rebuild_adnimerge_from_adnimerge2.py](scripts/rebuild_adnimerge_from_adnimerge2.py) (rebuild `ADNIMERGE.csv` from the ADNIMERGE2 R package).
- `data/`, `models/`, `reports/` — **gitignored and empty.** Data lives outside the repo.
- [docs/](docs/) — this documentation.

## Working conventions & cautions

- **There is no central config and no experiment tracker (no MLflow/W&B).** Paths are
  hardcoded per module — mostly Google-Colab Drive paths (`/content/gdrive/MyDrive/Lucas_Thimoteo/...`).
  Experiment provenance lives in the **dated notebook names** and **output-file names**
  (`RESULTS_*.csv`, `PREDICTIONS_<ARCH>*.csv`, `.pth` models suffixed with `%m%d%Y_%H%M`).
- **Expect to fix hardcoded paths and incomplete deps before a fresh run.**
  [requirements.txt](requirements.txt) is incomplete (missing `deepbrain`, `torchvision`,
  `scipy`, `nibabel`, `tensorflow`, `captum`, `interpret`, `pycaret`; lists the deprecated
  `sklearn`). Full list of pitfalls: [Known Issues](docs/reference/known-issues.md).
- **The code targets Python + PyTorch + ANTsPy + DeepBrain (TensorFlow) + PyCaret + interpret(EBM) + Captum.**
- **`ADNIMERGE.csv` no longer exists upstream (2026).** ADNI ships the **ADNIMERGE2 R package** (~200 `.rda` tables); the flat `adnimerge` table was discontinued. Rebuild the CSV the pipeline needs with [scripts/rebuild_adnimerge_from_adnimerge2.py](scripts/rebuild_adnimerge_from_adnimerge2.py) — full workflow in [docs/data/adnimerge2.md](docs/data/adnimerge2.md).
- **Key data contract:** the pipeline is glued by reference CSVs (`*_REFERENCE.csv`), 2D slice
  `.npz` files (array under key `arr_0`, 100×100), and a `CNN_SCORE` column that flows into the
  ensemble feature table. See [Data Structure](docs/data/data-structure.md) and [Data Semantics](docs/data/data-semantics.md).
- **When you change behavior, keep the docs in sync** — update the relevant page in `docs/`
  and its `path:line` citations.
- **Communication:** lead with the answer; be concise and direct; reference files with
  markdown links (e.g. [mri_train.py](src/model_training/mri_train.py)), not bare paths.

## Maintaining these docs

The `docs/` pages cite source as `path:line`. If you refactor, those citations can drift —
verify a citation still points at the right code before relying on it, and update it if not.
The research notes the docs were built from were kept under `docs/_research/` during
generation; they may have been removed after the docs were finalized.

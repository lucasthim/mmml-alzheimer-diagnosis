# Explainable Ensemble Learning for Alzheimer's Disease Diagnosis

This project aims to develop an explainable multi modality machine learning tool for the diagnosis of the Alzheimer's Disease.
Data for this project was collect from the ADNI initiative (http://adni.loni.usc.edu).

We use two types of data:

1. Tabular data containing neuropsychological exams of a patient
2. Tabular data containing demographic information of a patient
3. 3D images of Magnetic Resonance Scans of the brain of a patient

We integrate each data type through separate preprocessing and machine learning pipelines, and join all of them in a final classifier. The name ensemble learning is due to the fact that more than one classifier is used to make intermediate and final predictions. 

We provide explanations for predictions at patient level (local explanations) and at the population level (global explanation).


## Environment Setup

Single [uv](https://docs.astral.sh/uv/) venv on Python 3.11, built from `requirements.txt`.

```bash
# install uv if missing
curl -LsSf https://astral.sh/uv/install.sh | sh

# from the repo root
uv venv --python 3.11
uv pip install -r requirements.txt        # needs git + GitHub access (deepbrain is a fork)
```

**Linux GPU box (e.g. RTX 4090 / Ada) — one extra step:**

```bash
bash scripts/setup_gpu_linux.sh
```

`requirements.txt` already installs `tensorflow[and-cuda]` on Linux, but the venv ends up
with both cu12 and cu13 NVIDIA wheels and TensorFlow 2.21 needs the cu12 `libcusolver.so.11`
preloaded or it silently falls back to CPU. `setup_gpu_linux.sh` installs that preload and
verifies both PyTorch and TF see the GPU. **Skip it on macOS** — not needed there.

Verify the GPU at any time:

```bash
uv run python -c "import torch; print('torch CUDA:', torch.cuda.is_available())"
uv run python -c "import tensorflow as tf; print('TF GPUs:', tf.config.list_physical_devices('GPU'))"
uv run pytest tests/   # smoke-tests skull stripping on a synthetic dummy volume
```

If `TF GPUs:` is empty on Linux, re-run `bash scripts/setup_gpu_linux.sh` and check
`nvidia-smi` reports a recent driver/CUDA (12.x+ for Ada).

> **Gotcha:** TensorFlow must be imported *before* `ants` in any process, or TF's
> `session.run` deadlocks during skull stripping (both ship an OpenMP runtime). The library
> modules already enforce this — replicate it in any new notebook/script.


## Steps to Run Experiments:

1. Download ADNIMERGE.csv file from http://adni.loni.usc.edu.
2. Preprocess ADNIMERGE.csv file.

    Run cognitive_tests_preprocessing.py. Outputfile will be COGNITIVE_DATA_PROCESSED.csv

3. Preprocess metadata from potential MRIs and ADNIMERGE.csv (metadata_preprocessing.py)
    
    Run metadata_preprocessing.py and outputfile will indicate the right MRIs files to download and preprocess.

<!-- 3. Select diagnosis classes of interest (AD,CN,MCI) (subject_preprocess.py). -->

4. Download MRIs from selected classes and subjects at http://adni.loni.usc.edu. MACRO_GROUPs used were AD,CN and MCI.

5. Preprocess MRIs (mri_preprocess.py).

6. Process MRIs (3D to 2D+Augmentation - mri_preparation).

7. Train/Validate/Test CNNs generating prediction probabilities (mri_train.py).

8. Train/Validate/Test ML models with cognitive tests (cognitive_train.py).

8. Train/Validate/Test Ensemble models with cognitive tests (ensemble_train.py).

10. Results Report



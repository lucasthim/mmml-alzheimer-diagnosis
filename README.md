# Explainable Ensemble Learning for Alzheimer's Disease Diagnosis

This project aims to develop an explainable multi modality machine learning tool for the diagnosis of the Alzheimer's Disease.
Data for this project was collect from the ADNI initiative (http://adni.loni.usc.edu).

We use two types of data:

1. Tabular data containing neuropsychological exams of a patient
2. Tabular data containing demographic information of a patient
3. 3D images of Magnetic Resonance Scans of the brain of a patient

We integrate each data type through separate preprocessing and machine learning pipelines, and join all of them in a final classifier. The name ensemble learning is due to the fact that more than one classifier is used to make intermediate and final predictions. 

We provide explanations for predictions at patient level (local explanations) and at the population level (global explanation).




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



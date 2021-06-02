# mmml-alzheimer-diagnosis
Multi Modality Machine Learning for Alzheimer's Disease Diagnosis

This project aims to develop a multi modality machine learning tool for the diagnosis of the Alzheimer's Disease.
Data for this project was collect from the ADNI initiative (http://adni.loni.usc.edu).


## Steps to Run Experiments:

1. Download ADNIMERGE.csv file from http://adni.loni.usc.edu.
2. Preprocess ADNIMERGE.csv file.

    Run cognitive_tests_preprocessing.py. Outputfile will be COGNITIVE_DATA_PROCESSED.csv

3. Preprocess metadata from potential MRIs and ADNIMERGE.csv (metadata_preprocessing.py)
    
    Run metadata_preprocessing.py and outputfile will indicate the right MRIs files to download and preprocess.

<!-- 3. Select diagnosis classes of interest (AD,CN,MCI) (subject_preprocess.py). -->

4. Download MRIs from select classes and subjects at http://adni.loni.usc.edu.

5. Preprocess MRIs (mri_preprocess.py).

6. Process MRIs (3D to 2D+Augmentation - mri_preparation).

7. Train/Validate/Test CNNs generating prediction probabilities (mri_train.py).

8. Train/Validate/Test ML models with cognitive tests (cognitive_train.py).

8. Train/Validate/Test Ensemble models with cognitive tests (ensemble_train.py).

10. Results Report



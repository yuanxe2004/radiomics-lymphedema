# radiomics-lymphedema
Radiomics and morphological analysis framework for lower-limb lymphedema staging using CE-MRL imaging.

## Pipeline
1. DICOM → NIfTI conversion
2. Segmentation (nnU-Net)
3. Radiomics feature extraction
4. Feature selection (ANOVA + LASSO)
5. Model training (SVM / RF / XGBoost)
6. Evaluation (ROC, AUC, SHAP)

## Dataset
Multicenter CE-MRL dataset (n = XXX patients)

## Usage
python preprocess/dicom_to_nifti.py
python radiomics/feature_extraction.py
python model/train.py

## Results
AUC: 0.98 (internal test)
External validation: AUC 0.96

## Requirements
see requirements.txt

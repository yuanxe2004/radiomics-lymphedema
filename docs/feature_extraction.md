# Feature Extraction

This repository includes a reproducible PyRadiomics configuration and an example feature-extraction script for subcutaneous-tissue VOIs. The manuscript feature-extraction environment was:

- PyRadiomics 3.0.1
- SimpleITK 2.5.2
- NIfTI images and masks pre-resampled to 1 x 1 x 1 mm
- fixed bin width = 25
- z-score intensity normalization enabled
- 3D extraction (`force2D = false`)
- mask label = 1
- image types: Original, Wavelet, and LoG with sigma values 1.0, 2.0, and 3.0 mm
- feature classes: first-order, GLCM, GLRLM, GLSZM, GLDM, and NGTDM
- shape features are intentionally disabled because morphology features are analyzed separately

The main modeling pipeline starts from a tabular Excel file of already extracted radiomics and morphology features. It does not include DICOM-to-NIfTI conversion, VOI segmentation, or manual segmentation review.

Use `configs/pyradiomics.yaml` with `scripts/extract_radiomics_features.py` as a template. Edit input/output paths for your local data before running.

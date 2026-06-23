# radiomics-lymphedema

Radiomics and morphology modeling pipeline for lower-limb lymphedema staging.

This release provides the main open-source workflow used after feature-table construction: tabular Excel loading, train/validation/internal-test/external-test splitting, feature selection, model selection, final model evaluation, plotting, wrong-case analysis, and model package export.

## Scope of This Repository

Included:

- main modeling code for development, internal testing, and independent external testing
- model packaging for later independent inference
- PyRadiomics configuration and an example feature-extraction script for already converted and segmented NIfTI images
- plotting and interpretability outputs including ROC, calibration, DCA, SHAP, feature-distribution plots, and wrong-case analysis

Not included:

- DICOM-to-NIfTI conversion
- VOI segmentation or manual segmentation review
- full image preprocessing pipeline before the NIfTI feature-extraction stage
- repeated nested cross-validation; nested CV was used as supplementary robustness analysis and is not included in the main release

Prospective testing in the manuscript was performed by applying the exported locked model packages to an independent prospective feature table. The main release focuses on the development, internal-test, and external-test workflow; prospective inference can be reproduced by loading the exported model packages and applying the same selected feature columns to a prospective table.

## Installation

```bash
git clone https://github.com/yuanxe2004/radiomics-lymphedema.git
cd radiomics-lymphedema
python -m venv .venv
.venv\Scriptsctivate
pip install -e .
```

Install the feature-extraction dependencies when using the PyRadiomics example:

```bash
pip install -e ".[radiomics]"
```

The manuscript environment used `scikit-learn==1.7.2`, `PyRadiomics==3.0.1`, and `SimpleITK==2.5.2`.

## Usage

Run from the repository root:

```bash
python -m radiomics_project --input-excel path	oeatures.xlsx --output-dir outputs
```

Set cohort labels according to your Excel file:

```bash
python -m radiomics_project ^
  --input-excel path	oeatures.xlsx ^
  --output-dir outputs ^
  --development-cohort-label development ^
  --external-test-cohort-label external
```

Equivalent environment variables:

```bash
set RADIOMICS_INPUT_EXCEL=path	oeatures.xlsx
set RADIOMICS_OUTPUT_DIR=outputs
set RADIOMICS_DEVELOPMENT_COHORT_LABEL=development
set RADIOMICS_EXTERNAL_TEST_COHORT_LABEL=external
python -m radiomics_project
```

## Input Table

The Excel file should contain:

- a stage/label column configured by `LABEL_COL` (default: `标签`)
- a patient/case ID column configured by `ID_COL` (default: `序号`)
- a side column configured by `SIDE_COL` (default: `肢体`)
- a cohort/center split column configured by `CENTER_COL` (default: `对应中心`)
- radiomics feature columns
- morphology feature columns listed in `MORPH_FEATURES`

The values in `CENTER_COL` are user-defined cohort labels. Map all model-development centers to the development label and the independent external test cohort to the external-test label. For example, if your manuscript data use centers 1-3 for development and center 4 for external testing, create or recode a cohort column so centers 1-3 share one value such as `development`, and center 4 has a value such as `external`.

## Feature Extraction

The main modeling pipeline starts from an Excel feature table. For radiomics feature extraction from preprocessed NIfTI images and masks, see:

- `configs/pyradiomics.yaml`
- `scripts/extract_radiomics_features.py`
- `docs/feature_extraction.md`

The provided configuration matches the manuscript feature-extraction settings: PyRadiomics 3.0.1, SimpleITK 2.5.2, fixed bin width 25, pre-resampled 1 x 1 x 1 mm NIfTI images, 3D extraction, Original/Wavelet/LoG image types, first-order and texture feature classes, and no radiomics shape features.

## Outputs

The output directory contains:

- an Excel workbook with metrics, wrong cases, split samples, cross-validation summaries, selected features, and all-case predictions
- ROC, calibration, DCA, SHAP, top-feature distribution, and wrong-case figures
- serialized model packages and manifest files for locked-model inference

Confidence intervals are computed with 1000 bootstrap iterations. When `ID_COL` is available, bootstrap resampling is performed at the patient/case cluster level and then expanded to limb-level rows.

## Privacy

Patient data, images, spreadsheets, generated outputs, and serialized models are intentionally ignored by Git. Do not commit private clinical data or derived patient-level artifacts.

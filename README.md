# radiomics-lymphedema

Radiomics and morphology modeling pipeline for lower-limb lymphedema staging.

The current package runs a binary-classification workflow from a tabular Excel feature file. It performs train/validation/internal-test/external-test splitting, radiomics and morphology feature selection, model selection with cross-validation, final model evaluation, plots, wrong-case analysis, and model package export.

## Installation

```bash
git clone https://github.com/yuanxe2004/radiomics-lymphedema.git
cd radiomics-lymphedema
python -m venv .venv
.venv\Scripts\activate
pip install -e .
```

Install PyRadiomics only if you also need feature extraction:

```bash
pip install -e ".[radiomics]"
```

## Usage

Run from the repository root:

```bash
python -m radiomics_project --input-excel path\to\features.xlsx --output-dir outputs
```

Equivalent root wrapper:

```bash
python main.py -i path\to\features.xlsx -o outputs
```

After installation, the console command is also available:

```bash
radiomics-lymphedema -i path\to\features.xlsx -o outputs
```

Environment variables are supported:

```bash
set RADIOMICS_INPUT_EXCEL=path\to\features.xlsx
set RADIOMICS_OUTPUT_DIR=outputs
python -m radiomics_project
```

## Input Table

The Excel file should contain:

- a stage/label column configured in `radiomics_project.config.LABEL_COL`
- a case ID column configured in `ID_COL`
- a side column configured in `SIDE_COL`
- a center column configured in `CENTER_COL`
- radiomics feature columns
- morphology feature columns listed in `MORPH_FEATURES`

The default workflow expects internal and external center labels configured by `INTERNAL_CENTER` and `EXTERNAL_CENTER`.

## Outputs

The output directory contains:

- an Excel workbook with metrics, wrong cases, split samples, cross-validation summaries, selected features, and all-case predictions
- ROC, calibration, DCA, SHAP, top-feature distribution, and wrong-case figures
- serialized model packages and manifest files

## Privacy

Patient data, images, spreadsheets, generated outputs, and serialized models are intentionally ignored by Git. Do not commit private clinical data or derived patient-level artifacts.

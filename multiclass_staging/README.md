# Radiomics-Lymphedema

Multiclass radiomics + morphology pipeline for lower-extremity lymphedema staging from CE-MRL-derived tabular features.

This repository provides a leakage-controlled three-class workflow for Stage I / II / III classification. It includes model development, center-based validation, bootstrap confidence intervals, ROC / DCA / calibration analysis, confusion matrices, SHAP or permutation-based importance, and a reusable saved-model wrapper.

## Repository structure

```text
radiomics-lymphedema/
├── configs/
│   ├── train_config.example.yaml
│   ├── predict_config.example.yaml
│   └── validation_config.example.yaml
├── data/
│   └── README.md
├── docs/
│   └── METHOD_SUMMARY.md
├── legacy/
│   ├── 3-0_full_pipeline.py
│   └── foresight_multiclass_wrapper_validation.py
├── models/
│   └── README.md
├── outputs/
│   └── README.md
├── scripts/
│   ├── train_multiclass.py
│   ├── predict_with_model.py
│   └── validate_foresight.py
├── src/
│   └── radiomics_lymphedema/
│       ├── config.py
│       ├── data.py
│       ├── explain.py
│       ├── features.py
│       ├── metrics.py
│       ├── models.py
│       ├── pipeline.py
│       ├── plots.py
│       ├── utils.py
│       ├── validation.py
│       └── wrapper.py
├── .gitignore
├── requirements.txt
└── README.md
```

## What this pipeline does

The training workflow is designed for a table such as `final_deduplicated_with_new_morph-2.xlsx` and performs the following operations:

1. Reads the Excel table and cleans the label column.
2. Builds a multiclass Stage I / II / III task.
3. Uses center-based splitting:
   - internal centers for train / validation / internal test;
   - external center as a fully independent external test set.
4. Performs group-aware splitting by patient or limb identifier to reduce leakage.
5. Detects radiomics and morphology feature families.
6. Runs train-only feature selection:
   - radiomics: variance filtering, ANOVA, Pearson filtering, LASSO;
   - morphology: imputation / scaling and Pearson filtering.
7. Compares candidate models by 5-fold CV on train + validation data.
8. Selects the best model by:
   - highest AUC;
   - then highest ACC;
   - then highest PPV.
9. Evaluates train, validation, internal test, and external test sets.
10. Exports metrics, predictions, selected features, wrong cases, plots, and the reusable model wrapper.

## Installation

Create a clean Python environment. Python 3.9–3.11 is recommended.

```bash
conda create -n lel-radiomics python=3.10 -y
conda activate lel-radiomics
pip install -r requirements.txt
```

For editable local development:

```bash
pip install -e .
```

## Configure training

Copy the example config and edit paths / columns:

```bash
cp configs/train_config.example.yaml configs/train_config.yaml
```

Important fields:

```yaml
excel_file: "C:/path/to/final_deduplicated_with_new_morph-2.xlsx"
output_dir: "C:/path/to/output/3-0"

label_col: "标签"
center_col: "对应中心"
id_col: "序号"

internal_center_value: "中心134"
external_center_value: "中心2"
```

The default task is:

```yaml
labels: [1, 2, 3]
label_names:
  1: "Stage I"
  2: "Stage II"
  3: "Stage III"
```

## Train the multiclass model

```bash
python scripts/train_multiclass.py --config configs/train_config.yaml
```

Typical outputs:

```text
output_dir/
├── figures/
│   ├── ROC_*.png
│   ├── DCA_*.png
│   ├── Calibration_*.png
│   ├── CM_*.png
│   ├── SHAP_Top10_*.png
│   └── Boxplot_*.png
├── models/
│   ├── best_model_wrapper.joblib
│   └── best_model_metadata.json
└── multiclass_sci_results_*.xlsx
```

## Use the saved model for prediction

After training, load the saved model wrapper:

```python
from radiomics_lymphedema.wrapper import load_best_model
import pandas as pd

wrapper = load_best_model(r"C:/path/to/output/3-0/models/best_model_wrapper.joblib")
new_df = pd.read_excel(r"C:/path/to/new_cases.xlsx")
result = wrapper.predict_dataframe(new_df)
result.to_excel(r"C:/path/to/predictions.xlsx", index=False)
```

Or use the CLI:

```bash
python scripts/predict_with_model.py ^
  --model "C:/path/to/best_model_wrapper.joblib" ^
  --input "C:/path/to/new_cases.xlsx" ^
  --output "C:/path/to/predictions.xlsx"
```

## Validate on a prospective / foresight table

```bash
python scripts/validate_foresight.py ^
  --model "C:/Users/ALIENWARE/Desktop/train_mould/code/radiomics/drawing/output/3-0/models/best_model_wrapper.joblib" ^
  --input "F:/foresight/前瞻性数据_影像组学按标签分层校正后.xlsx" ^
  --output-dir "F:/foresight/output_multiclass_validation" ^
  --label-col "label" ^
  --n-bootstrap 1000
```

This produces:

- metrics with bootstrap 95% CI;
- all predictions;
- wrong cases;
- multiclass ROC curve;
- DCA curve with Model / Treat all / Treat none;
- calibration curve;
- blue confusion matrix.

## Data privacy note

Do not commit clinical spreadsheets, image files, masks, trained models, or any identifiable data to GitHub. The `.gitignore` file is configured to exclude common data and model artifact formats.

## Citation / method reporting

For manuscript reporting, document:

- center split design;
- group-aware patient or limb splitting;
- train-only feature selection within each CV fold;
- model selection criterion;
- independent external test set;
- bootstrap confidence intervals;
- decision-curve and calibration analysis.

A concise method summary is provided in `docs/METHOD_SUMMARY.md`.

## Legacy scripts

The original monolithic script and the earlier prospective validation script are preserved in `legacy/` for traceability. The recommended GitHub-facing implementation is the modular code under `src/radiomics_lymphedema/` plus the entry scripts under `scripts/`.

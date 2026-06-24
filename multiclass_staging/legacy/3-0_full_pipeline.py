# -*- coding: utf-8 -*-
"""
SCI Multiclass Radiomics + Morphology Pipeline for LEL staging (Stage I / II / III)
=================================================================================

This script is a strict, leakage-controlled, center-split, three-class pipeline.
It is designed for a table such as:
    final_deduplicated_with_new_morph-2.xlsx

Core design
-----------
1) Data normalization and task construction
   - Drop missing labels
   - Force labels to int
   - Check center distribution and label distribution
   - Task: Stage I vs Stage II vs Stage III, i.e. LABELS = [1, 2, 3]

2) Center-based split
   - INTERNAL_CENTER_VALUE: used for train / validation / internal test
   - EXTERNAL_CENTER_VALUE: fully independent external test
   - Internal center is split into Train 60%, Validation 20%, Internal test 20%
   - Supports group-aware split by ID_COL to reduce patient/limb leakage

3) Automatic feature-system construction
   - Radiomics features: detected by common pyradiomics keywords
   - Morphology features: user-defined list plus optional keyword detection
   - Combined: selected radiomics + selected morphology
   - Non-feature columns are excluded by name/pattern/type

4) Train-driven feature selection
   - Radiomics: VarianceThreshold -> ANOVA -> Pearson correlation filtering -> LASSO
   - Morphology: StandardScaler-equivalent imputation/scaling -> Pearson filtering
   - In 5-fold CV, feature selection is refit inside every fold using fold-train only

5) Model training and selection
   - Models: Logistic Regression, Random Forest, SVM, optional XGBoost if installed
   - Candidate selection uses 5-fold CV on Train + Validation
   - Best rule: highest AUC mean -> highest ACC mean -> highest PPV mean

6) Outputs
   - Multi-sheet Excel workbook
   - Internal/external ROC, DCA, calibration curves
   - SHAP/permutation importance for global best model
   - Boxplots for top features

Author note
-----------
Before running, check the CONFIG block, especially:
    EXCEL_FILE, LABEL_COL, CENTER_COL, ID_COL,
    INTERNAL_CENTER_VALUE, EXTERNAL_CENTER_VALUE,
    MORPHOLOGY_FEATURES
"""

import os
import re
import json
import math
import time
import warnings
import joblib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split, StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    HAS_STRATIFIED_GROUP_KFOLD = False
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# CONFIG
# =============================================================================

EXCEL_FILE = r"C:\Users\ALIENWARE\OneDrive\work\分类\final_deduplicated_with_new_morph-2.xlsx"
OUTPUT_DIR = r"C:\Users\ALIENWARE\Desktop\train_mould\code\radiomics\drawing\output\3-0"

LABEL_COL = "标签"
CENTER_COL = "对应中心"
ID_COL = "序号"

INTERNAL_CENTER_VALUE = "中心134"
EXTERNAL_CENTER_VALUE = "中心2"

TASKS = [(1, 2, 3)]
LABELS = [1, 2, 3]
LABEL_NAMES = {1: "Stage I", 2: "Stage II", 3: "Stage III"}

# =============================================================================
# Model Encapsulation (Best Model Wrapper)
# =============================================================================
class BestModelWrapper:
    """
    Lightweight callable wrapper for the final global-best model.

    Usage after training:
        wrapper = BestModelWrapper.load(r".../models/best_model_wrapper.joblib")
        pred = wrapper.predict(new_df)
        prob = wrapper.predict_proba(new_df)
        out_df = wrapper.predict_dataframe(new_df)

    The saved joblib file stores a dictionary rather than the wrapper object itself,
    which makes loading more stable across scripts.
    """

    def __init__(
        self,
        model: Any,
        feature_cols: List[str],
        label_order: Optional[List[int]] = None,
        label_names: Optional[Dict[int, str]] = None,
        feature_set: Optional[str] = None,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.model = model
        self.feature_cols = list(feature_cols)
        self.label_order = list(label_order or LABELS)
        self.label_names = label_names or LABEL_NAMES
        self.feature_set = feature_set
        self.model_name = model_name
        self.task = task
        self.metadata = metadata or {}

    def _prepare_X(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise ValueError(f"Input data is missing required model features: {missing}")
            X_model = X.loc[:, self.feature_cols].copy()
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_cols):
                raise ValueError(
                    f"Array input has {arr.shape[1]} columns, but the model expects "
                    f"{len(self.feature_cols)} features. For safety, pass a pandas DataFrame "
                    f"with the original feature names."
                )
            X_model = pd.DataFrame(arr, columns=self.feature_cols)
        for c in self.feature_cols:
            X_model[c] = pd.to_numeric(X_model[c], errors="coerce")
        return X_model

    def predict_proba(self, X: Any) -> np.ndarray:
        X_model = self._prepare_X(X)
        if self.model_name == "XGB":
            prob = np.asarray(self.model.predict_proba(X_model), dtype=float)[:, :len(self.label_order)]
        else:
            prob = predict_proba_aligned(self.model, X_model, self.label_order)
        row_sum = prob.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0
        return prob / row_sum

    def predict(self, X: Any) -> np.ndarray:
        prob = self.predict_proba(X)
        return np.asarray(self.label_order, dtype=int)[np.argmax(prob, axis=1)]

    def predict_dataframe(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            out = X.copy()
        else:
            out = self._prepare_X(X).copy()
        prob = self.predict_proba(X)
        pred = np.asarray(self.label_order, dtype=int)[np.argmax(prob, axis=1)]
        out["pred_label"] = pred
        out["pred_name"] = [self.label_names.get(int(v), str(v)) for v in pred]
        out["max_prob"] = np.max(prob, axis=1)
        out["uncertainty"] = 1.0 - out["max_prob"]
        for j, lab in enumerate(self.label_order):
            out[f"prob_stage_{lab}"] = prob[:, j]
        return out

    def save(self, path: str) -> None:
        payload = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "features": self.feature_cols,  # backward-compatible alias
            "label_order": self.label_order,
            "label_names": self.label_names,
            "feature_set": self.feature_set,
            "model_name": self.model_name,
            "task": self.task,
            "metadata": self.metadata,
            "saved_at": timestamp(),
        }
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> "BestModelWrapper":
        obj = joblib.load(path)
        if isinstance(obj, BestModelWrapper):
            return obj
        return BestModelWrapper(
            model=obj["model"],
            feature_cols=obj.get("feature_cols", obj.get("features")),
            label_order=obj.get("label_order", LABELS),
            label_names=obj.get("label_names", LABEL_NAMES),
            feature_set=obj.get("feature_set"),
            model_name=obj.get("model_name"),
            task=obj.get("task"),
            metadata=obj.get("metadata", {}),
        )


def load_best_model(path: str) -> BestModelWrapper:
    """Load the saved global-best model wrapper from a .joblib file."""
    return BestModelWrapper.load(path)


def save_best_model_artifacts(wrapper: BestModelWrapper, model_dir: Path) -> Dict[str, str]:
    """Save the global-best model and a standalone calling script."""
    model_dir = ensure_dir(str(model_dir))
    model_path = model_dir / "best_model_wrapper.joblib"
    metadata_path = model_dir / "best_model_metadata.json"
    caller_path = model_dir / "best_model_predictor.py"

    wrapper.save(str(model_path))

    metadata = {
        "task": wrapper.task,
        "feature_set": wrapper.feature_set,
        "model_name": wrapper.model_name,
        "label_order": wrapper.label_order,
        "label_names": wrapper.label_names,
        "n_features": len(wrapper.feature_cols),
        "feature_cols": wrapper.feature_cols,
        "metadata": wrapper.metadata,
        "model_path": str(model_path),
        "caller_path": str(caller_path),
        "saved_at": timestamp(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    caller_source = (
        "# -*- coding: utf-8 -*-\n"
        "\"\"\"Standalone loader for the saved global-best LEL staging model.\"\"\"\n\n"
        "import joblib\n"
        "import numpy as np\n"
        "import pandas as pd\n\n"
        "def _get_estimator_classes(model):\n"
        "    if hasattr(model, 'classes_'):\n"
        "        return np.asarray(model.classes_)\n"
        "    if hasattr(model, 'named_steps'):\n"
        "        for step in reversed(list(model.named_steps.values())):\n"
        "            if hasattr(step, 'classes_'):\n"
        "                return np.asarray(step.classes_)\n"
        "    return None\n\n"
        "def _predict_proba_aligned(model, X, label_order):\n"
        "    prob = np.asarray(model.predict_proba(X), dtype=float)\n"
        "    model_classes = _get_estimator_classes(model)\n"
        "    if model_classes is None:\n"
        "        aligned = prob[:, :len(label_order)]\n"
        "    else:\n"
        "        model_classes = list(model_classes)\n"
        "        aligned = np.zeros((len(X), len(label_order)), dtype=float)\n"
        "        for j, lab in enumerate(label_order):\n"
        "            if lab in model_classes:\n"
        "                aligned[:, j] = prob[:, model_classes.index(lab)]\n"
        "    row_sum = aligned.sum(axis=1, keepdims=True)\n"
        "    row_sum[row_sum == 0] = 1.0\n"
        "    return aligned / row_sum\n\n"
        "class BestModelWrapper:\n"
        "    def __init__(self, payload):\n"
        "        self.model = payload['model']\n"
        "        self.feature_cols = list(payload.get('feature_cols', payload.get('features')))\n"
        "        self.label_order = list(payload.get('label_order', [1, 2, 3]))\n"
        "        self.label_names = payload.get('label_names', {1: 'Stage I', 2: 'Stage II', 3: 'Stage III'})\n"
        "        self.feature_set = payload.get('feature_set')\n"
        "        self.model_name = payload.get('model_name')\n"
        "        self.task = payload.get('task')\n"
        "        self.metadata = payload.get('metadata', {})\n\n"
        "    def _prepare_X(self, X):\n"
        "        if isinstance(X, pd.DataFrame):\n"
        "            missing = [c for c in self.feature_cols if c not in X.columns]\n"
        "            if missing:\n"
        "                raise ValueError(f'Input data is missing required model features: {missing}')\n"
        "            X_model = X.loc[:, self.feature_cols].copy()\n"
        "        else:\n"
        "            arr = np.asarray(X)\n"
        "            if arr.ndim == 1:\n"
        "                arr = arr.reshape(1, -1)\n"
        "            if arr.shape[1] != len(self.feature_cols):\n"
        "                raise ValueError(f'Array input has {arr.shape[1]} columns, but the model expects {len(self.feature_cols)} features.')\n"
        "            X_model = pd.DataFrame(arr, columns=self.feature_cols)\n"
        "        for c in self.feature_cols:\n"
        "            X_model[c] = pd.to_numeric(X_model[c], errors='coerce')\n"
        "        return X_model\n\n"
        "    def predict_proba(self, X):\n"
        "        X_model = self._prepare_X(X)\n"
        "        if self.model_name == 'XGB':\n"
        "            prob = np.asarray(self.model.predict_proba(X_model), dtype=float)[:, :len(self.label_order)]\n"
        "            row_sum = prob.sum(axis=1, keepdims=True)\n"
        "            row_sum[row_sum == 0] = 1.0\n"
        "            return prob / row_sum\n"
        "        return _predict_proba_aligned(self.model, X_model, self.label_order)\n\n"
        "    def predict(self, X):\n"
        "        prob = self.predict_proba(X)\n"
        "        return np.asarray(self.label_order, dtype=int)[np.argmax(prob, axis=1)]\n\n"
        "    def predict_dataframe(self, X):\n"
        "        out = X.copy() if isinstance(X, pd.DataFrame) else self._prepare_X(X).copy()\n"
        "        prob = self.predict_proba(X)\n"
        "        pred = np.asarray(self.label_order, dtype=int)[np.argmax(prob, axis=1)]\n"
        "        out['pred_label'] = pred\n"
        "        out['pred_name'] = [self.label_names.get(int(v), str(v)) for v in pred]\n"
        "        out['max_prob'] = np.max(prob, axis=1)\n"
        "        out['uncertainty'] = 1.0 - out['max_prob']\n"
        "        for j, lab in enumerate(self.label_order):\n"
        "            out[f'prob_stage_{lab}'] = prob[:, j]\n"
        "        return out\n\n"
        "def load_model(model_path='best_model_wrapper.joblib'):\n"
        "    payload = joblib.load(model_path)\n"
        "    if isinstance(payload, BestModelWrapper):\n"
        "        return payload\n"
        "    return BestModelWrapper(payload)\n\n"
        "if __name__ == '__main__':\n"
        "    # Example:\n"
        "    # wrapper = load_model(r'best_model_wrapper.joblib')\n"
        "    # new_df = pd.read_excel(r'new_cases.xlsx')\n"
        "    # result = wrapper.predict_dataframe(new_df)\n"
        "    # result.to_excel(r'predictions.xlsx', index=False)\n"
        "    pass\n"
    )
    with open(caller_path, "w", encoding="utf-8") as f:
        f.write(caller_source)

    return {
        "model_path": str(model_path),
        "metadata_path": str(metadata_path),
        "caller_path": str(caller_path),
    }


RANDOM_STATE = 255
N_SPLITS = 5
N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 2026

USE_GROUP_SPLIT = True
USE_STRATIFIED_SPLIT = True

# Feature detection / selection
REMOVE_RADIOMICS_SHAPE_FEATURES = True
AUTO_DETECT_MORPHOLOGY_BY_KEYWORDS = True

# Please revise this list according to the exact column names in your Excel.
# The script also supports keyword-based morphology detection if enabled.
MORPHOLOGY_FEATURES = [
    "muscle_volume_ratio",
    "bone_volume_ratio",
    "underskin_volume_ratio",
    "subcutaneous_volume_ratio",
    "muscle_CSA_ratio",
    "bone_CSA_ratio",
    "underskin_CSA_ratio",
    "subcutaneous_CSA_ratio",
    "volume_ratio",
    "CSA_ratio",
    "muscle_ratio",
    "bone_ratio",
    "underskin_ratio",
]

RADIOMICS_KEYWORDS = [
    "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm",
    "wavelet", "log-sigma", "log_sigma", "logarithm", "exponential",
    "gradient", "lbp", "square", "squareroot", "original_",
]
MORPHOLOGY_KEYWORDS = [
    "muscle", "bone", "underskin", "subcutaneous", "skin", "fat",
    "volume_ratio", "csa", "area", "ratio",
    "肌肉", "骨", "皮下", "脂肪", "面积", "体积", "比例",
]
EXCLUDE_NAME_PATTERNS = [
    "label", "标签", "center", "中心", "id", "序号", "编号", "patient", "患者",
    "limb", "肢", "side", "左右", "name", "姓名", "path", "路径", "file", "文件",
    "nii", "dcm", "dicom", "mask", "seg", "分割", "日期", "date",
]

VARIANCE_THRESHOLD = 1e-8
ANOVA_TOP_K = 50
PEARSON_THRESHOLD_RADIOMICS = 0.90
PEARSON_THRESHOLD_MORPHOLOGY = 0.90
LASSO_C = 0.05
LASSO_MAX_ITER = 20000
MIN_FEATURES_AFTER_LASSO_FALLBACK = 10
COMBINED_FINAL_PEARSON = False

FEATURE_SETS = ["Radiomics_only", "Morphology_only", "Combined"]
MODEL_NAMES = ["LR", "RF", "SVM"]
USE_XGBOOST_IF_AVAILABLE = True

# Plot / interpretation
RUN_ROC = True
RUN_DCA = True
RUN_CALIBRATION = True
RUN_SHAP = True
SHAP_MAX_BACKGROUND = 80
SHAP_MAX_CASES = 150
BOXPLOT_TOP_N = 10

# Excel sheet safety
MAX_EXCEL_ROWS_PER_SHEET = 1_000_000


# =============================================================================
# Utilities
# =============================================================================

def ensure_dir(path: str) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_sheet_name(name: str) -> str:
    return re.sub(r"[\\/*?:\[\]]", "_", name)[:31]


def safe_filename(name: str) -> str:
    name = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", str(name))
    return name.strip("_")


def as_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    out = df.loc[:, cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def format_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def format_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(low) or pd.isna(high):
        return f"{mean:.{digits}f} (NA-NA)"
    return f"{mean:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def normalize_center_value(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def has_any_keyword(name: str, keywords: List[str]) -> bool:
    s = str(name).lower()
    return any(k.lower() in s for k in keywords)


def is_shape_radiomics_feature(name: str) -> bool:
    s = str(name).lower()
    return ("shape" in s) or ("shape2d" in s) or ("shape_" in s)


def get_estimator_classes(model: Any) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    return np.asarray(LABELS)


def predict_proba_aligned(model: Any, X: pd.DataFrame, labels: List[int]) -> np.ndarray:
    prob = model.predict_proba(X)
    model_classes = list(get_estimator_classes(model))
    aligned = np.zeros((len(X), len(labels)), dtype=float)
    for j, lab in enumerate(labels):
        if lab in model_classes:
            aligned[:, j] = prob[:, model_classes.index(lab)]
        else:
            aligned[:, j] = 0.0
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    aligned = aligned / row_sum
    return aligned


def print_table(title: str, obj: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(obj)


# =============================================================================
# Data loading and splitting
# =============================================================================

def load_and_normalize_data(excel_file: str) -> pd.DataFrame:
    df = pd.read_excel(excel_file)

    missing_cols = [c for c in [LABEL_COL, CENTER_COL, ID_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Excel: {missing_cols}")

    before_n = len(df)
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df[df[LABEL_COL].isin(LABELS)].copy()
    df[CENTER_COL] = df[CENTER_COL].map(normalize_center_value)

    print(f"Loaded rows: {before_n}; rows after label cleaning and LABELS filter: {len(df)}")
    print_table("Label distribution", df[LABEL_COL].value_counts().sort_index())
    print_table("Center distribution", df[CENTER_COL].value_counts(dropna=False))
    print_table("Center x Label cross table", pd.crosstab(df[CENTER_COL], df[LABEL_COL]))

    internal_n = (df[CENTER_COL] == INTERNAL_CENTER_VALUE).sum()
    external_n = (df[CENTER_COL] == EXTERNAL_CENTER_VALUE).sum()
    if internal_n == 0:
        raise ValueError(f"No rows found for INTERNAL_CENTER_VALUE={INTERNAL_CENTER_VALUE!r}")
    if external_n == 0:
        raise ValueError(f"No rows found for EXTERNAL_CENTER_VALUE={EXTERNAL_CENTER_VALUE!r}")

    return df.reset_index(drop=True)


def _group_stratum_table(df: pd.DataFrame) -> pd.DataFrame:
    def label_pattern(s: pd.Series) -> str:
        vals = sorted(pd.Series(s).dropna().astype(int).unique().tolist())
        return "_".join(map(str, vals))

    tab = (
        df.groupby(ID_COL, dropna=False)[LABEL_COL]
        .agg(stratum_pattern=label_pattern, dominant_label=lambda s: int(pd.Series(s).mode().iloc[0]))
        .reset_index()
    )
    pattern_counts = tab["stratum_pattern"].value_counts()
    if len(pattern_counts) > 0 and pattern_counts.min() >= 2:
        tab["stratum"] = tab["stratum_pattern"]
    else:
        dominant_counts = tab["dominant_label"].value_counts()
        if len(dominant_counts) > 0 and dominant_counts.min() >= 2:
            tab["stratum"] = tab["dominant_label"].astype(str)
        else:
            tab["stratum"] = "all"
    return tab


def group_aware_holdout_split(
    df: pd.DataFrame,
    test_size: float,
    random_state: int,
    name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Group-aware holdout split. Stratification is approximate at group level."""
    if not USE_GROUP_SPLIT or ID_COL not in df.columns:
        stratify = df[LABEL_COL] if USE_STRATIFIED_SPLIT else None
        tr, te = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
        return tr.copy(), te.copy()

    groups_tab = _group_stratum_table(df)
    stratify = None
    if USE_STRATIFIED_SPLIT:
        counts = groups_tab["stratum"].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = groups_tab["stratum"]

    train_groups, test_groups = train_test_split(
        groups_tab[ID_COL],
        test_size=test_size,
        stratify=stratify,
        random_state=random_state,
    )
    train_groups = set(train_groups.tolist())
    test_groups = set(test_groups.tolist())
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected during {name}: {len(overlap)} overlapping groups")

    tr = df[df[ID_COL].isin(train_groups)].copy()
    te = df[df[ID_COL].isin(test_groups)].copy()
    return tr, te


def center_and_internal_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    internal = df[df[CENTER_COL] == INTERNAL_CENTER_VALUE].copy()
    external = df[df[CENTER_COL] == EXTERNAL_CENTER_VALUE].copy()

    train_df, temp_df = group_aware_holdout_split(
        internal,
        test_size=0.40,
        random_state=RANDOM_STATE,
        name="internal train/temp split",
    )
    val_df, internal_test_df = group_aware_holdout_split(
        temp_df,
        test_size=0.50,
        random_state=RANDOM_STATE + 1,
        name="internal validation/test split",
    )

    train_df = train_df.copy(); train_df["__split__"] = "train"
    val_df = val_df.copy(); val_df["__split__"] = "validation"
    internal_test_df = internal_test_df.copy(); internal_test_df["__split__"] = "internal_test"
    external = external.copy(); external["__split__"] = "external_test"

    if USE_GROUP_SPLIT and ID_COL in df.columns:
        split_groups = {
            "train": set(train_df[ID_COL].astype(str)),
            "validation": set(val_df[ID_COL].astype(str)),
            "internal_test": set(internal_test_df[ID_COL].astype(str)),
            "external_test": set(external[ID_COL].astype(str)),
        }
        for a in split_groups:
            for b in split_groups:
                if a >= b:
                    continue
                # External can theoretically contain the same patient ID if IDs are not globally unique across centers.
                # Internal split leakage is the strict requirement.
                if a != "external_test" and b != "external_test":
                    inter = split_groups[a].intersection(split_groups[b])
                    if inter:
                        raise RuntimeError(f"Group leakage between {a} and {b}: {len(inter)} groups")

    print_table("Split x Label cross table", pd.crosstab(
        pd.concat([train_df, val_df, internal_test_df, external])["__split__"],
        pd.concat([train_df, val_df, internal_test_df, external])[LABEL_COL]
    ))

    return train_df, val_df, internal_test_df, external


# =============================================================================
# Feature detection
# =============================================================================

def detect_feature_columns(df: pd.DataFrame) -> Dict[str, List[str]]:
    exact_morph = [c for c in MORPHOLOGY_FEATURES if c in df.columns]

    numeric_cols = []
    for c in df.columns:
        if c.startswith("__"):
            continue
        if c in [LABEL_COL, CENTER_COL, ID_COL]:
            continue
        if has_any_keyword(c, EXCLUDE_NAME_PATTERNS):
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        non_na_ratio = converted.notna().mean()
        if non_na_ratio >= 0.60:
            numeric_cols.append(c)

    morph_cols = set(exact_morph)
    if AUTO_DETECT_MORPHOLOGY_BY_KEYWORDS:
        for c in numeric_cols:
            if c in morph_cols:
                continue
            if has_any_keyword(c, MORPHOLOGY_KEYWORDS) and not has_any_keyword(c, RADIOMICS_KEYWORDS):
                morph_cols.add(c)

    radiomics_cols = []
    for c in numeric_cols:
        if c in morph_cols:
            continue
        if has_any_keyword(c, RADIOMICS_KEYWORDS):
            if REMOVE_RADIOMICS_SHAPE_FEATURES and is_shape_radiomics_feature(c):
                continue
            radiomics_cols.append(c)

    # Fallback: if radiomics keywords fail, use all remaining numeric features as radiomics candidates.
    if len(radiomics_cols) == 0:
        for c in numeric_cols:
            if c not in morph_cols:
                if REMOVE_RADIOMICS_SHAPE_FEATURES and is_shape_radiomics_feature(c):
                    continue
                radiomics_cols.append(c)

    out = {
        "radiomics": sorted(list(dict.fromkeys(radiomics_cols))),
        "morphology": sorted(list(dict.fromkeys(morph_cols))),
        "numeric_all": sorted(list(dict.fromkeys(numeric_cols))),
    }

    print_table("Detected feature counts", pd.Series({
        "radiomics_candidates": len(out["radiomics"]),
        "morphology_candidates": len(out["morphology"]),
        "numeric_feature_candidates_total": len(out["numeric_all"]),
    }))
    print("\nFirst 20 radiomics candidates:", out["radiomics"][:20])
    print("First 20 morphology candidates:", out["morphology"][:20])
    return out


# =============================================================================
# Feature selection
# =============================================================================

class SelectionRecorder:
    def __init__(self):
        self.rows: List[Dict[str, Any]] = []

    def add(
        self,
        task: str,
        context: str,
        fold: Optional[int],
        feature_set: str,
        domain: str,
        step: str,
        features: List[str],
        scores: Optional[Dict[str, float]] = None,
    ) -> None:
        scores = scores or {}
        for rank, feat in enumerate(features, start=1):
            self.rows.append({
                "task": task,
                "context": context,
                "fold": fold,
                "feature_set": feature_set,
                "domain": domain,
                "step": step,
                "n_features_at_step": len(features),
                "feature_rank": rank,
                "feature": feat,
                "score": scores.get(feat, np.nan),
            })


def pearson_filter(
    X_scaled_df: pd.DataFrame,
    cols: List[str],
    threshold: float,
    scores: Optional[Dict[str, float]] = None,
) -> List[str]:
    if len(cols) <= 1:
        return list(cols)
    scores = scores or {c: 0.0 for c in cols}
    ranked = sorted(cols, key=lambda c: (scores.get(c, 0.0), c), reverse=True)
    corr = X_scaled_df[cols].corr(method="pearson").abs().fillna(0.0)

    kept: List[str] = []
    for c in ranked:
        if not kept:
            kept.append(c)
            continue
        max_corr = corr.loc[c, kept].max() if kept else 0.0
        if max_corr <= threshold:
            kept.append(c)
    return kept


def _impute_and_scale(X: pd.DataFrame) -> Tuple[np.ndarray, SimpleImputer, StandardScaler]:
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xi = imputer.fit_transform(X)
    Xs = scaler.fit_transform(Xi)
    return Xs, imputer, scaler


def select_radiomics_features(
    train_df: pd.DataFrame,
    y: pd.Series,
    candidate_cols: List[str],
    recorder: SelectionRecorder,
    task: str,
    context: str,
    fold: Optional[int],
    feature_set: str,
) -> List[str]:
    cols = [c for c in candidate_cols if c in train_df.columns]
    X = as_numeric_df(train_df, cols)
    cols = [c for c in X.columns if X[c].notna().sum() > 0]
    X = X[cols]
    recorder.add(task, context, fold, feature_set, "radiomics", "00_candidates", cols)
    if len(cols) == 0:
        return []

    Xs, _, _ = _impute_and_scale(X)

    # 1) Variance threshold
    try:
        vt = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        Xv = vt.fit_transform(Xs)
        vt_cols = list(np.array(cols)[vt.get_support()])
    except Exception:
        vt_cols = cols
        Xv = Xs
    recorder.add(task, context, fold, feature_set, "radiomics", "01_variance", vt_cols)
    if len(vt_cols) == 0:
        return []

    # 2) ANOVA
    k = min(ANOVA_TOP_K, len(vt_cols))
    try:
        skb = SelectKBest(score_func=f_classif, k=k)
        Xa = skb.fit_transform(Xv, y)
        anova_cols = list(np.array(vt_cols)[skb.get_support()])
        raw_scores = skb.scores_
        score_map = {c: float(s) if np.isfinite(s) else 0.0 for c, s in zip(vt_cols, raw_scores)}
        anova_cols = sorted(anova_cols, key=lambda c: score_map.get(c, 0.0), reverse=True)
    except Exception:
        anova_cols = vt_cols[:k]
        score_map = {c: 0.0 for c in vt_cols}
        Xa = Xv[:, :k]
    recorder.add(task, context, fold, feature_set, "radiomics", "02_anova", anova_cols, score_map)
    if len(anova_cols) == 0:
        return []

    # 3) Pearson correlation filtering
    X_anova = pd.DataFrame(Xs[:, [cols.index(c) for c in anova_cols]], columns=anova_cols, index=train_df.index)
    pearson_cols = pearson_filter(X_anova, anova_cols, PEARSON_THRESHOLD_RADIOMICS, score_map)
    recorder.add(task, context, fold, feature_set, "radiomics", "03_pearson", pearson_cols, score_map)
    if len(pearson_cols) == 0:
        return []

    # 4) LASSO multinomial logistic selection
    X_lasso = X_anova[pearson_cols].values
    lasso_cols: List[str] = []
    try:
        lasso = LogisticRegression(
            penalty="l1",
            solver="saga",
            C=LASSO_C,
            class_weight="balanced",
            max_iter=LASSO_MAX_ITER,
            random_state=RANDOM_STATE,
            multi_class="multinomial",
        )
        lasso.fit(X_lasso, y)
        coefs = np.asarray(lasso.coef_)
        nonzero = np.any(np.abs(coefs) > 1e-8, axis=0)
        lasso_cols = list(np.array(pearson_cols)[nonzero])
        coef_scores = {c: float(np.mean(np.abs(coefs[:, i]))) for i, c in enumerate(pearson_cols)}
        lasso_cols = sorted(lasso_cols, key=lambda c: coef_scores.get(c, 0.0), reverse=True)
    except Exception:
        coef_scores = score_map
        lasso_cols = []

    if len(lasso_cols) == 0:
        fallback_n = min(MIN_FEATURES_AFTER_LASSO_FALLBACK, len(pearson_cols))
        lasso_cols = pearson_cols[:fallback_n]

    recorder.add(task, context, fold, feature_set, "radiomics", "04_lasso", lasso_cols, coef_scores)
    return lasso_cols


def select_morphology_features(
    train_df: pd.DataFrame,
    y: pd.Series,
    candidate_cols: List[str],
    recorder: SelectionRecorder,
    task: str,
    context: str,
    fold: Optional[int],
    feature_set: str,
) -> List[str]:
    cols = [c for c in candidate_cols if c in train_df.columns]
    X = as_numeric_df(train_df, cols)
    cols = [c for c in X.columns if X[c].notna().sum() > 0]
    X = X[cols]
    recorder.add(task, context, fold, feature_set, "morphology", "00_candidates", cols)
    if len(cols) == 0:
        return []

    Xs, _, _ = _impute_and_scale(X)
    X_scaled = pd.DataFrame(Xs, columns=cols, index=train_df.index)

    # Simple ANOVA scores are used only to decide which correlated feature to keep.
    try:
        f_scores, _ = f_classif(X_scaled, y)
        score_map = {c: float(s) if np.isfinite(s) else 0.0 for c, s in zip(cols, f_scores)}
    except Exception:
        score_map = {c: 0.0 for c in cols}

    pearson_cols = pearson_filter(X_scaled, cols, PEARSON_THRESHOLD_MORPHOLOGY, score_map)
    recorder.add(task, context, fold, feature_set, "morphology", "01_pearson", pearson_cols, score_map)
    return pearson_cols


def select_features_for_set(
    train_df: pd.DataFrame,
    y: pd.Series,
    feature_set: str,
    feature_pools: Dict[str, List[str]],
    recorder: SelectionRecorder,
    task: str,
    context: str,
    fold: Optional[int],
) -> List[str]:
    if feature_set == "Radiomics_only":
        selected = select_radiomics_features(
            train_df, y, feature_pools["radiomics"], recorder, task, context, fold, feature_set
        )
        return selected

    if feature_set == "Morphology_only":
        selected = select_morphology_features(
            train_df, y, feature_pools["morphology"], recorder, task, context, fold, feature_set
        )
        return selected

    if feature_set == "Combined":
        r_cols = select_radiomics_features(
            train_df, y, feature_pools["radiomics"], recorder, task, context, fold, feature_set
        )
        m_cols = select_morphology_features(
            train_df, y, feature_pools["morphology"], recorder, task, context, fold, feature_set
        )
        combined = list(dict.fromkeys(r_cols + m_cols))
        recorder.add(task, context, fold, feature_set, "combined", "05_combined", combined)

        if COMBINED_FINAL_PEARSON and len(combined) > 1:
            X = as_numeric_df(train_df, combined)
            Xs, _, _ = _impute_and_scale(X)
            X_scaled = pd.DataFrame(Xs, columns=combined, index=train_df.index)
            combined = pearson_filter(X_scaled, combined, PEARSON_THRESHOLD_RADIOMICS)
            recorder.add(task, context, fold, feature_set, "combined", "06_combined_final_pearson", combined)
        return combined

    raise ValueError(f"Unknown feature_set: {feature_set}")


# =============================================================================
# Models and metrics
# =============================================================================

def get_model_names() -> List[str]:
    names = list(MODEL_NAMES)
    if USE_XGBOOST_IF_AVAILABLE:
        try:
            import xgboost  # noqa: F401
            if "XGB" not in names:
                names.append("XGB")
        except Exception:
            pass
    return names


def build_model(model_name: str) -> Any:
    if model_name == "LR":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=5000,
                random_state=RANDOM_STATE,
                multi_class="multinomial",
            )),
        ])

    if model_name == "RF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500,
                max_depth=None,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

    if model_name == "SVM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])

    if model_name == "XGB":
        try:
            from xgboost import XGBClassifier
        except Exception as e:
            raise ImportError("XGBoost is not installed. Set USE_XGBOOST_IF_AVAILABLE=False or install xgboost.") from e
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                objective="multi:softprob",
                num_class=len(LABELS),
                n_estimators=300,
                learning_rate=0.03,
                max_depth=3,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                eval_metric="mlogloss",
                n_jobs=-1,
            )),
        ])

    raise ValueError(f"Unknown model_name: {model_name}")


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    total = cm.sum()
    if total == 0:
        return {"ACC": np.nan, "AUC": np.nan, "SENS": np.nan, "SPEC": np.nan, "PPV": np.nan, "NPV": np.nan, "CM": cm}

    TP = np.diag(cm).astype(float)
    FP = cm.sum(axis=0).astype(float) - TP
    FN = cm.sum(axis=1).astype(float) - TP
    TN = total - (TP + FP + FN)

    sens = np.divide(TP, TP + FN, out=np.full_like(TP, np.nan, dtype=float), where=(TP + FN) != 0)
    spec = np.divide(TN, TN + FP, out=np.full_like(TN, np.nan, dtype=float), where=(TN + FP) != 0)
    ppv = np.divide(TP, TP + FP, out=np.full_like(TP, np.nan, dtype=float), where=(TP + FP) != 0)
    npv = np.divide(TN, TN + FN, out=np.full_like(TN, np.nan, dtype=float), where=(TN + FN) != 0)

    acc = float(np.trace(cm) / total)

    try:
        if len(np.unique(y_true)) == len(LABELS):
            y_bin = label_binarize(y_true, classes=LABELS)
            auc_score = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        else:
            auc_score = np.nan
    except Exception:
        auc_score = np.nan

    out = {
        "ACC": acc,
        "AUC": auc_score,
        "SENS": float(np.nanmean(sens)),
        "SPEC": float(np.nanmean(spec)),
        "PPV": float(np.nanmean(ppv)),
        "NPV": float(np.nanmean(npv)),
        "CM": cm,
    }
    for idx, lab in enumerate(LABELS):
        out[f"SENS_class_{lab}"] = sens[idx]
        out[f"SPEC_class_{lab}"] = spec[idx]
        out[f"PPV_class_{lab}"] = ppv[idx]
        out[f"NPV_class_{lab}"] = npv[idx]
    return out


def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP,
    seed: int = BOOTSTRAP_SEED,
) -> Dict[str, Tuple[float, float, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    point = multiclass_metrics(y_true, y_pred, y_prob)

    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric_names = ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]
    boot_values = {m: [] for m in metric_names}

    if n == 0:
        return {m: (np.nan, np.nan, np.nan) for m in metric_names}

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = multiclass_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for key in metric_names:
            if not pd.isna(m[key]):
                boot_values[key].append(float(m[key]))

    ci = {}
    for key in metric_names:
        vals = np.asarray(boot_values[key], dtype=float)
        if len(vals) == 0:
            ci[key] = (float(point[key]), np.nan, np.nan)
        else:
            ci[key] = (float(point[key]), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))
    return ci


def summarize_fold_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    summary = {}
    for key in ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]:
        vals = pd.to_numeric(pd.Series([r.get(key, np.nan) for r in rows]), errors="coerce")
        summary[f"{key}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
        summary[f"{key}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        summary[f"{key}_mean_sd"] = format_mean_sd(summary[f"{key}_mean"], summary[f"{key}_sd"])
    return summary


# =============================================================================
# Cross-validation, final training, evaluation
# =============================================================================

def effective_n_splits(y: pd.Series, requested: int) -> int:
    counts = y.value_counts()
    min_class = int(counts.min()) if len(counts) else 0
    n = min(requested, min_class)
    if USE_GROUP_SPLIT:
        n = min(n, y.shape[0])
    if n < 2:
        raise ValueError("Not enough samples per class to perform cross-validation.")
    return n


def get_cv_splits(dev_df: pd.DataFrame, y: pd.Series) -> List[Tuple[np.ndarray, np.ndarray]]:
    n_splits = effective_n_splits(y, N_SPLITS)
    if USE_GROUP_SPLIT and ID_COL in dev_df.columns and HAS_STRATIFIED_GROUP_KFOLD:
        groups = dev_df[ID_COL].astype(str).values
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        try:
            return list(cv.split(dev_df, y, groups=groups))
        except Exception as e:
            print(f"WARNING: StratifiedGroupKFold failed ({e}). Falling back to StratifiedKFold.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return list(cv.split(dev_df, y))


def run_cv_candidate(
    dev_df: pd.DataFrame,
    feature_pools: Dict[str, List[str]],
    feature_set: str,
    model_name: str,
    task: str,
    global_recorder: SelectionRecorder,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    y = dev_df[LABEL_COL].astype(int).reset_index(drop=True)
    dev_df_reset = dev_df.reset_index(drop=True)
    splits = get_cv_splits(dev_df_reset, y)

    fold_rows: List[Dict[str, Any]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        fold_train = dev_df_reset.iloc[tr_idx].copy()
        fold_valid = dev_df_reset.iloc[va_idx].copy()
        y_tr = fold_train[LABEL_COL].astype(int)
        y_va = fold_valid[LABEL_COL].astype(int).values

        selected = select_features_for_set(
            fold_train,
            y_tr,
            feature_set,
            feature_pools,
            global_recorder,
            task,
            context="cv_fold_train_only",
            fold=fold_idx,
        )

        row_base = {
            "task": task,
            "feature_set": feature_set,
            "model": model_name,
            "fold": fold_idx,
            "n_train_fold": len(fold_train),
            "n_valid_fold": len(fold_valid),
            "n_selected_features": len(selected),
            "selected_features": "; ".join(selected),
        }
        if len(selected) == 0:
            row_base.update({k: np.nan for k in ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]})
            fold_rows.append(row_base)
            continue

        model = build_model(model_name)
        X_tr = as_numeric_df(fold_train, selected)
        X_va = as_numeric_df(fold_valid, selected)

        # XGBoost expects 0-based classes. To keep a uniform API, remap labels only for XGB.
        if model_name == "XGB":
            y_tr_fit = y_tr.map({lab: i for i, lab in enumerate(LABELS)}).astype(int)
            model.fit(X_tr, y_tr_fit)
            raw_prob = model.predict_proba(X_va)
            prob = raw_prob[:, :len(LABELS)]
            pred = np.asarray([LABELS[i] for i in np.argmax(prob, axis=1)])
        else:
            model.fit(X_tr, y_tr)
            prob = predict_proba_aligned(model, X_va, LABELS)
            pred = np.asarray(LABELS)[np.argmax(prob, axis=1)]

        m = multiclass_metrics(y_va, pred, prob)
        row_base.update({k: m[k] for k in ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]})
        fold_rows.append(row_base)

    summary = summarize_fold_metrics(fold_rows)
    summary.update({
        "task": task,
        "feature_set": feature_set,
        "model": model_name,
        "n_folds": len(splits),
        "n_dev_train_plus_val": len(dev_df),
        "mean_n_selected_features": float(pd.Series([r["n_selected_features"] for r in fold_rows]).mean()),
    })
    return summary, fold_rows


def fit_final_model(
    train_df_for_fit: pd.DataFrame,
    feature_pools: Dict[str, List[str]],
    feature_set: str,
    model_name: str,
    task: str,
    recorder: SelectionRecorder,
) -> Tuple[Any, List[str]]:
    y_fit = train_df_for_fit[LABEL_COL].astype(int)
    selected = select_features_for_set(
        train_df_for_fit,
        y_fit,
        feature_set,
        feature_pools,
        recorder,
        task,
        context="final_train_plus_validation",
        fold=None,
    )
    if len(selected) == 0:
        raise RuntimeError(f"No selected features for final model: {feature_set} / {model_name}")

    X_fit = as_numeric_df(train_df_for_fit, selected)
    model = build_model(model_name)
    if model_name == "XGB":
        y_fit_remap = y_fit.map({lab: i for i, lab in enumerate(LABELS)}).astype(int)
        model.fit(X_fit, y_fit_remap)
    else:
        model.fit(X_fit, y_fit)
    return model, selected


def evaluate_dataset(
    model: Any,
    selected_features: List[str],
    df_eval: pd.DataFrame,
    split_name: str,
    task: str,
    feature_set: str,
    model_name: str,
    bootstrap_seed_offset: int = 0,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    X = as_numeric_df(df_eval, selected_features)
    y_true = df_eval[LABEL_COL].astype(int).values

    if model_name == "XGB":
        raw_prob = model.predict_proba(X)
        prob = raw_prob[:, :len(LABELS)]
        pred = np.asarray([LABELS[i] for i in np.argmax(prob, axis=1)])
    else:
        prob = predict_proba_aligned(model, X, LABELS)
        pred = np.asarray(LABELS)[np.argmax(prob, axis=1)]

    metrics = multiclass_metrics(y_true, pred, prob)
    ci = bootstrap_metrics_ci(
        y_true, pred, prob,
        n_bootstrap=N_BOOTSTRAP,
        seed=BOOTSTRAP_SEED + bootstrap_seed_offset,
    )

    row = {
        "task": task,
        "feature_set": feature_set,
        "model": model_name,
        "dataset": split_name,
        "n": len(df_eval),
        "n_selected_features": len(selected_features),
    }
    for key in ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]:
        row[key] = metrics[key]
        mean, low, high = ci[key]
        row[f"{key}_95CI"] = format_ci(mean, low, high)
        row[f"{key}_low"] = low
        row[f"{key}_high"] = high

    for i, true_lab in enumerate(LABELS):
        for j, pred_lab in enumerate(LABELS):
            row[f"CM_true{true_lab}_pred{pred_lab}"] = int(metrics["CM"][i, j])

    pred_df = df_eval[[c for c in [ID_COL, CENTER_COL, LABEL_COL, "__split__"] if c in df_eval.columns]].copy()
    pred_df["task"] = task
    pred_df["feature_set"] = feature_set
    pred_df["model"] = model_name
    pred_df["pred_label"] = pred
    pred_df["correct"] = (pred == y_true)
    pred_df["max_prob"] = np.max(prob, axis=1)
    pred_df["uncertainty"] = 1.0 - pred_df["max_prob"]
    for j, lab in enumerate(LABELS):
        pred_df[f"prob_stage_{lab}"] = prob[:, j]

    return row, pred_df


def choose_best_candidate(candidate_df: pd.DataFrame) -> pd.Series:
    if candidate_df.empty:
        raise RuntimeError("No candidate model results available.")
    tmp = candidate_df.copy()
    for c in ["AUC_mean", "ACC_mean", "PPV_mean"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(-999)
    tmp = tmp.sort_values(
        by=["AUC_mean", "ACC_mean", "PPV_mean"],
        ascending=[False, False, False],
    )
    return tmp.iloc[0]


def choose_best_by_feature_set(candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fs in FEATURE_SETS:
        sub = candidate_df[candidate_df["feature_set"] == fs]
        if sub.empty:
            continue
        rows.append(choose_best_candidate(sub).to_dict())
    return pd.DataFrame(rows)


# =============================================================================
# Plots
# =============================================================================

def macro_roc_curve(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    y_bin = label_binarize(y_true, classes=LABELS)
    if y_bin.shape[1] != len(LABELS):
        raise ValueError("Unexpected binarized label shape.")
    fpr_dict, tpr_dict = {}, {}
    for i, lab in enumerate(LABELS):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr_dict[lab], tpr_dict[lab], _ = roc_curve(y_bin[:, i], y_prob[:, i])
    if not fpr_dict:
        raise ValueError("Cannot compute ROC because at least one class is missing.")
    all_fpr = np.unique(np.concatenate([fpr_dict[lab] for lab in fpr_dict]))
    mean_tpr = np.zeros_like(all_fpr)
    for lab in fpr_dict:
        mean_tpr += np.interp(all_fpr, fpr_dict[lab], tpr_dict[lab])
    mean_tpr /= len(fpr_dict)
    roc_auc = auc(all_fpr, mean_tpr)
    return all_fpr, mean_tpr, float(roc_auc)


def plot_roc_comparison(
    predictions: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    task: str,
) -> None:
    plt.figure(figsize=(6.4, 5.2), dpi=300)
    plotted = False
    for fs in FEATURE_SETS:
        sub = predictions[(predictions["__split__"] == dataset_name) & (predictions["feature_set"] == fs)]
        if sub.empty:
            continue
        y_true = sub[LABEL_COL].astype(int).values
        y_prob = sub[[f"prob_stage_{lab}" for lab in LABELS]].values
        try:
            fpr, tpr, roc_auc = macro_roc_curve(y_true, y_prob)
            plt.plot(fpr, tpr, linewidth=2.0, label=f"{fs} (AUC={roc_auc:.3f})")
            plotted = True
        except Exception:
            continue
    if not plotted:
        plt.close()
        return
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC: {dataset_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"ROC_{safe_filename(task)}_{dataset_name}.png")
    plt.close()


def multiclass_dca_values(y_true: np.ndarray, y_pred: np.ndarray, max_prob: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    """
    Confidence-threshold multiclass DCA.

    Strategy definition used here:
    - model: accept model predictions only when max class probability >= threshold
    - treat_none: accept no model predictions, net benefit = 0 across thresholds
    - treat_all: accept all model predictions, using correct predictions as benefit and
      incorrect predictions as harm. This preserves the original script's multiclass
      correct/incorrect DCA framing while drawing all/none as full reference curves.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    max_prob = np.asarray(max_prob, dtype=float)
    n = len(y_true)
    rows = []
    if n == 0:
        return pd.DataFrame(columns=["threshold", "net_benefit", "strategy"])

    correct_all = (y_pred == y_true)
    tp_all_rate = float(np.mean(correct_all))
    fp_all_rate = float(np.mean(~correct_all))

    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            continue
        weight = pt / (1.0 - pt)

        selected = max_prob >= pt
        tp = np.sum(selected & correct_all)
        fp = np.sum(selected & (~correct_all))
        nb_model = (tp / n) - (fp / n) * weight

        nb_none = 0.0
        nb_all = tp_all_rate - fp_all_rate * weight

        rows.append({"threshold": pt, "net_benefit": nb_model, "strategy": "model"})
        rows.append({"threshold": pt, "net_benefit": nb_none, "strategy": "treat_none"})
        rows.append({"threshold": pt, "net_benefit": nb_all, "strategy": "treat_all"})
    return pd.DataFrame(rows)



def plot_dca_comparison(
    predictions: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    task: str,
) -> None:
    thresholds = np.linspace(0.05, 0.95, 91)
    plt.figure(figsize=(6.4, 5.2), dpi=300)
    plotted = False
    none_plotted = False

    for fs in FEATURE_SETS:
        sub = predictions[(predictions["__split__"] == dataset_name) & (predictions["feature_set"] == fs)]
        if sub.empty:
            continue

        dca = multiclass_dca_values(
            sub[LABEL_COL].astype(int).values,
            sub["pred_label"].astype(int).values,
            sub["max_prob"].astype(float).values,
            thresholds,
        )
        if dca.empty:
            continue

        model_curve = dca[dca["strategy"] == "model"]
        none_curve = dca[dca["strategy"] == "treat_none"]
        all_curve = dca[dca["strategy"] == "treat_all"]

        plt.plot(model_curve["threshold"], model_curve["net_benefit"], linewidth=2.0, label=f"{fs} model")
        if not none_plotted:
            plt.plot(none_curve["threshold"], none_curve["net_benefit"], linestyle="--", linewidth=1.2, label="None")
            none_plotted = True
        plt.plot(all_curve["threshold"], all_curve["net_benefit"], linestyle=":", linewidth=1.4, label=f"{fs} all")
        plotted = True

    if not plotted:
        plt.close()
        return
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"DCA: {dataset_name}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"DCA_{safe_filename(task)}_{dataset_name}.png")
    plt.close()


def plot_calibration_comparison(
    predictions: pd.DataFrame,
    output_dir: Path,
    dataset_name: str,
    task: str,
) -> None:
    plt.figure(figsize=(6.4, 5.2), dpi=300)
    plotted = False
    for fs in FEATURE_SETS:
        sub = predictions[(predictions["__split__"] == dataset_name) & (predictions["feature_set"] == fs)]
        if sub.empty:
            continue
        y_correct = sub["correct"].astype(int).values
        conf = sub["max_prob"].astype(float).values
        try:
            frac_pos, mean_pred = calibration_curve(y_correct, conf, n_bins=8, strategy="quantile")
            plt.plot(mean_pred, frac_pos, marker="o", linewidth=2.0, label=fs)
            plotted = True
        except Exception:
            continue
    if not plotted:
        plt.close()
        return
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title(f"Calibration: {dataset_name}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"Calibration_{safe_filename(task)}_{dataset_name}.png")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_dir: Path, title: str, filename: str) -> None:
    plt.figure(figsize=(4.8, 4.2), dpi=300)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(LABELS))
    plt.xticks(ticks, [LABEL_NAMES.get(l, str(l)) for l in LABELS], rotation=30, ha="right")
    plt.yticks(ticks, [LABEL_NAMES.get(l, str(l)) for l in LABELS])
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = "white" if cm[i, j] > thresh else "black"
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", color=text_color)
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(output_dir / filename)
    plt.close()


def make_all_main_plots(
    all_feature_predictions: pd.DataFrame,
    metrics_rows: List[Dict[str, Any]],
    output_dir: Path,
    task: str,
) -> None:
    if RUN_ROC:
        for dataset_name in ["internal_test", "external_test"]:
            plot_roc_comparison(all_feature_predictions, output_dir, dataset_name, task)
    if RUN_DCA:
        for dataset_name in ["internal_test", "external_test"]:
            plot_dca_comparison(all_feature_predictions, output_dir, dataset_name, task)
    if RUN_CALIBRATION:
        for dataset_name in ["internal_test", "external_test"]:
            plot_calibration_comparison(all_feature_predictions, output_dir, dataset_name, task)

    for row in metrics_rows:
        if row.get("dataset") in ["internal_test", "external_test"]:
            cm = np.zeros((len(LABELS), len(LABELS)), dtype=int)
            for i, true_lab in enumerate(LABELS):
                for j, pred_lab in enumerate(LABELS):
                    cm[i, j] = int(row.get(f"CM_true{true_lab}_pred{pred_lab}", 0))
            plot_confusion_matrix(
                cm,
                output_dir,
                title=f"{row.get('feature_set')} / {row.get('model')} / {row.get('dataset')}",
                filename=f"CM_{safe_filename(task)}_{safe_filename(row.get('feature_set'))}_{safe_filename(row.get('model'))}_{safe_filename(row.get('dataset'))}.png",
            )


# =============================================================================
# SHAP / fallback importance and boxplots
# =============================================================================

def permutation_importance_simple(
    model: Any,
    X: pd.DataFrame,
    y_true: np.ndarray,
    selected_features: List[str],
    model_name: str,
    n_repeats: int = 10,
    seed: int = RANDOM_STATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base_prob = predict_proba_aligned(model, X, LABELS) if model_name != "XGB" else model.predict_proba(X)[:, :len(LABELS)]
    base_pred = np.asarray(LABELS)[np.argmax(base_prob, axis=1)]
    base_acc = np.mean(base_pred == y_true)
    rows = []
    for feat in selected_features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[feat] = rng.permutation(Xp[feat].values)
            prob = predict_proba_aligned(model, Xp, LABELS) if model_name != "XGB" else model.predict_proba(Xp)[:, :len(LABELS)]
            pred = np.asarray(LABELS)[np.argmax(prob, axis=1)]
            drops.append(base_acc - np.mean(pred == y_true))
        rows.append({"feature": feat, "importance": float(np.mean(drops)), "method": "permutation_accuracy_drop"})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def make_shap_or_importance_outputs(
    model: Any,
    selected_features: List[str],
    trainval_df: pd.DataFrame,
    output_dir: Path,
    task: str,
    feature_set: str,
    model_name: str,
) -> pd.DataFrame:
    if len(selected_features) == 0:
        return pd.DataFrame()

    X_all = as_numeric_df(trainval_df, selected_features)
    y_all = trainval_df[LABEL_COL].astype(int).values
    rng = np.random.default_rng(RANDOM_STATE)
    sample_n = min(SHAP_MAX_CASES, len(X_all))
    sample_idx = rng.choice(len(X_all), size=sample_n, replace=False) if len(X_all) > sample_n else np.arange(len(X_all))
    X_sample = X_all.iloc[sample_idx].copy()

    importance_df = pd.DataFrame()
    shap_done = False

    if RUN_SHAP:
        try:
            import shap
            bg_n = min(SHAP_MAX_BACKGROUND, len(X_all))
            bg_idx = rng.choice(len(X_all), size=bg_n, replace=False) if len(X_all) > bg_n else np.arange(len(X_all))
            background = X_all.iloc[bg_idx].copy()

            def predict_fn(data):
                data_df = pd.DataFrame(data, columns=selected_features)
                if model_name == "XGB":
                    return model.predict_proba(data_df)[:, :len(LABELS)]
                return predict_proba_aligned(model, data_df, LABELS)

            explainer = shap.Explainer(predict_fn, background)
            shap_values = explainer(X_sample)
            values = np.asarray(shap_values.values)

            if values.ndim == 3:
                # shape may be n_samples x n_features x n_classes
                if values.shape[1] == len(selected_features):
                    mean_abs = np.mean(np.abs(values), axis=(0, 2))
                    signed_mean = np.mean(values, axis=(0, 2))
                else:
                    mean_abs = np.mean(np.abs(values), axis=(0, 1))
                    signed_mean = np.mean(values, axis=(0, 1))
            elif values.ndim == 2:
                mean_abs = np.mean(np.abs(values), axis=0)
                signed_mean = np.mean(values, axis=0)
            else:
                raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

            importance_df = pd.DataFrame({
                "feature": selected_features,
                "importance": mean_abs,
                "signed_mean_shap": signed_mean,
                "method": "shap_mean_abs_multiclass",
            }).sort_values("importance", ascending=False)

            top = importance_df.head(10).iloc[::-1]
            plt.figure(figsize=(7.2, 5.0), dpi=300)
            plt.barh(top["feature"], top["importance"])
            plt.xlabel("Mean absolute SHAP value")
            plt.title(f"Top 10 SHAP features: {feature_set} / {model_name}")
            plt.tight_layout()
            plt.savefig(output_dir / f"SHAP_Top10_{safe_filename(task)}_{safe_filename(feature_set)}_{safe_filename(model_name)}.png")
            plt.close()
            shap_done = True
        except Exception as e:
            print(f"WARNING: SHAP failed ({e}). Falling back to permutation importance.")

    if not shap_done:
        importance_df = permutation_importance_simple(
            model,
            X_all,
            y_all,
            selected_features,
            model_name=model_name,
        )
        top = importance_df.head(10).iloc[::-1]
        plt.figure(figsize=(7.2, 5.0), dpi=300)
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Permutation accuracy drop")
        plt.title(f"Top 10 feature importance: {feature_set} / {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"Importance_Top10_{safe_filename(task)}_{safe_filename(feature_set)}_{safe_filename(model_name)}.png")
        plt.close()

    importance_df.insert(0, "task", task)
    importance_df.insert(1, "feature_set", feature_set)
    importance_df.insert(2, "model", model_name)
    return importance_df


def make_boxplots(
    df_source: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir: Path,
    task: str,
    feature_set: str,
    model_name: str,
) -> None:
    if importance_df.empty:
        return
    top_features = importance_df.head(BOXPLOT_TOP_N)["feature"].tolist()
    top_features = [f for f in top_features if f in df_source.columns]
    if not top_features:
        return

    for feat in top_features:
        plt.figure(figsize=(5.6, 4.2), dpi=300)
        data = []
        labels = []
        for lab in LABELS:
            vals = pd.to_numeric(df_source.loc[df_source[LABEL_COL] == lab, feat], errors="coerce").dropna().values
            if len(vals) == 0:
                vals = np.array([np.nan])
            data.append(vals)
            labels.append(LABEL_NAMES.get(lab, str(lab)))
        plt.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        plt.ylabel(feat)
        plt.title(f"{feat}")
        plt.tight_layout()
        plt.savefig(output_dir / f"Boxplot_{safe_filename(task)}_{safe_filename(feature_set)}_{safe_filename(model_name)}_{safe_filename(feat)}.png")
        plt.close()


# =============================================================================
# Excel writing
# =============================================================================

def write_sheet_safe(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str) -> None:
    sheet_name = safe_sheet_name(sheet_name)
    if df is None:
        df = pd.DataFrame()
    if len(df) <= MAX_EXCEL_ROWS_PER_SHEET:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        n_parts = math.ceil(len(df) / MAX_EXCEL_ROWS_PER_SHEET)
        for i in range(n_parts):
            part = df.iloc[i * MAX_EXCEL_ROWS_PER_SHEET:(i + 1) * MAX_EXCEL_ROWS_PER_SHEET]
            part.to_excel(writer, sheet_name=safe_sheet_name(f"{sheet_name}_{i+1}"), index=False)


# =============================================================================
# Main pipeline
# =============================================================================

def run() -> None:
    out_dir = ensure_dir(OUTPUT_DIR)
    plot_dir = ensure_dir(str(out_dir / "figures"))
    model_dir = ensure_dir(str(out_dir / "models"))
    workbook_path = out_dir / f"multiclass_sci_results_{timestamp()}.xlsx"

    df = load_and_normalize_data(EXCEL_FILE)
    train_df, val_df, internal_test_df, external_df = center_and_internal_split(df)
    trainval_df = pd.concat([train_df, val_df], axis=0).copy().reset_index(drop=True)

    feature_pools = detect_feature_columns(df)
    if len(feature_pools["radiomics"]) == 0 and len(feature_pools["morphology"]) == 0:
        raise ValueError("No numeric feature columns detected. Check EXCLUDE_NAME_PATTERNS and feature column names.")

    task = "Stage_1_vs_2_vs_3"
    model_names = get_model_names()
    print("\nModels to evaluate:", model_names)

    selection_recorder = SelectionRecorder()
    candidate_summaries: List[Dict[str, Any]] = []
    candidate_fold_rows: List[Dict[str, Any]] = []

    # -------------------------------------------------------------------------
    # 5-fold CV for every feature set x model
    # -------------------------------------------------------------------------
    for fs in FEATURE_SETS:
        if fs == "Morphology_only" and len(feature_pools["morphology"]) == 0:
            print("WARNING: Morphology_only skipped because no morphology columns were detected.")
            continue
        if fs == "Radiomics_only" and len(feature_pools["radiomics"]) == 0:
            print("WARNING: Radiomics_only skipped because no radiomics columns were detected.")
            continue
        if fs == "Combined" and (len(feature_pools["radiomics"]) + len(feature_pools["morphology"])) == 0:
            print("WARNING: Combined skipped because no features were detected.")
            continue

        for model_name in model_names:
            print(f"\nRunning CV: {fs} / {model_name}")
            try:
                summary, folds = run_cv_candidate(
                    trainval_df,
                    feature_pools,
                    fs,
                    model_name,
                    task,
                    selection_recorder,
                )
                candidate_summaries.append(summary)
                candidate_fold_rows.extend(folds)
                print(f"CV AUC: {summary.get('AUC_mean_sd')} | ACC: {summary.get('ACC_mean_sd')}")
            except Exception as e:
                print(f"WARNING: CV failed for {fs} / {model_name}: {e}")
                candidate_summaries.append({
                    "task": task,
                    "feature_set": fs,
                    "model": model_name,
                    "error": str(e),
                    "ACC_mean": np.nan,
                    "AUC_mean": np.nan,
                    "SENS_mean": np.nan,
                    "SPEC_mean": np.nan,
                    "PPV_mean": np.nan,
                    "NPV_mean": np.nan,
                })

    candidate_df = pd.DataFrame(candidate_summaries)
    fold_df = pd.DataFrame(candidate_fold_rows)

    # Keep only successfully evaluated candidates for selection.
    valid_candidates = candidate_df[pd.to_numeric(candidate_df.get("AUC_mean", pd.Series(dtype=float)), errors="coerce").notna()].copy()
    if valid_candidates.empty:
        raise RuntimeError("All candidate models failed. Check feature columns, labels, and dependencies.")

    global_best = choose_best_candidate(valid_candidates)
    best_by_feature_set = choose_best_by_feature_set(valid_candidates)
    print_table("Global best candidate", global_best[["feature_set", "model", "AUC_mean", "ACC_mean", "PPV_mean"]])
    print_table("Best candidate by feature set", best_by_feature_set[["feature_set", "model", "AUC_mean", "ACC_mean", "PPV_mean"]])

    # -------------------------------------------------------------------------
    # Final training for the best model of each feature set, for R/M/M+R plots
    # -------------------------------------------------------------------------
    all_feature_eval_rows: List[Dict[str, Any]] = []
    all_feature_predictions: List[pd.DataFrame] = []
    final_models: Dict[str, Dict[str, Any]] = {}
    used_feature_rows: List[Dict[str, Any]] = []

    eval_datasets = {
        "train": train_df,
        "validation": val_df,
        "internal_test": internal_test_df,
        "external_test": external_df,
    }

    for _, row in best_by_feature_set.iterrows():
        fs = row["feature_set"]
        model_name = row["model"]
        print(f"\nFitting final best-by-feature-set model: {fs} / {model_name}")
        final_model, selected = fit_final_model(
            trainval_df,
            feature_pools,
            fs,
            model_name,
            task,
            selection_recorder,
        )
        final_models[fs] = {"model": final_model, "features": selected, "model_name": model_name}
        for rank, feat in enumerate(selected, start=1):
            used_feature_rows.append({
                "task": task,
                "feature_set": fs,
                "model": model_name,
                "is_global_best": bool(fs == global_best["feature_set"] and model_name == global_best["model"]),
                "feature_rank": rank,
                "feature": feat,
                "feature_domain": "morphology" if feat in feature_pools["morphology"] else "radiomics",
            })

        for k, (split_name, eval_df) in enumerate(eval_datasets.items(), start=1):
            eval_row, pred_df = evaluate_dataset(
                final_model,
                selected,
                eval_df,
                split_name,
                task,
                fs,
                model_name,
                bootstrap_seed_offset=1000 * k + len(all_feature_eval_rows),
            )
            all_feature_eval_rows.append(eval_row)
            all_feature_predictions.append(pred_df)

    metrics_all_feature_sets = pd.DataFrame(all_feature_eval_rows)
    predictions_all_feature_sets = pd.concat(all_feature_predictions, axis=0, ignore_index=True) if all_feature_predictions else pd.DataFrame()

    # -------------------------------------------------------------------------
    # Global best specific outputs
    # -------------------------------------------------------------------------
    best_fs = str(global_best["feature_set"])
    best_model_name = str(global_best["model"])
    if best_fs not in final_models:
        global_model, global_features = fit_final_model(
            trainval_df, feature_pools, best_fs, best_model_name, task, selection_recorder
        )
    else:
        global_model = final_models[best_fs]["model"]
        global_features = final_models[best_fs]["features"]

    global_eval_rows = []
    global_pred_rows = []
    for k, (split_name, eval_df) in enumerate(eval_datasets.items(), start=1):
        eval_row, pred_df = evaluate_dataset(
            global_model,
            global_features,
            eval_df,
            split_name,
            task,
            best_fs,
            best_model_name,
            bootstrap_seed_offset=5000 + k,
        )
        global_eval_rows.append(eval_row)
        global_pred_rows.append(pred_df)
    metrics_global_best = pd.DataFrame(global_eval_rows)
    all_cases_predictions = pd.concat(global_pred_rows, axis=0, ignore_index=True)

    wrong_global_best = all_cases_predictions[~all_cases_predictions["correct"]].copy()
    split_samples = pd.concat([train_df, val_df, internal_test_df, external_df], axis=0, ignore_index=True)

    # Add CV result columns to metrics_global_best.
    for key in ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]:
        metrics_global_best[f"CV_{key}_mean_sd"] = global_best.get(f"{key}_mean_sd", "NA")
        metrics_global_best[f"CV_{key}_mean"] = global_best.get(f"{key}_mean", np.nan)
        metrics_global_best[f"CV_{key}_sd"] = global_best.get(f"{key}_sd", np.nan)
    metrics_global_best["global_best_feature_set"] = best_fs
    metrics_global_best["global_best_model"] = best_model_name
    metrics_global_best["n_train"] = len(train_df)
    metrics_global_best["n_validation"] = len(val_df)
    metrics_global_best["n_internal_test"] = len(internal_test_df)
    metrics_global_best["n_external_test"] = len(external_df)

    # -------------------------------------------------------------------------
    # Save callable global-best model wrapper
    # -------------------------------------------------------------------------
    best_model_wrapper = BestModelWrapper(
        model=global_model,
        feature_cols=global_features,
        label_order=LABELS,
        label_names=LABEL_NAMES,
        feature_set=best_fs,
        model_name=best_model_name,
        task=task,
        metadata={
            "selection_rule": "highest CV AUC_mean, then ACC_mean, then PPV_mean",
            "cv_auc_mean": float(global_best.get("AUC_mean", np.nan)),
            "cv_acc_mean": float(global_best.get("ACC_mean", np.nan)),
            "cv_ppv_mean": float(global_best.get("PPV_mean", np.nan)),
            "n_train": int(len(train_df)),
            "n_validation": int(len(val_df)),
            "n_internal_test": int(len(internal_test_df)),
            "n_external_test": int(len(external_df)),
        },
    )
    model_artifact_paths = save_best_model_artifacts(best_model_wrapper, model_dir)
    metrics_global_best["saved_model_path"] = model_artifact_paths["model_path"]
    metrics_global_best["saved_model_metadata_path"] = model_artifact_paths["metadata_path"]
    metrics_global_best["saved_model_caller_path"] = model_artifact_paths["caller_path"]

    # -------------------------------------------------------------------------
    # Plots and interpretation
    # -------------------------------------------------------------------------
    make_all_main_plots(predictions_all_feature_sets, all_feature_eval_rows, plot_dir, task)

    importance_df = make_shap_or_importance_outputs(
        global_model,
        global_features,
        trainval_df,
        plot_dir,
        task,
        best_fs,
        best_model_name,
    )
    make_boxplots(
        trainval_df,
        importance_df,
        plot_dir,
        task,
        best_fs,
        best_model_name,
    )

    feature_selection_df = pd.DataFrame(selection_recorder.rows)
    used_features_df = pd.DataFrame(used_feature_rows)

    # Basic run info and diagnostics.
    run_info = pd.DataFrame([
        {"key": "EXCEL_FILE", "value": EXCEL_FILE},
        {"key": "OUTPUT_DIR", "value": str(out_dir)},
        {"key": "LABEL_COL", "value": LABEL_COL},
        {"key": "CENTER_COL", "value": CENTER_COL},
        {"key": "ID_COL", "value": ID_COL},
        {"key": "INTERNAL_CENTER_VALUE", "value": INTERNAL_CENTER_VALUE},
        {"key": "EXTERNAL_CENTER_VALUE", "value": EXTERNAL_CENTER_VALUE},
        {"key": "LABELS", "value": json.dumps(LABELS, ensure_ascii=False)},
        {"key": "USE_GROUP_SPLIT", "value": USE_GROUP_SPLIT},
        {"key": "USE_STRATIFIED_SPLIT", "value": USE_STRATIFIED_SPLIT},
        {"key": "N_SPLITS", "value": N_SPLITS},
        {"key": "N_BOOTSTRAP", "value": N_BOOTSTRAP},
        {"key": "REMOVE_RADIOMICS_SHAPE_FEATURES", "value": REMOVE_RADIOMICS_SHAPE_FEATURES},
        {"key": "ANOVA_TOP_K", "value": ANOVA_TOP_K},
        {"key": "PEARSON_THRESHOLD_RADIOMICS", "value": PEARSON_THRESHOLD_RADIOMICS},
        {"key": "PEARSON_THRESHOLD_MORPHOLOGY", "value": PEARSON_THRESHOLD_MORPHOLOGY},
        {"key": "LASSO_C", "value": LASSO_C},
        {"key": "global_best_feature_set", "value": best_fs},
        {"key": "global_best_model", "value": best_model_name},
        {"key": "global_best_n_features", "value": len(global_features)},
        {"key": "saved_model_path", "value": model_artifact_paths["model_path"]},
        {"key": "saved_model_metadata_path", "value": model_artifact_paths["metadata_path"]},
        {"key": "saved_model_caller_path", "value": model_artifact_paths["caller_path"]},
    ])

    feature_pool_df = pd.DataFrame(
        [{"domain": "radiomics", "feature": f} for f in feature_pools["radiomics"]] +
        [{"domain": "morphology", "feature": f} for f in feature_pools["morphology"]]
    )

    # -------------------------------------------------------------------------
    # Excel workbook
    # -------------------------------------------------------------------------
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        write_sheet_safe(writer, metrics_global_best, "metrics_global_best")
        write_sheet_safe(writer, candidate_df, "candidate_5fold_cv_all_models")
        write_sheet_safe(writer, fold_df, "candidate_5fold_cv_folds")
        write_sheet_safe(writer, metrics_all_feature_sets, "metrics_all_feature_sets")
        write_sheet_safe(writer, feature_selection_df, "feature_selection_features")
        write_sheet_safe(writer, used_features_df, "used_features_by_task")
        write_sheet_safe(writer, wrong_global_best, "wrong_global_best")
        write_sheet_safe(writer, split_samples, "split_samples")
        write_sheet_safe(writer, all_cases_predictions, "all_cases_predictions")
        write_sheet_safe(writer, predictions_all_feature_sets, "all_feature_set_predictions")
        write_sheet_safe(writer, importance_df, "shap_or_importance")
        write_sheet_safe(writer, feature_pool_df, "detected_feature_pools")
        write_sheet_safe(writer, run_info, "run_info")

    print("\nDONE")
    print(f"Excel saved to: {workbook_path}")
    print(f"Figures saved to: {plot_dir}")
    print(f"Best model wrapper saved to: {model_artifact_paths['model_path']}")
    print(f"Best model metadata saved to: {model_artifact_paths['metadata_path']}")
    print(f"Best model caller saved to: {model_artifact_paths['caller_path']}")


if __name__ == "__main__":
    run()

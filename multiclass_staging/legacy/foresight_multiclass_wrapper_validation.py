# -*- coding: utf-8 -*-
"""
Prospective / foresight validation for one packaged three-class LEL staging model.

本版修改目的
------------
1. 调用已经保存的三分类最优模型封装文件：best_model_wrapper.joblib；
2. 将原脚本中的两个二分类任务（1v2、2v3）改为一个三分类任务（Stage I / II / III）；
3. 其他验证功能尽量保持不变：
   - 读取前瞻性 Excel 表格；
   - 输出整体预测结果、错误病例；
   - 计算 ACC / AUC / SENS / SPEC / PPV / NPV 及 bootstrap 95%CI；
   - 保留 Brier、校准指标、DCA 指标；
   - 绘制 ROC、DCA、Calibration 曲线；
   - 绘制混淆矩阵；
   - 对三分类最优模型做 SHAP Top10 分析；
   - 绘制 SHAP 排名最高的 radiomics 与 morphology 特征 boxplot。

重要说明
--------
- 本脚本默认前瞻性表格真实标签列为 LABEL_COL = "label"，标签取值为 1/2/3。
- 模型封装文件来自训练脚本输出目录：OUTPUT_DIR/models/best_model_wrapper.joblib。
- 该 joblib 推荐保存为 dict payload；如果保存为 BestModelWrapper 对象，本脚本也提供兼容类用于加载。
"""

import os
import re
import json
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import shap
except ImportError:
    shap = None

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 文件路径
# =============================================================================

MODEL_WRAPPER_PATH = r"C:\Users\ALIENWARE\Desktop\train_mould\code\radiomics\drawing\output\3-0\models\best_model_wrapper.joblib"

input_excel = r"F:\foresight\前瞻性数据_影像组学按标签分层校正后.xlsx"

output_dir = r"F:\foresight\output-shap-multiclass"
os.makedirs(output_dir, exist_ok=True)

output_metrics = os.path.join(output_dir, "foresight_multiclass_metrics_with_95CI.xlsx")
output_wrong_cases = os.path.join(output_dir, "foresight_multiclass_wrong_cases_and_predictions.xlsx")

plot_dir = os.path.join(output_dir, "plots_multiclass")
os.makedirs(plot_dir, exist_ok=True)

shap_dir = os.path.join(output_dir, "SHAP_multiclass")
os.makedirs(shap_dir, exist_ok=True)

boxplot_dir = os.path.join(output_dir, "SHAP_multiclass_boxplots")
os.makedirs(boxplot_dir, exist_ok=True)

output_shap_top10 = os.path.join(output_dir, "SHAP_multiclass_top10_features.xlsx")


# =============================================================================
# 基础配置
# =============================================================================

LABEL_COL = "label"
LABELS = [1, 2, 3]
LABEL_NAMES = {1: "Stage I", 2: "Stage II", 3: "Stage III"}
TASK_NAME = "Stage_1_vs_2_vs_3"

N_BOOTSTRAP = 2000
RANDOM_STATE = 255

DCA_THRESHOLDS = np.linspace(0.05, 0.95, 91)
DCA_REPORT_THRESHOLDS = [0.10, 0.20, 0.30, 0.40, 0.50]

CALIBRATION_BINS = 6
SHAP_MAX_BACKGROUND = 80
SHAP_MAX_CASES = 150
BOXPLOT_TOP_N = 10

MAX_EXCEL_ROWS_PER_SHEET = 1_000_000


# =============================================================================
# 绘图基础设置
# =============================================================================

def set_publication_plot_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 600,
        "axes.grid": False,
    })


def apply_axis_style(ax):
    ax.grid(False)
    ax.tick_params(axis="both", length=4, width=0.9)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def safe_filename(text):
    text = str(text)
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)
    text = text.replace("+", "plus")
    return text.strip("_")


def safe_sheet_name(name):
    return re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31]


set_publication_plot_style()


# =============================================================================
# 模型封装兼容加载
# =============================================================================

def get_estimator_classes(model: Any) -> Optional[np.ndarray]:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    return None


def predict_proba_aligned(model: Any, X: pd.DataFrame, label_order: List[int]) -> np.ndarray:
    prob = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = get_estimator_classes(model)

    if model_classes is None:
        aligned = prob[:, :len(label_order)]
    else:
        model_classes = list(model_classes)
        aligned = np.zeros((len(X), len(label_order)), dtype=float)
        for j, lab in enumerate(label_order):
            if lab in model_classes:
                aligned[:, j] = prob[:, model_classes.index(lab)]
            else:
                # XGBoost 三分类模型可能内部类别为 0/1/2，而外部标签为 1/2/3。
                zero_based = j
                if zero_based in model_classes:
                    aligned[:, j] = prob[:, model_classes.index(zero_based)]

    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return aligned / row_sum


class BestModelWrapper:
    """兼容训练脚本保存的 BestModelWrapper 对象或 dict payload。"""

    def __init__(
        self,
        model: Any = None,
        feature_cols: Optional[List[str]] = None,
        label_order: Optional[List[int]] = None,
        label_names: Optional[Dict[int, str]] = None,
        feature_set: Optional[str] = None,
        model_name: Optional[str] = None,
        task: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
    ):
        if payload is not None:
            model = payload["model"]
            feature_cols = payload.get("feature_cols", payload.get("features"))
            label_order = payload.get("label_order", LABELS)
            label_names = payload.get("label_names", LABEL_NAMES)
            feature_set = payload.get("feature_set")
            model_name = payload.get("model_name")
            task = payload.get("task")
            metadata = payload.get("metadata", {})

        if model is None:
            raise ValueError("BestModelWrapper 初始化失败：model 为空。")
        if feature_cols is None:
            raise ValueError("BestModelWrapper 初始化失败：feature_cols/features 为空。")

        self.model = model
        self.feature_cols = list(feature_cols)
        self.label_order = [int(x) for x in list(label_order or LABELS)]
        self.label_names = label_names or LABEL_NAMES
        self.feature_set = feature_set or "Global_best"
        self.model_name = model_name or "Unknown"
        self.task = task or TASK_NAME
        self.metadata = metadata or {}

    def _prepare_X(self, X: Any) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            missing = [c for c in self.feature_cols if c not in X.columns]
            if missing:
                raise ValueError(
                    "输入表格缺少模型所需特征列：\n"
                    + "\n".join([str(c) for c in missing])
                )
            X_model = X.loc[:, self.feature_cols].copy()
        else:
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            if arr.shape[1] != len(self.feature_cols):
                raise ValueError(
                    f"Array input has {arr.shape[1]} columns, but the model expects "
                    f"{len(self.feature_cols)} features. 建议传入带原始特征名的 pandas DataFrame。"
                )
            X_model = pd.DataFrame(arr, columns=self.feature_cols)

        for c in self.feature_cols:
            X_model[c] = pd.to_numeric(X_model[c], errors="coerce")
        return X_model

    def predict_proba(self, X: Any) -> np.ndarray:
        X_model = self._prepare_X(X)
        if str(self.model_name).upper() in ["XGB", "XGBOOST"]:
            prob = np.asarray(self.model.predict_proba(X_model), dtype=float)[:, :len(self.label_order)]
            row_sum = prob.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1.0
            return prob / row_sum
        return predict_proba_aligned(self.model, X_model, self.label_order)

    def predict(self, X: Any) -> np.ndarray:
        prob = self.predict_proba(X)
        return np.asarray(self.label_order, dtype=int)[np.argmax(prob, axis=1)]

    def predict_dataframe(self, X: Any) -> pd.DataFrame:
        out = X.copy() if isinstance(X, pd.DataFrame) else self._prepare_X(X).copy()
        prob = self.predict_proba(X)
        pred = np.asarray(self.label_order, dtype=int)[np.argmax(prob, axis=1)]
        out["pred_label"] = pred
        out["pred_name"] = [self.label_names.get(int(v), str(v)) for v in pred]
        out["max_prob"] = np.max(prob, axis=1)
        out["uncertainty"] = 1.0 - out["max_prob"]
        for j, lab in enumerate(self.label_order):
            out[f"prob_stage_{lab}"] = prob[:, j]
        return out


class LoadedBestModelWrapper(BestModelWrapper):
    pass


def load_model_wrapper(model_path: str) -> BestModelWrapper:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"找不到三分类模型封装文件：{model_path}")

    obj = joblib.load(model_path)

    if isinstance(obj, BestModelWrapper):
        return obj

    if isinstance(obj, dict):
        return BestModelWrapper(payload=obj)

    # 兼容其他对象：只要有 predict_proba / feature_cols 或 features，也尝试包装。
    feature_cols = getattr(obj, "feature_cols", getattr(obj, "features", None))
    if hasattr(obj, "predict_proba") and feature_cols is not None:
        return BestModelWrapper(
            model=obj,
            feature_cols=list(feature_cols),
            label_order=getattr(obj, "label_order", LABELS),
            label_names=getattr(obj, "label_names", LABEL_NAMES),
            feature_set=getattr(obj, "feature_set", "Global_best"),
            model_name=getattr(obj, "model_name", "Unknown"),
            task=getattr(obj, "task", TASK_NAME),
            metadata=getattr(obj, "metadata", {}),
        )

    raise TypeError(f"无法识别的模型封装类型：{type(obj)}")


# =============================================================================
# 读取 Excel 与模型
# =============================================================================

def load_input_table(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)

    if LABEL_COL not in df.columns:
        raise ValueError(f"找不到真实标签列: {LABEL_COL}")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    df = df[df[LABEL_COL].isin(LABELS)].copy()

    if len(df) == 0:
        raise ValueError("筛选 label ∈ [1, 2, 3] 后没有可用样本。")

    return df.reset_index(drop=True)


def check_features(df: pd.DataFrame, features: List[str]) -> None:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(
            "前瞻性表格缺少三分类模型所需特征列：\n"
            + "\n".join([str(c) for c in missing])
        )


# =============================================================================
# 工具函数：指标、CI、校准、DCA
# =============================================================================

def format_with_ci(point, low, high):
    if pd.isna(point):
        return ""
    if pd.isna(low) or pd.isna(high):
        return f"{point:.3f}"
    return f"{point:.3f} ({low:.3f}-{high:.3f})"


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def one_hot_labels(y_true: np.ndarray, labels: List[int]) -> np.ndarray:
    y_true = np.asarray(y_true).astype(int)
    out = np.zeros((len(y_true), len(labels)), dtype=float)
    for j, lab in enumerate(labels):
        out[:, j] = (y_true == lab).astype(float)
    return out


def multiclass_brier_score(y_true: np.ndarray, y_prob: np.ndarray, labels: List[int]) -> float:
    y_onehot = one_hot_labels(y_true, labels)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def calibration_slope_intercept_for_confidence(y_correct, confidence):
    y_correct = np.asarray(y_correct).astype(int)
    confidence = np.asarray(confidence, dtype=float)

    if len(np.unique(y_correct)) < 2:
        return np.nan, np.nan

    try:
        x = safe_logit(confidence).reshape(-1, 1)
        lr = LogisticRegression(
            penalty="l2",
            C=1e6,
            solver="lbfgs",
            max_iter=5000,
        )
        lr.fit(x, y_correct)
        slope = float(lr.coef_[0][0])
        intercept = float(lr.intercept_[0])
        return slope, intercept
    except Exception:
        return np.nan, np.nan


def calibration_in_the_large_for_confidence(y_correct, confidence):
    y_correct = np.asarray(y_correct).astype(int)
    confidence = np.asarray(confidence, dtype=float)

    obs = np.mean(y_correct)
    pred = np.mean(confidence)

    if obs <= 0 or obs >= 1 or pred <= 0 or pred >= 1:
        return np.nan

    return float(np.log(obs / (1 - obs)) - np.log(pred / (1 - pred)))


def eo_ratio_for_confidence(y_correct, confidence):
    observed = np.sum(y_correct)
    expected = np.sum(confidence)

    if expected <= 0:
        return np.nan

    return float(observed / expected)


def macro_multiclass_confusion_metrics(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()

    TP = np.diag(cm).astype(float)
    FP = cm.sum(axis=0).astype(float) - TP
    FN = cm.sum(axis=1).astype(float) - TP
    TN = total - (TP + FP + FN)

    sens = np.divide(TP, TP + FN, out=np.full_like(TP, np.nan, dtype=float), where=(TP + FN) != 0)
    spec = np.divide(TN, TN + FP, out=np.full_like(TN, np.nan, dtype=float), where=(TN + FP) != 0)
    ppv = np.divide(TP, TP + FP, out=np.full_like(TP, np.nan, dtype=float), where=(TP + FP) != 0)
    npv = np.divide(TN, TN + FN, out=np.full_like(TN, np.nan, dtype=float), where=(TN + FN) != 0)

    out = {
        "CM": cm,
        "SENS": float(np.nanmean(sens)),
        "SPEC": float(np.nanmean(spec)),
        "PPV": float(np.nanmean(ppv)),
        "NPV": float(np.nanmean(npv)),
    }

    for idx, lab in enumerate(labels):
        out[f"SENS_class_{lab}"] = sens[idx]
        out[f"SPEC_class_{lab}"] = spec[idx]
        out[f"PPV_class_{lab}"] = ppv[idx]
        out[f"NPV_class_{lab}"] = npv[idx]
        out[f"TP_class_{lab}"] = TP[idx]
        out[f"FP_class_{lab}"] = FP[idx]
        out[f"FN_class_{lab}"] = FN[idx]
        out[f"TN_class_{lab}"] = TN[idx]

    return out


def multiclass_auc_macro_ovr(y_true, y_prob, labels):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true)) < 2:
        return np.nan

    try:
        y_bin = label_binarize(y_true, classes=labels)
        if y_bin.shape[1] != len(labels):
            return np.nan
        if any(len(np.unique(y_bin[:, j])) < 2 for j in range(y_bin.shape[1])):
            return np.nan
        return float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
    except Exception:
        return np.nan


def calculate_multiclass_net_benefit(y_true, y_pred, max_prob, thresholds):
    """
    三分类 DCA 的实现逻辑：
    - 当 max_prob >= threshold 时，模型给出一个可采纳的分期判断；
    - 判断正确视为 benefit，判断错误视为 harm；
    - Treat none：不采纳任何模型判断，net benefit = 0；
    - Treat all：所有病例均采纳模型判断，net benefit 随 threshold 变化。
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    max_prob = np.asarray(max_prob, dtype=float)
    thresholds = np.asarray(thresholds, dtype=float)

    n = len(y_true)
    is_correct = (y_pred == y_true)
    net_benefits = []

    for pt in thresholds:
        selected = max_prob >= pt
        tp = np.sum(selected & is_correct)
        fp = np.sum(selected & (~is_correct))
        nb = (tp / n) - (fp / n) * (pt / (1 - pt))
        net_benefits.append(nb)

    return np.asarray(net_benefits)


def net_benefit_at_threshold_multiclass(y_true, y_pred, max_prob, threshold):
    return float(calculate_multiclass_net_benefit(y_true, y_pred, max_prob, np.array([threshold]))[0])


def calc_point_metrics_multiclass(y_true, y_pred, y_prob, labels):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    max_prob = np.max(y_prob, axis=1)
    y_correct = (y_true == y_pred).astype(int)

    cm_metrics = macro_multiclass_confusion_metrics(y_true, y_pred, labels)

    acc = accuracy_score(y_true, y_pred)
    auc_value = multiclass_auc_macro_ovr(y_true, y_prob, labels)
    brier = multiclass_brier_score(y_true, y_prob, labels)

    cal_slope, cal_intercept = calibration_slope_intercept_for_confidence(y_correct, max_prob)
    cal_large = calibration_in_the_large_for_confidence(y_correct, max_prob)
    eo = eo_ratio_for_confidence(y_correct, max_prob)

    nb_curve = calculate_multiclass_net_benefit(y_true, y_pred, max_prob, DCA_THRESHOLDS)

    out = {
        "ACC": float(acc),
        "AUC": auc_value,
        "SENS": cm_metrics["SENS"],
        "SPEC": cm_metrics["SPEC"],
        "PPV": cm_metrics["PPV"],
        "NPV": cm_metrics["NPV"],
        "Brier": brier,
        "EO_ratio": eo,
        "Calibration_slope": cal_slope,
        "Calibration_intercept": cal_intercept,
        "Calibration_in_the_large": cal_large,
        "DCA_mean_net_benefit": float(np.nanmean(nb_curve)),
        "DCA_max_net_benefit": float(np.nanmax(nb_curve)),
    }

    for t in DCA_REPORT_THRESHOLDS:
        out[f"DCA_net_benefit_at_{t:.2f}"] = net_benefit_at_threshold_multiclass(y_true, y_pred, max_prob, t)

    for k, v in cm_metrics.items():
        if k != "CM":
            out[k] = v

    cm = cm_metrics["CM"]
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            out[f"CM_true{true_lab}_pred{pred_lab}"] = int(cm[i, j])

    return out


def bootstrap_ci_metrics_multiclass(y_true, y_pred, y_prob, labels):
    rng = np.random.default_rng(RANDOM_STATE)

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)

    metrics_for_ci = [
        "ACC", "AUC", "SENS", "SPEC", "PPV", "NPV",
        "Brier", "EO_ratio",
        "Calibration_slope", "Calibration_intercept", "Calibration_in_the_large",
        "DCA_mean_net_benefit", "DCA_max_net_benefit",
    ]

    for t in DCA_REPORT_THRESHOLDS:
        metrics_for_ci.append(f"DCA_net_benefit_at_{t:.2f}")

    boot = {k: [] for k in metrics_for_ci}

    for _ in range(N_BOOTSTRAP):
        idx = rng.choice(np.arange(n), size=n, replace=True)

        try:
            m = calc_point_metrics_multiclass(
                y_true=y_true[idx],
                y_pred=y_pred[idx],
                y_prob=y_prob[idx, :],
                labels=labels,
            )
        except Exception:
            continue

        for k in metrics_for_ci:
            if k in m and not pd.isna(m[k]):
                boot[k].append(float(m[k]))

    ci = {}
    for k, values in boot.items():
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            ci[f"{k}_low"] = np.nan
            ci[f"{k}_high"] = np.nan
        else:
            ci[f"{k}_low"] = float(np.percentile(values, 2.5))
            ci[f"{k}_high"] = float(np.percentile(values, 97.5))

    return ci


def calc_metrics_with_ci_multiclass(
    y_true,
    y_pred,
    y_prob,
    labels,
    task_name,
    feature_set_name,
    model_name,
    model_path,
    n_samples,
):
    point = calc_point_metrics_multiclass(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        labels=labels,
    )

    ci = bootstrap_ci_metrics_multiclass(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        labels=labels,
    )

    metric_names = [
        "ACC", "AUC", "SENS", "SPEC", "PPV", "NPV",
        "Brier", "EO_ratio",
        "Calibration_slope", "Calibration_intercept", "Calibration_in_the_large",
        "DCA_mean_net_benefit", "DCA_max_net_benefit",
    ]

    for t in DCA_REPORT_THRESHOLDS:
        metric_names.append(f"DCA_net_benefit_at_{t:.2f}")

    row = {
        "task": task_name,
        "feature_set": feature_set_name,
        "model_name": model_name,
        "model_path": model_path,
        "n_samples": n_samples,
        "labels": str(labels),
    }

    for metric in metric_names:
        row[f"{metric}_with_95CI"] = format_with_ci(
            point.get(metric, np.nan),
            ci.get(f"{metric}_low", np.nan),
            ci.get(f"{metric}_high", np.nan),
        )
        row[f"{metric}_raw"] = point.get(metric, np.nan)
        row[f"{metric}_low"] = ci.get(f"{metric}_low", np.nan)
        row[f"{metric}_high"] = ci.get(f"{metric}_high", np.nan)

    # 混淆矩阵与每类指标也写入 metrics 表。
    for true_lab in labels:
        for pred_lab in labels:
            row[f"CM_true{true_lab}_pred{pred_lab}"] = point.get(f"CM_true{true_lab}_pred{pred_lab}", np.nan)

    for lab in labels:
        for metric in ["SENS", "SPEC", "PPV", "NPV", "TP", "FP", "FN", "TN"]:
            row[f"{metric}_class_{lab}"] = point.get(f"{metric}_class_{lab}", np.nan)

    return row


# =============================================================================
# ROC / DCA / Calibration / Confusion Matrix 绘图
# =============================================================================

def macro_roc_curve_multiclass(y_true, y_prob, labels):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_bin = label_binarize(y_true, classes=labels)

    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    for i, lab in enumerate(labels):
        if len(np.unique(y_bin[:, i])) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        fpr_dict[lab] = fpr
        tpr_dict[lab] = tpr
        auc_dict[lab] = auc(fpr, tpr)

    if not fpr_dict:
        raise ValueError("Cannot compute ROC because fewer than two classes are present.")

    all_fpr = np.unique(np.concatenate([fpr_dict[lab] for lab in fpr_dict]))
    mean_tpr = np.zeros_like(all_fpr)
    for lab in fpr_dict:
        mean_tpr += np.interp(all_fpr, fpr_dict[lab], tpr_dict[lab])
    mean_tpr /= len(fpr_dict)
    macro_auc = auc(all_fpr, mean_tpr)

    return fpr_dict, tpr_dict, auc_dict, all_fpr, mean_tpr, macro_auc


def plot_roc_multiclass(y_true, y_prob, labels, task_name, save_path):
    plt.figure(figsize=(5.2, 5.0))
    ax = plt.gca()

    try:
        fpr_dict, tpr_dict, auc_dict, macro_fpr, macro_tpr, macro_auc = macro_roc_curve_multiclass(y_true, y_prob, labels)
    except Exception:
        plt.close()
        return

    for lab in labels:
        if lab not in fpr_dict:
            continue
        plt.plot(
            fpr_dict[lab],
            tpr_dict[lab],
            linewidth=1.7,
            label=f"Stage {lab} OvR (AUC = {auc_dict[lab]:.3f})",
        )

    plt.plot(
        macro_fpr,
        macro_tpr,
        linewidth=2.4,
        linestyle="-",
        color="black",
        label=f"Macro-average (AUC = {macro_auc:.3f})",
    )
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray", label="Reference")

    plt.xlabel("1 - Specificity")
    plt.ylabel("Sensitivity")
    plt.title(f"ROC Curve - {task_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.legend(loc="lower right", frameon=False)
    apply_axis_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_dca_multiclass(y_true, y_pred, y_prob, task_name, save_path):
    plt.figure(figsize=(6.0, 5.0))
    ax = plt.gca()

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    max_prob = np.max(y_prob, axis=1)
    is_correct = (y_true == y_pred)

    nb_model = calculate_multiclass_net_benefit(y_true, y_pred, max_prob, DCA_THRESHOLDS)

    accuracy = np.mean(is_correct)
    error_rate = 1.0 - accuracy
    nb_all = accuracy - error_rate * (DCA_THRESHOLDS / (1 - DCA_THRESHOLDS))
    nb_none = np.zeros_like(DCA_THRESHOLDS)

    plt.plot(DCA_THRESHOLDS, nb_model, linewidth=2.2, label="Model")
    plt.plot(DCA_THRESHOLDS, nb_all, linestyle="--", linewidth=1.5, color="black", label="Treat all")
    plt.plot(DCA_THRESHOLDS, nb_none, linestyle=":", linewidth=1.5, color="black", label="Treat none")

    plt.xlabel("Threshold Probability")
    plt.ylabel("Net Benefit")
    plt.title(f"DCA Curve - {task_name}")
    plt.xlim(float(DCA_THRESHOLDS.min()), float(DCA_THRESHOLDS.max()))
    y_min = min(-0.1, float(np.nanmin([np.nanmin(nb_model), np.nanmin(nb_all), 0])))
    y_max = max(0.6, float(np.nanmax([np.nanmax(nb_model), np.nanmax(nb_all), 0])) + 0.05)
    plt.ylim(y_min, y_max)
    plt.legend(loc="best", frameon=False)
    apply_axis_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def make_fixed_quantile_bins(y_prob, n_bins=6):
    y_prob = np.asarray(y_prob, dtype=float)
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(y_prob, quantiles)
    edges = np.unique(edges)

    if len(edges) < 3:
        edges = np.linspace(0, 1, n_bins + 1)

    edges[0] = -1e-8
    edges[-1] = 1 + 1e-8

    return edges


def calibration_points_fixed_bins(y_binary, y_prob, edges):
    y_binary = np.asarray(y_binary).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    prob_pred = []
    prob_true = []

    for i in range(len(edges) - 1):
        left, right = edges[i], edges[i + 1]

        if i == len(edges) - 2:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)

        if np.sum(mask) == 0:
            prob_pred.append(np.nan)
            prob_true.append(np.nan)
        else:
            prob_pred.append(float(np.mean(y_prob[mask])))
            prob_true.append(float(np.mean(y_binary[mask])))

    return np.asarray(prob_pred), np.asarray(prob_true)


def plot_calibration_multiclass(y_true, y_pred, y_prob, task_name, save_path, n_bins=6):
    plt.figure(figsize=(5.0, 5.0))
    ax = plt.gca()

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    max_prob = np.max(y_prob, axis=1)
    y_correct = (y_true == y_pred).astype(int)

    edges = make_fixed_quantile_bins(max_prob, n_bins=n_bins)
    prob_pred, prob_true = calibration_points_fixed_bins(y_correct, max_prob, edges)
    valid = ~np.isnan(prob_pred) & ~np.isnan(prob_true)

    if np.sum(valid) > 0:
        plt.plot(
            prob_pred[valid],
            prob_true[valid],
            marker="o",
            markersize=4,
            linewidth=2,
            label="Model confidence",
        )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="black", label="Perfect calibration")

    plt.xlabel("Predicted Confidence")
    plt.ylabel("Observed Accuracy")
    plt.title(f"Calibration Curve - {task_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(loc="best", frameon=False)
    apply_axis_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_confusion_matrix_multiclass(cm, labels, task_name, save_path):
    plt.figure(figsize=(4.8, 4.2))
    ax = plt.gca()

    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(f"Confusion Matrix - {task_name}")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(len(labels))
    tick_labels = [LABEL_NAMES.get(int(l), str(l)) for l in labels]
    plt.xticks(tick_marks, tick_labels, rotation=30, ha="right")
    plt.yticks(tick_marks, tick_labels)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(int(cm[i, j]), "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    apply_axis_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


# =============================================================================
# SHAP 分析与 boxplot
# =============================================================================

def is_morphology_feature(feature_name):
    name = str(feature_name).lower()

    morphology_prefixes = (
        "leg_", "underskin_", "skin_", "subskin_", "subcutaneous_",
        "muscle_", "bone_", "calf_", "tissue_", "limb_",
    )
    morphology_keywords = (
        "volume_mm3", "area_mm2", "csa_mm2", "max_csa", "min_csa",
        "diameter", "thickness", "circumference", "perimeter",
        "area_at_", "volume", "ratio", "体积", "面积", "比例", "皮下", "肌肉", "骨",
    )

    return name.startswith(morphology_prefixes) or any(k in name for k in morphology_keywords)


def is_radiomics_feature(feature_name):
    name = str(feature_name).lower()

    radiomics_prefixes = (
        "original_", "wavelet-", "log-sigma-", "square_", "squareroot_",
        "logarithm_", "exponential_", "gradient_", "lbp-2d_", "lbp-3d_",
    )
    radiomics_keywords = (
        "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm",
    )

    return name.startswith(radiomics_prefixes) or any(k in name for k in radiomics_keywords)


def unwrap_pipeline_for_shap(model, X):
    if not hasattr(model, "steps"):
        return model, X

    steps = list(model.steps)
    if len(steps) == 0:
        return model, X

    X_trans = X.copy()
    for _, step in steps[:-1]:
        if hasattr(step, "transform"):
            X_trans = step.transform(X_trans)
        else:
            return model, X

    final_model = steps[-1][1]

    if isinstance(X_trans, pd.DataFrame):
        X_final = X_trans
    else:
        X_trans = np.asarray(X_trans)
        if X_trans.ndim == 2 and X_trans.shape[1] == X.shape[1]:
            X_final = pd.DataFrame(X_trans, columns=X.columns, index=X.index)
        else:
            X_final = pd.DataFrame(
                X_trans,
                columns=[f"transformed_feature_{i}" for i in range(X_trans.shape[1])],
                index=X.index,
            )

    return final_model, X_final


def extract_shap_values_array(shap_values):
    if isinstance(shap_values, list):
        # list[class] of (n_samples, n_features) -> (n_samples, n_features, n_classes)
        arrays = [np.asarray(v) for v in shap_values]
        return np.stack(arrays, axis=2)

    arr = np.asarray(shap_values)
    if arr.ndim == 2:
        return arr[:, :, None]
    if arr.ndim == 3:
        return arr
    raise ValueError(f"Unexpected SHAP value shape: {arr.shape}")


def compute_shap_values_for_multiclass_model(wrapper, df_source, random_state=255):
    if shap is None:
        raise ImportError("当前环境未安装 shap，请先运行：pip install shap")

    X_all = wrapper._prepare_X(df_source)
    rng = np.random.default_rng(random_state)
    sample_n = min(SHAP_MAX_CASES, len(X_all))
    sample_idx = rng.choice(len(X_all), size=sample_n, replace=False) if len(X_all) > sample_n else np.arange(len(X_all))
    X_sample = X_all.iloc[sample_idx].copy()

    shap_model, X_for_shap = unwrap_pipeline_for_shap(wrapper.model, X_sample)

    try:
        explainer = shap.TreeExplainer(shap_model)
        shap_values_raw = explainer.shap_values(X_for_shap)
        shap_values = extract_shap_values_array(shap_values_raw)
        return shap_values, X_for_shap, "TreeExplainer"
    except Exception:
        # 回退方案：适用于 LR/SVM 或 TreeExplainer 不兼容的 Pipeline。
        bg_n = min(SHAP_MAX_BACKGROUND, len(X_all))
        bg_idx = rng.choice(len(X_all), size=bg_n, replace=False) if len(X_all) > bg_n else np.arange(len(X_all))
        background = X_all.iloc[bg_idx].copy()

        def predict_fn(data):
            data_df = pd.DataFrame(data, columns=X_all.columns)
            return wrapper.predict_proba(data_df)

        explainer = shap.Explainer(predict_fn, background)
        shap_exp = explainer(X_sample)
        shap_values = extract_shap_values_array(shap_exp.values)
        return shap_values, X_sample, "ModelAgnosticExplainer"


def plot_shap_summary_top10_multiclass(shap_values, X_for_shap, labels, task_name, save_prefix):
    # shap_values: (n_samples, n_features, n_classes)
    n_classes = shap_values.shape[2]

    # 多分类整体 bar summary：按各类别平均绝对 SHAP 展示。
    try:
        plt.figure(figsize=(7.2, 5.2))
        if n_classes > 1:
            shap.summary_plot(
                [shap_values[:, :, i] for i in range(n_classes)],
                X_for_shap,
                class_names=[LABEL_NAMES.get(int(l), str(l)) for l in labels[:n_classes]],
                max_display=10,
                plot_type="bar",
                show=False,
                plot_size=(7.2, 5.2),
            )
        else:
            shap.summary_plot(
                shap_values[:, :, 0],
                X_for_shap,
                max_display=10,
                show=False,
                plot_size=(7.2, 5.2),
            )
        plt.title(f"SHAP Summary - {task_name}", fontsize=11)
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_summary_top10.png", dpi=600, bbox_inches="tight")
        plt.close()
    except Exception as e:
        plt.close()
        print(f"WARNING: SHAP summary plot failed: {e}")

    # 每个类别单独输出 dot summary，便于查看方向性。
    for i in range(n_classes):
        try:
            plt.figure(figsize=(7.2, 5.2))
            shap.summary_plot(
                shap_values[:, :, i],
                X_for_shap,
                max_display=10,
                show=False,
                plot_size=(7.2, 5.2),
            )
            lab = labels[i] if i < len(labels) else i
            plt.title(f"SHAP Summary - {task_name} - Stage {lab}", fontsize=11)
            plt.tight_layout()
            plt.savefig(f"{save_prefix}_summary_top10_stage_{safe_filename(lab)}.png", dpi=600, bbox_inches="tight")
            plt.close()
        except Exception as e:
            plt.close()
            print(f"WARNING: SHAP class-specific plot failed for class index {i}: {e}")


def plot_top10_shap_bar(shap_rank_df, task_name, save_path):
    top = shap_rank_df.head(10).iloc[::-1]
    plt.figure(figsize=(7.2, 5.0))
    ax = plt.gca()
    ax.barh(top["feature"], top["mean_abs_shap"])
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title(f"Top 10 SHAP features - {task_name}")
    apply_axis_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_feature_boxplot_by_label(task_df, feature_name, task_name, feature_type, labels, save_path):
    plot_df = task_df[[LABEL_COL, feature_name]].copy()
    plot_df[feature_name] = pd.to_numeric(plot_df[feature_name], errors="coerce")
    plot_df = plot_df.dropna(subset=[LABEL_COL, feature_name])

    def remove_iqr_outliers(vals):
        vals = np.asarray(vals, dtype=float)
        vals = vals[~np.isnan(vals)]

        if len(vals) < 4:
            return vals

        q1, q3 = np.percentile(vals, [25, 75])
        iqr = q3 - q1

        if iqr == 0:
            return vals

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        return vals[(vals >= lower) & (vals <= upper)]

    data = []
    xtick_labels = []
    for label in labels:
        vals = plot_df.loc[plot_df[LABEL_COL] == label, feature_name].values
        vals = remove_iqr_outliers(vals)
        data.append(vals)
        xtick_labels.append(f"Stage {label}")

    sci_box_colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
    ]

    plt.figure(figsize=(4.8, 4.5))
    ax = plt.gca()
    bp = ax.boxplot(
        data,
        labels=xtick_labels,
        patch_artist=True,
        showfliers=False,
        widths=0.55,
        medianprops={"linewidth": 1.5, "color": "#1A1A1A"},
        boxprops={"linewidth": 1.1, "edgecolor": "#1A1A1A"},
        whiskerprops={"linewidth": 1.1, "color": "#1A1A1A"},
        capprops={"linewidth": 1.1, "color": "#1A1A1A"},
    )

    for i, patch in enumerate(bp["boxes"]):
        color = sci_box_colors[i % len(sci_box_colors)]
        patch.set_facecolor(color)
        patch.set_edgecolor("#1A1A1A")
        patch.set_alpha(0.78)

    rng = np.random.default_rng(RANDOM_STATE)
    for i, vals in enumerate(data, start=1):
        if len(vals) == 0:
            continue
        color = sci_box_colors[(i - 1) % len(sci_box_colors)]
        jitter = rng.normal(loc=0, scale=0.035, size=len(vals))
        ax.scatter(
            np.full(len(vals), i) + jitter,
            vals,
            s=14,
            alpha=0.68,
            color=color,
            edgecolors="none",
        )

    ax.set_xlabel("True label")
    ax.set_ylabel(feature_name)
    ax.set_title(f"{feature_type} feature - {task_name}", fontsize=11)
    apply_axis_style(ax)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def run_shap_analysis(df, wrapper):
    if shap is None:
        print("WARNING: 当前环境未安装 shap，跳过 SHAP 分析。")
        return {
            "top10": pd.DataFrame(),
            "all_rank": pd.DataFrame(),
            "boxplot_records": pd.DataFrame(),
        }

    shap_values, X_for_shap, explainer_name = compute_shap_values_for_multiclass_model(
        wrapper=wrapper,
        df_source=df,
        random_state=RANDOM_STATE,
    )

    if shap_values.shape[1] != X_for_shap.shape[1]:
        raise ValueError(
            f"SHAP维度与特征维度不一致：shap={shap_values.shape}, X={X_for_shap.shape}"
        )

    mean_abs_shap = np.mean(np.abs(shap_values), axis=(0, 2))
    shap_rank_df = pd.DataFrame({
        "task": TASK_NAME,
        "feature_set": wrapper.feature_set,
        "model_name": wrapper.model_name,
        "explainer": explainer_name,
        "feature": list(X_for_shap.columns),
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    shap_rank_df["rank"] = np.arange(1, len(shap_rank_df) + 1)
    shap_rank_df["feature_type"] = shap_rank_df["feature"].apply(
        lambda x: "Morphology" if is_morphology_feature(x) else ("Radiomics" if is_radiomics_feature(x) else "Unclassified")
    )

    top10_df = shap_rank_df.head(10).copy()

    save_prefix = os.path.join(shap_dir, f"SHAP_{safe_filename(TASK_NAME)}_multiclass")
    plot_shap_summary_top10_multiclass(
        shap_values=shap_values,
        X_for_shap=X_for_shap,
        labels=wrapper.label_order,
        task_name=TASK_NAME,
        save_prefix=save_prefix,
    )
    plot_top10_shap_bar(
        shap_rank_df=shap_rank_df,
        task_name=TASK_NAME,
        save_path=os.path.join(shap_dir, f"SHAP_Top10_bar_{safe_filename(TASK_NAME)}.png"),
    )

    top_radiomics = shap_rank_df[shap_rank_df["feature_type"] == "Radiomics"]
    top_morphology = shap_rank_df[shap_rank_df["feature_type"] == "Morphology"]

    top_radiomics_feature = top_radiomics.iloc[0]["feature"] if len(top_radiomics) > 0 else None
    top_morphology_feature = top_morphology.iloc[0]["feature"] if len(top_morphology) > 0 else None

    boxplot_records = []

    if top_radiomics_feature is not None and top_radiomics_feature in df.columns:
        path = os.path.join(
            boxplot_dir,
            f"Boxplot_{safe_filename(TASK_NAME)}_top_radiomics_{safe_filename(top_radiomics_feature)}.png",
        )
        plot_feature_boxplot_by_label(
            task_df=df,
            feature_name=top_radiomics_feature,
            task_name=TASK_NAME,
            feature_type="Top radiomics SHAP",
            labels=wrapper.label_order,
            save_path=path,
        )
        boxplot_records.append({
            "task": TASK_NAME,
            "feature_type": "Top radiomics",
            "feature": top_radiomics_feature,
            "boxplot_path": path,
        })

    if top_morphology_feature is not None and top_morphology_feature in df.columns:
        path = os.path.join(
            boxplot_dir,
            f"Boxplot_{safe_filename(TASK_NAME)}_top_morphology_{safe_filename(top_morphology_feature)}.png",
        )
        plot_feature_boxplot_by_label(
            task_df=df,
            feature_name=top_morphology_feature,
            task_name=TASK_NAME,
            feature_type="Top morphology SHAP",
            labels=wrapper.label_order,
            save_path=path,
        )
        boxplot_records.append({
            "task": TASK_NAME,
            "feature_type": "Top morphology",
            "feature": top_morphology_feature,
            "boxplot_path": path,
        })

    return {
        "top10": top10_df,
        "all_rank": shap_rank_df,
        "boxplot_records": pd.DataFrame(boxplot_records),
    }


# =============================================================================
# Excel 写入工具
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
            part.to_excel(writer, sheet_name=safe_sheet_name(f"{sheet_name}_{i + 1}"), index=False)


# =============================================================================
# 主流程
# =============================================================================

def main():
    df = load_input_table(input_excel)
    wrapper = load_model_wrapper(MODEL_WRAPPER_PATH)

    # 以模型封装中的 label_order 为准；若没有，则使用 [1,2,3]。
    labels = [int(x) for x in wrapper.label_order]
    if set(labels) != set(LABELS):
        print(f"WARNING: 模型 label_order={labels} 与脚本 LABELS={LABELS} 不完全一致，将以模型 label_order 为准。")

    check_features(df, wrapper.feature_cols)

    X = wrapper._prepare_X(df)
    y_true = df[LABEL_COL].astype(int).values
    y_prob = wrapper.predict_proba(df)
    y_pred = wrapper.predict(df)
    max_prob = np.max(y_prob, axis=1)

    metrics = calc_metrics_with_ci_multiclass(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        labels=labels,
        task_name=TASK_NAME,
        feature_set_name=wrapper.feature_set,
        model_name=wrapper.model_name,
        model_path=MODEL_WRAPPER_PATH,
        n_samples=len(df),
    )
    metrics_df = pd.DataFrame([metrics])

    pred_all = df.copy()
    pred_all["task"] = TASK_NAME
    pred_all["feature_set"] = wrapper.feature_set
    pred_all["model_name"] = wrapper.model_name
    pred_all["model_path"] = MODEL_WRAPPER_PATH
    pred_all["true_label"] = y_true
    pred_all["pred_label"] = y_pred
    pred_all["pred_name"] = [LABEL_NAMES.get(int(v), str(v)) for v in y_pred]
    pred_all["is_correct"] = (y_true == y_pred)
    pred_all["max_prob"] = max_prob
    pred_all["uncertainty"] = 1.0 - max_prob

    for j, lab in enumerate(labels):
        pred_all[f"prob_stage_{lab}"] = y_prob[:, j]

    wrong_all = pred_all[pred_all["is_correct"] == False].copy()

    # 曲线与混淆矩阵。
    plot_roc_multiclass(
        y_true=y_true,
        y_prob=y_prob,
        labels=labels,
        task_name=TASK_NAME,
        save_path=os.path.join(plot_dir, f"ROC_{safe_filename(TASK_NAME)}_multiclass.png"),
    )

    plot_dca_multiclass(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        task_name=TASK_NAME,
        save_path=os.path.join(plot_dir, f"DCA_{safe_filename(TASK_NAME)}_multiclass.png"),
    )

    plot_calibration_multiclass(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        task_name=TASK_NAME,
        save_path=os.path.join(plot_dir, f"Calibration_{safe_filename(TASK_NAME)}_multiclass.png"),
        n_bins=CALIBRATION_BINS,
    )

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plot_confusion_matrix_multiclass(
        cm=cm,
        labels=labels,
        task_name=TASK_NAME,
        save_path=os.path.join(plot_dir, f"CM_{safe_filename(TASK_NAME)}_multiclass.png"),
    )

    # SHAP 分析。
    shap_result = run_shap_analysis(df, wrapper)

    # 模型和运行信息。
    run_info = pd.DataFrame([
        {"key": "MODEL_WRAPPER_PATH", "value": MODEL_WRAPPER_PATH},
        {"key": "input_excel", "value": input_excel},
        {"key": "output_dir", "value": output_dir},
        {"key": "LABEL_COL", "value": LABEL_COL},
        {"key": "TASK_NAME", "value": TASK_NAME},
        {"key": "model_task", "value": wrapper.task},
        {"key": "model_feature_set", "value": wrapper.feature_set},
        {"key": "model_name", "value": wrapper.model_name},
        {"key": "label_order", "value": json.dumps(labels, ensure_ascii=False)},
        {"key": "n_model_features", "value": len(wrapper.feature_cols)},
        {"key": "n_samples", "value": len(df)},
        {"key": "n_wrong", "value": len(wrong_all)},
        {"key": "N_BOOTSTRAP", "value": N_BOOTSTRAP},
        {"key": "DCA_THRESHOLDS", "value": f"{float(DCA_THRESHOLDS.min()):.2f}-{float(DCA_THRESHOLDS.max()):.2f}"},
        {"key": "CALIBRATION_BINS", "value": CALIBRATION_BINS},
        {"key": "SHAP_MAX_BACKGROUND", "value": SHAP_MAX_BACKGROUND},
        {"key": "SHAP_MAX_CASES", "value": SHAP_MAX_CASES},
    ])

    feature_df = pd.DataFrame({
        "feature_rank": np.arange(1, len(wrapper.feature_cols) + 1),
        "feature": wrapper.feature_cols,
        "feature_type": [
            "Morphology" if is_morphology_feature(f) else ("Radiomics" if is_radiomics_feature(f) else "Unclassified")
            for f in wrapper.feature_cols
        ],
    })

    # metrics 单独保存。
    front_cols = [
        "task",
        "feature_set",
        "model_name",
        "n_samples",
        "labels",
        "ACC_with_95CI",
        "AUC_with_95CI",
        "SENS_with_95CI",
        "SPEC_with_95CI",
        "PPV_with_95CI",
        "NPV_with_95CI",
        "Brier_with_95CI",
        "EO_ratio_with_95CI",
        "Calibration_slope_with_95CI",
        "Calibration_intercept_with_95CI",
        "Calibration_in_the_large_with_95CI",
        "DCA_mean_net_benefit_with_95CI",
        "DCA_max_net_benefit_with_95CI",
        "DCA_net_benefit_at_0.10_with_95CI",
        "DCA_net_benefit_at_0.20_with_95CI",
        "DCA_net_benefit_at_0.30_with_95CI",
        "DCA_net_benefit_at_0.40_with_95CI",
        "DCA_net_benefit_at_0.50_with_95CI",
        "model_path",
    ]
    front_cols = [c for c in front_cols if c in metrics_df.columns]
    other_cols = [c for c in metrics_df.columns if c not in front_cols]
    metrics_df = metrics_df[front_cols + other_cols]
    metrics_df.to_excel(output_metrics, index=False)

    # 综合 workbook。
    with pd.ExcelWriter(output_wrong_cases, engine="openpyxl") as writer:
        write_sheet_safe(writer, metrics_df, "metrics_all")
        write_sheet_safe(writer, pred_all, "all_predictions")
        write_sheet_safe(writer, wrong_all, "wrong_cases_all")
        write_sheet_safe(writer, feature_df, "model_features")
        write_sheet_safe(writer, run_info, "run_info")

    # SHAP workbook。
    with pd.ExcelWriter(output_shap_top10, engine="openpyxl") as writer:
        write_sheet_safe(writer, shap_result.get("top10", pd.DataFrame()), "SHAP_top10")
        write_sheet_safe(writer, shap_result.get("all_rank", pd.DataFrame()), "SHAP_all_rank")
        write_sheet_safe(writer, shap_result.get("boxplot_records", pd.DataFrame()), "boxplot_index")

    display_cols = [
        "task",
        "feature_set",
        "model_name",
        "n_samples",
        "ACC_with_95CI",
        "AUC_with_95CI",
        "SENS_with_95CI",
        "SPEC_with_95CI",
        "PPV_with_95CI",
        "NPV_with_95CI",
        "Brier_with_95CI",
        "EO_ratio_with_95CI",
        "Calibration_slope_with_95CI",
        "Calibration_intercept_with_95CI",
        "Calibration_in_the_large_with_95CI",
        "DCA_mean_net_benefit_with_95CI",
        "DCA_max_net_benefit_with_95CI",
        "DCA_net_benefit_at_0.10_with_95CI",
        "DCA_net_benefit_at_0.20_with_95CI",
        "DCA_net_benefit_at_0.30_with_95CI",
        "DCA_net_benefit_at_0.40_with_95CI",
        "DCA_net_benefit_at_0.50_with_95CI",
    ]
    display_cols = [c for c in display_cols if c in metrics_df.columns]

    print("\n========== Multiclass foresight metrics with 95% CI ==========")
    print(metrics_df[display_cols])

    print("\n========== Wrong cases ==========")
    print(f"{TASK_NAME} 错误病例数: {len(wrong_all)} / {len(pred_all)}")

    print("\n已保存:")
    print(output_metrics)
    print(output_wrong_cases)
    print(plot_dir)
    print(output_shap_top10)
    print(shap_dir)
    print(boxplot_dir)


if __name__ == "__main__":
    main()

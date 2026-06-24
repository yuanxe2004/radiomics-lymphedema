# -*- coding: utf-8 -*-
"""Multiclass performance metrics and bootstrap confidence intervals."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from sklearn.linear_model import LogisticRegression

from .utils import format_mean_sd


def multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, labels: List[int]) -> Dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = cm.sum()
    if total == 0:
        return {"ACC": np.nan, "AUC": np.nan, "SENS": np.nan, "SPEC": np.nan, "PPV": np.nan, "NPV": np.nan, "CM": cm}

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0).astype(float) - tp
    fn = cm.sum(axis=1).astype(float) - tp
    tn = total - (tp + fp + fn)

    sens = np.divide(tp, tp + fn, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fn) != 0)
    spec = np.divide(tn, tn + fp, out=np.full_like(tn, np.nan, dtype=float), where=(tn + fp) != 0)
    ppv = np.divide(tp, tp + fp, out=np.full_like(tp, np.nan, dtype=float), where=(tp + fp) != 0)
    npv = np.divide(tn, tn + fn, out=np.full_like(tn, np.nan, dtype=float), where=(tn + fn) != 0)
    acc = float(np.trace(cm) / total)

    try:
        if len(np.unique(y_true)) == len(labels):
            y_bin = label_binarize(y_true, classes=labels)
            auc = float(roc_auc_score(y_bin, y_prob, average="macro", multi_class="ovr"))
        else:
            auc = np.nan
    except Exception:
        auc = np.nan

    out = {
        "ACC": acc,
        "AUC": auc,
        "SENS": float(np.nanmean(sens)),
        "SPEC": float(np.nanmean(spec)),
        "PPV": float(np.nanmean(ppv)),
        "NPV": float(np.nanmean(npv)),
        "CM": cm,
    }
    for idx, lab in enumerate(labels):
        out[f"SENS_class_{lab}"] = sens[idx]
        out[f"SPEC_class_{lab}"] = spec[idx]
        out[f"PPV_class_{lab}"] = ppv[idx]
        out[f"NPV_class_{lab}"] = npv[idx]
    return out


def bootstrap_metrics_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    labels: List[int],
    n_bootstrap: int = 1000,
    seed: int = 2026,
) -> Dict[str, Tuple[float, float, float]]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    point = multiclass_metrics(y_true, y_pred, y_prob, labels)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    metric_names = ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]
    boot_values = {m: [] for m in metric_names}
    if n == 0:
        return {m: (np.nan, np.nan, np.nan) for m in metric_names}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        m = multiclass_metrics(y_true[idx], y_pred[idx], y_prob[idx], labels)
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
    summary: Dict[str, Any] = {}
    for key in ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]:
        vals = pd.to_numeric(pd.Series([r.get(key, np.nan) for r in rows]), errors="coerce")
        summary[f"{key}_mean"] = float(vals.mean()) if vals.notna().any() else np.nan
        summary[f"{key}_sd"] = float(vals.std(ddof=1)) if vals.notna().sum() > 1 else np.nan
        summary[f"{key}_mean_sd"] = format_mean_sd(summary[f"{key}_mean"], summary[f"{key}_sd"])
    return summary


def safe_logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def calibration_slope_intercept(y_true_bin, y_prob):
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    if len(np.unique(y_true_bin)) < 2:
        return np.nan, np.nan
    try:
        x = safe_logit(y_prob).reshape(-1, 1)
        lr = LogisticRegression(penalty="l2", C=1e6, solver="lbfgs", max_iter=5000)
        lr.fit(x, y_true_bin)
        return float(lr.coef_[0][0]), float(lr.intercept_[0])
    except Exception:
        return np.nan, np.nan


def calibration_in_the_large(y_true_bin, y_prob):
    obs = np.mean(y_true_bin)
    pred = np.mean(y_prob)
    if obs <= 0 or obs >= 1 or pred <= 0 or pred >= 1:
        return np.nan
    return float(np.log(obs / (1 - obs)) - np.log(pred / (1 - pred)))


def eo_ratio(y_true_bin, y_prob):
    observed = np.sum(y_true_bin)
    expected = np.sum(y_prob)
    if expected <= 0:
        return np.nan
    return float(observed / expected)


def multiclass_validation_extra_metrics(y_true, y_pred, y_prob, thresholds=None):
    """Extra metrics for prospective validation using confidence-as-correctness calibration/DCA."""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    max_prob = np.max(y_prob, axis=1)
    y_correct = (y_true == y_pred).astype(int)
    out = {}
    try:
        out["Brier"] = float(brier_score_loss(y_correct, max_prob))
    except Exception:
        out["Brier"] = np.nan
    out["EO_ratio"] = eo_ratio(y_correct, max_prob)
    out["Calibration_slope"], out["Calibration_intercept"] = calibration_slope_intercept(y_correct, max_prob)
    out["Calibration_in_the_large"] = calibration_in_the_large(y_correct, max_prob)
    if thresholds is not None:
        nb = multiclass_net_benefit(y_true, y_pred, max_prob, thresholds)
        model_nb = nb[nb["strategy"] == "model"]["net_benefit"].values
        out["DCA_mean_net_benefit"] = float(np.nanmean(model_nb)) if len(model_nb) else np.nan
        out["DCA_max_net_benefit"] = float(np.nanmax(model_nb)) if len(model_nb) else np.nan
        for t in [0.10, 0.20, 0.30, 0.40, 0.50]:
            idx = np.argmin(np.abs(np.asarray(thresholds) - t))
            out[f"DCA_net_benefit_at_{t:.2f}"] = float(model_nb[idx]) if len(model_nb) > idx else np.nan
    return out


def multiclass_net_benefit(y_true, y_pred, max_prob, thresholds):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    max_prob = np.asarray(max_prob, dtype=float)
    n = len(y_true)
    rows = []
    if n == 0:
        return pd.DataFrame(columns=["threshold", "net_benefit", "strategy"])
    correct = (y_pred == y_true)
    tp_all_rate = float(np.mean(correct))
    fp_all_rate = float(np.mean(~correct))
    for pt in thresholds:
        if pt <= 0 or pt >= 1:
            continue
        weight = pt / (1 - pt)
        selected = max_prob >= pt
        tp = np.sum(selected & correct)
        fp = np.sum(selected & (~correct))
        rows.append({"threshold": pt, "net_benefit": (tp / n) - (fp / n) * weight, "strategy": "model"})
        rows.append({"threshold": pt, "net_benefit": 0.0, "strategy": "treat_none"})
        rows.append({"threshold": pt, "net_benefit": tp_all_rate - fp_all_rate * weight, "strategy": "treat_all"})
    return pd.DataFrame(rows)

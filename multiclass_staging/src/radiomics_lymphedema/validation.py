# -*- coding: utf-8 -*-
"""Prospective/external table validation with a saved multiclass wrapper."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import confusion_matrix

from .config import ensure_dir
from .metrics import bootstrap_metrics_ci, multiclass_metrics, multiclass_validation_extra_metrics, multiclass_net_benefit
from .plots import macro_roc_curve, plot_confusion_matrix, set_publication_plot_style
from .utils import format_ci, safe_filename, timestamp, write_sheet_safe
from .wrapper import load_best_model


def _format_with_ci(point, low, high):
    if pd.isna(point):
        return ""
    if pd.isna(low) or pd.isna(high):
        return f"{point:.3f}"
    return f"{point:.3f} ({low:.3f}-{high:.3f})"


def _bootstrap_extra_ci(y_true, y_pred, y_prob, thresholds, n_bootstrap=1000, seed=2026):
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true)
    keys = [
        "Brier", "EO_ratio", "Calibration_slope", "Calibration_intercept", "Calibration_in_the_large",
        "DCA_mean_net_benefit", "DCA_max_net_benefit",
        "DCA_net_benefit_at_0.10", "DCA_net_benefit_at_0.20", "DCA_net_benefit_at_0.30",
        "DCA_net_benefit_at_0.40", "DCA_net_benefit_at_0.50",
    ]
    boot = {k: [] for k in keys}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            m = multiclass_validation_extra_metrics(y_true[idx], y_pred[idx], y_prob[idx], thresholds)
        except Exception:
            continue
        for k in keys:
            if k in m and not pd.isna(m[k]):
                boot[k].append(float(m[k]))
    out = {}
    for k, vals in boot.items():
        vals = np.asarray(vals, dtype=float)
        if len(vals) == 0:
            out[f"{k}_low"] = np.nan
            out[f"{k}_high"] = np.nan
        else:
            out[f"{k}_low"] = float(np.percentile(vals, 2.5))
            out[f"{k}_high"] = float(np.percentile(vals, 97.5))
    return out


def plot_single_roc(y_true, y_prob, labels: List[int], save_path: Path):
    plt.figure(figsize=(5.2, 5.0), dpi=300)
    fpr, tpr, roc_auc = macro_roc_curve(np.asarray(y_true).astype(int), np.asarray(y_prob, dtype=float), labels)
    plt.plot(fpr, tpr, linewidth=2.0, label=f"Macro OvR AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Reference")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_single_dca(y_true, y_pred, max_prob, save_path: Path):
    thresholds = np.linspace(0.05, 0.95, 91)
    dca = multiclass_net_benefit(y_true, y_pred, max_prob, thresholds)
    plt.figure(figsize=(6, 5), dpi=300)
    for strategy, label, style in [("model", "Model", "-"), ("treat_all", "Treat all", "--"), ("treat_none", "Treat none", ":")]:
        sub = dca[dca["strategy"] == strategy]
        plt.plot(sub["threshold"], sub["net_benefit"], linestyle=style, linewidth=2 if strategy == "model" else 1.5, label=label)
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title("Decision Curve Analysis")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def plot_single_calibration(y_true, y_pred, max_prob, save_path: Path):
    y_correct = (np.asarray(y_true).astype(int) == np.asarray(y_pred).astype(int)).astype(int)
    try:
        frac_pos, mean_pred = calibration_curve(y_correct, max_prob, n_bins=8, strategy="quantile")
    except Exception:
        return
    plt.figure(figsize=(5.2, 5.0), dpi=300)
    plt.plot(mean_pred, frac_pos, marker="o", linewidth=2.0, label="Model confidence")
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Perfect calibration")
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title("Calibration Curve")
    plt.legend(loc="best", frameon=False)
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    plt.close()


def validate_table(
    model_path: str,
    input_excel: str,
    output_dir: str,
    label_col: str = "label",
    id_cols: Optional[List[str]] = None,
    n_bootstrap: int = 1000,
    seed: int = 2026,
) -> Dict[str, str]:
    set_publication_plot_style()
    out_dir = ensure_dir(output_dir)
    plot_dir = ensure_dir(out_dir / "figures")
    wrapper = load_best_model(model_path)
    labels = [int(x) for x in wrapper.label_order]
    label_names = {int(k): v for k, v in wrapper.label_names.items()}

    df = pd.read_excel(input_excel)
    if label_col not in df.columns:
        raise ValueError(f"Missing true label column: {label_col}")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce")
    df = df.dropna(subset=[label_col]).copy()
    df[label_col] = df[label_col].astype(int)
    df = df[df[label_col].isin(labels)].copy()
    if df.empty:
        raise ValueError("No samples remained after label filtering.")

    pred_df = wrapper.predict_dataframe(df)
    y_true = df[label_col].astype(int).values
    y_pred = pred_df["pred_label"].astype(int).values
    y_prob = pred_df[[f"prob_stage_{lab}" for lab in labels]].values
    max_prob = pred_df["max_prob"].values

    base = multiclass_metrics(y_true, y_pred, y_prob, labels)
    ci = bootstrap_metrics_ci(y_true, y_pred, y_prob, labels, n_bootstrap, seed)
    thresholds = np.linspace(0.05, 0.95, 91)
    extra = multiclass_validation_extra_metrics(y_true, y_pred, y_prob, thresholds)
    extra_ci = _bootstrap_extra_ci(y_true, y_pred, y_prob, thresholds, n_bootstrap=n_bootstrap, seed=seed + 1)

    row = {
        "task": wrapper.task or "Stage_1_vs_2_vs_3",
        "model_path": model_path,
        "input_excel": input_excel,
        "n_samples": len(df),
        "feature_set": wrapper.feature_set,
        "model_name": wrapper.model_name,
        "labels": str(labels),
    }
    for key in ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]:
        mean, low, high = ci[key]
        row[f"{key}_with_95CI"] = format_ci(mean, low, high)
        row[f"{key}_raw"] = base[key]
        row[f"{key}_low"] = low
        row[f"{key}_high"] = high
    for key, value in extra.items():
        row[f"{key}_with_95CI"] = _format_with_ci(value, extra_ci.get(f"{key}_low"), extra_ci.get(f"{key}_high"))
        row[f"{key}_raw"] = value
        row[f"{key}_low"] = extra_ci.get(f"{key}_low")
        row[f"{key}_high"] = extra_ci.get(f"{key}_high")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    for i, true_lab in enumerate(labels):
        for j, pred_lab in enumerate(labels):
            row[f"CM_true{true_lab}_pred{pred_lab}"] = int(cm[i, j])

    metrics_df = pd.DataFrame([row])
    pred_df["true_label"] = y_true
    pred_df["correct"] = y_true == y_pred
    wrong_df = pred_df[~pred_df["correct"]].copy()

    workbook_path = out_dir / f"foresight_multiclass_validation_{timestamp()}.xlsx"
    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        write_sheet_safe(writer, metrics_df, "metrics")
        write_sheet_safe(writer, pred_df, "all_predictions")
        write_sheet_safe(writer, wrong_df, "wrong_cases")

    plot_single_roc(y_true, y_prob, labels, plot_dir / "ROC_multiclass.png")
    plot_single_dca(y_true, y_pred, max_prob, plot_dir / "DCA_multiclass.png")
    plot_single_calibration(y_true, y_pred, max_prob, plot_dir / "Calibration_multiclass.png")
    plot_confusion_matrix(cm, plot_dir, "Confusion Matrix", "CM_multiclass.png", labels, label_names)

    print("Saved validation workbook:", workbook_path)
    print("Saved validation figures:", plot_dir)
    return {"workbook_path": str(workbook_path), "figures_dir": str(plot_dir)}

# -*- coding: utf-8 -*-
"""Publication-style plotting helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import auc, roc_curve
from sklearn.preprocessing import label_binarize

from .metrics import multiclass_net_benefit
from .utils import safe_filename


def set_publication_plot_style() -> None:
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


def apply_axis_style(ax) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", length=4, width=0.9)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def macro_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, labels: List[int]):
    y_bin = label_binarize(y_true, classes=labels)
    fpr_dict, tpr_dict = {}, {}
    for i, lab in enumerate(labels):
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
    return all_fpr, mean_tpr, float(auc(all_fpr, mean_tpr))


def plot_roc_comparison(predictions: pd.DataFrame, output_dir: Path, dataset_name: str, task: str, labels: List[int], label_col: str, feature_sets: List[str]) -> None:
    plt.figure(figsize=(6.4, 5.2), dpi=300)
    plotted = False
    for fs in feature_sets:
        sub = predictions[(predictions["__split__"] == dataset_name) & (predictions["feature_set"] == fs)]
        if sub.empty:
            continue
        y_true = sub[label_col].astype(int).values
        y_prob = sub[[f"prob_stage_{lab}" for lab in labels]].values
        try:
            fpr, tpr, roc_auc = macro_roc_curve(y_true, y_prob, labels)
            plt.plot(fpr, tpr, linewidth=2.0, label=f"{fs} (AUC={roc_auc:.3f})")
            plotted = True
        except Exception:
            continue
    if not plotted:
        plt.close(); return
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(f"ROC: {dataset_name}")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"ROC_{safe_filename(task)}_{dataset_name}.png")
    plt.close()


def plot_dca_comparison(predictions: pd.DataFrame, output_dir: Path, dataset_name: str, task: str, labels: List[int], label_col: str, feature_sets: List[str]) -> None:
    thresholds = np.linspace(0.05, 0.95, 91)
    plt.figure(figsize=(6.4, 5.2), dpi=300)
    plotted = False
    none_plotted = False
    for fs in feature_sets:
        sub = predictions[(predictions["__split__"] == dataset_name) & (predictions["feature_set"] == fs)]
        if sub.empty:
            continue
        dca = multiclass_net_benefit(
            sub[label_col].astype(int).values,
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
        plt.close(); return
    plt.xlabel("Threshold probability")
    plt.ylabel("Net benefit")
    plt.title(f"DCA: {dataset_name}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"DCA_{safe_filename(task)}_{dataset_name}.png")
    plt.close()


def plot_calibration_comparison(predictions: pd.DataFrame, output_dir: Path, dataset_name: str, task: str, feature_sets: List[str]) -> None:
    plt.figure(figsize=(6.4, 5.2), dpi=300)
    plotted = False
    for fs in feature_sets:
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
        plt.close(); return
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)
    plt.xlabel("Mean predicted confidence")
    plt.ylabel("Observed accuracy")
    plt.title(f"Calibration: {dataset_name}")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_dir / f"Calibration_{safe_filename(task)}_{dataset_name}.png")
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_dir: Path, title: str, filename: str, labels: List[int], label_names: Dict[int, str]) -> None:
    plt.figure(figsize=(4.8, 4.2), dpi=300)
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, [label_names.get(l, str(l)) for l in labels], rotation=30, ha="right")
    plt.yticks(ticks, [label_names.get(l, str(l)) for l in labels])
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

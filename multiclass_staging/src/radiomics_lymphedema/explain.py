# -*- coding: utf-8 -*-
"""SHAP/permutation importance and boxplots."""

from __future__ import annotations

from pathlib import Path
from typing import Any, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import PipelineConfig
from .models import predict_labels_and_prob
from .utils import as_numeric_df, safe_filename


def is_morphology_feature(feature_name: str) -> bool:
    name = str(feature_name).lower()
    prefixes = ("leg_", "underskin_", "skin_", "subskin_", "subcutaneous_", "muscle_", "bone_", "calf_", "tissue_", "limb_")
    keywords = ("volume_mm3", "area_mm2", "csa_mm2", "max_csa", "min_csa", "diameter", "thickness", "circumference", "perimeter", "area_at_", "volume", "ratio")
    return name.startswith(prefixes) or any(k in name for k in keywords)


def is_radiomics_feature(feature_name: str) -> bool:
    name = str(feature_name).lower()
    prefixes = ("original_", "wavelet-", "log-sigma-", "square_", "squareroot_", "logarithm_", "exponential_", "gradient_", "lbp-2d_", "lbp-3d_")
    keywords = ("firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
    return name.startswith(prefixes) or any(k in name for k in keywords)


def permutation_importance_simple(
    cfg: PipelineConfig,
    model: Any,
    X: pd.DataFrame,
    y_true: np.ndarray,
    selected_features: List[str],
    model_name: str,
    n_repeats: int = 10,
) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.random_state)
    base_pred, _ = predict_labels_and_prob(cfg, model, model_name, X)
    base_acc = np.mean(base_pred == y_true)
    rows = []
    for feat in selected_features:
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[feat] = rng.permutation(Xp[feat].values)
            pred, _ = predict_labels_and_prob(cfg, model, model_name, Xp)
            drops.append(base_acc - np.mean(pred == y_true))
        rows.append({"feature": feat, "importance": float(np.mean(drops)), "method": "permutation_accuracy_drop"})
    return pd.DataFrame(rows).sort_values("importance", ascending=False)


def make_shap_or_importance_outputs(
    cfg: PipelineConfig,
    model: Any,
    selected_features: List[str],
    trainval_df: pd.DataFrame,
    output_dir: Path,
    feature_set: str,
    model_name: str,
) -> pd.DataFrame:
    if len(selected_features) == 0:
        return pd.DataFrame()
    X_all = as_numeric_df(trainval_df, selected_features)
    y_all = trainval_df[cfg.label_col].astype(int).values
    rng = np.random.default_rng(cfg.random_state)
    sample_n = min(cfg.shap_max_cases, len(X_all))
    sample_idx = rng.choice(len(X_all), size=sample_n, replace=False) if len(X_all) > sample_n else np.arange(len(X_all))
    X_sample = X_all.iloc[sample_idx].copy()

    importance_df = pd.DataFrame()
    shap_done = False
    if cfg.run_shap:
        try:
            import shap
            bg_n = min(cfg.shap_max_background, len(X_all))
            bg_idx = rng.choice(len(X_all), size=bg_n, replace=False) if len(X_all) > bg_n else np.arange(len(X_all))
            background = X_all.iloc[bg_idx].copy()

            def predict_fn(data):
                data_df = pd.DataFrame(data, columns=selected_features)
                _, prob = predict_labels_and_prob(cfg, model, model_name, data_df)
                return prob

            explainer = shap.Explainer(predict_fn, background)
            shap_values = explainer(X_sample)
            values = np.asarray(shap_values.values)
            if values.ndim == 3:
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
            plt.savefig(output_dir / f"SHAP_Top10_{safe_filename(cfg.task_name)}_{safe_filename(feature_set)}_{safe_filename(model_name)}.png")
            plt.close()
            shap_done = True
        except Exception as e:
            print(f"WARNING: SHAP failed ({e}). Falling back to permutation importance.")

    if not shap_done:
        importance_df = permutation_importance_simple(cfg, model, X_all, y_all, selected_features, model_name)
        top = importance_df.head(10).iloc[::-1]
        plt.figure(figsize=(7.2, 5.0), dpi=300)
        plt.barh(top["feature"], top["importance"])
        plt.xlabel("Permutation accuracy drop")
        plt.title(f"Top 10 feature importance: {feature_set} / {model_name}")
        plt.tight_layout()
        plt.savefig(output_dir / f"Importance_Top10_{safe_filename(cfg.task_name)}_{safe_filename(feature_set)}_{safe_filename(model_name)}.png")
        plt.close()

    importance_df.insert(0, "task", cfg.task_name)
    importance_df.insert(1, "feature_set", feature_set)
    importance_df.insert(2, "model", model_name)
    importance_df["feature_type"] = importance_df["feature"].apply(lambda x: "Morphology" if is_morphology_feature(x) else ("Radiomics" if is_radiomics_feature(x) else "Unclassified"))
    return importance_df


def make_boxplots(
    cfg: PipelineConfig,
    df_source: pd.DataFrame,
    importance_df: pd.DataFrame,
    output_dir: Path,
    feature_set: str,
    model_name: str,
) -> None:
    if importance_df.empty:
        return
    top_features = importance_df.head(cfg.boxplot_top_n)["feature"].tolist()
    top_features = [f for f in top_features if f in df_source.columns]
    if not top_features:
        return

    for feat in top_features:
        plt.figure(figsize=(5.6, 4.2), dpi=300)
        data = []
        labels = []
        for lab in cfg.labels:
            vals = pd.to_numeric(df_source.loc[df_source[cfg.label_col] == lab, feat], errors="coerce").dropna().values
            if len(vals) == 0:
                vals = np.array([np.nan])
            data.append(vals)
            labels.append(cfg.label_names.get(lab, str(lab)))
        plt.boxplot(data, labels=labels, showfliers=False, patch_artist=True)
        plt.ylabel(feat)
        plt.title(feat)
        plt.tight_layout()
        plt.savefig(output_dir / f"Boxplot_{safe_filename(cfg.task_name)}_{safe_filename(feature_set)}_{safe_filename(model_name)}_{safe_filename(feat)}.png")
        plt.close()

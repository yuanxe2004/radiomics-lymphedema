# -*- coding: utf-8 -*-
"""Feature detection and train-only feature selection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from .config import PipelineConfig
from .utils import as_numeric_df, has_any_keyword, print_table


def is_shape_radiomics_feature(name: str) -> bool:
    s = str(name).lower()
    return ("shape" in s) or ("shape2d" in s) or ("shape_" in s)


def detect_feature_columns(cfg: PipelineConfig, df: pd.DataFrame) -> Dict[str, List[str]]:
    exact_morph = [c for c in cfg.morphology_features if c in df.columns]
    numeric_cols: List[str] = []
    for c in df.columns:
        if str(c).startswith("__"):
            continue
        if c in [cfg.label_col, cfg.center_col, cfg.id_col]:
            continue
        if has_any_keyword(c, cfg.exclude_name_patterns):
            continue
        converted = pd.to_numeric(df[c], errors="coerce")
        if converted.notna().mean() >= 0.60:
            numeric_cols.append(c)

    morph_cols = set(exact_morph)
    if cfg.auto_detect_morphology_by_keywords:
        for c in numeric_cols:
            if c in morph_cols:
                continue
            if has_any_keyword(c, cfg.morphology_keywords) and not has_any_keyword(c, cfg.radiomics_keywords):
                morph_cols.add(c)

    radiomics_cols: List[str] = []
    for c in numeric_cols:
        if c in morph_cols:
            continue
        if has_any_keyword(c, cfg.radiomics_keywords):
            if cfg.remove_radiomics_shape_features and is_shape_radiomics_feature(c):
                continue
            radiomics_cols.append(c)

    if len(radiomics_cols) == 0:
        for c in numeric_cols:
            if c not in morph_cols:
                if cfg.remove_radiomics_shape_features and is_shape_radiomics_feature(c):
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


@dataclass
class SelectionRecorder:
    rows: List[Dict[str, Any]] = field(default_factory=list)

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


def _impute_and_scale(X: pd.DataFrame):
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    Xi = imputer.fit_transform(X)
    Xs = scaler.fit_transform(Xi)
    return Xs, imputer, scaler


def select_radiomics_features(
    cfg: PipelineConfig,
    train_df: pd.DataFrame,
    y: pd.Series,
    candidate_cols: List[str],
    recorder: SelectionRecorder,
    context: str,
    fold: Optional[int],
    feature_set: str,
) -> List[str]:
    cols = [c for c in candidate_cols if c in train_df.columns]
    X = as_numeric_df(train_df, cols)
    cols = [c for c in X.columns if X[c].notna().sum() > 0]
    X = X[cols]
    recorder.add(cfg.task_name, context, fold, feature_set, "radiomics", "00_candidates", cols)
    if len(cols) == 0:
        return []

    Xs, _, _ = _impute_and_scale(X)
    try:
        vt = VarianceThreshold(threshold=cfg.variance_threshold)
        Xv = vt.fit_transform(Xs)
        vt_cols = list(np.array(cols)[vt.get_support()])
    except Exception:
        vt_cols = cols
        Xv = Xs
    recorder.add(cfg.task_name, context, fold, feature_set, "radiomics", "01_variance", vt_cols)
    if len(vt_cols) == 0:
        return []

    k = min(cfg.anova_top_k, len(vt_cols))
    try:
        skb = SelectKBest(score_func=f_classif, k=k)
        skb.fit_transform(Xv, y)
        anova_cols = list(np.array(vt_cols)[skb.get_support()])
        score_map = {c: float(s) if np.isfinite(s) else 0.0 for c, s in zip(vt_cols, skb.scores_)}
        anova_cols = sorted(anova_cols, key=lambda c: score_map.get(c, 0.0), reverse=True)
    except Exception:
        anova_cols = vt_cols[:k]
        score_map = {c: 0.0 for c in vt_cols}
    recorder.add(cfg.task_name, context, fold, feature_set, "radiomics", "02_anova", anova_cols, score_map)
    if len(anova_cols) == 0:
        return []

    X_anova = pd.DataFrame(Xs[:, [cols.index(c) for c in anova_cols]], columns=anova_cols, index=train_df.index)
    pearson_cols = pearson_filter(X_anova, anova_cols, cfg.pearson_threshold_radiomics, score_map)
    recorder.add(cfg.task_name, context, fold, feature_set, "radiomics", "03_pearson", pearson_cols, score_map)
    if len(pearson_cols) == 0:
        return []

    X_lasso = X_anova[pearson_cols].values
    lasso_cols: List[str] = []
    try:
        lasso = LogisticRegression(
            penalty="l1", solver="saga", C=cfg.lasso_c, class_weight="balanced",
            max_iter=cfg.lasso_max_iter, random_state=cfg.random_state,
        )
        lasso.fit(X_lasso, y)
        coefs = np.asarray(lasso.coef_)
        nonzero = np.any(np.abs(coefs) > 1e-8, axis=0)
        coef_scores = {c: float(np.mean(np.abs(coefs[:, i]))) for i, c in enumerate(pearson_cols)}
        lasso_cols = list(np.array(pearson_cols)[nonzero])
        lasso_cols = sorted(lasso_cols, key=lambda c: coef_scores.get(c, 0.0), reverse=True)
    except Exception:
        coef_scores = score_map
        lasso_cols = []
    if len(lasso_cols) == 0:
        lasso_cols = pearson_cols[:min(cfg.min_features_after_lasso_fallback, len(pearson_cols))]
    recorder.add(cfg.task_name, context, fold, feature_set, "radiomics", "04_lasso", lasso_cols, coef_scores)
    return lasso_cols


def select_morphology_features(
    cfg: PipelineConfig,
    train_df: pd.DataFrame,
    y: pd.Series,
    candidate_cols: List[str],
    recorder: SelectionRecorder,
    context: str,
    fold: Optional[int],
    feature_set: str,
) -> List[str]:
    cols = [c for c in candidate_cols if c in train_df.columns]
    X = as_numeric_df(train_df, cols)
    cols = [c for c in X.columns if X[c].notna().sum() > 0]
    X = X[cols]
    recorder.add(cfg.task_name, context, fold, feature_set, "morphology", "00_candidates", cols)
    if len(cols) == 0:
        return []
    Xs, _, _ = _impute_and_scale(X)
    X_scaled = pd.DataFrame(Xs, columns=cols, index=train_df.index)
    try:
        f_scores, _ = f_classif(X_scaled, y)
        score_map = {c: float(s) if np.isfinite(s) else 0.0 for c, s in zip(cols, f_scores)}
    except Exception:
        score_map = {c: 0.0 for c in cols}
    pearson_cols = pearson_filter(X_scaled, cols, cfg.pearson_threshold_morphology, score_map)
    recorder.add(cfg.task_name, context, fold, feature_set, "morphology", "01_pearson", pearson_cols, score_map)
    return pearson_cols


def select_features_for_set(
    cfg: PipelineConfig,
    train_df: pd.DataFrame,
    y: pd.Series,
    feature_set: str,
    feature_pools: Dict[str, List[str]],
    recorder: SelectionRecorder,
    context: str,
    fold: Optional[int],
) -> List[str]:
    if feature_set == "Radiomics_only":
        return select_radiomics_features(cfg, train_df, y, feature_pools["radiomics"], recorder, context, fold, feature_set)
    if feature_set == "Morphology_only":
        return select_morphology_features(cfg, train_df, y, feature_pools["morphology"], recorder, context, fold, feature_set)
    if feature_set == "Combined":
        r_cols = select_radiomics_features(cfg, train_df, y, feature_pools["radiomics"], recorder, context, fold, feature_set)
        m_cols = select_morphology_features(cfg, train_df, y, feature_pools["morphology"], recorder, context, fold, feature_set)
        combined = list(dict.fromkeys(r_cols + m_cols))
        recorder.add(cfg.task_name, context, fold, feature_set, "combined", "05_combined", combined)
        if cfg.combined_final_pearson and len(combined) > 1:
            X = as_numeric_df(train_df, combined)
            Xs, _, _ = _impute_and_scale(X)
            X_scaled = pd.DataFrame(Xs, columns=combined, index=train_df.index)
            combined = pearson_filter(X_scaled, combined, cfg.pearson_threshold_radiomics)
            recorder.add(cfg.task_name, context, fold, feature_set, "combined", "06_combined_final_pearson", combined)
        return combined
    raise ValueError(f"Unknown feature_set: {feature_set}")

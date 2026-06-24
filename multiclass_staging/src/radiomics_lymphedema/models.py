# -*- coding: utf-8 -*-
"""Model construction and probability alignment."""

from __future__ import annotations

from typing import Any, List

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import PipelineConfig


def available_model_names(cfg: PipelineConfig) -> List[str]:
    names = list(cfg.model_names)
    if cfg.use_xgboost_if_available:
        try:
            import xgboost  # noqa: F401
            if "XGB" not in names:
                names.append("XGB")
        except Exception:
            pass
    return names


def build_model(cfg: PipelineConfig, model_name: str) -> Any:
    if model_name == "LR":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                solver="lbfgs", class_weight="balanced", max_iter=5000,
                random_state=cfg.random_state,
            )),
        ])
    if model_name == "RF":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=500, max_depth=None, min_samples_leaf=2,
                class_weight="balanced_subsample", random_state=cfg.random_state, n_jobs=-1,
            )),
        ])
    if model_name == "SVM":
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", SVC(
                kernel="rbf", C=1.0, gamma="scale", probability=True,
                class_weight="balanced", random_state=cfg.random_state,
            )),
        ])
    if model_name == "XGB":
        try:
            from xgboost import XGBClassifier
        except Exception as e:  # pragma: no cover
            raise ImportError("XGBoost is not installed. Set use_xgboost_if_available=false or install xgboost.") from e
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", XGBClassifier(
                objective="multi:softprob", num_class=len(cfg.labels), n_estimators=300,
                learning_rate=0.03, max_depth=3, subsample=0.85,
                colsample_bytree=0.85, reg_lambda=1.0, random_state=cfg.random_state,
                eval_metric="mlogloss", n_jobs=-1,
            )),
        ])
    raise ValueError(f"Unknown model_name: {model_name}")


def get_estimator_classes(model: Any, labels: List[int]) -> np.ndarray:
    if hasattr(model, "classes_"):
        return np.asarray(model.classes_)
    if hasattr(model, "named_steps"):
        for step in reversed(list(model.named_steps.values())):
            if hasattr(step, "classes_"):
                return np.asarray(step.classes_)
    return np.asarray(labels)


def predict_proba_aligned(model: Any, X: pd.DataFrame, labels: List[int]) -> np.ndarray:
    prob = np.asarray(model.predict_proba(X), dtype=float)
    model_classes = list(get_estimator_classes(model, labels))
    aligned = np.zeros((len(X), len(labels)), dtype=float)
    for j, lab in enumerate(labels):
        if lab in model_classes:
            aligned[:, j] = prob[:, model_classes.index(lab)]
        elif j < prob.shape[1]:
            aligned[:, j] = prob[:, j]
    row_sum = aligned.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return aligned / row_sum


def fit_model(cfg: PipelineConfig, model_name: str, X: pd.DataFrame, y: pd.Series):
    model = build_model(cfg, model_name)
    if model_name == "XGB":
        remap = {lab: i for i, lab in enumerate(cfg.labels)}
        model.fit(X, y.map(remap).astype(int))
    else:
        model.fit(X, y.astype(int))
    return model


def predict_labels_and_prob(cfg: PipelineConfig, model: Any, model_name: str, X: pd.DataFrame):
    if model_name == "XGB":
        prob = np.asarray(model.predict_proba(X), dtype=float)[:, :len(cfg.labels)]
        pred = np.asarray([cfg.labels[i] for i in np.argmax(prob, axis=1)])
    else:
        prob = predict_proba_aligned(model, X, cfg.labels)
        pred = np.asarray(cfg.labels, dtype=int)[np.argmax(prob, axis=1)]
    return pred, prob

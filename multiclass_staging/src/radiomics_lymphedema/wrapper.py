# -*- coding: utf-8 -*-
"""Saved model wrapper for downstream prediction."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from .models import predict_proba_aligned
from .utils import timestamp


class BestModelWrapper:
    """Lightweight callable wrapper for the global-best multiclass model."""

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
        self.label_order = list(label_order or [1, 2, 3])
        self.label_names = label_names or {1: "Stage I", 2: "Stage II", 3: "Stage III"}
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
                    f"Array input has {arr.shape[1]} columns, but the model expects {len(self.feature_cols)} features. "
                    "For safety, pass a pandas DataFrame with the original feature names."
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

    def save(self, path: str) -> None:
        payload = {
            "model": self.model,
            "feature_cols": self.feature_cols,
            "features": self.feature_cols,
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
            label_order=obj.get("label_order", [1, 2, 3]),
            label_names=obj.get("label_names", {1: "Stage I", 2: "Stage II", 3: "Stage III"}),
            feature_set=obj.get("feature_set"),
            model_name=obj.get("model_name"),
            task=obj.get("task"),
            metadata=obj.get("metadata", {}),
        )


def load_best_model(path: str) -> BestModelWrapper:
    return BestModelWrapper.load(path)


def save_best_model_artifacts(wrapper: BestModelWrapper, model_dir: str | Path) -> Dict[str, str]:
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model_wrapper.joblib"
    metadata_path = model_dir / "best_model_metadata.json"
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
        "saved_at": timestamp(),
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return {"model_path": str(model_path), "metadata_path": str(metadata_path)}

# -*- coding: utf-8 -*-
"""Training pipeline entry point."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:  # pragma: no cover
    HAS_STRATIFIED_GROUP_KFOLD = False

from .config import PipelineConfig, ensure_dir
from .data import center_and_internal_split, load_and_normalize_data
from .explain import make_boxplots, make_shap_or_importance_outputs
from .features import SelectionRecorder, detect_feature_columns, select_features_for_set
from .metrics import bootstrap_metrics_ci, multiclass_metrics, summarize_fold_metrics
from .models import available_model_names, fit_model, predict_labels_and_prob
from .plots import (
    plot_calibration_comparison,
    plot_confusion_matrix,
    plot_dca_comparison,
    plot_roc_comparison,
    set_publication_plot_style,
)
from .utils import as_numeric_df, format_ci, safe_filename, timestamp, write_sheet_safe, print_table
from .wrapper import BestModelWrapper, save_best_model_artifacts


def effective_n_splits(cfg: PipelineConfig, y: pd.Series) -> int:
    counts = y.value_counts()
    min_class = int(counts.min()) if len(counts) else 0
    n = min(cfg.n_splits, min_class)
    if cfg.use_group_split:
        n = min(n, y.shape[0])
    if n < 2:
        raise ValueError("Not enough samples per class to perform cross-validation.")
    return n


def get_cv_splits(cfg: PipelineConfig, dev_df: pd.DataFrame, y: pd.Series):
    n_splits = effective_n_splits(cfg, y)
    if cfg.use_group_split and cfg.id_col in dev_df.columns and HAS_STRATIFIED_GROUP_KFOLD:
        groups = dev_df[cfg.id_col].astype(str).values
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
        try:
            return list(cv.split(dev_df, y, groups=groups))
        except Exception as e:
            print(f"WARNING: StratifiedGroupKFold failed ({e}). Falling back to StratifiedKFold.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    return list(cv.split(dev_df, y))


def run_cv_candidate(cfg: PipelineConfig, dev_df: pd.DataFrame, feature_pools, feature_set: str, model_name: str, recorder: SelectionRecorder):
    y = dev_df[cfg.label_col].astype(int).reset_index(drop=True)
    dev_df_reset = dev_df.reset_index(drop=True)
    splits = get_cv_splits(cfg, dev_df_reset, y)
    fold_rows: List[Dict[str, Any]] = []
    for fold_idx, (tr_idx, va_idx) in enumerate(splits, start=1):
        fold_train = dev_df_reset.iloc[tr_idx].copy()
        fold_valid = dev_df_reset.iloc[va_idx].copy()
        y_tr = fold_train[cfg.label_col].astype(int)
        y_va = fold_valid[cfg.label_col].astype(int).values
        selected = select_features_for_set(cfg, fold_train, y_tr, feature_set, feature_pools, recorder, "cv_fold_train_only", fold_idx)
        row_base = {
            "task": cfg.task_name,
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
        X_tr = as_numeric_df(fold_train, selected)
        X_va = as_numeric_df(fold_valid, selected)
        model = fit_model(cfg, model_name, X_tr, y_tr)
        pred, prob = predict_labels_and_prob(cfg, model, model_name, X_va)
        m = multiclass_metrics(y_va, pred, prob, cfg.labels)
        row_base.update({k: m[k] for k in ["ACC", "AUC", "SENS", "SPEC", "PPV", "NPV"]})
        fold_rows.append(row_base)
    summary = summarize_fold_metrics(fold_rows)
    summary.update({
        "task": cfg.task_name,
        "feature_set": feature_set,
        "model": model_name,
        "n_folds": len(splits),
        "n_dev_train_plus_val": len(dev_df),
        "mean_n_selected_features": float(pd.Series([r["n_selected_features"] for r in fold_rows]).mean()),
    })
    return summary, fold_rows


def fit_final_model(cfg: PipelineConfig, train_df_for_fit: pd.DataFrame, feature_pools, feature_set: str, model_name: str, recorder: SelectionRecorder):
    y_fit = train_df_for_fit[cfg.label_col].astype(int)
    selected = select_features_for_set(cfg, train_df_for_fit, y_fit, feature_set, feature_pools, recorder, "final_train_plus_validation", None)
    if len(selected) == 0:
        raise RuntimeError(f"No selected features for final model: {feature_set} / {model_name}")
    X_fit = as_numeric_df(train_df_for_fit, selected)
    model = fit_model(cfg, model_name, X_fit, y_fit)
    return model, selected


def evaluate_dataset(cfg: PipelineConfig, model, selected_features, df_eval, split_name, feature_set, model_name, bootstrap_seed_offset=0):
    X = as_numeric_df(df_eval, selected_features)
    y_true = df_eval[cfg.label_col].astype(int).values
    pred, prob = predict_labels_and_prob(cfg, model, model_name, X)
    metrics = multiclass_metrics(y_true, pred, prob, cfg.labels)
    ci = bootstrap_metrics_ci(y_true, pred, prob, cfg.labels, cfg.n_bootstrap, cfg.bootstrap_seed + bootstrap_seed_offset)
    row = {
        "task": cfg.task_name,
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
    for i, true_lab in enumerate(cfg.labels):
        for j, pred_lab in enumerate(cfg.labels):
            row[f"CM_true{true_lab}_pred{pred_lab}"] = int(metrics["CM"][i, j])
    pred_cols = [c for c in [cfg.id_col, cfg.center_col, cfg.label_col, "__split__"] if c in df_eval.columns]
    pred_df = df_eval[pred_cols].copy()
    pred_df["task"] = cfg.task_name
    pred_df["feature_set"] = feature_set
    pred_df["model"] = model_name
    pred_df["pred_label"] = pred
    pred_df["correct"] = (pred == y_true)
    pred_df["max_prob"] = np.max(prob, axis=1)
    pred_df["uncertainty"] = 1.0 - pred_df["max_prob"]
    for j, lab in enumerate(cfg.labels):
        pred_df[f"prob_stage_{lab}"] = prob[:, j]
    return row, pred_df


def choose_best_candidate(candidate_df: pd.DataFrame) -> pd.Series:
    if candidate_df.empty:
        raise RuntimeError("No candidate model results available.")
    tmp = candidate_df.copy()
    for c in ["AUC_mean", "ACC_mean", "PPV_mean"]:
        tmp[c] = pd.to_numeric(tmp[c], errors="coerce").fillna(-999)
    return tmp.sort_values(by=["AUC_mean", "ACC_mean", "PPV_mean"], ascending=[False, False, False]).iloc[0]


def choose_best_by_feature_set(cfg: PipelineConfig, candidate_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fs in cfg.feature_sets:
        sub = candidate_df[candidate_df["feature_set"] == fs]
        if not sub.empty:
            rows.append(choose_best_candidate(sub).to_dict())
    return pd.DataFrame(rows)


def make_all_main_plots(cfg: PipelineConfig, predictions: pd.DataFrame, metrics_rows: List[Dict[str, Any]], output_dir: Path) -> None:
    if cfg.run_roc:
        for dataset_name in ["internal_test", "external_test"]:
            plot_roc_comparison(predictions, output_dir, dataset_name, cfg.task_name, cfg.labels, cfg.label_col, cfg.feature_sets)
    if cfg.run_dca:
        for dataset_name in ["internal_test", "external_test"]:
            plot_dca_comparison(predictions, output_dir, dataset_name, cfg.task_name, cfg.labels, cfg.label_col, cfg.feature_sets)
    if cfg.run_calibration:
        for dataset_name in ["internal_test", "external_test"]:
            plot_calibration_comparison(predictions, output_dir, dataset_name, cfg.task_name, cfg.feature_sets)
    for row in metrics_rows:
        if row.get("dataset") in ["internal_test", "external_test"]:
            cm = np.zeros((len(cfg.labels), len(cfg.labels)), dtype=int)
            for i, true_lab in enumerate(cfg.labels):
                for j, pred_lab in enumerate(cfg.labels):
                    cm[i, j] = int(row.get(f"CM_true{true_lab}_pred{pred_lab}", 0))
            plot_confusion_matrix(
                cm, output_dir,
                title=f"{row.get('feature_set')} / {row.get('model')} / {row.get('dataset')}",
                filename=f"CM_{safe_filename(cfg.task_name)}_{safe_filename(row.get('feature_set'))}_{safe_filename(row.get('model'))}_{safe_filename(row.get('dataset'))}.png",
                labels=cfg.labels,
                label_names=cfg.label_names,
            )


def run_training_pipeline(cfg: PipelineConfig) -> Dict[str, str]:
    set_publication_plot_style()
    out_dir = ensure_dir(cfg.output_dir)
    plot_dir = ensure_dir(out_dir / "figures")
    model_dir = ensure_dir(out_dir / "models")
    workbook_path = out_dir / f"multiclass_sci_results_{timestamp()}.xlsx"

    df = load_and_normalize_data(cfg)
    train_df, val_df, internal_test_df, external_df = center_and_internal_split(cfg, df)
    trainval_df = pd.concat([train_df, val_df], axis=0).copy().reset_index(drop=True)
    feature_pools = detect_feature_columns(cfg, df)
    if len(feature_pools["radiomics"]) == 0 and len(feature_pools["morphology"]) == 0:
        raise ValueError("No numeric feature columns detected.")

    model_names = available_model_names(cfg)
    print("\nModels to evaluate:", model_names)
    selection_recorder = SelectionRecorder()
    candidate_summaries: List[Dict[str, Any]] = []
    candidate_fold_rows: List[Dict[str, Any]] = []

    for fs in cfg.feature_sets:
        if fs == "Morphology_only" and len(feature_pools["morphology"]) == 0:
            print("WARNING: Morphology_only skipped because no morphology columns were detected."); continue
        if fs == "Radiomics_only" and len(feature_pools["radiomics"]) == 0:
            print("WARNING: Radiomics_only skipped because no radiomics columns were detected."); continue
        for model_name in model_names:
            print(f"\nRunning CV: {fs} / {model_name}")
            try:
                summary, folds = run_cv_candidate(cfg, trainval_df, feature_pools, fs, model_name, selection_recorder)
                candidate_summaries.append(summary)
                candidate_fold_rows.extend(folds)
                print(f"CV AUC: {summary.get('AUC_mean_sd')} | ACC: {summary.get('ACC_mean_sd')}")
            except Exception as e:
                print(f"WARNING: CV failed for {fs} / {model_name}: {e}")
                candidate_summaries.append({"task": cfg.task_name, "feature_set": fs, "model": model_name, "error": str(e), "ACC_mean": np.nan, "AUC_mean": np.nan, "SENS_mean": np.nan, "SPEC_mean": np.nan, "PPV_mean": np.nan, "NPV_mean": np.nan})

    candidate_df = pd.DataFrame(candidate_summaries)
    fold_df = pd.DataFrame(candidate_fold_rows)
    valid_candidates = candidate_df[pd.to_numeric(candidate_df.get("AUC_mean", pd.Series(dtype=float)), errors="coerce").notna()].copy()
    if valid_candidates.empty:
        raise RuntimeError("All candidate models failed. Check features, labels, and dependencies.")
    global_best = choose_best_candidate(valid_candidates)
    best_by_feature_set = choose_best_by_feature_set(cfg, valid_candidates)
    print_table("Global best candidate", global_best[["feature_set", "model", "AUC_mean", "ACC_mean", "PPV_mean"]])
    print_table("Best candidate by feature set", best_by_feature_set[["feature_set", "model", "AUC_mean", "ACC_mean", "PPV_mean"]])

    eval_datasets = {"train": train_df, "validation": val_df, "internal_test": internal_test_df, "external_test": external_df}
    all_feature_eval_rows: List[Dict[str, Any]] = []
    all_feature_predictions: List[pd.DataFrame] = []
    final_models: Dict[str, Dict[str, Any]] = {}
    used_feature_rows: List[Dict[str, Any]] = []

    for _, row in best_by_feature_set.iterrows():
        fs = row["feature_set"]
        model_name = row["model"]
        print(f"\nFitting final best-by-feature-set model: {fs} / {model_name}")
        final_model, selected = fit_final_model(cfg, trainval_df, feature_pools, fs, model_name, selection_recorder)
        final_models[fs] = {"model": final_model, "features": selected, "model_name": model_name}
        for rank, feat in enumerate(selected, start=1):
            used_feature_rows.append({"task": cfg.task_name, "feature_set": fs, "model": model_name, "is_global_best": bool(fs == global_best["feature_set"] and model_name == global_best["model"]), "feature_rank": rank, "feature": feat, "feature_domain": "morphology" if feat in feature_pools["morphology"] else "radiomics"})
        for k, (split_name, eval_df) in enumerate(eval_datasets.items(), start=1):
            eval_row, pred_df = evaluate_dataset(cfg, final_model, selected, eval_df, split_name, fs, model_name, 1000 * k + len(all_feature_eval_rows))
            all_feature_eval_rows.append(eval_row)
            all_feature_predictions.append(pred_df)

    metrics_all_feature_sets = pd.DataFrame(all_feature_eval_rows)
    predictions_all_feature_sets = pd.concat(all_feature_predictions, axis=0, ignore_index=True) if all_feature_predictions else pd.DataFrame()

    best_fs = str(global_best["feature_set"])
    best_model_name = str(global_best["model"])
    if best_fs not in final_models:
        global_model, global_features = fit_final_model(cfg, trainval_df, feature_pools, best_fs, best_model_name, selection_recorder)
    else:
        global_model = final_models[best_fs]["model"]
        global_features = final_models[best_fs]["features"]

    global_eval_rows, global_pred_rows = [], []
    for k, (split_name, eval_df) in enumerate(eval_datasets.items(), start=1):
        eval_row, pred_df = evaluate_dataset(cfg, global_model, global_features, eval_df, split_name, best_fs, best_model_name, 5000 + k)
        global_eval_rows.append(eval_row)
        global_pred_rows.append(pred_df)
    metrics_global_best = pd.DataFrame(global_eval_rows)
    all_cases_predictions = pd.concat(global_pred_rows, axis=0, ignore_index=True)
    wrong_global_best = all_cases_predictions[~all_cases_predictions["correct"]].copy()
    split_samples = pd.concat([train_df, val_df, internal_test_df, external_df], axis=0, ignore_index=True)

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

    make_all_main_plots(cfg, predictions_all_feature_sets, all_feature_eval_rows, plot_dir)
    importance_df = make_shap_or_importance_outputs(cfg, global_model, global_features, trainval_df, plot_dir, best_fs, best_model_name)
    make_boxplots(cfg, trainval_df, importance_df, plot_dir, best_fs, best_model_name)

    wrapper = BestModelWrapper(
        model=global_model,
        feature_cols=global_features,
        label_order=cfg.labels,
        label_names=cfg.label_names,
        feature_set=best_fs,
        model_name=best_model_name,
        task=cfg.task_name,
        metadata={"excel_file": cfg.excel_file, "output_dir": str(out_dir), "n_train_plus_validation": len(trainval_df)},
    )
    model_paths = save_best_model_artifacts(wrapper, model_dir)
    metrics_global_best["best_model_wrapper_path"] = model_paths["model_path"]
    metrics_global_best["best_model_metadata_path"] = model_paths["metadata_path"]

    feature_selection_df = pd.DataFrame(selection_recorder.rows)
    used_features_df = pd.DataFrame(used_feature_rows)
    feature_pool_df = pd.DataFrame(
        [{"domain": "radiomics", "feature": f} for f in feature_pools["radiomics"]] +
        [{"domain": "morphology", "feature": f} for f in feature_pools["morphology"]]
    )
    run_info = pd.DataFrame([{"key": k, "value": json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v} for k, v in cfg.to_dict().items()] + [
        {"key": "global_best_feature_set", "value": best_fs},
        {"key": "global_best_model", "value": best_model_name},
        {"key": "global_best_n_features", "value": len(global_features)},
        {"key": "best_model_wrapper_path", "value": model_paths["model_path"]},
        {"key": "best_model_metadata_path", "value": model_paths["metadata_path"]},
    ])

    with pd.ExcelWriter(workbook_path, engine="openpyxl") as writer:
        write_sheet_safe(writer, metrics_global_best, "metrics_global_best", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, candidate_df, "candidate_5fold_cv_all_models", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, fold_df, "candidate_5fold_cv_folds", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, metrics_all_feature_sets, "metrics_all_feature_sets", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, feature_selection_df, "feature_selection_features", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, used_features_df, "used_features_by_task", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, wrong_global_best, "wrong_global_best", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, split_samples, "split_samples", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, all_cases_predictions, "all_cases_predictions", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, predictions_all_feature_sets, "predictions_all_feature_sets", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, importance_df, "shap_or_importance", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, feature_pool_df, "feature_pools", cfg.max_excel_rows_per_sheet)
        write_sheet_safe(writer, run_info, "run_info", cfg.max_excel_rows_per_sheet)

    print("\nSaved workbook:", workbook_path)
    print("Saved figures:", plot_dir)
    print("Saved best model:", model_paths["model_path"])
    return {"workbook_path": str(workbook_path), "figures_dir": str(plot_dir), **model_paths}

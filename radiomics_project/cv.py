
from .config import *
from .features import make_feature_selection_feature_rows, make_used_features_rows, select_features_by_feature_set
from .metrics import calc_binary_metrics_with_ci, calc_binary_metrics_without_ci
from .models import fit_model, fit_predict_model, predict_with_model
from .utils import (
    FinalArtifact,
    format_mean_sd,
    make_experiment_id,
    make_label_count_fields,
    task_name_from_classes,
    to_binary_labels,
)


def make_cv_splits(trainval_df, y_trainval, n_splits=CV_N_SPLITS, random_state=RANDOM_STATE):
    """
    在 Train + Validation 数据上生成交叉验证折。
    优先使用 StratifiedGroupKFold 保持 ID_COL 分组不泄漏；若不可用或失败，则退回 StratifiedKFold。
    """
    y_trainval = np.asarray(y_trainval).astype(int)
    class_counts = pd.Series(y_trainval).value_counts()
    min_class_count = int(class_counts.min())

    if min_class_count < 2:
        raise ValueError(f"Train + Validation 中至少一个类别样本数 < 2，无法进行交叉验证。class_counts={class_counts.to_dict()}")

    effective_splits = min(int(n_splits), min_class_count)
    if effective_splits < n_splits:
        logging.warning(
            "由于最少类别样本数为 %d，实际 CV 折数从 %d 调整为 %d。",
            min_class_count,
            n_splits,
            effective_splits,
        )

    if effective_splits < 2:
        raise ValueError("有效 CV 折数 < 2，无法进行交叉验证。")

    x_dummy = np.zeros((len(y_trainval), 1))

    use_group_cv = (
        USE_GROUP_SPLIT
        and HAS_STRATIFIED_GROUP_KFOLD
        and ID_COL in trainval_df.columns
        and trainval_df[ID_COL].notna().all()
        and trainval_df[ID_COL].astype(str).nunique() >= effective_splits
    )

    if use_group_cv:
        groups = trainval_df[ID_COL].astype(str).values
        try:
            cv = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
            splits = list(cv.split(x_dummy, y_trainval, groups=groups))
            return splits, f"StratifiedGroupKFold by {ID_COL}", effective_splits
        except Exception as e:
            logging.warning("StratifiedGroupKFold 失败，退回 StratifiedKFold。原因：%s", e)

    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=random_state)
    splits = list(cv.split(x_dummy, y_trainval))
    return splits, "StratifiedKFold", effective_splits


def aggregate_cv_fold_metrics(fold_metrics):
    metric_names = [
        "ACC", "AUC", "PPV", "NPV", "Sensitivity", "Specificity",
        "F1", "Balanced_ACC", "Brier", "Calibration_slope", "Calibration_intercept",
    ]

    out = {}
    for metric_name in metric_names:
        raw_key = f"{metric_name}_raw"
        values = np.asarray([m.get(raw_key, np.nan) for m in fold_metrics], dtype=float)
        valid_values = values[~np.isnan(values)]
        mean_value = float(np.mean(valid_values)) if len(valid_values) > 0 else np.nan
        sd_value = float(np.std(valid_values, ddof=1)) if len(valid_values) > 1 else np.nan

        out[f"CV_{metric_name}_mean"] = mean_value
        out[f"CV_{metric_name}_sd"] = sd_value
        out[f"CV_{metric_name}"] = format_mean_sd(mean_value, sd_value)
        out[f"CV_{metric_name}_有效折数"] = int(len(valid_values))

    # 为了沿用 is_better_model 的比较逻辑，将 raw 字段设为 CV mean。
    out["ACC_raw"] = out["CV_ACC_mean"]
    out["AUC_raw"] = out["CV_AUC_mean"]
    out["PPV_raw"] = out["CV_PPV_mean"]
    out["NPV_raw"] = out["CV_NPV_mean"]
    out["Sensitivity_raw"] = out["CV_Sensitivity_mean"]
    out["Specificity_raw"] = out["CV_Specificity_mean"]
    out["F1_raw"] = out["CV_F1_mean"]
    out["Balanced_ACC_raw"] = out["CV_Balanced_ACC_mean"]
    out["Brier_raw"] = out["CV_Brier_mean"]
    out["Calibration_slope_raw"] = out["CV_Calibration_slope_mean"]
    out["Calibration_intercept_raw"] = out["CV_Calibration_intercept_mean"]

    # 候选模型日志中直接显示 mean±sd。
    out["ACC"] = out["CV_ACC"]
    out["AUC"] = out["CV_AUC"]
    out["PPV"] = out["CV_PPV"]
    out["NPV"] = out["CV_NPV"]
    out["Sensitivity"] = out["CV_Sensitivity"]
    out["Specificity"] = out["CV_Specificity"]
    out["F1"] = out["CV_F1"]
    out["Balanced_ACC"] = out["CV_Balanced_ACC"]
    out["Brier"] = out["CV_Brier"]
    out["Calibration_slope"] = out["CV_Calibration_slope"]
    out["Calibration_intercept"] = out["CV_Calibration_intercept"]

    return out


def get_cv_summary_columns(cv_row):
    if cv_row is None:
        return {}

    keep_cols = [
        "CV方法", "CV折数", "CV模型选择数据", "CV特征筛选数据",
        "CV_ACC", "CV_ACC_mean", "CV_ACC_sd", "CV_ACC_有效折数",
        "CV_AUC", "CV_AUC_mean", "CV_AUC_sd", "CV_AUC_有效折数",
        "CV_PPV", "CV_PPV_mean", "CV_PPV_sd", "CV_PPV_有效折数",
        "CV_NPV", "CV_NPV_mean", "CV_NPV_sd", "CV_NPV_有效折数",
        "CV_Sensitivity", "CV_Sensitivity_mean", "CV_Sensitivity_sd", "CV_Sensitivity_有效折数",
        "CV_Specificity", "CV_Specificity_mean", "CV_Specificity_sd", "CV_Specificity_有效折数",
        "CV_F1", "CV_F1_mean", "CV_F1_sd", "CV_F1_有效折数",
        "CV_Balanced_ACC", "CV_Balanced_ACC_mean", "CV_Balanced_ACC_sd", "CV_Balanced_ACC_有效折数",
        "CV_Brier", "CV_Brier_mean", "CV_Brier_sd", "CV_Brier_有效折数",
        "CV_Calibration_slope", "CV_Calibration_slope_mean", "CV_Calibration_slope_sd", "CV_Calibration_slope_有效折数",
        "CV_Calibration_intercept", "CV_Calibration_intercept_mean", "CV_Calibration_intercept_sd", "CV_Calibration_intercept_有效折数",
    ]

    return {col: cv_row.get(col, np.nan) for col in keep_cols if col in cv_row}


def run_candidate_on_trainval_cv(analysis_label, data, class_a, class_b, feature_set_name, model_name, radiomics_cols, morph_cols):
    task_name = task_name_from_classes(class_a, class_b)
    positive_label = class_b
    labels_sorted = sorted([class_a, class_b])

    trainval_df = pd.concat([data.train_df, data.val_df], axis=0).reset_index(drop=True)
    x_trainval_all = pd.concat([data.x_train_all, data.x_val_all], axis=0).reset_index(drop=True)
    y_trainval = np.concatenate([data.y_train, data.y_val])

    cv_splits, cv_method, effective_splits = make_cv_splits(
        trainval_df=trainval_df,
        y_trainval=y_trainval,
        n_splits=CV_N_SPLITS,
        random_state=RANDOM_STATE,
    )

    fold_metrics = []
    fold_feature_counts = []
    feature_selection_rows = []

    for fold_idx, (cv_train_idx, cv_val_idx) in enumerate(cv_splits, start=1):
        x_fold_train = x_trainval_all.iloc[cv_train_idx].copy().reset_index(drop=True)
        y_fold_train = y_trainval[cv_train_idx]
        x_fold_val = x_trainval_all.iloc[cv_val_idx].copy().reset_index(drop=True)
        y_fold_val = y_trainval[cv_val_idx]

        fs_result = select_features_by_feature_set(
            feature_set_name=feature_set_name,
            x_train_all=x_fold_train,
            y_train=y_fold_train,
            valid_radio_cols=data.valid_radio_cols,
            valid_morph_cols=data.valid_morph_cols,
        )

        feature_selection_rows.extend(make_feature_selection_feature_rows(
            analysis_label=analysis_label,
            result_tag="Candidate_5fold_CV",
            task_name=task_name,
            feature_set_name=feature_set_name,
            model_name=model_name,
            stage_label=f"CV_fold_{fold_idx}",
            data_source="CV training fold only",
            fs_result=fs_result,
            valid_radio_cols=data.valid_radio_cols,
            valid_morph_cols=data.valid_morph_cols,
            cv_fold=fold_idx,
        ))

        val_pred, val_prob, _ = fit_predict_model(
            model_name=model_name,
            x_train=x_fold_train[fs_result.selected_cols],
            y_train=y_fold_train,
            x_test=x_fold_val[fs_result.selected_cols],
            labels_sorted=labels_sorted,
            positive_label=positive_label,
        )

        metrics = calc_binary_metrics_without_ci(
            y_true=y_fold_val,
            y_pred=val_pred,
            y_prob=val_prob,
            positive_label=positive_label,
            labels_sorted=labels_sorted,
        )
        fold_metrics.append(metrics)
        fold_feature_counts.append({
            "Radiomics最终特征数": len(fs_result.selected_radio_cols),
            "Morphology最终特征数": len(fs_result.selected_morph_cols),
            "总特征数": len(fs_result.selected_cols),
        })

        logging.info(
            "CV fold %d/%d | %s | %s | %s | AUC=%s ACC=%s PPV=%s Sens=%s Spec=%s F1=%s | 总特征数=%d | TN=%s FP=%s FN=%s TP=%s",
            fold_idx,
            effective_splits,
            task_name,
            feature_set_name,
            model_name,
            metrics["AUC"],
            metrics["ACC"],
            metrics["PPV"],
            metrics["Sensitivity"],
            metrics["Specificity"],
            metrics["F1"],
            len(fs_result.selected_cols),
            metrics["TN"],
            metrics["FP"],
            metrics["FN"],
            metrics["TP"],
        )

    cv_summary = aggregate_cv_fold_metrics(fold_metrics)

    final_fs_result = select_features_by_feature_set(
        feature_set_name=feature_set_name,
        x_train_all=x_trainval_all,
        y_train=y_trainval,
        valid_radio_cols=data.valid_radio_cols,
        valid_morph_cols=data.valid_morph_cols,
    )

    feature_selection_rows.extend(make_feature_selection_feature_rows(
        analysis_label=analysis_label,
        result_tag="Candidate_5fold_CV_final_trainval_refit",
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        stage_label="Final_Train_plus_Validation_feature_selection_for_candidate_summary",
        data_source="Train + Validation only",
        fs_result=final_fs_result,
        valid_radio_cols=data.valid_radio_cols,
        valid_morph_cols=data.valid_morph_cols,
        cv_fold=None,
    ))

    row = {
        **base_result_row(
            analysis_label=analysis_label,
            result_tag="Candidate_5fold_CV",
            task_name=task_name,
            feature_set_name=feature_set_name,
            model_name=model_name,
            dataset_name="Train_plus_Validation_5fold_CV",
            data=data,
            class_a=class_a,
            class_b=class_b,
            radiomics_cols=radiomics_cols,
            morph_cols=morph_cols,
            fs_result=final_fs_result,
        ),
        "CV方法": cv_method,
        "CV折数": effective_splits,
        "CV模型选择数据": "Train + Validation",
        "CV特征筛选数据": "Within each CV training fold only",
        "候选模型训练数据": "5-fold CV on Train + Validation",
        "候选模型筛选数据": "Each fold training subset only",
        **cv_summary,
    }

    return {
        "row": row,
        "feature_selection_rows": feature_selection_rows,
    }




# 10. 结果表构造


def base_result_row(analysis_label, result_tag, task_name, feature_set_name, model_name, dataset_name, data, class_a, class_b, radiomics_cols, morph_cols, fs_result):
    return {
        "分析类型": analysis_label,
        "结果类型": result_tag,
        "任务": task_name,
        "特征集": feature_set_name,
        "模型": model_name,
        "数据集": dataset_name,
        "内部中心": INTERNAL_CENTER,
        "外部中心": EXTERNAL_CENTER,
        "划分方式": data.split_method,
        "训练样本量": len(data.train_df),
        "验证样本量": len(data.val_df),
        "内部测试样本量": len(data.internal_test_df),
        "外部测试样本量": len(data.external_df),
        **make_label_count_fields(data, class_a, class_b),
        "Radiomics原始特征数_已删除shape": len(radiomics_cols),
        "Radiomics删除缺失特征数": len(data.dropped_radio_cols),
        "Morphology原始特征数": len(morph_cols),
        "Morphology删除缺失特征数": len(data.dropped_morph_cols),
        **fs_result.counts,
        "Radiomics最终特征数": len(fs_result.selected_radio_cols),
        "Morphology最终特征数": len(fs_result.selected_morph_cols),
        "总特征数": len(fs_result.selected_cols),
        "阳性标签": class_b,
    }


def make_wrong_df(base_df, y_true, y_pred, y_prob, analysis_label, task_name, feature_set_name, model_name, data_split, result_tag):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)

    wrong_mask = y_true != y_pred
    wrong_df = base_df.loc[wrong_mask].copy()

    keep_cols = [col for col in [ID_COL, SIDE_COL, CENTER_COL, LABEL_COL] if col in wrong_df.columns]
    wrong_df = wrong_df[keep_cols].copy()

    exp_id = make_experiment_id(
        analysis_label=analysis_label,
        result_tag=result_tag,
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        data_split=data_split,
    )

    wrong_df["实验ID"] = exp_id
    wrong_df["分析类型"] = analysis_label
    wrong_df["结果类型"] = result_tag
    wrong_df["任务"] = task_name
    wrong_df["特征集"] = feature_set_name
    wrong_df["模型"] = model_name
    wrong_df["数据集"] = data_split
    wrong_df["正确标签"] = y_true[wrong_mask]
    wrong_df["预测标签"] = y_pred[wrong_mask]
    wrong_df["预测为阳性概率"] = y_prob[wrong_mask]
    wrong_df["错误类型"] = np.where(
        y_pred[wrong_mask].astype(int) > y_true[wrong_mask].astype(int),
        "False_positive_or_higher_label",
        "False_negative_or_lower_label",
    )
    return wrong_df


def make_all_case_predictions_df(base_df, y_true, y_pred, y_prob, analysis_label, task_name, feature_set_name, model_name, data_split, result_tag, positive_label):
    """
    输出所有病例的预测结果。
    注意：Train 和 Validation 的预测来自最终模型，该模型使用 Train + Validation 训练，
    因此 Train / Validation 预测属于最终模型的拟合后预测，不是独立验证性能。
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)

    out_df = base_df.copy()
    keep_cols = [col for col in [ID_COL, SIDE_COL, CENTER_COL, LABEL_COL] if col in out_df.columns]
    out_df = out_df[keep_cols].copy()

    exp_id = make_experiment_id(
        analysis_label=analysis_label,
        result_tag=result_tag,
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        data_split=data_split,
    )

    out_df["实验ID"] = exp_id
    out_df["分析类型"] = analysis_label
    out_df["结果类型"] = result_tag
    out_df["任务"] = task_name
    out_df["特征集"] = feature_set_name
    out_df["模型"] = model_name
    out_df["数据集"] = data_split
    out_df["阳性标签"] = positive_label
    out_df["对应标签"] = y_true
    out_df["预测标签"] = y_pred
    out_df["预测值_预测为阳性概率"] = y_prob
    out_df["是否预测正确"] = y_true == y_pred
    out_df["错误类型"] = np.where(
        y_true == y_pred,
        "Correct",
        np.where(
            y_pred.astype(int) > y_true.astype(int),
            "False_positive_or_higher_label",
            "False_negative_or_lower_label",
        ),
    )
    return out_df


def make_split_samples_df(analysis_label, task_name, data):
    """
    输出每个任务的具体样本划分列表。
    Train / Validation / Internal test 来自内部中心中心134的6:2:2划分；
    External test 来自外部中心中心2，不参与随机划分。
    """
    rows = []
    split_items = [
        ("Train_center134_60percent", data.train_df),
        ("Validation_center134_20percent", data.val_df),
        ("Internal_test_center134_20percent", data.internal_test_df),
        ("External_test_center2", data.external_df),
    ]
    keep_cols = [col for col in [ID_COL, SIDE_COL, CENTER_COL, LABEL_COL] if col in data.task_df.columns]

    for split_name, split_df in split_items:
        if split_df is None or len(split_df) == 0:
            continue
        tmp = split_df[keep_cols].copy()
        tmp.insert(0, "分析类型", analysis_label)
        tmp.insert(1, "任务", task_name)
        tmp.insert(2, "数据集", split_name)
        tmp["随机种子"] = RANDOM_STATE
        tmp["划分方式"] = data.split_method
        tmp["是否外部测试集"] = split_name == "External_test_center2"
        rows.append(tmp)

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()




# 11. 最终模型


def evaluate_final_model(analysis_label, data, class_a, class_b, feature_set_name, model_name, radiomics_cols, morph_cols, result_tag="Global_best", cv_summary_row=None):
    task_name = task_name_from_classes(class_a, class_b)
    positive_label = class_b
    labels_sorted = sorted([class_a, class_b])

    trainval_df = pd.concat([data.train_df, data.val_df], axis=0).reset_index(drop=True)
    x_trainval_all = pd.concat([data.x_train_all, data.x_val_all], axis=0).reset_index(drop=True)
    y_trainval = np.concatenate([data.y_train, data.y_val])

    fs_result = select_features_by_feature_set(
        feature_set_name=feature_set_name,
        x_train_all=x_trainval_all,
        y_train=y_trainval,
        valid_radio_cols=data.valid_radio_cols,
        valid_morph_cols=data.valid_morph_cols,
    )

    final_model = fit_model(
        model_name=model_name,
        x_train=x_trainval_all[fs_result.selected_cols],
        y_train=y_trainval,
        labels_sorted=labels_sorted,
    )

    train_pred, train_prob = predict_with_model(
        model=final_model,
        model_name=model_name,
        x_test=data.x_train_all[fs_result.selected_cols],
        labels_sorted=labels_sorted,
        positive_label=positive_label,
    )

    val_pred, val_prob = predict_with_model(
        model=final_model,
        model_name=model_name,
        x_test=data.x_val_all[fs_result.selected_cols],
        labels_sorted=labels_sorted,
        positive_label=positive_label,
    )

    internal_pred, internal_prob = predict_with_model(
        model=final_model,
        model_name=model_name,
        x_test=data.x_internal_test_all[fs_result.selected_cols],
        labels_sorted=labels_sorted,
        positive_label=positive_label,
    )

    external_pred, external_prob = predict_with_model(
        model=final_model,
        model_name=model_name,
        x_test=data.x_external_all[fs_result.selected_cols],
        labels_sorted=labels_sorted,
        positive_label=positive_label,
    )

    internal_metrics = calc_binary_metrics_with_ci(
        y_true=data.y_internal_test,
        y_pred=internal_pred,
        y_prob=internal_prob,
        positive_label=positive_label,
        labels_sorted=labels_sorted,
        random_state=RANDOM_STATE + 100,
    )

    external_metrics = calc_binary_metrics_with_ci(
        y_true=data.y_external,
        y_pred=external_pred,
        y_prob=external_prob,
        positive_label=positive_label,
        labels_sorted=labels_sorted,
        random_state=RANDOM_STATE + 200,
    )

    cv_summary_cols = get_cv_summary_columns(cv_summary_row)

    feature_selection_rows = make_feature_selection_feature_rows(
        analysis_label=analysis_label,
        result_tag=result_tag,
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        stage_label="Final_model_Train_plus_Validation_feature_selection",
        data_source="Train + Validation only",
        fs_result=fs_result,
        valid_radio_cols=data.valid_radio_cols,
        valid_morph_cols=data.valid_morph_cols,
        cv_fold=None,
    )

    used_features_rows = make_used_features_rows(
        analysis_label=analysis_label,
        result_tag=result_tag,
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        fs_result=fs_result,
        radio_cols=radiomics_cols,
        morph_cols=morph_cols,
    )

    all_case_predictions_df = pd.concat([
        make_all_case_predictions_df(
            base_df=data.train_df,
            y_true=data.y_train,
            y_pred=train_pred,
            y_prob=train_prob,
            analysis_label=analysis_label,
            task_name=task_name,
            feature_set_name=feature_set_name,
            model_name=model_name,
            data_split="Train_center134_60percent",
            result_tag=result_tag,
            positive_label=positive_label,
        ),
        make_all_case_predictions_df(
            base_df=data.val_df,
            y_true=data.y_val,
            y_pred=val_pred,
            y_prob=val_prob,
            analysis_label=analysis_label,
            task_name=task_name,
            feature_set_name=feature_set_name,
            model_name=model_name,
            data_split="Validation_center134_20percent",
            result_tag=result_tag,
            positive_label=positive_label,
        ),
        make_all_case_predictions_df(
            base_df=data.internal_test_df,
            y_true=data.y_internal_test,
            y_pred=internal_pred,
            y_prob=internal_prob,
            analysis_label=analysis_label,
            task_name=task_name,
            feature_set_name=feature_set_name,
            model_name=model_name,
            data_split="Internal_test_center134_20percent",
            result_tag=result_tag,
            positive_label=positive_label,
        ),
        make_all_case_predictions_df(
            base_df=data.external_df,
            y_true=data.y_external,
            y_pred=external_pred,
            y_prob=external_prob,
            analysis_label=analysis_label,
            task_name=task_name,
            feature_set_name=feature_set_name,
            model_name=model_name,
            data_split="External_test_center2",
            result_tag=result_tag,
            positive_label=positive_label,
        ),
    ], ignore_index=True)

    base_internal = base_result_row(
        analysis_label=analysis_label,
        result_tag=result_tag,
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        dataset_name="Internal_test_center134_20percent",
        data=data,
        class_a=class_a,
        class_b=class_b,
        radiomics_cols=radiomics_cols,
        morph_cols=morph_cols,
        fs_result=fs_result,
    )

    internal_row = {
        **base_internal,
        "训练+验证样本量": len(trainval_df),
        "样本量": len(data.internal_test_df),
        "最终特征筛选数据": "Train + Validation only",
        "最终模型训练数据": "Train + Validation only",
        **cv_summary_cols,
        **internal_metrics,
    }

    external_row = internal_row.copy()
    external_row.update({
        "数据集": "External_test_center2",
        "样本量": len(data.external_df),
        **external_metrics,
    })

    internal_wrong_df = make_wrong_df(
        base_df=data.internal_test_df,
        y_true=data.y_internal_test,
        y_pred=internal_pred,
        y_prob=internal_prob,
        analysis_label=analysis_label,
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        data_split="Internal_test_center134_20percent",
        result_tag=result_tag,
    )

    external_wrong_df = make_wrong_df(
        base_df=data.external_df,
        y_true=data.y_external,
        y_pred=external_pred,
        y_prob=external_prob,
        analysis_label=analysis_label,
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        data_split="External_test_center2",
        result_tag=result_tag,
    )

    wrong_df = pd.concat([internal_wrong_df, external_wrong_df], ignore_index=True)

    artifact = FinalArtifact(
        analysis_label=analysis_label,
        task_name=task_name,
        feature_set_name=feature_set_name,
        model_name=model_name,
        result_tag=result_tag,
        positive_label=positive_label,
        labels_sorted=labels_sorted,
        selected_cols=fs_result.selected_cols,
        final_model=final_model,
        x_trainval_selected=x_trainval_all[fs_result.selected_cols].copy(),
        y_trainval=y_trainval.copy(),
        trainval_df=trainval_df.copy(),
        x_internal_selected=data.x_internal_test_all[fs_result.selected_cols].copy(),
        internal_df=data.internal_test_df.copy(),
        internal_true=data.y_internal_test.copy(),
        internal_true_bin=to_binary_labels(data.y_internal_test, positive_label),
        internal_pred=internal_pred.copy(),
        internal_prob=internal_prob.copy(),
        x_external_selected=data.x_external_all[fs_result.selected_cols].copy(),
        external_df=data.external_df.copy(),
        external_true=data.y_external.copy(),
        external_true_bin=to_binary_labels(data.y_external, positive_label),
        external_pred=external_pred.copy(),
        external_prob=external_prob.copy(),
    )

    return {
        "internal_row": internal_row,
        "external_row": external_row,
        "wrong_df": wrong_df,
        "all_case_predictions_df": all_case_predictions_df,
        "artifact": artifact,
        "feature_selection_rows": feature_selection_rows,
        "used_features_rows": used_features_rows,
    }




# 6. 特征筛选


def pearson_filter_by_train(x_train_scaled, feature_names, threshold=0.90):
    if len(feature_names) <= 1:
        return list(feature_names)

    x_df = pd.DataFrame(x_train_scaled, columns=feature_names)
    corr = x_df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

    drop_cols = set()
    for col in upper.columns:
        if any(upper[col] > threshold):
            drop_cols.add(col)

    return [c for c in feature_names if c not in drop_cols]


def select_radiomics_features_train_only(x_train, y_train, radiomics_cols):
    if not radiomics_cols:
        return {"var_cols": [], "uni_cols": [], "pearson_cols": [], "lasso_cols": []}

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train[radiomics_cols])

    var_selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD_VALUE)
    x_var = var_selector.fit_transform(x_scaled)
    var_cols = np.array(radiomics_cols)[var_selector.get_support()].tolist()

    if not var_cols:
        raise ValueError("VarianceThreshold 后 radiomics 特征为空。")

    k = min(UNIVARIATE_TOP_K, len(var_cols))
    uni_selector = SelectKBest(score_func=f_classif, k=k)
    x_uni = uni_selector.fit_transform(x_var, y_train)
    uni_cols = np.array(var_cols)[uni_selector.get_support()].tolist()

    if not uni_cols:
        raise ValueError("ANOVA/f_classif 后 radiomics 特征为空。")

    x_uni_df = pd.DataFrame(x_uni, columns=uni_cols)
    pearson_cols = pearson_filter_by_train(x_uni_df.values, uni_cols, threshold=PEARSON_THRESHOLD)

    if not pearson_cols:
        raise ValueError("Pearson 去冗余后 radiomics 特征为空。")

    x_pearson = x_uni_df[pearson_cols].values

    lasso = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=LASSO_C,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        max_iter=5000,
    )
    lasso.fit(x_pearson, y_train)

    coef = np.abs(lasso.coef_).ravel()
    lasso_cols = [c for c, w in zip(pearson_cols, coef) if w > 1e-8]

    if not lasso_cols:
        lasso_cols = pearson_cols

    return {
        "var_cols": var_cols,
        "uni_cols": uni_cols,
        "pearson_cols": pearson_cols,
        "lasso_cols": lasso_cols,
    }


def select_morph_features_train_only(x_train, morph_cols):
    if not morph_cols:
        return {"pearson_cols": []}
    if len(morph_cols) <= 1:
        return {"pearson_cols": list(morph_cols)}

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x_train[morph_cols])
    selected = pearson_filter_by_train(x_scaled, morph_cols, threshold=PEARSON_THRESHOLD)

    if not selected:
        selected = list(morph_cols)

    return {"pearson_cols": selected}


def build_selection_counts(feature_set_name, valid_radio_cols, valid_morph_cols, radio_result=None, morph_result=None):
    radio_result = radio_result or {}
    morph_result = morph_result or {}

    counts = {
        "Radiomics初步可用特征数": len(valid_radio_cols),
        "Radiomics_VarianceThreshold后特征数": 0,
        "Radiomics_ANOVA后特征数": 0,
        "Radiomics_Pearson后特征数": 0,
        "Radiomics_LASSO后特征数": 0,
        "Morphology初步可用特征数": len(valid_morph_cols),
        "Morphology_Pearson后特征数": 0,
    }

    if feature_set_name in ["Radiomics_only", "Combined"]:
        counts["Radiomics_VarianceThreshold后特征数"] = len(radio_result.get("var_cols", []))
        counts["Radiomics_ANOVA后特征数"] = len(radio_result.get("uni_cols", []))
        counts["Radiomics_Pearson后特征数"] = len(radio_result.get("pearson_cols", []))
        counts["Radiomics_LASSO后特征数"] = len(radio_result.get("lasso_cols", []))

    if feature_set_name in ["Morphology_only", "Combined"]:
        counts["Morphology_Pearson后特征数"] = len(morph_result.get("pearson_cols", []))

    return counts


def select_features_by_feature_set(feature_set_name, x_train_all, y_train, valid_radio_cols, valid_morph_cols):
    selected_radio_cols = []
    selected_morph_cols = []
    radio_result = {}
    morph_result = {}

    if feature_set_name == "Radiomics_only":
        radio_result = select_radiomics_features_train_only(x_train_all, y_train, valid_radio_cols)
        selected_radio_cols = radio_result["lasso_cols"]
        selected_cols = selected_radio_cols

    elif feature_set_name == "Morphology_only":
        morph_result = select_morph_features_train_only(x_train_all, valid_morph_cols)
        selected_morph_cols = morph_result["pearson_cols"]
        selected_cols = selected_morph_cols

    elif feature_set_name == "Combined":
        radio_result = select_radiomics_features_train_only(x_train_all, y_train, valid_radio_cols)
        selected_radio_cols = radio_result["lasso_cols"]

        morph_result = select_morph_features_train_only(x_train_all, valid_morph_cols)
        selected_morph_cols = morph_result["pearson_cols"]

        selected_cols = selected_radio_cols + selected_morph_cols

    else:
        raise ValueError(f"未知特征集：{feature_set_name}")

    if not selected_cols:
        raise ValueError(f"{feature_set_name} 筛选后没有可用特征。")

    counts = build_selection_counts(
        feature_set_name=feature_set_name,
        valid_radio_cols=valid_radio_cols,
        valid_morph_cols=valid_morph_cols,
        radio_result=radio_result,
        morph_result=morph_result,
    )

    return FeatureSelectionResult(
        selected_cols=selected_cols,
        selected_radio_cols=selected_radio_cols,
        selected_morph_cols=selected_morph_cols,
        counts=counts,
        radio_result=radio_result,
        morph_result=morph_result,
    )


def append_feature_rows(rows, common, feature_type, step_name, feature_names):
    feature_names = list(feature_names or [])
    for i, feature_name in enumerate(feature_names, start=1):
        rows.append({
            **common,
            "特征类型": feature_type,
            "筛选步骤": step_name,
            "该步骤剩余特征数": len(feature_names),
            "特征序号": i,
            "特征名称": feature_name,
        })


def make_feature_selection_feature_rows(
    analysis_label,
    result_tag,
    task_name,
    feature_set_name,
    model_name,
    stage_label,
    data_source,
    fs_result,
    valid_radio_cols,
    valid_morph_cols,
    cv_fold=None,
):
    """
    保存每次筛选后剩余的特征名称。
    采用长表格式：每个筛选步骤下，每个剩余特征一行。
    """
    rows = []
    common = {
        "分析类型": analysis_label,
        "结果类型": result_tag,
        "任务": task_name,
        "特征集": feature_set_name,
        "模型": model_name,
        "阶段": stage_label,
        "数据来源": data_source,
        "CV折号": cv_fold if cv_fold is not None else "",
    }

    if feature_set_name in ["Radiomics_only", "Combined"]:
        append_feature_rows(rows, common, "Radiomics", "00_initial_valid_after_missing_filter", valid_radio_cols)
        append_feature_rows(rows, common, "Radiomics", "01_after_VarianceThreshold", fs_result.radio_result.get("var_cols", []))
        append_feature_rows(rows, common, "Radiomics", "02_after_ANOVA_f_classif", fs_result.radio_result.get("uni_cols", []))
        append_feature_rows(rows, common, "Radiomics", "03_after_Pearson_filter", fs_result.radio_result.get("pearson_cols", []))
        append_feature_rows(rows, common, "Radiomics", "04_after_LASSO_final", fs_result.radio_result.get("lasso_cols", []))

    if feature_set_name in ["Morphology_only", "Combined"]:
        append_feature_rows(rows, common, "Morphology", "00_initial_valid_after_missing_filter", valid_morph_cols)
        append_feature_rows(rows, common, "Morphology", "01_after_Pearson_filter_final", fs_result.morph_result.get("pearson_cols", []))

    append_feature_rows(rows, common, "All", "final_selected_for_model", fs_result.selected_cols)
    return rows


def make_used_features_rows(
    analysis_label,
    result_tag,
    task_name,
    feature_set_name,
    model_name,
    fs_result,
    radio_cols,
    morph_cols,
):
    rows = []
    for i, feature_name in enumerate(fs_result.selected_cols, start=1):
        rows.append({
            "分析类型": analysis_label,
            "结果类型": result_tag,
            "任务": task_name,
            "特征集": feature_set_name,
            "模型": model_name,
            "最终模型训练数据": "Train + Validation only",
            "特征序号": i,
            "特征类型": feature_family(feature_name, radio_cols, morph_cols),
            "特征名称": feature_name,
            "总特征数": len(fs_result.selected_cols),
        })
    return rows




# 4. 数据读取与特征识别


def is_diagnostics_feature(col):
    return "diagnostics" in str(col).lower()


def is_radiomics_shape_feature(col):
    """
    判断是否为 PyRadiomics shape / shape2D 类特征。
    典型格式：
    - original_shape_MeshVolume
    - original_shape_SurfaceArea
    - original_shape_Sphericity
    - original_shape2D_Perimeter
    """
    c = str(col).lower()
    return re.search(r"(^|[_-])shape(?:2d)?([_-]|$)", c) is not None


def is_radiomics_feature(col):
    c = str(col).lower()
    radiomics_keywords = [
        "original_", "wavelet", "log-sigma", "log_", "square", "squareroot",
        "logarithm", "exponential", "gradient", "lbp", "glcm", "glrlm",
        "glszm", "gldm", "ngtdm", "firstorder", "shape", "diagnostics",
    ]
    return any(k in c for k in radiomics_keywords)


def is_exclude_col(col):
    exclude_keywords = [
        "标签", "label", "序号", "编号", "case", "patient",
        "肢体", "side", "center", "中心", "对应中心",
        "name", "file", "path", "路径",
    ]
    c = str(col).lower()
    return any(k.lower() in c for k in exclude_keywords)


def read_and_validate_data(excel_file):
    logging.info("读取数据：%s", excel_file)
    df = pd.read_excel(excel_file)

    logging.info("数据读取完成：样本数=%d，列数=%d", len(df), df.shape[1])

    if LABEL_COL not in df.columns:
        raise ValueError(f"没有找到标签列：{LABEL_COL}")
    if CENTER_COL not in df.columns:
        raise ValueError(f"没有找到中心列：{CENTER_COL}")

    df[LABEL_COL] = pd.to_numeric(df[LABEL_COL], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[LABEL_COL]).copy()
    df[LABEL_COL] = df[LABEL_COL].astype(int)
    logging.info("标签缺失样本删除：%d -> %d", before, len(df))

    df[CENTER_COL] = df[CENTER_COL].astype(str).str.strip()

    valid_centers = set(df[CENTER_COL].unique())
    if INTERNAL_CENTER not in valid_centers:
        raise ValueError(f"没有找到内部中心：{INTERNAL_CENTER}。当前中心包括：{sorted(valid_centers)}")
    if EXTERNAL_CENTER not in valid_centers:
        raise ValueError(f"没有找到外部中心：{EXTERNAL_CENTER}。当前中心包括：{sorted(valid_centers)}")

    logging.info("标签分布：\n%s", df[LABEL_COL].value_counts().sort_index().to_string())
    logging.info("中心分布：\n%s", df[CENTER_COL].value_counts().to_string())
    logging.info("各中心标签分布：\n%s", pd.crosstab(df[CENTER_COL], df[LABEL_COL]).to_string())

    return df


def identify_feature_columns(df):
    existing_morph_cols = [c for c in MORPH_FEATURES if c in df.columns]
    missing_morph_cols = [c for c in MORPH_FEATURES if c not in df.columns]

    logging.info(
        "手动 morphology 特征：应有=%d，实际存在=%d，不存在=%d",
        len(MORPH_FEATURES), len(existing_morph_cols), len(missing_morph_cols)
    )
    if missing_morph_cols:
        logging.warning("不存在的 morphology 特征：%s", missing_morph_cols)

    candidate_cols = []
    for col in df.columns:
        if is_exclude_col(col):
            continue
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.notna().sum() > 0:
            candidate_cols.append(col)

    radiomics_all_detected_cols = [
        c for c in candidate_cols
        if is_radiomics_feature(c) and c not in MORPH_FEATURES
    ]

    radiomics_shape_cols = [c for c in radiomics_all_detected_cols if is_radiomics_shape_feature(c)]
    radiomics_no_shape_cols = [c for c in radiomics_all_detected_cols if not is_radiomics_shape_feature(c)]

    if REMOVE_RADIOMICS_SHAPE_FEATURES:
        radiomics_cols = radiomics_no_shape_cols
    else:
        radiomics_cols = radiomics_all_detected_cols

    morph_cols = existing_morph_cols

    diagnostics_cols = [c for c in radiomics_cols if is_diagnostics_feature(c)]
    radiomics_no_diagnostics_cols = [c for c in radiomics_cols if not is_diagnostics_feature(c)]

    logging.info(
        "Radiomics 自动识别总数=%d；Radiomics shape/shape2D 数=%d；最终用于分析的 Radiomics=%d；Morphology=%d；Combined=%d",
        len(radiomics_all_detected_cols),
        len(radiomics_shape_cols),
        len(radiomics_cols),
        len(morph_cols),
        len(radiomics_cols) + len(morph_cols)
    )
    if REMOVE_RADIOMICS_SHAPE_FEATURES and radiomics_shape_cols:
        logging.info("已从 Radiomics 中删除 shape/shape2D 特征，例如：%s", radiomics_shape_cols[:20])

    logging.info(
        "diagnostics 特征数量：%d；删除 diagnostics 后 Radiomics=%d，Combined=%d",
        len(diagnostics_cols),
        len(radiomics_no_diagnostics_cols),
        len(radiomics_no_diagnostics_cols) + len(morph_cols)
    )

    return (
        radiomics_cols,
        morph_cols,
        diagnostics_cols,
        radiomics_no_diagnostics_cols,
        radiomics_shape_cols,
        radiomics_all_detected_cols,
    )




# 5. 数据划分与预处理


def row_level_stratified_split_indices(df_part, label_col, train_ratio, val_ratio, test_ratio, random_state):
    all_idx = np.arange(len(df_part))
    y = df_part[label_col].astype(int).values

    train_idx, temp_idx = train_test_split(
        all_idx,
        test_size=val_ratio + test_ratio,
        stratify=y,
        random_state=random_state,
    )

    temp_y = y[temp_idx]
    val_size_in_temp = val_ratio / (val_ratio + test_ratio)

    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=1 - val_size_in_temp,
        stratify=temp_y,
        random_state=random_state,
    )

    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


def stratified_group_split_indices(df_part, label_col, group_col, train_ratio, val_ratio, test_ratio, random_state=42):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    df_part = df_part.reset_index(drop=True).copy()

    use_group = (
        USE_GROUP_SPLIT
        and group_col in df_part.columns
        and df_part[group_col].notna().all()
        and df_part[group_col].astype(str).nunique() >= 10
    )

    if not use_group:
        train_idx, val_idx, test_idx = row_level_stratified_split_indices(
            df_part, label_col, train_ratio, val_ratio, test_ratio, random_state
        )
        return train_idx, val_idx, test_idx, "Row-level stratified split"

    try:
        group_df = (
            df_part
            .groupby(group_col)
            .agg(
                group_label=(label_col, lambda x: int(pd.Series(x).mode().iloc[0])),
                n=(label_col, "size"),
            )
            .reset_index()
        )

        groups = group_df[group_col].astype(str).values
        group_labels = group_df["group_label"].astype(int).values

        g_train, g_temp, _, y_g_temp = train_test_split(
            groups,
            group_labels,
            test_size=val_ratio + test_ratio,
            stratify=group_labels,
            random_state=random_state,
        )

        val_size_in_temp = val_ratio / (val_ratio + test_ratio)
        g_val, g_test = train_test_split(
            g_temp,
            test_size=1 - val_size_in_temp,
            stratify=y_g_temp,
            random_state=random_state,
        )

        train_idx = df_part.index[df_part[group_col].astype(str).isin(g_train)].to_numpy()
        val_idx = df_part.index[df_part[group_col].astype(str).isin(g_val)].to_numpy()
        test_idx = df_part.index[df_part[group_col].astype(str).isin(g_test)].to_numpy()

        return train_idx, val_idx, test_idx, f"Group-level stratified split by {group_col}"

    except Exception as e:
        logging.warning("分组划分失败，自动退回普通分层划分。原因：%s", e)
        train_idx, val_idx, test_idx = row_level_stratified_split_indices(
            df_part, label_col, train_ratio, val_ratio, test_ratio, random_state
        )
        return train_idx, val_idx, test_idx, "Fallback row-level stratified split"


def remove_missing_and_non_numeric_by_train(train_df, apply_dfs, feature_cols):
    valid_cols = []
    dropped_cols = []

    for col in feature_cols:
        if col not in train_df.columns:
            dropped_cols.append(col)
            continue

        s = pd.to_numeric(train_df[col], errors="coerce")
        if s.isna().any():
            dropped_cols.append(col)
        else:
            valid_cols.append(col)

    if valid_cols:
        x_train = train_df[valid_cols].apply(pd.to_numeric, errors="coerce")
    else:
        x_train = pd.DataFrame(index=train_df.index)

    nan_cols = x_train.columns[x_train.isna().any()].tolist()
    if nan_cols:
        x_train = x_train.drop(columns=nan_cols)
        valid_cols = x_train.columns.tolist()
        dropped_cols.extend(nan_cols)

    x_apply_list = []
    for name, d in apply_dfs:
        if not valid_cols:
            x_apply = pd.DataFrame(index=d.index)
        else:
            x_apply = d[valid_cols].apply(pd.to_numeric, errors="coerce")
            if x_apply.isna().any().any():
                bad_cols = x_apply.columns[x_apply.isna().any()].tolist()
                raise ValueError(f"{name} 中存在训练集保留特征的缺失/非数值：{bad_cols[:20]}")
        x_apply_list.append(x_apply)

    return x_train, x_apply_list, valid_cols, dropped_cols


def prepare_task_data(df, class_a, class_b, radiomics_cols, morph_cols):
    task_name = task_name_from_classes(class_a, class_b)
    logging.info("准备任务数据：%s", task_name)

    task_df = df[df[LABEL_COL].isin([class_a, class_b])].copy().reset_index(drop=True)
    internal_df = task_df[task_df[CENTER_COL] == INTERNAL_CENTER].copy().reset_index(drop=True)
    external_df = task_df[task_df[CENTER_COL] == EXTERNAL_CENTER].copy().reset_index(drop=True)

    if len(internal_df) == 0:
        raise ValueError(f"内部中心 {INTERNAL_CENTER} 样本为空。")
    if len(external_df) == 0:
        raise ValueError(f"外部中心 {EXTERNAL_CENTER} 样本为空。")
    if internal_df[LABEL_COL].nunique() < 2:
        raise ValueError(f"内部中心 {INTERNAL_CENTER} 在任务 {task_name} 中不足两个类别。")
    if external_df[LABEL_COL].nunique() < 2:
        logging.warning("外部中心 %s 在任务 %s 中不足两个类别，AUC 可能为空。", EXTERNAL_CENTER, task_name)

    train_idx, val_idx, test_idx, split_method = stratified_group_split_indices(
        df_part=internal_df,
        label_col=LABEL_COL,
        group_col=ID_COL,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=INTERNAL_TEST_RATIO,
        random_state=RANDOM_STATE,
    )

    train_df = internal_df.iloc[train_idx].copy().reset_index(drop=True)
    val_df = internal_df.iloc[val_idx].copy().reset_index(drop=True)
    internal_test_df = internal_df.iloc[test_idx].copy().reset_index(drop=True)

    x_radio_train, radio_apply_list, valid_radio_cols, dropped_radio_cols = remove_missing_and_non_numeric_by_train(
        train_df=train_df,
        apply_dfs=[("val_df", val_df), ("internal_test_df", internal_test_df), ("external_df", external_df)],
        feature_cols=radiomics_cols,
    )

    x_morph_train, morph_apply_list, valid_morph_cols, dropped_morph_cols = remove_missing_and_non_numeric_by_train(
        train_df=train_df,
        apply_dfs=[("val_df", val_df), ("internal_test_df", internal_test_df), ("external_df", external_df)],
        feature_cols=morph_cols,
    )

    x_radio_val, x_radio_internal_test, x_radio_external = radio_apply_list
    x_morph_val, x_morph_internal_test, x_morph_external = morph_apply_list

    x_train_all = pd.concat([x_radio_train, x_morph_train], axis=1)
    x_val_all = pd.concat([x_radio_val, x_morph_val], axis=1)
    x_internal_test_all = pd.concat([x_radio_internal_test, x_morph_internal_test], axis=1)
    x_external_all = pd.concat([x_radio_external, x_morph_external], axis=1)

    data = TaskData(
        task_df=task_df,
        internal_df=internal_df,
        external_df=external_df,
        train_df=train_df,
        val_df=val_df,
        internal_test_df=internal_test_df,
        x_train_all=x_train_all,
        x_val_all=x_val_all,
        x_internal_test_all=x_internal_test_all,
        x_external_all=x_external_all,
        y_train=train_df[LABEL_COL].astype(int).values,
        y_val=val_df[LABEL_COL].astype(int).values,
        y_internal_test=internal_test_df[LABEL_COL].astype(int).values,
        y_external=external_df[LABEL_COL].astype(int).values,
        valid_radio_cols=valid_radio_cols,
        valid_morph_cols=valid_morph_cols,
        dropped_radio_cols=dropped_radio_cols,
        dropped_morph_cols=dropped_morph_cols,
        split_method=split_method,
    )

    logging.info(
        "%s 划分完成：%s | train=%d, val=%d, internal_test=%d, external=%d | valid_radio_no_shape=%d, valid_morph=%d",
        task_name,
        split_method,
        len(train_df),
        len(val_df),
        len(internal_test_df),
        len(external_df),
        len(valid_radio_cols),
        len(valid_morph_cols),
    )
    logging.info(
        "%s 标签计数 | Train=%s | Validation=%s | Internal_test=%s | External_test=%s",
        task_name,
        format_label_counts(label_counts_for_df(train_df, sorted([class_a, class_b]))),
        format_label_counts(label_counts_for_df(val_df, sorted([class_a, class_b]))),
        format_label_counts(label_counts_for_df(internal_test_df, sorted([class_a, class_b]))),
        format_label_counts(label_counts_for_df(external_df, sorted([class_a, class_b]))),
    )

    return data



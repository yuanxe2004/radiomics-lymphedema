
# 12. Calibration / DCA / SHAP 图形


from .config import *
from .utils import (
    PUBLICATION_LINESTYLES,
    SCI_BOX_COLORS,
    SCI_BOX_POINT_COLOR,
    SCI_SHAP_CMAP,
    SHAP_NEGATIVE_COLOR,
    SHAP_POSITIVE_COLOR,
    apply_axis_style,
    ensure_directories,
    feature_family,
    get_feature_set_color,
    safe_filename,
    save_or_show_plot,
    short_feature_set_name,
)


def calibration_curve_uniform_bins(y_true_bin, y_prob, n_bins=10):
    """
    Calibration curve with uniform binning.

    返回每个非空分箱：
    - mean_pred: 分箱内平均预测概率；
    - observed_rate: 分箱内真实阳性比例；
    - n: 分箱样本数。

    说明：
    - uniform binning 将 [0, 1] 按等宽区间划分；
    - 空分箱不绘制；
    - 不绘制 95% CI 误差线。
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    finite_mask = np.isfinite(y_prob) & np.isfinite(y_true_bin)
    y_true_bin = y_true_bin[finite_mask]
    y_prob = y_prob[finite_mask]

    if len(y_true_bin) < 2:
        return np.array([]), np.array([]), np.array([])

    y_prob = np.clip(y_prob, 0.0, 1.0)
    effective_bins = max(1, int(n_bins))

    bin_edges = np.linspace(0.0, 1.0, effective_bins + 1)
    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=False)

    df_cal = pd.DataFrame({
        "y_true": y_true_bin,
        "y_prob": y_prob,
        "bin": bin_ids,
    })

    grouped = (
        df_cal
        .groupby("bin", dropna=True)
        .agg(
            mean_pred=("y_prob", "mean"),
            observed_rate=("y_true", "mean"),
            n=("y_true", "size"),
        )
        .reset_index(drop=True)
        .sort_values("mean_pred")
    )

    return grouped["mean_pred"].values, grouped["observed_rate"].values, grouped["n"].values


def calibration_curve_quantile_bins(y_true_bin, y_prob, n_bins=10):
    """
    Calibration curve with quantile binning.

    返回每个非空分箱：
    - mean_pred: 分箱内平均预测概率；
    - observed_rate: 分箱内真实阳性比例；
    - n: 分箱样本数。

    说明：
    - quantile binning 按预测概率分位数划分，使每个非空分箱样本数尽量接近；
    - 若存在大量相同预测概率，实际可用分箱数可能少于 n_bins；
    - 不绘制 95% CI 误差线。
    """
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    finite_mask = np.isfinite(y_prob) & np.isfinite(y_true_bin)
    y_true_bin = y_true_bin[finite_mask]
    y_prob = y_prob[finite_mask]

    if len(y_true_bin) < 2:
        return np.array([]), np.array([]), np.array([])

    y_prob = np.clip(y_prob, 0.0, 1.0)
    effective_bins = max(1, min(int(n_bins), len(y_prob)))

    df_cal = pd.DataFrame({
        "y_true": y_true_bin,
        "y_prob": y_prob,
    })

    try:
        df_cal["bin"] = pd.qcut(
            df_cal["y_prob"],
            q=effective_bins,
            labels=False,
            duplicates="drop",
        )
    except Exception:
        return calibration_curve_uniform_bins(y_true_bin, y_prob, n_bins=n_bins)

    grouped = (
        df_cal
        .dropna(subset=["bin"])
        .groupby("bin", dropna=True)
        .agg(
            mean_pred=("y_prob", "mean"),
            observed_rate=("y_true", "mean"),
            n=("y_true", "size"),
        )
        .reset_index(drop=True)
        .sort_values("mean_pred")
    )

    return grouped["mean_pred"].values, grouped["observed_rate"].values, grouped["n"].values


def plot_calibration_curve(curve_specs, title, save_path, n_bins=10):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    for i, spec in enumerate(curve_specs):
        x, y_point, counts = calibration_curve_quantile_bins(
            y_true_bin=spec["y_true_bin"],
            y_prob=spec["y_prob"],
            n_bins=n_bins,
        )
        if len(x) == 0:
            logging.warning("跳过 Calibration 曲线 %s，原因：可用点为空。", spec.get("label", ""))
            continue

        color = get_feature_set_color(spec.get("label", ""), i)
        linestyle = PUBLICATION_LINESTYLES[i % len(PUBLICATION_LINESTYLES)]
        mask = ~(np.isnan(x) | np.isnan(y_point))

        counts_text = ",".join([str(int(v)) for v in counts])
        logging.info("Calibration quantile bin counts | %s | counts=%s", spec["label"], counts_text)

        ax.plot(
            x[mask],
            y_point[mask],
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            marker="o",
            markersize=4,
            label=spec["label"],
        )

    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.1, label="Perfect calibration")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed probability")
    ax.set_title(title, pad=8)
    apply_axis_style(ax)
    ax.legend(frameon=False, loc="upper left", handlelength=2.5)
    save_or_show_plot(save_path)


def plot_roc_curve(curve_specs, title, save_path):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    plotted = False
    for i, spec in enumerate(curve_specs):
        y_true_bin = np.asarray(spec["y_true_bin"]).astype(int)
        y_prob = np.asarray(spec["y_prob"], dtype=float)

        finite_mask = np.isfinite(y_prob)
        y_true_bin = y_true_bin[finite_mask]
        y_prob = y_prob[finite_mask]

        if len(y_true_bin) < 2 or len(np.unique(y_true_bin)) < 2:
            logging.warning("跳过 ROC 曲线 %s，原因：样本为空或只有一个类别。", spec.get("label", ""))
            continue

        fpr, tpr, _ = roc_curve(y_true_bin, y_prob)
        auc_value = roc_auc_score(y_true_bin, y_prob)

        color = get_feature_set_color(spec.get("label", ""), i)
        linestyle = "-"
        ax.plot(
            fpr,
            tpr,
            color=color,
            linestyle=linestyle,
            linewidth=2.0,
            label=f"{spec['label']} (AUC={auc_value:.3f})",
        )
        plotted = True

    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.1, label="Chance")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("1 - Specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_title(title, pad=8)
    apply_axis_style(ax)
    ax.legend(frameon=False, loc="lower right", handlelength=2.5)

    if plotted:
        save_or_show_plot(save_path)
    else:
        plt.close()
        logging.warning("ROC 图未保存：%s，原因：没有可绘制曲线。", save_path)


def decision_curve_values(y_true_bin, y_prob, thresholds):
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    n = len(y_true_bin)

    net_benefit = []
    for threshold in thresholds:
        pred = (y_prob >= threshold).astype(int)
        tp = ((pred == 1) & (y_true_bin == 1)).sum()
        fp = ((pred == 1) & (y_true_bin == 0)).sum()
        nb = tp / n - fp / n * (threshold / (1 - threshold))
        net_benefit.append(nb)

    return np.asarray(net_benefit, dtype=float)


def plot_dca_curve(curve_specs, title, save_path, thresholds=DCA_THRESHOLDS):
    fig, ax = plt.subplots(figsize=(5.2, 5.0))

    for i, spec in enumerate(curve_specs):
        y = decision_curve_values(spec["y_true_bin"], spec["y_prob"], thresholds)

        color = get_feature_set_color(spec.get("label", ""), i)
        linestyle = "-"
        ax.plot(thresholds, y, color=color, linestyle=linestyle, linewidth=2.0, label=spec["label"])

    ax.axhline(0, color="0.35", linestyle="--", linewidth=1.1, label="Treat none")

    prevalence_records = []
    for spec in curve_specs:
        y_true_bin = np.asarray(spec["y_true_bin"]).astype(int)
        if len(y_true_bin) == 0:
            continue
        prevalence_records.append((spec["label"], float(np.mean(y_true_bin))))

    unique_prevalences = []
    for label, prevalence in prevalence_records:
        if not any(abs(prevalence - p) < 1e-12 for _, p in unique_prevalences):
            unique_prevalences.append((label, prevalence))

    if len(unique_prevalences) == 1:
        prevalence = unique_prevalences[0][1]
        treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
        ax.plot(thresholds, treat_all, color="0.35", linestyle="-.", linewidth=1.1, label="Treat all")
    elif len(unique_prevalences) > 1:
        for label, prevalence in unique_prevalences:
            treat_all = prevalence - (1 - prevalence) * thresholds / (1 - thresholds)
            ax.plot(
                thresholds,
                treat_all,
                color="0.35",
                linestyle=":",
                linewidth=1.0,
                label=f"Treat all ({label})",
            )

    ax.set_xlim(float(np.min(thresholds)), float(np.max(thresholds)))
    ax.set_ylim(bottom=DCA_YMIN)
    ax.set_xlabel("Threshold probability")
    ax.set_ylabel("Net benefit")
    ax.set_title(title, pad=8)
    apply_axis_style(ax)
    ax.legend(frameon=False, loc="best", handlelength=2.5)
    save_or_show_plot(save_path)


def make_curve_spec(label, true_bin, prob):
    return {
        "label": label,
        "y_true_bin": np.asarray(true_bin).astype(int),
        "y_prob": np.asarray(prob, dtype=float),
    }


def extract_shap_matrix(shap_values):
    if hasattr(shap_values, "values"):
        values = shap_values.values
    else:
        values = shap_values

    if isinstance(values, list):
        values = values[1] if len(values) >= 2 else values[0]

    values = np.asarray(values)

    if values.ndim == 3:
        if values.shape[-1] == 2:
            values = values[:, :, 1]
        elif values.shape[0] == 2:
            values = values[1, :, :]

    return np.asarray(values, dtype=float)


def compute_shap_explanation(model, model_name, x_df, positive_label, random_state=42):
    """
    计算 SHAP 矩阵、用于绘图的样本矩阵和特征重要性排序。

    返回：
    - importance_df：按 mean(|SHAP|) 从高到低排序，包含重要性排名、正负向平均贡献等；
    - shap_mat：shape=(n_samples, n_features) 的 SHAP 值矩阵；
    - x_exp：与 shap_mat 行对应的解释样本特征值。
    """
    if not HAS_SHAP:
        raise RuntimeError("未检测到 shap，请先安装：pip install shap")

    x_df = x_df.copy()
    feature_names = x_df.columns.tolist()
    if len(x_df) == 0:
        raise ValueError("SHAP 输入数据为空。")

    n_bg = min(SHAP_MAX_BACKGROUND, len(x_df))
    n_exp = min(SHAP_MAX_EXPLAIN, len(x_df))
    x_bg = x_df.sample(n=n_bg, random_state=random_state)
    x_exp = x_df.sample(n=n_exp, random_state=random_state)

    if model_name in ["RandomForest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_exp)
        shap_mat = extract_shap_matrix(shap_values)

    elif model_name == "LogisticRegression" and isinstance(model, Pipeline):
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        x_bg_scaled = scaler.transform(x_bg)
        x_exp_scaled = scaler.transform(x_exp)
        explainer = shap.LinearExplainer(clf, x_bg_scaled)
        shap_values = explainer(x_exp_scaled)
        shap_mat = extract_shap_matrix(shap_values)

    else:
        if model_name == "XGBoost":
            pos_index = 1
        else:
            if isinstance(model, Pipeline):
                class_list = list(model.named_steps["clf"].classes_)
            else:
                class_list = list(model.classes_)
            pos_index = class_list.index(positive_label)

        def predict_pos_prob(data_array):
            data_df = pd.DataFrame(np.asarray(data_array), columns=feature_names)
            return model.predict_proba(data_df)[:, pos_index]

        explainer = shap.KernelExplainer(predict_pos_prob, x_bg)
        shap_values = explainer.shap_values(x_exp, nsamples=SHAP_KERNEL_NSAMPLES)
        shap_mat = extract_shap_matrix(shap_values)

    shap_mat = np.asarray(shap_mat, dtype=float)
    if shap_mat.ndim == 1:
        shap_mat = shap_mat.reshape(1, -1)

    n_rows = min(len(x_exp), shap_mat.shape[0])
    x_exp = x_exp.iloc[:n_rows].copy().reset_index(drop=True)
    shap_mat = shap_mat[:n_rows, :]

    if shap_mat.shape[1] != len(feature_names):
        raise ValueError(f"SHAP 矩阵列数与特征数不一致：{shap_mat.shape[1]} vs {len(feature_names)}")

    mean_abs = np.abs(shap_mat).mean(axis=0)
    mean_signed = shap_mat.mean(axis=0)
    mean_positive = np.where(shap_mat > 0, shap_mat, 0.0).mean(axis=0)
    mean_negative = np.where(shap_mat < 0, shap_mat, 0.0).mean(axis=0)
    positive_frequency = (shap_mat > 0).mean(axis=0)
    negative_frequency = (shap_mat < 0).mean(axis=0)

    importance_df = (
        pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs,
            "mean_signed_shap": mean_signed,
            "mean_positive_shap": mean_positive,
            "mean_negative_shap": mean_negative,
            "positive_frequency": positive_frequency,
            "negative_frequency": negative_frequency,
            "feature_index": np.arange(len(feature_names), dtype=int),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    importance_df["importance_rank"] = np.arange(1, len(importance_df) + 1, dtype=int)

    return {
        "importance_df": importance_df,
        "shap_mat": shap_mat,
        "x_exp": x_exp,
    }


def compute_shap_importance(model, model_name, x_df, positive_label, random_state=42):
    return compute_shap_explanation(
        model=model,
        model_name=model_name,
        x_df=x_df,
        positive_label=positive_label,
        random_state=random_state,
    )["importance_df"]


def _normalize_feature_values_for_color(values):
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return np.full_like(values, 0.5, dtype=float)

    v_low, v_high = np.percentile(finite, [5, 95])
    if not np.isfinite(v_low) or not np.isfinite(v_high) or v_high <= v_low:
        v_low, v_high = float(np.min(finite)), float(np.max(finite))

    if v_high <= v_low:
        return np.full_like(values, 0.5, dtype=float)

    return np.clip((values - v_low) / (v_high - v_low), 0.0, 1.0)


def plot_shap_top10(model, model_name, x_df, positive_label, task_name, feature_set_name, save_path):
    logging.info("绘制 SHAP Top10：%s | %s | %s", task_name, feature_set_name, model_name)

    shap_result = compute_shap_explanation(
        model=model,
        model_name=model_name,
        x_df=x_df,
        positive_label=positive_label,
        random_state=RANDOM_STATE,
    )
    importance_df = shap_result["importance_df"]
    shap_mat = shap_result["shap_mat"]
    x_exp = shap_result["x_exp"]

    top10 = importance_df.head(10).copy()
    if top10.empty:
        raise ValueError("SHAP Top10 为空，无法绘图。")

    fig_height = max(4.8, 0.48 * len(top10) + 1.5)
    fig, ax = plt.subplots(figsize=(7.4, fig_height))
    rng = np.random.default_rng(RANDOM_STATE)

    for y_pos, (_, row) in enumerate(top10.iterrows()):
        feature_idx = int(row["feature_index"])
        shap_values = shap_mat[:, feature_idx]
        feature_values = pd.to_numeric(x_exp.iloc[:, feature_idx], errors="coerce").values
        color_values = _normalize_feature_values_for_color(feature_values)
        jitter = rng.normal(loc=0.0, scale=0.055, size=len(shap_values))

        ax.scatter(
            shap_values,
            np.full(len(shap_values), y_pos) + jitter,
            c=color_values,
            cmap=SCI_SHAP_CMAP,
            vmin=0.0,
            vmax=1.0,
            s=18,
            alpha=0.82,
            linewidths=0,
            rasterized=True,
        )

    ax.axvline(0, color="0.20", linestyle="-", linewidth=0.9)
    ax.set_yticks(np.arange(len(top10)))
    ax.set_yticklabels([
        f"{row.feature}"
        for row in top10.itertuples(index=False)
    ])
    ax.set_ylim(len(top10) - 0.5, -0.5)
    ax.set_xlabel("SHAP value (positive = higher predicted probability)")
    ax.set_ylabel("Features")
    ax.set_title(
        f"SHAP summary | {task_name} | {short_feature_set_name(feature_set_name)}",
        pad=8,
    )
    apply_axis_style(ax)
    ax.grid(False)

    sm = matplotlib.cm.ScalarMappable(
        cmap=SCI_SHAP_CMAP,
        norm=matplotlib.colors.Normalize(vmin=0.0, vmax=1.0),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.045)
    cbar.set_label("Feature value", rotation=90)
    cbar.set_ticks([0.0, 1.0])
    cbar.set_ticklabels(["Low", "High"])

    save_or_show_plot(save_path)

    return top10


def make_feature_set_curve_specs(feature_artifacts, split_name):
    """
    生成同一数据集下三种特征集的曲线规格。
    split_name 支持：internal / external。
    """
    specs = []
    for fs_name in FEATURE_SETS:
        if fs_name not in feature_artifacts:
            continue
        art = feature_artifacts[fs_name]
        label = short_feature_set_name(fs_name)
        if split_name == "internal":
            specs.append(make_curve_spec(label, art.internal_true_bin, art.internal_prob))
        elif split_name == "external":
            specs.append(make_curve_spec(label, art.external_true_bin, art.external_prob))
        else:
            raise ValueError(f"未知 split_name：{split_name}")
    return specs


def generate_plots(global_artifacts, feature_set_artifacts):
    """
    图形输出：
    1. 每个任务的全局最优模型 SHAP Top10；
    2. 每个任务、每个数据集分别绘制 ROC 曲线：R、M、M+R 放在同一张图；
    3. 每个任务、每个数据集分别绘制 DCA 曲线：R、M、M+R 放在同一张图；
    4. 每个任务保留一张 M+R 的 Internal/External 校准曲线，并新增 External test 中 R、M、M+R 三特征集同图。
    """
    if not global_artifacts and not feature_set_artifacts:
        return

    if not HAS_SHAP:
        logging.warning("未安装 shap，跳过 SHAP Top10 图。安装方法：pip install shap")

    for task_name, art in global_artifacts.items():
        if not HAS_SHAP:
            continue
        filename = f"SHAP_top10_global_best_{safe_filename(task_name)}_{safe_filename(art.feature_set_name)}_{safe_filename(art.model_name)}.png"
        save_path = os.path.join(SHAP_DIR, filename)
        try:
            plot_shap_top10(
                model=art.final_model,
                model_name=art.model_name,
                x_df=art.x_trainval_selected,
                positive_label=art.positive_label,
                task_name=task_name,
                feature_set_name=art.feature_set_name,
                save_path=save_path,
            )
            logging.info("SHAP Top10 图已保存：%s", save_path)
        except Exception as e:
            logging.warning("SHAP 绘制失败：%s | %s | %s，原因：%s", task_name, art.feature_set_name, art.model_name, e)

    for task_name, fs_artifacts in feature_set_artifacts.items():
        internal_specs = make_feature_set_curve_specs(fs_artifacts, split_name="internal")
        external_specs = make_feature_set_curve_specs(fs_artifacts, split_name="external")

        try:
            roc_path = os.path.join(ROC_DIR, f"ROC_FeatureSets_Internal_{safe_filename(task_name)}.png")
            plot_roc_curve(
                internal_specs,
                f"ROC curve | {task_name} | Internal test",
                roc_path,
            )
            logging.info("ROC Internal 图已保存：%s", roc_path)
        except Exception as e:
            logging.warning("ROC Internal 绘制失败：%s，原因：%s", task_name, e)

        try:
            roc_path = os.path.join(ROC_DIR, f"ROC_FeatureSets_External_{safe_filename(task_name)}.png")
            plot_roc_curve(
                external_specs,
                f"ROC curve | {task_name} | External test",
                roc_path,
            )
            logging.info("ROC External 图已保存：%s", roc_path)
        except Exception as e:
            logging.warning("ROC External 绘制失败：%s，原因：%s", task_name, e)

        try:
            dca_path = os.path.join(DCA_DIR, f"DCA_FeatureSets_Internal_{safe_filename(task_name)}.png")
            plot_dca_curve(
                internal_specs,
                f"Decision curve analysis | {task_name} | Internal test",
                dca_path,
                thresholds=DCA_THRESHOLDS,
            )
            logging.info("DCA Internal 图已保存：%s", dca_path)
        except Exception as e:
            logging.warning("DCA Internal 绘制失败：%s，原因：%s", task_name, e)

        try:
            dca_path = os.path.join(DCA_DIR, f"DCA_FeatureSets_External_{safe_filename(task_name)}.png")
            plot_dca_curve(
                external_specs,
                f"Decision curve analysis | {task_name} | External test",
                dca_path,
                thresholds=DCA_THRESHOLDS,
            )
            logging.info("DCA External 图已保存：%s", dca_path)
        except Exception as e:
            logging.warning("DCA External 绘制失败：%s，原因：%s", task_name, e)

        try:
            cal_external_path = os.path.join(CAL_DIR, f"Calibration_FeatureSets_External_{safe_filename(task_name)}.png")
            plot_calibration_curve(
                external_specs,
                f"Calibration curve | {task_name} | External test",
                cal_external_path,
                n_bins=CALIBRATION_BINS,
            )
            logging.info("Calibration External 三特征集图已保存：%s", cal_external_path)
        except Exception as e:
            logging.warning("Calibration External 三特征集图绘制失败：%s，原因：%s", task_name, e)

        if "Combined" not in fs_artifacts:
            logging.warning("%s 没有 Combined 特征集 artifact，跳过校准曲线。", task_name)
            continue

        art = fs_artifacts["Combined"]

        try:
            cal_path = os.path.join(CAL_DIR, f"Calibration_Combined_Internal_External_{safe_filename(task_name)}.png")
            plot_calibration_curve(
                [
                    make_curve_spec("Internal M+R", art.internal_true_bin, art.internal_prob),
                    make_curve_spec("External M+R", art.external_true_bin, art.external_prob),
                ],
                f"Calibration curve | {task_name} | M+R",
                cal_path,
                n_bins=CALIBRATION_BINS,
            )
            logging.info("Calibration Internal+External 合并图已保存：%s", cal_path)
        except Exception as e:
            logging.warning("Calibration Internal+External 合并图绘制失败：%s，原因：%s", task_name, e)





# 13. 新增模块：Top feature distributions + Wrong-case examples


def save_png_and_pdf(fig, base_path_without_ext):
    """
    同一张图同时保存 PNG 和 PDF，均为 600 dpi。
    base_path_without_ext 不需要扩展名。
    """
    fig.tight_layout()
    png_path = f"{base_path_without_ext}.png"
    pdf_path = f"{base_path_without_ext}.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=600, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close(fig)
    return png_path, pdf_path


def stage_label_text(label_value):
    mapping = {
        0: "Stage 0",
        1: "Stage I",
        2: "Stage II",
        3: "Stage III",
    }
    try:
        return mapping.get(int(label_value), f"Stage {label_value}")
    except Exception:
        return f"Stage {label_value}"


def normality_pvalue(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return np.nan, False
    if len(values) > 5000:
        # Shapiro-Wilk 对超大样本不稳定。这里采用保守策略：不判定为正态。
        return np.nan, False
    try:
        p_value = float(shapiro(values).pvalue)
        return p_value, p_value > 0.05
    except Exception:
        return np.nan, False


def compare_two_groups_for_distribution(values_a, values_b):
    values_a = np.asarray(values_a, dtype=float)
    values_b = np.asarray(values_b, dtype=float)
    values_a = values_a[np.isfinite(values_a)]
    values_b = values_b[np.isfinite(values_b)]

    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "test": "Insufficient sample size",
            "p_value": np.nan,
            "normality_p_group_a": np.nan,
            "normality_p_group_b": np.nan,
            "group_a_normal": False,
            "group_b_normal": False,
        }

    if not HAS_SCIPY:
        logging.warning("未检测到 scipy，无法进行正态性检验和两组统计检验。导入失败原因：%s", SCIPY_IMPORT_ERROR)
        return {
            "test": "scipy unavailable",
            "p_value": np.nan,
            "normality_p_group_a": np.nan,
            "normality_p_group_b": np.nan,
            "group_a_normal": False,
            "group_b_normal": False,
        }

    normality_p_a, normal_a = normality_pvalue(values_a)
    normality_p_b, normal_b = normality_pvalue(values_b)

    try:
        if normal_a and normal_b:
            stat = ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
            return {
                "test": "Independent t-test (Welch)",
                "p_value": float(stat.pvalue),
                "normality_p_group_a": normality_p_a,
                "normality_p_group_b": normality_p_b,
                "group_a_normal": bool(normal_a),
                "group_b_normal": bool(normal_b),
            }
        stat = mannwhitneyu(values_a, values_b, alternative="two-sided")
        return {
            "test": "Mann-Whitney U test",
            "p_value": float(stat.pvalue),
            "normality_p_group_a": normality_p_a,
            "normality_p_group_b": normality_p_b,
            "group_a_normal": bool(normal_a),
            "group_b_normal": bool(normal_b),
        }
    except Exception as e:
        logging.warning("两组统计检验失败：%s", e)
        return {
            "test": "test failed",
            "p_value": np.nan,
            "normality_p_group_a": normality_p_a,
            "normality_p_group_b": normality_p_b,
            "group_a_normal": bool(normal_a),
            "group_b_normal": bool(normal_b),
        }


def get_combined_artifacts(feature_set_artifacts):
    """
    从 feature_set_artifacts 中提取每个任务的最终 Combined 模型 artifact。
    """
    combined_artifacts = {}
    for task_name, fs_artifacts in feature_set_artifacts.items():
        if not isinstance(fs_artifacts, dict):
            continue
        if "Combined" not in fs_artifacts:
            logging.warning("%s 没有 Combined artifact，跳过新增 Combined 分析。", task_name)
            continue
        combined_artifacts[task_name] = fs_artifacts["Combined"]
    return combined_artifacts


def build_all_available_feature_table(artifact):
    """
    用 Train+Validation、Internal test、External test 的最终入模特征值组成描述性分析表。
    该表仅用于分布图和统计检验，不参与任何训练或模型选择。
    """
    parts = []

    trainval_x = artifact.x_trainval_selected.reset_index(drop=True).copy()
    trainval_x[LABEL_COL] = artifact.y_trainval.astype(int)
    trainval_x["dataset_split"] = "Train_plus_Validation"
    parts.append(trainval_x)

    internal_x = artifact.x_internal_selected.reset_index(drop=True).copy()
    internal_x[LABEL_COL] = artifact.internal_true.astype(int)
    internal_x["dataset_split"] = "Internal_test"
    parts.append(internal_x)

    external_x = artifact.x_external_selected.reset_index(drop=True).copy()
    external_x[LABEL_COL] = artifact.external_true.astype(int)
    external_x["dataset_split"] = "External_test"
    parts.append(external_x)

    return pd.concat(parts, axis=0, ignore_index=True)


def select_top_distribution_features(artifact, radiomics_cols, morph_cols):
    """
    从最终 Combined 模型的 SHAP 排名中选择：
    1 个 Top radiomics feature + 1 个 Top morphology feature。
    如果 SHAP 不可用，则按最终入模特征顺序兜底选择。
    """
    selected_records = []
    importance_df = None

    if HAS_SHAP:
        try:
            importance_df = compute_shap_importance(
                model=artifact.final_model,
                model_name=artifact.model_name,
                x_df=artifact.x_trainval_selected,
                positive_label=artifact.positive_label,
                random_state=RANDOM_STATE,
            )
            importance_df["shap_rank"] = np.arange(1, len(importance_df) + 1)
            importance_df["feature_type"] = importance_df["feature"].apply(
                lambda f: feature_family(f, radiomics_cols, morph_cols)
            )
        except Exception as e:
            logging.warning("%s | Combined SHAP 排名计算失败，将按最终特征顺序兜底选择。原因：%s", artifact.task_name, e)
            importance_df = None
    else:
        logging.warning("未安装 shap，%s 的 Top feature distributions 将按最终特征顺序兜底选择。", artifact.task_name)

    if importance_df is not None and len(importance_df) > 0:
        for family_name in ["Radiomics", "Morphology"]:
            family_df = importance_df[importance_df["feature_type"] == family_name].copy()
            if family_df.empty:
                logging.warning("%s | Combined SHAP 排名中未找到 %s 特征。", artifact.task_name, family_name)
                continue
            row = family_df.iloc[0]
            selected_records.append({
                "feature": row["feature"],
                "feature_type": family_name,
                "shap_rank": int(row["shap_rank"]),
                "mean_abs_shap": float(row["mean_abs_shap"]),
                "selection_source": "Combined SHAP ranking",
            })

    if not selected_records:
        # 兜底：若 SHAP 失败，则仍尽量给出一项 radiomics 和一项 morphology。
        for family_name, family_cols in [("Radiomics", radiomics_cols), ("Morphology", morph_cols)]:
            for feature_name in artifact.selected_cols:
                if feature_name in family_cols:
                    selected_records.append({
                        "feature": feature_name,
                        "feature_type": family_name,
                        "shap_rank": np.nan,
                        "mean_abs_shap": np.nan,
                        "selection_source": "fallback_final_selected_features",
                    })
                    break

    return selected_records


def plot_top_feature_distribution_for_artifact(artifact, radiomics_cols, morph_cols):
    ensure_directories()
    logging.info("新增分析：Top feature distributions | %s | Combined | %s", artifact.task_name, artifact.model_name)

    selected_records = select_top_distribution_features(artifact, radiomics_cols, morph_cols)
    if not selected_records:
        logging.warning("%s 未能选出用于分布图的关键特征，跳过。", artifact.task_name)
        return []

    feature_table = build_all_available_feature_table(artifact)
    class_labels = sorted([int(x) for x in np.unique(feature_table[LABEL_COL].values)])
    if len(class_labels) != 2:
        logging.warning("%s 的分布图要求两组标签，但实际标签为：%s，跳过。", artifact.task_name, class_labels)
        return []

    pvalue_rows = []
    valid_records = []

    for record in selected_records:
        feature_name = record["feature"]
        if feature_name not in feature_table.columns:
            logging.warning("SHAP 特征不存在于数据表中，已跳过：%s | %s", artifact.task_name, feature_name)
            continue

        values_a = pd.to_numeric(
            feature_table.loc[feature_table[LABEL_COL] == class_labels[0], feature_name],
            errors="coerce",
        ).dropna().values
        values_b = pd.to_numeric(
            feature_table.loc[feature_table[LABEL_COL] == class_labels[1], feature_name],
            errors="coerce",
        ).dropna().values

        test_result = compare_two_groups_for_distribution(values_a, values_b)
        pvalue_rows.append({
            "analysis_label": artifact.analysis_label,
            "task": artifact.task_name,
            "feature_set": "Combined",
            "model": artifact.model_name,
            "feature": feature_name,
            "feature_type": record["feature_type"],
            "selection_source": record["selection_source"],
            "shap_rank": record["shap_rank"],
            "mean_abs_shap": record["mean_abs_shap"],
            "group_a_label": class_labels[0],
            "group_b_label": class_labels[1],
            "group_a_stage": stage_label_text(class_labels[0]),
            "group_b_stage": stage_label_text(class_labels[1]),
            "group_a_n": len(values_a),
            "group_b_n": len(values_b),
            "group_a_median": float(np.median(values_a)) if len(values_a) else np.nan,
            "group_b_median": float(np.median(values_b)) if len(values_b) else np.nan,
            "group_a_mean": float(np.mean(values_a)) if len(values_a) else np.nan,
            "group_b_mean": float(np.mean(values_b)) if len(values_b) else np.nan,
            "normality_p_group_a": test_result["normality_p_group_a"],
            "normality_p_group_b": test_result["normality_p_group_b"],
            "group_a_normal": test_result["group_a_normal"],
            "group_b_normal": test_result["group_b_normal"],
            "statistical_test": test_result["test"],
            "p_value": test_result["p_value"],
        })
        valid_records.append(record)

    if not valid_records:
        logging.warning("%s 没有可绘制的 Top feature distribution 特征。", artifact.task_name)
        return pvalue_rows

    n_features = len(valid_records)
    fig_width = max(4.6, 3.8 * n_features)
    fig, axes = plt.subplots(1, n_features, figsize=(fig_width, 4.4), squeeze=False)
    rng = np.random.default_rng(RANDOM_STATE)

    for ax, record in zip(axes.ravel(), valid_records):
        feature_name = record["feature"]
        values = []
        for label in class_labels:
            vals = pd.to_numeric(
                feature_table.loc[feature_table[LABEL_COL] == label, feature_name],
                errors="coerce",
            ).dropna().values
            values.append(vals)

        box_colors = [SCI_BOX_COLORS[i % len(SCI_BOX_COLORS)] for i in range(len(class_labels))]
        bp = ax.boxplot(
            values,
            positions=np.arange(len(class_labels)),
            widths=0.48,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.25},
            boxprops={"edgecolor": "black", "linewidth": 1.0},
            whiskerprops={"color": "black", "linewidth": 0.9},
            capprops={"color": "black", "linewidth": 0.9},
        )
        for patch, box_color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(box_color)
            patch.set_alpha(0.58)
            patch.set_edgecolor("black")

        for x_pos, vals in enumerate(values):
            if len(vals) == 0:
                continue
            jitter = rng.normal(loc=0.0, scale=0.045, size=len(vals))
            ax.scatter(
                np.full(len(vals), x_pos) + jitter,
                vals,
                s=8,
                facecolors=SCI_BOX_POINT_COLOR,
                edgecolors="none",
                linewidths=0,
                alpha=0.72,
                zorder=3,
            )

        p_row = next((r for r in pvalue_rows if r["feature"] == feature_name), None)
        if p_row is None or pd.isna(p_row["p_value"]):
            p_text = "P = NA"
        elif p_row["p_value"] < 0.001:
            p_text = "P < 0.001"
        else:
            p_text = f"P = {p_row['p_value']:.3f}"

        finite_all = np.concatenate([v[np.isfinite(v)] for v in values if len(v) > 0]) if any(len(v) > 0 for v in values) else np.array([])
        if len(finite_all) > 0:
            y_min, y_max = float(np.min(finite_all)), float(np.max(finite_all))
            y_span = y_max - y_min if y_max > y_min else 1.0
            ax.text(
                0.5,
                y_max + 0.08 * y_span,
                p_text,
                ha="center",
                va="bottom",
                fontsize=9,
            )
            ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.22 * y_span)

        ax.set_xticks(np.arange(len(class_labels)))
        ax.set_xticklabels([stage_label_text(v) for v in class_labels])
        ax.set_xlabel("ISL stage")
        ax.set_ylabel("Feature value")
        ax.set_title(feature_name, pad=8)
        apply_axis_style(ax)
        ax.grid(False)

    fig.suptitle(f"Top feature distributions | {artifact.task_name} | Combined", y=1.02)
    base_path = os.path.join(
        TOP_FEATURE_DISTRIBUTION_DIR,
        f"top_feature_distribution_{safe_filename(artifact.task_name)}",
    )
    png_path, pdf_path = save_png_and_pdf(fig, base_path)
    logging.info("Top feature distribution 图已保存：%s；%s", png_path, pdf_path)

    return pvalue_rows


def generate_top_feature_distribution_analysis(feature_set_artifacts, radiomics_cols, morph_cols):
    """
    新增 Top feature distributions 分析。
    输出：
    - 每个任务一张 PNG + PDF；
    - top_feature_distribution_pvalues.xlsx；
    - top_feature_distribution_pvalues.csv。
    """
    ensure_directories()
    combined_artifacts = get_combined_artifacts(feature_set_artifacts)
    if not combined_artifacts:
        logging.warning("没有可用于 Top feature distributions 的 Combined artifacts。")
        return pd.DataFrame()

    all_pvalue_rows = []
    for task_name, artifact in combined_artifacts.items():
        try:
            rows = plot_top_feature_distribution_for_artifact(
                artifact=artifact,
                radiomics_cols=radiomics_cols,
                morph_cols=morph_cols,
            )
            all_pvalue_rows.extend(rows)
        except Exception as e:
            logging.warning("Top feature distributions 失败：%s，原因：%s", task_name, e)

    pvalue_df = pd.DataFrame(all_pvalue_rows)
    xlsx_path = os.path.join(TOP_FEATURE_DISTRIBUTION_DIR, "top_feature_distribution_pvalues.xlsx")
    csv_path = os.path.join(TOP_FEATURE_DISTRIBUTION_DIR, "top_feature_distribution_pvalues.csv")
    pvalue_df.to_excel(xlsx_path, index=False)
    pvalue_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    logging.info("Top feature distributions P value 表已保存：%s；%s", xlsx_path, csv_path)
    return pvalue_df


def compute_shap_values_for_samples(model, model_name, x_background_df, x_sample_df, positive_label, random_state=42):
    """
    计算指定样本的 SHAP 值矩阵，返回 shape=(n_samples, n_features)。
    SHAP 值均按阳性类别概率/输出方向解释。
    """
    if not HAS_SHAP:
        raise RuntimeError("未检测到 shap，请先安装：pip install shap")

    x_background_df = x_background_df.copy()
    x_sample_df = x_sample_df.copy()
    feature_names = x_background_df.columns.tolist()

    if len(x_sample_df) == 0:
        return np.zeros((0, len(feature_names)), dtype=float)

    if len(x_background_df) == 0:
        raise ValueError("SHAP background 数据为空。")

    n_bg = min(SHAP_MAX_BACKGROUND, len(x_background_df))
    x_bg = x_background_df.sample(n=n_bg, random_state=random_state)
    x_exp = x_sample_df[feature_names].copy()

    if model_name in ["RandomForest", "XGBoost"]:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_exp)
        shap_mat = extract_shap_matrix(shap_values)

    elif model_name == "LogisticRegression" and isinstance(model, Pipeline):
        scaler = model.named_steps["scaler"]
        clf = model.named_steps["clf"]
        x_bg_scaled = scaler.transform(x_bg)
        x_exp_scaled = scaler.transform(x_exp)
        explainer = shap.LinearExplainer(clf, x_bg_scaled)
        shap_values = explainer(x_exp_scaled)
        shap_mat = extract_shap_matrix(shap_values)

    else:
        if model_name == "XGBoost":
            pos_index = 1
        else:
            if isinstance(model, Pipeline):
                class_list = list(model.named_steps["clf"].classes_)
            else:
                class_list = list(model.classes_)
            pos_index = class_list.index(positive_label)

        def predict_pos_prob(data_array):
            data_df = pd.DataFrame(np.asarray(data_array), columns=feature_names)
            return model.predict_proba(data_df)[:, pos_index]

        explainer = shap.KernelExplainer(predict_pos_prob, x_bg)
        shap_values = explainer.shap_values(x_exp, nsamples=SHAP_KERNEL_NSAMPLES)
        shap_mat = extract_shap_matrix(shap_values)

    shap_mat = np.asarray(shap_mat, dtype=float)
    if shap_mat.ndim == 1:
        shap_mat = shap_mat.reshape(1, -1)
    if shap_mat.shape[1] != len(feature_names):
        raise ValueError(f"SHAP 矩阵列数与特征数不一致：{shap_mat.shape[1]} vs {len(feature_names)}")
    return shap_mat


def format_top_shap_contributors(feature_names, shap_values, top_n=5):
    feature_names = list(feature_names)
    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.size == 0:
        return ""
    order = np.argsort(np.abs(shap_values))[::-1][:top_n]
    items = []
    for idx in order:
        items.append(f"{feature_names[idx]} ({shap_values[idx]:+.4g})")
    return "; ".join(items)


def detect_existing_path_columns(df):
    """
    自动识别可能的原始图像、mask/VOI、QC 图路径列。
    只做宽松识别；不存在真实文件时不会中断程序。
    """
    if df is None or df.empty:
        return {"raw_image": [], "mask": [], "qc": [], "all_path_cols": []}

    image_keywords = ["image", "img", "dicom", "dcm", "nii", "mrl", "ce", "原始", "图像"]
    mask_keywords = ["mask", "seg", "voi", "labelmap", "label_map", "分割", "掩膜"]
    qc_keywords = ["qc", "overlay", "mip", "png", "jpg", "figure", "fig", "可视化"]
    path_keywords = ["path", "路径", "file", "文件"]

    result = {"raw_image": [], "mask": [], "qc": [], "all_path_cols": []}

    for col in df.columns:
        c = str(col).lower()
        is_path_like = any(k in c for k in path_keywords + image_keywords + mask_keywords + qc_keywords)
        if not is_path_like:
            continue

        sample_values = df[col].dropna().astype(str).head(20).tolist()
        has_path_text = any(
            (os.path.sep in v or "/" in v or "\\" in v or re.search(r"\.(nii\.gz|nii|mha|mhd|nrrd|dcm|png|jpg|jpeg|tif|tiff|bmp|npy)$", v, re.I))
            for v in sample_values
        )
        if not has_path_text:
            continue

        result["all_path_cols"].append(col)
        if any(k in c for k in qc_keywords):
            result["qc"].append(col)
        elif any(k in c for k in mask_keywords):
            result["mask"].append(col)
        elif any(k in c for k in image_keywords + path_keywords):
            result["raw_image"].append(col)

    return result


def first_existing_path(row, columns):
    for col in columns:
        if col not in row.index:
            continue
        value = row.get(col)
        if pd.isna(value):
            continue
        path = str(value)
        if os.path.exists(path):
            return path
    return None


def normalize_image_array(arr):
    arr = np.asarray(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim == 3:
        # 取中间层作为二维展示。
        arr = arr[:, :, arr.shape[2] // 2]
    arr = np.asarray(arr, dtype=float)
    finite = arr[np.isfinite(arr)]
    if len(finite) == 0:
        return arr
    p1, p99 = np.percentile(finite, [1, 99])
    if p99 <= p1:
        p1, p99 = float(np.min(finite)), float(np.max(finite))
    if p99 > p1:
        arr = np.clip((arr - p1) / (p99 - p1), 0, 1)
    return arr


def load_image_like(path):
    if path is None:
        return None
    lower_path = path.lower()
    try:
        if lower_path.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")):
            return plt.imread(path)
        if lower_path.endswith(".npy"):
            return normalize_image_array(np.load(path))
        if lower_path.endswith((".nii", ".nii.gz", ".mha", ".mhd", ".nrrd")):
            if not HAS_NIBABEL:
                logging.warning("检测到 NIfTI/医学图像路径，但 nibabel 不可用，跳过图像读取：%s", path)
                return None
            img = nib.load(path)
            return normalize_image_array(img.get_fdata())
    except Exception as e:
        logging.warning("图像读取失败：%s，原因：%s", path, e)
    return None


def build_wrong_case_table_for_artifact(artifact, split_key):
    """
    基于最终 Combined 模型生成 internal 或 external wrong-case 表格。
    split_key: "internal" or "external"
    """
    if split_key == "internal":
        base_df = artifact.internal_df.reset_index(drop=True).copy()
        x_df = artifact.x_internal_selected.reset_index(drop=True).copy()
        y_true = artifact.internal_true.copy()
        y_pred = artifact.internal_pred.copy()
        y_prob = artifact.internal_prob.copy()
        dataset_split = "internal test"
        dataset_split_raw = "Internal_test_center134_20percent"
    elif split_key == "external":
        base_df = artifact.external_df.reset_index(drop=True).copy()
        x_df = artifact.x_external_selected.reset_index(drop=True).copy()
        y_true = artifact.external_true.copy()
        y_pred = artifact.external_pred.copy()
        y_prob = artifact.external_prob.copy()
        dataset_split = "external test"
        dataset_split_raw = "External_test_center2"
    else:
        raise ValueError(f"未知 split_key：{split_key}")

    wrong_mask = np.asarray(y_true) != np.asarray(y_pred)
    wrong_indices = np.where(wrong_mask)[0]

    path_info = detect_existing_path_columns(base_df)
    keep_cols = [col for col in [ID_COL, SIDE_COL, CENTER_COL, LABEL_COL] if col in base_df.columns]
    keep_cols += [c for c in path_info["all_path_cols"] if c not in keep_cols]

    if len(wrong_indices) == 0:
        out = pd.DataFrame(columns=[
            "analysis_label", "task", "feature_set", "model", "dataset_split",
            "patient_id_or_case_id", "limb_id_or_side", "true_label", "predicted_label",
            "predicted_probability_positive", "predicted_probability_of_predicted_label",
            "is_borderline_case", "top_shap_contributing_features_top5",
        ] + keep_cols)
        return out, np.zeros((0, len(artifact.selected_cols)), dtype=float), x_df.iloc[[]].copy()

    wrong_base = base_df.iloc[wrong_indices].copy().reset_index(drop=True)
    wrong_x = x_df.iloc[wrong_indices].copy().reset_index(drop=True)

    try:
        shap_mat = compute_shap_values_for_samples(
            model=artifact.final_model,
            model_name=artifact.model_name,
            x_background_df=artifact.x_trainval_selected,
            x_sample_df=wrong_x,
            positive_label=artifact.positive_label,
            random_state=RANDOM_STATE,
        )
    except Exception as e:
        logging.warning("%s | %s wrong cases 的 SHAP 计算失败，top SHAP features 将留空。原因：%s", artifact.task_name, dataset_split, e)
        shap_mat = np.full((len(wrong_indices), len(artifact.selected_cols)), np.nan, dtype=float)

    rows = []
    id_candidates = [ID_COL, "patient_id", "PatientID", "case_id", "CaseID", "编号", "病例号"]
    side_candidates = [SIDE_COL, "side", "Side", "limb_id", "LimbID", "left_right"]

    for local_i, original_idx in enumerate(wrong_indices):
        src_row = wrong_base.iloc[local_i]
        patient_id = ""
        for col in id_candidates:
            if col in src_row.index and pd.notna(src_row[col]):
                patient_id = src_row[col]
                break
        limb_or_side = ""
        for col in side_candidates:
            if col in src_row.index and pd.notna(src_row[col]):
                limb_or_side = src_row[col]
                break

        true_label = int(y_true[original_idx])
        pred_label = int(y_pred[original_idx])
        prob_positive = float(y_prob[original_idx])
        prob_predicted = prob_positive if pred_label == int(artifact.positive_label) else 1.0 - prob_positive
        borderline_distance = abs(prob_positive - 0.5)
        is_borderline = borderline_distance <= 0.10

        if np.isnan(shap_mat[local_i]).all():
            top5 = ""
            top3 = ""
        else:
            top5 = format_top_shap_contributors(artifact.selected_cols, shap_mat[local_i], top_n=5)
            top3 = format_top_shap_contributors(artifact.selected_cols, shap_mat[local_i], top_n=3)

        record = {
            "analysis_label": artifact.analysis_label,
            "task": artifact.task_name,
            "feature_set": "Combined",
            "model": artifact.model_name,
            "dataset_split": dataset_split,
            "dataset_split_raw": dataset_split_raw,
            "row_index_in_split": int(original_idx),
            "patient_id_or_case_id": patient_id,
            "limb_id_or_side": limb_or_side,
            "true_label": true_label,
            "predicted_label": pred_label,
            "true_stage": stage_label_text(true_label),
            "predicted_stage": stage_label_text(pred_label),
            "predicted_probability_positive": prob_positive,
            "predicted_probability_of_predicted_label": prob_predicted,
            "borderline_distance_to_0_5": borderline_distance,
            "is_borderline_case": bool(is_borderline),
            "is_high_confidence_wrong": bool(prob_predicted >= 0.80),
            "top_shap_contributing_features_top5": top5,
            "top_shap_contributing_features_top3": top3,
        }
        for col in keep_cols:
            record[col] = src_row[col] if col in src_row.index else ""
        rows.append(record)

    wrong_df = pd.DataFrame(rows)
    return wrong_df, shap_mat, wrong_x


def select_typical_wrong_cases(wrong_df, max_cases=6):
    if wrong_df is None or wrong_df.empty:
        return []
    wrong_df = wrong_df.copy()
    wrong_df["_rank_borderline"] = pd.to_numeric(wrong_df["borderline_distance_to_0_5"], errors="coerce")
    wrong_df["_rank_high_conf"] = pd.to_numeric(wrong_df["predicted_probability_of_predicted_label"], errors="coerce")

    borderline = wrong_df.sort_values("_rank_borderline", ascending=True).head(min(3, max_cases))
    high_conf = wrong_df.sort_values("_rank_high_conf", ascending=False).head(min(3, max_cases))

    selected = pd.concat([borderline, high_conf], ignore_index=True)
    selected = selected.drop_duplicates(subset=["task", "dataset_split", "row_index_in_split"], keep="first")

    if len(selected) < min(max_cases, len(wrong_df)):
        filler = wrong_df.sort_values("_rank_borderline", ascending=True)
        selected = pd.concat([selected, filler], ignore_index=True)
        selected = selected.drop_duplicates(subset=["task", "dataset_split", "row_index_in_split"], keep="first")

    return selected.head(min(max_cases, len(wrong_df))).index.tolist(), selected.head(min(max_cases, len(wrong_df))).copy()


def plot_wrong_case_panel(row, artifact, shap_values, feature_values, save_base_path):
    """
    如果能自动读取 QC/原始图像/mask，则绘制影像 + SHAP；否则仅绘制文本 + SHAP bar plot。
    """
    path_info = detect_existing_path_columns(pd.DataFrame([row]))
    qc_path = first_existing_path(row, path_info["qc"])
    raw_path = first_existing_path(row, path_info["raw_image"])
    mask_path = first_existing_path(row, path_info["mask"])

    qc_img = load_image_like(qc_path)
    raw_img = load_image_like(raw_path)
    mask_img = load_image_like(mask_path)

    has_image = qc_img is not None or raw_img is not None
    if has_image:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2))
        ax_img, ax_bar = axes
        img_to_show = qc_img if qc_img is not None else raw_img
        if img_to_show is not None and np.asarray(img_to_show).ndim == 2:
            ax_img.imshow(img_to_show, cmap="gray")
        else:
            ax_img.imshow(img_to_show)
        if qc_img is None and mask_img is not None and np.asarray(mask_img).ndim == 2:
            ax_img.imshow(mask_img, cmap="Reds", alpha=0.30)
        ax_img.set_axis_off()
        ax_img.set_title("Image / mask overlay", pad=6)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.2), gridspec_kw={"width_ratios": [1.0, 1.25]})
        ax_text, ax_bar = axes
        ax_text.axis("off")
        text_lines = [
            f"Task: {row.get('task', '')}",
            f"Split: {row.get('dataset_split', '')}",
            f"Case: {row.get('patient_id_or_case_id', '')}",
            f"Limb/side: {row.get('limb_id_or_side', '')}",
            f"True: {row.get('true_stage', row.get('true_label', ''))}",
            f"Predicted: {row.get('predicted_stage', row.get('predicted_label', ''))}",
            f"P(positive): {row.get('predicted_probability_positive', np.nan):.3f}",
        ]
        ax_text.text(0.0, 0.95, "\n".join(text_lines), ha="left", va="top", fontsize=10)
        ax_text.set_title("Wrong-case metadata", pad=6)

    shap_values = np.asarray(shap_values, dtype=float)
    if shap_values.size == 0 or np.isnan(shap_values).all():
        ax_bar.axis("off")
        ax_bar.text(0.0, 0.5, "SHAP values unavailable", ha="left", va="center", fontsize=10)
    else:
        order = np.argsort(np.abs(shap_values))[::-1][:10]
        top_features = [artifact.selected_cols[i] for i in order][::-1]
        top_values = shap_values[order][::-1]
        y_pos = np.arange(len(top_features))
        bar_colors = [SHAP_POSITIVE_COLOR if v >= 0 else SHAP_NEGATIVE_COLOR for v in top_values]
        ax_bar.barh(y_pos, top_values, color=bar_colors, edgecolor="black", linewidth=0.35, alpha=0.86)
        ax_bar.axvline(0, color="black", linewidth=0.8)
        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(top_features)
        ax_bar.set_xlabel("SHAP value")
        ax_bar.set_title("Top SHAP contributors", pad=6)
        apply_axis_style(ax_bar)
        ax_bar.grid(False)

    fig.suptitle(
        f"{row.get('task', '')} | true {row.get('true_label', '')} vs predicted {row.get('predicted_label', '')} | P={row.get('predicted_probability_positive', np.nan):.3f}",
        y=1.02,
        fontsize=11,
    )
    png_path, pdf_path = save_png_and_pdf(fig, save_base_path)
    return png_path, pdf_path


def generate_wrong_case_analysis(feature_set_artifacts):
    """
    新增 Wrong-case examples / misclassified case analysis。
    基于最终 Combined 模型输出：
    - wrong_cases_internal.xlsx
    - wrong_cases_external.xlsx
    - wrong_case_summary.csv
    - wrong_case_figures/*.png 和 *.pdf

    表格保留 internal/external 中全部误分类病例；图像面板每个任务最多选择 6 个典型 wrong cases，
    优先包括 borderline wrong cases，其次包括 high-confidence wrong cases。
    """
    ensure_directories()
    combined_artifacts = get_combined_artifacts(feature_set_artifacts)
    if not combined_artifacts:
        logging.warning("没有可用于 wrong-case analysis 的 Combined artifacts。")
        return {
            "internal": pd.DataFrame(),
            "external": pd.DataFrame(),
            "summary": pd.DataFrame(),
        }

    internal_tables = []
    external_tables = []

    for task_name, artifact in combined_artifacts.items():
        logging.info("新增分析：Wrong-case analysis | %s | Combined | %s", task_name, artifact.model_name)

        split_cache = {}
        for split_key in ["internal", "external"]:
            try:
                wrong_df, shap_mat, wrong_x = build_wrong_case_table_for_artifact(artifact, split_key=split_key)
            except Exception as e:
                logging.warning("Wrong-case 表格生成失败：%s | %s，原因：%s", task_name, split_key, e)
                continue

            split_cache[split_key] = {
                "wrong_df": wrong_df,
                "shap_mat": shap_mat,
                "wrong_x": wrong_x,
            }

            if split_key == "internal":
                internal_tables.append(wrong_df)
            else:
                external_tables.append(wrong_df)

            if wrong_df.empty:
                logging.info("%s | %s 没有误分类病例。", task_name, split_key)
            else:
                logging.info("%s | %s 误分类病例数：%d", task_name, split_key, len(wrong_df))

        # 每个任务最多生成 6 个典型 wrong-case 图像面板。
        combined_wrong_for_selection = []
        for split_key, cache in split_cache.items():
            tmp = cache["wrong_df"].copy()
            if tmp.empty:
                continue
            tmp["_split_key_for_figure"] = split_key
            combined_wrong_for_selection.append(tmp)

        if not combined_wrong_for_selection:
            continue

        combined_wrong_df = pd.concat(combined_wrong_for_selection, ignore_index=True)
        _, selected_cases = select_typical_wrong_cases(combined_wrong_df, max_cases=6)
        if selected_cases.empty:
            continue

        for _, case_row in selected_cases.iterrows():
            split_key = case_row.get("_split_key_for_figure", "")
            if split_key not in split_cache:
                continue
            cache = split_cache[split_key]
            wrong_df = cache["wrong_df"]
            shap_mat = cache["shap_mat"]
            wrong_x = cache["wrong_x"]

            row_index = int(case_row["row_index_in_split"])
            match_positions = wrong_df.index[wrong_df["row_index_in_split"] == row_index].tolist()
            if not match_positions:
                continue
            local_pos = match_positions[0]
            shap_values = shap_mat[local_pos] if shap_mat.shape[0] > local_pos else np.array([])
            feature_values = wrong_x.iloc[local_pos] if len(wrong_x) > local_pos else pd.Series(dtype=float)

            case_id = safe_filename(case_row.get("patient_id_or_case_id", f"case_{row_index}"))
            limb_id = safe_filename(case_row.get("limb_id_or_side", "limb"))
            base_name = f"wrong_case_{safe_filename(task_name)}_{safe_filename(split_key)}_{case_id}_{limb_id}_idx{row_index}"
            save_base_path = os.path.join(WRONG_CASE_FIGURE_DIR, base_name)
            try:
                png_path, pdf_path = plot_wrong_case_panel(
                    row=case_row,
                    artifact=artifact,
                    shap_values=shap_values,
                    feature_values=feature_values,
                    save_base_path=save_base_path,
                )
                logging.info("Wrong-case 图已保存：%s；%s", png_path, pdf_path)
            except Exception as e:
                logging.warning("Wrong-case 图绘制失败：%s | %s | row=%s，原因：%s", task_name, split_key, row_index, e)

    internal_df = pd.concat(internal_tables, ignore_index=True) if internal_tables else pd.DataFrame()
    external_df = pd.concat(external_tables, ignore_index=True) if external_tables else pd.DataFrame()

    internal_path = os.path.join(WRONG_CASE_ANALYSIS_DIR, "wrong_cases_internal.xlsx")
    external_path = os.path.join(WRONG_CASE_ANALYSIS_DIR, "wrong_cases_external.xlsx")
    internal_df.to_excel(internal_path, index=False)
    external_df.to_excel(external_path, index=False)

    summary_frames = []
    for split_name, df_split in [("internal test", internal_df), ("external test", external_df)]:
        if df_split.empty:
            summary_frames.append(pd.DataFrame([{
                "dataset_split": split_name,
                "task": "all",
                "wrong_case_count": 0,
                "borderline_wrong_count": 0,
                "high_confidence_wrong_count": 0,
            }]))
            continue
        summary = (
            df_split
            .groupby(["task", "dataset_split"], dropna=False)
            .agg(
                wrong_case_count=("true_label", "size"),
                borderline_wrong_count=("is_borderline_case", "sum"),
                high_confidence_wrong_count=("is_high_confidence_wrong", "sum"),
            )
            .reset_index()
        )
        summary_frames.append(summary)

    summary_df = pd.concat(summary_frames, ignore_index=True) if summary_frames else pd.DataFrame()
    summary_path = os.path.join(WRONG_CASE_ANALYSIS_DIR, "wrong_case_summary.csv")
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    logging.info("Wrong-case internal 表已保存：%s", internal_path)
    logging.info("Wrong-case external 表已保存：%s", external_path)
    logging.info("Wrong-case summary 表已保存：%s", summary_path)

    return {
        "internal": internal_df,
        "external": external_df,
        "summary": summary_df,
    }



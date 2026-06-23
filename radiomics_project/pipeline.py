
# 13. 主流程：保留 Global_best 指标和错误病例，并新增所有候选模型 CV 与特征名称输出


def run_all_tasks(df, radiomics_cols, morph_cols, analysis_label="Primary"):
    model_names = list(BASE_MODEL_NAMES)
    if HAS_XGB:
        model_names.append("XGBoost")
    else:
        logging.info("未检测到 xgboost，将跳过 XGBoost。导入失败原因：%s", XGB_IMPORT_ERROR)

    final_metrics = []
    final_wrong = []
    split_samples = []
    candidate_cv_rows_all = []
    feature_selection_rows_all = []
    used_features_rows_all = []
    all_case_predictions = []
    global_artifacts = {}
    feature_set_artifacts = {}

    for class_a, class_b in TASKS:
        task_name = task_name_from_classes(class_a, class_b)
        logging.info("=" * 100)
        logging.info("开始任务：%s | 分析类型：%s", task_name, analysis_label)
        logging.info("=" * 100)

        data = prepare_task_data(
            df=df,
            class_a=class_a,
            class_b=class_b,
            radiomics_cols=radiomics_cols,
            morph_cols=morph_cols,
        )

        # 保存该任务的具体样本划分列表，方便复现。
        split_samples.append(make_split_samples_df(
            analysis_label=analysis_label,
            task_name=task_name,
            data=data,
        ))

        best_cv_row = None
        best_cv_by_feature_set = {}

        # 1. 在 Train + Validation 上进行 5 折交叉验证筛选：
        #    A. 全局最优模型；
        #    B. 每个特征集内部的最优模型；
        #    C. 保存每个特征集 × 每个模型的完整 5 折 CV 汇总；
        #    D. 保存每次 CV 折内筛选后剩余特征名称。
        for fs_name in FEATURE_SETS:
            for model_name in model_names:
                logging.info("候选模型 5折CV：%s | %s | %s | %s", analysis_label, task_name, fs_name, model_name)
                try:
                    candidate_result = run_candidate_on_trainval_cv(
                        analysis_label=analysis_label,
                        data=data,
                        class_a=class_a,
                        class_b=class_b,
                        feature_set_name=fs_name,
                        model_name=model_name,
                        radiomics_cols=radiomics_cols,
                        morph_cols=morph_cols,
                    )
                    row = candidate_result["row"]
                    feature_selection_rows_all.extend(candidate_result["feature_selection_rows"])
                    candidate_cv_rows_all.append(row)
                except Exception as e:
                    logging.warning("跳过候选：%s | %s | %s | %s，原因：%s", analysis_label, task_name, fs_name, model_name, e)
                    continue

                logging.info(
                    "5折CV mean±sd：AUC=%s, ACC=%s, PPV=%s, Sens=%s, Spec=%s, F1=%s, Brier=%s | 最终总特征数=%d",
                    row["CV_AUC"], row["CV_ACC"], row["CV_PPV"], row["CV_Sensitivity"], row["CV_Specificity"], row["CV_F1"], row["CV_Brier"],
                    row["总特征数"],
                )

                # 全局最优：所有特征集、所有候选模型一起比较，比较依据为 CV mean。
                if is_better_model(row, best_cv_row):
                    best_cv_row = row.copy()

                # 特征集内部最优：每个特征集内部单独比较，比较依据为 CV mean。
                if fs_name not in best_cv_by_feature_set:
                    best_cv_by_feature_set[fs_name] = row.copy()
                elif is_better_model(row, best_cv_by_feature_set[fs_name]):
                    best_cv_by_feature_set[fs_name] = row.copy()

        if best_cv_row is None:
            raise RuntimeError(f"{task_name} 未能筛选出全局最优模型。")

        best_feature_set = best_cv_row["特征集"]
        best_model_name = best_cv_row["模型"]

        logging.info(
            "全局最优模型：%s | %s | %s | 5折CV AUC=%s ACC=%s PPV=%s",
            task_name,
            best_feature_set,
            best_model_name,
            best_cv_row["CV_AUC"],
            best_cv_row["CV_ACC"],
            best_cv_row["CV_PPV"],
        )

        # 2. 评估全局最优模型。
        global_eval = evaluate_final_model(
            analysis_label=analysis_label,
            data=data,
            class_a=class_a,
            class_b=class_b,
            feature_set_name=best_feature_set,
            model_name=best_model_name,
            radiomics_cols=radiomics_cols,
            morph_cols=morph_cols,
            result_tag="Global_best",
            cv_summary_row=best_cv_row,
        )

        final_metrics.extend([global_eval["internal_row"], global_eval["external_row"]])
        final_wrong.append(global_eval["wrong_df"])
        all_case_predictions.append(global_eval["all_case_predictions_df"])
        feature_selection_rows_all.extend(global_eval["feature_selection_rows"])
        used_features_rows_all.extend(global_eval["used_features_rows"])
        global_artifacts[task_name] = global_eval["artifact"]

        logging.info(
            "全局最优内部测试：AUC=%s ACC=%s PPV=%s Sens=%s Spec=%s F1=%s Brier=%s CalSlope=%s | 错误例数=%d",
            global_eval["internal_row"]["AUC"], global_eval["internal_row"]["ACC"],
            global_eval["internal_row"]["PPV"], global_eval["internal_row"]["Sensitivity"],
            global_eval["internal_row"]["Specificity"], global_eval["internal_row"]["F1"],
            global_eval["internal_row"]["Brier"], global_eval["internal_row"]["Calibration_slope"],
            len(global_eval["wrong_df"][global_eval["wrong_df"]["数据集"] == "Internal_test_center134_20percent"]),
        )
        logging.info(
            "全局最优外部测试：AUC=%s ACC=%s PPV=%s Sens=%s Spec=%s F1=%s Brier=%s CalSlope=%s | 错误例数=%d",
            global_eval["external_row"]["AUC"], global_eval["external_row"]["ACC"],
            global_eval["external_row"]["PPV"], global_eval["external_row"]["Sensitivity"],
            global_eval["external_row"]["Specificity"], global_eval["external_row"]["F1"],
            global_eval["external_row"]["Brier"], global_eval["external_row"]["Calibration_slope"],
            len(global_eval["wrong_df"][global_eval["wrong_df"]["数据集"] == "External_test_center2"]),
        )

        # 3. 评估三种特征集各自的最佳模型。
        for fs_name in FEATURE_SETS:
            if fs_name not in best_cv_by_feature_set:
                logging.warning("%s | %s | %s 没有可用候选模型，无法输出该特征集结果。", analysis_label, task_name, fs_name)
                continue

            fs_best_row = best_cv_by_feature_set[fs_name]
            fs_best_model = fs_best_row["模型"]

            logging.info(
                "该特征集最佳模型：%s | %s | %s | 5折CV AUC=%s ACC=%s PPV=%s",
                task_name,
                fs_name,
                fs_best_model,
                fs_best_row["CV_AUC"],
                fs_best_row["CV_ACC"],
                fs_best_row["CV_PPV"],
            )

            try:
                fs_eval = evaluate_final_model(
                    analysis_label=analysis_label,
                    data=data,
                    class_a=class_a,
                    class_b=class_b,
                    feature_set_name=fs_name,
                    model_name=fs_best_model,
                    radiomics_cols=radiomics_cols,
                    morph_cols=morph_cols,
                    result_tag="Feature_set_best",
                    cv_summary_row=fs_best_row,
                )
            except Exception as e:
                logging.warning(
                    "该特征集最佳模型测试集评估失败：%s | %s | %s | %s，原因：%s",
                    analysis_label,
                    task_name,
                    fs_name,
                    fs_best_model,
                    e,
                )
                continue

            final_metrics.extend([fs_eval["internal_row"], fs_eval["external_row"]])
            all_case_predictions.append(fs_eval["all_case_predictions_df"])
            feature_selection_rows_all.extend(fs_eval["feature_selection_rows"])
            used_features_rows_all.extend(fs_eval["used_features_rows"])

            feature_set_artifacts.setdefault(task_name, {})[fs_name] = fs_eval["artifact"]

            # 全局最优特征集的错误病例已经以 Global_best 输出过，这里不重复输出。
            if fs_name != best_feature_set:
                final_wrong.append(fs_eval["wrong_df"])

            logging.info(
                "%s | %s | %s 内部测试：AUC=%s ACC=%s PPV=%s Sens=%s Spec=%s F1=%s Brier=%s | 错误例数=%d",
                analysis_label,
                task_name,
                fs_name,
                fs_eval["internal_row"]["AUC"],
                fs_eval["internal_row"]["ACC"],
                fs_eval["internal_row"]["PPV"],
                fs_eval["internal_row"]["Sensitivity"],
                fs_eval["internal_row"]["Specificity"],
                fs_eval["internal_row"]["F1"],
                fs_eval["internal_row"]["Brier"],
                len(fs_eval["wrong_df"][fs_eval["wrong_df"]["数据集"] == "Internal_test_center134_20percent"]),
            )
            logging.info(
                "%s | %s | %s 外部测试：AUC=%s ACC=%s PPV=%s Sens=%s Spec=%s F1=%s Brier=%s | 错误例数=%d",
                analysis_label,
                task_name,
                fs_name,
                fs_eval["external_row"]["AUC"],
                fs_eval["external_row"]["ACC"],
                fs_eval["external_row"]["PPV"],
                fs_eval["external_row"]["Sensitivity"],
                fs_eval["external_row"]["Specificity"],
                fs_eval["external_row"]["F1"],
                fs_eval["external_row"]["Brier"],
                len(fs_eval["wrong_df"][fs_eval["wrong_df"]["数据集"] == "External_test_center2"]),
            )

    final_metrics_df_raw = pd.DataFrame(final_metrics)
    final_wrong_df = pd.concat(final_wrong, ignore_index=True) if final_wrong else pd.DataFrame()
    split_samples_df = pd.concat(split_samples, ignore_index=True) if split_samples else pd.DataFrame()
    candidate_cv_df_raw = pd.DataFrame(candidate_cv_rows_all)
    feature_selection_df = pd.DataFrame(feature_selection_rows_all)
    used_features_df = pd.DataFrame(used_features_rows_all)
    all_case_predictions_df = pd.concat(all_case_predictions, ignore_index=True) if all_case_predictions else pd.DataFrame()

    return {
        "final_metrics_df": clean_table_for_export(final_metrics_df_raw),
        "final_wrong_df": final_wrong_df,
        "split_samples_df": split_samples_df,
        "candidate_cv_df": clean_table_for_export(candidate_cv_df_raw),
        "feature_selection_df": feature_selection_df,
        "used_features_df": used_features_df,
        "all_case_predictions_df": all_case_predictions_df,
        "global_artifacts": global_artifacts,
        "feature_set_artifacts": feature_set_artifacts,
    }




# 14. Excel 输出：指标、错误病例、样本划分、候选模型 CV、特征名称、所有病例预测


def save_results_to_excel(metrics_df, wrong_df, split_samples_df, candidate_cv_df, feature_selection_df, used_features_df, all_case_predictions_df):
    logging.info("保存 Excel：%s", SAVE_XLSX)

    ensure_directories()
    with pd.ExcelWriter(SAVE_XLSX, engine="openpyxl") as writer:
        metrics_df.to_excel(writer, sheet_name="metrics_global_best", index=False)
        wrong_df.to_excel(writer, sheet_name="wrong_global_best", index=False)
        split_samples_df.to_excel(writer, sheet_name="split_samples", index=False)
        candidate_cv_df.to_excel(writer, sheet_name="candidate_5fold_cv_all_models", index=False)
        feature_selection_df.to_excel(writer, sheet_name="feature_selection_features", index=False)
        used_features_df.to_excel(writer, sheet_name="used_features_by_task", index=False)
        all_case_predictions_df.to_excel(writer, sheet_name="all_cases_predictions", index=False)

    logging.info(
        "Excel 保存完成。包含 sheet: metrics_global_best, wrong_global_best, split_samples, "
        "candidate_5fold_cv_all_models, feature_selection_features, used_features_by_task, all_cases_predictions"
    )




# 14.1 最终模型封装保存：供独立前瞻性数据推理使用


def save_global_model_packages(global_artifacts, model_package_dir=MODEL_PACKAGE_DIR):
    """
    只封装每个任务的 Global_best 最终模型，不改变原有训练、评估、绘图和 Excel 输出逻辑。

    每个 joblib 包含：
    - model: 已训练好的 sklearn / xgboost / Pipeline 模型
    - selected_cols: 最终模型实际使用的特征列，推理 Excel 必须包含这些列
    - labels_sorted: 二分类任务原始标签，例如 [1, 2] 或 [2, 3]
    - positive_label: 阳性类别，1v2 为 2，2v3 为 3
    - negative_label: 阴性类别，1v2 为 1，2v3 为 2
    - task_name / feature_set_name / model_name / analysis_label / result_tag: 模型来源信息
    - trainval_feature_mean / trainval_feature_std: 训练+验证集特征分布，便于后续做输入检查或 OOD 评估
    """
    os.makedirs(model_package_dir, exist_ok=True)

    manifest_rows = []

    for task_key, artifact in global_artifacts.items():
        if artifact is None:
            continue

        labels_sorted = [int(x) for x in artifact.labels_sorted]
        positive_label = int(artifact.positive_label)
        negative_candidates = [x for x in labels_sorted if x != positive_label]
        negative_label = int(negative_candidates[0]) if negative_candidates else None

        selected_cols = list(artifact.selected_cols)

        # 保存训练+验证集的特征分布，后续新病例推理前可用于检查是否明显偏离训练分布。
        trainval_feature_mean = artifact.x_trainval_selected.mean(axis=0).to_dict()
        trainval_feature_std = artifact.x_trainval_selected.std(axis=0).replace(0, np.nan).to_dict()

        package = {
            "model": artifact.final_model,
            "selected_cols": selected_cols,
            "labels_sorted": labels_sorted,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "task_name": artifact.task_name,
            "feature_set_name": artifact.feature_set_name,
            "model_name": artifact.model_name,
            "analysis_label": artifact.analysis_label,
            "result_tag": artifact.result_tag,
            "random_state": RANDOM_STATE,
            "train_ratio": TRAIN_RATIO,
            "val_ratio": VAL_RATIO,
            "internal_test_ratio": INTERNAL_TEST_RATIO,
            "internal_center": INTERNAL_CENTER,
            "external_center": EXTERNAL_CENTER,
            "label_col": LABEL_COL,
            "id_col": ID_COL,
            "side_col": SIDE_COL,
            "center_col": CENTER_COL,
            "trainval_feature_mean": trainval_feature_mean,
            "trainval_feature_std": trainval_feature_std,
            "package_note": "Global_best model trained on Train + Validation only. Use selected_cols exactly for inference.",
        }

        filename = (
            f"global_best_{safe_filename(artifact.analysis_label)}_"
            f"{safe_filename(artifact.task_name)}.joblib"
        )
        save_path = os.path.join(model_package_dir, filename)
        joblib.dump(package, save_path)

        manifest_rows.append({
            "analysis_label": artifact.analysis_label,
            "task_name": artifact.task_name,
            "feature_set_name": artifact.feature_set_name,
            "model_name": artifact.model_name,
            "result_tag": artifact.result_tag,
            "positive_label": positive_label,
            "negative_label": negative_label,
            "labels_sorted": str(labels_sorted),
            "n_selected_features": len(selected_cols),
            "package_path": save_path,
            "selected_cols": ";".join(selected_cols),
        })

        logging.info(
            "Global_best 模型封装已保存：%s | task=%s | feature_set=%s | model=%s | 特征数=%d",
            save_path,
            artifact.task_name,
            artifact.feature_set_name,
            artifact.model_name,
            len(selected_cols),
        )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(model_package_dir, "model_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    logging.info("模型封装清单已保存：%s", manifest_path)

    return manifest_df



def save_feature_set_best_model_packages(feature_set_artifacts, model_package_dir=MODEL_PACKAGE_DIR):
    """
    封装每个任务下 Radiomics_only / Morphology_only / Combined 三种特征集各自的 Feature_set_best 最终模型。

    说明：
    - 不改变原有 Global_best 模型封装逻辑；
    - 不改变原有 Excel、绘图、错误病例分析等输出逻辑；
    - 每个 joblib 包结构与 Global_best package 保持一致，便于后续统一推理；
    - 额外生成 feature_set_best_model_manifest.csv，记录 R / M / M+R 各自最优模型 package 路径。
    """
    os.makedirs(model_package_dir, exist_ok=True)

    manifest_rows = []

    for task_key, fs_artifacts in feature_set_artifacts.items():
        if not isinstance(fs_artifacts, dict):
            continue

        for fs_name in FEATURE_SETS:
            artifact = fs_artifacts.get(fs_name)
            if artifact is None:
                logging.warning("%s | %s 没有可封装的 Feature_set_best artifact，已跳过。", task_key, fs_name)
                continue

            labels_sorted = [int(x) for x in artifact.labels_sorted]
            positive_label = int(artifact.positive_label)
            negative_candidates = [x for x in labels_sorted if x != positive_label]
            negative_label = int(negative_candidates[0]) if negative_candidates else None

            selected_cols = list(artifact.selected_cols)

            # 保存训练+验证集的特征分布，后续新病例推理前可用于检查是否明显偏离训练分布。
            trainval_feature_mean = artifact.x_trainval_selected.mean(axis=0).to_dict()
            trainval_feature_std = artifact.x_trainval_selected.std(axis=0).replace(0, np.nan).to_dict()

            package = {
                "model": artifact.final_model,
                "selected_cols": selected_cols,
                "labels_sorted": labels_sorted,
                "positive_label": positive_label,
                "negative_label": negative_label,
                "task_name": artifact.task_name,
                "feature_set_name": artifact.feature_set_name,
                "feature_set_short_name": short_feature_set_name(artifact.feature_set_name),
                "model_name": artifact.model_name,
                "analysis_label": artifact.analysis_label,
                "result_tag": artifact.result_tag,
                "package_scope": "Feature_set_best",
                "random_state": RANDOM_STATE,
                "train_ratio": TRAIN_RATIO,
                "val_ratio": VAL_RATIO,
                "internal_test_ratio": INTERNAL_TEST_RATIO,
                "internal_center": INTERNAL_CENTER,
                "external_center": EXTERNAL_CENTER,
                "label_col": LABEL_COL,
                "id_col": ID_COL,
                "side_col": SIDE_COL,
                "center_col": CENTER_COL,
                "trainval_feature_mean": trainval_feature_mean,
                "trainval_feature_std": trainval_feature_std,
                "package_note": "Feature_set_best model trained on Train + Validation only. Use selected_cols exactly for inference.",
            }

            filename = (
                f"feature_set_best_{safe_filename(artifact.analysis_label)}_"
                f"{safe_filename(artifact.task_name)}_"
                f"{safe_filename(short_feature_set_name(artifact.feature_set_name))}.joblib"
            )
            save_path = os.path.join(model_package_dir, filename)
            joblib.dump(package, save_path)

            manifest_rows.append({
                "analysis_label": artifact.analysis_label,
                "task_name": artifact.task_name,
                "feature_set_name": artifact.feature_set_name,
                "feature_set_short_name": short_feature_set_name(artifact.feature_set_name),
                "model_name": artifact.model_name,
                "result_tag": artifact.result_tag,
                "package_scope": "Feature_set_best",
                "positive_label": positive_label,
                "negative_label": negative_label,
                "labels_sorted": str(labels_sorted),
                "n_selected_features": len(selected_cols),
                "package_path": save_path,
                "selected_cols": ";".join(selected_cols),
            })

            logging.info(
                "Feature_set_best 模型封装已保存：%s | task=%s | feature_set=%s(%s) | model=%s | 特征数=%d",
                save_path,
                artifact.task_name,
                artifact.feature_set_name,
                short_feature_set_name(artifact.feature_set_name),
                artifact.model_name,
                len(selected_cols),
            )

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(model_package_dir, "feature_set_best_model_manifest.csv")
    manifest_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    logging.info("Feature_set_best 模型封装清单已保存：%s", manifest_path)

    return manifest_df



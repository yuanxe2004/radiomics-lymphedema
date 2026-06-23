
# 15. 主入口


"""Command-line runner for the radiomics pipeline."""

import argparse
import os
import sys as _sys

from . import config as _config
from . import cv as _cv
from . import data as _data
from . import pipeline as _pipeline
from . import plotting as _plotting
from . import utils as _utils
from .config import *
from .data import identify_feature_columns, read_and_validate_data
from .pipeline import (
    run_all_tasks,
    save_feature_set_best_model_packages,
    save_global_model_packages,
    save_results_to_excel,
)
from .plotting import generate_plots, generate_top_feature_distribution_analysis, generate_wrong_case_analysis
from .utils import ensure_directories, set_publication_plot_style, setup_logging


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the lower-limb lymphedema radiomics/morphology modeling pipeline.",
    )
    parser.add_argument(
        "-i",
        "--input-excel",
        default=None,
        help="Input Excel table containing labels, center IDs, radiomics features, and morphology features.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default=None,
        help="Directory for result workbooks, plots, and model packages. Defaults to ./outputs.",
    )
    parser.add_argument(
        "--run-sensitivity-analysis",
        action="store_true",
        help="Also run the sensitivity analysis without diagnostics features.",
    )
    parser.add_argument(
        "--show-figures",
        action="store_true",
        help="Display matplotlib figures in addition to saving them.",
    )
    return parser.parse_args(argv)


def _sync_runtime_config(input_excel=None, output_dir=None, run_sensitivity_analysis=None, show_figures=None):
    if input_excel:
        _config.EXCEL_FILE = os.path.abspath(input_excel)

    if output_dir:
        _config.OUTPUT_DIR = os.path.abspath(output_dir)

    _config.SAVE_XLSX = os.path.join(
        _config.OUTPUT_DIR,
        "binary_classification_center134_7_2_1_center2_external_no_radiomics_shape.xlsx",
    )
    _config.PLOT_DIR = os.path.join(_config.OUTPUT_DIR, "plots")
    _config.SHAP_DIR = os.path.join(_config.PLOT_DIR, "shap")
    _config.ROC_DIR = os.path.join(_config.PLOT_DIR, "roc")
    _config.CAL_DIR = os.path.join(_config.PLOT_DIR, "calibration")
    _config.DCA_DIR = os.path.join(_config.PLOT_DIR, "dca")
    _config.TOP_FEATURE_DISTRIBUTION_DIR = os.path.join(_config.OUTPUT_DIR, "top_feature_distributions")
    _config.WRONG_CASE_ANALYSIS_DIR = os.path.join(_config.OUTPUT_DIR, "wrong_case_analysis")
    _config.WRONG_CASE_FIGURE_DIR = os.path.join(_config.WRONG_CASE_ANALYSIS_DIR, "wrong_case_figures")
    _config.MODEL_PACKAGE_DIR = os.path.join(_config.OUTPUT_DIR, "model_packages")

    if run_sensitivity_analysis is not None:
        _config.RUN_SENSITIVITY_ANALYSIS = bool(run_sensitivity_analysis)
    if show_figures is not None:
        _config.SHOW_FIGURES = bool(show_figures)

    names = [
        "EXCEL_FILE",
        "OUTPUT_DIR",
        "SAVE_XLSX",
        "PLOT_DIR",
        "SHAP_DIR",
        "ROC_DIR",
        "CAL_DIR",
        "DCA_DIR",
        "TOP_FEATURE_DISTRIBUTION_DIR",
        "WRONG_CASE_ANALYSIS_DIR",
        "WRONG_CASE_FIGURE_DIR",
        "MODEL_PACKAGE_DIR",
        "RUN_SENSITIVITY_ANALYSIS",
        "SHOW_FIGURES",
    ]
    modules = [_utils, _data, _cv, _plotting, _pipeline, _sys.modules[__name__]]
    for module in modules:
        for name in names:
            if hasattr(_config, name):
                setattr(module, name, getattr(_config, name))


def main(argv=None):
    args = parse_args(argv)
    _sync_runtime_config(
        input_excel=args.input_excel,
        output_dir=args.output_dir,
        run_sensitivity_analysis=args.run_sensitivity_analysis,
        show_figures=args.show_figures,
    )

    if not EXCEL_FILE:
        raise SystemExit("Input Excel file is required. Pass --input-excel or set RADIOMICS_INPUT_EXCEL.")
    if not os.path.exists(EXCEL_FILE):
        raise FileNotFoundError(f"Input Excel file does not exist: {EXCEL_FILE}")

    ensure_directories()
    setup_logging()
    set_publication_plot_style()

    logging.info("输出目录：%s", OUTPUT_DIR)
    logging.info("Excel 输出文件：%s", SAVE_XLSX)
    logging.info("图形输出目录：%s", PLOT_DIR)
    logging.info("ROC 图目录：%s", ROC_DIR)
    logging.info("图形：Global_best SHAP Top10；ROC/DCA 按 Internal 和 External 分开绘制，并在每张图内比较 R、M、M+R。")
    logging.info("校准曲线：保留 M+R 的 Internal/External 合并图，并新增 External test 中 R、M、M+R 三特征集同图；使用 quantile binning；CALIBRATION_BINS = 6；不绘制 95%% CI 误差线。")
    logging.info("生成 ROC 图，不保存日志文件。")
    logging.info("Excel 输出 sheet: metrics_global_best, wrong_global_best, split_samples, candidate_5fold_cv_all_models, feature_selection_features, used_features_by_task, all_cases_predictions")
    logging.info("metrics_global_best 包含 Global_best 和三种特征集各自 Feature_set_best 的指标。")
    logging.info("candidate_5fold_cv_all_models 保存每个特征集 × 每个模型的完整 5 折 CV mean±sd 结果。")
    logging.info("feature_selection_features 保存每次 CV 折和最终 Train+Validation 筛选后剩余的特征名称。")
    logging.info("used_features_by_task 保存每个任务最终模型实际使用的特征名称。")
    logging.info("all_cases_predictions 保存每个最终模型在 Train/Validation/Internal/External 所有病例上的预测值、真实标签和预测标签。")
    logging.info("新增 Top feature distributions 输出目录：%s", TOP_FEATURE_DISTRIBUTION_DIR)
    logging.info("新增 Wrong-case analysis 输出目录：%s", WRONG_CASE_ANALYSIS_DIR)
    logging.info("模型封装输出目录：%s", MODEL_PACKAGE_DIR)
    logging.info("metrics_global_best 同时包含 5折CV mean±sd 结果，以及 Train/Validation/Internal/External 各标签数量。")
    logging.info("wrong_global_best 包含三种特征组合的错误病例，且不重复输出全局最优特征集。")
    logging.info("split_samples 报告每个任务的具体样本划分列表，用于复现。")
    logging.info("删除 Radiomics shape/shape2D 特征：%s", REMOVE_RADIOMICS_SHAPE_FEATURES)
    logging.info("候选模型选择：Train + Validation 上 %d 折交叉验证 mean，AUC -> ACC -> PPV。", CV_N_SPLITS)
    logging.info("运行敏感性分析并合并到同几个 sheet：%s", RUN_SENSITIVITY_ANALYSIS)
    logging.info("SHAP 可用：%s", HAS_SHAP)
    logging.info("SciPy 可用：%s", HAS_SCIPY)
    logging.info("nibabel 可用：%s", HAS_NIBABEL)
    logging.info("XGBoost 可用：%s", HAS_XGB)
    if HAS_XGB:
        logging.info("XGBoost 版本：%s", xgboost.__version__)
        logging.info("XGBoost 路径：%s", xgboost.__file__)
    else:
        logging.info("XGBoost 导入失败原因：%s", XGB_IMPORT_ERROR)
    logging.info("StratifiedGroupKFold 可用：%s", HAS_STRATIFIED_GROUP_KFOLD)
    logging.info("当前 Python：%s", sys.executable)

    df = read_and_validate_data(EXCEL_FILE)
    (
        radiomics_cols,
        morph_cols,
        diagnostics_cols,
        radiomics_no_diagnostics_cols,
        radiomics_shape_cols,
        radiomics_all_detected_cols,
    ) = identify_feature_columns(df)

    logging.info("=" * 100)
    logging.info("开始主分析：Primary_no_radiomics_shape")
    logging.info("=" * 100)
    primary_results = run_all_tasks(
        df=df,
        radiomics_cols=radiomics_cols,
        morph_cols=morph_cols,
        analysis_label="Primary_no_radiomics_shape",
    )

    all_metrics = [primary_results["final_metrics_df"]]
    all_wrong = [primary_results["final_wrong_df"]]
    all_split_samples = [primary_results["split_samples_df"]]
    all_candidate_cv = [primary_results["candidate_cv_df"]]
    all_feature_selection = [primary_results["feature_selection_df"]]
    all_used_features = [primary_results["used_features_df"]]
    all_case_predictions = [primary_results["all_case_predictions_df"]]

    all_global_artifacts = dict(primary_results["global_artifacts"])
    all_feature_set_artifacts = dict(primary_results["feature_set_artifacts"])

    if RUN_SENSITIVITY_ANALYSIS:
        logging.info("=" * 100)
        logging.info("开始敏感性分析：Sensitivity_no_shape_no_diagnostics")
        logging.info("删除 diagnostics 特征数：%d", len(diagnostics_cols))
        logging.info("=" * 100)

        sensitivity_results = run_all_tasks(
            df=df,
            radiomics_cols=radiomics_no_diagnostics_cols,
            morph_cols=morph_cols,
            analysis_label="Sensitivity_no_shape_no_diagnostics",
        )

        all_metrics.append(sensitivity_results["final_metrics_df"])
        all_wrong.append(sensitivity_results["final_wrong_df"])
        all_split_samples.append(sensitivity_results["split_samples_df"])
        all_candidate_cv.append(sensitivity_results["candidate_cv_df"])
        all_feature_selection.append(sensitivity_results["feature_selection_df"])
        all_used_features.append(sensitivity_results["used_features_df"])
        all_case_predictions.append(sensitivity_results["all_case_predictions_df"])

        for k, v in sensitivity_results["global_artifacts"].items():
            all_global_artifacts[f"Sensitivity_{k}"] = v
        for k, v in sensitivity_results["feature_set_artifacts"].items():
            all_feature_set_artifacts[f"Sensitivity_{k}"] = v

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    wrong_df = pd.concat(all_wrong, ignore_index=True)
    split_samples_df = pd.concat(all_split_samples, ignore_index=True)
    candidate_cv_df = pd.concat(all_candidate_cv, ignore_index=True)
    feature_selection_df = pd.concat(all_feature_selection, ignore_index=True)
    used_features_df = pd.concat(all_used_features, ignore_index=True)
    all_case_predictions_df = pd.concat(all_case_predictions, ignore_index=True)

    save_results_to_excel(
        metrics_df=metrics_df,
        wrong_df=wrong_df,
        split_samples_df=split_samples_df,
        candidate_cv_df=candidate_cv_df,
        feature_selection_df=feature_selection_df,
        used_features_df=used_features_df,
        all_case_predictions_df=all_case_predictions_df,
    )

    # 新增：保存每个二分类任务的 Global_best 训练模型 package。
    # 不影响原有 Excel、绘图、错误病例分析等输出。
    save_global_model_packages(all_global_artifacts)

    # 新增：保存每个任务下 R、M、M+R 三种特征集各自的 Feature_set_best 训练模型 package。
    # 这样即使 Global_best 是 M+R，也会额外封装 R 和 M 各自最优模型。
    save_feature_set_best_model_packages(all_feature_set_artifacts)

    generate_plots(
        global_artifacts=all_global_artifacts,
        feature_set_artifacts=all_feature_set_artifacts,
    )

    generate_top_feature_distribution_analysis(
        feature_set_artifacts=all_feature_set_artifacts,
        radiomics_cols=radiomics_cols,
        morph_cols=morph_cols,
    )

    generate_wrong_case_analysis(
        feature_set_artifacts=all_feature_set_artifacts,
    )

    logging.info("全部流程完成。")
    logging.info("结果 Excel：%s", SAVE_XLSX)
    logging.info("图形目录：%s", PLOT_DIR)


if __name__ == "__main__":
    main()

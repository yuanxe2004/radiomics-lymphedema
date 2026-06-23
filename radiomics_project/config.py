
# 1. 基础配置与导入


import os
import re
import sys
import logging
import warnings
from dataclasses import dataclass

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

os.environ["MPLBACKEND"] = "Agg"

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_STRATIFIED_GROUP_KFOLD = True
except Exception:
    StratifiedGroupKFold = None
    HAS_STRATIFIED_GROUP_KFOLD = False

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix

try:
    from scipy.stats import shapiro, ttest_ind, mannwhitneyu
    HAS_SCIPY = True
except Exception as e:
    HAS_SCIPY = False
    shapiro = None
    ttest_ind = None
    mannwhitneyu = None
    SCIPY_IMPORT_ERROR = repr(e)
else:
    SCIPY_IMPORT_ERROR = ""

try:
    import nibabel as nib
    HAS_NIBABEL = True
except Exception as e:
    HAS_NIBABEL = False
    nib = None
    NIBABEL_IMPORT_ERROR = repr(e)
else:
    NIBABEL_IMPORT_ERROR = ""

try:
    import shap
    HAS_SHAP = True
except Exception:
    HAS_SHAP = False
    shap = None

try:
    import xgboost
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception as e:
    HAS_XGB = False
    XGBClassifier = None
    XGB_IMPORT_ERROR = repr(e)
else:
    XGB_IMPORT_ERROR = ""




# 2. 路径与全局参数


EXCEL_FILE = r"C:\Users\ALIENWARE\OneDrive\work\分类\final_deduplicated_with_new_morph-2.xlsx"

OUTPUT_DIR = r"C:\Users\ALIENWARE\Desktop\train_mould\code\radiomics\drawing\output\2-8"
SAVE_XLSX = os.path.join(
    OUTPUT_DIR,
    "binary_classification_center134_7_2_1_center2_external_no_radiomics_shape.xlsx"
)

PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
SHAP_DIR = os.path.join(PLOT_DIR, "shap")
ROC_DIR = os.path.join(PLOT_DIR, "roc")
CAL_DIR = os.path.join(PLOT_DIR, "calibration")
DCA_DIR = os.path.join(PLOT_DIR, "dca")

# 新增分析输出目录。为避免改变既有输出逻辑，新增结果均保存到 OUTPUT_DIR 下的独立子文件夹。
TOP_FEATURE_DISTRIBUTION_DIR = os.path.join(OUTPUT_DIR, "top_feature_distributions")
WRONG_CASE_ANALYSIS_DIR = os.path.join(OUTPUT_DIR, "wrong_case_analysis")
WRONG_CASE_FIGURE_DIR = os.path.join(WRONG_CASE_ANALYSIS_DIR, "wrong_case_figures")

# 模型封装输出目录：保存最终 Global_best 模型，以及 R / M / M+R 各自 Feature_set_best 模型，供新 Excel 直接推理使用。
MODEL_PACKAGE_DIR = os.path.join(OUTPUT_DIR, "model_packages")

LABEL_COL = "标签"
ID_COL = "序号"
SIDE_COL = "肢体"
CENTER_COL = "对应中心"

INTERNAL_CENTER = "中心134"
EXTERNAL_CENTER = "中心2"

RANDOM_STATE = 255

TRAIN_RATIO = 0.6
VAL_RATIO = 0.2
INTERNAL_TEST_RATIO = 0.2
USE_GROUP_SPLIT = True

REMOVE_RADIOMICS_SHAPE_FEATURES = True
RUN_SENSITIVITY_ANALYSIS = False

N_BOOTSTRAP = 2000
CI_ALPHA = 0.05

CALIBRATION_BINS = 6
DCA_THRESHOLDS = np.linspace(0.05, 0.80, 76)
DCA_YMIN = -0.1
SHOW_FIGURES = False

SHAP_MAX_BACKGROUND = 80
SHAP_MAX_EXPLAIN = 200
SHAP_KERNEL_NSAMPLES = 150

VARIANCE_THRESHOLD_VALUE = 1e-8
UNIVARIATE_TOP_K = 100
PEARSON_THRESHOLD = 0.90
LASSO_C = 0.10

# 旧版的 AUC/ACC 容忍阈值已不再用于模型选择，保留变量仅便于追踪历史配置。
AUC_TIE_THRESHOLD = 0.01
ACC_TIE_THRESHOLD = 0.01

# 候选模型选择使用 Train + Validation 上的 K 折交叉验证均值。
CV_N_SPLITS = 5

TASKS = [(1, 2), (2, 3)]
FEATURE_SETS = ["Radiomics_only", "Morphology_only", "Combined"]
BASE_MODEL_NAMES = ["LogisticRegression", "RandomForest", "SVM"]

MORPH_FEATURES = [
    "leg_middle1of3_volume_mm3",
    "underskin_middle1of3_volume_mm3",
    "leg_max_csa_mm2",
    "underskin_area_at_leg_max_csa_mm2",
    "underskin_middle1of3_top_bottom_area_ratio",
    "underskin_muscle_middle1of3_volume_ratio",
    "underskin_bone_middle1of3_volume_ratio",
    "underskin_muscle_max_csa_ratio_at_legmax",
    "underskin_bone_max_csa_ratio_at_legmax",
]



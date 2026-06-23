
# 3. 日志、绘图与基础工具


def ensure_directories():
    for d in [
        OUTPUT_DIR, PLOT_DIR, SHAP_DIR, ROC_DIR, CAL_DIR, DCA_DIR,
        TOP_FEATURE_DISTRIBUTION_DIR, WRONG_CASE_ANALYSIS_DIR, WRONG_CASE_FIGURE_DIR, MODEL_PACKAGE_DIR,
    ]:
        os.makedirs(d, exist_ok=True)


def setup_logging():
    """
    不保存日志文件，只向控制台输出。
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)


def set_publication_plot_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 8,
        "figure.titlesize": 12,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "savefig.dpi": 600,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.grid": False,
    })


PUBLICATION_COLORS = [
    "#009E73",  # Radiomics_only / R: green
    "#E69F00",  # Morphology_only / M: orange
    "#D62728",  # Combined / M+R: red
    "#0072B2",
    "#CC79A7",
    "#56B4E9",
]
PUBLICATION_LINESTYLES = ["-", "--", "-.", ":"]

FEATURE_SET_COLOR_MAP = {
    "Radiomics_only": "#009E73",
    "R": "#009E73",
    "Morphology_only": "#E69F00",
    "M": "#E69F00",
    "Combined": "#D62728",
    "R+M": "#D62728",
    "M+R": "#D62728",
}

SCI_BOX_COLORS = [
    "#0B7670",  # muted teal green
    "#C47E2E",  # muted ochre orange
]
SCI_BOX_POINT_COLOR = "#5B5A5A"

SHAP_POSITIVE_COLOR = "#D10000"
SHAP_NEGATIVE_COLOR = "#0950CB"
SCI_SHAP_CMAP = matplotlib.colors.LinearSegmentedColormap.from_list(
    "sci_shap_blue_white_red",
    ["#0950CB", "#EEEEEE", "#D10000"],
)


def get_feature_set_color(label, default_index=0):
    label_text = str(label).strip()

    if label_text in FEATURE_SET_COLOR_MAP:
        return FEATURE_SET_COLOR_MAP[label_text]

    # 优先识别组合特征集，避免 "M+R" 中的 "R" 被误判为 Radiomics。
    if "M+R" in label_text or "R+M" in label_text or "Combined" in label_text:
        return FEATURE_SET_COLOR_MAP["Combined"]
    if "Radiomics_only" in label_text or re.search(r"(^|\s)R($|\s|\()", label_text):
        return FEATURE_SET_COLOR_MAP["Radiomics_only"]
    if "Morphology_only" in label_text or re.search(r"(^|\s)M($|\s|\()", label_text):
        return FEATURE_SET_COLOR_MAP["Morphology_only"]

    return PUBLICATION_COLORS[default_index % len(PUBLICATION_COLORS)]


def save_or_show_plot(save_path):
    plt.tight_layout()
    plt.savefig(save_path, dpi=600, bbox_inches="tight")
    if SHOW_FIGURES:
        plt.show()
    plt.close()


def apply_axis_style(ax):
    ax.grid(False)
    ax.tick_params(axis="both", length=4, width=0.9)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)


def safe_filename(text):
    text = str(text)
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", text)
    text = text.replace("+", "plus")
    return text.strip("_")


def short_feature_set_name(feature_set_name):
    mapping = {
        "Radiomics_only": "R",
        "Morphology_only": "M",
        "Combined": "M+R",
    }
    return mapping.get(feature_set_name, feature_set_name)


def feature_family(feature_name, radio_cols, morph_cols):
    if feature_name in morph_cols:
        return "Morphology"
    if feature_name in radio_cols:
        return "Radiomics"
    return "Unknown"


@dataclass
class TaskData:
    task_df: pd.DataFrame
    internal_df: pd.DataFrame
    external_df: pd.DataFrame
    train_df: pd.DataFrame
    val_df: pd.DataFrame
    internal_test_df: pd.DataFrame

    x_train_all: pd.DataFrame
    x_val_all: pd.DataFrame
    x_internal_test_all: pd.DataFrame
    x_external_all: pd.DataFrame

    y_train: np.ndarray
    y_val: np.ndarray
    y_internal_test: np.ndarray
    y_external: np.ndarray

    valid_radio_cols: list
    valid_morph_cols: list
    dropped_radio_cols: list
    dropped_morph_cols: list
    split_method: str


@dataclass
class FeatureSelectionResult:
    selected_cols: list
    selected_radio_cols: list
    selected_morph_cols: list
    counts: dict
    radio_result: dict
    morph_result: dict


@dataclass
class FinalArtifact:
    analysis_label: str
    task_name: str
    feature_set_name: str
    model_name: str
    result_tag: str
    positive_label: int
    labels_sorted: list
    selected_cols: list
    final_model: object
    x_trainval_selected: pd.DataFrame
    y_trainval: np.ndarray
    trainval_df: pd.DataFrame
    x_internal_selected: pd.DataFrame
    internal_df: pd.DataFrame
    internal_true: np.ndarray
    internal_true_bin: np.ndarray
    internal_pred: np.ndarray
    internal_prob: np.ndarray
    x_external_selected: pd.DataFrame
    external_df: pd.DataFrame
    external_true: np.ndarray
    external_true_bin: np.ndarray
    external_pred: np.ndarray
    external_prob: np.ndarray


def task_name_from_classes(class_a, class_b):
    return f"{class_a}v{class_b}"


def to_binary_labels(y, positive_label):
    return (np.asarray(y) == positive_label).astype(int)


def format_float(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def format_metric_with_ci(point, low, high):
    if pd.isna(point) or pd.isna(low) or pd.isna(high):
        return ""
    return f"{point:.3f} (95% CI: {low:.3f}–{high:.3f})"


def format_mean_sd(mean_value, sd_value):
    if pd.isna(mean_value):
        return ""
    if pd.isna(sd_value):
        return f"{mean_value:.3f} ± NA"
    return f"{mean_value:.3f} ± {sd_value:.3f}"


def clean_table_for_export(df):
    df = df.copy()
    raw_cols = [
        "ACC_raw", "AUC_raw", "PPV_raw", "NPV_raw",
        "Sensitivity_raw", "Specificity_raw", "F1_raw", "Brier_raw",
        "Calibration_slope_raw", "Calibration_intercept_raw",
        "Balanced_ACC_raw",
    ]
    return df.drop(columns=[c for c in raw_cols if c in df.columns], errors="ignore")


def make_experiment_id(analysis_label, result_tag, task_name, feature_set_name, model_name, data_split):
    return "|".join([
        str(analysis_label),
        str(result_tag),
        str(task_name),
        str(feature_set_name),
        str(model_name),
        str(data_split),
    ])


def label_counts_for_df(df, labels_sorted):
    vc = df[LABEL_COL].astype(int).value_counts().to_dict()
    return {int(label): int(vc.get(int(label), 0)) for label in labels_sorted}


def format_label_counts(count_dict):
    return "; ".join([f"{label}:{count}" for label, count in count_dict.items()])


def make_label_count_fields(data, class_a, class_b):
    labels_sorted = sorted([class_a, class_b])
    split_specs = [
        ("训练集", data.train_df),
        ("验证集", data.val_df),
        ("内部测试集", data.internal_test_df),
        ("外部测试集", data.external_df),
    ]

    fields = {}
    for split_name, split_df in split_specs:
        counts = label_counts_for_df(split_df, labels_sorted)
        fields[f"{split_name}标签分布"] = format_label_counts(counts)
        for label in labels_sorted:
            fields[f"{split_name}_标签{label}_数量"] = counts[int(label)]
    return fields



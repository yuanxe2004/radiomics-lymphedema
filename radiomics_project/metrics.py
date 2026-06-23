"""Binary classification metrics and model-selection helpers."""

from .config import *
from .utils import format_float, format_metric_with_ci, to_binary_labels


def calibration_slope_intercept(y_true_bin, y_prob):
    y_true_bin = np.asarray(y_true_bin).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)

    if len(np.unique(y_true_bin)) < 2:
        return np.nan, np.nan

    p = np.clip(y_prob, 1e-6, 1 - 1e-6)
    logit_p = np.log(p / (1 - p)).reshape(-1, 1)

    try:
        lr = LogisticRegression(
            penalty="l2",
            C=1e6,
            solver="lbfgs",
            max_iter=5000,
        )
        lr.fit(logit_p, y_true_bin)
        slope = float(lr.coef_.ravel()[0])
        intercept = float(lr.intercept_.ravel()[0])
        return slope, intercept
    except Exception:
        return np.nan, np.nan


def calc_binary_metrics_point(y_true, y_pred, y_prob, positive_label, labels_sorted=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)

    if labels_sorted is None:
        labels_sorted = sorted(np.unique(np.concatenate([y_true, y_pred])))

    acc_value = accuracy_score(y_true, y_pred)
    y_true_bin = to_binary_labels(y_true, positive_label)

    try:
        auc_value = roc_auc_score(y_true_bin, y_prob)
    except Exception:
        auc_value = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan

    ppv = tp / (tp + fp) if pd.notna(tp) and (tp + fp) > 0 else np.nan
    npv = tn / (tn + fn) if pd.notna(tn) and (tn + fn) > 0 else np.nan
    sensitivity = tp / (tp + fn) if pd.notna(tp) and (tp + fn) > 0 else np.nan
    specificity = tn / (tn + fp) if pd.notna(tn) and (tn + fp) > 0 else np.nan
    f1 = 2 * tp / (2 * tp + fp + fn) if pd.notna(tp) and (2 * tp + fp + fn) > 0 else np.nan
    balanced_acc = np.nanmean([sensitivity, specificity])
    brier = np.mean((y_true_bin - y_prob) ** 2) if len(y_prob) > 0 else np.nan
    cal_slope, cal_intercept = calibration_slope_intercept(y_true_bin, y_prob)

    return {
        "ACC_raw": acc_value,
        "AUC_raw": auc_value,
        "PPV_raw": ppv,
        "NPV_raw": npv,
        "Sensitivity_raw": sensitivity,
        "Specificity_raw": specificity,
        "F1_raw": f1,
        "Balanced_ACC_raw": balanced_acc,
        "Brier_raw": brier,
        "Calibration_slope_raw": cal_slope,
        "Calibration_intercept_raw": cal_intercept,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
    }


def bootstrap_ci_binary_metrics(y_true, y_pred, y_prob, positive_label, labels_sorted, n_bootstrap=2000, alpha=0.05, random_state=42):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob, dtype=float)

    n = len(y_true)
    rng = np.random.default_rng(random_state)

    boot_values = {
        "ACC": [],
        "AUC": [],
        "PPV": [],
        "NPV": [],
        "Sensitivity": [],
        "Specificity": [],
        "F1": [],
        "Balanced_ACC": [],
        "Brier": [],
        "Calibration_slope": [],
        "Calibration_intercept": [],
    }

    mapping = {
        "ACC": "ACC_raw",
        "AUC": "AUC_raw",
        "PPV": "PPV_raw",
        "NPV": "NPV_raw",
        "Sensitivity": "Sensitivity_raw",
        "Specificity": "Specificity_raw",
        "F1": "F1_raw",
        "Balanced_ACC": "Balanced_ACC_raw",
        "Brier": "Brier_raw",
        "Calibration_slope": "Calibration_slope_raw",
        "Calibration_intercept": "Calibration_intercept_raw",
    }

    for _ in range(n_bootstrap):
        idx = rng.choice(np.arange(n), size=n, replace=True)
        m = calc_binary_metrics_point(
            y_true=y_true[idx],
            y_pred=y_pred[idx],
            y_prob=y_prob[idx],
            positive_label=positive_label,
            labels_sorted=labels_sorted,
        )
        for out_key, raw_key in mapping.items():
            if pd.notna(m[raw_key]):
                boot_values[out_key].append(m[raw_key])

    ci_result = {}
    for k, values in boot_values.items():
        values = np.asarray(values, dtype=float)
        if len(values) == 0:
            ci_result[f"{k}_low"] = np.nan
            ci_result[f"{k}_high"] = np.nan
        else:
            ci_result[f"{k}_low"] = np.percentile(values, 100 * alpha / 2)
            ci_result[f"{k}_high"] = np.percentile(values, 100 * (1 - alpha / 2))

    ci_result["Bootstrap_times"] = n_bootstrap
    ci_result["Bootstrap_valid_times_AUC"] = len(boot_values["AUC"])
    ci_result["Bootstrap_valid_times_CalSlope"] = len(boot_values["Calibration_slope"])
    return ci_result


def calc_binary_metrics_with_ci(y_true, y_pred, y_prob, positive_label, labels_sorted, n_bootstrap=N_BOOTSTRAP, alpha=CI_ALPHA, random_state=RANDOM_STATE):
    point = calc_binary_metrics_point(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        positive_label=positive_label,
        labels_sorted=labels_sorted,
    )

    ci = bootstrap_ci_binary_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        positive_label=positive_label,
        labels_sorted=labels_sorted,
        n_bootstrap=n_bootstrap,
        alpha=alpha,
        random_state=random_state,
    )

    return {
        "ACC_raw": point["ACC_raw"],
        "AUC_raw": point["AUC_raw"],
        "PPV_raw": point["PPV_raw"],
        "NPV_raw": point["NPV_raw"],
        "Sensitivity_raw": point["Sensitivity_raw"],
        "Specificity_raw": point["Specificity_raw"],
        "F1_raw": point["F1_raw"],
        "Balanced_ACC_raw": point["Balanced_ACC_raw"],
        "Brier_raw": point["Brier_raw"],
        "Calibration_slope_raw": point["Calibration_slope_raw"],
        "Calibration_intercept_raw": point["Calibration_intercept_raw"],
        "ACC": format_metric_with_ci(point["ACC_raw"], ci["ACC_low"], ci["ACC_high"]),
        "AUC": format_metric_with_ci(point["AUC_raw"], ci["AUC_low"], ci["AUC_high"]),
        "PPV": format_metric_with_ci(point["PPV_raw"], ci["PPV_low"], ci["PPV_high"]),
        "NPV": format_metric_with_ci(point["NPV_raw"], ci["NPV_low"], ci["NPV_high"]),
        "Sensitivity": format_metric_with_ci(point["Sensitivity_raw"], ci["Sensitivity_low"], ci["Sensitivity_high"]),
        "Specificity": format_metric_with_ci(point["Specificity_raw"], ci["Specificity_low"], ci["Specificity_high"]),
        "F1": format_metric_with_ci(point["F1_raw"], ci["F1_low"], ci["F1_high"]),
        "Balanced_ACC": format_metric_with_ci(point["Balanced_ACC_raw"], ci["Balanced_ACC_low"], ci["Balanced_ACC_high"]),
        "Brier": format_metric_with_ci(point["Brier_raw"], ci["Brier_low"], ci["Brier_high"]),
        "Calibration_slope": format_metric_with_ci(point["Calibration_slope_raw"], ci["Calibration_slope_low"], ci["Calibration_slope_high"]),
        "Calibration_intercept": format_metric_with_ci(point["Calibration_intercept_raw"], ci["Calibration_intercept_low"], ci["Calibration_intercept_high"]),
        "TN": point["TN"],
        "FP": point["FP"],
        "FN": point["FN"],
        "TP": point["TP"],
        "Bootstrap_times": ci["Bootstrap_times"],
        "Bootstrap_valid_times_AUC": ci["Bootstrap_valid_times_AUC"],
        "Bootstrap_valid_times_CalSlope": ci["Bootstrap_valid_times_CalSlope"],
    }


def calc_binary_metrics_without_ci(y_true, y_pred, y_prob, positive_label, labels_sorted):
    point = calc_binary_metrics_point(
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        positive_label=positive_label,
        labels_sorted=labels_sorted,
    )
    return {
        "ACC_raw": point["ACC_raw"],
        "AUC_raw": point["AUC_raw"],
        "PPV_raw": point["PPV_raw"],
        "NPV_raw": point["NPV_raw"],
        "Sensitivity_raw": point["Sensitivity_raw"],
        "Specificity_raw": point["Specificity_raw"],
        "F1_raw": point["F1_raw"],
        "Balanced_ACC_raw": point["Balanced_ACC_raw"],
        "Brier_raw": point["Brier_raw"],
        "Calibration_slope_raw": point["Calibration_slope_raw"],
        "Calibration_intercept_raw": point["Calibration_intercept_raw"],
        "ACC": format_float(point["ACC_raw"]),
        "AUC": format_float(point["AUC_raw"]),
        "PPV": format_float(point["PPV_raw"]),
        "NPV": format_float(point["NPV_raw"]),
        "Sensitivity": format_float(point["Sensitivity_raw"]),
        "Specificity": format_float(point["Specificity_raw"]),
        "F1": format_float(point["F1_raw"]),
        "Balanced_ACC": format_float(point["Balanced_ACC_raw"]),
        "Brier": format_float(point["Brier_raw"]),
        "Calibration_slope": format_float(point["Calibration_slope_raw"]),
        "Calibration_intercept": format_float(point["Calibration_intercept_raw"]),
        "TN": point["TN"],
        "FP": point["FP"],
        "FN": point["FN"],
        "TP": point["TP"],
    }


def is_better_model(candidate, current_best):
    if current_best is None:
        return True

    cand_auc = -np.inf if pd.isna(candidate["AUC_raw"]) else candidate["AUC_raw"]
    best_auc = -np.inf if pd.isna(current_best["AUC_raw"]) else current_best["AUC_raw"]

    cand_acc = -np.inf if pd.isna(candidate["ACC_raw"]) else candidate["ACC_raw"]
    best_acc = -np.inf if pd.isna(current_best["ACC_raw"]) else current_best["ACC_raw"]

    cand_ppv = -np.inf if pd.isna(candidate["PPV_raw"]) else candidate["PPV_raw"]
    best_ppv = -np.inf if pd.isna(current_best["PPV_raw"]) else current_best["PPV_raw"]

    if cand_auc != best_auc:
        return cand_auc > best_auc
    if cand_acc != best_acc:
        return cand_acc > best_acc
    return cand_ppv > best_ppv

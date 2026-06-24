# -*- coding: utf-8 -*-
"""Configuration objects and YAML loading helpers."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class PipelineConfig:
    # I/O
    excel_file: str = r"C:\Users\ALIENWARE\OneDrive\work\分类\final_deduplicated_with_new_morph-2.xlsx"
    output_dir: str = r"C:\Users\ALIENWARE\Desktop\train_mould\code\radiomics\drawing\output\3-0"

    # Columns
    label_col: str = "标签"
    center_col: str = "对应中心"
    id_col: str = "序号"

    # Center split
    internal_center_value: str = "中心134"
    external_center_value: str = "中心2"

    # Task
    labels: List[int] = field(default_factory=lambda: [1, 2, 3])
    label_names: Dict[int, str] = field(default_factory=lambda: {1: "Stage I", 2: "Stage II", 3: "Stage III"})
    task_name: str = "Stage_1_vs_2_vs_3"

    # Randomness and validation
    random_state: int = 255
    n_splits: int = 5
    n_bootstrap: int = 1000
    bootstrap_seed: int = 2026
    use_group_split: bool = True
    use_stratified_split: bool = True

    # Feature detection
    remove_radiomics_shape_features: bool = True
    auto_detect_morphology_by_keywords: bool = True
    morphology_features: List[str] = field(default_factory=lambda: [
        "muscle_volume_ratio", "bone_volume_ratio", "underskin_volume_ratio",
        "subcutaneous_volume_ratio", "muscle_CSA_ratio", "bone_CSA_ratio",
        "underskin_CSA_ratio", "subcutaneous_CSA_ratio", "volume_ratio",
        "CSA_ratio", "muscle_ratio", "bone_ratio", "underskin_ratio",
    ])
    radiomics_keywords: List[str] = field(default_factory=lambda: [
        "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm",
        "wavelet", "log-sigma", "log_sigma", "logarithm", "exponential",
        "gradient", "lbp", "square", "squareroot", "original_",
    ])
    morphology_keywords: List[str] = field(default_factory=lambda: [
        "muscle", "bone", "underskin", "subcutaneous", "skin", "fat",
        "volume_ratio", "csa", "area", "ratio", "肌肉", "骨", "皮下", "脂肪", "面积", "体积", "比例",
    ])
    exclude_name_patterns: List[str] = field(default_factory=lambda: [
        "label", "标签", "center", "中心", "id", "序号", "编号", "patient", "患者",
        "limb", "肢", "side", "左右", "name", "姓名", "path", "路径", "file", "文件",
        "nii", "dcm", "dicom", "mask", "seg", "分割", "日期", "date",
    ])

    # Feature selection
    variance_threshold: float = 1e-8
    anova_top_k: int = 50
    pearson_threshold_radiomics: float = 0.90
    pearson_threshold_morphology: float = 0.90
    lasso_c: float = 0.05
    lasso_max_iter: int = 20000
    min_features_after_lasso_fallback: int = 10
    combined_final_pearson: bool = False

    # Models
    feature_sets: List[str] = field(default_factory=lambda: ["Radiomics_only", "Morphology_only", "Combined"])
    model_names: List[str] = field(default_factory=lambda: ["LR", "RF", "SVM"])
    use_xgboost_if_available: bool = True

    # Plot / explainability
    run_roc: bool = True
    run_dca: bool = True
    run_calibration: bool = True
    run_shap: bool = True
    shap_max_background: int = 80
    shap_max_cases: int = 150
    boxplot_top_n: int = 10
    max_excel_rows_per_sheet: int = 1_000_000

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _normalize_label_names(value: Dict[Any, Any]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    for k, v in value.items():
        try:
            out[int(k)] = str(v)
        except Exception:
            out[k] = str(v)
    return out


def load_config(path: Optional[str] = None) -> PipelineConfig:
    """Load config from YAML. If no path is given, return defaults."""
    cfg = PipelineConfig()
    if path is None:
        return cfg
    if yaml is None:
        raise ImportError("PyYAML is required for YAML config loading. Install with: pip install pyyaml")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    data = {k: v for k, v in data.items() if hasattr(cfg, k)}
    if "label_names" in data and isinstance(data["label_names"], dict):
        data["label_names"] = _normalize_label_names(data["label_names"])
    return PipelineConfig(**{**cfg.to_dict(), **data})


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

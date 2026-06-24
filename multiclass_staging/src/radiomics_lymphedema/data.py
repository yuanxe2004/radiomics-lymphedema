# -*- coding: utf-8 -*-
"""Data loading and leakage-controlled splitting."""

from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import PipelineConfig
from .utils import normalize_center_value, print_table


def load_and_normalize_data(cfg: PipelineConfig) -> pd.DataFrame:
    df = pd.read_excel(cfg.excel_file)
    missing_cols = [c for c in [cfg.label_col, cfg.center_col, cfg.id_col] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in Excel: {missing_cols}")

    before_n = len(df)
    df = df.dropna(subset=[cfg.label_col]).copy()
    df[cfg.label_col] = pd.to_numeric(df[cfg.label_col], errors="coerce")
    df = df.dropna(subset=[cfg.label_col]).copy()
    df[cfg.label_col] = df[cfg.label_col].astype(int)
    df = df[df[cfg.label_col].isin(cfg.labels)].copy()
    df[cfg.center_col] = df[cfg.center_col].map(normalize_center_value)

    print(f"Loaded rows: {before_n}; rows after label cleaning and LABELS filter: {len(df)}")
    print_table("Label distribution", df[cfg.label_col].value_counts().sort_index())
    print_table("Center distribution", df[cfg.center_col].value_counts(dropna=False))
    print_table("Center x Label cross table", pd.crosstab(df[cfg.center_col], df[cfg.label_col]))

    internal_n = (df[cfg.center_col] == cfg.internal_center_value).sum()
    external_n = (df[cfg.center_col] == cfg.external_center_value).sum()
    if internal_n == 0:
        raise ValueError(f"No rows found for internal_center_value={cfg.internal_center_value!r}")
    if external_n == 0:
        raise ValueError(f"No rows found for external_center_value={cfg.external_center_value!r}")
    return df.reset_index(drop=True)


def _group_stratum_table(df: pd.DataFrame, cfg: PipelineConfig) -> pd.DataFrame:
    def label_pattern(s: pd.Series) -> str:
        vals = sorted(pd.Series(s).dropna().astype(int).unique().tolist())
        return "_".join(map(str, vals))

    tab = (
        df.groupby(cfg.id_col, dropna=False)[cfg.label_col]
        .agg(stratum_pattern=label_pattern, dominant_label=lambda s: int(pd.Series(s).mode().iloc[0]))
        .reset_index()
    )
    pattern_counts = tab["stratum_pattern"].value_counts()
    if len(pattern_counts) > 0 and pattern_counts.min() >= 2:
        tab["stratum"] = tab["stratum_pattern"]
    else:
        dominant_counts = tab["dominant_label"].value_counts()
        if len(dominant_counts) > 0 and dominant_counts.min() >= 2:
            tab["stratum"] = tab["dominant_label"].astype(str)
        else:
            tab["stratum"] = "all"
    return tab


def group_aware_holdout_split(
    df: pd.DataFrame,
    cfg: PipelineConfig,
    test_size: float,
    random_state: int,
    name: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not cfg.use_group_split or cfg.id_col not in df.columns:
        stratify = df[cfg.label_col] if cfg.use_stratified_split else None
        tr, te = train_test_split(df, test_size=test_size, stratify=stratify, random_state=random_state)
        return tr.copy(), te.copy()

    groups_tab = _group_stratum_table(df, cfg)
    stratify = None
    if cfg.use_stratified_split:
        counts = groups_tab["stratum"].value_counts()
        if len(counts) > 1 and counts.min() >= 2:
            stratify = groups_tab["stratum"]

    train_groups, test_groups = train_test_split(
        groups_tab[cfg.id_col], test_size=test_size, stratify=stratify, random_state=random_state
    )
    train_groups = set(train_groups.tolist())
    test_groups = set(test_groups.tolist())
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(f"Group leakage detected during {name}: {len(overlap)} overlapping groups")
    tr = df[df[cfg.id_col].isin(train_groups)].copy()
    te = df[df[cfg.id_col].isin(test_groups)].copy()
    return tr, te


def center_and_internal_split(cfg: PipelineConfig, df: pd.DataFrame):
    internal = df[df[cfg.center_col] == cfg.internal_center_value].copy()
    external = df[df[cfg.center_col] == cfg.external_center_value].copy()

    train_df, temp_df = group_aware_holdout_split(
        internal, cfg, test_size=0.40, random_state=cfg.random_state, name="internal train/temp split"
    )
    val_df, internal_test_df = group_aware_holdout_split(
        temp_df, cfg, test_size=0.50, random_state=cfg.random_state + 1, name="internal validation/test split"
    )

    train_df = train_df.copy(); train_df["__split__"] = "train"
    val_df = val_df.copy(); val_df["__split__"] = "validation"
    internal_test_df = internal_test_df.copy(); internal_test_df["__split__"] = "internal_test"
    external = external.copy(); external["__split__"] = "external_test"

    if cfg.use_group_split and cfg.id_col in df.columns:
        split_groups = {
            "train": set(train_df[cfg.id_col].astype(str)),
            "validation": set(val_df[cfg.id_col].astype(str)),
            "internal_test": set(internal_test_df[cfg.id_col].astype(str)),
            "external_test": set(external[cfg.id_col].astype(str)),
        }
        for a in split_groups:
            for b in split_groups:
                if a >= b:
                    continue
                if a != "external_test" and b != "external_test":
                    inter = split_groups[a].intersection(split_groups[b])
                    if inter:
                        raise RuntimeError(f"Group leakage between {a} and {b}: {len(inter)} groups")

    all_split = pd.concat([train_df, val_df, internal_test_df, external])
    print_table("Split x Label cross table", pd.crosstab(all_split["__split__"], all_split[cfg.label_col]))
    return train_df, val_df, internal_test_df, external

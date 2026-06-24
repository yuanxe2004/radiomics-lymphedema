# -*- coding: utf-8 -*-
"""General utilities."""

from __future__ import annotations

import re
import time
from typing import Any, List

import numpy as np
import pandas as pd


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def safe_sheet_name(name: str) -> str:
    return re.sub(r"[\\/*?:\[\]]", "_", str(name))[:31]


def safe_filename(name: Any) -> str:
    name = re.sub(r"[^0-9a-zA-Z_\-\.]+", "_", str(name))
    return name.strip("_")


def normalize_center_value(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def has_any_keyword(name: str, keywords: List[str]) -> bool:
    s = str(name).lower()
    return any(k.lower() in s for k in keywords)


def as_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(index=df.index)
    out = df.loc[:, cols].copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def format_mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(sd):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {sd:.{digits}f}"


def format_ci(mean: float, low: float, high: float, digits: int = 3) -> str:
    if pd.isna(mean):
        return "NA"
    if pd.isna(low) or pd.isna(high):
        return f"{mean:.{digits}f} (NA-NA)"
    return f"{mean:.{digits}f} ({low:.{digits}f}-{high:.{digits}f})"


def write_sheet_safe(writer: pd.ExcelWriter, df: pd.DataFrame, sheet_name: str, max_rows: int = 1_000_000) -> None:
    sheet_name = safe_sheet_name(sheet_name)
    if df is None:
        df = pd.DataFrame()
    if len(df) <= max_rows:
        df.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        import math
        n_parts = math.ceil(len(df) / max_rows)
        for i in range(n_parts):
            part = df.iloc[i * max_rows:(i + 1) * max_rows]
            part.to_excel(writer, sheet_name=safe_sheet_name(f"{sheet_name}_{i + 1}"), index=False)


def print_table(title: str, obj: Any) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(obj)

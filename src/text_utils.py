from __future__ import annotations

import re
import pandas as pd


def contains_keyword(series: pd.Series, keyword: str) -> pd.Series:
    if not keyword:
        return pd.Series([True] * len(series), index=series.index)
    terms = [t.strip().lower() for t in re.split(r"[,|\s]+", keyword) if t.strip()]
    if not terms:
        return pd.Series([True] * len(series), index=series.index)
    text = series.fillna("").astype(str).str.lower()
    mask = pd.Series([False] * len(series), index=series.index)
    for term in terms:
        mask = mask | text.str.contains(re.escape(term), regex=True, na=False)
    return mask


def fmt_num(x, digits: int = 0) -> str:
    try:
        v = float(x)
    except Exception:
        return "0"
    if digits == 0:
        return f"{v:,.0f}"
    return f"{v:,.{digits}f}"

from __future__ import annotations

from pathlib import Path
import pandas as pd


def read_stats_table(stats_dir: str | Path, stem: str) -> pd.DataFrame:
    base = Path(stats_dir)
    for ext in ["parquet", "csv"]:
        p = base / f"{stem}.{ext}"
        if p.exists():
            if ext == "parquet":
                return pd.read_parquet(p)
            return pd.read_csv(p, dtype="object", low_memory=False, encoding="utf-8-sig")
    return pd.DataFrame()


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["hhi", "entropy", "entropy_norm", "effective_issue_count", "top1_share", "top3_share", "share", "ci_low", "ci_high", "weight_sum", "total_weight", "n_issue_rows", "n_boot"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def filter_date_overlap(df: pd.DataFrame, start, end) -> pd.DataFrame:
    if df.empty or "window_start" not in df.columns or "window_end" not in df.columns:
        return df
    ws = pd.to_datetime(df["window_start"], errors="coerce")
    we = pd.to_datetime(df["window_end"], errors="coerce")
    # Overall tables have empty window fields; keep them.
    overall = ws.isna() & we.isna()
    return df[overall | ((ws <= pd.to_datetime(end)) & (we >= pd.to_datetime(start)))].copy()

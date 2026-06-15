from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def load_metadata(data_dir: str | Path) -> dict:
    p = Path(data_dir) / "metadata.json"
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_table(data_dir: str | Path, stem: str) -> pd.DataFrame:
    d = Path(data_dir)
    pq = d / f"{stem}.parquet"
    csv = d / f"{stem}.csv"
    if pq.exists():
        return pd.read_parquet(pq)
    if csv.exists():
        return pd.read_csv(csv, dtype="object", low_memory=False, encoding="utf-8-sig")
    return pd.DataFrame()


def coerce_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    return out

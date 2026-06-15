from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Optional, Sequence

import numpy as np
import pandas as pd

FILTER_MODES = ("strict", "balanced", "exploratory")
DEFAULT_BAD_TYPES = {"other", "unknown", "noise_or_nonissue", "media_meta", "general_news_digest"}
DEFAULT_EXCLUDE_STATUS = {"exclude_candidate", "excluded", "exclude"}
REVIEW_NEEDED_STATUS = {"review_needed", "확인 필요"}
GOOD_REVIEW_STATUS = {"publish_ready", "바로 설명 가능", "evidence_ready", "reviewed_final"}
GOOD_REPORT_USE = {"evidence_ready", "reviewed_final", "publish_ready", "triage_only_publish_ready"}


def safe_str(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except Exception:
        pass
    return str(x)


def table_path(data_dir: str | Path, table_name: str) -> Path:
    base = Path(data_dir)
    for ext in ("parquet", "csv"):
        p = base / f"{table_name}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"Cannot find {table_name}.parquet or {table_name}.csv under {base}")


def read_table(data_dir: str | Path, table_name: str) -> pd.DataFrame:
    p = table_path(data_dir, table_name)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p, dtype="object", low_memory=False, encoding="utf-8-sig")


def write_table(df: pd.DataFrame, out_dir: str | Path, stem: str, fmt: str = "csv") -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    if fmt in {"parquet", "both"}:
        try:
            p = out / f"{stem}.parquet"
            df.to_parquet(p, index=False)
            written.append(str(p))
        except Exception:
            if fmt == "parquet":
                # Parquet may be unavailable in minimal runtime; fall back to CSV instead of failing.
                p = out / f"{stem}.csv"
                df.to_csv(p, index=False, encoding="utf-8-sig")
                written.append(str(p))
    if fmt in {"csv", "both"} or not written:
        p = out / f"{stem}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        written.append(str(p))
    return written


def first_col(df: pd.DataFrame, candidates: Sequence[str], default: str = "") -> pd.Series:
    for c in candidates:
        if c in df.columns:
            return df[c].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce").fillna(default).astype(float)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def clamp01(value: float) -> float:
    if pd.isna(value):
        return np.nan
    return float(np.clip(value, 0.0, 1.0))


def standardize_issue_briefs(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize issue_briefs from v31/v31.1/v31.2 into a common stats schema."""
    out = pd.DataFrame(index=df.index)
    out["period_issue_id"] = first_col(df, ["period_issue_id"])
    out["stable_issue_id"] = first_col(df, ["stable_issue_id"])
    out["issue_key"] = first_col(df, ["stable_issue_id"])
    out["issue_key"] = out["issue_key"].mask(out["issue_key"].astype(str).eq(""), first_col(df, ["period_issue_id"]))
    issue_name = first_col(df, ["issue_name", "issue_name_clean_final", "issue_name_clean_rule", "issue_label_auto", "stable_issue_label_auto"])
    out["issue_name"] = issue_name.map(lambda x: safe_str(x).strip())
    out["issue_key"] = out["issue_key"].mask(out["issue_key"].astype(str).eq(""), out["issue_name"])
    out["issue_key"] = out["issue_key"].mask(out["issue_key"].astype(str).eq(""), pd.Series([f"row_{i}" for i in range(len(out))], index=df.index))

    out["candidate"] = first_col(df, ["candidate"])
    out["source"] = first_col(df, ["source"])
    out["window_type"] = first_col(df, ["window_type"])
    out["window_start"] = first_col(df, ["window_start"]).astype(str).str[:10]
    out["window_end"] = first_col(df, ["window_end"]).astype(str).str[:10]
    out["rank_in_window"] = numeric(df, "rank_in_window", 999).astype(int)
    out["issue_type"] = first_col(df, ["issue_type", "issue_type_final", "issue_type_rule", "frame_rule"], "unknown").astype(str)
    out["review_status"] = first_col(df, ["review_status", "review_status_final", "review_status_rule"], "unknown").astype(str)
    out["report_use_level"] = first_col(df, ["report_use_level", "report_use_level_rule"], "unknown").astype(str)
    out["evidence_source_method"] = first_col(df, ["evidence_source_method"], "unknown").astype(str)
    out["source_label"] = first_col(df, ["source_label"], "")
    out["candidate_label"] = first_col(df, ["candidate_label"], "")
    out["summary_1line"] = first_col(df, ["summary_1line", "issue_summary_1line_final", "issue_summary_1line_rule"], "")

    out["doc_count_weighted"] = numeric(df, "doc_count_weighted", 0)
    # fallback names for comment/view/like
    out["comment_count"] = numeric(df, "comment_count", 0)
    if out["comment_count"].sum() == 0:
        out["comment_count"] = numeric(df, "comment_count_weighted", 0)
    out["view_count"] = numeric(df, "view_count", 0)
    if out["view_count"].sum() == 0:
        out["view_count"] = numeric(df, "view_count_weighted", 0)
    out["like_count"] = numeric(df, "like_count", 0)
    if out["like_count"].sum() == 0:
        out["like_count"] = numeric(df, "like_count_weighted", 0)
    out["row_weight"] = 1.0
    out["weight_doc_count_weighted"] = out["doc_count_weighted"].clip(lower=0)
    out["weight_row_count"] = 1.0
    return out


def apply_filter_mode(df: pd.DataFrame, filter_mode: str = "balanced", include_other: bool | None = None) -> pd.DataFrame:
    if filter_mode not in FILTER_MODES:
        raise ValueError(f"filter_mode must be one of {FILTER_MODES}; got {filter_mode}")
    out = df.copy()
    out = out[~out["review_status"].isin(DEFAULT_EXCLUDE_STATUS)].copy()
    if filter_mode == "strict":
        # Strict mode for report-like interpretation: remove low-confidence and generic types.
        # Some current dashboard builds mark otherwise usable rows as report_use_level=triage_only,
        # so review_status is the primary quality gate here.
        out = out[~out["review_status"].isin(REVIEW_NEEDED_STATUS)].copy()
        out = out[~out["issue_type"].isin(DEFAULT_BAD_TYPES)].copy()
        if "report_use_level" in out.columns:
            out = out[~out["report_use_level"].isin(DEFAULT_EXCLUDE_STATUS)].copy()
    elif filter_mode == "balanced":
        out = out[~out["issue_type"].isin(DEFAULT_BAD_TYPES)].copy()
        # review_needed remains; this is the default because automatic issue type errors should be visible.
    elif filter_mode == "exploratory":
        pass
    if include_other is not None:
        if not include_other:
            out = out[~out["issue_type"].isin(DEFAULT_BAD_TYPES)].copy()
    return out.reset_index(drop=True)


def get_weight(df: pd.DataFrame, weight_metric: str = "doc_count_weighted") -> pd.Series:
    if weight_metric == "row_count":
        return pd.Series([1.0] * len(df), index=df.index)
    if weight_metric in df.columns:
        return pd.to_numeric(df[weight_metric], errors="coerce").fillna(0).clip(lower=0).astype(float)
    # fallback
    return pd.to_numeric(df.get("doc_count_weighted", pd.Series([1.0] * len(df), index=df.index)), errors="coerce").fillna(0).clip(lower=0).astype(float)


def concentration_for_group(group: pd.DataFrame, weight_metric: str = "doc_count_weighted") -> dict:
    if group.empty:
        return {}
    tmp = group.copy()
    tmp["_w"] = get_weight(tmp, weight_metric)
    # Consolidate by issue key within the group so repeated rows don't become separate "issues".
    by = tmp.groupby(["issue_key", "issue_name", "issue_type"], dropna=False)["_w"].sum().reset_index()
    by = by[by["_w"] > 0].copy()
    if by.empty:
        return {"issue_count": 0, "total_weight": 0.0, "hhi": np.nan, "entropy": np.nan, "entropy_norm": np.nan, "effective_issue_count": np.nan, "top1_share": np.nan, "top3_share": np.nan, "top1_issue_name": "", "top1_issue_type": ""}
    total = float(by["_w"].sum())
    p = by["_w"].to_numpy(dtype=float) / total
    k = len(p)
    hhi = float(np.sum(p ** 2))
    entropy = float(-np.sum(p * np.log(p + 1e-15)))
    entropy_norm = clamp01(float(entropy / np.log(k)) if k > 1 else 0.0)
    eff = float(1.0 / hhi) if hhi > 0 else np.nan
    by["share"] = p
    by = by.sort_values("share", ascending=False)
    top1 = by.iloc[0]
    top3_share = clamp01(float(by.head(3)["share"].sum()))
    return {
        "issue_count": int(k),
        "total_weight": total,
        "hhi": hhi,
        "hhi_10000": hhi * 10000,
        "entropy": entropy,
        "entropy_norm": entropy_norm,
        "effective_issue_count": eff,
        "top1_share": clamp01(float(by.iloc[0]["share"])),
        "top3_share": top3_share,
        "top1_issue_name": safe_str(top1["issue_name"]),
        "top1_issue_type": safe_str(top1["issue_type"]),
    }


def build_concentration_metrics(df: pd.DataFrame, weight_metric: str = "doc_count_weighted", filter_mode: str = "balanced", include_all_sources: bool = True) -> pd.DataFrame:
    """Vectorized concentration metrics for source-level and all-source groups."""
    work = apply_filter_mode(standardize_issue_briefs(df) if "issue_key" not in df.columns else df, filter_mode=filter_mode)
    if work.empty:
        return pd.DataFrame()

    def compute_for_scope(x: pd.DataFrame, source_scope: str) -> pd.DataFrame:
        tmp = x.copy()
        tmp["_w"] = get_weight(tmp, weight_metric)
        tmp = tmp[tmp["_w"] > 0].copy()
        if tmp.empty:
            return pd.DataFrame()
        group_cols = ["candidate", "source", "window_type", "window_start", "window_end"]
        issue_cols = group_cols + ["issue_key", "issue_name", "issue_type"]
        issue = tmp.groupby(issue_cols, dropna=False, as_index=False)["_w"].sum()
        total = issue.groupby(group_cols, dropna=False)["_w"].sum().rename("total_weight").reset_index()
        issue = issue.merge(total, on=group_cols, how="left")
        issue["p"] = issue["_w"] / issue["total_weight"]
        issue["p2"] = issue["p"] ** 2
        issue["plogp"] = -issue["p"] * np.log(issue["p"] + 1e-15)
        agg = issue.groupby(group_cols, dropna=False).agg(
            issue_count=("issue_key", "nunique"),
            total_weight=("total_weight", "first"),
            hhi=("p2", "sum"),
            entropy=("plogp", "sum"),
        ).reset_index()
        agg["hhi_10000"] = agg["hhi"] * 10000
        agg["entropy_norm"] = np.where(agg["issue_count"] > 1, agg["entropy"] / np.log(agg["issue_count"]), 0.0)
        agg["entropy_norm"] = agg["entropy_norm"].clip(lower=0, upper=1)
        agg["effective_issue_count"] = np.where(agg["hhi"] > 0, 1.0 / agg["hhi"], np.nan)
        issue_sorted = issue.sort_values(group_cols + ["p"], ascending=[True, True, True, True, True, False])
        top1 = issue_sorted.groupby(group_cols, dropna=False).head(1)[group_cols + ["issue_name", "issue_type", "p"]]
        top1 = top1.rename(columns={"issue_name": "top1_issue_name", "issue_type": "top1_issue_type", "p": "top1_share"})
        top3 = issue_sorted.groupby(group_cols, dropna=False).head(3).groupby(group_cols, dropna=False)["p"].sum().rename("top3_share").reset_index()
        top1["top1_share"] = top1["top1_share"].clip(lower=0, upper=1)
        top3["top3_share"] = top3["top3_share"].clip(lower=0, upper=1)
        out = agg.merge(top1, on=group_cols, how="left").merge(top3, on=group_cols, how="left")
        out["weight_metric"] = weight_metric
        out["filter_mode"] = filter_mode
        out["source_scope"] = source_scope
        return out

    frames = [compute_for_scope(work, "source")]
    if include_all_sources:
        tmp = work.copy()
        tmp["source"] = "ALL"
        frames.append(compute_for_scope(tmp, "all_sources"))
    return pd.concat([f for f in frames if not f.empty], ignore_index=True) if frames else pd.DataFrame()


def issue_type_share_once(group: pd.DataFrame, weight_metric: str = "doc_count_weighted") -> pd.DataFrame:
    tmp = group.copy()
    tmp["_w"] = get_weight(tmp, weight_metric)
    tmp = tmp[tmp["_w"] > 0]
    if tmp.empty:
        return pd.DataFrame(columns=["issue_type", "weight_sum", "share"])
    total = float(tmp["_w"].sum())
    out = tmp.groupby("issue_type", dropna=False)["_w"].sum().reset_index(name="weight_sum")
    out["share"] = out["weight_sum"] / total if total > 0 else np.nan
    out["total_weight"] = total
    return out


def bootstrap_issue_type_shares(
    group: pd.DataFrame,
    weight_metric: str = "doc_count_weighted",
    n_boot: int = 500,
    seed: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:
    group = group.reset_index(drop=True).copy()
    n = len(group)
    base = issue_type_share_once(group, weight_metric)
    if base.empty:
        return base.assign(ci_low=np.nan, ci_high=np.nan, n_issue_rows=n, n_boot=0)
    issue_types = sorted(base["issue_type"].astype(str).unique())
    # If too few rows, do not pretend the CI is meaningful. Return degenerate CI.
    if n < 2 or n_boot <= 0:
        out = base.copy()
        out["ci_low"] = out["share"]
        out["ci_high"] = out["share"]
        out["n_issue_rows"] = n
        out["n_boot"] = 0
        return out
    rng = np.random.default_rng(seed)
    shares = {t: [] for t in issue_types}
    idx = np.arange(n)
    for _ in range(n_boot):
        sample_idx = rng.choice(idx, size=n, replace=True)
        sample = group.iloc[sample_idx]
        ss = issue_type_share_once(sample, weight_metric)
        mapping = dict(zip(ss["issue_type"].astype(str), ss["share"].astype(float)))
        for t in issue_types:
            shares[t].append(mapping.get(t, 0.0))
    low_q = (1.0 - ci) / 2.0
    high_q = 1.0 - low_q
    out = base.copy()
    ci_low = []
    ci_high = []
    for t in out["issue_type"].astype(str):
        arr = np.array(shares.get(t, [np.nan]), dtype=float)
        ci_low.append(float(np.nanquantile(arr, low_q)))
        ci_high.append(float(np.nanquantile(arr, high_q)))
    out["ci_low"] = ci_low
    out["ci_high"] = ci_high
    out["n_issue_rows"] = n
    out["n_boot"] = n_boot
    return out


def build_candidate_issue_type_share_ci(
    df: pd.DataFrame,
    weight_metric: str = "doc_count_weighted",
    filter_mode: str = "balanced",
    n_boot: int = 500,
    seed: int = 42,
    include_all_sources: bool = True,
    min_issue_rows: int = 3,
    min_total_weight: float = 1.0,
) -> pd.DataFrame:
    """Build candidate issue-type shares with bootstrap CIs.

    For runtime stability on the full dashboard data, confidence intervals are
    computed for overall candidate-level distributions and source-specific
    overall distributions, not for every individual window. Window-level
    concentration metrics are handled separately by build_concentration_metrics.
    """
    work = apply_filter_mode(standardize_issue_briefs(df) if "issue_key" not in df.columns else df, filter_mode=filter_mode)
    rows: list[pd.DataFrame] = []
    group_id = 0

    def add_group(g: pd.DataFrame, base: dict):
        nonlocal group_id
        group_id += 1
        total_weight = float(get_weight(g, weight_metric).sum())
        if len(g) < min_issue_rows or total_weight < min_total_weight:
            return
        res = bootstrap_issue_type_shares(g, weight_metric=weight_metric, n_boot=n_boot, seed=seed + group_id)
        for k, v in base.items():
            res[k] = v
        res["source_scope"] = base.get("source_scope", "source")
        res["weight_metric"] = weight_metric
        res["filter_mode"] = filter_mode
        rows.append(res)

    # Candidate-level distribution across all selected sources.
    if include_all_sources:
        for cand, g in work.groupby("candidate", dropna=False):
            add_group(g, {
                "candidate": cand,
                "source": "ALL",
                "window_type": "ALL",
                "window_start": "",
                "window_end": "",
                "source_scope": "all_sources",
                "period_scope": "overall",
            })

    # Candidate-source distribution.
    for (cand, src), g in work.groupby(["candidate", "source"], dropna=False):
        add_group(g, {
            "candidate": cand,
            "source": src,
            "window_type": "ALL",
            "window_start": "",
            "window_end": "",
            "source_scope": "source",
            "period_scope": "overall",
        })

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def candidate_share_difference(share_df: pd.DataFrame, cand_a: str = "JWO", cand_b: str = "OSH") -> pd.DataFrame:
    if share_df.empty:
        return pd.DataFrame()
    desired_keys = ["source", "source_scope", "period_scope", "window_type", "window_start", "window_end", "issue_type", "weight_metric", "filter_mode"]
    a = share_df[share_df["candidate"].eq(cand_a)].copy()
    b = share_df[share_df["candidate"].eq(cand_b)].copy()
    if a.empty or b.empty:
        return pd.DataFrame()
    key_cols = [c for c in desired_keys if c in a.columns and c in b.columns]
    if not key_cols or "issue_type" not in key_cols:
        return pd.DataFrame()
    merged = a.merge(b, on=key_cols, how="inner", suffixes=(f"_{cand_a}", f"_{cand_b}"))
    if merged.empty:
        return merged
    merged["share_diff"] = merged[f"share_{cand_a}"] - merged[f"share_{cand_b}"]
    # Conservative interval by subtracting marginal CI endpoints; not a paired bootstrap.
    merged["share_diff_ci_low_conservative"] = merged[f"ci_low_{cand_a}"] - merged[f"ci_high_{cand_b}"]
    merged["share_diff_ci_high_conservative"] = merged[f"ci_high_{cand_a}"] - merged[f"ci_low_{cand_b}"]
    merged["candidate_a"] = cand_a
    merged["candidate_b"] = cand_b
    return merged


def build_candidate_issue_type_share_ci_grouped(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    weight_metric: str = "doc_count_weighted",
    filter_mode: str = "balanced",
    n_boot: int = 500,
    seed: int = 42,
    min_issue_rows: int = 3,
    min_total_weight: float = 1.0,
) -> pd.DataFrame:
    """Bootstrap issue-type shares for configurable grouping.

    This is preferred for dashboards because bootstrapping every single window can be
    expensive. For interactive use, filter the data first, then group by candidate/source
    or candidate only.
    """
    work = apply_filter_mode(standardize_issue_briefs(df) if "issue_key" not in df.columns else df, filter_mode=filter_mode)
    rows: list[pd.DataFrame] = []
    group_id = 0
    for keys, g in work.groupby(list(group_cols), dropna=False):
        group_id += 1
        if not isinstance(keys, tuple):
            keys = (keys,)
        total_weight = float(get_weight(g, weight_metric).sum())
        if len(g) < min_issue_rows or total_weight < min_total_weight:
            continue
        res = bootstrap_issue_type_shares(g, weight_metric=weight_metric, n_boot=n_boot, seed=seed + group_id)
        for c, k in zip(group_cols, keys):
            res[c] = k
        res["weight_metric"] = weight_metric
        res["filter_mode"] = filter_mode
        res["grouping"] = "+".join(group_cols)
        rows.append(res)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

# Fast vectorized override for full dashboard data.
def build_concentration_metrics(df: pd.DataFrame, weight_metric: str = "doc_count_weighted", filter_mode: str = "balanced", include_all_sources: bool = True) -> pd.DataFrame:  # type: ignore[override]
    work = apply_filter_mode(standardize_issue_briefs(df) if "issue_key" not in df.columns else df, filter_mode=filter_mode)
    work = work.copy()
    work["_w"] = get_weight(work, weight_metric)
    if work.empty:
        return pd.DataFrame()
    frames = []

    def compute(sub: pd.DataFrame, source_scope: str) -> pd.DataFrame:
        if sub.empty:
            return pd.DataFrame()
        group_cols = ["candidate", "source", "window_type", "window_start", "window_end"]
        issue_cols = group_cols + ["issue_key", "issue_name", "issue_type"]
        collapsed = sub.groupby(issue_cols, dropna=False, as_index=False)["_w"].sum()
        # If zero weights appear in a group, they simply do not contribute. Groups with total 0 are removed.
        total = collapsed.groupby(group_cols, dropna=False)["_w"].transform("sum")
        collapsed = collapsed[total > 0].copy()
        if collapsed.empty:
            return pd.DataFrame()
        total = collapsed.groupby(group_cols, dropna=False)["_w"].transform("sum")
        collapsed["share"] = collapsed["_w"] / total
        collapsed["p2"] = collapsed["share"] ** 2
        collapsed["plogp"] = collapsed["share"] * np.log(collapsed["share"].clip(lower=1e-15))
        agg = collapsed.groupby(group_cols, dropna=False).agg(
            issue_count=("issue_key", "nunique"),
            total_weight=("_w", "sum"),
            hhi=("p2", "sum"),
            entropy=("plogp", lambda s: -float(s.sum())),
        ).reset_index()
        agg["hhi_10000"] = agg["hhi"] * 10000
        agg["entropy_norm"] = agg.apply(lambda r: float(r["entropy"] / np.log(r["issue_count"])) if r["issue_count"] > 1 else 0.0, axis=1)
        agg["entropy_norm"] = agg["entropy_norm"].clip(lower=0, upper=1)
        agg["effective_issue_count"] = 1.0 / agg["hhi"].replace(0, np.nan)
        # top1/top3
        collapsed = collapsed.sort_values(group_cols + ["share"], ascending=[True, True, True, True, True, False])
        collapsed["rank"] = collapsed.groupby(group_cols, dropna=False).cumcount() + 1
        top1 = collapsed[collapsed["rank"].eq(1)][group_cols + ["share", "issue_name", "issue_type"]].rename(columns={"share": "top1_share", "issue_name": "top1_issue_name", "issue_type": "top1_issue_type"})
        top3 = collapsed[collapsed["rank"].le(3)].groupby(group_cols, dropna=False)["share"].sum().reset_index(name="top3_share")
        top1["top1_share"] = top1["top1_share"].clip(lower=0, upper=1)
        top3["top3_share"] = top3["top3_share"].clip(lower=0, upper=1)
        out = agg.merge(top1, on=group_cols, how="left").merge(top3, on=group_cols, how="left")
        out["source_scope"] = source_scope
        out["weight_metric"] = weight_metric
        out["filter_mode"] = filter_mode
        return out

    frames.append(compute(work, "source"))
    if include_all_sources:
        allw = work.copy()
        allw["source"] = "ALL"
        frames.append(compute(allw, "all_sources"))
    return pd.concat([f for f in frames if f is not None and not f.empty], ignore_index=True) if frames else pd.DataFrame()

# Robust override: support both overall and window-specific share tables.
def candidate_share_difference(share_df: pd.DataFrame, cand_a: str = "JWO", cand_b: str = "OSH") -> pd.DataFrame:  # type: ignore[override]
    if share_df.empty:
        return pd.DataFrame()
    base_keys = ["source", "source_scope", "window_type", "window_start", "window_end", "period_scope", "issue_type", "weight_metric", "filter_mode"]
    key_cols = [c for c in base_keys if c in share_df.columns]
    a = share_df[share_df["candidate"].eq(cand_a)].copy()
    b = share_df[share_df["candidate"].eq(cand_b)].copy()
    if a.empty or b.empty or not key_cols:
        return pd.DataFrame()
    merged = a.merge(b, on=key_cols, how="inner", suffixes=(f"_{cand_a}", f"_{cand_b}"))
    if merged.empty:
        return merged
    merged["share_diff"] = merged[f"share_{cand_a}"] - merged[f"share_{cand_b}"]
    if f"ci_low_{cand_a}" in merged.columns and f"ci_high_{cand_b}" in merged.columns:
        merged["share_diff_ci_low_conservative"] = merged[f"ci_low_{cand_a}"] - merged[f"ci_high_{cand_b}"]
        merged["share_diff_ci_high_conservative"] = merged[f"ci_high_{cand_a}"] - merged[f"ci_low_{cand_b}"]
    merged["candidate_a"] = cand_a
    merged["candidate_b"] = cand_b
    return merged

# Vectorized bootstrap override for large groups.
def bootstrap_issue_type_shares(
    group: pd.DataFrame,
    weight_metric: str = "doc_count_weighted",
    n_boot: int = 500,
    seed: int = 42,
    ci: float = 0.95,
) -> pd.DataFrame:  # type: ignore[override]
    group = group.reset_index(drop=True).copy()
    n = len(group)
    base = issue_type_share_once(group, weight_metric)
    if base.empty:
        return base.assign(ci_low=np.nan, ci_high=np.nan, n_issue_rows=n, n_boot=0)
    if n < 2 or n_boot <= 0:
        out = base.copy()
        out["ci_low"] = out["share"]
        out["ci_high"] = out["share"]
        out["n_issue_rows"] = n
        out["n_boot"] = 0
        return out
    weights = get_weight(group, weight_metric).to_numpy(dtype=float)
    if np.nansum(weights) <= 0:
        weights = np.ones(n, dtype=float)
    type_values = group["issue_type"].astype(str).to_numpy()
    issue_types = sorted(pd.unique(type_values).tolist())
    type_to_code = {t: i for i, t in enumerate(issue_types)}
    codes = np.array([type_to_code[t] for t in type_values], dtype=np.int16)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n), dtype=np.int32)
    sw = weights[idx]
    scodes = codes[idx]
    totals = sw.sum(axis=1)
    totals_safe = np.where(totals <= 0, 1.0, totals)
    low_q = (1.0 - ci) / 2.0
    high_q = 1.0 - low_q
    ci_map = {}
    for t, code in type_to_code.items():
        shares = np.where(scodes == code, sw, 0.0).sum(axis=1) / totals_safe
        ci_map[t] = (float(np.quantile(shares, low_q)), float(np.quantile(shares, high_q)))
    out = base.copy()
    out["ci_low"] = out["issue_type"].astype(str).map(lambda t: ci_map.get(t, (np.nan, np.nan))[0])
    out["ci_high"] = out["issue_type"].astype(str).map(lambda t: ci_map.get(t, (np.nan, np.nan))[1])
    out["n_issue_rows"] = n
    out["n_boot"] = n_boot
    return out

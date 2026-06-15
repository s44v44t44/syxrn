from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

try:
    from scipy.stats import chi2_contingency
except Exception:  # pragma: no cover
    chi2_contingency = None

BAD_TYPES = {"other", "unknown", "noise_or_nonissue", "media_meta", "general_news_digest"}
EXCLUDE_STATUS = {"exclude_candidate", "excluded", "exclude"}
REVIEW_NEEDED_STATUS = {"review_needed", "확인 필요"}
GENERIC_TERMS = {
    "정원오",
    "오세훈",
    "서울시장",
    "후보",
    "기사",
    "뉴스",
    "영상",
    "관련",
    "이슈",
    "공방",
    "논란",
    "서울",
    "시장",
    "민주당",
    "국민의힘",
    "했다",
    "대한",
    "위해",
    "이슈다",
    "이슈다.",
    "연결된",
    "후보가",
    "묶인",
    "둘러싼",
    "여러",
    "기타",
    "관한",
    "보도가",
    "맥락에서",
    "프레임의",
    "함께",
}


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
    raise FileNotFoundError(f"Cannot find {table_name}.parquet or .csv under {base}")


def read_table(data_dir: str | Path, table_name: str) -> pd.DataFrame:
    p = table_path(data_dir, table_name)
    if p.suffix.lower() == ".parquet":
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
                p = out / f"{stem}.csv"
                df.to_csv(p, index=False, encoding="utf-8-sig")
                written.append(str(p))
    if fmt in {"csv", "both"} or not written:
        p = out / f"{stem}.csv"
        df.to_csv(p, index=False, encoding="utf-8-sig")
        written.append(str(p))
    return written


def write_json(obj: dict, out_dir: str | Path, stem: str) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    p = out / f"{stem}.json"
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return p


def first_col(df: pd.DataFrame, names: Sequence[str], default: object = "") -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name].fillna(default)
    return pd.Series([default] * len(df), index=df.index)


def numeric_col(df: pd.DataFrame, name: str, default: float = 0.0) -> pd.Series:
    if name in df.columns:
        return pd.to_numeric(df[name], errors="coerce").fillna(default).astype(float)
    return pd.Series([default] * len(df), index=df.index, dtype=float)


def standardize_issue_briefs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["period_issue_id"] = first_col(df, ["period_issue_id"])
    out["stable_issue_id"] = first_col(df, ["stable_issue_id"])
    out["issue_key"] = out["stable_issue_id"].mask(out["stable_issue_id"].astype(str).eq(""), out["period_issue_id"])
    out["issue_name"] = first_col(
        df,
        ["issue_display_name", "issue_name", "issue_name_clean_final", "issue_name_clean_rule", "issue_label_auto", "stable_issue_label_auto"],
    ).map(lambda x: safe_str(x).strip())
    out["issue_key"] = out["issue_key"].mask(out["issue_key"].astype(str).eq(""), out["issue_name"])
    out["issue_key"] = out["issue_key"].mask(out["issue_key"].astype(str).eq(""), pd.Series([f"row_{i}" for i in range(len(out))], index=df.index))
    out["candidate"] = first_col(df, ["candidate"]).astype(str)
    out["candidate_label"] = first_col(df, ["candidate_label"], "").astype(str)
    out["source"] = first_col(df, ["source"]).astype(str)
    out["source_label"] = first_col(df, ["source_label"], "").astype(str)
    out["window_type"] = first_col(df, ["window_type"]).astype(str)
    out["window_start"] = first_col(df, ["window_start"]).astype(str).str[:10]
    out["window_end"] = first_col(df, ["window_end"]).astype(str).str[:10]
    out["rank_in_window"] = numeric_col(df, "rank_in_window", 999).astype(int)
    out["issue_type"] = first_col(df, ["issue_type", "issue_type_final", "issue_type_rule", "frame_rule"], "unknown").astype(str)
    out["review_status"] = first_col(df, ["review_status", "review_status_final", "review_status_rule"], "unknown").astype(str)
    out["report_use_level"] = first_col(df, ["report_use_level", "report_use_level_rule"], "unknown").astype(str)
    out["doc_count_weighted"] = numeric_col(df, "doc_count_weighted", 0).clip(lower=0)
    out["doc_count_raw"] = numeric_col(df, "doc_count_raw", 0).clip(lower=0)
    out["row_count"] = 1.0
    out["search_text"] = (
        first_col(df, ["search_text"], "").astype(str)
        + " "
        + first_col(df, ["issue_name", "issue_display_name", "issue_name_clean_rule", "issue_label_auto"], "").astype(str)
        + " "
        + first_col(df, ["top_terms"], "").astype(str)
        + " "
        + first_col(df, ["representative_title"], "").astype(str)
        + " "
        + first_col(df, ["evidence_titles", "evidence_titles_topN"], "").astype(str)
    ).str.lower()
    return out


def apply_filter_mode(df: pd.DataFrame, filter_mode: str = "balanced") -> pd.DataFrame:
    out = df.copy()
    out = out[~out["review_status"].isin(EXCLUDE_STATUS)].copy()
    if filter_mode == "strict":
        out = out[~out["review_status"].isin(REVIEW_NEEDED_STATUS)].copy()
        out = out[~out["issue_type"].isin(BAD_TYPES)].copy()
    elif filter_mode == "balanced":
        out = out[~out["issue_type"].isin(BAD_TYPES)].copy()
    elif filter_mode == "exploratory":
        pass
    else:
        raise ValueError("filter_mode must be strict, balanced, or exploratory")
    return out.reset_index(drop=True)


def collapse_distribution_unit(df: pd.DataFrame, distribution_unit: str, group_cols: list[str], weight_metric: str) -> pd.DataFrame:
    if distribution_unit == "window_issue":
        return df.copy()
    if distribution_unit != "stable_issue":
        raise ValueError("distribution_unit must be stable_issue or window_issue")

    # Windowed tables repeat the same stable issue across overlapping windows.
    # Collapsing keeps the distribution closer to unique issue units and reduces pseudo-replication.
    dedupe_cols = [c for c in group_cols + ["source", "issue_type", "issue_key"] if c in df.columns]
    if not dedupe_cols:
        return df.copy()
    work = df.copy()
    agg = {
        "period_issue_id": "first",
        "stable_issue_id": "first",
        "issue_name": "first",
        "review_status": "first",
        "report_use_level": "first",
        "doc_count_weighted": "max",
        "doc_count_raw": "max",
        "row_count": "first",
        "search_text": "first",
    }
    agg = {k: v for k, v in agg.items() if k in work.columns}
    collapsed = work.groupby(dedupe_cols, dropna=False, as_index=False).agg(agg)
    collapsed["row_count"] = 1.0
    return collapsed


def get_weight(df: pd.DataFrame, weight_metric: str = "row_count") -> pd.Series:
    if weight_metric == "row_count":
        return pd.Series([1.0] * len(df), index=df.index, dtype=float)
    if weight_metric in df.columns:
        return pd.to_numeric(df[weight_metric], errors="coerce").fillna(0).clip(lower=0).astype(float)
    return pd.to_numeric(df.get("doc_count_weighted", 0), errors="coerce").fillna(0).clip(lower=0).astype(float)


def cramers_v_from_chi2(chi2: float, n: float, r: int, c: int) -> float:
    denom = n * max(1, min(r - 1, c - 1))
    return float(math.sqrt(max(0.0, chi2) / denom)) if denom > 0 else np.nan


def contingency_expected(observed: np.ndarray) -> np.ndarray:
    total = observed.sum()
    if total <= 0:
        return np.zeros_like(observed, dtype=float)
    row_sum = observed.sum(axis=1, keepdims=True)
    col_sum = observed.sum(axis=0, keepdims=True)
    return row_sum @ col_sum / total


def chi_square_manual(observed: np.ndarray) -> tuple[float, int, np.ndarray]:
    expected = contingency_expected(observed)
    mask = expected > 0
    chi2 = float(((observed[mask] - expected[mask]) ** 2 / expected[mask]).sum()) if mask.any() else np.nan
    dof = max(0, (observed.shape[0] - 1) * (observed.shape[1] - 1))
    return chi2, dof, expected


def build_source_issue_type_tests(
    issue_briefs: pd.DataFrame,
    weight_metric: str = "row_count",
    filter_mode: str = "balanced",
    group_scope: str = "candidate_window_type",
    distribution_unit: str = "stable_issue",
    min_total_weight: float = 5.0,
    min_sources: int = 2,
    min_issue_types: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    work = standardize_issue_briefs(issue_briefs) if "issue_key" not in issue_briefs.columns else issue_briefs.copy()
    work = apply_filter_mode(work, filter_mode)
    if group_scope == "candidate_window_type":
        group_cols = ["candidate", "window_type"]
    elif group_scope == "candidate_overall":
        group_cols = ["candidate"]
    elif group_scope == "overall":
        work["all_group"] = "ALL"
        group_cols = ["all_group"]
    else:
        raise ValueError("group_scope must be candidate_window_type, candidate_overall, or overall")

    work = collapse_distribution_unit(work, distribution_unit, group_cols, weight_metric)
    work["_w"] = get_weight(work, weight_metric)
    work = work[work["_w"] > 0].copy()
    if work.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    test_rows: list[dict] = []
    residual_rows: list[dict] = []
    table_rows: list[dict] = []
    for keys, g in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {c: k for c, k in zip(group_cols, keys)}
        if "all_group" in base:
            base = {"candidate": "ALL", "window_type": "ALL"}
        elif "window_type" not in base:
            base["window_type"] = "ALL"

        tab_long = g.groupby(["source", "issue_type"], dropna=False)["_w"].sum().reset_index(name="observed")
        sources = sorted(tab_long["source"].astype(str).unique())
        types = sorted(tab_long["issue_type"].astype(str).unique())
        if len(sources) < min_sources or len(types) < min_issue_types:
            continue
        observed_df = tab_long.pivot(index="source", columns="issue_type", values="observed").fillna(0.0)
        observed_df = observed_df.reindex(index=sources, columns=types).fillna(0.0)
        observed = observed_df.to_numpy(dtype=float)
        total = float(observed.sum())
        if total < min_total_weight:
            continue
        if chi2_contingency is not None:
            try:
                chi2, p, dof, expected = chi2_contingency(observed, correction=False)
                chi2 = float(chi2)
                p = float(p)
                dof = int(dof)
            except Exception:
                chi2, dof, expected = chi_square_manual(observed)
                p = np.nan
        else:
            chi2, dof, expected = chi_square_manual(observed)
            p = np.nan

        cramer_v = cramers_v_from_chi2(chi2, total, len(sources), len(types))
        min_expected = float(np.min(expected)) if expected.size else np.nan
        low_expected_share = float((expected < 5).mean()) if expected.size else np.nan
        base_row = {
            **base,
            "filter_mode": filter_mode,
            "weight_metric": weight_metric,
            "distribution_unit": distribution_unit,
            "group_scope": group_scope,
            "n_sources": len(sources),
            "n_issue_types": len(types),
            "total_weight": total,
            "chi2": chi2,
            "dof": dof,
            "p_value": p,
            "cramers_v": cramer_v,
            "min_expected": min_expected,
            "low_expected_cell_share": low_expected_share,
            "method_note": "row_count_chi_square_diagnostic" if weight_metric == "row_count" else "weighted_exposure_diagnostic",
        }
        test_rows.append(base_row)
        for i, source in enumerate(sources):
            for j, issue_type in enumerate(types):
                obs = float(observed[i, j])
                exp = float(expected[i, j])
                resid = float((obs - exp) / math.sqrt(exp)) if exp > 0 else np.nan
                cell = {
                    **base,
                    "filter_mode": filter_mode,
                    "weight_metric": weight_metric,
                    "distribution_unit": distribution_unit,
                    "group_scope": group_scope,
                    "source": source,
                    "issue_type": issue_type,
                    "observed": obs,
                    "expected": exp,
                    "std_residual": resid,
                }
                residual_rows.append(cell)
                table_rows.append({k: v for k, v in cell.items() if k != "expected" and k != "std_residual"})
    return pd.DataFrame(test_rows), pd.DataFrame(residual_rows), pd.DataFrame(table_rows)


def standardize_base_docs(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["doc_id"] = first_col(df, ["doc_id", "document_id"])
    out["candidate"] = first_col(df, ["candidate"]).astype(str)
    out["source"] = first_col(df, ["source"]).astype(str)
    out["date"] = first_col(df, ["date"]).astype(str).str[:10]
    out["title"] = first_col(df, ["title", "representative_title"])
    search_text = first_col(df, ["search_text"], "")
    if search_text.astype(str).str.len().sum() == 0:
        search_text = (
            first_col(df, ["title"], "").astype(str)
            + " "
            + first_col(df, ["snippet"], "").astype(str)
            + " "
            + first_col(df, ["keywords"], "").astype(str)
            + " "
            + first_col(df, ["text_for_issue"], "").astype(str)
        )
    out["search_text"] = search_text.astype(str).str.lower()
    out["count_for_ranking"] = numeric_col(df, "count_for_ranking", 1.0).clip(lower=0)
    return out


def clean_keyword_token(token: object) -> str:
    text = safe_str(token).strip().lower().replace("_", " ")
    text = re.sub(r"^[\W_]+|[\W_]+$", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def extract_auto_keywords(issue_briefs: pd.DataFrame, max_keywords: int = 20, min_len: int = 2) -> list[str]:
    df = standardize_issue_briefs(issue_briefs) if "search_text" not in issue_briefs.columns else issue_briefs.copy()
    text_groups: list[tuple[str, float]] = []
    if "top_terms" in issue_briefs.columns:
        text_groups.extend((x, 2.0) for x in issue_briefs["top_terms"].fillna("").astype(str).tolist())
    fallback_texts: list[str] = []
    for col in ["issue_name", "representative_title"]:
        if col in issue_briefs.columns:
            fallback_texts.extend(issue_briefs[col].fillna("").astype(str).tolist())
    if "issue_name" in df.columns:
        fallback_texts.extend(df["issue_name"].fillna("").astype(str).tolist())
    text_groups.extend((x, 1.0) for x in fallback_texts)
    counter: dict[str, float] = {}
    for text, weight in text_groups:
        splitter = r"[|,;/]+" if "|" in safe_str(text) else r"[|,;/\s·]+"
        for token in re.split(splitter, safe_str(text)):
            t = clean_keyword_token(token)
            if len(t) < min_len or len(t) > 18 or t in GENERIC_TERMS:
                continue
            if re.fullmatch(r"\d+", t):
                continue
            if any(bad in t for bad in ["이슈다", "맥락", "프레임", "보도·게시글"]):
                continue
            counter[t] = counter.get(t, 0.0) + weight
    return [k for k, _ in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:max_keywords]]


def daily_keyword_counts(base_docs: pd.DataFrame, keywords: Sequence[str], weight_metric: str = "count_for_ranking") -> pd.DataFrame:
    docs = standardize_base_docs(base_docs) if "search_text" not in base_docs.columns or "count_for_ranking" not in base_docs.columns else base_docs.copy()
    docs["date"] = pd.to_datetime(docs["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    docs = docs.dropna(subset=["date"])
    docs["_w"] = pd.to_numeric(docs.get(weight_metric, docs.get("count_for_ranking", 1)), errors="coerce").fillna(1).clip(lower=0).astype(float)
    rows: list[pd.DataFrame] = []
    for keyword in keywords:
        keyword = safe_str(keyword).strip().lower()
        if not keyword:
            continue
        mask = docs["search_text"].fillna("").astype(str).str.contains(re.escape(keyword), na=False, regex=True)
        sub = docs[mask].copy()
        if sub.empty:
            continue
        group = sub.groupby(["date", "candidate", "source"], dropna=False).agg(
            doc_count=("doc_id", "nunique"),
            doc_count_weighted=("_w", "sum"),
        ).reset_index()
        group["keyword"] = keyword
        rows.append(group)
        all_sources = sub.groupby(["date", "candidate"], dropna=False).agg(
            doc_count=("doc_id", "nunique"),
            doc_count_weighted=("_w", "sum"),
        ).reset_index()
        all_sources["source"] = "ALL"
        all_sources["keyword"] = keyword
        rows.append(all_sources)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def add_complete_daily_grid(df: pd.DataFrame, key_col: str = "keyword") -> pd.DataFrame:
    if df.empty:
        return df
    frames: list[pd.DataFrame] = []
    key_cols = [key_col, "candidate", "source"]
    for keys, group in df.groupby(key_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        dates = pd.to_datetime(group["date"], errors="coerce")
        if dates.dropna().empty:
            continue
        grid = pd.DataFrame({"date": pd.date_range(dates.min(), dates.max(), freq="D").strftime("%Y-%m-%d")})
        for col, value in zip(key_cols, keys):
            grid[col] = value
        merged = grid.merge(group, on=["date", *key_cols], how="left")
        for col in ["doc_count", "doc_count_weighted"]:
            if col in merged.columns:
                merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0)
        for col in [c for c in merged.columns if c not in {"date", *key_cols, "doc_count", "doc_count_weighted"}]:
            if merged[col].notna().any():
                merged[col] = merged[col].ffill().bfill()
        frames.append(merged)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def robust_spikes(
    counts: pd.DataFrame,
    value_col: str = "doc_count_weighted",
    key_col: str = "keyword",
    window: int = 14,
    spike_threshold: float = 2.5,
    strong_threshold: float = 4.0,
    min_baseline_days: int = 5,
    min_total_count: float = 5.0,
) -> pd.DataFrame:
    if counts.empty:
        return counts
    work = add_complete_daily_grid(counts, key_col=key_col)
    rows: list[pd.DataFrame] = []
    for _, group in work.groupby([key_col, "candidate", "source"], dropna=False):
        group = group.sort_values("date").reset_index(drop=True).copy()
        vals = pd.to_numeric(group[value_col], errors="coerce").fillna(0).to_numpy(dtype=float)
        total = float(vals.sum())
        medians: list[float] = []
        mads: list[float] = []
        zscores: list[float] = []
        for i in range(len(vals)):
            start = max(0, i - window)
            base = vals[start:i]
            if len(base) < min_baseline_days:
                med = np.nan
                mad = np.nan
                z = np.nan
            else:
                med = float(np.median(base))
                mad_raw = float(np.median(np.abs(base - med)))
                scale = max(1.4826 * mad_raw, 1.0)
                z = float((vals[i] - med) / scale)
                mad = mad_raw
            medians.append(med)
            mads.append(mad)
            zscores.append(z)
        group["rolling_median_prev"] = medians
        group["rolling_mad_prev"] = mads
        group["robust_z"] = zscores
        group["is_spike"] = (group["robust_z"] >= spike_threshold) & (total >= min_total_count)
        group["is_strong_spike"] = (group["robust_z"] >= strong_threshold) & (total >= min_total_count)
        group["spike_level"] = np.select([group["is_strong_spike"], group["is_spike"]], ["strong_spike", "spike"], default="normal")
        group["baseline_window"] = window
        group["spike_threshold"] = spike_threshold
        group["strong_threshold"] = strong_threshold
        group["total_series_count"] = total
        rows.append(group)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def prepare_issue_spike_counts(
    issue_timeseries: pd.DataFrame,
    min_total_count: float = 5.0,
    min_active_days: int = 3,
    min_span_days: int = 6,
    max_series: int = 1000,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if issue_timeseries.empty:
        return pd.DataFrame(), pd.DataFrame()
    work = issue_timeseries.copy()
    work["date"] = pd.to_datetime(first_col(work, ["date"]), errors="coerce")
    work = work.dropna(subset=["date"])
    if work.empty:
        return pd.DataFrame(), pd.DataFrame()
    work["candidate"] = first_col(work, ["candidate"]).astype(str)
    work["source"] = first_col(work, ["source"]).astype(str)
    work["stable_issue_id"] = first_col(work, ["stable_issue_id", "period_issue_id", "issue_id"], "unknown_issue").astype(str)
    work["issue_name"] = first_col(work, ["issue_display_name", "issue_name", "representative_title", "stable_issue_label_auto"], "이슈").astype(str)
    work["issue_type"] = first_col(work, ["issue_type", "issue_type_final", "issue_type_rule"], "unknown").astype(str)
    metric_col = "doc_count_weighted" if "doc_count_weighted" in work.columns else "doc_count_raw" if "doc_count_raw" in work.columns else None
    if metric_col is None:
        return pd.DataFrame(), pd.DataFrame()
    work["doc_count_weighted"] = pd.to_numeric(work[metric_col], errors="coerce").fillna(0).clip(lower=0)
    work["date_str"] = work["date"].dt.strftime("%Y-%m-%d")

    daily = work.groupby(["date_str", "candidate", "source", "stable_issue_id"], dropna=False).agg(
        doc_count=("doc_count_weighted", "sum"),
        doc_count_weighted=("doc_count_weighted", "sum"),
        issue_name=("issue_name", "last"),
        issue_type=("issue_type", "last"),
    ).reset_index().rename(columns={"date_str": "date", "stable_issue_id": "keyword"})

    summary = daily.groupby(["keyword", "candidate", "source"], dropna=False).agg(
        active_days=("date", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max"),
        total_count=("doc_count_weighted", "sum"),
        issue_name=("issue_name", "last"),
        issue_type=("issue_type", "last"),
    ).reset_index()
    summary["span_days"] = (pd.to_datetime(summary["last_date"]) - pd.to_datetime(summary["first_date"])).dt.days + 1
    keep = summary[
        (summary["total_count"] >= min_total_count)
        & (summary["active_days"] >= min_active_days)
        & (summary["span_days"] >= min_span_days)
    ].copy()
    keep = keep.sort_values(["total_count", "active_days", "span_days"], ascending=[False, False, False])
    if max_series and len(keep) > max_series:
        keep = keep.head(max_series).copy()

    counts = daily.merge(keep[["keyword", "candidate", "source"]], on=["keyword", "candidate", "source"], how="inner")
    counts = counts.merge(
        keep[["keyword", "candidate", "source", "issue_name", "issue_type", "active_days", "span_days", "total_count"]],
        on=["keyword", "candidate", "source"],
        how="left",
        suffixes=("", "_series"),
    )
    for col in ["issue_name", "issue_type"]:
        series_col = f"{col}_series"
        if series_col in counts.columns:
            counts[col] = counts[series_col].fillna(counts[col])
            counts = counts.drop(columns=[series_col])
    return counts, keep

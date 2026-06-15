from __future__ import annotations

import numpy as np
import pandas as pd

from .text_utils import contains_keyword


def _ts(x):
    return pd.to_datetime(x, errors="coerce")


def _apply_common(df: pd.DataFrame, candidate: str, sources=None) -> pd.DataFrame:
    out = df.copy()
    if candidate and "candidate" in out.columns:
        out = out[out["candidate"].astype(str).eq(candidate)]
    if sources and "source" in out.columns:
        out = out[out["source"].isin(list(sources))]
    return out


def _first_nonempty(series: pd.Series) -> str:
    for value in series:
        if pd.notna(value):
            text = str(value).strip()
            if text:
                return text
    return ""


def _safe_row_value(row: pd.Series, key: str) -> str:
    try:
        value = row.get(key, "")
    except Exception:
        return ""
    if pd.isna(value):
        return ""
    return str(value).strip()


def _coerce_evidence_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["count_for_ranking", "comment_count", "view_count", "like_count"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0)
    if "reaction_total" not in out.columns:
        parts = []
        for c in ["comment_count", "view_count", "like_count"]:
            if c in out.columns:
                parts.append(pd.to_numeric(out[c], errors="coerce").fillna(0))
        if parts:
            out["reaction_total"] = sum(parts)
    return out


def _sort_evidence(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = _coerce_evidence_metrics(df)
    sort_cols = [c for c in ["count_for_ranking", "comment_count", "view_count", "like_count"] if c in out.columns]
    if sort_cols:
        out = out.sort_values(sort_cols, ascending=False)
    return out


def query_existing_window_top10(issue_briefs: pd.DataFrame, candidate: str, sources, start_date, end_date, window_types=None, keyword: str = "", max_rows: int = 50) -> pd.DataFrame:
    df = _apply_common(issue_briefs, candidate, sources)
    if df.empty:
        return df
    if window_types and "window_type" in df.columns:
        df = df[df["window_type"].isin(list(window_types))]
    ws = _ts(df.get("window_start"))
    we = _ts(df.get("window_end"))
    s = pd.to_datetime(start_date)
    e = pd.to_datetime(end_date)
    df = df[(ws <= e) & (we >= s)].copy()
    if keyword:
        search = df.get("search_text", df.get("issue_name", pd.Series(index=df.index, dtype=str)))
        df = df[contains_keyword(search, keyword)]
    if df.empty:
        return df
    ws = _ts(df.get("window_start"))
    we = _ts(df.get("window_end"))
    overlap_start = np.maximum(ws.values.astype("datetime64[ns]"), np.datetime64(s))
    overlap_end = np.minimum(we.values.astype("datetime64[ns]"), np.datetime64(e))
    overlap_days = ((overlap_end - overlap_start) / np.timedelta64(1, "D") + 1).clip(min=0)
    df["overlap_days"] = overlap_days
    df["exact_window"] = df["window_start"].astype(str).str[:10].eq(str(s.date())) & df["window_end"].astype(str).str[:10].eq(str(e.date()))
    for c in ["rank_in_window", "doc_count_weighted", "comment_count", "view_count", "rank_score"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.sort_values(["exact_window", "overlap_days", "rank_in_window", "doc_count_weighted"], ascending=[False, False, True, False]).head(max_rows)


def query_custom_period_top10_from_timeseries(issue_timeseries: pd.DataFrame, candidate: str, sources, start_date, end_date, keyword: str = "", top_n: int = 10) -> pd.DataFrame:
    df = _apply_common(issue_timeseries, candidate, sources)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame()
    d = _ts(df["date"])
    df = df[d.between(pd.to_datetime(start_date), pd.to_datetime(end_date))].copy()
    if keyword:
        search = df.get("search_text", df.get("issue_name", pd.Series(index=df.index, dtype=str)))
        df = df[contains_keyword(search, keyword)]
    if df.empty:
        return df
    for c in ["doc_count_weighted", "comment_count_weighted", "view_count_weighted", "like_count_weighted"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)
    group_cols = [c for c in ["candidate", "source", "stable_issue_id"] if c in df.columns]
    if not group_cols:
        return pd.DataFrame()
    agg = {
        "doc_count_weighted": ("doc_count_weighted", "sum"),
        "comment_count_weighted": ("comment_count_weighted", "sum"),
        "view_count_weighted": ("view_count_weighted", "sum"),
        "like_count_weighted": ("like_count_weighted", "sum"),
        "active_days": ("date", "nunique"),
    }
    for col in [
        "issue_name", "issue_display_name", "summary_display", "representative_title", "issue_type",
        "summary_1line", "top_terms", "review_status", "report_use_level", "dashboard_quality_flags",
        "dashboard_quality_tier", "search_text",
    ]:
        if col in df.columns and col not in group_cols:
            agg[col] = (col, _first_nonempty)
    g = df.groupby(group_cols, dropna=False).agg(**agg).reset_index()
    g["dashboard_rank_score"] = (
        0.60 * np.log1p(g["doc_count_weighted"].astype(float))
        + 0.20 * np.log1p(g["comment_count_weighted"].astype(float))
        + 0.15 * np.log1p(g["view_count_weighted"].astype(float))
        + 0.05 * np.log1p(g["active_days"].astype(float))
    )
    if "issue_name" not in g.columns:
        g["issue_name"] = ""
    if "issue_display_name" not in g.columns:
        g["issue_display_name"] = ""
    missing_display = g["issue_display_name"].fillna("").astype(str).str.strip().eq("")
    if missing_display.any() and "issue_name" in g.columns:
        g.loc[missing_display, "issue_display_name"] = g.loc[missing_display, "issue_name"].fillna("").astype(str)
    missing_name = g["issue_name"].fillna("").astype(str).str.strip().eq("")
    if missing_name.any() and "issue_display_name" in g.columns:
        g.loc[missing_name, "issue_name"] = g.loc[missing_name, "issue_display_name"].fillna("").astype(str)
    missing_name = g["issue_name"].fillna("").astype(str).str.strip().eq("")
    if missing_name.any() and "representative_title" in g.columns:
        g.loc[missing_name, "issue_name"] = g.loc[missing_name, "representative_title"].fillna("").astype(str)
    missing_name = g["issue_name"].fillna("").astype(str).str.strip().eq("")
    if missing_name.any() and "stable_issue_id" in g.columns:
        g.loc[missing_name, "issue_name"] = g.loc[missing_name, "stable_issue_id"].fillna("").astype(str)
    return g.sort_values("dashboard_rank_score", ascending=False).head(top_n)


def keyword_daily_timeline(base_docs_light: pd.DataFrame, candidate: str, sources, start_date, end_date, keyword: str) -> pd.DataFrame:
    df = _apply_common(base_docs_light, candidate, sources)
    if df.empty or "date" not in df.columns:
        return pd.DataFrame(columns=["date", "source", "doc_count", "doc_count_weighted", "comment_count", "view_count", "like_count"])
    d = _ts(df["date"])
    df = df[d.between(pd.to_datetime(start_date), pd.to_datetime(end_date))].copy()
    if keyword:
        search = df.get("search_text", df.get("title", pd.Series(index=df.index, dtype=str)))
        df = df[contains_keyword(search, keyword)]
    if df.empty:
        return pd.DataFrame(columns=["date", "source", "doc_count", "doc_count_weighted", "comment_count", "view_count", "like_count"])
    for c in ["count_for_ranking", "comment_count", "view_count", "like_count"]:
        df[c] = pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0)
    out = df.groupby([_ts(df["date"]).dt.strftime("%Y-%m-%d"), "source"], dropna=False).agg(
        doc_count=("doc_id", "nunique"),
        doc_count_weighted=("count_for_ranking", "sum"),
        comment_count=("comment_count", "sum"),
        view_count=("view_count", "sum"),
        like_count=("like_count", "sum"),
    ).reset_index().rename(columns={"date": "date"})
    if out.columns[0] != "date":
        out = out.rename(columns={out.columns[0]: "date"})
    return out.sort_values(["date", "source"])


def keyword_platform_reactions(base_docs_light: pd.DataFrame, candidate: str, sources, start_date, end_date, keyword: str) -> pd.DataFrame:
    tl = keyword_daily_timeline(base_docs_light, candidate, sources, start_date, end_date, keyword)
    if tl.empty:
        return pd.DataFrame(columns=["source", "doc_count", "doc_count_weighted", "comment_count", "view_count", "like_count"])
    return tl.groupby("source", dropna=False).agg(
        doc_count=("doc_count", "sum"),
        doc_count_weighted=("doc_count_weighted", "sum"),
        comment_count=("comment_count", "sum"),
        view_count=("view_count", "sum"),
        like_count=("like_count", "sum"),
    ).reset_index().sort_values(["doc_count_weighted", "view_count"], ascending=False)


def filter_evidence(evidence_docs: pd.DataFrame, candidate: str, sources, start_date, end_date, keyword: str = "", period_issue_id: str = "", stable_issue_id: str = "", limit: int = 200) -> pd.DataFrame:
    df = _apply_common(evidence_docs, candidate, sources)
    if df.empty:
        return df
    if "date" in df.columns:
        d = _ts(df["date"])
        df = df[d.between(pd.to_datetime(start_date), pd.to_datetime(end_date))].copy()
    if period_issue_id and "period_issue_id" in df.columns:
        df = df[df["period_issue_id"].astype(str).eq(str(period_issue_id))]
    if stable_issue_id and "stable_issue_id" in df.columns:
        df = df[df["stable_issue_id"].astype(str).eq(str(stable_issue_id))]
    if keyword:
        search = df.get("search_text", df.get("title", pd.Series(index=df.index, dtype=str)))
        df = df[contains_keyword(search, keyword)]
    return _sort_evidence(df).head(limit)


def evidence_for_issue(evidence_docs: pd.DataFrame, issue_row: pd.Series, candidate: str, sources, start_date, end_date, keyword: str = "", limit: int = 200) -> pd.DataFrame:
    """Return evidence docs for one selected issue.

    Period issue id is most precise. Custom-period rows usually only have a stable
    issue id, so this falls back to stable id when the exact period id is missing
    or produces no rows.
    """
    if evidence_docs.empty:
        return evidence_docs
    period_issue_id = _safe_row_value(issue_row, "period_issue_id")
    stable_issue_id = _safe_row_value(issue_row, "stable_issue_id")
    if not period_issue_id and not stable_issue_id:
        return pd.DataFrame(columns=evidence_docs.columns)

    if period_issue_id:
        exact = filter_evidence(
            evidence_docs,
            candidate,
            sources,
            start_date,
            end_date,
            keyword=keyword,
            period_issue_id=period_issue_id,
            limit=limit,
        )
        if not exact.empty or not stable_issue_id:
            return exact

    return filter_evidence(
        evidence_docs,
        candidate,
        sources,
        start_date,
        end_date,
        keyword=keyword,
        stable_issue_id=stable_issue_id,
        limit=limit,
    )


def top_reaction_documents(evidence_docs: pd.DataFrame, limit: int = 20) -> pd.DataFrame:
    if evidence_docs.empty:
        return evidence_docs
    df = _coerce_evidence_metrics(evidence_docs)
    metric_cols = [c for c in ["comment_count", "view_count", "like_count"] if c in df.columns]
    if not metric_cols:
        return df.head(limit)
    df["reaction_total"] = sum(pd.to_numeric(df[c], errors="coerce").fillna(0) for c in metric_cols)
    positive = df[df["reaction_total"].gt(0)].copy()
    if positive.empty:
        positive = df.copy()
    sort_cols = [c for c in ["comment_count", "view_count", "like_count", "count_for_ranking"] if c in positive.columns]
    if sort_cols:
        positive = positive.sort_values(sort_cols, ascending=False)
    return positive.head(limit)

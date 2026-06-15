from __future__ import annotations

import pandas as pd

from .text_utils import contains_keyword


COMMENT_METRIC_COLS = [
    "collected_comment_count",
    "top_level_comment_count",
    "reply_comment_count",
    "comment_like_sum",
    "comment_like_count",
    "sympathy_sum",
    "sympathy_count",
    "antipathy_sum",
    "antipathy_count",
    "reply_count_sum",
    "reply_count",
    "unique_author_hash_count",
]


def _ts(x):
    return pd.to_datetime(x, errors="coerce")


def _apply_common(df: pd.DataFrame, candidate: str = "", sources=None) -> pd.DataFrame:
    out = df.copy()
    if candidate and "candidate" in out.columns:
        out = out[out["candidate"].astype(str).eq(candidate)]
    if sources and "source" in out.columns:
        out = out[out["source"].isin(list(sources))]
    return out


def _between_dates(df: pd.DataFrame, date_col: str, start_date=None, end_date=None) -> pd.DataFrame:
    if df.empty or date_col not in df.columns or start_date is None or end_date is None:
        return df
    d = _ts(df[date_col])
    return df[d.between(pd.to_datetime(start_date), pd.to_datetime(end_date))].copy()


def _coerce_comment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in COMMENT_METRIC_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def _issue_keys(
    issue_comment_map: pd.DataFrame,
    candidate: str = "",
    sources=None,
    period_issue_id: str = "",
    stable_issue_id: str = "",
    issue_id: str = "",
) -> list[str]:
    if issue_comment_map is None or issue_comment_map.empty:
        return []
    m = _apply_common(issue_comment_map, candidate, sources)
    target = str(issue_id or "").strip()
    if target:
        if "period_issue_id" in m.columns:
            by_period = m["period_issue_id"].astype(str).eq(target)
        else:
            by_period = pd.Series(False, index=m.index)
        if "stable_issue_id" in m.columns:
            by_stable = m["stable_issue_id"].astype(str).eq(target)
        else:
            by_stable = pd.Series(False, index=m.index)
        m = m[by_period | by_stable]
    if period_issue_id and "period_issue_id" in m.columns:
        m = m[m["period_issue_id"].astype(str).eq(str(period_issue_id))]
    if stable_issue_id and "stable_issue_id" in m.columns:
        m = m[m["stable_issue_id"].astype(str).eq(str(stable_issue_id))]
    if "parent_doc_key" not in m.columns:
        return []
    return m["parent_doc_key"].dropna().astype(str).unique().tolist()


def parent_docs_for_issue_or_keyword(
    parent_docs: pd.DataFrame,
    issue_comment_map: pd.DataFrame | None = None,
    *,
    issue_id: str | None = None,
    period_issue_id: str | None = None,
    stable_issue_id: str | None = None,
    keyword: str | None = None,
    candidate: str | None = None,
    sources: list[str] | None = None,
    start_date=None,
    end_date=None,
    limit: int = 200,
) -> pd.DataFrame:
    """Return parent articles/videos/posts before exposing their comments."""
    if parent_docs is None or parent_docs.empty:
        return pd.DataFrame()
    docs = _apply_common(parent_docs, candidate or "", sources)
    if docs.empty:
        return docs
    if "date" in docs.columns and docs["date"].fillna("").astype(str).str.len().gt(0).any():
        docs = _between_dates(docs, "date", start_date, end_date)
    elif "first_comment_date" in docs.columns:
        docs = _between_dates(docs, "first_comment_date", start_date, end_date)

    keys = _issue_keys(
        issue_comment_map if issue_comment_map is not None else pd.DataFrame(),
        candidate or "",
        sources,
        period_issue_id or "",
        stable_issue_id or "",
        issue_id or "",
    )
    if (issue_id or period_issue_id or stable_issue_id) and "parent_doc_key" in docs.columns:
        docs = docs[docs["parent_doc_key"].astype(str).isin(keys)].copy()

    if keyword:
        search_parts = []
        for col in ["parent_title", "parent_url", "publisher_or_channel", "top_comment_sample"]:
            if col in docs.columns:
                search_parts.append(docs[col].fillna("").astype(str))
        if search_parts:
            search = search_parts[0]
            for part in search_parts[1:]:
                search = search + " " + part
            docs = docs[contains_keyword(search, keyword)]

    docs = _coerce_comment_metrics(docs)
    sort_cols = [c for c in ["collected_comment_count", "comment_like_sum", "sympathy_sum", "reply_count_sum"] if c in docs.columns]
    if sort_cols:
        docs = docs.sort_values(sort_cols, ascending=False)
    return docs.head(limit)


def comments_for_parent_docs(
    comments: pd.DataFrame,
    parent_doc_keys: list[str],
    *,
    keyword: str | None = None,
    start_date=None,
    end_date=None,
    sort_by: str = "반응 많은 순",
    ascending: bool = False,
    limit: int = 500,
) -> pd.DataFrame:
    """Return comments that belong to selected parent documents."""
    if comments is None or comments.empty:
        return pd.DataFrame()
    keys = [str(k) for k in parent_doc_keys if str(k)]
    if not keys or "parent_doc_key" not in comments.columns:
        return pd.DataFrame(columns=comments.columns)
    df = comments[comments["parent_doc_key"].astype(str).isin(keys)].copy()
    if "comment_date" in df.columns:
        df = _between_dates(df, "comment_date", start_date, end_date)
    if keyword:
        text_col = "comment_text_clean" if "comment_text_clean" in df.columns else "comment_text_masked"
        df = df[contains_keyword(df.get(text_col, pd.Series(index=df.index, dtype=str)), keyword)]
    df = _coerce_comment_metrics(df)

    if sort_by == "최신순" and "comment_datetime" in df.columns:
        df = df.sort_values("comment_datetime", ascending=False)
    elif sort_by == "공감/좋아요 많은 순":
        cols = [c for c in ["sympathy_count", "comment_like_count", "reply_count"] if c in df.columns]
        if cols:
            df = df.sort_values(cols, ascending=[ascending] * len(cols))
    elif sort_by == "답글 많은 순" and "reply_count" in df.columns:
        df = df.sort_values("reply_count", ascending=ascending)
    else:
        cols = [c for c in ["comment_like_count", "sympathy_count", "reply_count"] if c in df.columns]
        if cols:
            df = df.sort_values(cols, ascending=[ascending] * len(cols))
    return df.head(limit)


def comment_timeline_for_parent_docs(comments: pd.DataFrame, parent_doc_keys: list[str]) -> pd.DataFrame:
    """Return daily comment timeline for selected parent documents."""
    if comments is None or comments.empty:
        return pd.DataFrame(columns=["date", "source", "collected_comment_count", "comment_like_sum", "sympathy_sum", "antipathy_sum", "reply_count_sum"])
    keys = [str(k) for k in parent_doc_keys if str(k)]
    if not keys:
        return pd.DataFrame(columns=["date", "source", "collected_comment_count", "comment_like_sum", "sympathy_sum", "antipathy_sum", "reply_count_sum"])
    df = comments.copy()
    if "parent_doc_key" in df.columns:
        df = df[df["parent_doc_key"].astype(str).isin(keys)]
    if df.empty or "comment_date" not in df.columns:
        return pd.DataFrame()
    df = _coerce_comment_metrics(df)
    out = df.groupby(["comment_date", "source"], dropna=False).agg(
        collected_comment_count=("comment_uid", "nunique"),
        comment_like_sum=("comment_like_count", "sum"),
        sympathy_sum=("sympathy_count", "sum"),
        antipathy_sum=("antipathy_count", "sum"),
        reply_count_sum=("reply_count", "sum"),
    ).reset_index().rename(columns={"comment_date": "date"})
    out = out[out["date"].fillna("").astype(str).str.len().gt(0)].copy()
    return out.sort_values(["date", "source"])


def comment_platform_for_parent_docs(
    comments: pd.DataFrame,
    parent_docs: pd.DataFrame,
    parent_doc_keys: list[str],
) -> pd.DataFrame:
    """Return source-level comment reactions for selected parent documents."""
    if parent_docs is None or parent_docs.empty:
        return pd.DataFrame(columns=["source", "parent_doc_count", "collected_comment_count", "comment_like_sum", "sympathy_sum", "antipathy_sum", "reply_count_sum", "unique_author_hash_count"])
    keys = [str(k) for k in parent_doc_keys if str(k)]
    if not keys:
        return pd.DataFrame(columns=["source", "parent_doc_count", "collected_comment_count", "comment_like_sum", "sympathy_sum", "antipathy_sum", "reply_count_sum", "unique_author_hash_count"])
    df = parent_docs.copy()
    if "parent_doc_key" in df.columns:
        df = df[df["parent_doc_key"].astype(str).isin(keys)]
    if df.empty:
        return pd.DataFrame()
    df = _coerce_comment_metrics(df)
    for col in ["collected_comment_count", "comment_like_sum", "sympathy_sum", "antipathy_sum", "reply_count_sum", "unique_author_hash_count"]:
        if col not in df.columns:
            df[col] = 0
    return df.groupby("source", dropna=False).agg(
        parent_doc_count=("parent_doc_key", "nunique"),
        collected_comment_count=("collected_comment_count", "sum"),
        comment_like_sum=("comment_like_sum", "sum"),
        sympathy_sum=("sympathy_sum", "sum"),
        antipathy_sum=("antipathy_sum", "sum"),
        reply_count_sum=("reply_count_sum", "sum"),
        unique_author_hash_count=("unique_author_hash_count", "sum"),
    ).reset_index().sort_values(["collected_comment_count", "parent_doc_count"], ascending=False)


# Backward-compatible aliases for older smoke scripts.
def filter_comment_parent_documents(doc_summary: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return parent_docs_for_issue_or_keyword(doc_summary, **kwargs)


def filter_comments(comments: pd.DataFrame, issue_comment_map: pd.DataFrame | None = None, parent_doc_keys=None, **kwargs) -> pd.DataFrame:
    allowed = {k: kwargs[k] for k in ["keyword", "start_date", "end_date", "sort_by", "ascending", "limit"] if k in kwargs}
    return comments_for_parent_docs(comments, parent_doc_keys or [], **allowed)


def comment_platform_summary(comments: pd.DataFrame, parent_docs: pd.DataFrame) -> pd.DataFrame:
    keys = parent_docs["parent_doc_key"].dropna().astype(str).tolist() if parent_docs is not None and "parent_doc_key" in parent_docs.columns else []
    return comment_platform_for_parent_docs(comments, parent_docs, keys)


def comment_timeline(comment_timeseries: pd.DataFrame, candidate: str = "", sources=None, start_date=None, end_date=None) -> pd.DataFrame:
    df = _apply_common(comment_timeseries, candidate, sources)
    if "date" in df.columns:
        df = _between_dates(df, "date", start_date, end_date)
    return df

from __future__ import annotations

from datetime import date
from pathlib import Path
import re

import pandas as pd
import pyarrow.parquet as pq
import streamlit as st

from src.data_loader import load_metadata, load_table, coerce_numeric
from src.queries import (
    evidence_for_issue,
    filter_evidence,
    keyword_daily_timeline,
    keyword_platform_reactions,
    query_custom_period_top10_from_timeseries,
    query_existing_window_top10,
    top_reaction_documents,
)
from src.charts import line_chart, bar_chart
from src.comment_charts import comment_platform_chart, comment_timeline_chart
from src.comment_queries import (
    comment_platform_for_parent_docs,
    comment_timeline_for_parent_docs,
    comments_for_parent_docs,
    parent_docs_for_issue_or_keyword,
)
from src.text_utils import fmt_num
from src.display_labels import (
    CANDIDATE_LABELS,
    SOURCE_LABELS,
    WINDOW_LABELS,
    display_table,
    glossary_markdown,
    label_issue_type,
    label_source,
    label_window,
    metric_label,
    reverse_lookup,
)

DATA_DIR = Path("dashboard_data")

st.set_page_config(page_title="서울시장 후보 온라인 이슈 레이더", layout="wide", initial_sidebar_state="collapsed")
st.markdown(
    """
    <style>
    [data-testid="stToolbar"], .stDeployButton, #MainMenu, footer,
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapseButton"],
    [data-testid="stExpandSidebarButton"],
    [data-testid="stSidebarNav"] {display: none !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

NUMERIC_COLS = [
    "rank_in_window", "doc_count_weighted", "doc_count_raw", "comment_count", "view_count", "like_count",
    "comment_count_weighted", "view_count_weighted", "like_count_weighted", "count_for_ranking", "rank_score",
    "cluster_coherence", "parent_doc_count", "collected_comment_count", "platform_comment_count",
    "top_level_comment_count", "reply_comment_count", "comment_like_sum", "comment_like_count",
    "sympathy_sum", "sympathy_count", "antipathy_sum", "antipathy_count", "reply_count_sum",
    "reply_count", "unique_author_hash_count",
]

BRIEF_INDEX_COLS = ["candidate", "source", "window_type", "window_start", "window_end"]
BRIEF_VIEW_COLS = [
    "period_issue_id", "stable_issue_id", "candidate", "source", "window_type", "window_start", "window_end",
    "rank_in_window", "issue_name", "issue_type", "summary_1line", "representative_title",
    "representative_url", "top_terms", "doc_count_raw", "doc_count_weighted", "comment_count",
    "view_count", "like_count", "rank_score", "cluster_coherence", "issue_display_name",
    "summary_display", "issue_label_auto", "doc_count",
]
TIMESERIES_VIEW_COLS = [
    "date", "candidate", "source", "stable_issue_id", "issue_name", "issue_display_name",
    "summary_display", "representative_title", "issue_type", "summary_1line", "top_terms",
    "review_status", "report_use_level", "dashboard_quality_flags", "dashboard_quality_tier",
    "doc_count_weighted", "comment_count_weighted", "view_count_weighted", "like_count_weighted",
    "search_text",
]
EVIDENCE_VIEW_COLS = [
    "period_issue_id", "stable_issue_id", "candidate", "source", "date", "doc_id", "title", "url",
    "publisher_or_channel", "comment_count", "view_count", "like_count", "count_for_ranking",
]
BASE_VIEW_COLS = [
    "candidate", "source", "date", "doc_id", "title", "search_text", "count_for_ranking",
    "comment_count", "view_count", "like_count",
]


@st.cache_data(show_spinner=False)
def available_table_columns(data_dir: str, stem: str) -> tuple[str, ...]:
    d = Path(data_dir)
    pq_path = d / f"{stem}.parquet"
    csv_path = d / f"{stem}.csv"
    if pq_path.exists():
        return tuple(pq.read_schema(pq_path).names)
    if csv_path.exists():
        return tuple(pd.read_csv(csv_path, nrows=0, encoding="utf-8-sig").columns)
    return tuple()


def read_table_subset(data_dir: str, stem: str, columns: list[str], filters=None) -> pd.DataFrame:
    d = Path(data_dir)
    available = set(available_table_columns(data_dir, stem))
    keep = [c for c in columns if c in available]
    if not keep:
        return pd.DataFrame()
    pq_path = d / f"{stem}.parquet"
    csv_path = d / f"{stem}.csv"
    if pq_path.exists():
        return pd.read_parquet(pq_path, columns=keep, filters=filters)
    if csv_path.exists():
        df = pd.read_csv(csv_path, usecols=keep, dtype="object", low_memory=False, encoding="utf-8-sig")
        return df
    return pd.DataFrame()


def add_filter(filters: list[tuple], available: set[str], column: str, op: str, value) -> None:
    if column in available and value not in (None, "", [], ()):
        filters.append((column, op, value))


def add_in_filter(filters: list[tuple], available: set[str], column: str, values: tuple[str, ...]) -> None:
    if column not in available or not values:
        return
    if len(values) == 1:
        filters.append((column, "==", values[0]))
    else:
        filters.append((column, "in", list(values)))


@st.cache_data(show_spinner="기본 분석 데이터를 불러오는 중...")
def load_dashboard_index_data(data_dir: str):
    meta = load_metadata(data_dir)
    briefs = read_table_subset(data_dir, "issue_briefs", BRIEF_INDEX_COLS)
    return meta, coerce_numeric(briefs, NUMERIC_COLS)


@st.cache_data(show_spinner="선택한 분석 데이터를 불러오는 중...")
def load_dashboard_table(data_dir: str, stem: str):
    return coerce_numeric(load_table(data_dir, stem), NUMERIC_COLS)


@st.cache_data(show_spinner="선택한 기간 이슈를 불러오는 중...")
def load_issue_briefs_view(
    data_dir: str,
    candidate: str,
    sources: tuple[str, ...],
    window_types: tuple[str, ...],
    start_date_key: str,
    end_date_key: str,
    keyword: str,
):
    if not sources or not window_types:
        return pd.DataFrame()
    available = set(available_table_columns(data_dir, "issue_briefs"))
    filters: list[tuple] = []
    add_filter(filters, available, "candidate", "==", candidate)
    add_in_filter(filters, available, "source", sources)
    add_in_filter(filters, available, "window_type", window_types)
    add_filter(filters, available, "window_end", ">=", start_date_key)
    add_filter(filters, available, "window_start", "<=", end_date_key)
    cols = list(BRIEF_VIEW_COLS)
    if keyword.strip():
        cols.append("search_text")
    df = read_table_subset(data_dir, "issue_briefs", cols, filters=filters)
    return coerce_numeric(df, NUMERIC_COLS)


@st.cache_data(show_spinner="선택 기간 시계열을 불러오는 중...")
def load_issue_timeseries_view(
    data_dir: str,
    candidate: str,
    sources: tuple[str, ...],
    start_date_key: str,
    end_date_key: str,
):
    if not sources:
        return pd.DataFrame()
    available = set(available_table_columns(data_dir, "issue_timeseries"))
    filters: list[tuple] = []
    add_filter(filters, available, "candidate", "==", candidate)
    add_in_filter(filters, available, "source", sources)
    add_filter(filters, available, "date", ">=", start_date_key)
    add_filter(filters, available, "date", "<=", end_date_key)
    df = read_table_subset(data_dir, "issue_timeseries", TIMESERIES_VIEW_COLS, filters=filters)
    return coerce_numeric(df, NUMERIC_COLS)


@st.cache_data(show_spinner="키워드 분석 데이터를 불러오는 중...")
def load_base_docs_view(
    data_dir: str,
    candidate: str,
    sources: tuple[str, ...],
    start_date_key: str,
    end_date_key: str,
):
    if not sources:
        return pd.DataFrame()
    available = set(available_table_columns(data_dir, "base_docs_light"))
    filters: list[tuple] = []
    add_filter(filters, available, "candidate", "==", candidate)
    add_in_filter(filters, available, "source", sources)
    add_filter(filters, available, "date", ">=", start_date_key)
    add_filter(filters, available, "date", "<=", end_date_key)
    df = read_table_subset(data_dir, "base_docs_light", BASE_VIEW_COLS, filters=filters)
    return coerce_numeric(df, NUMERIC_COLS)


@st.cache_data(show_spinner="근거 문서를 불러오는 중...")
def load_evidence_filter_view(
    data_dir: str,
    candidate: str,
    sources: tuple[str, ...],
    start_date_key: str,
    end_date_key: str,
):
    if not sources:
        return pd.DataFrame()
    available = set(available_table_columns(data_dir, "evidence_docs"))
    filters: list[tuple] = []
    add_filter(filters, available, "candidate", "==", candidate)
    add_in_filter(filters, available, "source", sources)
    add_filter(filters, available, "date", ">=", start_date_key)
    add_filter(filters, available, "date", "<=", end_date_key)
    df = read_table_subset(data_dir, "evidence_docs", EVIDENCE_VIEW_COLS, filters=filters)
    return coerce_numeric(df, NUMERIC_COLS)


@st.cache_data(show_spinner="선택 이슈의 근거 문서를 불러오는 중...")
def load_evidence_for_issue_view(
    data_dir: str,
    candidate: str,
    sources: tuple[str, ...],
    start_date_key: str,
    end_date_key: str,
    keyword: str,
    period_issue_id: str,
    stable_issue_id: str,
    representative_url: str = "",
    representative_title: str = "",
    row_window_start: str = "",
    row_window_end: str = "",
):
    if not period_issue_id and not stable_issue_id and not representative_url and not representative_title:
        return pd.DataFrame()
    available = set(available_table_columns(data_dir, "evidence_docs"))
    evidence_sources = tuple(s for s in sources if s != "all_sources_combined")

    def _read_with_id(id_column: str, id_value: str) -> pd.DataFrame:
        filters: list[tuple] = []
        add_filter(filters, available, "candidate", "==", candidate)
        add_in_filter(filters, available, "source", evidence_sources or sources)
        add_filter(filters, available, "date", ">=", start_date_key)
        add_filter(filters, available, "date", "<=", end_date_key)
        add_filter(filters, available, id_column, "==", id_value)
        return read_table_subset(data_dir, "evidence_docs", EVIDENCE_VIEW_COLS, filters=filters)

    def _read_representative() -> pd.DataFrame:
        filters: list[tuple] = []
        add_filter(filters, available, "candidate", "==", candidate)
        add_in_filter(filters, available, "source", evidence_sources)
        add_filter(filters, available, "date", ">=", row_window_start or start_date_key)
        add_filter(filters, available, "date", "<=", row_window_end or end_date_key)
        df = read_table_subset(data_dir, "evidence_docs", EVIDENCE_VIEW_COLS, filters=filters)
        if df.empty:
            return df
        mask = pd.Series(False, index=df.index)
        if representative_url and "url" in df.columns:
            mask = mask | df["url"].fillna("").astype(str).eq(representative_url)
        if representative_title and "title" in df.columns:
            mask = mask | df["title"].fillna("").astype(str).eq(representative_title)
        out = df[mask].copy()
        if out.empty:
            return out
        if "url" in out.columns:
            out = out.drop_duplicates(["url", "title"] if "title" in out.columns else ["url"])
        sort_cols = [c for c in ["comment_count", "view_count", "like_count", "count_for_ranking"] if c in out.columns]
        if sort_cols:
            for col in sort_cols:
                out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
            out = out.sort_values(sort_cols, ascending=False)
        return out.head(30)

    if period_issue_id:
        exact = _read_with_id("period_issue_id", period_issue_id)
        exact = filter_evidence(exact, candidate, sources, start_date_key, end_date_key, keyword=keyword, period_issue_id=period_issue_id, limit=300)
        if not exact.empty:
            return coerce_numeric(exact, NUMERIC_COLS)

    if stable_issue_id:
        fallback = _read_with_id("stable_issue_id", stable_issue_id)
        fallback = filter_evidence(fallback, candidate, sources, start_date_key, end_date_key, keyword=keyword, stable_issue_id=stable_issue_id, limit=300)
        if not fallback.empty:
            return coerce_numeric(fallback, NUMERIC_COLS)

    representative = _read_representative()
    return coerce_numeric(representative, NUMERIC_COLS)


@st.cache_data(show_spinner="댓글 분석 데이터를 불러오는 중...")
def load_dashboard_comment_data(data_dir: str):
    comments = load_table(data_dir, "comments_light")
    doc_comments = load_table(data_dir, "document_comment_summary")
    issue_comment_map = load_table(data_dir, "issue_comment_map")
    comment_timeseries = load_table(data_dir, "comment_timeseries")
    if not comments.empty and "comment_text_masked" not in comments.columns and "comment_text_clean" in comments.columns:
        comments = comments.copy()
        comments["comment_text_masked"] = comments["comment_text_clean"]
    return (
        coerce_numeric(comments, NUMERIC_COLS),
        coerce_numeric(doc_comments, NUMERIC_COLS),
        coerce_numeric(issue_comment_map, NUMERIC_COLS),
        coerce_numeric(comment_timeseries, NUMERIC_COLS),
    )


def source_display(x: str) -> str:
    return label_source(x)


def reverse_lookup_available(mapping: dict[str, str], label: str, available_keys: list[str] | tuple[str, ...]) -> str:
    """Resolve duplicated display labels using keys that actually exist in the loaded data."""
    matches = [key for key, value in mapping.items() if value == label]
    available = set(available_keys)
    for key in matches:
        if key in available:
            return key
    return matches[0] if matches else label


def render_page_nav():
    nav1, nav2, nav3 = st.columns(3)
    nav1.page_link("app.py", label="이슈 레이더")
    nav2.page_link("pages/6_통계_분석.py", label="통계 분석")
    nav3.page_link("pages/7_분포검정_급등탐지.py", label="분포 검정·급등 탐지")


def display_text(value) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return "" if value is None else str(value)


def first_value(row: pd.Series, *cols: str, default=0):
    for col in cols:
        if col in row.index:
            value = row.get(col)
            try:
                if pd.notna(value):
                    return value
            except Exception:
                if value is not None:
                    return value
    return default


WEAK_SUMMARY_MARKERS = [
    "맥락에서 후보가 연결된 이슈다",
    "관련 여러 보도·게시글이 묶인 기타 이슈다",
    "부동산·주택·세금·정비사업을 둘러싼 정책 프레임의 이슈다",
    "선거 캠페인 행보·진영 결집·투표 독려와 관련된 이슈다",
    "방송·토론·인터뷰를 통해 후보 메시지가 노출된 이슈다",
    "후보 확정·경선 결과와 본선 구도 형성에 관한 이슈다",
    "후보 이슈라기보다 UI·광고·주식·재게시성 잡음일 가능성이 큰 항목이다",
]
WEAK_NAME_TERMS = {
    "정책", "vs", "대통령", "시장", "시정", "시민", "이재명", "함께", "공약", "정부", "대표",
    "게시글", "댓글", "목록", "삭제", "수정", "스팸처리", "공유하기", "북마크", "이전글", "더보기",
    "daum", "가입하기", "계정", "카카오", "로그인", "re", "저장", "됩니다", "주세요", "필독",
    "내용", "방송", "모음", "제목", "신문기사", "정원", "감사",
}
STRONG_NAME_TERMS = {
    "여론조사", "본선", "판세", "TV토론", "토론", "공방", "GTX", "삼성역", "철근", "누락",
    "서소문", "고가", "사고", "안전", "칸쿤", "출장", "의혹", "선거법", "고발", "장특공",
    "세금", "경선", "확정", "사전투표", "투표", "박원순", "노무현", "부동산",
}


def shorten_text(value, max_len: int = 76) -> str:
    text = re.sub(r"\s+", " ", display_text(value)).strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def split_terms(value) -> list[str]:
    text = display_text(value)
    if not text:
        return []
    parts = re.split(r"\s*(?:\||·|,|/|;|\n)\s*", text)
    return [p.strip() for p in parts if p and p.strip()]


def first_evidence_title(row: pd.Series) -> str:
    for col in ["representative_title", "evidence_titles"]:
        value = display_text(row.get(col, ""))
        if not value:
            continue
        parts = re.split(r"\s*(?:\|\||\||\n)\s*", value)
        for part in parts:
            part = part.strip()
            if part:
                return part
    return ""


def weak_summary(value) -> bool:
    text = display_text(value)
    return not text or any(marker in text for marker in WEAK_SUMMARY_MARKERS)


def weak_issue_name(value) -> bool:
    name = display_text(value)
    if not name:
        return True
    tokens = split_terms(name)
    if not tokens:
        return True
    if any(term in name for term in STRONG_NAME_TERMS):
        generic_hits = sum(1 for token in tokens if token.lower() in WEAK_NAME_TERMS)
        return len(tokens) >= 3 and generic_hits >= len(tokens) - 1 and "공방" not in name
    if "_" in name:
        return True
    generic_hits = sum(1 for token in tokens if token.lower() in WEAK_NAME_TERMS)
    return generic_hits >= max(1, len(tokens) - 1)


def issue_display_name(row: pd.Series) -> str:
    raw = display_text(row.get("issue_display_name", ""))
    if raw:
        return raw
    name = display_text(row.get("issue_name", "")) or display_text(row.get("issue_label_auto", ""))
    if name and not weak_issue_name(name):
        return name
    title = first_evidence_title(row)
    if title:
        return shorten_text(title, 54)
    terms = [t for t in split_terms(row.get("top_terms", "")) if t.lower() not in WEAK_NAME_TERMS][:4]
    if terms:
        return "·".join(terms)
    return display_text(row.get("stable_issue_id", "")) or "이름 없는 이슈"


def issue_display_summary(row: pd.Series) -> str:
    raw = display_text(row.get("summary_display", ""))
    if raw:
        return raw
    summary = display_text(row.get("summary_1line", ""))
    title = first_evidence_title(row)
    terms = [t for t in split_terms(row.get("top_terms", "")) if t.lower() not in WEAK_NAME_TERMS][:5]
    if title and (weak_summary(summary) or weak_issue_name(row.get("issue_name", ""))):
        return f"대표 근거: {shorten_text(title, 92)}"
    if summary and not weak_summary(summary):
        return summary
    if terms:
        return f"핵심어: {' · '.join(terms)}"
    if title:
        return f"대표 근거: {shorten_text(title, 92)}"
    return "대표 근거 확인이 필요한 자동 묶음입니다."


def issue_name(row: pd.Series) -> str:
    return issue_display_name(row)


def issue_option_label(row: pd.Series, rank: int) -> str:
    source = label_source(row.get("source", ""))
    window = label_window(row.get("window_type", "")) if "window_type" in row.index else "사용자 지정"
    doc_count = fmt_num(first_value(row, "doc_count_weighted", "doc_count", default=0), 1)
    return f"{rank}. {issue_name(row)} · {source} · {window} · 분석 문서 {doc_count}"


def prepare_top_table(top: pd.DataFrame, limit: int | None = None) -> pd.DataFrame:
    out = top.reset_index(drop=True).copy()
    out["issue_display_name"] = [issue_display_name(row) for _, row in out.iterrows()]
    out["summary_display"] = [issue_display_summary(row) for _, row in out.iterrows()]
    if "representative_title" in out.columns:
        out["representative_title_short"] = out["representative_title"].map(lambda x: shorten_text(x, 88))
    dedupe_cols = ["issue_display_name"]
    if "source" in out.columns:
        dedupe_cols.append("source")
    out = out.drop_duplicates(dedupe_cols, keep="first")
    if limit is not None:
        out = out.head(limit)
    out = out.reset_index(drop=True)
    out["display_rank"] = range(1, len(out) + 1)
    return out


def issue_card(row: pd.Series, rank: int, *, heading: str = "선택 이슈 상세"):
    name = issue_name(row)
    issue_type = row.get("issue_type", row.get("issue_type_rule", ""))
    source = row.get("source", "")
    window = row.get("window_type", "")
    comment_value = row.get("comment_count", row.get("comment_count_weighted", 0))
    view_value = row.get("view_count", row.get("view_count_weighted", 0))
    like_value = row.get("like_count", row.get("like_count_weighted", 0))
    with st.container(border=True):
        st.markdown(f"### {heading}")
        st.markdown(f"#### {rank}. {name}")
        st.caption(
            f"자료원: {label_source(source)} · 기간 유형: {label_window(window) if display_text(window) else '사용자 지정'} · "
            f"이슈 유형: {label_issue_type(issue_type)}"
        )
        summary = issue_display_summary(row)
        if summary:
            st.write(summary)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("분석 문서 수", fmt_num(row.get("doc_count_weighted", row.get("doc_count", 0)), 1))
        c2.metric("댓글 수", fmt_num(comment_value))
        c3.metric("조회수", fmt_num(view_value))
        c4.metric("좋아요 수", fmt_num(like_value))
        representative_title = display_text(row.get("representative_title", ""))
        if representative_title:
            st.write(f"대표 제목: {representative_title}")
        representative_url = display_text(row.get("representative_url", ""))
        if representative_url.startswith("http"):
            st.markdown(f"[대표 근거 열기]({representative_url})")
        with st.expander("핵심어·근거 보기"):
            st.write("핵심어:", row.get("top_terms", ""))
            st.write("근거 제목:", row.get("evidence_titles", ""))


def evidence_metrics(evidence_df: pd.DataFrame):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("근거 문서", f"{len(evidence_df):,}")
    c2.metric("댓글 수", fmt_num(evidence_df.get("comment_count", pd.Series([0])).sum()))
    c3.metric("조회수", fmt_num(evidence_df.get("view_count", pd.Series([0])).sum()))
    c4.metric("좋아요 수", fmt_num(evidence_df.get("like_count", pd.Series([0])).sum()))


def render_issue_evidence(evidence_df: pd.DataFrame):
    if evidence_df.empty:
        st.info("선택한 이슈와 연결된 근거 문서가 없습니다.")
        return
    evidence_metrics(evidence_df)
    st.markdown("##### 대표 근거 문서")
    st.dataframe(display_table(evidence_df, "evidence", max_rows=30), width="stretch", height=320, hide_index=True)
    reaction_docs = top_reaction_documents(evidence_df, limit=15)
    if not reaction_docs.empty:
        st.markdown("##### 댓글·조회 반응이 큰 문서")
        st.caption("이 표는 댓글 원문이 아니라 댓글 수·조회수·좋아요 수가 큰 문서를 반응 근거로 보여줍니다. 댓글 원문은 원문·댓글 반응 탭에서 확인할 수 있습니다.")
        st.dataframe(display_table(reaction_docs, "reaction_docs"), width="stretch", height=260, hide_index=True)

def render_comment_drilldown(
    briefs: pd.DataFrame,
    comments: pd.DataFrame,
    doc_comments: pd.DataFrame,
    issue_comment_map: pd.DataFrame,
    candidate: str,
    selected_sources: list[str],
    selected_windows: list[str],
    start_date,
    end_date,
    keyword: str,
):
    st.subheader("원문·댓글 반응")
    st.caption("이슈 또는 키워드와 연결된 원문 기사·영상·게시글을 먼저 확인하고, 그 원문에 달린 수집 댓글을 함께 봅니다.")
    st.info("댓글은 수집된 원문 단위 데이터입니다. 플랫폼 전체 댓글 수와 다를 수 있고, 공개/시연용 보기에서는 민감정보가 마스킹된 댓글을 기본 표시합니다.")

    if comments.empty or doc_comments.empty:
        st.warning("댓글 분석용 데이터가 준비되지 않아 이 탭을 표시할 수 없습니다.")
        return

    comment_issue_sources = selected_sources
    period_ids: set[str] = set()
    stable_ids: set[str] = set()
    if not issue_comment_map.empty:
        map_for_options = issue_comment_map
        if "candidate" in map_for_options.columns:
            map_for_options = map_for_options[map_for_options["candidate"].astype(str).eq(candidate)]
        if "source" in map_for_options.columns:
            mapped_sources = set(map_for_options["source"].dropna().astype(str).unique().tolist())
            comment_issue_sources = [s for s in selected_sources if s in mapped_sources]
            map_for_options = map_for_options[map_for_options["source"].astype(str).isin(comment_issue_sources)]
        if "period_issue_id" in map_for_options.columns:
            period_ids = set(map_for_options["period_issue_id"].dropna().astype(str).drop_duplicates().tolist())
        if "stable_issue_id" in map_for_options.columns:
            stable_ids = set(map_for_options["stable_issue_id"].dropna().astype(str).drop_duplicates().tolist())

    issue_candidates = pd.DataFrame()
    if comment_issue_sources:
        issue_candidates = query_existing_window_top10(
            briefs,
            candidate,
            comment_issue_sources,
            start_date,
            end_date,
            selected_windows,
            keyword=keyword,
            max_rows=100,
        )
        if not issue_candidates.empty and (period_ids or stable_ids):
            period_match = issue_candidates.get("period_issue_id", pd.Series(index=issue_candidates.index, dtype=str)).fillna("").astype(str).isin(period_ids)
            stable_match = issue_candidates.get("stable_issue_id", pd.Series(index=issue_candidates.index, dtype=str)).fillna("").astype(str).isin(stable_ids)
            issue_candidates = issue_candidates[period_match | stable_match].head(50).copy()
    option_map: dict[str, tuple[str, str]] = {"키워드/기간 조건으로 원문 찾기": ("", "")}
    if not issue_candidates.empty:
        issue_candidates = issue_candidates.reset_index(drop=True)
        for idx, row in issue_candidates.iterrows():
            label = issue_option_label(row, idx + 1)
            option_map[label] = (
                display_text(row.get("period_issue_id", "")),
                display_text(row.get("stable_issue_id", "")),
            )

    f1, f2 = st.columns([1.4, 1])
    selected_issue_label = f1.selectbox("댓글까지 볼 이슈", list(option_map.keys()))
    comment_keyword = f2.text_input("댓글 내용에서 추가 검색", value="", placeholder="예: 비판, 응원, 정책")

    f3, f4, f5, f6 = st.columns([1.2, 0.9, 0.9, 0.9])
    sort_mode = f3.radio("댓글 정렬", ["반응 많은 순", "최신순", "공감/좋아요 많은 순", "답글 많은 순"], horizontal=True)
    public_mode = f4.checkbox("공개/시연용 보기", value=True, help="마스킹된 댓글 내용과 비식별 작성자만 표시합니다.")
    parent_limit = f5.slider("원문 수", 10, 300, 80, 10)
    comment_limit = f6.slider("댓글 수", 50, 1000, 300, 50)

    period_issue_id, stable_issue_id = option_map[selected_issue_label]
    parent_docs = parent_docs_for_issue_or_keyword(
        doc_comments,
        issue_comment_map,
        candidate=candidate,
        sources=selected_sources,
        start_date=start_date,
        end_date=end_date,
        keyword=keyword,
        period_issue_id=period_issue_id,
        stable_issue_id=stable_issue_id,
        limit=parent_limit,
    )

    if parent_docs.empty:
        st.info("현재 조건에 맞는 댓글 원문이 없습니다. 기간·자료원·키워드 조건을 넓혀보세요.")
        return

    parent_keys = parent_docs["parent_doc_key"].dropna().astype(str).tolist() if "parent_doc_key" in parent_docs.columns else []
    comment_rows = comments_for_parent_docs(
        comments,
        parent_keys,
        keyword=comment_keyword,
        start_date=start_date,
        end_date=end_date,
        sort_by=sort_mode,
        limit=comment_limit,
    )
    timeline = comment_timeline_for_parent_docs(comments, parent_keys)
    platform = comment_platform_for_parent_docs(comments, parent_docs, parent_keys)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("관련 원문", f"{len(parent_docs):,}")
    m2.metric("수집 댓글", fmt_num(parent_docs.get("collected_comment_count", pd.Series([0])).sum()))
    m3.metric("비식별 작성자", fmt_num(parent_docs.get("unique_author_hash_count", pd.Series([0])).sum()))
    m4.metric("댓글 반응", fmt_num(parent_docs.get("comment_like_sum", pd.Series([0])).sum() + parent_docs.get("sympathy_sum", pd.Series([0])).sum()))

    c1, c2 = st.columns([1.2, 1])
    with c1:
        fig = comment_timeline_chart(timeline, "collected_comment_count", "선택 원문의 댓글 일별 추이")
        if fig is not None:
            st.plotly_chart(fig, width="stretch")
    with c2:
        fig = comment_platform_chart(platform, "collected_comment_count", "자료원별 수집 댓글")
        if fig is not None:
            st.plotly_chart(fig, width="stretch")

    st.markdown("#### 관련 원문")
    parent_display = display_table(parent_docs, "parent_docs", max_rows=parent_limit)
    st.dataframe(
        parent_display,
        width="stretch",
        height=360,
        hide_index=True,
        column_config={"원문 링크": st.column_config.LinkColumn("원문 링크", display_text="열기")},
    )

    st.markdown("#### 댓글 목록")
    if comment_rows.empty:
        st.info("선택 원문에는 현재 조건에 맞는 댓글이 없습니다.")
    else:
        comment_kind = "comments" if public_mode else "comments_internal"
        st.dataframe(
            display_table(comment_rows, comment_kind, max_rows=comment_limit),
            width="stretch",
            height=460,
            hide_index=True,
            column_config={"원문 링크": st.column_config.LinkColumn("원문 링크", display_text="열기")},
        )

    st.markdown("#### 원문별 댓글 펼쳐보기")
    for _, doc in parent_docs.head(20).iterrows():
        doc_key = display_text(doc.get("parent_doc_key", ""))
        title = display_text(doc.get("parent_title", "")) or "(제목 없음)"
        count = first_value(doc, "collected_comment_count", default=0)
        with st.expander(f"{title} · 수집 댓글 {fmt_num(count)}"):
            url = display_text(doc.get("parent_url", ""))
            if url.startswith("http"):
                st.markdown(f"[원문 열기]({url})")
            one_doc_comments = comments_for_parent_docs(
                comments,
                [doc_key],
                keyword=comment_keyword,
                start_date=start_date,
                end_date=end_date,
                sort_by=sort_mode,
                limit=50,
            )
            st.dataframe(
                display_table(one_doc_comments, "comments" if public_mode else "comments_internal", max_rows=50),
                width="stretch",
                height=260,
                hide_index=True,
                column_config={"원문 링크": st.column_config.LinkColumn("원문 링크", display_text="열기")},
            )


def main():
    st.title("서울시장 후보 온라인 이슈 레이더")
    st.caption("후보별·자료원별·기간별 주요 이슈, 키워드 일별 추이, 플랫폼별 반응, 근거 문서, 원문·댓글 반응을 탐색합니다.")
    render_page_nav()

    if not DATA_DIR.exists():
        st.error("대시보드 데이터 폴더를 찾지 못했습니다. 분석 산출물 위치를 확인해 주세요.")
        st.stop()

    meta, briefs = load_dashboard_index_data(str(DATA_DIR))
    if briefs.empty:
        st.error("기간별 이슈 요약 데이터를 읽지 못했습니다. 분석 산출물 구성을 확인해 주세요.")
        st.stop()

    # Defaults
    date_candidates = []
    for c in ["window_start", "window_end"]:
        if c in briefs.columns:
            date_candidates.extend(pd.to_datetime(briefs[c], errors="coerce").dropna().tolist())
    min_dt = min(date_candidates).date() if date_candidates else date(2026, 1, 1)
    max_dt = max(date_candidates).date() if date_candidates else date(2026, 5, 31)

    candidates = [c for c in ["JWO", "OSH"] if c in briefs.get("candidate", pd.Series([])).astype(str).unique().tolist()]
    if not candidates:
        candidates = sorted(briefs.get("candidate", pd.Series(["JWO", "OSH"])).dropna().unique().tolist())
    source_values = sorted(briefs.get("source", pd.Series([], dtype=str)).dropna().unique().tolist())
    window_values = sorted(briefs.get("window_type", pd.Series([], dtype=str)).dropna().unique().tolist())

    with st.container(border=True):
        st.markdown("#### 조회 조건")
        r1c1, r1c2, r1c3 = st.columns([1.1, 1, 1])
        cand_label = r1c1.selectbox("후보", [CANDIDATE_LABELS.get(c, c) for c in candidates])
        start_date = r1c2.date_input("시작일", min_dt, min_value=min_dt, max_value=max_dt)
        end_date = r1c3.date_input("종료일", max_dt, min_value=min_dt, max_value=max_dt)

        r2c1, r2c2 = st.columns([1.2, 1])
        selected_source_labels = r2c1.multiselect(
            "자료원",
            [source_display(s) for s in source_values],
            default=[source_display(s) for s in source_values],
        )
        window_options = [label_window(w) for w in window_values]
        selected_window_labels = r2c2.multiselect("기간 유형", window_options, default=window_options)

        r3c1, r3c2, r3c3 = st.columns([1.4, 0.8, 1.2])
        keyword = r3c1.text_input("키워드", value="", placeholder="예: 칸쿤, 여론조사, 서소문, GTX")
        top_n = r3c2.slider("표시할 이슈 수", 5, 30, 10)
        view_mode_label = r3c3.radio("조회 방식", ["기간별 산출 결과 조회", "선택 기간 요약"], index=0)
        view_mode = "custom 기간 시계열 집계" if view_mode_label.startswith("선택") else "기존 window 결과"

    candidate = {v: k for k, v in CANDIDATE_LABELS.items()}.get(cand_label, cand_label)
    if start_date > end_date:
        st.error("시작일이 종료일보다 늦습니다.")
        st.stop()
    selected_sources = [reverse_lookup_available(SOURCE_LABELS, lab, source_values) for lab in selected_source_labels]
    selected_windows = [reverse_lookup_available(WINDOW_LABELS, lab, window_values) for lab in selected_window_labels]
    source_key = tuple(selected_sources)
    window_key = tuple(selected_windows)
    start_key = str(start_date)
    end_key = str(end_date)

    sections = ["기간별 주요 이슈", "키워드 일별 추이", "플랫폼별 반응", "근거 문서", "원문·댓글 반응", "설명"]
    active_section = st.radio("탐색 섹션", sections, horizontal=True, label_visibility="collapsed")

    if active_section == sections[0]:
        st.subheader("기간별 주요 이슈")
        st.caption("선택한 후보·기간·자료원 조건에 맞는 주요 이슈를 표로 먼저 보고, 하나를 골라 근거 문서까지 확인합니다.")
        if view_mode == "custom 기간 시계열 집계":
            timeseries = load_issue_timeseries_view(str(DATA_DIR), candidate, source_key, start_key, end_key)
            st.info("선택 기간 요약은 사전에 산출된 이슈 시계열을 선택 기간 기준으로 합산한 탐색 결과입니다.")
            top = query_custom_period_top10_from_timeseries(timeseries, candidate, selected_sources, start_date, end_date, keyword=keyword, top_n=top_n * 3)
        else:
            briefs_view = load_issue_briefs_view(str(DATA_DIR), candidate, source_key, window_key, start_key, end_key, keyword)
            top = query_existing_window_top10(briefs_view, candidate, selected_sources, start_date, end_date, selected_windows, keyword=keyword, max_rows=top_n * 3)
        if top.empty:
            st.info("조건에 맞는 이슈가 없습니다.")
        else:
            top = prepare_top_table(top, limit=top_n)
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("표시 이슈", f"{len(top):,}")
            c2.metric("분석 문서 수", fmt_num(top.get("doc_count_weighted", pd.Series([0])).sum(), 1))
            c3.metric("댓글 수", fmt_num(top.get("comment_count", top.get("comment_count_weighted", pd.Series([0]))).sum()))
            c4.metric("조회수", fmt_num(top.get("view_count", top.get("view_count_weighted", pd.Series([0]))).sum()))

            st.markdown("#### Top 이슈 요약표")
            st.dataframe(display_table(top, "top10"), width="stretch", height=360, hide_index=True)

            options = [issue_option_label(row, int(row["display_rank"])) for _, row in top.iterrows()]
            selected_label = st.selectbox("상세로 볼 이슈", options, index=0)
            selected_pos = options.index(selected_label)
            selected_row = top.iloc[selected_pos]

            st.divider()
            issue_card(selected_row, int(selected_row["display_rank"]))

            selected_evidence = load_evidence_for_issue_view(
                str(DATA_DIR),
                candidate,
                source_key,
                start_key,
                end_key,
                keyword,
                display_text(selected_row.get("period_issue_id", "")),
                display_text(selected_row.get("stable_issue_id", "")),
                display_text(selected_row.get("representative_url", "")),
                display_text(selected_row.get("representative_title", "")),
                display_text(selected_row.get("window_start", ""))[:10],
                display_text(selected_row.get("window_end", ""))[:10],
            )
            if selected_evidence.empty:
                st.warning("근거 문서 데이터가 준비되지 않아 선택 이슈의 근거 문서를 표시할 수 없습니다.")
            else:
                render_issue_evidence(selected_evidence)

    elif active_section == sections[1]:
        st.subheader("키워드별 일별 이슈 수치")
        st.caption("선택한 키워드가 날짜별·자료원별로 얼마나 등장했는지 보여줍니다.")
        if not keyword.strip():
            st.info("키워드를 입력하세요. 예: 칸쿤, 여론조사, 서소문, GTX")
        else:
            base = load_base_docs_view(str(DATA_DIR), candidate, source_key, start_key, end_key)
            if base.empty:
                st.warning("키워드 추이 분석용 원문 데이터가 준비되지 않았습니다.")
                st.stop()
            tl = keyword_daily_timeline(base, candidate, selected_sources, start_date, end_date, keyword)
            if tl.empty:
                st.info("조건에 맞는 문서가 없습니다.")
            else:
                metric_options = ["doc_count_weighted", "doc_count", "comment_count", "view_count", "like_count"]
                metric_label_map = {metric_label(m): m for m in metric_options}
                selected_metric_label = st.radio("지표", list(metric_label_map.keys()), horizontal=True)
                metric = metric_label_map[selected_metric_label]
                fig = line_chart(tl, metric, f"'{keyword}' 일별 {metric_label(metric)}")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(display_table(tl, "timeline"), width="stretch", hide_index=True)

    elif active_section == sections[2]:
        st.subheader("키워드 플랫폼별 반응")
        st.caption("선택한 키워드가 어느 자료원에서 많이 나타났고, 댓글·조회수 반응이 어디서 컸는지 비교합니다.")
        if not keyword.strip():
            st.info("키워드를 입력하세요.")
        else:
            base = load_base_docs_view(str(DATA_DIR), candidate, source_key, start_key, end_key)
            if base.empty:
                st.warning("base_docs_light 데이터가 없습니다.")
                st.stop()
            pr = keyword_platform_reactions(base, candidate, selected_sources, start_date, end_date, keyword)
            if pr.empty:
                st.info("조건에 맞는 문서가 없습니다.")
            else:
                metric_options = ["doc_count_weighted", "doc_count", "comment_count", "view_count", "like_count"]
                metric_label_map = {metric_label(m): m for m in metric_options}
                selected_metric_label = st.radio("플랫폼 지표", list(metric_label_map.keys()), horizontal=True, key="platform_metric")
                metric = metric_label_map[selected_metric_label]
                fig = bar_chart(pr, metric, f"'{keyword}' 플랫폼별 {metric_label(metric)}")
                st.plotly_chart(fig, width="stretch")
                st.dataframe(display_table(pr, "platform"), width="stretch", hide_index=True)

    elif active_section == sections[3]:
        st.subheader("근거 문서")
        st.caption("현재 조회 조건 또는 키워드와 연결된 대표 기사·영상·게시글을 확인합니다.")
        evidence = load_evidence_filter_view(str(DATA_DIR), candidate, source_key, start_key, end_key)
        if evidence.empty:
            st.warning("근거 문서 데이터가 준비되지 않았습니다.")
        else:
            ev = filter_evidence(evidence, candidate, selected_sources, start_date, end_date, keyword=keyword, limit=500)
            if ev.empty:
                st.info("조건에 맞는 근거 문서가 없습니다.")
            else:
                evidence_metrics(ev)
                st.dataframe(display_table(ev, "evidence"), width="stretch", height=560, hide_index=True)
                reaction_docs = top_reaction_documents(ev, limit=30)
                if not reaction_docs.empty:
                    with st.expander("댓글·조회 반응이 큰 문서 보기"):
                        st.caption("댓글 원문이 아니라, 수집된 댓글 수·조회수·좋아요 수가 큰 문서를 보여줍니다. 댓글 원문은 원문·댓글 반응 탭에서 확인할 수 있습니다.")
                        st.dataframe(display_table(reaction_docs, "reaction_docs"), width="stretch", height=360, hide_index=True)
    elif active_section == sections[4]:
        comments, doc_comments, issue_comment_map, _comment_timeseries = load_dashboard_comment_data(str(DATA_DIR))
        comment_briefs = load_issue_briefs_view(str(DATA_DIR), candidate, source_key, window_key, start_key, end_key, keyword)
        render_comment_drilldown(
            comment_briefs,
            comments,
            doc_comments,
            issue_comment_map,
            candidate,
            selected_sources,
            selected_windows,
            start_date,
            end_date,
            keyword,
        )

    elif active_section == sections[5]:
        st.subheader("분석 방법과 해석 기준")
        st.markdown(glossary_markdown())
        st.markdown(
            """
### 해석 주의

- 이 대시보드는 후보 관련 온라인 이슈의 **노출과 확산**을 탐색합니다.
- `기간별 산출 결과 조회`는 사전에 산출된 기간별 주요 이슈를 보여줍니다.
- `선택 기간 요약`은 기존 이슈 시계열을 선택 기간 기준으로 합산한 탐색 결과입니다.
- 댓글 원문은 원문 기사·영상·게시글과 함께 확인합니다. 공개/시연용 보기에서는 민감정보가 마스킹된 댓글을 기본 표시합니다.
- 최종 보고서 문장에는 대표 근거 문서를 확인한 뒤 정리한 문장을 쓰는 것을 권장합니다.
- 후보에 대한 긍정/부정 판단은 별도 감성·입장·프레임 분석 이후 가능합니다.
            """
        )


if __name__ == "__main__":
    main()

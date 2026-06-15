from __future__ import annotations

import pandas as pd
import plotly.express as px

try:
    from src.display_labels import (
        CANDIDATE_LABELS,
        ISSUE_TYPE_LABELS,
        SOURCE_LABELS,
        WINDOW_LABELS,
    )
except Exception:  # pragma: no cover
    CANDIDATE_LABELS = {"JWO": "정원오", "OSH": "오세훈"}
    SOURCE_LABELS = {}
    WINDOW_LABELS = {}
    ISSUE_TYPE_LABELS = {}

SOURCE_LABELS_WITH_ALL = {"ALL": "전체 자료원", **SOURCE_LABELS}
CANDIDATE_LABELS_WITH_ALL = {"ALL": "전체 후보", **CANDIDATE_LABELS}
WINDOW_LABELS_WITH_ALL = {"ALL": "전체 기간 유형", **WINDOW_LABELS}
SPIKE_LABELS = {"normal": "일반", "spike": "급등", "strong_spike": "강한 급등"}
GROUP_SCOPE_LABELS = {
    "candidate_window_type": "후보·기간유형별",
    "candidate_overall": "후보별 전체",
    "overall": "전체 후보 통합",
}
DISTRIBUTION_UNIT_LABELS = {
    "stable_issue": "연결 이슈 단위",
    "window_issue": "기간 이슈 row 단위",
}


def label_map_value(mapping: dict[str, str], value: object) -> str:
    text = "" if value is None else str(value)
    return mapping.get(text, text)


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate" in out.columns:
        out["후보"] = out["candidate"].map(lambda x: label_map_value(CANDIDATE_LABELS_WITH_ALL, x))
    if "source" in out.columns:
        out["자료원"] = out["source"].map(lambda x: label_map_value(SOURCE_LABELS_WITH_ALL, x))
    if "window_type" in out.columns:
        out["기간 유형"] = out["window_type"].map(lambda x: label_map_value(WINDOW_LABELS_WITH_ALL, x))
    if "issue_type" in out.columns:
        out["이슈 유형"] = out["issue_type"].map(lambda x: label_map_value(ISSUE_TYPE_LABELS, x))
    if "group_scope" in out.columns:
        out["검정 범위"] = out["group_scope"].map(lambda x: label_map_value(GROUP_SCOPE_LABELS, x))
    if "distribution_unit" in out.columns:
        out["분석 단위"] = out["distribution_unit"].map(lambda x: label_map_value(DISTRIBUTION_UNIT_LABELS, x))
    if "spike_level" in out.columns:
        out["급등 수준"] = out["spike_level"].map(lambda x: label_map_value(SPIKE_LABELS, x))
    if "issue_name" in out.columns:
        out["이슈명"] = out["issue_name"].astype(str)
    if "keyword" in out.columns:
        out["키워드"] = out["keyword"].astype(str)
    return out


def coerce_stats34_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = [
        "n_sources",
        "n_issue_types",
        "total_weight",
        "chi2",
        "dof",
        "p_value",
        "cramers_v",
        "min_expected",
        "low_expected_cell_share",
        "observed",
        "expected",
        "std_residual",
        "doc_count",
        "doc_count_weighted",
        "rolling_median_prev",
        "rolling_mad_prev",
        "robust_z",
        "baseline_window",
        "spike_threshold",
        "strong_threshold",
        "total_series_count",
        "active_days",
        "span_days",
        "total_count",
    ]
    for col in cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["is_spike", "is_strong_spike"]:
        if col in out.columns:
            out[col] = out[col].astype(str).str.lower().isin(["true", "1", "yes"])
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out


def observed_bar(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    work = add_labels(coerce_stats34_numeric(df))
    fig = px.bar(
        work,
        x="자료원",
        y="observed",
        color="이슈 유형",
        barmode="stack",
        labels={"observed": "관측 수"},
        title="자료원별 이슈유형 분포",
    )
    fig.update_layout(height=440, margin=dict(l=20, r=20, t=60, b=20), legend_title_text="이슈 유형")
    return fig


def residual_heatmap(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    work = add_labels(coerce_stats34_numeric(df))
    if "std_residual" not in work.columns:
        return None
    pivot = work.pivot_table(index="자료원", columns="이슈 유형", values="std_residual", aggfunc="mean")
    if pivot.empty:
        return None
    fig = px.imshow(
        pivot,
        text_auto=".1f",
        aspect="auto",
        labels=dict(color="표준화 잔차"),
        color_continuous_scale="RdBu_r",
        color_continuous_midpoint=0,
        title="자료원 × 이슈유형 표준화 잔차",
    )
    fig.update_layout(height=540, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def spike_line(df: pd.DataFrame, series_col: str = "keyword", series_value: str = "", candidate: str = "ALL", source: str = "ALL", title_prefix: str = "급등 탐지"):
    if df is None or df.empty:
        return None
    work = add_labels(coerce_stats34_numeric(df))
    if series_value:
        work = work[work[series_col].astype(str).eq(str(series_value))]
    if candidate != "ALL" and "candidate" in work.columns:
        work = work[work["candidate"].astype(str).eq(candidate)]
    if source != "ALL" and "source" in work.columns:
        work = work[work["source"].astype(str).eq(source)]
    if work.empty:
        return None
    line_color = "자료원" if "자료원" in work.columns else "source"
    fig = px.line(
        work.sort_values("date"),
        x="date",
        y="doc_count_weighted",
        color=line_color,
        line_group="source",
        markers=True,
        labels={"date": "날짜", "doc_count_weighted": "보정 문서 수"},
        title=title_prefix,
    )
    spikes = work[work.get("is_spike", False).astype(bool)] if "is_spike" in work.columns else work.head(0)
    if not spikes.empty:
        fig.add_scatter(
            x=spikes["date"],
            y=spikes["doc_count_weighted"],
            mode="markers",
            marker=dict(size=12, symbol="x"),
            name="급등일",
            text=spikes.get("robust_z", ""),
        )
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=60, b=20), legend_title_text="자료원")
    return fig

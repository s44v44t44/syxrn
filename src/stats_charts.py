from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from .display_labels import (
    CANDIDATE_LABELS as BASE_CANDIDATE_LABELS,
    ISSUE_TYPE_LABELS as BASE_ISSUE_TYPE_LABELS,
    SOURCE_LABELS as BASE_SOURCE_LABELS,
)

CANDIDATE_LABELS = dict(BASE_CANDIDATE_LABELS)
SOURCE_LABELS = {"ALL": "전체 자료원", **BASE_SOURCE_LABELS}
ISSUE_TYPE_LABELS = dict(BASE_ISSUE_TYPE_LABELS)
ISSUE_TYPE_LABELS.update({
    "campaign_presence": "선거 행보",
    "policy_livelihood": "민생·생활정책",
})


def add_display_cols(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate" in out.columns:
        out["후보"] = out["candidate"].map(CANDIDATE_LABELS).fillna(out["candidate"])
    if "source" in out.columns:
        out["자료원"] = out["source"].map(SOURCE_LABELS).fillna(out["source"])
    if "issue_type" in out.columns:
        out["이슈 유형"] = out["issue_type"].map(ISSUE_TYPE_LABELS).fillna(out["issue_type"])
    return out


def concentration_line_chart(df: pd.DataFrame, metric: str = "hhi"):
    if df is None or df.empty:
        return None
    d = add_display_cols(df)
    if "window_end" in d.columns:
        d["기간 종료일"] = pd.to_datetime(d["window_end"], errors="coerce")
    else:
        d["기간 종료일"] = range(len(d))
    y_label = {
        "hhi": "이슈 집중도(HHI)",
        "entropy_norm": "다양성 지수(표준화 entropy)",
        "effective_issue_count": "실질 이슈 수",
        "top1_share": "1위 이슈 점유율",
        "top3_share": "상위 3개 이슈 점유율",
    }.get(metric, metric)
    d[metric] = pd.to_numeric(d[metric], errors="coerce")
    fig = px.line(d.sort_values("기간 종료일"), x="기간 종료일", y=metric, color="후보", line_dash="자료원", markers=True, labels={metric: y_label})
    fig.update_layout(height=430, margin=dict(l=20, r=20, t=50, b=20))
    return fig


def issue_type_share_bar(df: pd.DataFrame):
    if df is None or df.empty:
        return None
    d = add_display_cols(df)
    d["share"] = pd.to_numeric(d["share"], errors="coerce")
    d["ci_low"] = pd.to_numeric(d.get("ci_low", pd.Series([0]*len(d))), errors="coerce")
    d["ci_high"] = pd.to_numeric(d.get("ci_high", pd.Series([0]*len(d))), errors="coerce")
    fig = go.Figure()
    for cand, sub in d.groupby("후보", dropna=False):
        sub = sub.sort_values("share", ascending=True)
        fig.add_trace(go.Bar(
            name=str(cand),
            y=sub["이슈 유형"],
            x=sub["share"],
            orientation="h",
            error_x=dict(type="data", symmetric=False, array=(sub["ci_high"]-sub["share"]).clip(lower=0), arrayminus=(sub["share"]-sub["ci_low"]).clip(lower=0)),
            hovertemplate="후보=%{name}<br>이슈=%{y}<br>점유율=%{x:.1%}<extra></extra>",
        ))
    fig.update_layout(barmode="group", height=max(420, 32*max(1, d["이슈 유형"].nunique())), xaxis_tickformat=".0%", xaxis_title="이슈 유형 점유율", yaxis_title="이슈 유형", margin=dict(l=20, r=20, t=50, b=20))
    return fig

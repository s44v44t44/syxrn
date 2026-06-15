from __future__ import annotations

import pandas as pd
import plotly.express as px

from .display_labels import chart_df_with_source_label, metric_label

COMMENT_LABELS = {
    "collected_comment_count": "수집 댓글 수",
    "comment_like_sum": "댓글 좋아요 합계",
    "sympathy_sum": "공감 합계",
    "antipathy_sum": "비공감 합계",
    "reply_count_sum": "답글 수",
    "unique_author_hash_count": "비식별 작성자 수",
}


def comment_timeline_chart(df: pd.DataFrame, metric: str = "collected_comment_count", title: str = "댓글 일별 추이"):
    if df is None or df.empty:
        return None
    work = chart_df_with_source_label(df.copy())
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work = work.dropna(subset=["date"]).sort_values(["date", "자료원" if "자료원" in work.columns else "source"])
    y_label = COMMENT_LABELS.get(metric, metric_label(metric))
    fig = px.line(
        work,
        x="date",
        y=metric,
        color="자료원" if "자료원" in work.columns else "source",
        markers=True,
        title=title,
        labels={"date": "날짜", metric: y_label, "source": "자료원", "자료원": "자료원"},
    )
    fig.update_layout(legend_title_text="자료원", hovermode="x unified")
    return fig


def comment_platform_chart(df: pd.DataFrame, metric: str = "collected_comment_count", title: str = "자료원별 댓글 반응"):
    if df is None or df.empty:
        return None
    work = chart_df_with_source_label(df.copy())
    y_label = COMMENT_LABELS.get(metric, metric_label(metric))
    return px.bar(
        work.sort_values(metric, ascending=False),
        x="자료원" if "자료원" in work.columns else "source",
        y=metric,
        title=title,
        labels={metric: y_label, "source": "자료원", "자료원": "자료원"},
    )


comment_line_chart = comment_timeline_chart
comment_bar_chart = comment_platform_chart


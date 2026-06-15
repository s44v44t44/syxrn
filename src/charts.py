from __future__ import annotations

import pandas as pd
import plotly.express as px

from .display_labels import label_source, metric_label


def _ensure_source_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source" in out.columns and "자료원" not in out.columns:
        out["자료원"] = out["source"].map(label_source)
    return out


def line_chart(df: pd.DataFrame, y: str, title: str):
    if df.empty:
        return None
    work = _ensure_source_label(df)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.sort_values(["date", "자료원" if "자료원" in work.columns else "source"])
    y_label = metric_label(y)
    fig = px.line(
        work,
        x="date",
        y=y,
        color="자료원" if "자료원" in work.columns else "source",
        markers=True,
        title=title.replace(y, y_label),
        labels={"date": "날짜", y: y_label, "source": "자료원", "자료원": "자료원"},
    )
    fig.update_layout(legend_title_text="자료원", hovermode="x unified")
    return fig


def bar_chart(df: pd.DataFrame, y: str, title: str):
    if df.empty:
        return None
    work = _ensure_source_label(df)
    y_label = metric_label(y)
    return px.bar(
        work.sort_values(y, ascending=False),
        x="자료원" if "자료원" in work.columns else "source",
        y=y,
        hover_data=[c for c in ["doc_count", "comment_count", "view_count", "like_count"] if c in work.columns],
        title=title.replace(y, y_label),
        labels={"source": "자료원", "자료원": "자료원", y: y_label},
    )

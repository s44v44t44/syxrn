from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stats34_methods import read_table  # noqa: E402
from src.stats34_charts import (  # noqa: E402
    CANDIDATE_LABELS_WITH_ALL,
    GROUP_SCOPE_LABELS,
    SOURCE_LABELS_WITH_ALL,
    WINDOW_LABELS_WITH_ALL,
    add_labels,
    coerce_stats34_numeric,
    observed_bar,
    residual_heatmap,
    spike_line,
)

STATS_DIR = Path("dashboard_stats")
MODE_LABELS = {
    "balanced": "기본 분석",
    "strict": "보수적 분석",
    "exploratory": "탐색 분석",
}
SPIKE_FILTERS = {
    "all": "전체",
    "spike": "급등 이상",
    "strong_spike": "강한 급등만",
}

st.set_page_config(page_title="분포 검정·급등 탐지 | 이슈 레이더", layout="wide", initial_sidebar_state="collapsed")
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


def render_page_nav():
    nav1, nav2, nav3 = st.columns(3)
    nav1.page_link("app.py", label="이슈 레이더")
    nav2.page_link("pages/6_통계_분석.py", label="통계 분석")
    nav3.page_link("pages/7_분포검정_급등탐지.py", label="분포 검정·급등 탐지")


st.title("자료원별 이슈 분포와 급등 신호 분석")
st.caption("온라인 이슈가 자료원에 따라 어떻게 다르게 구성되는지, 특정 키워드와 이슈가 언제 평소보다 크게 증가했는지 확인합니다.")
render_page_nav()


def read_stats(stem: str) -> pd.DataFrame:
    try:
        return coerce_stats34_numeric(read_table(STATS_DIR, stem))
    except Exception:
        return pd.DataFrame()


def options_with_labels(values: list[str], labels: dict[str, str]) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    seen: set[str] = set()
    for value in values:
        label = labels.get(value, value)
        if label in seen:
            label = f"{label} [{value}]"
        seen.add(label)
        out.append((label, value))
    return out


def apply_spike_level(df: pd.DataFrame, level: str) -> pd.DataFrame:
    if df.empty:
        return df
    if level == "strong_spike" and "is_strong_spike" in df.columns:
        return df[df["is_strong_spike"].astype(bool)].copy()
    if level == "spike" and "is_spike" in df.columns:
        return df[df["is_spike"].astype(bool)].copy()
    return df


def filter_date(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df.empty or "date" not in df.columns:
        return df
    date_values = pd.to_datetime(df["date"], errors="coerce")
    return df[(date_values >= pd.to_datetime(start_date)) & (date_values <= pd.to_datetime(end_date))].copy()


def format_p_value(value: object) -> str:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return "-"
    if val < 0.001:
        return "< 0.001"
    return f"{val:.3f}"


def interpret_v(value: object) -> str:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return "-"
    if val < 0.10:
        return "매우 약함"
    if val < 0.20:
        return "약한 차이"
    if val < 0.30:
        return "뚜렷한 차이"
    return "강한 차이"


def expected_count_note(value: object) -> str:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return "-"
    if val >= 0.20:
        return "기대빈도 주의"
    if val > 0:
        return "일부 낮음"
    return "양호"


def residual_note(value: object) -> str:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return "-"
    if val >= 2:
        return "기대보다 많음"
    if val <= -2:
        return "기대보다 적음"
    return "기대와 유사"


def v_range_label(df: pd.DataFrame) -> str:
    if df.empty or "cramers_v" not in df.columns:
        return "-"
    vals = pd.to_numeric(df["cramers_v"], errors="coerce").dropna()
    if vals.empty:
        return "-"
    return f"{vals.min():.3f}~{vals.max():.3f}"


def distribution_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    show = add_labels(df)
    out = pd.DataFrame()
    out["후보"] = show.get("후보", "")
    out["기간"] = show.get("기간 유형", "")
    out["검정 범위"] = show.get("검정 범위", "")
    out["자료원 수"] = show.get("n_sources", "")
    out["이슈유형 수"] = show.get("n_issue_types", "")
    out["분석 이슈 수"] = show.get("total_weight", pd.Series(dtype=float)).round(0).astype("Int64")
    out["χ²"] = show.get("chi2", pd.Series(dtype=float)).round(2)
    out["자유도"] = show.get("dof", "")
    out["p-value"] = show.get("p_value", pd.Series(dtype=float)).map(format_p_value)
    out["Cramér's V"] = show.get("cramers_v", pd.Series(dtype=float)).round(3)
    out["효과크기 해석"] = show.get("cramers_v", pd.Series(dtype=float)).map(interpret_v)
    out["기대빈도 점검"] = show.get("low_expected_cell_share", pd.Series(dtype=float)).map(expected_count_note)
    return out


def residual_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    show = add_labels(df)
    out = pd.DataFrame()
    out["후보"] = show.get("후보", "")
    out["기간"] = show.get("기간 유형", "")
    out["자료원"] = show.get("자료원", "")
    out["이슈 유형"] = show.get("이슈 유형", "")
    out["관측 이슈 수"] = show.get("observed", pd.Series(dtype=float)).round(1)
    out["기대 이슈 수"] = show.get("expected", pd.Series(dtype=float)).round(1)
    out["표준화 잔차"] = show.get("std_residual", pd.Series(dtype=float)).round(2)
    out["해석"] = show.get("std_residual", pd.Series(dtype=float)).map(residual_note)
    return out.sort_values("표준화 잔차", key=lambda s: s.abs(), ascending=False)


def spike_table(df: pd.DataFrame, include_issue_name: bool = False) -> pd.DataFrame:
    if df.empty:
        return df
    show = add_labels(df)
    out = pd.DataFrame()
    out["날짜"] = pd.to_datetime(show["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    if include_issue_name:
        out["이슈명"] = show.get("이슈명", "")
    else:
        out["키워드"] = show.get("키워드", "")
    out["후보"] = show.get("후보", "")
    out["자료원"] = show.get("자료원", "")
    out["분석 문서 수"] = show.get("doc_count_weighted", pd.Series(dtype=float)).round(1)
    out["이전 기준선"] = show.get("rolling_median_prev", pd.Series(dtype=float)).round(1)
    out["급등 점수"] = show.get("robust_z", pd.Series(dtype=float)).round(2)
    out["급등 수준"] = show.get("급등 수준", "")
    return out.sort_values(["날짜", "급등 점수"], ascending=[False, False])


def method_summary_table(qc_df: pd.DataFrame, issue_series_count: int) -> pd.DataFrame:
    rows = []
    if not tests.empty:
        rows.append({"구분": "자료원별 이슈유형 분포 검정", "산출 결과": f"{len(tests):,}개 검정", "상태": "완료"})
    if not keyword_spikes.empty:
        n = int(keyword_spikes.get("is_spike", pd.Series(dtype=bool)).sum())
        rows.append({"구분": "키워드 급등 탐지", "산출 결과": f"{n:,}개 급등일", "상태": "완료"})
    if not issue_spikes.empty:
        n = int(issue_spikes.get("is_spike", pd.Series(dtype=bool)).sum())
        rows.append({"구분": "이슈 급등 탐지", "산출 결과": f"{n:,}개 급등일 / {issue_series_count:,}개 이슈 시계열", "상태": "완료"})
    if qc_df.empty:
        return pd.DataFrame(rows)
    status = "확인 완료" if qc_df.get("status", pd.Series()).astype(str).eq("ok").all() else "일부 확인 필요"
    rows.append({"구분": "자료 구성 점검", "산출 결과": "분석 테이블 구성 확인", "상태": status})
    return pd.DataFrame(rows)


if not STATS_DIR.exists():
    st.error("분포 검정·급등 탐지 산출물 폴더를 찾지 못했습니다. 분석 산출물 구성을 확인해 주세요.")
    st.stop()

tests = read_stats("source_issue_type_test")
residuals = read_stats("source_issue_type_residuals")
observed = read_stats("source_issue_type_observed")
keyword_spikes = read_stats("keyword_spike_detection")
issue_spikes = read_stats("issue_spike_detection")
issue_summary = read_stats("issue_spike_series_summary")
qc = read_stats("99_dashboard_stats_34_qc")

if tests.empty and keyword_spikes.empty and issue_spikes.empty:
    st.error("분포 검정·급등 탐지 산출물을 읽지 못했습니다.")
    st.stop()

date_values: list[pd.Timestamp] = []
for frame in [keyword_spikes, issue_spikes]:
    if not frame.empty and "date" in frame.columns:
        date_values.extend(pd.to_datetime(frame["date"], errors="coerce").dropna().tolist())
min_date = min(date_values).date() if date_values else date(2026, 1, 1)
max_date = max(date_values).date() if date_values else date(2026, 5, 31)

mode_values = [m for m in ["balanced", "strict", "exploratory"] if m in tests.get("filter_mode", pd.Series(dtype=str)).dropna().astype(str).unique().tolist()]
mode_values = mode_values or sorted(tests.get("filter_mode", pd.Series(["balanced"])).dropna().astype(str).unique().tolist())
mode_labels = [MODE_LABELS.get(m, m) for m in mode_values]

candidate_values = sorted(set(tests.get("candidate", pd.Series(dtype=str)).dropna().astype(str)) | set(keyword_spikes.get("candidate", pd.Series(dtype=str)).dropna().astype(str)) | set(issue_spikes.get("candidate", pd.Series(dtype=str)).dropna().astype(str)))
candidate_values = ["ALL"] + [c for c in candidate_values if c != "ALL"]
source_values = sorted(set(observed.get("source", pd.Series(dtype=str)).dropna().astype(str)) | set(keyword_spikes.get("source", pd.Series(dtype=str)).dropna().astype(str)) | set(issue_spikes.get("source", pd.Series(dtype=str)).dropna().astype(str)))
source_values = ["ALL"] + [s for s in source_values if s != "ALL"]
window_values = sorted(tests.get("window_type", pd.Series(dtype=str)).dropna().astype(str).unique().tolist())
window_values = ["ALL"] + [w for w in window_values if w != "ALL"]
group_values = sorted(tests.get("group_scope", pd.Series(["candidate_window_type"])).dropna().astype(str).unique().tolist())

with st.container(border=True):
    st.markdown("#### 분석 조건")
    r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1])
    mode_label = r1c1.selectbox("분석 모드", mode_labels, index=0)
    filter_mode = mode_values[mode_labels.index(mode_label)]
    candidate_options = options_with_labels(candidate_values, CANDIDATE_LABELS_WITH_ALL)
    candidate_label = r1c2.selectbox("후보", [label for label, _ in candidate_options], index=0)
    candidate = dict(candidate_options)[candidate_label]
    source_options = options_with_labels(source_values, SOURCE_LABELS_WITH_ALL)
    source_label = r1c3.selectbox("자료원", [label for label, _ in source_options], index=0)
    source = dict(source_options)[source_label]
    group_options = options_with_labels(group_values, GROUP_SCOPE_LABELS)
    group_label = r1c4.selectbox("검정 범위", [label for label, _ in group_options], index=0)
    group_scope = dict(group_options)[group_label]

    r2c1, r2c2, r2c3, r2c4 = st.columns([1, 1, 1, 1])
    window_options = options_with_labels(window_values, WINDOW_LABELS_WITH_ALL)
    window_label = r2c1.selectbox("기간 유형", [label for label, _ in window_options], index=0)
    window_type = dict(window_options)[window_label]
    start = r2c2.date_input("분석 시작일", min_date, min_value=min_date, max_value=max_date)
    end = r2c3.date_input("분석 종료일", max_date, min_value=min_date, max_value=max_date)
    spike_label = r2c4.selectbox("급등 표시", list(SPIKE_FILTERS.values()), index=2)
    spike_level = {v: k for k, v in SPIKE_FILTERS.items()}[spike_label]
    if start > end:
        st.error("시작일이 종료일보다 늦습니다.")
        st.stop()


def filter_distribution(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if "filter_mode" in out.columns:
        out = out[out["filter_mode"].astype(str).eq(filter_mode)]
    if "group_scope" in out.columns:
        out = out[out["group_scope"].astype(str).eq(group_scope)]
    if candidate != "ALL" and "candidate" in out.columns:
        out = out[out["candidate"].astype(str).eq(candidate)]
    if source != "ALL" and "source" in out.columns:
        out = out[out["source"].astype(str).eq(source)]
    if window_type != "ALL" and "window_type" in out.columns:
        out = out[out["window_type"].astype(str).eq(window_type)]
    return out


def filter_spikes(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    if candidate != "ALL" and "candidate" in out.columns:
        out = out[out["candidate"].astype(str).eq(candidate)]
    if source != "ALL" and "source" in out.columns:
        out = out[out["source"].astype(str).eq(source)]
    out = filter_date(out, start, end)
    return apply_spike_level(out, spike_level)


def selected_spike_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    if spike_level == "all" and "is_spike" in df.columns:
        return int(df["is_spike"].astype(bool).sum())
    return len(df)


ft = filter_distribution(tests)
fr = filter_distribution(residuals)
fo = filter_distribution(observed)
fk = filter_spikes(keyword_spikes)
fi = filter_spikes(issue_spikes)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("분포 검정 수", f"{len(ft):,}")
k2.metric("비교 항목 수", f"{len(fr):,}")
k3.metric("연관강도 범위", v_range_label(ft))
k4.metric("선택 기준 키워드 급등일", f"{selected_spike_count(fk):,}")
k5.metric("선택 기준 이슈 급등일", f"{selected_spike_count(fi):,}")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["자료원별 분포 차이", "분포 차이 세부 항목", "키워드 급등", "이슈 급등", "방법 및 한계"])

with tab1:
    st.subheader("자료원별 이슈유형 분포 차이")
    st.caption("자료원별로 이슈유형 분포가 서로 다른지 검정하고, 차이의 크기를 Cramér's V로 함께 확인합니다.")
    if ft.empty:
        st.warning("조건에 맞는 검정 결과가 없습니다.")
    else:
        st.dataframe(distribution_table(ft), width="stretch", hide_index=True)
    if not fo.empty:
        fig = observed_bar(fo)
        if fig:
            st.plotly_chart(fig, width="stretch")

with tab2:
    st.subheader("분포 차이가 두드러진 자료원·이슈유형")
    st.caption("표준화 잔차 기준으로, 양수는 기대보다 많이 관측된 조합이고 음수는 기대보다 적게 관측된 조합입니다.")
    if fr.empty:
        st.warning("조건에 맞는 잔차 결과가 없습니다.")
    else:
        fig = residual_heatmap(fr)
        if fig:
            st.plotly_chart(fig, width="stretch")
        st.dataframe(residual_table(fr), width="stretch", height=420, hide_index=True)

with tab3:
    st.subheader("키워드 급등 탐지")
    if keyword_spikes.empty:
        st.warning("키워드 급등 산출물이 없습니다.")
    else:
        key_pool = filter_date(keyword_spikes, start, end)
        key_pool = key_pool.copy()
        key_pool["_spike_score"] = key_pool.get("is_strong_spike", pd.Series(False, index=key_pool.index)).astype(bool).astype(int) * 2 + key_pool.get("is_spike", pd.Series(False, index=key_pool.index)).astype(bool).astype(int)
        key_rank = key_pool.groupby("keyword", dropna=False).agg(spikes=("is_spike", "sum"), max_z=("robust_z", "max"), rows=("keyword", "size")).reset_index()
        key_rank = key_rank.sort_values(["spikes", "max_z", "rows"], ascending=[False, False, False])
        keywords = key_rank["keyword"].astype(str).tolist() or sorted(keyword_spikes["keyword"].dropna().astype(str).unique().tolist())
        selected_keyword = st.selectbox("키워드", keywords)
        chart_df = filter_spikes(keyword_spikes[keyword_spikes["keyword"].astype(str).eq(selected_keyword)])
        fig = spike_line(keyword_spikes, series_col="keyword", series_value=selected_keyword, candidate=candidate, source=source, title_prefix=f"키워드 급등: {selected_keyword}")
        if fig:
            st.plotly_chart(fig, width="stretch")
        if chart_df.empty:
            st.info("조건에 맞는 키워드 급등일이 없습니다.")
        else:
            st.dataframe(spike_table(chart_df), width="stretch", height=420, hide_index=True)

with tab4:
    st.subheader("이슈 급등 탐지")
    if issue_spikes.empty:
        st.warning("이슈 급등 산출물이 없습니다.")
    else:
        issue_pool = filter_date(issue_spikes, start, end)
        if candidate != "ALL":
            issue_pool = issue_pool[issue_pool["candidate"].astype(str).eq(candidate)]
        if source != "ALL":
            issue_pool = issue_pool[issue_pool["source"].astype(str).eq(source)]
        issue_pool = apply_spike_level(issue_pool, spike_level)
        if issue_pool.empty:
            st.info("조건에 맞는 이슈 급등일이 없습니다.")
        else:
            rank = issue_pool.groupby(["keyword", "candidate", "source"], dropna=False).agg(
                issue_name=("issue_name", "last"),
                spikes=("is_spike", "sum"),
                strong_spikes=("is_strong_spike", "sum"),
                max_z=("robust_z", "max"),
                total=("total_series_count", "max"),
            ).reset_index()
            rank = rank.sort_values(["strong_spikes", "spikes", "max_z", "total"], ascending=[False, False, False, False])
            options = []
            for i, (_, row) in enumerate(rank.iterrows(), start=1):
                source_text = SOURCE_LABELS_WITH_ALL.get(str(row["source"]), str(row["source"]))
                cand_text = CANDIDATE_LABELS_WITH_ALL.get(str(row["candidate"]), str(row["candidate"]))
                label = f'{i}. {row["issue_name"]} · {cand_text} · {source_text} · 급등 {int(row["spikes"])}일'
                options.append((label, str(row["keyword"]), str(row["candidate"]), str(row["source"])))
            selected_label = st.selectbox("이슈", [o[0] for o in options])
            _, selected_issue, selected_candidate, selected_source = options[[o[0] for o in options].index(selected_label)]
            issue_chart_source = issue_spikes[
                issue_spikes["keyword"].astype(str).eq(selected_issue)
                & issue_spikes["candidate"].astype(str).eq(selected_candidate)
                & issue_spikes["source"].astype(str).eq(selected_source)
            ].copy()
            issue_title = issue_chart_source["issue_name"].dropna().astype(str).iloc[0] if not issue_chart_source.empty else selected_issue
            fig = spike_line(issue_chart_source, series_col="keyword", series_value=selected_issue, candidate=selected_candidate, source=selected_source, title_prefix=f"이슈 급등: {issue_title}")
            if fig:
                st.plotly_chart(fig, width="stretch")
            sub = filter_date(issue_chart_source, start, end)
            sub = apply_spike_level(sub, spike_level)
            st.dataframe(spike_table(sub, include_issue_name=True), width="stretch", height=420, hide_index=True)

with tab5:
    st.subheader("분석 방법 및 해석 한계")
    st.markdown(
        """
### 분포 검정
- 중복 집계를 줄이기 위해 동일 이슈는 하나의 분석 단위로 묶었습니다.
- 자료원과 이슈유형의 분포 차이는 카이제곱 검정과 Cramér's V로 확인했습니다.
- p-value는 자동 분류된 이슈 기준의 진단값이므로, 효과크기와 잔차를 함께 해석해야 합니다.

### 급등 탐지
- 급등일은 이전 기간의 중앙값과 MAD를 기준으로 산출한 강건 z-score 신호입니다.
- 급등은 원인효과나 지지율 변화를 뜻하지 않습니다.
        """
    )
    issue_series_count = len(issue_summary) if not issue_summary.empty else 0
    summary = method_summary_table(qc, issue_series_count)
    if not summary.empty:
        st.dataframe(summary, width="stretch", hide_index=True)

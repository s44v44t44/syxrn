from __future__ import annotations

from pathlib import Path
from datetime import date
import sys

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.stats_queries import read_stats_table, coerce_numeric, filter_date_overlap  # noqa: E402
from src.stats_charts import concentration_line_chart, issue_type_share_bar, add_display_cols  # noqa: E402
from src.display_labels import CANDIDATE_LABELS, ISSUE_TYPE_LABELS, SOURCE_LABELS, WINDOW_LABELS  # noqa: E402

STATS_DIR = Path("dashboard_stats")
MODE_LABELS = {
    "balanced": "기본 분석",
    "strict": "보수적 분석",
    "exploratory": "탐색 분석",
}
SOURCE_LABELS_WITH_ALL = {"ALL": "전체 자료원", **SOURCE_LABELS}
WINDOW_LABELS_WITH_ALL = {"ALL": "전체 기간 유형", **WINDOW_LABELS}
CANDIDATE_LABELS_WITH_ALL = {"ALL": "전체 후보", **CANDIDATE_LABELS}

st.set_page_config(page_title="통계 분석 | 이슈 레이더", layout="wide", initial_sidebar_state="collapsed")
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


st.title("이슈 집중도와 후보별 이슈유형 분석")
st.caption("후보별 온라인 이슈가 특정 주제에 집중되는지, 이슈유형 구성은 어떻게 다른지 통계 지표로 확인합니다.")
render_page_nav()

if not STATS_DIR.exists():
    st.error("통계 분석 산출물 폴더를 찾지 못했습니다. 분석 산출물 구성을 확인해 주세요.")
    st.stop()

conc = coerce_numeric(read_stats_table(STATS_DIR, "issue_concentration_metrics"))
share = coerce_numeric(read_stats_table(STATS_DIR, "candidate_issue_type_share_ci"))
diff = coerce_numeric(read_stats_table(STATS_DIR, "candidate_issue_type_share_diff"))
qc = read_stats_table(STATS_DIR, "99_dashboard_stats_qc")

if conc.empty and share.empty:
    st.error("통계 분석 테이블을 읽지 못했습니다.")
    st.stop()

all_dates = []
if not conc.empty and "window_start" in conc.columns:
    all_dates.extend(pd.to_datetime(conc["window_start"], errors="coerce").dropna().tolist())
    all_dates.extend(pd.to_datetime(conc["window_end"], errors="coerce").dropna().tolist())
min_date = min(all_dates).date() if all_dates else date(2026, 1, 1)
max_date = max(all_dates).date() if all_dates else date(2026, 5, 31)

cands = sorted(set(conc.get("candidate", pd.Series(dtype=str)).dropna().astype(str)) | set(share.get("candidate", pd.Series(dtype=str)).dropna().astype(str)))
sources = sorted(set(conc.get("source", pd.Series(dtype=str)).dropna().astype(str)) | set(share.get("source", pd.Series(dtype=str)).dropna().astype(str)))
wtypes = sorted(set(conc.get("window_type", pd.Series(dtype=str)).dropna().astype(str)))
fmodes = sorted(set(conc.get("filter_mode", pd.Series(dtype=str)).dropna().astype(str)) | set(share.get("filter_mode", pd.Series(dtype=str)).dropna().astype(str)))

mode_values = [m for m in ["balanced", "strict", "exploratory"] if m in fmodes] or (fmodes or ["balanced"])
mode_labels = [MODE_LABELS.get(m, m) for m in mode_values]
candidate_values = ["ALL"] + [c for c in cands if c != "ALL"]
source_values = ["ALL"] + [s for s in sources if s != "ALL"]
window_values = ["ALL"] + [w for w in wtypes if w != "ALL"]


def options_with_labels(values: list[str], labels: dict[str, str]) -> list[tuple[str, str]]:
    out = []
    seen = set()
    for value in values:
        label = labels.get(value, value)
        if label in seen:
            label = f"{label} [{value}]"
        seen.add(label)
        out.append((label, value))
    return out


def pct(value: object) -> str:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return "-"
    return f"{val:.1%}"


def num(value: object, digits: int = 3) -> str:
    val = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(val):
        return "-"
    return f"{val:.{digits}f}"


def issue_type_label(value: object) -> str:
    text = "" if value is None else str(value)
    return ISSUE_TYPE_LABELS.get(text, text)


def concentration_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    show = add_display_cols(df)
    out = pd.DataFrame()
    out["후보"] = show.get("후보", "")
    out["자료원"] = show.get("자료원", "")
    out["기간 유형"] = show.get("window_type", "").map(lambda x: WINDOW_LABELS.get(str(x), str(x)))
    out["시작일"] = show.get("window_start", "")
    out["종료일"] = show.get("window_end", "")
    out["이슈 수"] = show.get("issue_count", "")
    out["집중도(HHI)"] = show.get("hhi", pd.Series(dtype=float)).map(lambda x: num(x, 3))
    out["다양성 지수"] = show.get("entropy_norm", pd.Series(dtype=float)).map(lambda x: num(x, 3))
    out["실질 이슈 수"] = show.get("effective_issue_count", pd.Series(dtype=float)).map(lambda x: num(x, 1))
    out["최대 이슈 점유율"] = show.get("top1_share", pd.Series(dtype=float)).map(pct)
    out["상위 3개 이슈 점유율"] = show.get("top3_share", pd.Series(dtype=float)).map(pct)
    out["대표 이슈"] = show.get("top1_issue_name", "")
    out["대표 이슈 유형"] = show.get("top1_issue_type", "").map(issue_type_label)
    return out


def share_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    show = add_display_cols(df)
    out = pd.DataFrame()
    out["후보"] = show.get("후보", "")
    out["자료원"] = show.get("자료원", "")
    out["이슈 유형"] = show.get("이슈 유형", "")
    out["점유율"] = show.get("share", pd.Series(dtype=float)).map(pct)
    out["95% 하한"] = show.get("ci_low", pd.Series(dtype=float)).map(pct)
    out["95% 상한"] = show.get("ci_high", pd.Series(dtype=float)).map(pct)
    out["분석 이슈 수"] = show.get("n_issue_rows", "")
    return out


def diff_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    show = add_display_cols(df)
    out = pd.DataFrame()
    out["자료원"] = show.get("source", "").map(lambda x: SOURCE_LABELS_WITH_ALL.get(str(x), str(x)))
    out["기간 유형"] = show.get("window_type", "").map(lambda x: WINDOW_LABELS_WITH_ALL.get(str(x), str(x)))
    out["이슈 유형"] = show.get("issue_type", "").map(issue_type_label)
    out["정원오 점유율"] = show.get("share_JWO", pd.Series(dtype=float)).map(pct)
    out["오세훈 점유율"] = show.get("share_OSH", pd.Series(dtype=float)).map(pct)
    out["차이(정원오-오세훈)"] = show.get("share_diff", pd.Series(dtype=float)).map(pct)
    out["보수적 하한"] = show.get("share_diff_ci_low_conservative", pd.Series(dtype=float)).map(pct)
    out["보수적 상한"] = show.get("share_diff_ci_high_conservative", pd.Series(dtype=float)).map(pct)
    return out


def aggregate_share_for_view(df: pd.DataFrame, source_value: str, window_value: str) -> pd.DataFrame:
    if df.empty:
        return df
    work = df.copy()
    for col in ["weight_sum", "total_weight", "n_issue_rows"]:
        if col in work.columns:
            work[col] = pd.to_numeric(work[col], errors="coerce").fillna(0)
    denom_keys = [c for c in ["candidate", "source", "window_type", "filter_mode"] if c in work.columns]
    denom = work.drop_duplicates(denom_keys).groupby("candidate", dropna=False)["total_weight"].sum().reset_index(name="total_weight")
    numer = work.groupby(["candidate", "issue_type"], dropna=False).agg(
        weight_sum=("weight_sum", "sum"),
        n_issue_rows=("n_issue_rows", "sum"),
    ).reset_index()
    out = numer.merge(denom, on="candidate", how="left")
    out["share"] = out["weight_sum"] / out["total_weight"].replace(0, pd.NA)
    out["ci_low"] = out["share"]
    out["ci_high"] = out["share"]
    out["source"] = source_value if source_value != "ALL" else "ALL"
    out["window_type"] = window_value if window_value != "ALL" else "ALL"
    return out


def aggregate_diff_from_share(df: pd.DataFrame, source_value: str, window_value: str) -> pd.DataFrame:
    if df.empty:
        return df
    agg = aggregate_share_for_view(df, source_value, window_value)
    if agg.empty or not {"JWO", "OSH"}.issubset(set(agg["candidate"].astype(str))):
        return pd.DataFrame()
    jwo = agg[agg["candidate"].astype(str).eq("JWO")][["issue_type", "share"]].rename(columns={"share": "share_JWO"})
    osh = agg[agg["candidate"].astype(str).eq("OSH")][["issue_type", "share"]].rename(columns={"share": "share_OSH"})
    out = jwo.merge(osh, on="issue_type", how="inner")
    out["share_diff"] = out["share_JWO"] - out["share_OSH"]
    out["share_diff_ci_low_conservative"] = out["share_diff"]
    out["share_diff_ci_high_conservative"] = out["share_diff"]
    out["source"] = source_value if source_value != "ALL" else "ALL"
    out["window_type"] = window_value if window_value != "ALL" else "ALL"
    return out.sort_values("share_diff", key=lambda s: s.abs(), ascending=False)


with st.container(border=True):
    st.markdown("#### 통계 분석 조건")
    r1c1, r1c2, r1c3, r1c4 = st.columns([1, 1, 1, 1])
    mode_label = r1c1.selectbox("분석 모드", mode_labels, index=0)
    filter_mode = mode_values[mode_labels.index(mode_label)]
    candidate_options = options_with_labels(candidate_values, CANDIDATE_LABELS_WITH_ALL)
    candidate_label = r1c2.selectbox("후보", [label for label, _ in candidate_options], index=0)
    candidate = dict(candidate_options)[candidate_label]
    source_options = options_with_labels(source_values, SOURCE_LABELS_WITH_ALL)
    source_label = r1c3.selectbox("자료원", [label for label, _ in source_options], index=0)
    source = dict(source_options)[source_label]
    window_options = options_with_labels(window_values, WINDOW_LABELS_WITH_ALL)
    window_label = r1c4.selectbox("기간 유형", [label for label, _ in window_options], index=0)
    window_type = dict(window_options)[window_label]

    r2c1, r2c2 = st.columns([1, 1])
    start = r2c1.date_input("시작일", min_date, min_value=min_date, max_value=max_date)
    end = r2c2.date_input("종료일", max_date, min_value=min_date, max_value=max_date)
    if start > end:
        st.error("시작일이 종료일보다 늦습니다.")
        st.stop()

cf = conc.copy()
sf = share.copy()
df_diff = diff.copy()
for frame_name, frame in [("conc", cf), ("share", sf), ("diff", df_diff)]:
    if frame.empty:
        continue
    if "filter_mode" in frame.columns:
        frame = frame[frame["filter_mode"].astype(str).eq(filter_mode)].copy()
    if candidate != "ALL" and "candidate" in frame.columns:
        frame = frame[frame["candidate"].astype(str).eq(candidate)].copy()
    if source != "ALL" and "source" in frame.columns:
        frame = frame[frame["source"].astype(str).eq(source)].copy()
    if window_type != "ALL" and "window_type" in frame.columns:
        frame = frame[frame["window_type"].astype(str).eq(window_type)].copy()
    frame = filter_date_overlap(frame, start, end)
    if frame_name == "conc":
        cf = frame
    elif frame_name == "share":
        sf = frame
    else:
        df_diff = frame

# Candidate issue-type shares are overall distributions in the current stats builder; if a date filter removes all, fall back to overall.
if sf.empty:
    sf = share.copy()
    if "filter_mode" in sf.columns:
        sf = sf[sf["filter_mode"].astype(str).eq(filter_mode)].copy()
    if candidate != "ALL" and "candidate" in sf.columns:
        sf = sf[sf["candidate"].astype(str).eq(candidate)].copy()
    if source != "ALL" and "source" in sf.columns:
        sf = sf[sf["source"].astype(str).eq(source)].copy()

sf_view = aggregate_share_for_view(sf, source, window_type)
diff_share_base = share.copy()
if "filter_mode" in diff_share_base.columns:
    diff_share_base = diff_share_base[diff_share_base["filter_mode"].astype(str).eq(filter_mode)].copy()
if source != "ALL" and "source" in diff_share_base.columns:
    diff_share_base = diff_share_base[diff_share_base["source"].astype(str).eq(source)].copy()
if window_type != "ALL" and "window_type" in diff_share_base.columns:
    diff_share_base = diff_share_base[diff_share_base["window_type"].astype(str).eq(window_type)].copy()
df_diff_view = aggregate_diff_from_share(diff_share_base, source, window_type)

k1, k2, k3, k4 = st.columns(4)
k1.metric("분석 구간", f"{len(cf):,}")
k2.metric("이슈유형 항목", f"{len(sf_view):,}")
k3.metric("평균 집중도(HHI)", f"{cf['hhi'].mean():.3f}" if not cf.empty and "hhi" in cf else "-")
k4.metric("평균 실질 이슈 수", f"{cf['effective_issue_count'].mean():.1f}" if not cf.empty and "effective_issue_count" in cf else "-")

tab1, tab2, tab3, tab4 = st.tabs(["이슈 집중도·다양성", "후보별 이슈유형 점유율", "후보 간 차이", "방법 및 한계"])

with tab1:
    st.subheader("이슈 집중도·다양성")
    st.info("HHI가 높을수록 특정 이슈에 집중된 구조이고, 실질 이슈 수가 높을수록 여러 이슈로 분산된 구조입니다.")
    if cf.empty:
        st.warning("조건에 맞는 집중도 지표가 없습니다.")
    else:
        metric_options = {
            "집중도(HHI)": "hhi",
            "실질 이슈 수": "effective_issue_count",
            "다양성 지수": "entropy_norm",
            "최대 이슈 점유율": "top1_share",
            "상위 3개 이슈 점유율": "top3_share",
        }
        metric_label = st.radio("차트 지표", list(metric_options.keys()), horizontal=True)
        metric = metric_options[metric_label]
        fig = concentration_line_chart(cf, metric=metric)
        if fig:
            st.plotly_chart(fig, width="stretch")
        st.dataframe(concentration_table(cf), width="stretch", height=420, hide_index=True)

with tab2:
    st.subheader("후보별 이슈유형 점유율과 95% 신뢰구간")
    st.caption("이 신뢰구간은 자동 추출된 이슈 단위를 재표본추출한 구간입니다. 여론조사 표본오차가 아닙니다.")
    if sf_view.empty:
        st.warning("조건에 맞는 이슈유형 점유율 결과가 없습니다.")
    else:
        fig = issue_type_share_bar(sf_view)
        if fig:
            st.plotly_chart(fig, width="stretch")
        table = share_table(sf_view)
        st.dataframe(table, width="stretch", height=420, hide_index=True)

with tab3:
    st.subheader("정원오-오세훈 이슈유형 점유율 차이")
    st.caption("양 후보가 모두 있는 이슈유형에 대해 점유율 차이를 계산합니다. 구간은 보수적 endpoint 차이입니다.")
    if df_diff_view.empty:
        st.info("후보 간 차이 테이블이 없습니다.")
    else:
        st.dataframe(diff_table(df_diff_view), width="stretch", height=480, hide_index=True)

with tab4:
    st.subheader("분석 방법 및 해석 한계")
    st.markdown(
        """
### 분석 모드
- **보수적 분석**: 자동 분류 결과 중 확인 필요 항목을 더 엄격히 제외합니다.
- **기본 분석**: 명시적 제외 항목과 기타·잡음 유형을 제외한 기본 결과입니다.

### 지표
- **HHI** = 이슈별 점유율 제곱합. 높을수록 특정 이슈 집중.
- **Entropy** = 이슈 분포 다양성.
- **Effective issue count** = 1 / HHI. 실질적으로 몇 개 이슈가 주도했는지 보여줌.
- **부트스트랩 95% 신뢰구간** = 자동 추출 이슈 단위의 재표본추출로 계산한 점유율 불확실성.
        """
    )
    summary = pd.DataFrame(
        [
            {"구분": "이슈 집중도·다양성", "산출 결과": f"{len(conc):,}개 분석 구간", "상태": "완료"},
            {"구분": "후보별 이슈유형 점유율", "산출 결과": f"{len(sf_view):,}개 이슈유형 항목", "상태": "완료"},
            {"구분": "후보 간 점유율 차이", "산출 결과": f"{len(df_diff_view):,}개 비교 항목", "상태": "완료" if not df_diff_view.empty else "해당 없음"},
        ]
    )
    st.dataframe(summary, width="stretch", hide_index=True)

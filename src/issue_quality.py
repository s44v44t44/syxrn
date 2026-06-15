from __future__ import annotations

import re
from typing import Iterable

import pandas as pd


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
    "내용", "방송", "모음", "제목", "신문기사", "정원", "감사", "하는", "누가",
}

STRONG_NAME_TERMS = {
    "여론조사", "본선", "판세", "TV토론", "토론", "공방", "GTX", "삼성역", "철근", "누락",
    "서소문", "고가", "사고", "안전", "칸쿤", "출장", "의혹", "선거법", "고발", "장특공",
    "세금", "경선", "확정", "사전투표", "투표", "박원순", "노무현", "부동산", "후보 선출",
}

CATEGORY_LIKE_NAMES = {
    "후보 출마·정치적 입지",
    "사진·현장 스케치 노출",
    "사진·현장 노출 이벤트",
    "행정 공지·지역 서비스 안내",
    "행정 성과·지역 행정",
    "행정·지역 정책 메시지",
    "지역 방문·현장 행보",
    "언론 칼럼·뉴스요약 콘텐츠",
    "관련 여러 보도·게시글",
}

BAD_ISSUE_TYPES = {"noise_or_nonissue", "other", "media_meta"}
EXCLUDE_STATUSES = {"exclude_candidate", "excluded", "exclude"}
REVIEW_NEEDED_STATUSES = {"review_needed", "확인 필요"}


def safe_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return "" if value is None else str(value)


def normalize_spaces(value: object) -> str:
    return re.sub(r"\s+", " ", safe_text(value)).strip()


def shorten_text(value: object, max_len: int = 76) -> str:
    text = normalize_spaces(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def split_terms(value: object) -> list[str]:
    text = safe_text(value)
    if not text:
        return []
    parts = re.split(r"\s*(?:\|\||\||·|,|/|;|\n)\s*", text)
    return [p.strip() for p in parts if p and p.strip()]


def first_present(row: pd.Series, cols: Iterable[str]) -> str:
    for col in cols:
        if col not in row.index:
            continue
        text = normalize_spaces(row.get(col, ""))
        if text:
            return text
    return ""


def first_evidence_title(row: pd.Series) -> str:
    for col in ["representative_title", "evidence_titles", "evidence_titles_topN"]:
        value = safe_text(row.get(col, ""))
        if not value:
            continue
        for part in split_terms(value):
            if part:
                return part
    return ""


def boolish(value: object) -> bool:
    return safe_text(value).strip().lower() in {"true", "1", "1.0", "yes", "y"}


def weak_summary(value: object) -> bool:
    text = normalize_spaces(value)
    return not text or any(marker in text for marker in WEAK_SUMMARY_MARKERS)


def category_like_name(value: object) -> bool:
    name = normalize_spaces(value)
    if not name:
        return True
    return name in CATEGORY_LIKE_NAMES or any(name.startswith(prefix) for prefix in CATEGORY_LIKE_NAMES)


def weak_issue_name(value: object) -> bool:
    name = normalize_spaces(value)
    if not name:
        return True
    if name.startswith(("SI3_", "PI_", "row_")):
        return True
    if category_like_name(name):
        return True
    tokens = split_terms(name)
    if not tokens:
        return True
    if "_" in name:
        return True
    if any(term in name for term in STRONG_NAME_TERMS):
        generic_hits = sum(1 for token in tokens if token.lower() in WEAK_NAME_TERMS)
        return len(tokens) >= 3 and generic_hits >= len(tokens) - 1 and "공방" not in name
    generic_hits = sum(1 for token in tokens if token.lower() in WEAK_NAME_TERMS)
    return generic_hits >= max(1, len(tokens) - 1)


def infer_display_name(row: pd.Series) -> str:
    existing = normalize_spaces(row.get("issue_display_name", ""))
    if existing and not weak_issue_name(existing):
        return existing
    name = first_present(row, ["issue_name", "issue_name_clean_rule", "issue_label_auto"])
    if name and not weak_issue_name(name):
        return name
    title = first_evidence_title(row)
    if title:
        return shorten_text(title, 58)
    terms = [t for t in split_terms(row.get("top_terms", "")) if t.lower() not in WEAK_NAME_TERMS][:4]
    if terms:
        return "·".join(terms)
    return first_present(row, ["stable_issue_id", "period_issue_id"]) or "이름 없는 이슈"


def infer_summary_display(row: pd.Series) -> str:
    existing = normalize_spaces(row.get("summary_display", ""))
    if existing and not weak_summary(existing):
        return existing
    summary = first_present(row, ["summary_1line", "issue_summary_1line_rule"])
    title = first_evidence_title(row)
    issue_name = first_present(row, ["issue_name", "issue_name_clean_rule"])
    if title and (weak_summary(summary) or weak_issue_name(issue_name)):
        return f"대표 근거: {shorten_text(title, 96)}"
    if summary and not weak_summary(summary):
        return summary
    terms = [t for t in split_terms(row.get("top_terms", "")) if t.lower() not in WEAK_NAME_TERMS][:5]
    if terms:
        return f"핵심어: {' · '.join(terms)}"
    if title:
        return f"대표 근거: {shorten_text(title, 96)}"
    return "대표 근거 확인이 필요한 자동 묶음입니다."


def quality_flags(row: pd.Series) -> list[str]:
    flags: list[str] = []
    issue_name = first_present(row, ["issue_name", "issue_name_clean_rule"])
    summary = first_present(row, ["summary_1line", "issue_summary_1line_rule"])
    issue_type = safe_text(row.get("issue_type", row.get("issue_type_rule", ""))).strip()
    review_status = safe_text(row.get("review_status", row.get("review_status_rule", ""))).strip()
    report_use = safe_text(row.get("report_use_level", row.get("report_use_level_rule", ""))).strip()
    evidence_method = safe_text(row.get("evidence_source_method", "")).strip()
    real_evidence = pd.to_numeric(pd.Series([row.get("real_evidence_title_count", 0)]), errors="coerce").fillna(0).iloc[0]

    if weak_issue_name(issue_name):
        flags.append("weak_issue_name")
    display_name = safe_text(row.get("issue_display_name", "")).strip()
    if display_name.startswith(("SI3_", "PI_", "row_")) or issue_name.startswith(("SI3_", "PI_", "row_")):
        flags.append("internal_id_display")
    if weak_summary(summary):
        flags.append("weak_summary")
    if issue_type in BAD_ISSUE_TYPES:
        flags.append(f"issue_type_{issue_type}")
    if review_status in EXCLUDE_STATUSES:
        flags.append("exclude_status")
    elif review_status in REVIEW_NEEDED_STATUSES:
        flags.append("review_needed_status")
    if report_use in {"triage_only", "review_needed"}:
        flags.append(f"report_use_{report_use}")
    if evidence_method and evidence_method != "docmap_topN":
        flags.append(f"evidence_{evidence_method}")
    if real_evidence < 1:
        flags.append("no_real_evidence_title")
    if boolish(row.get("low_coherence_flag", row.get("qc_low_coherence_flag", False))):
        flags.append("low_coherence")
    quality_include = safe_text(row.get("quality_include_for_final_l4_v31", "")).strip()
    if quality_include and not boolish(quality_include):
        flags.append("quality_excluded_v31")
    text = " ".join([
        safe_text(row.get("issue_name", "")),
        safe_text(row.get("top_terms", "")),
        safe_text(row.get("representative_title", "")),
    ]).lower()
    if any(term in text for term in ["가입하기", "카카오 계정", "스크랩", "댓글내용", "480p", "관련주", "특징주"]):
        flags.append("ui_or_market_noise_terms")
    return sorted(set(flags))


def quality_tier(flags: list[str]) -> str:
    if "exclude_status" in flags or "quality_excluded_v31" in flags or "issue_type_noise_or_nonissue" in flags:
        return "exclude"
    if "issue_type_other" in flags or "review_needed_status" in flags:
        return "review_needed"
    caution = {"weak_issue_name", "weak_summary", "low_coherence", "report_use_triage_only", "no_real_evidence_title"}
    if any(flag in caution for flag in flags):
        return "use_with_caution"
    return "ready"


def apply_issue_quality_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    out = df.copy()
    displays = []
    summaries = []
    flag_values = []
    tiers = []
    for _, row in out.iterrows():
        flags = quality_flags(row)
        displays.append(infer_display_name(row))
        summaries.append(infer_summary_display(row))
        flag_values.append("|".join(flags))
        tiers.append(quality_tier(flags))
    out["issue_display_name"] = displays
    out["summary_display"] = summaries
    out["dashboard_quality_flags"] = flag_values
    out["dashboard_quality_tier"] = tiers
    out["dashboard_include_default"] = ~out["dashboard_quality_tier"].isin(["exclude"])
    return out


def make_quality_search_text(df: pd.DataFrame) -> pd.Series:
    if df is None or df.empty:
        return pd.Series([], index=df.index, dtype="object")
    cols = [
        "issue_display_name", "summary_display", "issue_name", "summary_1line", "summary_detail",
        "top_terms", "representative_title", "evidence_titles",
    ]
    existing = [c for c in cols if c in df.columns]
    if not existing:
        return pd.Series([""] * len(df), index=df.index)
    joined = df[existing].fillna("").astype(str).apply(lambda row: " ".join(row.values.tolist()), axis=1)
    return joined.str.replace(r"\s+", " ", regex=True).str.lower().str.strip()

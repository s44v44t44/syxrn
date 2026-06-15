from __future__ import annotations

import pandas as pd

CANDIDATE_LABELS = {"JWO": "정원오", "OSH": "오세훈"}
SOURCE_LABELS = {
    "bigkinds_article": "빅카인즈 기사",
    "naver_article": "네이버 기사",
    "daum_article": "다음 기사",
    "youtube": "유튜브",
    "naver_blog_cafe": "네이버 블로그·카페",
    "daum_blog_cafe": "다음 블로그·카페",
    "all_sources_combined": "전체 자료원",
}
WINDOW_LABELS = {
    "daily": "당일",
    "rolling_7d": "최근 7일",
    "rolling_14d": "최근 14일",
    "calendar_week": "주간",
    "calendar_month_exact": "월간",
    "monthly": "월간",
    "custom": "사용자 지정",
}
REVIEW_STATUS_LABELS = {
    "publish_ready": "바로 설명 가능",
    "review_needed": "확인 필요",
    "exclude_candidate": "제외 후보",
    "": "-",
}
REPORT_USE_LABELS = {
    "triage_only": "빠른 탐색용",
    "evidence_ready": "근거 확인됨",
    "review_needed": "확인 필요",
    "exclude_candidate": "제외 후보",
    "reviewed_final": "검토 완료",
    "": "-",
}
ISSUE_TYPE_LABELS = {
    "candidate_selection": "후보 선출·경선",
    "race_polling": "여론조사·판세",
    "campaign_strategy": "캠페인·선거전략",
    "controversy_attack_defense": "검증·공세·반박",
    "legal_election_rule": "법적 공방·선거법",
    "policy_real_estate_tax": "부동산·세금 정책",
    "policy_transport_infrastructure": "교통·인프라 정책",
    "policy_welfare_livelihood": "민생·복지 정책",
    "safety_accident": "안전·사고",
    "administrative_performance": "행정 성과·실적",
    "administrative_notice": "행정 공지·안내",
    "candidate_positioning": "후보 포지셔닝",
    "regional_visit": "지역 방문·현장 행보",
    "media_photo_event": "포토·현장 노출",
    "media_column_summary": "칼럼·뉴스요약",
    "debate_broadcast": "토론·방송",
    "community_reaction": "온라인 반응",
    "media_meta": "미디어·메타 이슈",
    "symbolic_event": "상징 행보",
    "noise_or_nonissue": "잡음·비이슈",
    "other": "기타/분류 필요",
    "": "-",
}
METRIC_LABELS = {
    "doc_count": "문서 수",
    "doc_count_raw": "문서 수",
    "doc_count_weighted": "보정 문서 수",
    "comment_count": "댓글 수",
    "comment_count_weighted": "보정 댓글 수",
    "view_count": "조회수",
    "view_count_weighted": "보정 조회수",
    "like_count": "좋아요 수",
    "like_count_weighted": "보정 좋아요 수",
    "rank_score": "순위 점수",
    "dashboard_rank_score": "탐색 순위 점수",
    "cluster_coherence": "묶음 일관성",
    "cluster_coherence_mean": "묶음 일관성",
    "active_days": "관측일 수",
    "reaction_total": "반응 합계",
    "parent_doc_count": "원문 수",
    "collected_comment_count": "수집 댓글 수",
    "platform_comment_count": "플랫폼 댓글 수",
    "top_level_comment_count": "원댓글 수",
    "reply_comment_count": "답글 댓글 수",
    "comment_like_sum": "댓글 좋아요 합계",
    "comment_like_count": "댓글 좋아요",
    "sympathy_sum": "공감 합계",
    "sympathy_count": "공감",
    "antipathy_sum": "비공감 합계",
    "antipathy_count": "비공감",
    "reply_count_sum": "답글 수",
    "reply_count": "답글 수",
    "unique_author_hash_count": "비식별 작성자 수",
}
COLUMN_LABELS = {
    "display_rank": "순위",
    "candidate": "후보",
    "candidate_label": "후보",
    "source": "자료원",
    "source_label": "자료원",
    "date": "날짜",
    "window_type": "기간 유형",
    "window_start": "시작일",
    "window_end": "종료일",
    "rank_in_window": "순위",
    "issue_name": "이슈명",
    "issue_display_name": "이슈명",
    "issue_name_clean_rule": "이슈명",
    "issue_label_auto": "자동 이슈명",
    "issue_type": "이슈 유형",
    "issue_type_rule": "이슈 유형",
    "summary_1line": "한 줄 요약",
    "summary_display": "한 줄 요약",
    "issue_summary_1line_rule": "한 줄 요약",
    "review_status": "검토 상태",
    "review_status_rule": "검토 상태",
    "report_use_level": "사용 수준",
    "report_use_level_rule": "사용 수준",
    "representative_title": "대표 제목",
    "representative_title_short": "대표 제목",
    "title": "제목",
    "top_terms": "핵심어",
    "evidence_titles": "근거 제목",
    "evidence_titles_topN": "근거 제목",
    "publisher_or_channel": "언론사·채널",
    "url": "URL",
    "parent_doc_key": "원문 ID",
    "parent_title": "원문 제목",
    "parent_url": "원문 링크",
    "first_comment_date": "첫 댓글일",
    "latest_comment_date": "마지막 댓글일",
    "comment_date": "댓글 날짜",
    "comment_datetime": "댓글 작성시각",
    "comment_text_masked": "댓글 내용",
    "comment_text_clean": "댓글 내용",
    "author_hash_public": "작성자(비식별)",
    "is_reply": "답글 여부",
    "mapping_method": "댓글 연결 방식",
    "mapping_confidence": "연결 신뢰도",
    "period_issue_id": "기간 이슈 ID",
    "stable_issue_id": "연결 이슈 ID",
    **METRIC_LABELS,
}
HELP_TEXT = {
    "보정 문서 수": "같은 기사·같은 제목 반복이 과도하게 세어지지 않도록 보정한 문서 수입니다.",
    "문서 수": "조건에 맞는 고유 문서 수입니다.",
    "댓글 수": "수집된 댓글 수의 합입니다. 자료원별 수집 가능 여부에 따라 0일 수 있습니다.",
    "조회수": "유튜브·블로그/카페 등 조회수가 있는 자료원의 조회수 합입니다.",
    "기간 유형": "당일, 최근 7일, 최근 14일, 주간, 월간 등 이슈를 묶은 기간 기준입니다.",
    "사용 수준": "빠른 탐색용인지, 근거 문서가 확인된 결과인지, 사람이 확인해야 하는지 표시합니다.",
    "검토 상태": "자동 결과를 바로 설명해도 되는지, 사람이 확인해야 하는지, 제외 후보인지 표시합니다.",
    "묶음 일관성": "같은 이슈로 묶인 문서들이 얼마나 비슷한지 보는 보조 지표입니다. 낮을수록 검토가 필요합니다.",
    "수집 댓글 수": "현재 데이터에 원문 단위로 수집된 댓글 수입니다. 플랫폼 전체 댓글 수와 다를 수 있습니다.",
    "원문 링크": "댓글이 달린 기사·영상·게시글의 원문 링크입니다.",
    "댓글 연결 방식": "댓글이 이슈와 연결된 방식입니다. parent_doc_issue_map은 부모 원문이 해당 이슈에 속해 연결된 경우입니다.",
}


def safe_text(value: object) -> str:
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return "" if value is None else str(value)


def reverse_lookup(mapping: dict[str, str], label: str) -> str:
    return {v: k for k, v in mapping.items()}.get(label, label)


def label_from(mapping: dict[str, str], value: object) -> str:
    text = safe_text(value)
    return mapping.get(text, text)


def label_candidate(x: object) -> str:
    return label_from(CANDIDATE_LABELS, x)


def label_source(x: object) -> str:
    return label_from(SOURCE_LABELS, x)


def label_window(x: object) -> str:
    return label_from(WINDOW_LABELS, x)


def label_review_status(x: object) -> str:
    return label_from(REVIEW_STATUS_LABELS, x)


def label_report_use(x: object) -> str:
    return label_from(REPORT_USE_LABELS, x)


def label_issue_type(x: object) -> str:
    return label_from(ISSUE_TYPE_LABELS, x)


def metric_label(metric: str) -> str:
    return METRIC_LABELS.get(metric, metric)


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "candidate" in out.columns:
        out["후보"] = out["candidate"].map(label_candidate)
    if "source" in out.columns:
        out["자료원"] = out["source"].map(label_source)
    if "window_type" in out.columns:
        out["기간 유형"] = out["window_type"].map(label_window)
    if "review_status" in out.columns:
        out["검토 상태"] = out["review_status"].map(label_review_status)
    elif "review_status_rule" in out.columns:
        out["검토 상태"] = out["review_status_rule"].map(label_review_status)
    if "report_use_level" in out.columns:
        out["사용 수준"] = out["report_use_level"].map(label_report_use)
    elif "report_use_level_rule" in out.columns:
        out["사용 수준"] = out["report_use_level_rule"].map(label_report_use)
    if "issue_type" in out.columns:
        out["이슈 유형"] = out["issue_type"].map(label_issue_type)
    elif "issue_type_rule" in out.columns:
        out["이슈 유형"] = out["issue_type_rule"].map(label_issue_type)
    return out


def display_table(df: pd.DataFrame, kind: str = "generic", max_rows: int | None = None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = add_display_columns(df)
    if kind == "top10":
        order = [
            "display_rank", "issue_display_name", "summary_display", "자료원", "기간 유형",
            "window_start", "window_end", "active_days", "doc_count_weighted", "comment_count",
            "comment_count_weighted", "view_count", "view_count_weighted", "like_count", "like_count_weighted",
            "top_terms",
        ]
    elif kind == "timeline":
        order = ["date", "자료원", "doc_count", "doc_count_weighted", "comment_count", "view_count", "like_count"]
    elif kind == "platform":
        order = ["자료원", "doc_count", "doc_count_weighted", "comment_count", "view_count", "like_count"]
    elif kind == "evidence":
        order = ["date", "자료원", "publisher_or_channel", "title", "url", "comment_count", "view_count", "like_count"]
    elif kind == "evidence_debug":
        order = ["date", "자료원", "publisher_or_channel", "title", "url", "comment_count", "view_count", "like_count", "period_issue_id", "stable_issue_id", "doc_id"]
    elif kind == "reaction_docs":
        order = ["date", "자료원", "publisher_or_channel", "title", "url", "comment_count", "view_count", "like_count", "reaction_total"]
    elif kind in {"parent_docs", "comment_docs"}:
        order = [
            "date", "자료원", "publisher_or_channel", "parent_title", "parent_url",
            "collected_comment_count", "top_level_comment_count", "reply_comment_count",
            "comment_like_sum", "sympathy_sum", "antipathy_sum", "reply_count_sum",
            "unique_author_hash_count", "first_comment_date", "latest_comment_date",
        ]
    elif kind == "parent_docs_debug":
        order = [
            "date", "자료원", "publisher_or_channel", "parent_title", "parent_url",
            "collected_comment_count", "parent_doc_key", "period_issue_id", "stable_issue_id",
        ]
    elif kind == "comments":
        order = [
            "comment_datetime", "자료원", "parent_title", "parent_url", "comment_text_masked",
            "comment_like_count", "sympathy_count", "antipathy_count", "reply_count",
            "is_reply", "author_hash_public",
        ]
    elif kind == "comments_internal":
        order = [
            "comment_datetime", "자료원", "parent_title", "parent_url", "comment_text_clean",
            "comment_like_count", "sympathy_count", "antipathy_count", "reply_count",
            "is_reply", "author_hash_public", "parent_doc_key",
        ]
    elif kind == "comment_timeline":
        order = ["date", "자료원", "collected_comment_count", "comment_like_sum", "sympathy_sum", "antipathy_sum", "reply_count_sum"]
    elif kind == "comment_platform":
        order = ["자료원", "parent_doc_count", "collected_comment_count", "comment_like_sum", "sympathy_sum", "antipathy_sum", "reply_count_sum", "unique_author_hash_count"]
    else:
        order = list(out.columns)
    keep = [c for c in order if c in out.columns]
    if not keep:
        keep = list(out.columns)
    out = out[keep].copy()
    out = out.rename(columns={c: COLUMN_LABELS.get(c, c) for c in out.columns})
    if max_rows is not None:
        out = out.head(max_rows)
    return out


def chart_df_with_source_label(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "source" in out.columns and "자료원" not in out.columns:
        out["자료원"] = out["source"].map(label_source)
    return out


def glossary_markdown() -> str:
    return """
### 화면 용어 설명

| 화면 표시 | 뜻 |
|---|---|
| 당일 | 하루 단위로 잡은 이슈입니다. |
| 최근 7일 | 선택한 종료일을 포함한 최근 7일 이슈입니다. |
| 최근 14일 | 선택한 종료일을 포함한 최근 14일 이슈입니다. |
| 주간 | 월~일 단위 주간 이슈입니다. |
| 보정 문서 수 | 같은 기사·같은 제목 반복이 과도하게 세어지지 않도록 보정한 문서 수입니다. |
| 수집 댓글 수 | 대시보드 데이터에 실제로 수집된 댓글 원문 수입니다. 플랫폼 전체 댓글 수와 다를 수 있습니다. |
| 원문·댓글 반응 | 이슈/키워드와 관련된 원문 기사·영상·게시글과 그 댓글을 함께 보는 화면입니다. |
"""

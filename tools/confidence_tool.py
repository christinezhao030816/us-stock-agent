from datetime import date
from typing import Any


def _parse_date(value: str) -> date:
    cleaned = value.strip()[:10]

    try:
        return date.fromisoformat(cleaned)
    except ValueError as error:
        raise ValueError(
            f"日期格式不正确：{value}，应使用YYYY-MM-DD"
        ) from error


def _age_days(
    analysis_date: date,
    data_date: str,
) -> int:
    parsed_data_date = _parse_date(data_date)
    difference = (analysis_date - parsed_data_date).days

    return max(difference, 0)


def _freshness_score(
    age_days: int,
    thresholds: list[tuple[int, int]],
) -> int:
    for maximum_age, score in thresholds:
        if age_days <= maximum_age:
            return score

    return 0


def calculate_confidence_score(
    analysis_date: str,
    technical_data_date: str,
    news_data_date: str,
    fundamental_filing_date: str,
    technical_metrics_present: int,
    fundamental_metrics_present: int,
    relevant_news_count: int,
    has_sec_primary_text: bool,
    has_independent_high_quality_news: bool,
    has_data_read_errors: bool,
    sec_text_truncated: bool,
    signal_alignment: str,
) -> dict[str, Any]:
    """
    根据固定规则计算综合分析置信度。

    technical_metrics_present范围为0至4：
    1. 价格和日期
    2. 均线与RSI
    3. 成交量
    4. 支撑位与阻力位

    fundamental_metrics_present范围为0至5：
    1. 财报期间与来源
    2. 收入、利润和同比增长
    3. 经营现金流与自由现金流
    4. 现金、债务、资产和权益
    5. 数据期间区别与限制

    signal_alignment只能是：
    aligned、mixed或conflicting。
    """

    technical_metrics_present = max(
        0,
        min(int(technical_metrics_present), 4),
    )

    fundamental_metrics_present = max(
        0,
        min(int(fundamental_metrics_present), 5),
    )

    relevant_news_count = max(
        0,
        min(int(relevant_news_count), 5),
    )

    normalized_alignment = (
        signal_alignment.strip().lower()
    )

    allowed_alignments = {
        "aligned",
        "mixed",
        "conflicting",
    }

    if normalized_alignment not in allowed_alignments:
        raise ValueError(
            "signal_alignment必须是aligned、"
            "mixed或conflicting"
        )

    parsed_analysis_date = _parse_date(
        analysis_date
    )

    technical_age_days = _age_days(
        parsed_analysis_date,
        technical_data_date,
    )

    news_age_days = _age_days(
        parsed_analysis_date,
        news_data_date,
    )

    fundamental_age_days = _age_days(
        parsed_analysis_date,
        fundamental_filing_date,
    )

    technical_score = (
        technical_metrics_present * 5
    )

    fundamental_score = (
        fundamental_metrics_present * 5
    )

    news_count_score = relevant_news_count

    sec_primary_score = (
        10 if has_sec_primary_text else 0
    )

    independent_news_score = (
        7
        if has_independent_high_quality_news
        else 0
    )

    read_quality_score = (
        3 if not has_data_read_errors else 0
    )

    news_score = (
        news_count_score
        + sec_primary_score
        + independent_news_score
        + read_quality_score
    )

    alignment_scores = {
        "aligned": 15,
        "mixed": 8,
        "conflicting": 0,
    }

    alignment_score = alignment_scores[
        normalized_alignment
    ]

    technical_freshness_score = _freshness_score(
        technical_age_days,
        [
            (1, 6),
            (3, 4),
            (7, 2),
        ],
    )

    news_freshness_score = _freshness_score(
        news_age_days,
        [
            (1, 5),
            (3, 3),
            (7, 1),
        ],
    )

    fundamental_freshness_score = _freshness_score(
        fundamental_age_days,
        [
            (45, 4),
            (120, 3),
            (240, 1),
        ],
    )

    freshness_score = (
        technical_freshness_score
        + news_freshness_score
        + fundamental_freshness_score
    )

    raw_score = (
        technical_score
        + fundamental_score
        + news_score
        + alignment_score
        + freshness_score
    )

    deductions = []
    deduction_total = 0

    if has_data_read_errors:
        deduction_total += 10
        deductions.append(
            {
                "reason": "存在数据读取错误",
                "points": -10,
            }
        )

    if sec_text_truncated:
        deduction_total += 3
        deductions.append(
            {
                "reason": "SEC正文被截断",
                "points": -3,
            }
        )

    adjusted_score = max(
        0,
        min(100, raw_score - deduction_total),
    )

    score_cap = 100
    cap_reasons = []

    if technical_metrics_present < 3:
        score_cap = min(score_cap, 79)
        cap_reasons.append(
            "技术面关键指标不足"
        )

    if fundamental_metrics_present < 4:
        score_cap = min(score_cap, 79)
        cap_reasons.append(
            "基本面关键指标不足"
        )

    if not has_independent_high_quality_news:
        score_cap = min(score_cap, 79)
        cap_reasons.append(
            "缺少独立高质量新闻来源"
        )

    if sec_text_truncated:
        score_cap = min(score_cap, 79)
        cap_reasons.append(
            "SEC正文未完整读取"
        )

    if has_data_read_errors:
        score_cap = min(score_cap, 79)
        cap_reasons.append(
            "存在数据读取错误"
        )

    if (
        relevant_news_count == 0
        and not has_sec_primary_text
    ):
        score_cap = min(score_cap, 59)
        cap_reasons.append(
            "消息面缺少可用证据"
        )

    if technical_age_days > 7:
        score_cap = min(score_cap, 59)
        cap_reasons.append(
            "技术面行情超过7天"
        )

    final_score = min(
        adjusted_score,
        score_cap,
    )

    if final_score >= 80:
        confidence_level = "高"
    elif final_score >= 60:
        confidence_level = "中"
    else:
        confidence_level = "低"

    return {
        "final_score": final_score,
        "raw_score": raw_score,
        "score_cap": score_cap,
        "confidence_level": confidence_level,
        "score_breakdown": {
            "technical_quality": {
                "score": technical_score,
                "maximum": 20,
            },
            "fundamental_quality": {
                "score": fundamental_score,
                "maximum": 25,
            },
            "news_quality": {
                "score": news_score,
                "maximum": 25,
            },
            "signal_alignment": {
                "score": alignment_score,
                "maximum": 15,
                "classification": normalized_alignment,
            },
            "freshness": {
                "score": freshness_score,
                "maximum": 15,
            },
        },
        "data_age_days": {
            "technical": technical_age_days,
            "news": news_age_days,
            "fundamental_filing": (
                fundamental_age_days
            ),
        },
        "deductions": deductions,
        "deduction_total": deduction_total,
        "cap_reasons": cap_reasons,
        "method": (
            "固定100分评分规则，并应用数据质量上限"
        ),
    }

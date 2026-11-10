import re
from datetime import date, datetime, timezone
from typing import Any


FRESHNESS_THRESHOLDS_DAYS = {
    "quarterly": 180,
    "annual": 450,
}
CAUTION_MULTIPLIER = 1.5
CORE_FUNDAMENTAL_METRICS = {
    "revenue",
    "net_income",
    "operating_cash_flow",
    "total_assets",
}
ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
QUARTERLY_FORMS = {"10-Q", "10-Q/A"}
FOREIGN_INTERIM_FORMS = {"6-K", "6-K/A"}


def analysis_date(value: str | date | None = None) -> date:
    if value is None:
        return datetime.now(timezone.utc).date()
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"analysis_as_of 日期格式不正确：{value}，应为YYYY-MM-DD"
        ) from error


def _optional_date(value: Any) -> date | None:
    if value in {None, ""}:
        return None
    try:
        return date.fromisoformat(str(value).strip()[:10])
    except (TypeError, ValueError):
        return None


def period_basis(
    form: str | None,
    period_start: str | None,
    period_end: str | None,
) -> str | None:
    normalized = str(form or "").upper()
    if normalized in ANNUAL_FORMS:
        return "annual"
    if normalized in QUARTERLY_FORMS:
        return "quarterly"
    if normalized in FOREIGN_INTERIM_FORMS:
        start = _optional_date(period_start)
        end = _optional_date(period_end)
        if start and end and 300 <= (end - start).days <= 390:
            return "annual"
        return "quarterly"
    return None


def evaluate_fact_freshness(
    *,
    metric_name: str,
    value: Any,
    unit: str | None,
    form: str | None,
    fiscal_year: Any,
    fiscal_period: Any,
    period_start: str | None,
    period_end: str | None,
    filed_date: str | None,
    accession: str | None,
    analysis_as_of: str | date | None = None,
) -> dict[str, Any]:
    as_of = analysis_date(analysis_as_of)
    end = _optional_date(period_end)
    filed = _optional_date(filed_date)
    basis = period_basis(form, period_start, period_end)
    threshold = FRESHNESS_THRESHOLDS_DAYS.get(str(basis))
    period_age = (as_of - end).days if end else None
    filing_age = (as_of - filed).days if filed else None

    status = "unknown"
    reason = "缺少或无法解析期间结束日期"
    if value is None:
        reason = "该指标没有可用数值"
    elif end is None or filed is None:
        missing = []
        if end is None:
            missing.append("period_end")
        if filed is None:
            missing.append("filed_date")
        reason = "缺少或无法解析日期：" + ", ".join(missing)
    elif period_age is not None and period_age < 0:
        reason = "期间结束日期晚于分析日，事实被降级为unknown"
    elif filing_age is not None and filing_age < 0:
        reason = "提交日期晚于分析日，事实被降级为unknown"
    elif basis is None or threshold is None:
        reason = f"申报类型{form or '缺失'}无法确定统一新鲜度阈值"
    elif period_age <= threshold:
        status = "fresh"
        reason = (
            f"{basis}事实距分析日{period_age}天，"
            f"不超过{threshold}天阈值"
        )
    elif period_age <= int(threshold * CAUTION_MULTIPLIER):
        status = "caution"
        reason = (
            f"{basis}事实距分析日{period_age}天，超过{threshold}天"
            f"fresh阈值但未超过{int(threshold * CAUTION_MULTIPLIER)}天"
        )
    else:
        status = "stale"
        reason = (
            f"{basis}事实距分析日{period_age}天，超过"
            f"{int(threshold * CAUTION_MULTIPLIER)}天stale界限"
        )

    return {
        "metric_name": metric_name,
        "value": value,
        "unit": unit,
        "form": form,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
        "period_start": period_start,
        "period_end": period_end,
        "filed_date": filed_date,
        "accession": accession,
        "analysis_as_of": as_of.isoformat(),
        "period_age_days": period_age,
        "filing_age_days": filing_age,
        "freshness_threshold_days": threshold,
        "age_freshness_status": status,
        "age_freshness_reason": reason,
        "freshness_status": status,
        "freshness_reason": reason,
        "period_basis": basis,
    }


def _record_metadata(record: Any, fallback: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(record, dict):
        return {
            "form": fallback.get("form"),
            "fiscal_year": fallback.get("fiscal_year"),
            "fiscal_period": fallback.get("fiscal_period"),
            "period_start": None,
            "period_end": fallback.get("income_period_end"),
            "filed_date": fallback.get("filing_date"),
            "accession": None,
        }
    concepts = record.get("concepts")
    primary = concepts[0] if isinstance(concepts, list) and concepts else record
    return {
        "form": primary.get("form") or fallback.get("form"),
        "fiscal_year": primary.get("fiscal_year") or fallback.get("fiscal_year"),
        "fiscal_period": primary.get("fiscal_period") or fallback.get("fiscal_period"),
        "period_start": primary.get("start"),
        "period_end": primary.get("end") or record.get("end") or fallback.get("income_period_end"),
        "filed_date": primary.get("filed") or record.get("filed") or fallback.get("filing_date"),
        "accession": primary.get("accession"),
    }


def build_structured_freshness(
    metrics: dict[str, Any],
    analysis_as_of: str | date | None = None,
    latest_filing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    amounts = metrics.get("financial_amounts")
    amounts = amounts if isinstance(amounts, dict) else {}
    records = metrics.get("source_records")
    records = records if isinstance(records, dict) else {}

    def amount_unit(amount: Any, fallback: str) -> str:
        if not isinstance(amount, dict):
            return fallback
        currency = str(amount.get("currency") or "").strip()
        scale = str(amount.get("scale") or "").strip()
        return " ".join(part for part in (currency, scale) if part) or fallback

    eps_value = metrics.get("eps_diluted_per_share")
    if eps_value is None:
        eps_value = metrics.get("eps_diluted_usd_per_share")
    default_currency = str(metrics.get("reporting_currency") or "USD")
    default_amount_unit = f"{default_currency} billion"
    definitions = [
        ("revenue", amounts.get("revenue"), amount_unit(amounts.get("revenue"), default_amount_unit), "revenue"),
        ("net_income", amounts.get("net_income"), amount_unit(amounts.get("net_income"), default_amount_unit), "net_income"),
        ("EPS", eps_value, str(metrics.get("eps_unit") or f"{default_currency}/share"), "eps"),
        ("operating_cash_flow", amounts.get("operating_cash_flow"), amount_unit(amounts.get("operating_cash_flow"), default_amount_unit), "operating_cash_flow"),
        ("capital_expenditure", amounts.get("capital_expenditure"), amount_unit(amounts.get("capital_expenditure"), default_amount_unit), "capital_expenditure"),
        ("free_cash_flow", amounts.get("free_cash_flow"), amount_unit(amounts.get("free_cash_flow"), default_amount_unit), "free_cash_flow"),
        ("cash", amounts.get("cash"), amount_unit(amounts.get("cash"), default_amount_unit), "cash"),
        ("total_assets", amounts.get("assets"), amount_unit(amounts.get("assets"), default_amount_unit), "assets"),
        ("total_liabilities", amounts.get("liabilities"), amount_unit(amounts.get("liabilities"), default_amount_unit), "liabilities"),
        ("total_debt", amounts.get("total_debt"), amount_unit(amounts.get("total_debt"), default_amount_unit), "total_debt"),
        ("equity", amounts.get("equity"), amount_unit(amounts.get("equity"), default_amount_unit), "equity"),
        ("revenue_yoy", metrics.get("revenue_yoy_percent"), "percent", "revenue_yoy"),
        ("net_income_yoy", metrics.get("net_income_yoy_percent"), "percent", "net_income_yoy"),
        ("net_margin", metrics.get("net_margin"), "ratio", "net_margin"),
        ("equity_ratio", metrics.get("equity_ratio"), "ratio", "equity_ratio"),
        (
            "operating_cash_flow_to_net_income",
            metrics.get("operating_cash_flow_to_net_income"),
            "ratio",
            "operating_cash_flow_to_net_income",
        ),
    ]
    facts = []
    for metric_name, raw_value, unit, record_key in definitions:
        value = raw_value.get("value") if isinstance(raw_value, dict) else raw_value
        metadata = _record_metadata(records.get(record_key), metrics)
        facts.append(
            evaluate_fact_freshness(
                metric_name=metric_name,
                value=value,
                unit=unit,
                analysis_as_of=analysis_as_of,
                **metadata,
            )
        )
    return apply_latest_period_alignment(
        facts,
        analysis_as_of=analysis_as_of,
        selected_period_end=metrics.get("income_period_end"),
        latest_filing=latest_filing,
    )


def apply_latest_period_alignment(
    facts: list[dict[str, Any]],
    *,
    analysis_as_of: str | date | None,
    selected_period_end: str | None,
    latest_filing: dict[str, Any] | None,
) -> dict[str, Any]:
    selected = _optional_date(selected_period_end)
    latest_end_value = (
        latest_filing.get("period_end")
        if isinstance(latest_filing, dict)
        else None
    )
    latest = _optional_date(latest_end_value)
    lag_days = (latest - selected).days if selected and latest else None

    if selected is None or latest is None:
        alignment = "unknown"
        alignment_reason = (
            "缺少系统选中期间或SEC最新适用申报期间，无法验证相对期间一致性"
        )
    elif lag_days == 0:
        alignment = "aligned"
        alignment_reason = "系统选中期间与SEC最新适用申报期间一致"
    elif lag_days > 0:
        alignment = "historical_fallback"
        alignment_reason = (
            f"SEC已有更新期间{latest_end_value}，系统仅取得"
            f"{selected_period_end}，落后{lag_days}天"
        )
    else:
        alignment = "unknown"
        alignment_reason = (
            "系统选中期间晚于SEC submissions中的最新适用期间，元数据不一致"
        )

    for fact in facts:
        fact["latest_filing_period_end"] = latest_end_value
        fact["selected_period_end"] = selected_period_end
        fact["period_lag_days"] = lag_days
        fact["latest_period_alignment_status"] = alignment
        if fact.get("value") is None:
            continue
        if alignment == "historical_fallback":
            fact["freshness_status"] = "stale"
            fact["freshness_reason"] = (
                f"{fact.get('age_freshness_reason')}; {alignment_reason}; "
                "该值仅可作为historical_fallback"
            )
        elif alignment == "unknown":
            fact["freshness_status"] = "unknown"
            fact["freshness_reason"] = (
                f"{fact.get('age_freshness_reason')}; {alignment_reason}"
            )

    result = summarize_freshness(facts, analysis_as_of)
    summary = result["summary"]
    age_statuses = [
        fact.get("age_freshness_status")
        for fact in facts
        if fact.get("value") is not None
    ]
    if not age_statuses:
        age_overall = "unknown"
    elif "stale" in age_statuses:
        age_overall = "stale"
    elif "unknown" in age_statuses or "caution" in age_statuses:
        age_overall = "caution"
    else:
        age_overall = "fresh"
    summary.update(
        {
            "age_freshness_status": age_overall,
            "latest_filing_period_end": latest_end_value,
            "latest_filing_date": (
                latest_filing.get("filing_date")
                if isinstance(latest_filing, dict)
                else None
            ),
            "latest_filing_form": (
                latest_filing.get("form")
                if isinstance(latest_filing, dict)
                else None
            ),
            "latest_filing_accession": (
                latest_filing.get("accession_number")
                if isinstance(latest_filing, dict)
                else None
            ),
            "selected_period_end": selected_period_end,
            "period_lag_days": lag_days,
            "latest_period_alignment_status": alignment,
            "latest_period_alignment_reason": alignment_reason,
            "historical_fallback": alignment == "historical_fallback",
        }
    )
    if alignment != "aligned":
        summary["current_analysis_usable"] = False
        summary["all_available_core_facts_fresh"] = False
    return result


def build_filing_freshness(
    filing_data: dict[str, Any],
    analysis_as_of: str | date | None = None,
) -> dict[str, Any]:
    facts = []
    for index, filing in enumerate(filing_data.get("filings") or [], start=1):
        primary = filing.get("primary_document")
        primary = primary if isinstance(primary, dict) else {}
        value = str(primary.get("text") or "").strip() or None
        facts.append(
            evaluate_fact_freshness(
                metric_name=f"foreign_filing_{index}",
                value=value,
                unit="filing_text",
                form=filing.get("form"),
                fiscal_year=None,
                fiscal_period=None,
                period_start=None,
                period_end=filing.get("report_date") or filing.get("filing_date"),
                filed_date=filing.get("filing_date"),
                accession=filing.get("accession_number"),
                analysis_as_of=analysis_as_of,
            )
        )
    result = summarize_freshness(
        facts,
        analysis_as_of,
        structured_core=False,
    )
    filing_status = result["summary"]["fundamental_freshness_status"]
    result["summary"]["filing_evidence_freshness_status"] = filing_status
    # A recent filing proves that the document is timely, not that the
    # structured core metrics inside it have been obtained and period-matched.
    result["summary"]["fundamental_freshness_status"] = "unknown"
    result["summary"]["filing_evidence_usable"] = bool(
        any(fact["freshness_status"] == "fresh" for fact in facts)
    )
    result["summary"]["current_analysis_usable"] = False
    result["summary"]["all_available_core_facts_fresh"] = False
    result["summary"]["fresh_core_fact_count"] = 0
    result["summary"]["available_core_fact_count"] = 0
    result["summary"].update(
        {
            "age_freshness_status": "unknown",
            "latest_filing_period_end": None,
            "selected_period_end": None,
            "period_lag_days": None,
            "latest_period_alignment_status": "unknown",
            "latest_period_alignment_reason": (
                "外国发行人备用申报文本未形成结构化核心财务期间映射"
            ),
            "historical_fallback": False,
        }
    )
    return result


def summarize_freshness(
    facts: list[dict[str, Any]],
    analysis_as_of: str | date | None = None,
    *,
    structured_core: bool = True,
) -> dict[str, Any]:
    counts = {
        status: sum(fact["freshness_status"] == status for fact in facts)
        for status in ("fresh", "caution", "stale", "unknown")
    }
    core = [
        fact for fact in facts
        if fact["metric_name"] in CORE_FUNDAMENTAL_METRICS
    ] if structured_core else facts
    available_core = [fact for fact in core if fact.get("value") is not None]
    fresh_core = [fact for fact in available_core if fact["freshness_status"] == "fresh"]
    degraded_core = [
        fact for fact in available_core
        if fact["freshness_status"] in {"caution", "stale", "unknown"}
    ]
    if not facts:
        overall = "unknown"
    elif counts["stale"]:
        overall = "stale"
    elif counts["unknown"] or counts["caution"]:
        overall = "caution"
    else:
        overall = "fresh"
    current_usable = bool(fresh_core) if structured_core else bool(
        any(fact["freshness_status"] == "fresh" for fact in facts)
    )
    all_core_fresh = bool(available_core) and not degraded_core
    ages = [
        fact["period_age_days"]
        for fact in facts
        if isinstance(fact.get("period_age_days"), int)
        and fact["period_age_days"] >= 0
    ]
    return {
        "schema_version": "1.0",
        "analysis_as_of": analysis_date(analysis_as_of).isoformat(),
        "thresholds_days": dict(FRESHNESS_THRESHOLDS_DAYS),
        "caution_multiplier": CAUTION_MULTIPLIER,
        "facts": facts,
        "summary": {
            "fundamental_freshness_status": overall,
            "fresh_fact_count": counts["fresh"],
            "caution_fact_count": counts["caution"],
            "stale_fact_count": counts["stale"],
            "unknown_freshness_count": counts["unknown"],
            "oldest_fact_age_days": max(ages) if ages else None,
            "fresh_core_fact_count": len(fresh_core),
            "available_core_fact_count": len(available_core),
            "all_available_core_facts_fresh": all_core_fresh,
            "current_analysis_usable": current_usable,
        },
    }


def freshness_banner(
    report: str,
    freshness: dict[str, Any] | None,
    *,
    language: str = "zh",
) -> str:
    if not isinstance(freshness, dict):
        return report
    summary = freshness.get("summary") or {}
    facts = freshness.get("facts") or []
    status = summary.get("fundamental_freshness_status", "unknown")
    age_status = summary.get("age_freshness_status", status)
    alignment = summary.get("latest_period_alignment_status", "unknown")
    usable = bool(summary.get("current_analysis_usable"))
    cleaned = str(report or "")
    if not usable:
        # Preserve the meaning of explicit prohibitions before replacing
        # unsupported recency wording in affirmative prose.
        negative_prefix = r"((?:不能|不得)[^。\n]{0,100}(?:称为|写成|描述为))"
        cleaned = re.sub(
            negative_prefix + r"([“\"]?)最近一期([”\"]?)",
            r"\1\2当前有效期间\3",
            cleaned,
        )
        cleaned = re.sub(
            negative_prefix + r"([“\"]?)最新财务数据([”\"]?)",
            r"\1\2当前有效财务数据\3",
            cleaned,
        )
        cleaned = cleaned.replace("最新或最近一期", "当前有效")
        cleaned = cleaned.replace("最近一期", "历史期间（未通过新鲜度检查）")
        cleaned = cleaned.replace("最新财务数据", "历史财务数据（未通过新鲜度检查）")
        cleaned = cleaned.replace("最新一期", "历史期间（未通过新鲜度检查）")
        cleaned = cleaned.replace("最新期间", "历史期间（未通过新鲜度检查）")
    title = "Financial Data Freshness" if language == "en" else "财务数据新鲜度"
    warning = (
        "SEC已有更新申报；当前数值仅为historical_fallback，"
        "不得用于正常的当前基本面结论。"
        if alignment == "historical_fallback" and language != "en"
        else "A newer SEC filing exists; selected values are historical fallback only."
        if alignment == "historical_fallback"
        else
        "Current fundamental conclusions must not use these facts."
        if language == "en" and not usable
        else "这些事实不得用于正常的当前基本面结论。"
        if not usable
        else "Non-fresh facts are present; conclusions must be qualified."
        if language == "en" and status != "fresh"
        else "存在非fresh事实，结论必须降级并披露。"
        if status != "fresh"
        else "核心财务事实通过新鲜度检查。"
    )
    lines = [
        f"## {title}",
        "",
        f"- status: **{status}**",
        f"- age_freshness_status: **{age_status}**",
        f"- latest_period_alignment_status: **{alignment}**",
        f"- selected_period_end: {summary.get('selected_period_end')}",
        f"- latest_filing_period_end: {summary.get('latest_filing_period_end')}",
        f"- latest_filing_date: {summary.get('latest_filing_date')}",
        f"- latest_filing_form: {summary.get('latest_filing_form')}",
        f"- latest_filing_accession: {summary.get('latest_filing_accession')}",
        f"- period_lag_days: {summary.get('period_lag_days')}",
        f"- analysis_as_of: {freshness.get('analysis_as_of')}",
        f"- fresh/caution/stale/unknown: "
        f"{summary.get('fresh_fact_count', 0)}/"
        f"{summary.get('caution_fact_count', 0)}/"
        f"{summary.get('stale_fact_count', 0)}/"
        f"{summary.get('unknown_freshness_count', 0)}",
        f"- warning: **{warning}**",
    ]
    for fact in facts:
        if fact.get("metric_name") not in CORE_FUNDAMENTAL_METRICS:
            continue
        lines.append(
            f"- {fact['metric_name']}: period={fact.get('period_start')}.."
            f"{fact.get('period_end')}, filed={fact.get('filed_date')}, "
            f"period_age_days={fact.get('period_age_days')}, "
            f"age_freshness={fact.get('age_freshness_status')}, "
            f"effective_freshness={fact.get('freshness_status')}"
        )
    return "\n".join(lines) + "\n\n" + cleaned

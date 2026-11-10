from __future__ import annotations

from datetime import date
from typing import Any

from tools.financial_amount import (
    format_financial_per_share_en,
    format_financial_per_share_zh,
    make_financial_amount,
)
from tools.financial_freshness import analysis_date, build_structured_freshness


IFRS_PILOT_ISSUERS = {
    "TSM": {"reporting_currency": "TWD"},
    "SONY": {"reporting_currency": "JPY"},
    "TM": {"reporting_currency": "JPY"},
}

IFRS_CONCEPT_MAP: dict[str, tuple[str, ...]] = {
    "revenue": ("Revenue",),
    "net_income": (
        "ProfitLossAttributableToOwnersOfParent",
        "ProfitLoss",
    ),
    "eps": (
        "DilutedEarningsLossPerShare",
        "BasicEarningsLossPerShare",
    ),
    "operating_cash_flow": (
        "CashFlowsFromUsedInOperatingActivities",
    ),
    "capital_expenditure": (
        "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities",
        "PurchaseOfPropertyPlantAndEquipmentIntangibleAssetsOtherThanGoodwillInvestmentPropertyAndOtherNoncurrentAssets",
        "PurchaseOfPropertyPlantAndEquipment",
    ),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "cash": ("CashAndCashEquivalents",),
    "equity": ("Equity",),
    "total_debt_aggregate": ("Borrowings",),
    "total_debt_current": (
        "CurrentBorrowingsAndCurrentPortionOfNoncurrentBorrowings",
        "CurrentPortionOfLongtermBorrowings",
        "ShorttermBorrowings",
    ),
    "total_debt_noncurrent": (
        "LongtermBorrowings",
        "NoncurrentBorrowings",
    ),
}

IFRS_STRUCTURED_FORMS = {"20-F", "20-F/A", "6-K", "6-K/A"}


def _date(value: Any) -> date | None:
    try:
        return date.fromisoformat(str(value or "")[:10])
    except ValueError:
        return None


def _duration_days(record: dict[str, Any]) -> int | None:
    start = _date(record.get("start"))
    end = _date(record.get("end"))
    return (end - start).days if start and end else None


def flow_period_granularity(record: dict[str, Any]) -> str | None:
    duration = _duration_days(record)
    if duration is None:
        return None
    if 70 <= duration <= 110:
        return "quarterly"
    if 150 <= duration <= 200:
        return "half_year_ytd"
    if 240 <= duration <= 300:
        return "nine_month_ytd"
    if 300 <= duration <= 390:
        return "annual"
    return None


def _filing_record(recent: dict[str, Any], index: int) -> dict[str, Any]:
    def field(name: str) -> Any:
        values = recent.get(name) or []
        return values[index] if index < len(values) else None

    return {
        "filing_date": field("filingDate"),
        "form": field("form"),
        "period_end": field("reportDate"),
        "accession_number": field("accessionNumber"),
        "primary_document": field("primaryDocument"),
    }


def latest_foreign_filings(
    submissions: dict[str, Any],
    analysis_as_of: str | date | None = None,
) -> dict[str, Any]:
    """Return latest eligible 20-F and 6-K metadata by filing date."""

    as_of = analysis_date(analysis_as_of)
    recent = submissions.get("filings", {}).get("recent", {})
    by_group: dict[str, list[dict[str, Any]]] = {
        "latest_20f": [],
        "latest_6k": [],
    }
    excluded = []
    for index, form in enumerate(recent.get("form") or []):
        if form in {"20-F", "20-F/A"}:
            group = "latest_20f"
        elif form in {"6-K", "6-K/A"}:
            group = "latest_6k"
        else:
            continue
        record = _filing_record(recent, index)
        filed = _date(record.get("filing_date"))
        period_end = _date(record.get("period_end"))
        if filed is None or period_end is None:
            excluded.append({**record, "reason": "missing_or_invalid_date"})
        elif filed > as_of:
            excluded.append({**record, "reason": "filing_after_analysis_date"})
        elif period_end > as_of:
            excluded.append({**record, "reason": "period_after_analysis_date"})
        else:
            by_group[group].append(record)

    def latest(records: list[dict[str, Any]]) -> dict[str, Any] | None:
        return max(
            records,
            key=lambda item: (
                str(item.get("filing_date") or ""),
                str(item.get("period_end") or ""),
                str(item.get("accession_number") or ""),
            ),
            default=None,
        )

    return {
        "analysis_as_of": as_of.isoformat(),
        "latest_20f": latest(by_group["latest_20f"]),
        "latest_6k": latest(by_group["latest_6k"]),
        "excluded_candidates": excluded,
    }


def _entries(
    taxonomy_facts: dict[str, Any],
    concepts: tuple[str, ...],
    unit: str,
) -> list[dict[str, Any]]:
    results = []
    for priority, concept in enumerate(concepts):
        fact = taxonomy_facts.get(concept)
        if not isinstance(fact, dict):
            continue
        for item in fact.get("units", {}).get(unit, []):
            record = dict(item)
            record.update(
                {
                    "_taxonomy": "ifrs-full",
                    "_concept": concept,
                    "_unit": unit,
                    "_concept_priority": len(concepts) - priority,
                }
            )
            results.append(record)
    return results


def _eligible_flow(
    record: dict[str, Any],
    as_of: date,
) -> tuple[bool, str | None]:
    start = _date(record.get("start"))
    end = _date(record.get("end"))
    filed = _date(record.get("filed"))
    if start is None or end is None or filed is None:
        return False, "missing_or_invalid_date"
    if record.get("form") not in IFRS_STRUCTURED_FORMS:
        return False, "unsupported_form"
    if end > as_of or filed > as_of:
        return False, "future_period_or_filing"
    if end < start or flow_period_granularity(record) is None:
        return False, "abnormal_or_ambiguous_duration"
    if str(record.get("form")).startswith("20-F") and flow_period_granularity(record) != "annual":
        return False, "20f_nonannual_period"
    return True, None


def select_latest_ifrs_reporting_period(
    taxonomy_facts: dict[str, Any],
    currency: str,
    analysis_as_of: str | date | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    as_of = analysis_date(analysis_as_of)
    candidates = _entries(
        taxonomy_facts,
        IFRS_CONCEPT_MAP["revenue"],
        currency,
    )
    valid = []
    excluded = []
    for record in candidates:
        eligible, reason = _eligible_flow(record, as_of)
        if eligible:
            valid.append(record)
        else:
            excluded.append(_audit_record(record, reason or "excluded"))
    if not valid:
        raise ValueError(
            f"SEC Company Facts has no usable ifrs-full Revenue in {currency}."
        )
    valid.sort(
        key=lambda item: (
            str(item.get("end") or ""),
            str(item.get("filed") or ""),
            int(item.get("_concept_priority") or 0),
        )
    )
    selected = valid[-1]
    for record in valid[:-1]:
        reason = (
            "older_period"
            if record.get("end") != selected.get("end")
            else "superseded_duplicate"
        )
        excluded.append(_audit_record(record, reason))
    return selected, {
        "analysis_as_of": as_of.isoformat(),
        "taxonomy": "ifrs-full",
        "reporting_currency": currency,
        "candidate_count": len(candidates),
        "eligible_candidate_count": len(valid),
        "selected": source_record(
            selected,
            "latest eligible IFRS Revenue period in original reporting currency",
        ),
        "selected_reason": (
            "latest period_end with valid IFRS flow duration and filing date; "
            "original reporting currency only"
        ),
        "excluded_candidate_count": len(excluded),
        "excluded_candidates": sorted(
            excluded,
            key=lambda item: (
                str(item.get("end") or ""),
                str(item.get("filed") or ""),
            ),
            reverse=True,
        )[:100],
    }


def _audit_record(record: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "taxonomy": record.get("_taxonomy"),
        "concept": record.get("_concept"),
        "unit": record.get("_unit"),
        "form": record.get("form"),
        "start": record.get("start"),
        "end": record.get("end"),
        "filed": record.get("filed"),
        "accession": record.get("accn"),
        "reason": reason,
    }


def source_record(
    record: dict[str, Any] | None,
    selection_reason: str,
) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "taxonomy": record.get("_taxonomy", "ifrs-full"),
        "concept": record.get("_concept"),
        "original_value": record.get("val"),
        "original_unit": record.get("_unit"),
        "unit": record.get("_unit"),
        "start": record.get("start"),
        "end": record.get("end"),
        "filed": record.get("filed"),
        "form": record.get("form"),
        "fiscal_year": record.get("fy"),
        "fiscal_period": record.get("fp"),
        "accession": record.get("accn"),
        "frame": record.get("frame"),
        "period_granularity": (
            flow_period_granularity(record)
            if record.get("start")
            else "instant"
        ),
        "selection_reason": selection_reason,
    }


def _matching_record(
    taxonomy_facts: dict[str, Any],
    concepts: tuple[str, ...],
    unit: str,
    target: dict[str, Any],
    *,
    instant: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    candidates = _entries(taxonomy_facts, concepts, unit)
    matching = []
    excluded = []
    for record in candidates:
        reason = None
        if record.get("form") != target.get("form"):
            reason = "form_mismatch"
        elif record.get("end") != target.get("end"):
            reason = "period_end_mismatch"
        elif instant and record.get("start"):
            reason = "expected_instant_fact"
        elif not instant and record.get("start") != target.get("start"):
            reason = "flow_scope_mismatch"
        elif not instant and flow_period_granularity(record) != flow_period_granularity(target):
            reason = "flow_granularity_mismatch"
        if reason:
            excluded.append(_audit_record(record, reason))
        else:
            matching.append(record)
    matching.sort(
        key=lambda item: (
            str(item.get("filed") or ""),
            int(item.get("_concept_priority") or 0),
            str(item.get("accn") or ""),
        )
    )
    selected = matching[-1] if matching else None
    selected_reason = (
        "exact taxonomy, original unit, form and period match; latest filed "
        "duplicate and concept priority win"
        if selected
        else "no exact taxonomy, original unit, form and period match"
    )
    selected_accession = selected.get("accn") if selected else None
    for record in matching[:-1]:
        if record.get("accn") != selected_accession or record.get("_concept") != selected.get("_concept"):
            excluded.append(_audit_record(record, "lower_concept_or_duplicate_priority"))
    return selected, {
        "selected": source_record(selected, selected_reason),
        "selected_reason": selected_reason,
        "excluded_candidate_count": len(excluded),
        "excluded_candidates": excluded[:100],
    }


def _previous_flow_record(
    taxonomy_facts: dict[str, Any],
    concepts: tuple[str, ...],
    unit: str,
    target: dict[str, Any],
) -> dict[str, Any] | None:
    target_end = _date(target.get("end"))
    target_duration = _duration_days(target)
    if target_end is None or target_duration is None:
        return None
    candidates = []
    for record in _entries(taxonomy_facts, concepts, unit):
        end = _date(record.get("end"))
        duration = _duration_days(record)
        if end is None or duration is None:
            continue
        if record.get("form") != target.get("form"):
            continue
        if 330 <= (target_end - end).days <= 400 and abs(duration - target_duration) <= 20:
            candidates.append(record)
    return max(
        candidates,
        key=lambda item: (str(item.get("end") or ""), str(item.get("filed") or "")),
        default=None,
    )


def _billions(value: float | int | None) -> float | None:
    return round(float(value) / 1_000_000_000, 2) if value is not None else None


def _amount(
    value: float | int | None,
    *,
    currency: str,
    period: dict[str, str | None],
    statement_scope: str,
) -> dict[str, Any] | None:
    amount = make_financial_amount(
        _billions(value),
        currency=currency,
        scale="billion",
        period=period,
        statement_scope=statement_scope,
        display_scale="billion",
    )
    if amount is not None:
        amount["original_value"] = value
        amount["original_currency"] = currency
        amount["original_unit"] = currency
        amount["conversion_applied"] = False
    return amount


def _growth(current: float | None, previous: float | None) -> float | None:
    if current is None or previous in {None, 0}:
        return None
    return round((current / previous - 1) * 100, 2)


def _same_period_currency(*records: dict[str, Any] | None) -> bool:
    if not records or any(record is None for record in records):
        return False
    usable = [record for record in records if record is not None]
    return bool(usable) and len({record.get("_unit") for record in usable}) == 1 and len(
        {(record.get("start"), record.get("end")) for record in usable}
    ) == 1


def _accession_fact_count(company_facts: dict[str, Any], accession: str | None) -> int:
    if not accession:
        return 0
    count = 0
    for concepts in company_facts.get("facts", {}).values():
        for fact in concepts.values():
            for records in fact.get("units", {}).values():
                count += sum(item.get("accn") == accession for item in records)
    return count


def _accession_core_fact_count(
    company_facts: dict[str, Any],
    accession: str | None,
    currency: str,
) -> int:
    if not accession:
        return 0
    concepts = {
        concept
        for metric in (
            "revenue",
            "net_income",
            "operating_cash_flow",
            "assets",
        )
        for concept in IFRS_CONCEPT_MAP[metric]
    }
    taxonomy = company_facts.get("facts", {}).get("ifrs-full", {})
    count = 0
    for concept in concepts:
        for record in taxonomy.get(concept, {}).get("units", {}).get(currency, []):
            count += record.get("accn") == accession
    return count


def build_ifrs_fundamental_metrics(
    *,
    symbol: str,
    symbols: dict[str, str],
    company_facts: dict[str, Any],
    submissions: dict[str, Any],
    cik: str,
    source_url: str,
    submissions_url: str,
    analysis_as_of: str | date | None = None,
) -> dict[str, Any]:
    config = IFRS_PILOT_ISSUERS.get(symbol)
    if config is None:
        raise ValueError(f"{symbol} is outside the IFRS pilot scope.")
    currency = config["reporting_currency"]
    taxonomy_facts = company_facts.get("facts", {}).get("ifrs-full")
    if not isinstance(taxonomy_facts, dict):
        raise ValueError("SEC Company Facts has no usable ifrs-full structured facts.")
    as_of = analysis_date(analysis_as_of)
    filings = latest_foreign_filings(submissions, as_of)
    latest_20f = filings.get("latest_20f")
    latest_6k = filings.get("latest_6k")
    target, selection_audit = select_latest_ifrs_reporting_period(
        taxonomy_facts,
        currency,
        as_of,
    )

    selected: dict[str, dict[str, Any] | None] = {}
    audits: dict[str, dict[str, Any]] = {}
    for metric in (
        "revenue",
        "net_income",
        "eps",
        "operating_cash_flow",
        "capital_expenditure",
    ):
        unit = f"{currency}/shares" if metric == "eps" else currency
        selected[metric], audits[metric] = _matching_record(
            taxonomy_facts,
            IFRS_CONCEPT_MAP[metric],
            unit,
            target,
        )
    for metric in ("assets", "liabilities", "cash", "equity"):
        selected[metric], audits[metric] = _matching_record(
            taxonomy_facts,
            IFRS_CONCEPT_MAP[metric],
            currency,
            target,
            instant=True,
        )

    debt_aggregate, debt_audit = _matching_record(
        taxonomy_facts,
        IFRS_CONCEPT_MAP["total_debt_aggregate"],
        currency,
        target,
        instant=True,
    )
    debt_records: list[dict[str, Any]] = []
    if debt_aggregate is not None:
        debt_records = [debt_aggregate]
        debt_reason = "IFRS Borrowings aggregate selected; components not summed"
    else:
        current_debt, current_audit = _matching_record(
            taxonomy_facts,
            IFRS_CONCEPT_MAP["total_debt_current"],
            currency,
            target,
            instant=True,
        )
        noncurrent_debt, noncurrent_audit = _matching_record(
            taxonomy_facts,
            IFRS_CONCEPT_MAP["total_debt_noncurrent"],
            currency,
            target,
            instant=True,
        )
        audits["total_debt_current"] = current_audit
        audits["total_debt_noncurrent"] = noncurrent_audit
        if current_debt is not None and noncurrent_debt is not None:
            debt_records = [current_debt, noncurrent_debt]
            debt_reason = "exact-date current and noncurrent borrowings summed in original currency"
        else:
            debt_reason = "no complete non-overlapping debt aggregate or component pair"
    audits["total_debt"] = debt_audit

    def raw(metric: str) -> float | None:
        record = selected.get(metric)
        return float(record["val"]) if record is not None else None

    revenue = raw("revenue")
    net_income = raw("net_income")
    eps = raw("eps")
    operating_cash_flow = raw("operating_cash_flow")
    capex = raw("capital_expenditure")
    assets = raw("assets")
    liabilities = raw("liabilities")
    cash = raw("cash")
    equity = raw("equity")
    total_debt = sum(float(item["val"]) for item in debt_records) if debt_records else None

    free_cash_flow = None
    if _same_period_currency(selected["operating_cash_flow"], selected["capital_expenditure"]):
        free_cash_flow = operating_cash_flow - capex
    net_margin = None
    if revenue not in {None, 0} and _same_period_currency(selected["revenue"], selected["net_income"]):
        net_margin = round(net_income / revenue, 4)
    equity_ratio = round(equity / assets, 4) if assets not in {None, 0} and equity is not None else None
    operating_cash_flow_to_net_income = (
        round(operating_cash_flow / net_income, 4)
        if net_income not in {None, 0}
        and _same_period_currency(
            selected["operating_cash_flow"],
            selected["net_income"],
        )
        else None
    )

    previous_revenue = _previous_flow_record(
        taxonomy_facts, IFRS_CONCEPT_MAP["revenue"], currency, target
    )
    previous_net_income = _previous_flow_record(
        taxonomy_facts, IFRS_CONCEPT_MAP["net_income"], currency, target
    )
    income_period = {"start": target.get("start"), "end": target.get("end")}
    balance_period = {"start": None, "end": target.get("end")}
    cash_flow_period = {
        "start": selected["operating_cash_flow"].get("start") if selected["operating_cash_flow"] else None,
        "end": selected["operating_cash_flow"].get("end") if selected["operating_cash_flow"] else None,
    }
    amounts = {
        "revenue": _amount(revenue, currency=currency, period=income_period, statement_scope="annual_income_statement"),
        "net_income": _amount(net_income, currency=currency, period=income_period, statement_scope="annual_income_statement"),
        "operating_cash_flow": _amount(operating_cash_flow, currency=currency, period=cash_flow_period, statement_scope="annual_cash_flow"),
        "capital_expenditure": _amount(capex, currency=currency, period=cash_flow_period, statement_scope="annual_cash_flow"),
        "free_cash_flow": _amount(free_cash_flow, currency=currency, period=cash_flow_period, statement_scope="annual_cash_flow"),
        "cash": _amount(cash, currency=currency, period=balance_period, statement_scope="period_end_balance_sheet"),
        "total_debt": _amount(total_debt, currency=currency, period=balance_period, statement_scope="period_end_balance_sheet"),
        "assets": _amount(assets, currency=currency, period=balance_period, statement_scope="period_end_balance_sheet"),
        "liabilities": _amount(liabilities, currency=currency, period=balance_period, statement_scope="period_end_balance_sheet"),
        "equity": _amount(equity, currency=currency, period=balance_period, statement_scope="period_end_balance_sheet"),
    }
    sources = {
        metric: source_record(record, audits[metric]["selected_reason"])
        for metric, record in selected.items()
    }
    sources["total_debt"] = (
        {
            "concepts": [source_record(item, debt_reason) for item in debt_records],
            "end": target.get("end"),
            "filed": max((str(item.get("filed") or "") for item in debt_records), default="") or None,
            "selection_reason": debt_reason,
        }
        if debt_records
        else None
    )
    sources["free_cash_flow"] = (
        {
            "concepts": [sources["operating_cash_flow"], sources["capital_expenditure"]],
            "end": cash_flow_period["end"],
            "filed": selected["operating_cash_flow"].get("filed") if selected["operating_cash_flow"] else None,
            "selection_reason": "same-period same-currency operating cash flow minus capital expenditure",
        }
        if free_cash_flow is not None
        else None
    )
    sources["revenue_yoy"] = (
        {**(sources["revenue"] or {}), "comparison": source_record(previous_revenue, "same-duration prior-year flow")}
        if previous_revenue is not None else None
    )
    sources["net_income_yoy"] = (
        {**(sources["net_income"] or {}), "comparison": source_record(previous_net_income, "same-duration prior-year flow")}
        if previous_net_income is not None else None
    )
    sources["net_margin"] = (
        {"concepts": [sources["net_income"], sources["revenue"]], "end": target.get("end"), "filed": target.get("filed"), "selection_reason": "same-period same-currency ratio"}
        if net_margin is not None else None
    )
    sources["equity_ratio"] = (
        {"concepts": [sources["equity"], sources["assets"]], "end": target.get("end"), "filed": target.get("filed"), "selection_reason": "same-date same-currency ratio"}
        if equity_ratio is not None else None
    )
    sources["operating_cash_flow_to_net_income"] = (
        {
            "concepts": [
                sources["operating_cash_flow"],
                sources["net_income"],
            ],
            "end": target.get("end"),
            "filed": target.get("filed"),
            "selection_reason": "same-period same-currency ratio",
        }
        if operating_cash_flow_to_net_income is not None
        else None
    )

    latest_20f_accn = latest_20f.get("accession_number") if latest_20f else None
    latest_6k_accn = latest_6k.get("accession_number") if latest_6k else None
    latest_20f_count = _accession_fact_count(company_facts, latest_20f_accn)
    latest_6k_count = _accession_fact_count(company_facts, latest_6k_accn)
    latest_6k_core_count = _accession_core_fact_count(
        company_facts,
        latest_6k_accn,
        currency,
    )
    latest_financial_filing = (
        latest_6k if latest_6k_core_count else latest_20f
    )
    result: dict[str, Any] = {
        "symbol": symbol,
        "source_symbol": symbols["sec"],
        "symbol_mapping": symbols,
        "company": company_facts.get("entityName"),
        "cik": cik,
        "taxonomy": "ifrs-full",
        "reporting_currency": currency,
        "form": target.get("form"),
        "fiscal_year": target.get("fy"),
        "fiscal_period": target.get("fp"),
        "period_granularity": flow_period_granularity(target),
        "income_period_start": target.get("start"),
        "income_period_end": target.get("end"),
        "cash_flow_period_start": cash_flow_period["start"],
        "cash_flow_period_end": cash_flow_period["end"],
        "filing_date": target.get("filed"),
        "financial_amounts": amounts,
        "eps_diluted_per_share": eps,
        "eps_unit": f"{currency}/shares",
        "eps_display_zh": (
            format_financial_per_share_zh(eps, currency) if eps is not None else None
        ),
        "eps_display_en": (
            format_financial_per_share_en(eps, currency) if eps is not None else None
        ),
        "eps_diluted_usd_per_share": eps if currency == "USD" else None,
        "source_url": source_url,
        "source_records": sources,
        "analysis_as_of": as_of.isoformat(),
        "selection_audit": selection_audit,
        "metric_selection_audit": audits,
        "latest_applicable_filing": latest_financial_filing,
        "latest_20f": latest_20f,
        "latest_6k": latest_6k,
        "latest_filing_source_url": submissions_url,
        "latest_filing_selection_audit": filings,
        "latest_20f_companyfacts_record_count": latest_20f_count,
        "latest_6k_companyfacts_record_count": latest_6k_count,
        "latest_6k_structured_core_fact_count": latest_6k_core_count,
        "latest_6k_structured_core_present": latest_6k_core_count > 0,
        "structured_period_alignment": (
            "aligned" if latest_20f and target.get("end") == latest_20f.get("period_end") else "historical_fallback"
        ),
        "financial_amount_display_rule": (
            "Use financial_amounts.display_en for English output and preserve "
            "the original reporting currency without converting to USD."
        ),
        "revenue_billion_original_currency": _billions(revenue),
        "net_income_billion_original_currency": _billions(net_income),
        "operating_cash_flow_billion_original_currency": _billions(operating_cash_flow),
        "capital_expenditure_billion_original_currency": _billions(capex),
        "free_cash_flow_billion_original_currency": _billions(free_cash_flow),
        "cash_billion_original_currency": _billions(cash),
        "total_debt_billion_original_currency": _billions(total_debt),
        "assets_billion_original_currency": _billions(assets),
        "liabilities_billion_original_currency": _billions(liabilities),
        "equity_billion_original_currency": _billions(equity),
        "net_margin": net_margin,
        "equity_ratio": equity_ratio,
        "operating_cash_flow_to_net_income": operating_cash_flow_to_net_income,
        "revenue_yoy_percent": _growth(revenue, float(previous_revenue["val"]) if previous_revenue else None),
        "net_income_yoy_percent": _growth(net_income, float(previous_net_income["val"]) if previous_net_income else None),
        "cash_flow_basis": "IFRS flow facts retain exact period; annual, quarterly and FYTD are never mixed",
        "free_cash_flow_definition": "same-period same-currency operating cash flow minus capital expenditure",
        "currency_conversion": None,
        "source": "SEC Company Facts API (ifrs-full)",
    }
    result["fundamental_evidence"] = {
        "schema_version": "1.0",
        "source": "SEC Company Facts API",
        "source_url": source_url,
        "canonical_symbol": symbol,
        "vendor_symbol": symbols["sec"],
        "cik": cik,
        "taxonomy": "ifrs-full",
        "reporting_currency": currency,
        "analysis_as_of": as_of.isoformat(),
        "selected_period": {
            "form": target.get("form"),
            "fiscal_year": target.get("fy"),
            "fiscal_period": target.get("fp"),
            "period_granularity": flow_period_granularity(target),
            "start": target.get("start"),
            "end": target.get("end"),
            "filed": target.get("filed"),
            "accession": target.get("accn"),
        },
        "latest_20f": latest_20f,
        "latest_6k": latest_6k,
        "metrics": {
            metric: {
                "canonical_metric": metric,
                "amount": amounts.get(metric),
                "source_record": sources.get(metric),
                "selection_audit": audits.get(metric),
                "supported": amounts.get(metric) is not None,
            }
            for metric in (
                "revenue",
                "net_income",
                "operating_cash_flow",
                "capital_expenditure",
                "free_cash_flow",
                "assets",
                "liabilities",
                "cash",
                "total_debt",
                "equity",
            )
        },
        "eps": {
            "canonical_metric": "EPS",
            "value": eps,
            "unit": f"{currency}/shares",
            "source_record": sources.get("eps"),
            "selection_audit": audits.get("eps"),
            "supported": eps is not None,
        },
        "currency_conversion": None,
    }
    result["fundamental_freshness"] = build_structured_freshness(
        result,
        as_of,
        latest_filing=latest_financial_filing,
    )
    summary = result["fundamental_freshness"]["summary"]
    summary.update(
        {
            "taxonomy": "ifrs-full",
            "reporting_currency": currency,
            "latest_20f": latest_20f,
            "latest_6k": latest_6k,
            "latest_20f_companyfacts_record_count": latest_20f_count,
            "latest_6k_companyfacts_record_count": latest_6k_count,
            "latest_6k_structured_core_fact_count": latest_6k_core_count,
            "latest_6k_structured_core_present": latest_6k_core_count > 0,
        }
    )
    if latest_6k and latest_6k_core_count == 0:
        summary["foreign_interim_status"] = "unknown"
        summary["foreign_interim_reason"] = (
            "The latest 6-K is not represented by structured core facts in SEC "
            "Company Facts; values must not be inferred from filing text or news."
        )
        summary["current_analysis_usable"] = False
        summary["all_available_core_facts_fresh"] = False
    return result

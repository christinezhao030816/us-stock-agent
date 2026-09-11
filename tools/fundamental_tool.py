import os
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv

from tools.financial_amount import make_financial_amount
from tools.ticker_symbols import (
    canonicalize_symbol,
    sec_symbol,
    ticker_mapping,
)


load_dotenv()

SEC_BASE_URL = "https://data.sec.gov"
SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")

if not SEC_USER_AGENT:
    raise RuntimeError("未读取到 SEC_USER_AGENT")


def _get_json(url: str) -> dict:
    headers = {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }

    with httpx.Client(timeout=30.0, headers=headers) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _get_company(symbol: str) -> dict:
    companies = _get_json(
        "https://www.sec.gov/files/company_tickers.json"
    )

    symbol = sec_symbol(symbol)

    for company in companies.values():
        if company["ticker"].upper() == symbol:
            return company

    raise ValueError(f"SEC 中找不到股票代码：{symbol}")


def _entries(
    us_gaap: dict,
    concepts: list[str],
    unit: str = "USD",
) -> list[dict]:
    results = []

    for concept in concepts:
        fact = us_gaap.get(concept, {})
        units = fact.get("units", {})

        for item in units.get(unit, []):
            record = dict(item)
            record["_concept"] = concept
            record["_unit"] = unit
            results.append(record)

    return results


def _duration_days(item: dict) -> int:
    if not item.get("start") or not item.get("end"):
        return 0

    start = datetime.fromisoformat(item["start"])
    end = datetime.fromisoformat(item["end"])

    return (end - start).days


def _latest_reporting_period(us_gaap: dict) -> dict:
    revenue_entries = _entries(
        us_gaap,
        [
            "RevenueFromContractWithCustomerExcludingAssessedTax",
            "Revenues",
            "SalesRevenueNet",
        ],
    )

    valid = [
        item
        for item in revenue_entries
        if item.get("form") in {"10-Q", "10-K"}
        and item.get("start")
        and item.get("end")
        and item.get("filed")
    ]

    if not valid:
        raise ValueError("未找到可用的 SEC 营业收入记录")

    valid.sort(
        key=lambda item: (
            item["end"],
            item["filed"],
        )
    )

    return valid[-1]


def _period_record(
    us_gaap: dict,
    concepts: list[str],
    target: dict,
    instant: bool = False,
    match_start: bool = True,
) -> dict | None:
    candidates = []

    for item in _entries(us_gaap, concepts):
        if item.get("form") not in {"10-Q", "10-K"}:
            continue

        if item.get("end") != target.get("end"):
            continue

        if (
            not instant
            and match_start
            and item.get("start") != target.get("start")
        ):
            continue

        candidates.append(item)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item.get("filed", ""),
            _duration_days(item),
        )
    )

    return candidates[-1]


def _period_value(
    us_gaap: dict,
    concepts: list[str],
    target: dict,
    instant: bool = False,
    match_start: bool = True,
) -> float | None:
    record = _period_record(
        us_gaap=us_gaap,
        concepts=concepts,
        target=target,
        instant=instant,
        match_start=match_start,
    )

    if record is None:
        return None

    return float(record["val"])


def _prior_period_value(
    us_gaap: dict,
    concepts: list[str],
    target: dict,
) -> float | None:
    record = _prior_period_record(us_gaap, concepts, target)
    return float(record["val"]) if record is not None else None


def _prior_period_record(
    us_gaap: dict,
    concepts: list[str],
    target: dict,
) -> dict | None:
    target_start = datetime.fromisoformat(target["start"])
    target_end = datetime.fromisoformat(target["end"])
    target_duration = (target_end - target_start).days
    candidates = []

    for item in _entries(us_gaap, concepts):
        if item.get("form") != target.get("form"):
            continue

        if not item.get("start") or not item.get("end"):
            continue

        item_start = datetime.fromisoformat(item["start"])
        item_end = datetime.fromisoformat(item["end"])
        item_duration = (item_end - item_start).days
        days_difference = (target_end - item_end).days

        if 330 <= days_difference <= 400:
            if abs(item_duration - target_duration) <= 20:
                candidates.append(item)

    if not candidates:
        return None

    candidates.sort(
        key=lambda item: (
            item["end"],
            item.get("filed", ""),
        )
    )

    return candidates[-1]


def _source_record(record: dict | None) -> dict[str, Any] | None:
    if record is None:
        return None
    return {
        "concept": record.get("_concept"),
        "unit": record.get("_unit", "USD"),
        "start": record.get("start"),
        "end": record.get("end"),
        "filed": record.get("filed"),
        "form": record.get("form"),
        "fiscal_year": record.get("fy"),
        "fiscal_period": record.get("fp"),
        "accession": record.get("accn"),
        "frame": record.get("frame"),
    }


def _billions(value: float | None) -> float | None:
    if value is None:
        return None

    return round(value / 1_000_000_000, 2)


def _growth_rate(
    current: float | None,
    previous: float | None,
) -> float | None:
    if current is None or previous in {None, 0}:
        return None

    return round((current / previous - 1) * 100, 2)


def _amount(
    value: float | None,
    *,
    period: dict[str, str | None],
    statement_scope: str,
) -> dict[str, Any] | None:
    return make_financial_amount(
        _billions(value),
        currency="USD",
        scale="billion",
        period=period,
        statement_scope=statement_scope,
        display_scale="billion",
    )


def get_fundamental_metrics(symbol: str) -> dict[str, Any]:
    symbol = canonicalize_symbol(symbol)
    symbols = ticker_mapping(symbol)
    company = _get_company(symbol)

    cik = str(company["cik_str"]).zfill(10)

    company_facts = _get_json(
        f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    )

    us_gaap = company_facts["facts"]["us-gaap"]
    target = _latest_reporting_period(us_gaap)

    revenue_concepts = [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
    ]

    revenue_record = _period_record(
        us_gaap,
        revenue_concepts,
        target,
    )
    revenue = float(revenue_record["val"]) if revenue_record else None

    net_income_record = _period_record(
        us_gaap,
        ["NetIncomeLoss"],
        target,
    )
    net_income = (
        float(net_income_record["val"]) if net_income_record else None
    )

    eps_facts = {
        concept: {
            "units": {
                "USD": fact.get("units", {}).get("USD/shares", [])
            }
        }
        for concept, fact in us_gaap.items()
    }
    eps_record = _period_record(
        eps_facts,
        ["EarningsPerShareDiluted"],
        target,
    ) or _period_record(
        eps_facts,
        ["EarningsPerShareBasic"],
        target,
    )
    if eps_record is not None:
        eps_record["_unit"] = "USD/shares"
    eps = float(eps_record["val"]) if eps_record else None

    operating_cash_flow_record = _period_record(
        us_gaap,
        ["NetCashProvidedByUsedInOperatingActivities"],
        target,
        match_start=False,
    )

    operating_cash_flow = None
    cash_flow_target = None

    if operating_cash_flow_record is not None:
        operating_cash_flow = float(
            operating_cash_flow_record["val"]
        )

        cash_flow_target = {
            **target,
            "start": operating_cash_flow_record.get("start"),
            "end": operating_cash_flow_record.get("end"),
        }

    capital_expenditure = None
    capital_expenditure_record = None
    cash_flow_net_income = None
    cash_flow_net_income_record = None

    if cash_flow_target is not None:
        capital_expenditure_record = _period_record(
            us_gaap,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsForAdditionsToPropertyPlantAndEquipment",
            ],
            cash_flow_target,
        )
        capital_expenditure = (
            float(capital_expenditure_record["val"])
            if capital_expenditure_record
            else None
        )

        cash_flow_net_income_record = _period_record(
            us_gaap,
            ["NetIncomeLoss"],
            cash_flow_target,
        )
        cash_flow_net_income = (
            float(cash_flow_net_income_record["val"])
            if cash_flow_net_income_record
            else None
        )

    assets_record = _period_record(
        us_gaap,
        ["Assets"],
        target,
        instant=True,
    )
    assets = float(assets_record["val"]) if assets_record else None

    equity_record = _period_record(
        us_gaap,
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        target,
        instant=True,
    )
    equity = float(equity_record["val"]) if equity_record else None

    cash_record = _period_record(
        us_gaap,
        [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ],
        target,
        instant=True,
    )
    cash = float(cash_record["val"]) if cash_record else None

    short_term_debt_record = _period_record(
        us_gaap,
        [
            "ShortTermBorrowings",
            "ShortTermDebtCurrent",
        ],
        target,
        instant=True,
    )
    short_term_debt = (
        float(short_term_debt_record["val"])
        if short_term_debt_record
        else None
    )

    current_long_term_debt_record = _period_record(
        us_gaap,
        [
            "LongTermDebtCurrent",
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
        ],
        target,
        instant=True,
    )
    current_long_term_debt = (
        float(current_long_term_debt_record["val"])
        if current_long_term_debt_record
        else None
    )

    noncurrent_debt_record = _period_record(
        us_gaap,
        [
            "LongTermDebtNoncurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        ],
        target,
        instant=True,
    )
    noncurrent_debt = (
        float(noncurrent_debt_record["val"])
        if noncurrent_debt_record
        else None
    )

    previous_revenue_record = _prior_period_record(
        us_gaap,
        revenue_concepts,
        target,
    )
    previous_revenue = (
        float(previous_revenue_record["val"])
        if previous_revenue_record
        else None
    )

    previous_net_income_record = _prior_period_record(
        us_gaap,
        ["NetIncomeLoss"],
        target,
    )
    previous_net_income = (
        float(previous_net_income_record["val"])
        if previous_net_income_record
        else None
    )

    free_cash_flow = None

    if (
        operating_cash_flow is not None
        and capital_expenditure is not None
    ):
        free_cash_flow = (
            operating_cash_flow - capital_expenditure
        )

    debt_values = [
        value
        for value in [
            short_term_debt,
            current_long_term_debt,
            noncurrent_debt,
        ]
        if value is not None
    ]

    total_debt = sum(debt_values) if debt_values else None

    net_margin = None

    if revenue and net_income is not None:
        net_margin = round(net_income / revenue, 4)

    equity_ratio = None

    if assets and equity is not None:
        equity_ratio = round(equity / assets, 4)

    operating_cash_flow_to_net_income = None

    if (
        cash_flow_net_income
        and operating_cash_flow is not None
    ):
        operating_cash_flow_to_net_income = round(
            operating_cash_flow / cash_flow_net_income,
            4,
        )

    income_period = {
        "start": target.get("start"),
        "end": target.get("end"),
    }
    cash_flow_period = {
        "start": (
            cash_flow_target.get("start")
            if cash_flow_target
            else None
        ),
        "end": (
            cash_flow_target.get("end")
            if cash_flow_target
            else None
        ),
    }
    balance_sheet_period = {
        "start": None,
        "end": target.get("end"),
    }

    financial_amounts = {
        "revenue": _amount(
            revenue,
            period=income_period,
            statement_scope="income_statement_period",
        ),
        "net_income": _amount(
            net_income,
            period=income_period,
            statement_scope="income_statement_period",
        ),
        "operating_cash_flow": _amount(
            operating_cash_flow,
            period=cash_flow_period,
            statement_scope="fiscal_year_to_date_cash_flow",
        ),
        "capital_expenditure": _amount(
            capital_expenditure,
            period=cash_flow_period,
            statement_scope="fiscal_year_to_date_cash_flow",
        ),
        "free_cash_flow": _amount(
            free_cash_flow,
            period=cash_flow_period,
            statement_scope="fiscal_year_to_date_cash_flow",
        ),
        "cash": _amount(
            cash,
            period=balance_sheet_period,
            statement_scope="period_end_balance_sheet",
        ),
        "total_debt": _amount(
            total_debt,
            period=balance_sheet_period,
            statement_scope="period_end_balance_sheet",
        ),
        "assets": _amount(
            assets,
            period=balance_sheet_period,
            statement_scope="period_end_balance_sheet",
        ),
        "equity": _amount(
            equity,
            period=balance_sheet_period,
            statement_scope="period_end_balance_sheet",
        ),
    }

    debt_source_records = [
        _source_record(record)
        for record in (
            short_term_debt_record,
            current_long_term_debt_record,
            noncurrent_debt_record,
        )
        if record is not None
    ]
    source_records = {
        "revenue": _source_record(revenue_record),
        "net_income": _source_record(net_income_record),
        "eps": _source_record(eps_record),
        "operating_cash_flow": _source_record(
            operating_cash_flow_record
        ),
        "capital_expenditure": _source_record(
            capital_expenditure_record
        ),
        "free_cash_flow": {
            "concepts": [
                item
                for item in (
                    _source_record(operating_cash_flow_record),
                    _source_record(capital_expenditure_record),
                )
                if item is not None
            ],
            "end": cash_flow_period.get("end"),
            "filed": (
                operating_cash_flow_record.get("filed")
                if operating_cash_flow_record
                else target.get("filed")
            ),
        } if free_cash_flow is not None else None,
        "assets": _source_record(assets_record),
        "equity": _source_record(equity_record),
        "cash": _source_record(cash_record),
        "total_debt": {
            "concepts": debt_source_records,
            "end": target.get("end"),
            "filed": max(
                (str(item.get("filed") or "") for item in debt_source_records),
                default="",
            ) or target.get("filed"),
        } if debt_source_records else None,
        "revenue_yoy": {
            **(_source_record(revenue_record) or {}),
            "comparison": _source_record(previous_revenue_record),
        } if revenue_record and previous_revenue_record else None,
        "net_income_yoy": {
            **(_source_record(net_income_record) or {}),
            "comparison": _source_record(previous_net_income_record),
        } if net_income_record and previous_net_income_record else None,
        "net_margin": {
            "concepts": [
                item
                for item in (
                    _source_record(net_income_record),
                    _source_record(revenue_record),
                )
                if item is not None
            ],
            "end": target.get("end"),
            "filed": target.get("filed"),
        } if net_margin is not None else None,
        "equity_ratio": {
            "concepts": [
                item
                for item in (
                    _source_record(equity_record),
                    _source_record(assets_record),
                )
                if item is not None
            ],
            "end": target.get("end"),
            "filed": target.get("filed"),
        } if equity_ratio is not None else None,
        "operating_cash_flow_to_net_income": {
            "concepts": [
                item
                for item in (
                    _source_record(operating_cash_flow_record),
                    _source_record(cash_flow_net_income_record),
                )
                if item is not None
            ],
            "end": cash_flow_period.get("end"),
            "filed": target.get("filed"),
        } if operating_cash_flow_to_net_income is not None else None,
    }

    source_url = (
        f"{SEC_BASE_URL}/api/xbrl/companyfacts/CIK{cik}.json"
    )

    return {
        "symbol": symbol,
        "source_symbol": symbols["sec"],
        "symbol_mapping": symbols,
        "company": company_facts.get("entityName"),
        "cik": cik,
        "form": target.get("form"),
        "fiscal_year": target.get("fy"),
        "fiscal_period": target.get("fp"),
        "income_period_start": target.get("start"),
        "income_period_end": target.get("end"),
        "cash_flow_period_start": (
            cash_flow_target.get("start")
            if cash_flow_target
            else None
        ),
        "cash_flow_period_end": (
            cash_flow_target.get("end")
            if cash_flow_target
            else None
        ),
        "filing_date": target.get("filed"),
        "financial_amounts": financial_amounts,
        "eps_diluted_usd_per_share": eps,
        "source_url": source_url,
        "source_records": source_records,
        "financial_amount_display_rule": (
            "报告金额时逐字使用 financial_amounts 中的 "
            "display_zh；不得改写数值或单位"
        ),
        "revenue_billion_usd": _billions(revenue),
        "net_income_billion_usd": _billions(net_income),
        "operating_cash_flow_billion_usd": _billions(
            operating_cash_flow
        ),
        "capital_expenditure_billion_usd": _billions(
            capital_expenditure
        ),
        "free_cash_flow_billion_usd": _billions(
            free_cash_flow
        ),
        "cash_billion_usd": _billions(cash),
        "total_debt_billion_usd": _billions(total_debt),
        "assets_billion_usd": _billions(assets),
        "equity_billion_usd": _billions(equity),
        "net_margin": net_margin,
        "equity_ratio": equity_ratio,
        "operating_cash_flow_to_net_income": (
            operating_cash_flow_to_net_income
        ),
        "revenue_yoy_percent": _growth_rate(
            revenue,
            previous_revenue,
        ),
        "net_income_yoy_percent": _growth_rate(
            net_income,
            previous_net_income,
        ),
        "cash_flow_basis": (
            "经营现金流、资本支出和自由现金流采用"
            "SEC财年初至报告期末累计值"
        ),
        "free_cash_flow_definition": (
            "自由现金流等于经营现金流减资本支出"
        ),
        "source": "SEC Company Facts API",
    }

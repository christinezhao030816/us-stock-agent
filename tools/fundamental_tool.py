import os
from datetime import datetime
from typing import Any

import httpx
from dotenv import load_dotenv


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

    symbol = symbol.upper()

    for company in companies.values():
        if company["ticker"].upper() == symbol:
            return company

    raise ValueError(f"SEC 中找不到股票代码：{symbol}")


def _entries(
    us_gaap: dict,
    concepts: list[str],
) -> list[dict]:
    results = []

    for concept in concepts:
        fact = us_gaap.get(concept, {})
        units = fact.get("units", {})

        for item in units.get("USD", []):
            record = dict(item)
            record["_concept"] = concept
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

    return float(candidates[-1]["val"])


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


def get_fundamental_metrics(symbol: str) -> dict[str, Any]:
    symbol = symbol.strip().upper()
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

    revenue = _period_value(
        us_gaap,
        revenue_concepts,
        target,
    )

    net_income = _period_value(
        us_gaap,
        ["NetIncomeLoss"],
        target,
    )

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
    cash_flow_net_income = None

    if cash_flow_target is not None:
        capital_expenditure = _period_value(
            us_gaap,
            [
                "PaymentsToAcquirePropertyPlantAndEquipment",
                "PaymentsForAdditionsToPropertyPlantAndEquipment",
            ],
            cash_flow_target,
        )

        cash_flow_net_income = _period_value(
            us_gaap,
            ["NetIncomeLoss"],
            cash_flow_target,
        )

    assets = _period_value(
        us_gaap,
        ["Assets"],
        target,
        instant=True,
    )

    equity = _period_value(
        us_gaap,
        [
            "StockholdersEquity",
            "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
        ],
        target,
        instant=True,
    )

    cash = _period_value(
        us_gaap,
        [
            "CashAndCashEquivalentsAtCarryingValue",
            "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents",
        ],
        target,
        instant=True,
    )

    short_term_debt = _period_value(
        us_gaap,
        [
            "ShortTermBorrowings",
            "ShortTermDebtCurrent",
        ],
        target,
        instant=True,
    )

    current_long_term_debt = _period_value(
        us_gaap,
        [
            "LongTermDebtCurrent",
            "LongTermDebtAndFinanceLeaseObligationsCurrent",
        ],
        target,
        instant=True,
    )

    noncurrent_debt = _period_value(
        us_gaap,
        [
            "LongTermDebtNoncurrent",
            "LongTermDebtAndFinanceLeaseObligationsNoncurrent",
        ],
        target,
        instant=True,
    )

    previous_revenue = _prior_period_value(
        us_gaap,
        revenue_concepts,
        target,
    )

    previous_net_income = _prior_period_value(
        us_gaap,
        ["NetIncomeLoss"],
        target,
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

    return {
        "symbol": symbol,
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

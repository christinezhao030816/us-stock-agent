import os

import httpx
from dotenv import load_dotenv


load_dotenv()

sec_user_agent = os.getenv("SEC_USER_AGENT")

if not sec_user_agent:
    raise RuntimeError("未读取到 SEC_USER_AGENT")

headers = {
    "User-Agent": sec_user_agent,
    "Accept-Encoding": "gzip, deflate",
    "Accept": "application/json",
}


def get_cik(symbol: str) -> str:
    url = "https://www.sec.gov/files/company_tickers.json"

    response = httpx.get(
        url,
        headers=headers,
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()

    companies = response.json()
    symbol = symbol.upper()

    for company in companies.values():
        if company["ticker"].upper() == symbol:
            return str(company["cik_str"]).zfill(10)

    raise ValueError(f"SEC 中没有找到股票代码：{symbol}")


def get_company_facts(cik: str) -> dict:
    url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"

    response = httpx.get(
        url,
        headers=headers,
        timeout=30,
        follow_redirects=True,
    )
    response.raise_for_status()

    return response.json()


def get_latest_usd_fact(
    company_facts: dict,
    concepts: list[str],
) -> dict | None:
    us_gaap = company_facts.get("facts", {}).get("us-gaap", {})

    for concept in concepts:
        fact = us_gaap.get(concept)

        if not fact:
            continue

        entries = fact.get("units", {}).get("USD", [])

        valid_entries = [
            entry
            for entry in entries
            if entry.get("form") in {"10-K", "10-Q"}
            and entry.get("end")
            and entry.get("filed")
        ]

        if not valid_entries:
            continue

        latest = max(
            valid_entries,
            key=lambda entry: (
                entry["end"],
                bool(entry.get("frame")),
                entry["filed"],
            ),
        )

        return {
            "concept": concept,
            "value": latest["val"],
            "period_end": latest["end"],
            "filed": latest["filed"],
            "form": latest["form"],
        }

    return None


symbol = "AAPL"
cik = get_cik(symbol)
company_facts = get_company_facts(cik)

metrics = {
    "营业收入": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
    ],
    "净利润": [
        "NetIncomeLoss",
    ],
    "总资产": [
        "Assets",
    ],
    "股东权益": [
        "StockholdersEquity",
        "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest",
    ],
}

print(f"股票：{symbol}")
print(f"公司：{company_facts.get('entityName')}")
print(f"SEC CIK：{cik}")

for metric_name, concepts in metrics.items():
    result = get_latest_usd_fact(company_facts, concepts)

    if result:
        value_billions = result["value"] / 1_000_000_000

        print(
            f"{metric_name}：{value_billions:.2f} 十亿美元"
            f"｜期间截止：{result['period_end']}"
            f"｜披露日期：{result['filed']}"
            f"｜表格：{result['form']}"
        )
    else:
        print(f"{metric_name}：未找到")

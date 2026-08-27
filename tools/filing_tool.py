import os
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()

SEC_DATA_URL = "https://data.sec.gov"
SEC_ARCHIVES_URL = "https://www.sec.gov/Archives/edgar/data"
SEC_TICKERS_URL = (
    "https://www.sec.gov/files/company_tickers.json"
)

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")

if not SEC_USER_AGENT:
    raise RuntimeError("未读取到 SEC_USER_AGENT")


ITEM_DESCRIPTIONS = {
    "1.01": "签订重大合同",
    "1.02": "终止重大合同",
    "1.03": "破产或接管",
    "2.01": "完成资产收购或处置",
    "2.02": "经营结果和财务状况",
    "2.03": "新增重大直接财务义务",
    "2.04": "触发加速或增加财务义务的事件",
    "2.05": "退出或处置活动相关成本",
    "2.06": "重大资产减值",
    "3.01": "退市或持续上市规则相关通知",
    "3.02": "未注册股权证券销售",
    "3.03": "证券持有人权利发生重大变化",
    "4.01": "注册会计师发生变化",
    "4.02": "历史财务报表不应继续依赖",
    "5.01": "公司控制权发生变化",
    "5.02": "董事或高级管理人员变动",
    "5.03": "公司章程或细则修订",
    "5.07": "股东投票事项",
    "7.01": "Regulation FD披露",
    "8.01": "其他重大事件",
    "9.01": "财务报表和附件",
}


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "application/json",
    }


def _get_json(url: str) -> dict:
    with httpx.Client(
        timeout=30,
        headers=_sec_headers(),
    ) as client:
        response = client.get(url)
        response.raise_for_status()
        return response.json()


def _get_company(symbol: str) -> dict:
    companies = _get_json(SEC_TICKERS_URL)
    symbol = symbol.upper().strip()

    for company in companies.values():
        if str(company.get("ticker", "")).upper() == symbol:
            return company

    raise ValueError(f"SEC中找不到股票代码：{symbol}")


def _value_at(
    values: list,
    index: int,
) -> Any:
    if index >= len(values):
        return None

    return values[index]


def _parse_items(value: str | None) -> list[str]:
    if not value:
        return []

    items = []

    for item in value.split(","):
        cleaned = item.strip()

        if cleaned:
            items.append(cleaned)

    return items


def get_recent_sec_filings(
    symbol: str,
    lookback_days: int = 90,
    limit: int = 10,
) -> dict[str, Any]:
    """获取目标公司最近的8-K或6-K重大事项申报。"""

    symbol = symbol.upper().strip()
    lookback_days = max(1, min(lookback_days, 365))
    limit = max(1, min(limit, 20))

    company = _get_company(symbol)

    cik_integer = int(company["cik_str"])
    cik_padded = str(cik_integer).zfill(10)

    submissions = _get_json(
        f"{SEC_DATA_URL}/submissions/CIK{cik_padded}.json"
    )

    recent = submissions.get(
        "filings",
        {},
    ).get(
        "recent",
        {},
    )

    now = datetime.now(timezone.utc)
    cutoff_date = (
        now - timedelta(days=lookback_days)
    ).date()

    forms = recent.get("form", [])
    filing_dates = recent.get("filingDate", [])
    accession_numbers = recent.get(
        "accessionNumber",
        [],
    )
    primary_documents = recent.get(
        "primaryDocument",
        [],
    )
    report_dates = recent.get("reportDate", [])
    acceptance_times = recent.get(
        "acceptanceDateTime",
        [],
    )
    items_values = recent.get("items", [])
    descriptions = recent.get(
        "primaryDocDescription",
        [],
    )

    filings = []

    accepted_forms = {
        "8-K",
        "8-K/A",
        "6-K",
        "6-K/A",
    }

    for index, form in enumerate(forms):
        if form not in accepted_forms:
            continue

        filing_date_text = _value_at(
            filing_dates,
            index,
        )

        if not filing_date_text:
            continue

        filing_date = datetime.fromisoformat(
            filing_date_text
        ).date()

        if filing_date < cutoff_date:
            continue

        accession_number = _value_at(
            accession_numbers,
            index,
        )

        primary_document = _value_at(
            primary_documents,
            index,
        )

        if not accession_number:
            continue

        accession_without_dashes = (
            accession_number.replace("-", "")
        )

        filing_index_url = (
            f"{SEC_ARCHIVES_URL}/"
            f"{cik_integer}/"
            f"{accession_without_dashes}/"
            f"{accession_number}-index.html"
        )

        primary_document_url = None

        if primary_document:
            primary_document_url = (
                f"{SEC_ARCHIVES_URL}/"
                f"{cik_integer}/"
                f"{accession_without_dashes}/"
                f"{primary_document}"
            )

        filing_items = _parse_items(
            _value_at(items_values, index)
        )

        item_descriptions = [
            {
                "item": item,
                "description": ITEM_DESCRIPTIONS.get(
                    item,
                    "工具未配置该项目的中文说明",
                ),
            }
            for item in filing_items
        ]

        filings.append(
            {
                "form": form,
                "filing_date": filing_date_text,
                "report_date": _value_at(
                    report_dates,
                    index,
                ),
                "acceptance_time": _value_at(
                    acceptance_times,
                    index,
                ),
                "accession_number": accession_number,
                "items": filing_items,
                "item_descriptions": item_descriptions,
                "primary_document": primary_document,
                "primary_document_description": _value_at(
                    descriptions,
                    index,
                ),
                "filing_index_url": filing_index_url,
                "primary_document_url": (
                    primary_document_url
                ),
                "source_quality": "一手监管申报",
            }
        )

    filings.sort(
        key=lambda filing: (
            filing.get("acceptance_time") or "",
            filing.get("filing_date") or "",
        ),
        reverse=True,
    )

    selected_filings = filings[:limit]

    return {
        "symbol": symbol,
        "company": submissions.get("name"),
        "cik": cik_padded,
        "analysis_time_utc": now.isoformat(),
        "lookback_days": lookback_days,
        "period_start": cutoff_date.isoformat(),
        "period_end": now.date().isoformat(),
        "filing_count": len(selected_filings),
        "filings": selected_filings,
        "included_forms": sorted(accepted_forms),
        "source": "SEC EDGAR Submissions API",
        "source_quality": "一手监管申报",
        "interpretation_note": (
            "申报项目说明仅概括8-K项目类别，"
            "具体事实、金额和影响必须打开SEC原文核实"
        ),
        "limitations": (
            "该工具只读取近期8-K、8-K/A、6-K和6-K/A元数据，"
            "不自动提取或总结申报文件正文"
        ),
    }

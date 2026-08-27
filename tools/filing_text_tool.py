import os
import re
from typing import Any
from urllib.parse import urljoin

import httpx
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from tools.filing_tool import get_recent_sec_filings


load_dotenv()

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT")

if not SEC_USER_AGENT:
    raise RuntimeError("未读取到 SEC_USER_AGENT")


def _sec_headers() -> dict[str, str]:
    return {
        "User-Agent": SEC_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
        "Accept": "text/html,application/xhtml+xml",
    }


def _clean_html(
    html: str,
    max_characters: int,
) -> dict[str, Any]:
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "ix:header",
            "ix:hidden",
        ]
    ):
        tag.decompose()

    for tag in soup.find_all(style=True):
        style = str(tag.get("style", "")).lower()

        if "display:none" in style.replace(" ", ""):
            tag.decompose()

    raw_text = soup.get_text("\n", strip=True)

    lines = []

    for line in raw_text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()

        if cleaned:
            lines.append(cleaned)

    text = "\n".join(lines)
    original_length = len(text)
    truncated = original_length > max_characters

    if truncated:
        text = text[:max_characters].rstrip()
        text += "\n[正文因长度限制被截断]"

    return {
        "text": text,
        "original_character_count": original_length,
        "returned_character_count": len(text),
        "truncated": truncated,
    }


def _fetch_document(
    client: httpx.Client,
    url: str | None,
    max_characters: int,
) -> dict[str, Any]:
    if not url:
        return {
            "url": None,
            "text": None,
            "error": "未提供文件链接",
        }

    try:
        response = client.get(url)
        response.raise_for_status()

        result = _clean_html(
            response.text,
            max_characters,
        )

        return {
            "url": url,
            **result,
            "error": None,
        }

    except Exception as error:
        return {
            "url": url,
            "text": None,
            "error": (
                f"{type(error).__name__}: {error}"
            ),
        }


def _find_exhibit_links(
    index_html: str,
    index_url: str,
    maximum_exhibits: int = 2,
) -> list[dict[str, str]]:
    soup = BeautifulSoup(index_html, "html.parser")
    exhibits = []
    seen_urls = set()

    description_keywords = {
        "earnings release",
        "press release",
        "news release",
        "financial results",
    }

    for row in soup.find_all("tr"):
        cells = row.find_all("td")

        if len(cells) < 4:
            continue

        description = cells[1].get_text(
            " ",
            strip=True,
        )

        document_cell = cells[2]
        document_type = cells[3].get_text(
            " ",
            strip=True,
        ).upper()

        link = document_cell.find("a", href=True)

        if link is None:
            continue

        description_lower = description.lower()

        is_exhibit_99 = document_type.startswith(
            "EX-99"
        )

        is_relevant_description = any(
            keyword in description_lower
            for keyword in description_keywords
        )

        if not (
            is_exhibit_99
            or is_relevant_description
        ):
            continue

        document_url = urljoin(
            index_url,
            link["href"],
        )

        if document_url in seen_urls:
            continue

        seen_urls.add(document_url)

        exhibits.append(
            {
                "document_type": document_type,
                "description": description,
                "url": document_url,
            }
        )

        if len(exhibits) >= maximum_exhibits:
            break

    return exhibits


def get_recent_sec_filing_texts(
    symbol: str,
    lookback_days: int = 90,
    filing_limit: int = 3,
    max_characters_per_document: int = 7000,
) -> dict[str, Any]:
    """读取近期SEC重大事项申报的主文件和EX-99附件正文。"""

    symbol = symbol.upper().strip()
    filing_limit = max(1, min(filing_limit, 5))

    max_characters_per_document = max(
        1000,
        min(max_characters_per_document, 12000),
    )

    metadata = get_recent_sec_filings(
        symbol=symbol,
        lookback_days=lookback_days,
        limit=filing_limit,
    )

    filings_with_text = []

    with httpx.Client(
        timeout=30,
        headers=_sec_headers(),
        follow_redirects=True,
    ) as client:
        for filing in metadata.get("filings", []):
            filing_index_url = filing.get(
                "filing_index_url"
            )

            primary_document_url = filing.get(
                "primary_document_url"
            )

            primary_document = _fetch_document(
                client=client,
                url=primary_document_url,
                max_characters=(
                    max_characters_per_document
                ),
            )

            exhibit_results = []
            index_error = None

            if filing_index_url:
                try:
                    index_response = client.get(
                        filing_index_url
                    )

                    index_response.raise_for_status()

                    exhibit_links = _find_exhibit_links(
                        index_html=index_response.text,
                        index_url=filing_index_url,
                    )

                    for exhibit in exhibit_links:
                        document = _fetch_document(
                            client=client,
                            url=exhibit["url"],
                            max_characters=(
                                max_characters_per_document
                            ),
                        )

                        exhibit_results.append(
                            {
                                **exhibit,
                                **document,
                            }
                        )

                except Exception as error:
                    index_error = (
                        f"{type(error).__name__}: "
                        f"{error}"
                    )

            filings_with_text.append(
                {
                    "form": filing.get("form"),
                    "filing_date": filing.get(
                        "filing_date"
                    ),
                    "report_date": filing.get(
                        "report_date"
                    ),
                    "acceptance_time": filing.get(
                        "acceptance_time"
                    ),
                    "accession_number": filing.get(
                        "accession_number"
                    ),
                    "items": filing.get("items", []),
                    "item_descriptions": filing.get(
                        "item_descriptions",
                        [],
                    ),
                    "filing_index_url": (
                        filing_index_url
                    ),
                    "primary_document": (
                        primary_document
                    ),
                    "exhibits": exhibit_results,
                    "index_error": index_error,
                    "source_quality": (
                        "一手监管申报"
                    ),
                }
            )

    return {
        "symbol": symbol,
        "company": metadata.get("company"),
        "analysis_time_utc": metadata.get(
            "analysis_time_utc"
        ),
        "lookback_days": metadata.get(
            "lookback_days"
        ),
        "filing_count": len(filings_with_text),
        "filings": filings_with_text,
        "source": "SEC EDGAR",
        "source_quality": "一手监管申报",
        "text_scope": (
            "读取8-K或6-K主文件，并最多读取两个"
            "EX-99或新闻稿类附件"
        ),
        "interpretation_note": (
            "返回内容是SEC文件正文的截取文本；"
            "涉及金额、日期和事件时仍需通过原文链接核查"
        ),
    }

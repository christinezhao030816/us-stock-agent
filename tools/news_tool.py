import os
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()

ALPACA_NEWS_URL = (
    "https://data.alpaca.markets/v1beta1/news"
)

ALPACA_API_KEY = (
    os.getenv("ALPACA_API_KEY")
    or os.getenv("APCA_API_KEY_ID")
)

ALPACA_SECRET_KEY = (
    os.getenv("ALPACA_SECRET_KEY")
    or os.getenv("APCA_API_SECRET_KEY")
)

if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
    raise RuntimeError(
        "未读取到 Alpaca API Key 或 Secret Key"
    )


HIGH_QUALITY_SOURCES = {
    "reuters",
    "bloomberg",
    "cnbc",
    "financial times",
    "the wall street journal",
    "wall street journal",
    "associated press",
    "ap news",
    "barron's",
    "barrons",
}

MEDIUM_QUALITY_SOURCES = {
    "benzinga",
    "marketwatch",
    "yahoo finance",
    "investor's business daily",
    "seeking alpha",
    "the motley fool",
    "business insider",
    "fortune",
    "forbes",
}


def _alpaca_headers() -> dict[str, str]:
    return {
        "APCA-API-KEY-ID": ALPACA_API_KEY,
        "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        "Accept": "application/json",
    }


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""

    text = value.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)

    return " ".join(text.split())


def _source_quality(source: str | None) -> dict:
    normalized = _normalize_text(source)

    if normalized in HIGH_QUALITY_SOURCES:
        return {
            "level": "high",
            "label": "较高",
            "score": 20,
        }

    if normalized in MEDIUM_QUALITY_SOURCES:
        return {
            "level": "medium",
            "label": "中等",
            "score": 12,
        }

    return {
        "level": "unrated",
        "label": "未评级",
        "score": 5,
    }


def _token_similarity(
    first: str,
    second: str,
) -> float:
    first_tokens = set(_normalize_text(first).split())
    second_tokens = set(_normalize_text(second).split())

    if not first_tokens or not second_tokens:
        return 0.0

    intersection = first_tokens & second_tokens
    union = first_tokens | second_tokens

    return len(intersection) / len(union)


def _is_duplicate(
    headline: str,
    url: str,
    accepted_articles: list[dict],
) -> bool:
    normalized_headline = _normalize_text(headline)

    for article in accepted_articles:
        if url and url == article.get("url"):
            return True

        existing_headline = article.get("headline", "")

        if (
            normalized_headline
            == _normalize_text(existing_headline)
        ):
            return True

        if (
            _token_similarity(
                headline,
                existing_headline,
            )
            >= 0.82
        ):
            return True

    return False


def _relevance_score(
    symbol: str,
    headline: str,
    summary: str,
    article_symbols: list[str],
    source_score: int,
) -> int:
    normalized_symbols = {
        str(item).upper()
        for item in article_symbols
    }

    score = 0

    if symbol in normalized_symbols:
        score += 60

    symbol_pattern = re.compile(
        rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])",
        re.IGNORECASE,
    )

    if symbol_pattern.search(headline):
        score += 15

    if symbol_pattern.search(summary):
        score += 5

    if 1 <= len(normalized_symbols) <= 3:
        score += 5

    score += source_score

    return min(score, 100)


def get_stock_news(
    symbol: str,
    limit: int = 5,
) -> dict[str, Any]:
    """获取并筛选指定股票最近7天的相关新闻。"""

    symbol = symbol.upper().strip()
    limit = max(1, min(limit, 10))

    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=7)

    fetch_limit = min(max(limit * 6, 30), 50)

    params = {
        "symbols": symbol,
        "start": start_time.isoformat(),
        "end": now.isoformat(),
        "sort": "desc",
        "limit": fetch_limit,
        "include_content": "false",
    }

    response = httpx.get(
        ALPACA_NEWS_URL,
        headers=_alpaca_headers(),
        params=params,
        timeout=30,
    )

    response.raise_for_status()
    data = response.json()

    accepted_articles = []
    rejected_count = 0
    duplicate_count = 0

    for item in data.get("news", []):
        headline = str(item.get("headline") or "").strip()
        summary = str(item.get("summary") or "").strip()
        url = str(item.get("url") or "").strip()
        source = str(item.get("source") or "未知来源").strip()


        article_symbols = [
            str(value).upper()
            for value in item.get("symbols", [])
        ]

        if symbol not in article_symbols:
            rejected_count += 1
            continue

        unique_symbols = set(article_symbols)

        symbol_pattern = re.compile(
            rf"(?<![A-Z0-9]){re.escape(symbol)}(?![A-Z0-9])",
            re.IGNORECASE,
        )

        directly_mentions_symbol = bool(
            symbol_pattern.search(
                f"{headline} {summary}"
            )
        )

        if (
            len(unique_symbols) > 1
            and not directly_mentions_symbol
        ):
            rejected_count += 1
            continue

        if not headline or not url:
            rejected_count += 1
            continue

        if _is_duplicate(
            headline,
            url,
            accepted_articles,
        ):
            duplicate_count += 1
            continue

        quality = _source_quality(source)

        relevance_score = _relevance_score(
            symbol=symbol,
            headline=headline,
            summary=summary,
            article_symbols=article_symbols,
            source_score=quality["score"],
        )

        accepted_articles.append(
            {
                "headline": headline,
                "summary": summary or None,
                "source": source,
                "source_quality": quality["label"],
                "source_quality_level": quality["level"],
                "author": item.get("author"),
                "created_at": item.get("created_at"),
                "updated_at": item.get("updated_at"),
                "url": url,
                "symbols": article_symbols,
                "relevance_score": relevance_score,
                "relevance_reason": (
                    f"Alpaca 将该新闻明确关联至 {symbol}"
                ),
            }
        )

    accepted_articles.sort(
        key=lambda article: (
            article["relevance_score"],
            article.get("created_at") or "",
        ),
        reverse=True,
    )

    selected_articles = accepted_articles[:limit]

    quality_counts = {
        "较高": 0,
        "中等": 0,
        "未评级": 0,
    }

    for article in selected_articles:
        label = article["source_quality"]
        quality_counts[label] += 1

    return {
        "symbol": symbol,
        "analysis_time_utc": now.isoformat(),
        "news_period_start_utc": start_time.isoformat(),
        "news_period_end_utc": now.isoformat(),
        "lookback_days": 7,
        "article_count": len(selected_articles),
        "articles": selected_articles,
        "source_quality_counts": quality_counts,
        "rejected_article_count": rejected_count,
        "duplicate_article_count": duplicate_count,
        "selection_rules": [
            "仅保留Alpaca明确关联至目标股票代码的新闻",
            "删除缺少标题或链接的记录",
            "过滤链接相同、标题相同或标题高度相似的重复新闻",
            "优先选择相关性评分较高且发布时间较新的新闻",
            "来源评级仅用于提示信息质量，不代表内容一定准确",
        ],
        "source": "Alpaca News API",
        "possible_delay_minutes": 15,
        "limitations": (
            "来源评级基于预设媒体名单；未评级不等于不可信。"
            "新闻摘要可能不包含报道全文，重要信息仍需打开原文核实。"
        ),
    }

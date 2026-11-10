from __future__ import annotations

from typing import Any


VALID_EVIDENCE_DEPTHS = {
    "fully_supported",
    "headline_only",
    "truncated_source",
    "unavailable",
}


def classify_article_evidence(
    article: dict[str, Any],
    *,
    company_match: bool,
) -> dict[str, Any]:
    title = str(article.get("headline") or "").strip()
    summary = str(article.get("summary") or "").strip()
    body = str(
        article.get("content")
        or article.get("article_body")
        or article.get("excerpt")
        or ""
    ).strip()
    url = str(article.get("url") or "").strip()
    published = str(article.get("created_at") or "").strip()
    metadata_complete = bool(title and url and published and company_match)
    if metadata_complete and (summary or body):
        depth = "fully_supported"
    elif metadata_complete:
        depth = "headline_only"
    else:
        depth = "unavailable"
    strategy_eligible = depth == "fully_supported" and bool(body)
    limitation_reason = (
        None
        if strategy_eligible
        else "full_text_unavailable"
        if depth == "fully_supported"
        else depth
    )
    return {
        "evidence_depth": depth,
        "full_text_available": bool(body),
        "claim_scope_limited": depth != "fully_supported",
        "limitation_disclosed": depth != "fully_supported",
        "strategy_eligible": strategy_eligible,
        "strategy_limitation_reason": limitation_reason,
        "source_content": {
            "title": title or None,
            "summary": summary or None,
            "article_body_or_excerpt": body or None,
            "publisher": str(article.get("source") or "").strip() or None,
            "published_date": published or None,
            "url": url or None,
        },
    }


def classify_filing_evidence(filing: dict[str, Any]) -> dict[str, Any]:
    primary = filing.get("primary_document")
    primary = primary if isinstance(primary, dict) else {}
    exhibits = [item for item in filing.get("exhibits") or [] if isinstance(item, dict)]
    documents = [primary, *exhibits]
    readable = [
        item
        for item in documents
        if str(item.get("text") or "").strip() and not item.get("error")
    ]
    full_text_available = bool(readable) and not any(item.get("truncated") for item in readable)
    if not readable:
        depth = "unavailable"
    elif any(item.get("truncated") for item in readable):
        depth = "truncated_source"
    else:
        depth = "fully_supported"
    strategy_eligible = depth == "fully_supported" and full_text_available
    return {
        "evidence_depth": depth,
        "full_text_available": full_text_available,
        "claim_scope_limited": depth != "fully_supported",
        "limitation_disclosed": depth != "fully_supported",
        "strategy_eligible": strategy_eligible,
        "strategy_limitation_reason": (
            None if strategy_eligible else depth
        ),
        "source_content": {
            "form": filing.get("form"),
            "filing_date": filing.get("filing_date"),
            "accession_number": filing.get("accession_number"),
            "primary_document_url": primary.get("url"),
            "primary_document_excerpt": str(primary.get("text") or "") or None,
            "primary_document_truncated": bool(primary.get("truncated")),
            "exhibits": [
                {
                    "document_type": item.get("document_type"),
                    "url": item.get("url"),
                    "excerpt": str(item.get("text") or "") or None,
                    "truncated": bool(item.get("truncated")),
                    "error": item.get("error"),
                }
                for item in exhibits
            ],
        },
    }


def scoped_news_claim_text(
    title: str,
    description: str,
    evidence_depth: str,
) -> str:
    if evidence_depth == "fully_supported":
        return f"{title}: {description}" if description else title
    if evidence_depth == "headline_only":
        return f"Headline-only evidence; full article text was not available: {title}"
    if evidence_depth == "truncated_source":
        return f"The filing attachment was incomplete. Only the retrieved excerpt was used: {title}"
    return f"Unverified item; full text unavailable: {title or 'source metadata only'}"

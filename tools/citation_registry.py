import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import quote

from tools.financial_freshness import (
    build_filing_freshness,
    evaluate_fact_freshness,
)


VALID_CITATION_STATUSES = {"supported", "partial", "unsupported"}
_CLAIM_MARKER = re.compile(r"\[([TFN]\d+)\]")
_URL_PATTERN = re.compile(r"https?://[^\s<>\])]+")
_SOURCE_HEADING = re.compile(
    r"^#{1,3}\s+(?:\d+\.\s*)?(?:来源与证据|Sources and Evidence)\s*$",
    re.IGNORECASE,
)


def _claim(
    *,
    claim_id: str,
    agent: str,
    claim_type: str,
    metric_name: str,
    claim_text: str,
    value: Any,
    unit: str | None,
    period_or_as_of_date: str | None,
    source_name: str | None,
    source_type: str | None,
    source_url: str | None,
    source_date: str | None,
    source_anchor: str | None,
    evidence_path: str | None,
    citation_status: str,
) -> dict[str, Any]:
    if citation_status not in VALID_CITATION_STATUSES:
        raise ValueError(f"无效引用状态：{citation_status}")

    return {
        "claim_id": claim_id,
        "agent": agent,
        "claim_type": claim_type,
        "metric_name": metric_name,
        "claim_text": claim_text,
        "value": value,
        "unit": unit,
        "period_or_as_of_date": period_or_as_of_date,
        "source_name": source_name,
        "source_type": source_type,
        "source_url": source_url,
        "source_date": source_date,
        "source_anchor": source_anchor,
        "evidence_path": evidence_path,
        "citation_status": citation_status,
    }


def _status_for(
    *,
    value: Any,
    source_url: str | None,
    source_date: str | None,
    source_anchor: str | None,
    evidence_path: str | None = None,
    require_evidence_path: bool = False,
) -> str:
    if value is None:
        return "unsupported"
    required = [source_url, source_date, source_anchor]
    if require_evidence_path:
        required.append(evidence_path)
    return "supported" if all(required) else "partial"


def build_technical_citations(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = metrics.get("technical_evidence")
    if not isinstance(evidence, dict):
        evidence = {}
    indicators = evidence.get("indicator_evidence")
    if not isinstance(indicators, dict):
        indicators = {}

    canonical = str(
        evidence.get("canonical_symbol") or metrics.get("symbol") or ""
    )
    vendor = str(
        evidence.get("vendor_symbol")
        or metrics.get("source_symbol")
        or canonical
    )
    source_url = str(evidence.get("source_url") or "") or (
        "https://finance.yahoo.com/quote/"
        f"{quote(vendor, safe='')}/history/"
        if vendor
        else None
    )
    source_date = str(evidence.get("retrieved_at_utc") or "") or None
    as_of = str(evidence.get("last_trading_date") or "") or None
    evidence_path = str(evidence.get("evidence_path") or "") or None
    source_name = str(evidence.get("source") or "Yahoo Finance via yfinance")

    definitions = [
        ("current_price", "current_price_usd", "USD"),
        ("previous_close", "previous_close_usd", "USD"),
        ("daily_change", "daily_change_percent", "percent"),
        ("return_5d", "return_5d_percent", "percent"),
        ("return_20d", "return_20d_percent", "percent"),
        ("return_60d", "return_60d_percent", "percent"),
        ("sma20", "sma20_usd", "USD"),
        ("sma50", "sma50_usd", "USD"),
        ("sma200", "sma200_usd", "USD"),
        ("rsi14", "rsi14", "index"),
        ("latest_volume", "latest_volume", "shares"),
        ("average_volume_20d", "average_volume_20d", "shares"),
        ("average_volume_60d", "average_volume_60d", "shares"),
        ("volume_ratio_to_20d", "volume_ratio_to_20d", "ratio"),
        ("support_20d", "support_20d_usd", "USD"),
        ("resistance_20d", "resistance_20d_usd", "USD"),
        ("support_60d", "support_60d_usd", "USD"),
        ("resistance_60d", "resistance_60d_usd", "USD"),
        ("distance_from_support_20d", "distance_from_support_20d_percent", "percent"),
        ("distance_from_resistance_20d", "distance_from_resistance_20d_percent", "percent"),
        ("high_52_week", "high_52_week_usd", "USD"),
        ("low_52_week", "low_52_week_usd", "USD"),
        ("distance_from_52_week_high", "distance_from_52_week_high_percent", "percent"),
        ("distance_from_52_week_low", "distance_from_52_week_low_percent", "percent"),
    ]
    claims = []
    for index, (metric_name, key, unit) in enumerate(definitions, start=1):
        value = metrics.get(key)
        anchor = f"technical_evidence.indicator_evidence.{key}"
        status = _status_for(
            value=value,
            source_url=source_url,
            source_date=source_date,
            source_anchor=anchor if key in indicators else None,
            evidence_path=evidence_path,
            require_evidence_path=True,
        )
        claims.append(
            _claim(
                claim_id=f"T{index}",
                agent="technical",
                claim_type="technical_metric",
                metric_name=metric_name,
                claim_text=f"{canonical} {metric_name} = {value} {unit}",
                value=value,
                unit=unit,
                period_or_as_of_date=as_of,
                source_name=source_name,
                source_type="market_data",
                source_url=source_url,
                source_date=source_date,
                source_anchor=anchor if key in indicators else None,
                evidence_path=evidence_path,
                citation_status=status,
            )
        )
    return claims


def _record_anchor(record: Any) -> str | None:
    if not isinstance(record, dict):
        return None
    concepts = record.get("concepts")
    if isinstance(concepts, list):
        anchors = []
        for item in concepts:
            if not item.get("concept"):
                continue
            anchor = (
                f"facts.us-gaap.{item['concept']}.units."
                f"{item.get('unit') or 'USD'}"
                f" period={item.get('start') or 'instant'}..{item.get('end')}"
            )
            if item.get("accession"):
                anchor += f" accession={item['accession']}"
            anchors.append(anchor)
        return "; ".join(anchors) or None
    concept = record.get("concept")
    unit = record.get("unit")
    if not concept:
        return None
    anchor = f"facts.us-gaap.{concept}.units.{unit or 'USD'}"
    anchor += f" period={record.get('start') or 'instant'}..{record.get('end')}"
    if record.get("accession"):
        anchor += f" accession={record['accession']}"
    comparison = record.get("comparison")
    if isinstance(comparison, dict) and comparison.get("concept"):
        comparison_anchor = (
            f"facts.us-gaap.{comparison['concept']}.units."
            f"{comparison.get('unit') or 'USD'}"
        )
        anchor += (
            f"; compare {comparison_anchor}"
            f" period_end={comparison.get('end')}"
        )
    return anchor


def build_fundamental_citations(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    records = metrics.get("source_records")
    records = records if isinstance(records, dict) else {}
    source_url = str(metrics.get("source_url") or "") or None
    company = str(metrics.get("company") or metrics.get("symbol") or "")
    filing_date = str(metrics.get("filing_date") or "") or None
    amounts = metrics.get("financial_amounts")
    amounts = amounts if isinstance(amounts, dict) else {}

    definitions = [
        ("revenue", amounts.get("revenue"), "USD billion", records.get("revenue")),
        ("net_income", amounts.get("net_income"), "USD billion", records.get("net_income")),
        ("EPS", metrics.get("eps_diluted_usd_per_share"), "USD/share", records.get("eps")),
        ("operating_cash_flow", amounts.get("operating_cash_flow"), "USD billion", records.get("operating_cash_flow")),
        ("capital_expenditure", amounts.get("capital_expenditure"), "USD billion", records.get("capital_expenditure")),
        ("free_cash_flow", amounts.get("free_cash_flow"), "USD billion", records.get("free_cash_flow")),
        ("cash", amounts.get("cash"), "USD billion", records.get("cash")),
        ("total_assets", amounts.get("assets"), "USD billion", records.get("assets")),
        ("total_debt", amounts.get("total_debt"), "USD billion", records.get("total_debt")),
        ("equity", amounts.get("equity"), "USD billion", records.get("equity")),
        ("reporting_period", metrics.get("income_period_end"), "date", records.get("revenue")),
        ("filing_date", filing_date, "date", records.get("revenue")),
        ("revenue_yoy", metrics.get("revenue_yoy_percent"), "percent", records.get("revenue_yoy")),
        ("net_income_yoy", metrics.get("net_income_yoy_percent"), "percent", records.get("net_income_yoy")),
        ("net_margin", metrics.get("net_margin"), "ratio", records.get("net_margin")),
        ("equity_ratio", metrics.get("equity_ratio"), "ratio", records.get("equity_ratio")),
        ("operating_cash_flow_to_net_income", metrics.get("operating_cash_flow_to_net_income"), "ratio", records.get("operating_cash_flow_to_net_income")),
    ]
    claims = []
    freshness = metrics.get("fundamental_freshness")
    freshness_facts = (
        freshness.get("facts")
        if isinstance(freshness, dict)
        else []
    ) or []
    freshness_by_metric = {
        str(item.get("metric_name")): item
        for item in freshness_facts
        if isinstance(item, dict)
    }
    for index, (metric_name, raw_value, unit, record) in enumerate(definitions, start=1):
        if isinstance(raw_value, dict):
            value = raw_value.get("value")
            period = raw_value.get("period", {}).get("end")
            claim_value = raw_value.get("display_zh") or value
        else:
            value = raw_value
            period = (
                str(record.get("end") or "") if isinstance(record, dict) else ""
            ) or metrics.get("income_period_end")
            claim_value = raw_value
        record_date = (
            str(record.get("filed") or "") if isinstance(record, dict) else ""
        ) or filing_date
        anchor = _record_anchor(record)
        status = _status_for(
            value=value,
            source_url=source_url,
            source_date=record_date,
            source_anchor=anchor,
        )
        claim = _claim(
                claim_id=f"F{index}",
                agent="fundamental",
                claim_type="fundamental_metric",
                metric_name=metric_name,
                claim_text=f"{company} {metric_name} = {claim_value}",
                value=value,
                unit=unit,
                period_or_as_of_date=str(period or "") or None,
                source_name="SEC Company Facts API",
                source_type="sec_company_facts",
                source_url=source_url,
                source_date=record_date,
                source_anchor=anchor,
                evidence_path=source_url,
                citation_status=status,
            )
        claim["company_or_ticker"] = str(
            metrics.get("symbol") or company
        ) or None
        claim["publisher"] = claim["source_name"]
        fact_freshness = freshness_by_metric.get(metric_name)
        if fact_freshness is None and metric_name in {
            "reporting_period",
            "filing_date",
        }:
            fact_freshness = freshness_by_metric.get("revenue")
        claim["freshness"] = fact_freshness
        claim["freshness_status"] = (
            fact_freshness.get("freshness_status")
            if isinstance(fact_freshness, dict)
            else "unknown"
        )
        claims.append(claim)
    latest_filing = metrics.get("latest_applicable_filing")
    if isinstance(latest_filing, dict):
        latest_url = str(
            metrics.get("latest_filing_source_url") or ""
        ).strip() or None
        accession = str(
            latest_filing.get("accession_number") or ""
        ).strip()
        latest_period = str(
            latest_filing.get("period_end") or ""
        ).strip() or None
        latest_date = str(
            latest_filing.get("filing_date") or ""
        ).strip() or None
        latest_form = str(latest_filing.get("form") or "").strip()
        latest_claim = _claim(
            claim_id=f"F{len(claims) + 1}",
            agent="fundamental",
            claim_type="latest_periodic_filing",
            metric_name="latest_filing_period",
            claim_text=(
                f"{company} latest applicable filing is {latest_form} "
                f"for {latest_period}, filed {latest_date}"
            ),
            value={
                "form": latest_form,
                "period_end": latest_period,
                "filing_date": latest_date,
                "accession_number": accession or None,
            },
            unit="filing_metadata",
            period_or_as_of_date=latest_period,
            source_name="SEC Submissions API",
            source_type="sec_submissions",
            source_url=latest_url,
            source_date=latest_date,
            source_anchor=(
                f"filings.recent accession:{accession}"
                if accession else None
            ),
            evidence_path=latest_url,
            citation_status=_status_for(
                value=latest_period,
                source_url=latest_url,
                source_date=latest_date,
                source_anchor=(
                    f"filings.recent accession:{accession}"
                    if accession else None
                ),
            ),
        )
        latest_claim["company_or_ticker"] = str(
            metrics.get("symbol") or company
        ) or None
        latest_claim["publisher"] = "SEC EDGAR"
        latest_fact_freshness = evaluate_fact_freshness(
            metric_name="latest_filing_period",
            value=latest_period,
            unit="filing_metadata",
            form=latest_form,
            fiscal_year=None,
            fiscal_period=None,
            period_start=None,
            period_end=latest_period,
            filed_date=latest_date,
            accession=accession or None,
            analysis_as_of=(
                freshness.get("analysis_as_of")
                if isinstance(freshness, dict)
                else None
            ),
        )
        latest_claim["freshness"] = latest_fact_freshness
        latest_claim["freshness_status"] = latest_fact_freshness[
            "freshness_status"
        ]
        claims.append(latest_claim)
    return claims


def build_fundamental_filing_citations(
    filing_data: dict[str, Any],
) -> list[dict[str, Any]]:
    freshness = build_filing_freshness(filing_data)
    freshness_facts = freshness.get("facts") or []
    claims = []
    for index, filing in enumerate(filing_data.get("filings") or [], start=1):
        primary = filing.get("primary_document")
        primary = primary if isinstance(primary, dict) else {}
        url = str(
            filing.get("filing_index_url") or primary.get("url") or ""
        ).strip() or None
        accession = str(filing.get("accession_number") or "").strip()
        form = str(filing.get("form") or "").strip()
        filing_date = str(filing.get("filing_date") or "").strip() or None
        text = " ".join(str(primary.get("text") or "").split())
        description = text[:300]
        supported = bool(
            url and accession and filing_date and description
            and not primary.get("error")
        )
        fact_freshness = (
            freshness_facts[index - 1]
            if index <= len(freshness_facts)
            else None
        )
        claim = _claim(
            claim_id=f"F{index}",
            agent="fundamental",
            claim_type="foreign_filing_evidence",
            metric_name=f"foreign_filing_{index}",
            claim_text=(
                f"SEC {form} filing {accession}: {description}"
                if description else f"SEC {form} filing {accession}"
            ),
            value={
                "form": form,
                "description": description or None,
                "accession_number": accession,
            },
            unit="filing_text",
            period_or_as_of_date=str(
                filing.get("report_date") or filing_date or ""
            ) or None,
            source_name="SEC EDGAR",
            source_type="sec_filing",
            source_url=url,
            source_date=filing_date,
            source_anchor=(
                f"accession:{accession}; primary_document.text"
                if accession else None
            ),
            evidence_path=url,
            citation_status="supported" if supported else "partial",
        )
        claim["company_or_ticker"] = filing_data.get("symbol")
        claim["publisher"] = "SEC EDGAR"
        claim["freshness"] = fact_freshness
        claim["freshness_status"] = (
            fact_freshness.get("freshness_status")
            if isinstance(fact_freshness, dict)
            else "unknown"
        )
        claims.append(claim)
    return claims


def build_news_citations(payloads: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    claims: list[dict[str, Any]] = []
    news_values = payloads.get("get_stock_news", [])
    news = news_values[-1] if news_values else {}
    canonical = str(news.get("symbol") or "")
    articles = news.get("articles")
    articles = articles if isinstance(articles, list) else []

    for article in articles:
        claim_id = f"N{len(claims) + 1}"
        title = str(article.get("headline") or "").strip()
        description = str(article.get("summary") or "").strip()
        url = str(article.get("url") or "").strip() or None
        date = str(article.get("created_at") or "").strip() or None
        symbols = article.get("symbols") or []
        company_match = canonical in symbols if canonical else bool(symbols)
        status = (
            "supported"
            if title and description and url and date and company_match
            else "partial" if title or url else "unsupported"
        )
        claim = _claim(
                claim_id=claim_id,
                agent="news",
                claim_type="news_event",
                metric_name="event",
                claim_text=f"{title}: {description}" if description else title,
                value={
                    "title": title,
                    "description": description or None,
                    "company_match": company_match,
                },
                unit=None,
                period_or_as_of_date=date,
                source_name=str(article.get("source") or "") or None,
                source_type="news_article",
                source_url=url,
                source_date=date,
                source_anchor=title or None,
                evidence_path=url,
                citation_status=status,
            )
        claim["company_or_ticker"] = canonical or None
        claim["publisher"] = claim["source_name"]
        claims.append(claim)

    filing_values = payloads.get("get_recent_sec_filing_texts", [])
    filing_data = filing_values[-1] if filing_values else {}
    filings = filing_data.get("filings")
    filings = filings if isinstance(filings, list) else []
    for filing in filings:
        claim_id = f"N{len(claims) + 1}"
        accession = str(filing.get("accession_number") or "").strip()
        form = str(filing.get("form") or "").strip()
        date = str(filing.get("filing_date") or "").strip() or None
        primary = filing.get("primary_document")
        primary = primary if isinstance(primary, dict) else {}
        url = str(
            filing.get("filing_index_url")
            or primary.get("url")
            or filing.get("primary_document_url")
            or ""
        ).strip() or None
        text_present = bool(str(primary.get("text") or "").strip())
        item_descriptions = filing.get("item_descriptions") or []
        description = "; ".join(
            str(item).strip() for item in item_descriptions if str(item).strip()
        )
        if not description and text_present:
            description = " ".join(
                str(primary.get("text") or "").split()
            )[:300]
        status = (
            "supported"
            if accession and form and date and url and text_present and not primary.get("error")
            else "partial" if accession or url else "unsupported"
        )
        claim = _claim(
                claim_id=claim_id,
                agent="news",
                claim_type="sec_filing_event",
                metric_name="filing_event",
                claim_text=(
                    f"SEC {form} filing {accession}: {description}"
                    if description else f"SEC {form} filing {accession}"
                ),
                value={
                    "title": f"SEC {form} filing",
                    "description": description or None,
                    "form": form,
                    "accession_number": accession,
                    "company_match": True,
                },
                unit=None,
                period_or_as_of_date=str(filing.get("report_date") or date or "") or None,
                source_name="SEC EDGAR",
                source_type="sec_filing",
                source_url=url,
                source_date=date,
                source_anchor=(
                    f"accession:{accession}; primary_document.text"
                    if accession else None
                ),
                evidence_path=url,
                citation_status=status,
            )
        claim["company_or_ticker"] = str(
            filing_data.get("symbol") or canonical or ""
        ) or None
        claim["publisher"] = "SEC EDGAR"
        claims.append(claim)

    if not claims:
        claims.append(
            _claim(
                claim_id="N1",
                agent="news",
                claim_type="news_event",
                metric_name="event",
                claim_text=f"{canonical or 'ticker'} 无可验证新闻或申报事件",
                value=None,
                unit=None,
                period_or_as_of_date=str(news.get("news_period_end_utc") or "") or None,
                source_name=str(news.get("source") or "Alpaca News API"),
                source_type="news_api",
                source_url=None,
                source_date=str(news.get("analysis_time_utc") or "") or None,
                source_anchor=None,
                evidence_path=None,
                citation_status="unsupported",
            )
        )
    return claims


def build_agent_citations(
    agent_role: str,
    payloads: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if agent_role == "technical":
        values = payloads.get("get_technical_metrics", [])
        return build_technical_citations(values[-1]) if values else []
    if agent_role == "fundamental":
        values = payloads.get("get_fundamental_metrics", [])
        if values:
            return build_fundamental_citations(values[-1])
        filings = payloads.get("get_recent_sec_filing_texts", [])
        return (
            build_fundamental_filing_citations(filings[-1])
            if filings
            else []
        )
    if agent_role == "news":
        return build_news_citations(payloads)
    raise ValueError(f"未知 Agent：{agent_role}")


def _valid_supported(claim: dict[str, Any]) -> bool:
    required = (
        "claim_id",
        "agent",
        "claim_type",
        "metric_name",
        "claim_text",
        "value",
        "period_or_as_of_date",
        "source_name",
        "source_type",
        "source_url",
        "source_date",
        "source_anchor",
    )
    return (
        claim.get("citation_status") == "supported"
        and all(
            claim.get(key) is not None and claim.get(key) != ""
            for key in required
        )
        and str(claim.get("source_url", "")).startswith(("http://", "https://"))
    )


def build_source_registry(citations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for claim in citations:
        url = str(claim.get("source_url") or "")
        key = url or f"unsupported:{claim.get('claim_id')}"
        entry = registry.setdefault(
            key,
            {
                "source_id": f"SRC{len(registry) + 1}",
                "source_name": claim.get("source_name"),
                "source_type": claim.get("source_type"),
                "source_url": claim.get("source_url"),
                "source_date": claim.get("source_date"),
                "source_anchors": [],
                "claim_ids": [],
            },
        )
        entry["claim_ids"].append(claim.get("claim_id"))
        anchor = claim.get("source_anchor")
        if anchor and anchor not in entry["source_anchors"]:
            entry["source_anchors"].append(anchor)
    return list(registry.values())


def citation_summary(
    citations: list[dict[str, Any]],
    report: str = "",
    source_registry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    ids = [str(claim.get("claim_id") or "") for claim in citations]
    duplicate_ids = sorted({item for item in ids if ids.count(item) > 1})
    valid_ids = {claim["claim_id"] for claim in citations if _valid_supported(claim)}
    report_ids = set(_CLAIM_MARKER.findall(report))
    unresolved = sorted(report_ids - set(ids))
    uncited_supported = sorted(valid_ids - report_ids) if report else []
    registry = source_registry if source_registry is not None else build_source_registry(citations)
    used_ids = set(ids)
    unused_sources = [
        entry.get("source_id")
        for entry in registry
        if not set(entry.get("claim_ids") or []) & used_ids
    ]
    required = len(citations)
    cited = len(valid_ids & report_ids) if report else len(valid_ids)
    return {
        "citation_required_claims": required,
        "cited_claims": cited,
        "citation_coverage_pct": round(cited / required * 100, 2) if required else 0.0,
        "citation_integrity": not duplicate_ids and not unresolved and not uncited_supported and not unused_sources,
        "duplicate_claim_ids": duplicate_ids,
        "unresolved_report_citations": unresolved,
        "uncited_supported_claims": uncited_supported,
        "unused_source_ids": unused_sources,
    }


def finalize_cited_report(
    report: str,
    citations: list[dict[str, Any]],
    *,
    language: str = "zh",
) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
    if not citations:
        registry: list[dict[str, Any]] = []
        return str(report or ""), registry, citation_summary([], str(report or ""), registry)

    report_lines = str(report or "").splitlines()
    for index, line in enumerate(report_lines):
        if _SOURCE_HEADING.match(line.strip()):
            report_lines = report_lines[:index]
            break
    report = "\n".join(report_lines).rstrip()

    ids = {str(claim.get("claim_id")) for claim in citations}
    allowed_urls = {str(claim.get("source_url")) for claim in citations if claim.get("source_url")}

    cleaned = _CLAIM_MARKER.sub(
        lambda match: match.group(0) if match.group(1) in ids else "[未解析引用已移除]",
        str(report or ""),
    )
    cleaned = _URL_PATTERN.sub(
        lambda match: match.group(0) if match.group(0).rstrip(".,") in allowed_urls else "[未验证URL已移除]",
        cleaned,
    ).rstrip()

    registry = build_source_registry(citations)
    title = "Sources and Evidence" if language == "en" else "来源与证据"
    fact_title = "Claim-level evidence" if language == "en" else "逐项事实证据"
    source_title = "Source registry" if language == "en" else "来源注册表"
    strategy_title = "Strategy evidence basis" if language == "en" else "综合策略事实依据"
    lines = [cleaned, "", f"## {title}", "", f"### {fact_title}", ""]
    for claim in citations:
        marker = f"[{claim['claim_id']}]"
        status = claim.get("citation_status")
        lines.append(f"- {marker} {claim.get('claim_text')}（{status}）")
    lines.extend(["", f"### {strategy_title}", ""])
    for agent, label in (
        ("technical", "Technical/技术"),
        ("fundamental", "Fundamental/基本面"),
        ("news", "News & catalysts/消息与催化剂"),
    ):
        markers = " ".join(
            f"[{claim['claim_id']}]"
            for claim in citations
            if claim.get("agent") == agent and claim.get("citation_status") == "supported"
        )
        lines.append(f"- {label}: {markers or '无完整支持项'}")
    lines.extend(["", f"### {source_title}", ""])
    for entry in registry:
        claim_ids = ", ".join(f"[{item}]" for item in entry.get("claim_ids") or [])
        url = entry.get("source_url") or "URL unavailable / URL缺失"
        lines.append(
            f"- {entry['source_id']} — {entry.get('source_name') or '未知来源'}; "
            f"{entry.get('source_date') or '日期缺失'}; {url}; claims: {claim_ids}"
        )
    final_report = "\n".join(lines).strip() + "\n"
    summary = citation_summary(citations, final_report, registry)
    return final_report, registry, summary


def save_source_registry(
    path: Path,
    *,
    citations: list[dict[str, Any]],
    source_registry: list[dict[str, Any]],
    summary: dict[str, Any],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "citations": citations,
                "source_registry": source_registry,
                "citation_summary": summary,
            },
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    return path

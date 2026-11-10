import ast
import json
from collections.abc import Callable
from typing import Any

from tools.citation_registry import build_agent_citations
from tools.financial_freshness import build_filing_freshness


AGENT_ROLES = ("technical", "fundamental", "news", "strategy")
VALID_STATUSES = {"success", "partial", "failed"}

_AGENT_TOOL_NAMES = {
    "technical_analysis": "technical",
    "fundamental_analysis": "fundamental",
    "news_analysis": "news",
}


def _tool_name(item: Any, call_names: dict[str, str]) -> str | None:
    resolved_name = getattr(item, "_resolved_tool_name", None)
    if resolved_name:
        return str(resolved_name)

    tool_name = getattr(item, "tool_name", None)
    if tool_name:
        return str(tool_name)

    raw_item = getattr(item, "raw_item", None)
    if isinstance(raw_item, dict):
        raw_name = raw_item.get("name")
        if raw_name:
            return str(raw_name)

    call_id = getattr(item, "call_id", None)
    if call_id:
        called_name = call_names.get(str(call_id))
        if called_name:
            return called_name

    # ``tool_origin.agent_tool_name`` identifies the enclosing agent-as-tool.
    # It is useful for top-level specialist outputs, but for nested function
    # tools the call item's resolved name is the authoritative identity.
    origin = getattr(item, "tool_origin", None)
    origin_name = getattr(origin, "agent_tool_name", None)
    if origin_name:
        return str(origin_name)

    return None


def _coerce_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value

    if not isinstance(value, str):
        return None

    stripped = value.strip()
    if not stripped:
        return None

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            return None

    return parsed if isinstance(parsed, dict) else None


def collect_tool_results(
    run_result: Any,
) -> tuple[dict[str, list[dict[str, Any]]], list[str]]:
    payloads: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    call_names: dict[str, str] = {}

    for item in getattr(run_result, "new_items", []):
        item_type = getattr(item, "type", "")

        if item_type == "tool_call_item":
            name = _tool_name(item, call_names)
            call_id = getattr(item, "call_id", None)
            if name and call_id:
                call_names[str(call_id)] = name
            continue

        if item_type != "tool_call_output_item":
            continue

        name = _tool_name(item, call_names) or "unknown_tool"
        output = getattr(item, "output", None)
        payload = _coerce_mapping(output)

        if payload is not None:
            payloads.setdefault(name, []).append(payload)
        else:
            message = str(output or "工具未返回结构化结果").strip()
            errors.append(f"{name}: {message[:300]}")

    return payloads, errors


def _source_symbols(
    payloads: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    symbols: dict[str, str] = {}

    for values in payloads.values():
        for payload in values:
            mapping = payload.get("symbol_mapping")
            if not isinstance(mapping, dict):
                continue
            for source, symbol in mapping.items():
                if symbol:
                    symbols[str(source)] = str(symbol)

    return symbols


def _status(
    status: str,
    *,
    core_data_present: bool,
    missing_items: list[str],
    errors: list[str],
    data_sources: list[str],
    source_symbols: dict[str, str],
) -> dict[str, Any]:
    return {
        "status": status,
        "core_data_present": core_data_present,
        "missing_items": missing_items,
        "errors": errors,
        "data_sources": data_sources,
        "source_symbols": source_symbols,
    }


def evaluate_technical_status(
    payloads: dict[str, list[dict[str, Any]]],
    errors: list[str],
) -> dict[str, Any]:
    results = payloads.get("get_technical_metrics", [])
    metrics = results[-1] if results else None
    required = [
        "data_end",
        "trading_days",
        "current_price_usd",
        "return_20d_percent",
        "sma20_usd",
        "sma50_usd",
        "sma200_usd",
        "rsi14",
        "latest_volume",
        "support_20d_usd",
        "resistance_20d_usd",
    ]

    if metrics is None:
        return _status(
            "failed",
            core_data_present=False,
            missing_items=required,
            errors=errors or ["技术工具没有返回结构化数据"],
            data_sources=[],
            source_symbols={},
        )

    missing = [key for key in required if metrics.get(key) is None]
    technical_evidence = metrics.get("technical_evidence")

    if not isinstance(technical_evidence, dict):
        missing.append("technical_evidence")
    elif not technical_evidence.get("reproducible"):
        evidence_missing = technical_evidence.get(
            "reproducibility_missing"
        ) or ["unknown_reproducibility_requirement"]
        missing.extend(
            f"technical_evidence:{item}"
            for item in evidence_missing
        )

    core_present = all(
        metrics.get(key) is not None
        for key in ("data_end", "trading_days", "current_price_usd")
    )

    if not core_present:
        status = "failed"
    elif missing or errors:
        status = "partial"
    else:
        status = "success"

    return _status(
        status,
        core_data_present=core_present,
        missing_items=missing,
        errors=errors,
        data_sources=[str(metrics.get("price_source") or "Yahoo Finance")],
        source_symbols=_source_symbols(payloads),
    )


def evaluate_fundamental_status(
    payloads: dict[str, list[dict[str, Any]]],
    errors: list[str],
) -> dict[str, Any]:
    results = payloads.get("get_fundamental_metrics", [])
    metrics = results[-1] if results else None
    filings = payloads.get("get_recent_sec_filing_texts", [])
    filing_data = filings[-1] if filings else None
    required = [
        "company",
        "form",
        "income_period_end",
        "filing_date",
        "revenue",
        "net_income",
        "operating_cash_flow",
        "capital_expenditure",
        "free_cash_flow",
        "cash",
        "total_debt",
        "assets",
        "equity",
    ]

    if metrics is None:
        filing_count = (
            int(filing_data.get("filing_count") or 0)
            if filing_data
            else 0
        )
        if filing_count > 0:
            status = _status(
                "partial",
                core_data_present=True,
                missing_items=required,
                errors=errors,
                data_sources=["SEC EDGAR filing text"],
                source_symbols=_source_symbols(payloads),
            )
            freshness = build_filing_freshness(filing_data)
            status["fundamental_freshness_status"] = (
                freshness["summary"]["fundamental_freshness_status"]
            )
            status["current_analysis_usable"] = False
            return status

        return _status(
            "failed",
            core_data_present=False,
            missing_items=required,
            errors=errors or ["基本面工具没有返回核心财务数据"],
            data_sources=[],
            source_symbols=_source_symbols(payloads),
        )

    amounts = metrics.get("financial_amounts")
    amounts = amounts if isinstance(amounts, dict) else {}
    values = {
        "company": metrics.get("company"),
        "form": metrics.get("form"),
        "income_period_end": metrics.get("income_period_end"),
        "filing_date": metrics.get("filing_date"),
        "revenue": amounts.get("revenue"),
        "net_income": amounts.get("net_income"),
        "operating_cash_flow": amounts.get("operating_cash_flow"),
        "capital_expenditure": amounts.get("capital_expenditure"),
        "free_cash_flow": amounts.get("free_cash_flow"),
        "cash": amounts.get("cash"),
        "total_debt": amounts.get("total_debt"),
        "assets": amounts.get("assets"),
        "equity": amounts.get("equity"),
    }
    missing = [key for key in required if values.get(key) is None]
    freshness = metrics.get("fundamental_freshness")
    freshness_summary = (
        freshness.get("summary")
        if isinstance(freshness, dict)
        else None
    )
    if not isinstance(freshness_summary, dict):
        missing.append("fundamental_freshness")
        freshness_status = "unknown"
        alignment_status = "unknown"
        current_usable = False
        all_core_fresh = False
    else:
        freshness_status = str(
            freshness_summary.get("fundamental_freshness_status")
            or "unknown"
        )
        current_usable = bool(
            freshness_summary.get("current_analysis_usable")
        )
        alignment_status = str(
            freshness_summary.get("latest_period_alignment_status")
            or (
                "aligned"
                if current_usable and freshness_status == "fresh"
                else "unknown"
            )
        )
        all_core_fresh = bool(
            freshness_summary.get("all_available_core_facts_fresh")
        )
        for fact in freshness.get("facts") or []:
            if (
                fact.get("metric_name")
                in {"revenue", "net_income", "operating_cash_flow", "total_assets"}
                and fact.get("value") is not None
                and fact.get("freshness_status") != "fresh"
            ):
                missing.append(
                    f"freshness:{fact.get('metric_name')}="
                    f"{fact.get('freshness_status')}"
                )
        if not current_usable:
            missing.append("fresh_current_core_data")
        if alignment_status != "aligned":
            missing.append(
                f"latest_period_alignment={alignment_status}"
            )
    core_present = bool(
        metrics.get("company")
        and any(amounts.get(key) is not None for key in amounts)
    )

    if not core_present:
        status = "failed"
    elif missing or errors or not all_core_fresh:
        status = "partial"
    else:
        status = "success"

    sources = [str(metrics.get("source") or "SEC Company Facts API")]
    if filing_data:
        sources.append(str(filing_data.get("source") or "SEC EDGAR"))

    result = _status(
        status,
        core_data_present=core_present,
        missing_items=missing,
        errors=errors,
        data_sources=sources,
        source_symbols=_source_symbols(payloads),
    )
    result["fundamental_freshness_status"] = freshness_status
    result["latest_period_alignment_status"] = alignment_status
    result["current_analysis_usable"] = current_usable
    return result


def evaluate_news_status(
    payloads: dict[str, list[dict[str, Any]]],
    errors: list[str],
) -> dict[str, Any]:
    news_results = payloads.get("get_stock_news", [])
    news = news_results[-1] if news_results else None
    filing_results = payloads.get("get_recent_sec_filing_texts", [])
    filings = filing_results[-1] if filing_results else None

    if news is None and filings is None:
        return _status(
            "failed",
            core_data_present=False,
            missing_items=["news_query", "sec_filing_query"],
            errors=errors or ["消息面工具没有返回结构化数据"],
            data_sources=[],
            source_symbols={},
        )

    article_count = int(news.get("article_count") or 0) if news else 0
    filing_count = int(filings.get("filing_count") or 0) if filings else 0
    missing = []

    if news is None:
        missing.append("news_query")
    elif article_count == 0:
        missing.append("filtered_news_articles")
    else:
        articles = news.get("articles") or []
        headline_only_count = sum(
            item.get("evidence_depth") == "headline_only"
            for item in articles
            if isinstance(item, dict)
        )
        unavailable_count = sum(
            item.get("evidence_depth") == "unavailable"
            for item in articles
            if isinstance(item, dict)
        )
        if headline_only_count:
            missing.append(f"headline_only_articles:{headline_only_count}")
        if unavailable_count:
            missing.append(f"unavailable_articles:{unavailable_count}")

    if filings is None:
        missing.append("sec_filing_query")
    elif filing_count == 0:
        missing.append("recent_sec_filings")
    else:
        filing_items = filings.get("filings") or []
        truncated_count = sum(
            bool((item.get("primary_document") or {}).get("truncated"))
            or any(
                bool(exhibit.get("truncated"))
                for exhibit in item.get("exhibits") or []
                if isinstance(exhibit, dict)
            )
            for item in filing_items
            if isinstance(item, dict)
        )
        if truncated_count:
            missing.append(f"truncated_sec_sources:{truncated_count}")

    core_present = article_count > 0 or filing_count > 0

    if news is None and not core_present:
        status = "failed"
    elif missing or errors:
        status = "partial"
    else:
        status = "success"

    sources = []
    if news:
        sources.append(str(news.get("source") or "Alpaca News API"))
    if filings:
        sources.append(str(filings.get("source") or "SEC EDGAR"))

    return _status(
        status,
        core_data_present=core_present,
        missing_items=missing,
        errors=errors,
        data_sources=sources,
        source_symbols=_source_symbols(payloads),
    )


_EVALUATORS: dict[
    str,
    Callable[
        [dict[str, list[dict[str, Any]]], list[str]],
        dict[str, Any],
    ],
] = {
    "technical": evaluate_technical_status,
    "fundamental": evaluate_fundamental_status,
    "news": evaluate_news_status,
}


def build_specialist_envelope(agent_role: str, run_result: Any) -> str:
    payloads, errors = collect_tool_results(run_result)
    status = _EVALUATORS[agent_role](payloads, errors)
    citations = build_agent_citations(agent_role, payloads)

    envelope = {
        "agent": agent_role,
        "agent_status": status,
        "report": str(getattr(run_result, "final_output", "") or ""),
        "citations": citations,
    }

    if agent_role == "technical":
        results = payloads.get("get_technical_metrics", [])
        metrics = results[-1] if results else {}
        evidence = metrics.get("technical_evidence")

        if isinstance(evidence, dict):
            envelope["technical_evidence"] = evidence

    if agent_role == "fundamental":
        results = payloads.get("get_fundamental_metrics", [])
        metrics = results[-1] if results else {}
        freshness = metrics.get("fundamental_freshness")
        if not isinstance(freshness, dict):
            filings = payloads.get("get_recent_sec_filing_texts", [])
            freshness = (
                build_filing_freshness(filings[-1])
                if filings
                else None
            )
        if isinstance(freshness, dict):
            envelope["fundamental_freshness"] = freshness
        evidence = metrics.get("fundamental_evidence")
        if isinstance(evidence, dict):
            envelope["fundamental_evidence"] = evidence

    return json.dumps(
        envelope,
        ensure_ascii=False,
    )


async def extract_technical_agent_output(run_result: Any) -> str:
    return build_specialist_envelope("technical", run_result)


async def extract_fundamental_agent_output(run_result: Any) -> str:
    return build_specialist_envelope("fundamental", run_result)


async def extract_news_agent_output(run_result: Any) -> str:
    return build_specialist_envelope("news", run_result)


def extract_top_level_technical_evidence(
    run_result: Any,
) -> dict[str, Any] | None:
    payloads, _ = collect_tool_results(run_result)

    for payload in reversed(payloads.get("technical_analysis", [])):
        evidence = payload.get("technical_evidence")
        if isinstance(evidence, dict):
            return evidence

    return None


def extract_top_level_citations(run_result: Any) -> list[dict[str, Any]]:
    payloads, _ = collect_tool_results(run_result)
    citations: list[dict[str, Any]] = []

    for tool_name in (
        "technical_analysis",
        "fundamental_analysis",
        "news_analysis",
    ):
        for payload in payloads.get(tool_name, []):
            values = payload.get("citations")
            if isinstance(values, list):
                citations.extend(
                    item for item in values if isinstance(item, dict)
                )

    return citations


def extract_top_level_fundamental_freshness(
    run_result: Any,
) -> dict[str, Any] | None:
    payloads, _ = collect_tool_results(run_result)
    for payload in reversed(payloads.get("fundamental_analysis", [])):
        freshness = payload.get("fundamental_freshness")
        if isinstance(freshness, dict):
            return freshness
    return None


def extract_top_level_fundamental_evidence(
    run_result: Any,
) -> dict[str, Any] | None:
    payloads, _ = collect_tool_results(run_result)
    for payload in reversed(payloads.get("fundamental_analysis", [])):
        evidence = payload.get("fundamental_evidence")
        if isinstance(evidence, dict):
            return evidence
    return None


def extract_top_level_statuses(
    run_result: Any,
) -> dict[str, dict[str, Any]]:
    payloads, _ = collect_tool_results(run_result)
    statuses: dict[str, dict[str, Any]] = {}

    for tool_name, role in _AGENT_TOOL_NAMES.items():
        for payload in payloads.get(tool_name, []):
            status = payload.get("agent_status")
            if isinstance(status, dict):
                statuses[role] = status

    for role in ("technical", "fundamental", "news"):
        if role not in statuses:
            statuses[role] = _status(
                "failed",
                core_data_present=False,
                missing_items=["structured_agent_status"],
                errors=["未取得结构化 Agent 状态"],
                data_sources=[],
                source_symbols={},
            )

    report = str(getattr(run_result, "final_output", "") or "").strip()
    specialist_statuses = [
        statuses[role]["status"]
        for role in ("technical", "fundamental", "news")
    ]

    if not report:
        strategy_status = "failed"
    elif all(status == "success" for status in specialist_statuses):
        strategy_status = "success"
    else:
        strategy_status = "partial"

    incomplete_roles = [
        role
        for role in ("technical", "fundamental", "news")
        if statuses[role]["status"] != "success"
    ]
    statuses["strategy"] = _status(
        strategy_status,
        core_data_present=bool(report),
        missing_items=incomplete_roles,
        errors=[] if report else ["策略报告为空"],
        data_sources=["specialist agent reports"] if report else [],
        source_symbols={},
    )

    return statuses


def warnings_from_statuses(
    statuses: dict[str, dict[str, Any]],
) -> list[str]:
    warnings = []

    for role in AGENT_ROLES:
        detail = statuses.get(role, {})
        if detail.get("status") == "success":
            continue

        missing = detail.get("missing_items") or []
        errors = detail.get("errors") or []
        parts = [f"{role}={detail.get('status', 'failed')}"]
        if missing:
            parts.append("缺失：" + ", ".join(map(str, missing)))
        if errors:
            parts.append("错误：" + "; ".join(map(str, errors)))
        warnings.append("；".join(parts))

    return warnings

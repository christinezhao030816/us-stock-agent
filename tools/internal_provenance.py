from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from tools.agent_status import collect_tool_results


def _record(
    provenance_id: str,
    category: str,
    metric_name: str,
    input_fields: dict[str, Any],
    formula_or_rule: str,
    result: Any,
    timestamp: str,
    path: str,
) -> dict[str, Any]:
    return {
        "provenance_id": provenance_id,
        "claim_category": category,
        "metric_name": metric_name,
        "input_fields": input_fields,
        "formula_or_rule": formula_or_rule,
        "result": result,
        "calculation_timestamp": timestamp,
        "provenance_path": path,
        "deterministic": True,
    }


def build_internal_provenance(
    run_result: Any,
    *,
    agent_statuses: dict[str, dict[str, Any]],
    fundamental_freshness: dict[str, Any] | None,
    technical_evidence: dict[str, Any] | None = None,
    citations: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Build auditable provenance for system scores, diagnostics and limits."""

    payloads, collection_errors = collect_tool_results(run_result)
    now = datetime.now(timezone.utc).isoformat()
    records: list[dict[str, Any]] = []
    confidence_values = payloads.get("calculate_confidence_score", [])
    if confidence_values:
        result = confidence_values[-1]
        metadata = result.get("internal_provenance") or {}
        records.append(
            _record(
                "I1",
                "system_derived",
                "confidence_score",
                dict(metadata.get("input_fields") or {}),
                str(metadata.get("formula_or_rule") or "tools.confidence_tool.calculate_confidence_score/v1"),
                {key: value for key, value in result.items() if key != "internal_provenance"},
                str(metadata.get("calculation_timestamp") or now),
                str(metadata.get("provenance_path") or "tool_outputs.calculate_confidence_score[-1]"),
            )
        )

    next_id = len(records) + 1
    technical = technical_evidence if isinstance(technical_evidence, dict) else {}
    claim_records = [item for item in (citations or []) if isinstance(item, dict)]
    news_dates = [
        str(item.get("source_date"))
        for item in claim_records
        if item.get("agent") == "news" and item.get("source_date")
    ]
    context_inputs = {
        "technical_last_trading_date": technical.get("last_trading_date"),
        "technical_price_basis": technical.get("price_basis"),
        "technical_evidence_path": technical.get("evidence_path"),
        "news_source_dates": news_dates,
    }
    if any(value for value in context_inputs.values()):
        records.append(
            _record(
                f"I{next_id}",
                "system_derived",
                "analysis_data_context",
                context_inputs,
                "technical date from saved evidence; news cutoff=max structured news source_date",
                {
                    "technical_data_date": technical.get("last_trading_date"),
                    "news_data_date": max(news_dates) if news_dates else None,
                },
                now,
                "response.technical_evidence + response.citations[agent=news]",
            )
        )
        next_id += 1
    for claim in claim_records:
        if claim.get("agent") != "news" or claim.get("evidence_depth") != "unavailable":
            continue
        records.append(
            _record(
                f"I{next_id}",
                "operational_disclosure",
                f"news_evidence_unavailable:{claim.get('claim_id')}",
                {
                    "claim_id": claim.get("claim_id"),
                    "source_url": claim.get("source_url"),
                    "source_content": claim.get("source_content") or {},
                },
                "tools.news_evidence.classify_article_evidence/v1",
                "unavailable",
                now,
                f"response.citations[{claim.get('claim_id')}]",
            )
        )
        next_id += 1
    freshness = fundamental_freshness if isinstance(fundamental_freshness, dict) else {}
    if freshness:
        records.append(
            _record(
                f"I{next_id}",
                "system_derived",
                "fundamental_freshness_and_alignment",
                {
                    "analysis_as_of": freshness.get("analysis_as_of"),
                    "thresholds": freshness.get("thresholds"),
                    "facts": freshness.get("facts"),
                    "latest_filing": freshness.get("latest_filing"),
                },
                "tools.financial_freshness: age thresholds plus latest applicable filing alignment",
                freshness.get("summary") or {},
                now,
                "response.fundamental_freshness",
            )
        )
        next_id += 1

    for role in ("technical", "fundamental", "news", "strategy"):
        detail = agent_statuses.get(role) or {}
        records.append(
            _record(
                f"I{next_id}",
                "operational_disclosure",
                f"{role}_agent_status",
                {
                    "core_data_present": detail.get("core_data_present"),
                    "missing_items": detail.get("missing_items") or [],
                    "errors": detail.get("errors") or [],
                    "data_sources": detail.get("data_sources") or [],
                },
                f"tools.agent_status.evaluate_{role}_status/v1",
                detail.get("status") or "failed",
                now,
                f"response.agent_statuses.{role}",
            )
        )
        next_id += 1

    if collection_errors:
        records.append(
            _record(
                f"I{next_id}",
                "operational_disclosure",
                "tool_collection_errors",
                {"errors": collection_errors},
                "tools.agent_status.collect_tool_results/v1",
                "errors_detected",
                now,
                "run_result.new_items",
            )
        )
    return records


def internal_traceability_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    required = len(records)
    traced = sum(
        bool(item.get("input_fields"))
        and bool(item.get("formula_or_rule"))
        and "result" in item
        and bool(item.get("calculation_timestamp"))
        and bool(item.get("provenance_path"))
        and item.get("deterministic") is True
        for item in records
    )
    return {
        "internal_system_claims": required,
        "traceable_internal_claims": traced,
        "internal_traceability_coverage_pct": round(traced / required * 100, 2) if required else 100.0,
    }


def append_internal_provenance_note(
    report: str,
    records: list[dict[str, Any]],
    *,
    language: str = "en",
) -> str:
    if not records:
        return report
    title = (
        "System Diagnostics and Internal Provenance"
        if language == "en"
        else "系统诊断与内部溯源"
    )
    lines = [str(report or "").rstrip(), "", f"## {title}", ""]
    for item in records:
        if language == "en":
            lines.append(
                f"- [{item['provenance_id']}] `{item['metric_name']}`: "
                f"result={item['result']!s}; rule={item['formula_or_rule']}; "
                f"path={item['provenance_path']}; deterministic=true"
            )
        else:
            lines.append(
                f"- [{item['provenance_id']}] 系统诊断 `{item['metric_name']}`："
                f"结果={item['result']!s}; 规则={item['formula_or_rule']}; "
                f"路径={item['provenance_path']}; deterministic=true"
            )
    return "\n".join(lines).rstrip() + "\n"

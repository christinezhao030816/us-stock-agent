#!/usr/bin/env python3
"""Targeted Phase 7A regression for the frozen Phase 6B P0 claims."""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
BASELINE_ROOT = (
    PROJECT_ROOT
    / "evaluation"
    / "final_evaluation_regression"
    / "phase6b-final-19stocks-20261107T074706Z-44ed0bf"
)
DEFAULT_ROOT = PROJECT_ROOT / "evaluation" / "phase7a_p0_regression"
SYMBOLS = ("ABBV", "CAT", "COST", "JNJ", "NVDA", "TM", "XOM", "SONY")
TARGETS = (
    ("ABBV", "F2", "financial", "net_income"),
    ("CAT", "F5", "financial", "capital_expenditure"),
    ("CAT", "N1", "headline", "event"),
    ("COST", "N3", "headline", "event"),
    ("JNJ", "N5", "headline", "event"),
    ("NVDA", "N4", "headline", "event"),
    ("TM", "N2", "headline", "event"),
    ("TM", "N4", "headline", "event"),
    ("XOM", "N6", "news_gating", "event"),
    ("XOM", "N7", "news_gating", "event"),
    ("XOM", "N8", "news_gating", "event"),
    ("XOM", "N9", "news_gating", "event"),
    ("XOM", "N10", "news_gating", "event"),
    ("SONY", "N6", "news_gating", "event"),
)
NUMERICAL_TARGETS = {
    ("ABBV", "F2"),
    ("CAT", "F5"),
    ("CAT", "N1"),
    ("COST", "N3"),
    ("JNJ", "N5"),
    ("NVDA", "N4"),
    ("TM", "N2"),
    ("TM", "N4"),
}
CORE_SECTION_TERMS = (
    "看多", "看空", "风险", "催化", "综合判断", "最终结论", "投资建议",
    "bull", "bear", "risk", "catalyst", "final synthesis", "recommendation",
)
NUMBER_TOKEN = re.compile(
    r"(?:[$¥]\d[\d.]*B?|\d+(?:\.\d+)?\s*Percent|FY\d{4}|2027)",
    re.IGNORECASE,
)


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty audit: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _baseline_claims(ticker: str) -> list[dict[str, Any]]:
    payload = json.loads(
        (BASELINE_ROOT / "source_registry" / f"{ticker}.json").read_text(
            encoding="utf-8"
        )
    )
    return [item for item in payload.get("citations") or [] if isinstance(item, dict)]


def _match_after_claim(
    before: dict[str, Any],
    after_claims: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if before.get("agent") == "fundamental":
        return next(
            (
                item
                for item in after_claims
                if item.get("agent") == "fundamental"
                and item.get("metric_name") == before.get("metric_name")
            ),
            None,
        )
    for field in ("source_url", "source_anchor"):
        value = before.get(field)
        if value:
            matched = next(
                (
                    item
                    for item in after_claims
                    if item.get("agent") == "news"
                    and item.get(field) == value
                ),
                None,
            )
            if matched:
                return matched
    before_title = str((before.get("source_content") or {}).get("title") or "")
    return next(
        (
            item
            for item in after_claims
            if str((item.get("source_content") or {}).get("title") or "")
            == before_title
        ),
        None,
    )


def _marker_lines(report: str, claim_id: str) -> list[str]:
    marker = f"[{claim_id}]"
    return [line.strip() for line in report.splitlines() if marker in line]


def _core_marker_lines(report: str, claim_id: str) -> list[str]:
    marker = f"[{claim_id}]"
    section = ""
    matches = []
    for raw in report.splitlines():
        heading = re.match(r"^\s*#{1,6}\s+(.*)$", raw)
        if heading:
            section = re.sub(r"[*_`]", "", heading.group(1)).strip().lower()
            continue
        if marker in raw and any(term in section for term in CORE_SECTION_TERMS):
            matches.append(raw.strip())
    return matches


def _canonical_headline(value: str) -> str:
    return re.sub(r"[;；]+\s*", "，", str(value or "")).strip().lower()


def _baseline_ground_truth_value(
    fact_rows: list[dict[str, str]], ticker: str, claim_id: str
) -> Any:
    row = next(
        item
        for item in fact_rows
        if item["ticker"] == ticker and item["claim_id"] == claim_id
    )
    ground_truth = json.loads(row["ground_truth"])
    return ground_truth.get("value") if isinstance(ground_truth, dict) else None


def _unknown_citations(summary: dict[str, Any]) -> list[str]:
    unresolved = summary.get("unresolved_report_citations") or []
    return [str(item) for item in unresolved]


def _bool_cell(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def reaudit_existing(output_directory: Path) -> dict[str, Any]:
    """Reconcile frozen claims that are absent from the current news window.

    This is deliberately offline: it reads only the already saved Phase 7A
    responses/reports and never calls an Agent or an external data source.
    """

    news_path = output_directory / "news_boundary_audit.csv"
    strategy_path = output_directory / "strategy_gating_audit.csv"
    batch_path = output_directory / "batch_results.csv"
    validation_path = output_directory / "validation.json"
    news_rows = _read_csv(news_path)
    strategy_rows = _read_csv(strategy_path)
    batch_rows = _read_csv(batch_path)
    report_by_ticker = {
        row["ticker"]: PROJECT_ROOT / row["report_path"]
        for row in batch_rows
    }

    for row in news_rows:
        matched = bool(row.get("after_claim_id"))
        row["current_disposition"] = (
            "matched_current_claim"
            if matched
            else "not_returned_in_current_window"
        )
        if not matched:
            report = report_by_ticker[row["ticker"]].read_text(
                encoding="utf-8"
            )
            old_claim_present = _canonical_headline(row["title"]) in (
                _canonical_headline(report)
            )
            row["baseline_claim_present_unregistered"] = old_claim_present
            row["boundary_pass"] = not old_claim_present
        else:
            row["baseline_claim_present_unregistered"] = False

    news_disposition = {
        (row["ticker"], row["before_claim_id"]): row["current_disposition"]
        for row in news_rows
    }
    for row in strategy_rows:
        row["current_disposition"] = news_disposition[
            (row["ticker"], row["before_claim_id"])
        ]

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    passed_news = sum(_bool_cell(row["boundary_pass"]) for row in news_rows)
    passed_gating = sum(_bool_cell(row["gating_pass"]) for row in strategy_rows)
    validation["news_boundary_passed"] = passed_news
    validation["news_targets_not_returned_in_current_window"] = sum(
        row["current_disposition"] == "not_returned_in_current_window"
        for row in news_rows
    )
    validation["acceptance_passed"] = bool(
        validation.get("all_original_numerical_errors_eliminated")
        and passed_news == len(news_rows)
        and passed_gating == len(strategy_rows)
        and validation.get("invalid_unknown_citation_count") == 0
        and validation.get("all_news_modules_retained")
    )
    validation["offline_reaudit"] = True
    validation["offline_reaudit_at_utc"] = datetime.now(timezone.utc).isoformat()
    _write_csv(news_path, news_rows)
    _write_csv(strategy_path, strategy_rows)
    _write_json(validation_path, validation)
    return validation


async def run(output_directory: Path) -> dict[str, Any]:
    if not BASELINE_ROOT.is_dir():
        raise FileNotFoundError(f"frozen baseline not found: {BASELINE_ROOT}")
    output_directory.mkdir(parents=True, exist_ok=False)
    reports_directory = output_directory / "reports"
    responses_directory = output_directory / "responses"
    reports_directory.mkdir()
    responses_directory.mkdir()
    os.environ["TECHNICAL_EVIDENCE_DIR"] = str(
        output_directory / "technical_evidence"
    )

    from app import AnalysisRequest, analyze_stock
    from evaluation.batch_runner import (
        extract_agent_statuses,
        save_citation_registry,
        save_technical_evidence,
    )
    from evaluation.final_evaluation_runner import (
        _news_scope_consistent,
        _sentence_units,
    )

    baseline_fact_rows = _read_csv(BASELINE_ROOT / "fact_evaluation.csv")
    baseline_by_ticker = {ticker: _baseline_claims(ticker) for ticker in SYMBOLS}
    responses: dict[str, dict[str, Any]] = {}
    batch_rows: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()

    for ticker in SYMBOLS:
        ticker_start = datetime.now(timezone.utc)
        response = await analyze_stock(AnalysisRequest(symbol=ticker))
        data = response.model_dump()
        responses[ticker] = data
        safe = ticker.replace(".", "_")
        response_path = responses_directory / f"{safe}.json"
        report_path = reports_directory / f"{safe}.md"
        _write_json(response_path, data)
        report_path.write_text(data["report"], encoding="utf-8")
        citation_fields = save_citation_registry(
            data, data["report"], ticker, output_directory
        )
        evidence_fields = save_technical_evidence(
            data, ticker, output_directory
        )
        statuses, status_source = extract_agent_statuses(data, data["report"])
        citation_summary = data.get("citation_summary") or {}
        unknown = _unknown_citations(citation_summary)
        ticker_end = datetime.now(timezone.utc)
        batch_rows.append(
            {
                "ticker": ticker,
                "start_time_utc": ticker_start.isoformat(),
                "end_time_utc": ticker_end.isoformat(),
                "latency_seconds": round(
                    (ticker_end - ticker_start).total_seconds(), 3
                ),
                "technical_status": statuses["technical"],
                "fundamental_status": statuses["fundamental"],
                "news_status": statuses["news"],
                "strategy_status": statuses["strategy"],
                "status_source": status_source,
                "citation_integrity": citation_summary.get(
                    "citation_integrity", False
                ),
                "invalid_unknown_citation_count": len(unknown),
                "news_module_retained": (
                    "消息面摘要" in data["report"]
                    or "news" in data["report"].lower()
                ),
                "report_path": _portable(report_path),
                "response_path": _portable(response_path),
                **citation_fields,
                **evidence_fields,
            }
        )

    before_after_rows: list[dict[str, Any]] = []
    numerical_rows: list[dict[str, Any]] = []
    news_rows: list[dict[str, Any]] = []
    strategy_rows: list[dict[str, Any]] = []

    for ticker, before_id, issue_type, metric in TARGETS:
        before = next(
            item
            for item in baseline_by_ticker[ticker]
            if item.get("claim_id") == before_id
        )
        data = responses[ticker]
        after = _match_after_claim(before, data.get("citations") or [])
        after_id = str((after or {}).get("claim_id") or "")
        report = data["report"]
        occurrence_lines = _marker_lines(report, after_id) if after_id else []
        core_lines = _core_marker_lines(report, after_id) if after_id else []
        title = str((before.get("source_content") or {}).get("title") or "")
        number_tokens = NUMBER_TOKEN.findall(title)
        tokens_preserved = all(
            any(token.lower() in line.lower() for line in occurrence_lines)
            for token in number_tokens
        )
        scope_consistent = True
        if after and after.get("agent") == "news":
            relevant_units = [
                sentence
                for _, sentence in _sentence_units(report)
                if f"[{after_id}]" in sentence
            ]
            scope_consistent = bool(relevant_units) and all(
                _news_scope_consistent(sentence, after)
                or "待核实" in sentence
                for sentence in relevant_units
            )
        baseline_claim_present_unregistered = bool(
            not after
            and _canonical_headline(title) in _canonical_headline(report)
        )
        current_disposition = (
            "matched_current_claim"
            if after
            else "not_returned_in_current_window"
        )
        before_after_rows.append(
            {
                "ticker": ticker,
                "before_claim_id": before_id,
                "after_claim_id": after_id,
                "issue_type": issue_type,
                "metric_name": metric,
                "before_value": json.dumps(
                    before.get("value"), ensure_ascii=False
                ),
                "after_value": json.dumps(
                    (after or {}).get("value"), ensure_ascii=False
                ),
                "before_evidence_depth": before.get("evidence_depth", ""),
                "after_evidence_depth": (after or {}).get(
                    "evidence_depth", ""
                ),
                "after_strategy_eligible": (after or {}).get(
                    "strategy_eligible", ""
                ),
                "report_occurrence_count": len(occurrence_lines),
                "core_strategy_occurrence_count": len(core_lines),
                "number_tokens_preserved": tokens_preserved,
                "scope_consistent": scope_consistent,
                "matched_after_claim": after is not None,
                "current_disposition": current_disposition,
            }
        )

        if (ticker, before_id) in NUMERICAL_TARGETS:
            if issue_type == "financial":
                expected = _baseline_ground_truth_value(
                    baseline_fact_rows, ticker, before_id
                )
                after_value = (after or {}).get("value")
                registry_exact = (
                    isinstance(after_value, (int, float))
                    and abs(float(after_value) - float(expected)) < 1e-12
                )
                display = str((after or {}).get("claim_text") or "").split(
                    " = ", 1
                )[-1]
                report_exact = bool(
                    display
                    and any(display in line for line in occurrence_lines)
                )
                eliminated = bool(registry_exact and report_exact)
                expected_value = expected
            else:
                expected_value = title
                registry_exact = after is not None and str(
                    (after.get("source_content") or {}).get("title") or ""
                ) == title
                report_exact = bool(tokens_preserved and scope_consistent)
                eliminated = bool(registry_exact and report_exact)
            numerical_rows.append(
                {
                    "ticker": ticker,
                    "before_claim_id": before_id,
                    "after_claim_id": after_id,
                    "metric_or_event": metric,
                    "expected_value_or_title": expected_value,
                    "after_registry_value": json.dumps(
                        (after or {}).get("value"), ensure_ascii=False
                    ),
                    "registry_exact": registry_exact,
                    "report_exact": report_exact,
                    "original_numerical_error_eliminated": eliminated,
                }
            )

        if before.get("agent") == "news":
            full_text = bool((after or {}).get("full_text_available"))
            eligible = bool((after or {}).get("strategy_eligible"))
            boundary_pass = bool(
                (
                    after
                    and scope_consistent
                    and tokens_preserved
                    and (eligible or not core_lines)
                )
                or (not after and not baseline_claim_present_unregistered)
            )
            news_rows.append(
                {
                    "ticker": ticker,
                    "before_claim_id": before_id,
                    "after_claim_id": after_id,
                    "title": title,
                    "evidence_depth": (after or {}).get(
                        "evidence_depth", ""
                    ),
                    "full_text_available": full_text,
                    "strategy_eligible": eligible,
                    "strategy_limitation_reason": (after or {}).get(
                        "strategy_limitation_reason", ""
                    ),
                    "number_tokens": ";".join(number_tokens),
                    "number_tokens_preserved": tokens_preserved,
                    "scope_consistent": scope_consistent,
                    "core_strategy_occurrence_count": len(core_lines),
                    "limitation_disclosed": (
                        "未读取完整正文" in report
                        or "附件内容不完整" in report
                        or "待核实" in report
                    ),
                    "boundary_pass": boundary_pass,
                    "current_disposition": current_disposition,
                    "baseline_claim_present_unregistered": (
                        baseline_claim_present_unregistered
                    ),
                }
            )
            strategy_rows.append(
                {
                    "ticker": ticker,
                    "before_claim_id": before_id,
                    "after_claim_id": after_id,
                    "strategy_eligible": eligible,
                    "core_strategy_occurrence_count": len(core_lines),
                    "gating_pass": bool(eligible or not core_lines),
                    "news_module_retained": (
                        "消息面摘要" in report or "news" in report.lower()
                    ),
                    "current_disposition": current_disposition,
                }
            )

    numerical_eliminated = sum(
        bool(row["original_numerical_error_eliminated"])
        for row in numerical_rows
    )
    invalid_total = sum(
        int(row["invalid_unknown_citation_count"])
        for row in batch_rows
    )
    xom_sony_gating = [
        row
        for row in strategy_rows
        if row["ticker"] in {"XOM", "SONY"}
    ]
    validation = {
        "baseline_root": _portable(BASELINE_ROOT),
        "output_directory": _portable(output_directory),
        "symbols": list(SYMBOLS),
        "run_started_at_utc": started_at,
        "run_finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "single_attempt_per_ticker": True,
        "numerical_error_targets": len(numerical_rows),
        "numerical_errors_eliminated": numerical_eliminated,
        "all_original_numerical_errors_eliminated": (
            numerical_eliminated == len(NUMERICAL_TARGETS) == 8
        ),
        "news_boundary_targets": len(news_rows),
        "news_boundary_passed": sum(
            bool(row["boundary_pass"]) for row in news_rows
        ),
        "news_targets_not_returned_in_current_window": sum(
            row["current_disposition"] == "not_returned_in_current_window"
            for row in news_rows
        ),
        "xom_sony_core_gating_targets": len(xom_sony_gating),
        "xom_sony_core_gating_passed": sum(
            bool(row["gating_pass"]) for row in xom_sony_gating
        ),
        "all_xom_sony_core_gating_passed": bool(
            xom_sony_gating
            and all(row["gating_pass"] for row in xom_sony_gating)
        ),
        "invalid_unknown_citation_count": invalid_total,
        "all_news_modules_retained": all(
            bool(row["news_module_retained"]) for row in batch_rows
        ),
        "acceptance_passed": bool(
            numerical_eliminated == 8
            and all(row["boundary_pass"] for row in news_rows)
            and xom_sony_gating
            and all(row["gating_pass"] for row in xom_sony_gating)
            and invalid_total == 0
            and all(bool(row["news_module_retained"]) for row in batch_rows)
        ),
    }

    _write_csv(output_directory / "batch_results.csv", batch_rows)
    _write_csv(
        output_directory / "affected_claims_before_after.csv",
        before_after_rows,
    )
    _write_csv(
        output_directory / "numerical_error_audit.csv",
        numerical_rows,
    )
    _write_csv(
        output_directory / "news_boundary_audit.csv",
        news_rows,
    )
    _write_csv(
        output_directory / "strategy_gating_audit.csv",
        strategy_rows,
    )
    _write_json(output_directory / "validation.json", validation)
    return validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--audit-existing",
        action="store_true",
        help="offline-only reconciliation of an already saved run",
    )
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or DEFAULT_ROOT / timestamp
    result = (
        reaudit_existing(output)
        if args.audit_existing
        else asyncio.run(run(output))
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

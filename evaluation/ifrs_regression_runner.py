import argparse
import asyncio
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_ROOT = PROJECT_ROOT / "evaluation" / "ifrs_regression"
SYMBOLS = ("TSM", "SONY", "TM", "AAPL")
FOREIGN_SYMBOLS = {"TSM", "SONY", "TM"}
CORE_METRICS = (
    "revenue",
    "net_income",
    "EPS",
    "operating_cash_flow",
    "capital_expenditure",
    "total_assets",
    "total_liabilities",
    "cash",
    "total_debt",
)


def _portable(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _fundamental_claims(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(claim.get("metric_name")): claim
        for claim in data.get("citations") or []
        if claim.get("agent") == "fundamental"
        and claim.get("metric_name") in CORE_METRICS
    }


def _base_currency(unit: Any) -> str | None:
    value = str(unit or "").strip()
    if not value:
        return None
    return value.split()[0].split("/")[0]


async def run(output_directory: Path) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=False)
    os.environ["TECHNICAL_EVIDENCE_DIR"] = str(
        output_directory / "technical_evidence"
    )
    from app import AnalysisRequest, analyze_stock
    from evaluation.batch_runner import (
        extract_agent_statuses,
        extract_freshness_fields,
        save_citation_registry,
        save_technical_evidence,
    )
    from tools.citation_registry import citation_summary

    rows = []
    validations: dict[str, Any] = {}
    for symbol in SYMBOLS:
        response = await analyze_stock(AnalysisRequest(symbol=symbol))
        data = response.model_dump()
        safe = symbol.replace(".", "_")
        response_path = output_directory / f"{safe}_response.json"
        report_path = output_directory / f"{safe}_report.md"
        freshness_path = output_directory / f"{safe}_freshness.json"
        _write_json(response_path, data)
        _write_json(freshness_path, data.get("fundamental_freshness"))
        report_path.write_text(data["report"], encoding="utf-8")
        evidence_fields = save_technical_evidence(data, symbol, output_directory)
        citation_fields = save_citation_registry(
            data, data["report"], symbol, output_directory
        )
        freshness_fields = extract_freshness_fields(data)
        statuses, status_source = extract_agent_statuses(data, data["report"])
        claims = _fundamental_claims(data)
        freshness = data.get("fundamental_freshness") or {}
        summary = freshness.get("summary") or {}
        independently_recomputed = citation_summary(
            data.get("citations") or [],
            data["report"],
            data.get("source_registry") or [],
        )
        coverage = {}
        for metric in CORE_METRICS:
            claim = claims.get(metric) or {}
            coverage[metric] = {
                "available": claim.get("value") is not None,
                "value": claim.get("value"),
                "unit": claim.get("unit"),
                "period": claim.get("period_or_as_of_date"),
                "source_date": claim.get("source_date"),
                "source_anchor": claim.get("source_anchor"),
                "citation_status": claim.get("citation_status", "unsupported"),
                "freshness_status": claim.get("freshness_status", "unknown"),
            }
        available = sum(item["available"] for item in coverage.values())
        currencies = {
            _base_currency(item["unit"])
            for item in coverage.values()
            if item["available"] and item.get("unit")
            and _base_currency(item["unit"]) not in {"date", "percent", "ratio"}
        }
        validations[symbol] = {
            "pre_fix_available_core_metrics": 0 if symbol in FOREIGN_SYMBOLS else None,
            "post_fix_available_core_metrics": available,
            "coverage": coverage,
            "currencies": sorted(currencies),
            "no_cross_currency_output": (
                currencies <= ({"TWD"} if symbol == "TSM" else {"JPY"})
                if symbol in FOREIGN_SYMBOLS
                else True
            ),
            "fundamental_status": statuses["fundamental"],
            "strategy_status": statuses["strategy"],
            "all_agent_statuses": statuses,
            "status_source": status_source,
            "freshness_summary": summary,
            "fundamental_evidence": data.get("fundamental_evidence"),
            "citation_summary": data.get("citation_summary") or {},
            "citation_summary_matches": (
                data.get("citation_summary") == independently_recomputed
            ),
            "citation_integrity": independently_recomputed["citation_integrity"],
            "latest_20f": summary.get("latest_20f"),
            "latest_6k": summary.get("latest_6k"),
            "latest_20f_companyfacts_record_count": summary.get(
                "latest_20f_companyfacts_record_count"
            ),
            "latest_6k_companyfacts_record_count": summary.get(
                "latest_6k_companyfacts_record_count"
            ),
            "latest_6k_structured_core_present": summary.get(
                "latest_6k_structured_core_present"
            ),
            "report_path": _portable(report_path),
            "response_path": _portable(response_path),
            "freshness_path": _portable(freshness_path),
            "source_registry_path": citation_fields["source_registry_path"],
        }
        rows.append(
            {
                "ticker": symbol,
                "post_fix_available_core_metrics": available,
                **{f"{role}_agent_status": value for role, value in statuses.items()},
                **freshness_fields,
                **citation_fields,
                **evidence_fields,
                "report_path": _portable(report_path),
                "freshness_path": _portable(freshness_path),
                "response_path": _portable(response_path),
            }
        )

    csv_path = output_directory / "batch_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    validation_path = output_directory / "validation.json"
    _write_json(validation_path, validations)
    return {
        "output_directory": _portable(output_directory),
        "batch_results": _portable(csv_path),
        "validation": _portable(validation_path),
        "symbols": validations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or DEFAULT_ROOT / timestamp
    print(json.dumps(asyncio.run(run(output)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

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
DEFAULT_ROOT = PROJECT_ROOT / "evaluation" / "freshness_regression"
URL_PATTERN = re.compile(r"https?://[^\s<>\])]+")


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
    validations = {}
    for symbol in ("DUK", "AAPL", "TSM"):
        response = await analyze_stock(AnalysisRequest(symbol=symbol))
        data = response.model_dump()
        safe_symbol = symbol.replace(".", "_")
        response_path = output_directory / f"{safe_symbol}_response.json"
        report_path = output_directory / f"{safe_symbol}_report.md"
        freshness_path = output_directory / f"{safe_symbol}_freshness.json"
        _write_json(response_path, data)
        _write_json(freshness_path, data.get("fundamental_freshness"))
        report_path.write_text(data["report"], encoding="utf-8")

        evidence_fields = save_technical_evidence(
            data,
            symbol,
            output_directory,
        )
        citation_fields = save_citation_registry(
            data,
            data["report"],
            symbol,
            output_directory,
        )
        freshness_fields = extract_freshness_fields(data)
        statuses, status_source = extract_agent_statuses(data, data["report"])
        independently_recomputed_citations = citation_summary(
            data["citations"],
            data["report"],
            data["source_registry"],
        )
        freshness = data.get("fundamental_freshness") or {}
        facts = freshness.get("facts") or []
        independently_counted = {
            status: sum(
                fact.get("freshness_status") == status for fact in facts
            )
            for status in ("fresh", "caution", "stale", "unknown")
        }
        registered_urls = {
            str(item.get("source_url"))
            for item in data["source_registry"]
            if item.get("source_url")
        }
        report_urls = {
            match.rstrip(".,;:")
            for match in URL_PATTERN.findall(data["report"])
        }
        old_current_facts = [
            fact.get("metric_name")
            for fact in facts
            if str(fact.get("period_end") or "").startswith("2018")
            and fact.get("freshness_status") == "fresh"
        ]
        validations[symbol] = {
            "freshness_summary": freshness.get("summary") or {},
            "financial_periods": {
                fact.get("metric_name"): {
                    "form": fact.get("form"),
                    "period_start": fact.get("period_start"),
                    "period_end": fact.get("period_end"),
                    "filed_date": fact.get("filed_date"),
                    "period_age_days": fact.get("period_age_days"),
                    "freshness_status": fact.get("freshness_status"),
                }
                for fact in facts
            },
            "freshness_counts_match": independently_counted == {
                "fresh": freshness_fields["fresh_fact_count"],
                "caution": freshness_fields["caution_fact_count"],
                "stale": freshness_fields["stale_fact_count"],
                "unknown": freshness_fields["unknown_freshness_count"],
            },
            "old_2018_facts_marked_fresh": old_current_facts,
            "agent_statuses": statuses,
            "status_source": status_source,
            "citation_summary_matches": (
                data["citation_summary"]
                == independently_recomputed_citations
            ),
            "unregistered_report_urls": sorted(report_urls - registered_urls),
            "stale_claim_count": len(data.get("stale_claims") or []),
        }
        rows.append(
            {
                "ticker": symbol,
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

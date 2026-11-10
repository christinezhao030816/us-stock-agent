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
DEFAULT_ROOT = PROJECT_ROOT / "evaluation" / "citation_regression"
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
    technical_directory = output_directory / "technical_evidence"
    os.environ["TECHNICAL_EVIDENCE_DIR"] = str(technical_directory)

    from app import AnalysisRequest, analyze_stock
    from evaluation.batch_runner import (
        extract_agent_statuses,
        save_citation_registry,
        save_technical_evidence,
    )
    from tools.citation_registry import citation_summary

    result_rows = []
    validations = {}
    for symbol in ("AAPL", "BRK.B"):
        response = await analyze_stock(AnalysisRequest(symbol=symbol))
        data = response.model_dump()
        safe_symbol = symbol.replace(".", "_")
        response_path = output_directory / f"{safe_symbol}_response.json"
        report_path = output_directory / f"{safe_symbol}_report.md"
        _write_json(response_path, data)
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
        statuses, status_source = extract_agent_statuses(data, data["report"])
        recomputed = citation_summary(
            data["citations"],
            data["report"],
            data["source_registry"],
        )
        registered_urls = {
            str(item.get("source_url"))
            for item in data["source_registry"]
            if item.get("source_url")
        }
        report_urls = {
            match.rstrip(".,;:")
            for match in URL_PATTERN.findall(data["report"])
        }
        claims_by_agent = {
            agent: [
                item for item in data["citations"] if item.get("agent") == agent
            ]
            for agent in ("technical", "fundamental", "news")
        }
        validations[symbol] = {
            "api_summary_matches_independent_recalculation": (
                data["citation_summary"] == recomputed
            ),
            "unregistered_report_urls": sorted(report_urls - registered_urls),
            "duplicate_claim_ids": recomputed["duplicate_claim_ids"],
            "unresolved_report_citations": recomputed[
                "unresolved_report_citations"
            ],
            "unused_source_ids": recomputed["unused_source_ids"],
            "fundamental_claim_count": len(claims_by_agent["fundamental"]),
            "technical_claim_count": len(claims_by_agent["technical"]),
            "news_claim_count": len(claims_by_agent["news"]),
            "fundamental_has_sources": all(
                item.get("source_url") and item.get("source_anchor")
                for item in claims_by_agent["fundamental"]
                if item.get("citation_status") == "supported"
            ),
            "technical_points_to_p2_evidence": all(
                item.get("evidence_path")
                and str(item.get("source_anchor", "")).startswith(
                    "technical_evidence.indicator_evidence."
                )
                for item in claims_by_agent["technical"]
                if item.get("citation_status") == "supported"
            ),
            "news_supported_items_have_real_metadata": all(
                item.get("source_url")
                and item.get("source_date")
                and item.get("source_anchor")
                for item in claims_by_agent["news"]
                if item.get("citation_status") == "supported"
            ),
            "agent_statuses": statuses,
            "status_source": status_source,
        }
        result_rows.append(
            {
                "ticker": symbol,
                **{f"{role}_agent_status": value for role, value in statuses.items()},
                **evidence_fields,
                **citation_fields,
                "report_path": _portable(report_path),
                "response_path": _portable(response_path),
            }
        )

    csv_path = output_directory / "batch_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(result_rows[0]))
        writer.writeheader()
        writer.writerows(result_rows)
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
    summary = asyncio.run(run(output))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

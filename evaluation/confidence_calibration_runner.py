#!/usr/bin/env python3
"""Offline confidence calibration for a frozen final-evaluation run.

This script never calls production agents or external data sources. It treats the
Phase 6B directory as immutable, verifies its SHA-256 manifest, and writes all
derived Phase 6C artifacts to a separate, new directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


EXPECTED_COMMIT = "44ed0bf4d271c6cac3786b184da832b74041cf6f"
EXPECTED_TICKERS = [
    "AAPL", "ABBV", "BRK.B", "CAT", "COST", "DUK", "EQIX", "GOOGL",
    "JNJ", "JPM", "LIN", "MSFT", "NVDA", "SONY", "TM", "TSLA", "TSM",
    "WELL", "XOM",
]
BAD_CASE_THRESHOLDS = {
    "factual_accuracy_pct_min": 95.0,
    "numerical_accuracy_pct_min": 95.0,
    "unsupported_claim_rate_pct_max": 10.0,
    "citation_coverage_pct_min": 95.0,
    "completeness_score_min": 100.0,
    "numerically_incorrect_max": 0,
    "undisclosed_major_data_limitation_max": 0,
}
BOOTSTRAP_SEED = 20261107
BOOTSTRAP_SAMPLES = 10_000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=BOOTSTRAP_SAMPLES)
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")


def to_float(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    return float(value)


def to_int(value: Any) -> int:
    return int(round(to_float(value)))


def to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_sha_manifest(root: Path) -> dict[str, Any]:
    manifest = json.loads((root / "sha256_manifest.json").read_text(encoding="utf-8"))
    files = manifest.get("files", {})
    missing: list[str] = []
    mismatched: list[str] = []
    for relative, record in files.items():
        path = root / relative
        if not path.is_file():
            missing.append(relative)
            continue
        if sha256_file(path) != record.get("sha256") or path.stat().st_size != record.get("bytes"):
            mismatched.append(relative)
    return {
        "manifest_file_count": manifest.get("file_count"),
        "checked_file_count": len(files),
        "matched_file_count": len(files) - len(missing) - len(mismatched),
        "missing_files": missing,
        "mismatched_files": mismatched,
        "all_match": not missing and not mismatched and len(files) == manifest.get("file_count"),
        "manifest_sha256": sha256_file(root / "sha256_manifest.json"),
    }


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.fmean(values) if values else math.nan


def median(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.median(values) if values else math.nan


def pearson(x: list[float], y: list[float]) -> float:
    if len(x) != len(y) or len(x) < 2:
        return math.nan
    xbar, ybar = mean(x), mean(y)
    dx = [value - xbar for value in x]
    dy = [value - ybar for value in y]
    denom = math.sqrt(sum(value * value for value in dx) * sum(value * value for value in dy))
    if denom == 0:
        return math.nan
    return sum(a * b for a, b in zip(dx, dy)) / denom


def ranks(values: list[float]) -> list[float]:
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    result = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][1] == ordered[start][1]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            result[ordered[position][0]] = average_rank
        start = end
    return result


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(ranks(x), ranks(y))


def quantile(values: list[float], probability: float) -> float:
    values = sorted(values)
    if not values:
        return math.nan
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    fraction = position - lower
    return values[lower] * (1 - fraction) + values[upper] * fraction


def bootstrap_correlations(
    x: list[float], y: list[float], samples: int, rng: random.Random
) -> dict[str, Any]:
    rho_values: list[float] = []
    r_values: list[float] = []
    n = len(x)
    for _ in range(samples):
        indexes = [rng.randrange(n) for _ in range(n)]
        bx = [x[index] for index in indexes]
        by = [y[index] for index in indexes]
        rho = spearman(bx, by)
        r = pearson(bx, by)
        if not math.isnan(rho):
            rho_values.append(rho)
        if not math.isnan(r):
            r_values.append(r)
    return {
        "spearman_bootstrap_valid_samples": len(rho_values),
        "spearman_ci95_low": quantile(rho_values, 0.025),
        "spearman_ci95_high": quantile(rho_values, 0.975),
        "pearson_bootstrap_valid_samples": len(r_values),
        "pearson_ci95_low": quantile(r_values, 0.025),
        "pearson_ci95_high": quantile(r_values, 0.975),
    }


def bucket_for(score: float) -> str:
    if 50 <= score <= 59:
        return "Low"
    if 60 <= score <= 69:
        return "Medium"
    if 70 <= score <= 79:
        return "High"
    return "Outside"


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def status_counts(batch_row: dict[str, str]) -> tuple[int, int, int]:
    statuses = [batch_row[f"{agent}_status"] for agent in ("technical", "fundamental", "news", "strategy")]
    return statuses.count("success"), statuses.count("partial"), statuses.count("failed")


def validate_inputs(root: Path) -> tuple[dict[str, Any], dict[str, list[dict[str, str]]]]:
    required = [
        "batch_results.csv", "report_evaluation.csv", "fact_evaluation.csv",
        "claim_inventory.csv", "evaluation_summary.json", "run_manifest.json",
        "sha256_manifest.json", "evaluation_exceptions.csv",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise RuntimeError(f"Missing frozen inputs: {missing}")

    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("git_commit_sha") != EXPECTED_COMMIT:
        raise RuntimeError("Frozen run commit does not match the Phase 6C baseline")

    tables = {
        "batch": read_csv(root / "batch_results.csv"),
        "report": read_csv(root / "report_evaluation.csv"),
        "fact": read_csv(root / "fact_evaluation.csv"),
        "inventory": read_csv(root / "claim_inventory.csv"),
        "exceptions": read_csv(root / "evaluation_exceptions.csv"),
    }
    summary = json.loads((root / "evaluation_summary.json").read_text(encoding="utf-8"))

    ticker_sets: dict[str, list[str]] = {}
    for name in ("batch", "report"):
        tickers = [row["ticker"] for row in tables[name]]
        if len(tickers) != len(set(tickers)):
            raise RuntimeError(f"Duplicate tickers in {name}")
        ticker_sets[name] = sorted(tickers)
    for name in ("fact", "inventory"):
        ticker_sets[name] = sorted({row["ticker"] for row in tables[name]})
    ticker_sets["exceptions"] = sorted({row["ticker"] for row in tables["exceptions"]})
    ticker_sets["evaluation_summary"] = sorted(summary.get("tickers", []))
    expected = sorted(EXPECTED_TICKERS)
    mismatched = {
        name: tickers
        for name, tickers in ticker_sets.items()
        if name != "exceptions" and tickers != expected
    }
    unexpected_exception_tickers = sorted(set(ticker_sets["exceptions"]) - set(expected))
    if unexpected_exception_tickers:
        mismatched["exceptions_unexpected"] = unexpected_exception_tickers
    if mismatched:
        raise RuntimeError(f"Ticker mismatch across frozen inputs: {mismatched}")

    report_required = [
        "factual_correct", "factual_incorrect", "factual_accuracy_denominator",
        "unsupported_claims", "total_factual_claims", "numerically_correct",
        "numerically_incorrect", "numerical_accuracy_denominator",
        "citation_correct", "citation_incorrect", "citation_accuracy_denominator",
        "validly_cited_claims", "citation_required_claims", "completed_modules",
        "completeness_score",
    ]
    missing_metric_values = []
    for row in tables["report"]:
        for field in report_required:
            if row.get(field, "") == "":
                missing_metric_values.append(f"{row['ticker']}:{field}")
    missing_confidence = [row["ticker"] for row in tables["batch"] if row.get("confidence_score", "") == ""]
    if missing_metric_values or missing_confidence:
        raise RuntimeError(
            f"Incomplete quality inputs: metrics={missing_metric_values}, confidence={missing_confidence}"
        )

    sha = verify_sha_manifest(root)
    if not sha["all_match"]:
        raise RuntimeError(f"Frozen SHA-256 manifest failed: {sha}")

    integrity = {
        "frozen_input_path": str(root.resolve()),
        "git_commit_sha": manifest.get("git_commit_sha"),
        "run_id": manifest.get("run_id"),
        "ticker_count": len(EXPECTED_TICKERS),
        "tickers": EXPECTED_TICKERS,
        "ticker_sets_match": True,
        "confidence_complete": True,
        "quality_numerators_denominators_complete": True,
        "row_counts": {name: len(rows) for name, rows in tables.items()},
        "sha256": sha,
    }
    tables["summary"] = [summary]
    return integrity, tables


def build_confidence_rows(tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    batch = {row["ticker"]: row for row in tables["batch"]}
    reports = {row["ticker"]: row for row in tables["report"]}
    disclosures: dict[str, int] = Counter()
    for row in tables["inventory"]:
        if row.get("item_category") == "operational_disclosure":
            disclosures[row["ticker"]] += 1

    rows: list[dict[str, Any]] = []
    for ticker in EXPECTED_TICKERS:
        b, r = batch[ticker], reports[ticker]
        success, partial, failed = status_counts(b)
        news_total = sum(
            to_int(b[field])
            for field in ("fully_supported_news", "headline_only_news", "truncated_source_news", "unavailable_news")
        )
        limited_news = to_int(b["headline_only_news"]) + to_int(b["truncated_source_news"])
        historical = to_bool(b["historical_fallback"])
        current_available = (
            b["latest_period_alignment_status"] == "aligned"
            and b["fundamental_freshness_status"] in {"fresh", "caution"}
            and failed == 0
        )
        major_limit_present = historical or partial > 0 or failed > 0 or to_int(b["unavailable_news"]) > 0
        undisclosed_major_limit = bool(major_limit_present and disclosures[ticker] == 0)
        row = {
            "ticker": ticker,
            "confidence_score": to_float(b["confidence_score"]),
            "confidence_bucket": bucket_for(to_float(b["confidence_score"])),
            "factual_correct": to_int(r["factual_correct"]),
            "factual_incorrect": to_int(r["factual_incorrect"]),
            "factual_accuracy_denominator": to_int(r["factual_accuracy_denominator"]),
            "factual_accuracy_pct": to_float(r["factual_accuracy_pct"]),
            "unsupported_claims": to_int(r["unsupported_claims"]),
            "total_factual_claims": to_int(r["total_factual_claims"]),
            "unsupported_claim_rate_pct": to_float(r["unsupported_claim_rate_pct"]),
            "numerically_correct": to_int(r["numerically_correct"]),
            "numerically_incorrect": to_int(r["numerically_incorrect"]),
            "numerical_accuracy_denominator": to_int(r["numerical_accuracy_denominator"]),
            "numerical_accuracy_pct": to_float(r["numerical_accuracy_pct"]),
            "citation_correct": to_int(r["citation_correct"]),
            "citation_incorrect": to_int(r["citation_incorrect"]),
            "citation_accuracy_denominator": to_int(r["citation_accuracy_denominator"]),
            "citation_accuracy_pct": to_float(r["citation_accuracy_pct"]),
            "validly_cited_claims": to_int(r["validly_cited_claims"]),
            "citation_required_claims": to_int(r["citation_required_claims"]),
            "citation_coverage_pct": to_float(r["citation_coverage_pct"]),
            "completed_modules": to_int(r["completed_modules"]),
            "completeness_denominator": 8,
            "completeness_score": to_float(r["completeness_score"]),
            "agent_success_count": success,
            "agent_partial_count": partial,
            "agent_failed_count": failed,
            "technical_status": b["technical_status"],
            "fundamental_status": b["fundamental_status"],
            "news_status": b["news_status"],
            "strategy_status": b["strategy_status"],
            "fundamental_freshness_status": b["fundamental_freshness_status"],
            "latest_period_alignment_status": b["latest_period_alignment_status"],
            "historical_fallback": historical,
            "current_data_available": int(current_available),
            "headline_only_count": to_int(b["headline_only_news"]),
            "truncated_source_count": to_int(b["truncated_source_news"]),
            "headline_truncated_share_pct": 100.0 * safe_ratio(limited_news, news_total),
            "operational_disclosure_count": disclosures[ticker],
            "undisclosed_major_data_limitation": int(undisclosed_major_limit),
        }
        rows.append(row)
    return rows


def bad_case_triggers(row: dict[str, Any]) -> list[str]:
    triggers: list[str] = []
    if row["factual_accuracy_pct"] < 95.0:
        triggers.append("factual_accuracy_below_95")
    if row["numerical_accuracy_pct"] < 95.0:
        triggers.append("numerical_accuracy_below_95")
    if row["unsupported_claim_rate_pct"] > 10.0:
        triggers.append("unsupported_claim_rate_above_10")
    if row["citation_coverage_pct"] < 95.0:
        triggers.append("citation_coverage_below_95")
    if row["completeness_score"] < 100.0:
        triggers.append("completeness_below_100")
    if row["numerically_incorrect"] > 0:
        triggers.append("any_numerical_incorrect")
    if row["undisclosed_major_data_limitation"] > 0:
        triggers.append("undisclosed_major_data_limitation")
    return triggers


def build_bad_cases(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        triggers = bad_case_triggers(row)
        is_bad = bool(triggers)
        output.append({
            "ticker": row["ticker"],
            "confidence_score": row["confidence_score"],
            "confidence_bucket": row["confidence_bucket"],
            "is_bad_case": is_bad,
            "trigger_count": len(triggers),
            "trigger_conditions": ";".join(triggers),
            "low_confidence_correctly_identified": bool(is_bad and row["confidence_bucket"] == "Low"),
            "factual_accuracy_pct": row["factual_accuracy_pct"],
            "unsupported_claim_rate_pct": row["unsupported_claim_rate_pct"],
            "numerical_accuracy_pct": row["numerical_accuracy_pct"],
            "citation_coverage_pct": row["citation_coverage_pct"],
            "completeness_score": row["completeness_score"],
            "numerically_incorrect": row["numerically_incorrect"],
            "undisclosed_major_data_limitation": row["undisclosed_major_data_limitation"],
        })
    return output


def build_correlations(
    rows: list[dict[str, Any]], bootstrap_samples: int, seed: int
) -> list[dict[str, Any]]:
    metrics = [
        ("factual_accuracy_pct", "positive"),
        ("numerical_accuracy_pct", "positive"),
        ("citation_accuracy_pct", "positive"),
        ("citation_coverage_pct", "positive"),
        ("completeness_score", "positive"),
        ("agent_success_count", "positive"),
        ("current_data_available", "positive"),
        ("unsupported_claim_rate_pct", "negative"),
        ("factual_incorrect", "negative"),
        ("numerically_incorrect", "negative"),
        ("agent_partial_count", "negative"),
        ("historical_fallback", "negative"),
        ("headline_truncated_share_pct", "negative"),
    ]
    x = [row["confidence_score"] for row in rows]
    output: list[dict[str, Any]] = []
    for index, (metric, expected) in enumerate(metrics):
        y = [float(row[metric]) for row in rows]
        rho, r = spearman(x, y), pearson(x, y)
        bootstrap = bootstrap_correlations(x, y, bootstrap_samples, random.Random(seed + index))
        loo: list[tuple[str, float]] = []
        for omitted in range(len(rows)):
            lx = [value for pos, value in enumerate(x) if pos != omitted]
            ly = [value for pos, value in enumerate(y) if pos != omitted]
            loo.append((rows[omitted]["ticker"], spearman(lx, ly)))
        valid_loo = [(ticker, value) for ticker, value in loo if not math.isnan(value)]
        max_delta_ticker, max_delta_rho = max(
            valid_loo, key=lambda item: abs(item[1] - rho), default=("", math.nan)
        )
        max_delta = abs(max_delta_rho - rho) if not math.isnan(max_delta_rho) else math.nan
        sign_flip = any(value * rho < 0 for _, value in valid_loo) if rho else False
        direction_ok = (expected == "positive" and rho > 0) or (expected == "negative" and rho < 0)
        output.append({
            "metric": metric,
            "expected_direction": expected,
            "n": len(rows),
            "spearman_rho": rho,
            "pearson_r": r,
            **bootstrap,
            "direction_matches_expectation": direction_ok,
            "leave_one_out_rho_min": min((value for _, value in valid_loo), default=math.nan),
            "leave_one_out_rho_max": max((value for _, value in valid_loo), default=math.nan),
            "max_abs_leave_one_out_delta": max_delta,
            "most_influential_ticker": max_delta_ticker,
            "sign_flip_under_leave_one_out": sign_flip,
            "extreme_value_sensitive": bool(sign_flip or (not math.isnan(max_delta) and max_delta >= 0.20)),
        })
    return output


def build_buckets(rows: list[dict[str, Any]], bad_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bad_map = {row["ticker"]: row["is_bad_case"] for row in bad_cases}
    output = []
    for bucket in ("Low", "Medium", "High"):
        group = [row for row in rows if row["confidence_bucket"] == bucket]
        output.append({
            "confidence_bucket": bucket,
            "stock_count": len(group),
            "tickers": ";".join(row["ticker"] for row in group),
            "mean_factual_accuracy_pct": mean(row["factual_accuracy_pct"] for row in group),
            "median_factual_accuracy_pct": median(row["factual_accuracy_pct"] for row in group),
            "mean_unsupported_claim_rate_pct": mean(row["unsupported_claim_rate_pct"] for row in group),
            "median_unsupported_claim_rate_pct": median(row["unsupported_claim_rate_pct"] for row in group),
            "mean_numerical_accuracy_pct": mean(row["numerical_accuracy_pct"] for row in group),
            "median_numerical_accuracy_pct": median(row["numerical_accuracy_pct"] for row in group),
            "mean_citation_coverage_pct": mean(row["citation_coverage_pct"] for row in group),
            "median_citation_coverage_pct": median(row["citation_coverage_pct"] for row in group),
            "agent_partial_total": sum(row["agent_partial_count"] for row in group),
            "agent_partial_mean_per_stock": mean(row["agent_partial_count"] for row in group),
            "historical_fallback_count": sum(int(row["historical_fallback"]) for row in group),
            "historical_fallback_pct": 100.0 * safe_ratio(sum(int(row["historical_fallback"]) for row in group), len(group)),
            "bad_case_count": sum(int(bad_map[row["ticker"]]) for row in group),
            "bad_case_pct": 100.0 * safe_ratio(sum(int(bad_map[row["ticker"]]) for row in group), len(group)),
        })
    return output


def threshold_row(rows: list[dict[str, Any]], bad_map: dict[str, bool], threshold: int) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    for row in rows:
        predicted_bad = row["confidence_score"] < threshold
        actual_bad = bad_map[row["ticker"]]
        if predicted_bad and actual_bad:
            tp += 1
        elif predicted_bad and not actual_bad:
            fp += 1
        elif not predicted_bad and actual_bad:
            fn += 1
        else:
            tn += 1
    return {
        "low_confidence_threshold_exclusive": threshold,
        "rule": f"predict_bad_case_if_confidence_score<{threshold}",
        "true_positive": tp,
        "false_positive": fp,
        "true_negative": tn,
        "false_negative": fn,
        "precision": safe_ratio(tp, tp + fp),
        "recall": safe_ratio(tp, tp + fn),
        "specificity": safe_ratio(tn, tn + fp),
        "accuracy": safe_ratio(tp + tn, len(rows)),
    }


def build_mismatches(rows: list[dict[str, Any]], bad_cases: list[dict[str, Any]]) -> list[dict[str, Any]]:
    bad_map = {row["ticker"]: row for row in bad_cases}
    output: list[dict[str, Any]] = []

    def add(row: dict[str, Any], mismatch_type: str, explanation: str) -> None:
        output.append({
            "ticker": row["ticker"],
            "mismatch_type": mismatch_type,
            "confidence_score": row["confidence_score"],
            "confidence_bucket": row["confidence_bucket"],
            "is_bad_case": bad_map[row["ticker"]]["is_bad_case"],
            "bad_case_triggers": bad_map[row["ticker"]]["trigger_conditions"],
            "factual_accuracy_pct": row["factual_accuracy_pct"],
            "numerical_accuracy_pct": row["numerical_accuracy_pct"],
            "unsupported_claim_rate_pct": row["unsupported_claim_rate_pct"],
            "citation_coverage_pct": row["citation_coverage_pct"],
            "historical_fallback": row["historical_fallback"],
            "agent_partial_count": row["agent_partial_count"],
            "explanation": explanation,
        })

    for row in rows:
        bad = bad_map[row["ticker"]]["is_bad_case"]
        if row["confidence_bucket"] == "High" and bad:
            add(row, "high_confidence_low_quality", "High confidence despite one or more pre-registered Bad Case triggers.")
        if row["confidence_bucket"] == "Low" and not bad:
            add(row, "low_confidence_high_quality", "Low confidence although no pre-registered Bad Case trigger fired.")
        if row["historical_fallback"] and row["factual_accuracy_pct"] >= 95 and row["numerical_accuracy_pct"] >= 95:
            add(row, "stale_but_verified_content", "Verified content is accurate, but historical fallback limits current research usability.")
        if row["factual_incorrect"] > 0 and row["confidence_score"] >= 60:
            add(row, "content_error_not_reflected_in_confidence", "One or more factual errors coexist with Medium/High confidence.")
    return output


def root_cause(notes: str, category: str) -> str:
    if "no_registry_match" in notes:
        return "unmatched_body_claim_no_registry"
    if "article_url_not_found_in_snapshot" in notes:
        return "frozen_news_snapshot_missing_article"
    if "filing accession not present" in notes:
        return "filing_text_snapshot_missing"
    if "body_value_mismatch" in notes and category == "news_events":
        return "news_claim_scope_or_value_mismatch"
    if "body_value_mismatch" in notes and category == "fundamental":
        return "financial_report_value_mismatch"
    if "body_value_mismatch" in notes:
        return "external_claim_value_mismatch"
    if "value_mismatch" in notes:
        return "derived_financial_value_mismatch"
    return "unsupported_or_mismatched_evidence"


def agent_for(category: str) -> str:
    return {
        "technical": "Technical",
        "fundamental": "Fundamental",
        "news_events": "News",
        "body": "Strategy",
    }.get(category, "Unknown")


def root_cause_attributes(cause: str, issue_class: str, affects_conclusion: bool) -> tuple[str, str, str]:
    upstream = cause in {"frozen_news_snapshot_missing_article", "filing_text_snapshot_missing"}
    ownership = "upstream_data_limit" if upstream else "system_issue"
    if issue_class == "numerical_incorrect" or (issue_class == "factual_incorrect" and affects_conclusion):
        return "critical", ownership, "P0"
    if issue_class in {"factual_incorrect", "unsupported_claim", "completeness_gap"}:
        return "high" if affects_conclusion or issue_class != "unsupported_claim" else "medium", ownership, "P1"
    return "low", ownership, "P2"


def build_root_cause_matrix(root: Path, tables: dict[str, list[dict[str, str]]]) -> list[dict[str, Any]]:
    inventory = {row["item_id"]: row for row in tables["inventory"]}
    issue_filters = {
        "factual_incorrect": ("accuracy_label", "incorrect"),
        "numerical_incorrect": ("numerical_label", "incorrect"),
        "unsupported_claim": ("support_label", "unsupported"),
    }
    grouped: dict[tuple[Any, ...], dict[str, Any]] = {}
    conclusion_tokens = ("看多", "看空", "风险", "催化", "综合判断", "投资建议", "后续需要关注")
    for issue_class, (field, target) in issue_filters.items():
        unique_issues: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
        for row in tables["fact"]:
            if row.get(field) != target:
                continue
            unique = row.get("unique_claim_key") or row.get("claim_id") or row.get("agent_claim")
            unique_issues[(row["ticker"], unique)].append(row)

        for (_, unique), occurrences in unique_issues.items():
            cause_priority = {
                "financial_report_value_mismatch": 0,
                "derived_financial_value_mismatch": 1,
                "news_claim_scope_or_value_mismatch": 2,
                "external_claim_value_mismatch": 3,
                "frozen_news_snapshot_missing_article": 4,
                "filing_text_snapshot_missing": 5,
                "unmatched_body_claim_no_registry": 6,
                "unsupported_or_mismatched_evidence": 7,
            }
            classified = [
                (root_cause(row.get("notes", ""), row.get("category", "")), row)
                for row in occurrences
            ]
            cause, representative = min(classified, key=lambda item: cause_priority.get(item[0], 99))
            sections = [inventory.get(row.get("occurrence_id", ""), {}).get("section", "") for row in occurrences]
            affects_conclusion = any(any(token in section for token in conclusion_tokens) for section in sections)
            key = (
                issue_class, representative["ticker"], agent_for(representative.get("category", "")),
                representative.get("claim_type", ""), cause, affects_conclusion,
            )
            record = grouped.setdefault(key, {
                "issue_class": issue_class,
                "ticker": representative["ticker"],
                "agent": agent_for(representative.get("category", "")),
                "claim_type": representative.get("claim_type", ""),
                "root_cause": cause,
                "unique_claim_keys": set(),
                "occurrence_count": 0,
                "affects_numerical": False,
                "affects_final_investment_conclusion": affects_conclusion,
                "example_claim_ids": set(),
            })
            record["unique_claim_keys"].add(unique)
            record["occurrence_count"] += len(occurrences)
            record["affects_numerical"] = record["affects_numerical"] or any(
                row.get("numerical_label") == "incorrect" for row in occurrences
            )
            record["example_claim_ids"].update(
                row["claim_id"] for row in occurrences if row.get("claim_id")
            )

    rows: list[dict[str, Any]] = []
    for key, record in grouped.items():
        severity, ownership, priority = root_cause_attributes(
            record["root_cause"], record["issue_class"], record["affects_final_investment_conclusion"]
        )
        rows.append({
            "ticker": record["ticker"],
            "agent": record["agent"],
            "claim_type": record["claim_type"],
            "issue_class": record["issue_class"],
            "root_cause": record["root_cause"],
            "unique_claim_count": len(record["unique_claim_keys"]),
            "frequency_occurrences": record["occurrence_count"],
            "severity": severity,
            "affects_numerical": record["affects_numerical"],
            "affects_final_investment_conclusion": record["affects_final_investment_conclusion"],
            "ownership": ownership,
            "recommended_priority": priority,
            "example_claim_ids": ";".join(sorted(record["example_claim_ids"])),
        })

    report_rows = {row["ticker"]: row for row in tables["report"]}
    for ticker, row in report_rows.items():
        if to_float(row["completeness_score"]) < 100:
            severity, ownership, priority = root_cause_attributes("missing_final_synthesis", "completeness_gap", True)
            rows.append({
                "ticker": ticker, "agent": "Strategy", "claim_type": "final_synthesis",
                "issue_class": "completeness_gap", "root_cause": "missing_final_synthesis",
                "unique_claim_count": 1, "frequency_occurrences": 1, "severity": severity,
                "affects_numerical": False, "affects_final_investment_conclusion": True,
                "ownership": ownership, "recommended_priority": priority, "example_claim_ids": "",
            })

    forbidden = re.compile(r"十亿美元|百万美元|\bbillion\b|\bmillion\b", re.IGNORECASE)
    for report_path in sorted((root / "reports").glob("*.md")):
        matches = forbidden.findall(report_path.read_text(encoding="utf-8"))
        if matches:
            ticker = report_path.stem.removesuffix("_report")
            rows.append({
                "ticker": ticker, "agent": "Fundamental/Strategy", "claim_type": "display_format",
                "issue_class": "format_violation", "root_cause": "forbidden_amount_abbreviation",
                "unique_claim_count": len(matches), "frequency_occurrences": len(matches), "severity": "low",
                "affects_numerical": False, "affects_final_investment_conclusion": False,
                "ownership": "system_issue", "recommended_priority": "P2", "example_claim_ids": "",
            })
    return sorted(rows, key=lambda row: (row["recommended_priority"], row["ticker"], row["issue_class"], row["root_cause"]))


def finite(value: Any) -> Any:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: finite(value) for key, value in row.items()} for row in rows]


def methodology_text(input_root: Path, output_root: Path, bootstrap_samples: int, seed: int) -> str:
    return f"""# Phase 6C Confidence Calibration Methodology

## Frozen input

- Input: `{input_root.resolve()}`
- Production commit: `{EXPECTED_COMMIT}`
- Unit of analysis: one frozen report per ticker, n=19.
- No Agent, market, SEC, or news calls are made.
- The input SHA-256 manifest is verified before and after analysis.
- All derived files are written only to `{output_root.resolve()}`.

## Pre-registered Bad Case rule

A report is a Bad Case if any condition is true. These thresholds are copied from the Phase 6C request and are not tuned to the observed results:

- Factual Accuracy < 95%
- Numerical Accuracy < 95%
- Unsupported Claim Rate > 10%
- Citation Coverage < 95%
- Completeness < 100%
- Any numerical incorrect claim
- Any undisclosed major data limitation

An undisclosed major data limitation is operationalized as: a structured major limitation exists (historical fallback, partial/failed Agent, or unavailable news evidence), while the claim inventory contains no operational disclosure for that ticker. This deterministic proxy is reported separately and does not reinterpret report prose.

## Correlation analysis

- Spearman rank correlation is primary; Pearson correlation is supplementary.
- Pairwise bootstrap: {bootstrap_samples:,} resamples with replacement, deterministic seed {seed}.
- 95% percentile intervals are reported for both correlations.
- Influence is checked by leave-one-ticker-out Spearman correlation. A metric is marked extreme-value-sensitive if any omission flips the sign or changes rho by at least 0.20.
- The sample is only 19 reports. Intervals are descriptive; no statistical-significance or causal claim is made.

## Confidence threshold analysis

- Low: 50–59, Medium: 60–69, High: 70–79.
- Threshold sensitivity predicts Bad Case when Confidence is strictly below 55, 60, 65, or 70.
- Precision and recall treat a pre-registered Bad Case as the positive class.

## Separate quality dimensions

No opaque composite quality score is used.

1. Verified Content Quality is represented by factual accuracy, numerical accuracy, unsupported rate, citation accuracy/coverage, and completeness.
2. Current Research Usability is represented by freshness, latest-period alignment, historical fallback, Agent status, and limited-depth news evidence.

These dimensions are intentionally kept separate so a report such as TSM can be factually correct while still being less useful for a current investment decision.

## Root-cause accounting

- Incorrect, numerical-incorrect, and unsupported issues are deduplicated by the frozen evaluator's `unique_claim_key`; occurrence counts are also retained.
- Root-cause categories are derived only from frozen evaluator notes, claim categories, sections, and evidence metadata.
- A single claim can appear in more than one issue class, such as both incorrect and numerical-incorrect. Root-cause rows therefore reconcile within each issue class, not across overlapping issue classes.
- Completeness and amount-format findings are added from frozen report-level results and deterministic report scans.
"""


def main() -> None:
    args = parse_args()
    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    if input_root == output_root or input_root in output_root.parents:
        raise RuntimeError("Output directory must not be inside the frozen input directory")
    output_root.mkdir(parents=True, exist_ok=False)

    started = datetime.now(timezone.utc).isoformat()
    integrity_before, tables = validate_inputs(input_root)
    confidence_rows = build_confidence_rows(tables)
    bad_cases = build_bad_cases(confidence_rows)
    correlations = build_correlations(confidence_rows, args.bootstrap_samples, args.seed)
    buckets = build_buckets(confidence_rows, bad_cases)
    bad_map = {row["ticker"]: row["is_bad_case"] for row in bad_cases}
    thresholds = [threshold_row(confidence_rows, bad_map, threshold) for threshold in (55, 60, 65, 70)]
    mismatches = build_mismatches(confidence_rows, bad_cases)
    root_causes = build_root_cause_matrix(input_root, tables)

    # Reconcile the frozen headline issue counts at unique-claim grain.
    root_reconciliation = {}
    for issue_class, expected in {
        "factual_incorrect": 19,
        "numerical_incorrect": 8,
        "unsupported_claim": 97,
    }.items():
        actual = sum(row["unique_claim_count"] for row in root_causes if row["issue_class"] == issue_class)
        root_reconciliation[issue_class] = {"expected": expected, "actual": actual, "matches": actual == expected}
        if actual != expected:
            raise RuntimeError(f"Root-cause reconciliation failed for {issue_class}: {actual} != {expected}")

    confidence_fields = list(confidence_rows[0])
    write_csv(output_root / "confidence_vs_quality.csv", clean_rows(confidence_rows), confidence_fields)
    write_csv(output_root / "confidence_buckets.csv", clean_rows(buckets), list(buckets[0]))
    write_csv(output_root / "bad_case_classification.csv", clean_rows(bad_cases), list(bad_cases[0]))
    write_csv(output_root / "threshold_sensitivity.csv", clean_rows(thresholds), list(thresholds[0]))
    write_csv(output_root / "mismatch_cases.csv", clean_rows(mismatches), list(mismatches[0]))
    write_csv(output_root / "root_cause_matrix.csv", clean_rows(root_causes), list(root_causes[0]))

    methodology = methodology_text(input_root, output_root, args.bootstrap_samples, args.seed)
    (output_root / "methodology.md").write_text(methodology, encoding="utf-8")

    low = [row for row in bad_cases if row["confidence_bucket"] == "Low"]
    high = [row for row in bad_cases if row["confidence_bucket"] == "High"]
    threshold_60 = next(row for row in thresholds if row["low_confidence_threshold_exclusive"] == 60)
    strongest = max(correlations, key=lambda row: abs(row["spearman_rho"]))
    bad_tickers = [row["ticker"] for row in bad_cases if row["is_bad_case"]]
    summary = {
        "phase": "6C",
        "baseline_version": "v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_started_at_utc": started,
        "input_integrity_before": integrity_before,
        "bad_case_thresholds": BAD_CASE_THRESHOLDS,
        "bootstrap": {"samples": args.bootstrap_samples, "seed": args.seed, "method": "paired percentile bootstrap"},
        "correlations": clean_rows(correlations),
        "strongest_absolute_spearman_relationship": clean_rows([strongest])[0],
        "bad_case_summary": {
            "bad_case_count": len(bad_tickers),
            "bad_case_tickers": bad_tickers,
            "non_bad_case_count": len(EXPECTED_TICKERS) - len(bad_tickers),
            "low_confidence_bad_case_count": sum(int(row["is_bad_case"]) for row in low),
            "low_confidence_total": len(low),
            "low_confidence_bad_case_pct": 100.0 * safe_ratio(sum(int(row["is_bad_case"]) for row in low), len(low)),
            "high_confidence_bad_case_count": sum(int(row["is_bad_case"]) for row in high),
            "high_confidence_total": len(high),
            "high_confidence_bad_case_pct": 100.0 * safe_ratio(sum(int(row["is_bad_case"]) for row in high), len(high)),
            "threshold_60_confusion_matrix": threshold_60,
        },
        "confidence_buckets": clean_rows(buckets),
        "threshold_sensitivity": clean_rows(thresholds),
        "mismatch_case_count": len(mismatches),
        "root_cause_reconciliation": root_reconciliation,
        "quality_dimensions": {
            "verified_content_quality": [
                "factual_accuracy_pct", "numerical_accuracy_pct", "unsupported_claim_rate_pct",
                "citation_accuracy_pct", "citation_coverage_pct", "completeness_score",
            ],
            "current_research_usability": [
                "fundamental_freshness_status", "latest_period_alignment_status", "historical_fallback",
                "agent_success_count", "agent_partial_count", "headline_truncated_share_pct",
            ],
            "composite_ground_truth_used": False,
        },
        "files": [
            "calibration_summary.json", "confidence_vs_quality.csv", "confidence_buckets.csv",
            "bad_case_classification.csv", "threshold_sensitivity.csv", "mismatch_cases.csv",
            "root_cause_matrix.csv", "methodology.md",
        ],
    }
    integrity_after = verify_sha_manifest(input_root)
    if integrity_after != integrity_before["sha256"]:
        raise RuntimeError("Frozen input manifest verification changed during analysis")
    summary["input_integrity_after"] = integrity_after
    summary["frozen_input_unchanged"] = True
    write_json(output_root / "calibration_summary.json", summary)


if __name__ == "__main__":
    main()

import argparse
import csv
import json
import re
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

from tools.citation_registry import citation_summary, save_source_registry


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "smif_equity_holdings.csv"
)

DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "evaluation"
    / "batch_results.csv"
)

RESULT_FIELDS = [
    "ticker",
    "sector",
    "portfolio_weight_pct",
    "test_date",
    "agent_version",
    "model",
    "start_time_utc",
    "end_time_utc",
    "latency_seconds",
    "technical_agent_success",
    "fundamental_agent_success",
    "news_agent_success",
    "strategy_agent_success",
    "technical_agent_status",
    "fundamental_agent_status",
    "news_agent_status",
    "strategy_agent_status",
    "technical_evidence_path",
    "technical_price_date",
    "technical_price_basis",
    "technical_reproducible",
    "citation_required_claims",
    "cited_claims",
    "citation_coverage_pct",
    "citation_integrity",
    "source_registry_path",
    "confidence_score",
    "status",
    "http_status",
    "error_type",
    "error_message",
    "raw_output_path",
]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def has_any(report: str, markers: list[str]) -> bool:
    return any(marker in report for marker in markers)


def detect_sections(report: str) -> dict[str, bool]:
    return {
        "technical_agent_success": has_any(
            report,
            [
                "技术面摘要",
                "技术面分析",
                "技术面结论",
            ],
        ),
        "fundamental_agent_success": has_any(
            report,
            [
                "基本面摘要",
                "基本面分析",
                "基本面结论",
            ],
        ),
        "news_agent_success": has_any(
            report,
            [
                "消息面摘要",
                "消息面分析",
                "消息面结论",
            ],
        ),
        "strategy_agent_success": has_any(
            report,
            [
                "综合判断",
                "综合结论",
                "看多因素",
                "看空与风险因素",
            ],
        ),
    }


def extract_agent_statuses(
    response_data: dict,
    report: str,
) -> tuple[dict[str, str], str]:
    structured = response_data.get("agent_statuses")
    roles = ("technical", "fundamental", "news", "strategy")

    if isinstance(structured, dict):
        statuses = {}

        for role in roles:
            detail = structured.get(role)
            value = (
                detail.get("status")
                if isinstance(detail, dict)
                else None
            )
            statuses[role] = (
                value
                if value in {"success", "partial", "failed"}
                else "failed"
            )

        return statuses, "structured"

    sections = detect_sections(report)
    statuses = {
        role: (
            "success"
            if sections[f"{role}_agent_success"]
            else "failed"
        )
        for role in roles
    }

    return statuses, "legacy_section_fallback"


def overall_status(agent_statuses: dict[str, str]) -> str:
    values = list(agent_statuses.values())

    if values and all(value == "success" for value in values):
        return "success"

    if any(value in {"success", "partial"} for value in values):
        return "partial"

    return "failed"


def _portable_path(path: Path) -> str:
    resolved = path.resolve()

    try:
        return str(resolved.relative_to(PROJECT_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def save_technical_evidence(
    response_data: dict,
    ticker: str,
    raw_directory: Path,
) -> dict[str, str | bool]:
    result: dict[str, str | bool] = {
        "technical_evidence_path": "",
        "technical_price_date": "",
        "technical_price_basis": "",
        "technical_reproducible": "",
    }
    evidence = response_data.get("technical_evidence")

    if not isinstance(evidence, dict):
        return result

    result["technical_price_date"] = str(
        evidence.get("last_trading_date") or ""
    )
    price_basis = evidence.get("price_basis")

    if isinstance(price_basis, dict):
        column = str(price_basis.get("column") or "")
        auto_adjust = price_basis.get("auto_adjust")
        result["technical_price_basis"] = (
            f"{column}; auto_adjust={str(auto_adjust).lower()}"
            if column and isinstance(auto_adjust, bool)
            else column
        )
    elif price_basis:
        result["technical_price_basis"] = str(price_basis)

    result["technical_reproducible"] = bool(
        evidence.get("reproducible")
    )
    evidence_directory = raw_directory / "technical_evidence"
    evidence_directory.mkdir(parents=True, exist_ok=True)
    safe_ticker = ticker.replace("/", "_")
    destination = evidence_directory / f"{safe_ticker}.json"
    source_value = evidence.get("evidence_path")

    if source_value:
        source = Path(str(source_value)).expanduser()

        if not source.is_file():
            result["technical_reproducible"] = False
            return result

        # Parsing first ensures the copied artifact is valid JSON evidence.
        json.loads(source.read_text(encoding="utf-8"))

        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
    elif isinstance(evidence.get("raw_bars"), list):
        destination.write_text(
            json.dumps(
                evidence,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        result["technical_reproducible"] = False
        return result

    result["technical_evidence_path"] = _portable_path(destination)
    return result


def save_citation_registry(
    response_data: dict,
    report: str,
    ticker: str,
    raw_directory: Path,
) -> dict[str, str | int | float | bool]:
    empty: dict[str, str | int | float | bool] = {
        "citation_required_claims": "",
        "cited_claims": "",
        "citation_coverage_pct": "",
        "citation_integrity": False,
        "source_registry_path": "",
    }
    citations = response_data.get("citations")
    registry = response_data.get("source_registry")
    if not isinstance(citations, list) or not isinstance(registry, list):
        return empty

    clean_citations = [item for item in citations if isinstance(item, dict)]
    clean_registry = [item for item in registry if isinstance(item, dict)]
    summary = citation_summary(clean_citations, report, clean_registry)
    directory = raw_directory / "source_registry"
    destination = directory / f"{ticker.replace('/', '_')}.json"
    save_source_registry(
        destination,
        citations=clean_citations,
        source_registry=clean_registry,
        summary=summary,
    )
    return {
        "citation_required_claims": summary["citation_required_claims"],
        "cited_claims": summary["cited_claims"],
        "citation_coverage_pct": summary["citation_coverage_pct"],
        "citation_integrity": summary["citation_integrity"],
        "source_registry_path": _portable_path(destination),
    }


def extract_confidence_score(report: str) -> int | None:
    cleaned = report.replace("*", "")

    patterns = [
        r"最终分数\s*[:：]\s*(\d{1,3})",
        r"最终分数[^0-9]{0,20}(\d{1,3})\s*/\s*100",
        r"综合置信度评分[^0-9]{0,30}(\d{1,3})",
    ]

    for pattern in patterns:
        match = re.search(pattern, cleaned)

        if match:
            score = int(match.group(1))

            if 0 <= score <= 100:
                return score

    return None


def load_holdings(path: Path) -> list[dict]:
    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    return [
        row
        for row in rows
        if row.get("in_scope", "").strip().lower()
        == "true"
    ]


def initialize_results_file(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )
        writer.writeheader()


def append_result(path: Path, row: dict) -> None:
    with path.open(
        "a",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=RESULT_FIELDS,
        )
        writer.writerow(row)

def load_completed_tickers(path: Path) -> set[str]:
    if not path.exists() or path.stat().st_size == 0:
        return set()

    with path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        return {
            row.get("ticker", "").strip().upper()
            for row in csv.DictReader(file)
            if row.get("status", "").strip().lower() == "success"
        }

def analyze_one(
    client: httpx.Client,
    base_url: str,
    holding: dict,
    raw_directory: Path,
    agent_version: str,
    model: str,
) -> dict:
    ticker = holding["ticker"].strip().upper()
    started_at = utc_now()
    started_clock = time.perf_counter()

    report = ""
    http_status = ""
    error_type = ""
    error_message = ""

    section_status = {
        "technical_agent_success": False,
        "fundamental_agent_success": False,
        "news_agent_success": False,
        "strategy_agent_success": False,
    }
    agent_statuses = {
        "technical": "failed",
        "fundamental": "failed",
        "news": "failed",
        "strategy": "failed",
    }

    confidence_score = None
    status = "failed"
    evidence_fields: dict[str, str | bool] = {
        "technical_evidence_path": "",
        "technical_price_date": "",
        "technical_price_basis": "",
        "technical_reproducible": "",
    }
    citation_fields: dict[str, str | int | float | bool] = {
        "citation_required_claims": "",
        "cited_claims": "",
        "citation_coverage_pct": "",
        "citation_integrity": False,
        "source_registry_path": "",
    }

    try:
        response = client.post(
            f"{base_url}/analyze",
            json={"symbol": ticker},
        )

        http_status = response.status_code
        response.raise_for_status()

        data = response.json()
        report = str(data.get("report") or "").strip()

        if not report:
            raise ValueError(
                "接口成功，但 report 为空"
            )

        agent_statuses, status_source = extract_agent_statuses(
            data,
            report,
        )
        section_status = {
            f"{role}_agent_success": value == "success"
            for role, value in agent_statuses.items()
        }
        confidence_score = extract_confidence_score(
            report
        )

        incomplete_agents = [
            f"{role}={value}"
            for role, value in agent_statuses.items()
            if value != "success"
        ]

        status = overall_status(agent_statuses)
        evidence_fields = save_technical_evidence(
            data,
            ticker,
            raw_directory,
        )
        citation_fields = save_citation_registry(
            data,
            report,
            ticker,
            raw_directory,
        )

        if incomplete_agents:
            error_type = "incomplete_agent_data"
            error_message = (
                f"状态来源={status_source}；"
                "未完整成功的 Agent："
                + ", ".join(incomplete_agents)
            )

    except httpx.TimeoutException as error:
        error_type = "timeout"
        error_message = str(error)

    except httpx.HTTPStatusError as error:
        error_type = "http_error"
        error_message = (
            f"{error}; "
            f"response={error.response.text[:500]}"
        )

    except httpx.RequestError as error:
        error_type = "request_error"
        error_message = str(error)

    except Exception as error:
        error_type = type(error).__name__
        error_message = str(error)

    ended_at = utc_now()
    latency_seconds = round(
        time.perf_counter() - started_clock,
        2,
    )

    raw_path = raw_directory / f"{ticker}.md"

    raw_path.write_text(
        report
        if report
        else (
            f"# {ticker} 分析失败\n\n"
            f"- error_type: {error_type}\n"
            f"- error_message: {error_message}\n"
        ),
        encoding="utf-8",
    )

    return {
        "ticker": ticker,
        "sector": holding.get("sector", ""),
        "portfolio_weight_pct": holding.get(
            "portfolio_weight_pct",
            "",
        ),
        "test_date": started_at.date().isoformat(),
        "agent_version": agent_version,
        "model": model,
        "start_time_utc": started_at.isoformat(),
        "end_time_utc": ended_at.isoformat(),
        "latency_seconds": latency_seconds,
        **section_status,
        "technical_agent_status": agent_statuses["technical"],
        "fundamental_agent_status": agent_statuses["fundamental"],
        "news_agent_status": agent_statuses["news"],
        "strategy_agent_status": agent_statuses["strategy"],
        **evidence_fields,
        **citation_fields,
        "confidence_score": (
            confidence_score
            if confidence_score is not None
            else ""
        ),
        "status": status,
        "http_status": http_status,
        "error_type": error_type,
        "error_message": error_message[:1000],
        "raw_output_path": str(
            raw_path.relative_to(PROJECT_ROOT)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "对 SMIF Equity Holdings "
            "运行 US Stock Agent 批量评测"
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="只测试前 N 只；0 表示全部",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=180.0,
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--agent-version",
        default="v1.0",
    )
    parser.add_argument(
        "--model",
        default="gpt-5.6-luna",
    )

    args = parser.parse_args()

    holdings = load_holdings(args.input)

    if args.limit > 0:
        holdings = holdings[: args.limit]

    run_id = utc_now().strftime(
        "%Y%m%dT%H%M%SZ"
    )

    raw_directory = (
        PROJECT_ROOT
        / "evaluation"
        / "raw_outputs"
        / run_id
    )
    raw_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    completed_tickers = load_completed_tickers(args.output)

    if args.output.exists():
        with args.output.open(
            "r",
            encoding="utf-8-sig",
            newline="",
        ) as file:
            existing_rows = list(csv.DictReader(file))

        initialize_results_file(args.output)

        for existing_row in existing_rows:
            if existing_row.get("status", "").strip().lower() == "success":
                append_result(args.output, existing_row)
    else:
        initialize_results_file(args.output)

    if completed_tickers:
        print(
            f"发现 {len(completed_tickers)} 个已完成结果，"
            "将自动跳过。"
        )
        holdings = [
            holding
            for holding in holdings
            if holding["ticker"].strip().upper()
            not in completed_tickers
        ]
    with httpx.Client(
        timeout=args.timeout
    ) as client:
        health_response = client.get(
            f"{args.base_url}/health"
        )
        health_response.raise_for_status()

        total = len(holdings)
        successful = 0

        for index, holding in enumerate(
            holdings,
            start=1,
        ):
            ticker = holding["ticker"].strip()

            print(
                f"[{index}/{total}] "
                f"正在分析 {ticker}..."
            )

            result = analyze_one(
                client=client,
                base_url=args.base_url,
                holding=holding,
                raw_directory=raw_directory,
                agent_version=args.agent_version,
                model=args.model,
            )

            append_result(
                args.output,
                result,
            )

            if result["status"] == "success":
                successful += 1

            print(
                f"  状态：{result['status']}，"
                f"耗时：{result['latency_seconds']} 秒"
            )

            if index < total:
                time.sleep(args.delay)

    overall_total = len(completed_tickers) + len(holdings)
    overall_successful = len(completed_tickers) + successful
    completion_rate = (
        overall_successful / overall_total * 100
        if overall_total
        else 0
    )

    print()
    print("Batch Evaluation 完成")
    print(f"测试数量：{overall_total}")
    print(f"完整成功：{overall_successful}")
    print(
        f"Completion Rate："
        f"{completion_rate:.2f}%"
    )
    print(f"结果文件：{args.output}")
    print(f"原始报告目录：{raw_directory}")


if __name__ == "__main__":
    main()

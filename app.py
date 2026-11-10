import asyncio
import logging
import os
import sys
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Literal

from agents import Runner
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator


project_directory = Path(__file__).resolve().parent
agent_directory = project_directory / "agents"
static_directory = project_directory / "static"
prototype_path = project_directory / "parallax_prototype.html"

sys.path.insert(0, str(agent_directory))

from strategist_agent import strategist_agent
from tools.agent_status import (
    extract_top_level_citations,
    extract_top_level_fundamental_evidence,
    extract_top_level_fundamental_freshness,
    extract_top_level_statuses,
    extract_top_level_technical_evidence,
    warnings_from_statuses,
)
from tools.citation_registry import (
    citations_for_language,
    finalize_cited_report,
)
from tools.financial_amount import normalize_financial_amount_display
from tools.financial_freshness import freshness_banner
from tools.internal_provenance import (
    build_internal_provenance,
    internal_traceability_summary,
)
from tools.output_language import assert_english_output_contract


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

analysis_rate_limit = max(
    1,
    int(os.getenv("ANALYSIS_RATE_LIMIT_PER_HOUR", "5")),
)
analysis_slots = asyncio.Semaphore(
    max(1, int(os.getenv("MAX_CONCURRENT_ANALYSES", "2")))
)
request_history: dict[str, deque[float]] = defaultdict(deque)

app = FastAPI(
    title="Parallax Research Multi-Agent Equity Intelligence",
    version="1.0.0",
)

app.mount(
    "/static",
    StaticFiles(directory=str(static_directory)),
    name="static",
)


class AnalysisRequest(BaseModel):
    symbol: str
    language: Literal["zh", "en"] = "zh"

    @field_validator("symbol")
    @classmethod
    def validate_symbol(cls, value: str) -> str:
        symbol = value.strip().upper()

        if not symbol:
            raise ValueError("Ticker symbol is required.")

        if not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("Ticker symbol format is invalid.")

        if len(symbol) > 10:
            raise ValueError("Ticker symbol is too long.")

        return symbol


class AnalysisResponse(BaseModel):
    symbol: str
    report: str
    agent_statuses: dict[str, dict[str, Any]]
    warnings: list[str]
    technical_evidence: dict[str, Any] | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    source_registry: list[dict[str, Any]] = Field(default_factory=list)
    citation_summary: dict[str, Any] = Field(default_factory=dict)
    fundamental_freshness: dict[str, Any] | None = None
    fundamental_evidence: dict[str, Any] | None = None
    freshness_summary: dict[str, Any] = Field(default_factory=dict)
    stale_claims: list[dict[str, Any]] = Field(default_factory=list)
    internal_provenance: list[dict[str, Any]] = Field(default_factory=list)
    internal_traceability_summary: dict[str, Any] = Field(default_factory=dict)


@app.get("/", include_in_schema=False)
def home_page() -> FileResponse:
    return FileResponse(str(prototype_path))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "us-stock-agent",
    }


async def analyze_stock(
    request: AnalysisRequest,
    client_ip: str | None = None,
) -> AnalysisResponse:
    symbol = request.symbol
    if request.language == "en":
        prompt = f"""
Analyze US-listed ticker {symbol}.

ENGLISH OUTPUT CONTRACT
- Call the Technical, Fundamental, and News tools and instruct each one to
  return professional English only.
- Return every user-facing heading, narrative, status explanation, freshness
  warning, evidence limitation, and fallback message in English.
- Do not output CJK characters.
- Preserve every number, date, ticker, currency, period, accession, URL, and
  [Tn]/[Fn]/[Nn] claim ID exactly.
- Generate English directly. Do not translate a completed report afterward.
- Use complete accounting amounts such as USD 127,577,000,000, JPY
  1,141,600,000,000, TWD 2,894,308,000,000, and USD 2.18 per share.
- Never convert currencies or abbreviate amounts as billion or million.

Compare the three specialist results and provide Technical Analysis,
Fundamental Analysis, News Analysis, Bull Factors, Bear Factors and Risks,
Catalysts and Watch Items, Overall Judgment, Data Limitations, a disclaimer,
and Sources and Evidence. Do not fabricate data. This is not investment advice.
"""
    else:
        prompt = f"""
请对美股 {symbol} 进行完整分析。

必须调用以下三个分析工具：
1. 技术面分析师
2. 基本面分析师
3. 消息面分析师

综合三方面结果，给出：
- 数据日期与来源
- 技术面摘要
- 基本面摘要
- 消息面摘要
- 正面因素
- 负面因素
- 综合判断
- 判断置信度
- 后续需要关注的指标或事件
- 数据限制

不得虚构数据，并明确说明这不是投资建议。

最终综合报告必须使用简体中文。股票代码、公司名、
数据源名称和财务单位须保持准确，不得因翻译改变数字、日期、
币种、链接或专有名词。
"""

    if client_ip is not None:
        now = time.monotonic()
        history = request_history[client_ip]

        while history and now - history[0] >= 3600:
            history.popleft()

        if len(history) >= analysis_rate_limit:
            raise HTTPException(
                status_code=429,
                detail=(
                    "请求过于频繁，请稍后再试"
                    if request.language == "zh"
                    else "Too many requests. Please try again later."
                ),
                headers={"Retry-After": "3600"},
            )

        history.append(now)

    try:
        async with analysis_slots:
            result = await Runner.run(strategist_agent, prompt)
        agent_statuses = extract_top_level_statuses(result)
        technical_evidence = extract_top_level_technical_evidence(
            result
        )
        raw_citations = extract_top_level_citations(result)
        citations = citations_for_language(
            raw_citations,
            request.language,
        )
        fundamental_freshness = (
            extract_top_level_fundamental_freshness(result)
        )
        fundamental_evidence = extract_top_level_fundamental_evidence(
            result
        )
        report, source_registry, citation_summary = (
            finalize_cited_report(
                str(result.final_output or ""),
                citations,
                language=request.language,
            )
        )
        report = freshness_banner(
            report,
            fundamental_freshness,
            language=request.language,
        )
        report = normalize_financial_amount_display(
            report,
            language=request.language,
        )
        freshness_summary = (
            fundamental_freshness.get("summary", {})
            if isinstance(fundamental_freshness, dict)
            else {}
        )
        stale_claims = [
            claim
            for claim in citations
            if claim.get("agent") == "fundamental"
            and claim.get("freshness_status")
            in {"caution", "stale", "unknown"}
        ]
        internal_provenance = build_internal_provenance(
            result,
            agent_statuses=agent_statuses,
            fundamental_freshness=fundamental_freshness,
            technical_evidence=technical_evidence,
            citations=citations,
        )

        response = AnalysisResponse(
            symbol=symbol,
            report=report,
            agent_statuses=agent_statuses,
            warnings=warnings_from_statuses(agent_statuses),
            technical_evidence=technical_evidence,
            citations=citations,
            source_registry=source_registry,
            citation_summary=citation_summary,
            fundamental_freshness=fundamental_freshness,
            fundamental_evidence=fundamental_evidence,
            freshness_summary=freshness_summary,
            stale_claims=stale_claims,
            internal_provenance=internal_provenance,
            internal_traceability_summary=internal_traceability_summary(
                internal_provenance
            ),
        )
        if request.language == "en":
            public_fields = response.model_dump(
                include={
                    "report",
                    "agent_statuses",
                    "warnings",
                    "citations",
                    "source_registry",
                    "freshness_summary",
                    "internal_provenance",
                }
            )
            assert_english_output_contract(public_fields)
        return response

    except Exception:
        logger.exception("Analysis failed for %s", symbol)

        raise HTTPException(
            status_code=500,
            detail=(
                "Analysis failed. Review the server logs for details."
                if request.language == "en"
                else "分析失败，请查看终端中的错误信息"
            ),
        )


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock_endpoint(
    request: AnalysisRequest,
    http_request: Request,
) -> AnalysisResponse:
    client_ip = (
        http_request.client.host
        if http_request.client is not None
        else None
    )
    return await analyze_stock(request, client_ip=client_ip)

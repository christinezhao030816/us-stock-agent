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
from tools.citation_registry import finalize_cited_report
from tools.financial_amount import normalize_financial_amount_display
from tools.financial_freshness import freshness_banner
from tools.internal_provenance import (
    build_internal_provenance,
    internal_traceability_summary,
)


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
    title="美股多 Agent 分析系统",
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
            raise ValueError("股票代码不能为空")

        if not symbol.replace(".", "").replace("-", "").isalnum():
            raise ValueError("股票代码格式不正确")

        if len(symbol) > 10:
            raise ValueError("股票代码过长")

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
    output_language = (
        "English" if request.language == "en" else "简体中文"
    )

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

最终综合报告必须使用{output_language}。股票代码、公司名、
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
        citations = extract_top_level_citations(result)
        fundamental_freshness = (
            extract_top_level_fundamental_freshness(result)
        )
        fundamental_evidence = extract_top_level_fundamental_evidence(
            result
        )
        internal_provenance = build_internal_provenance(
            result,
            agent_statuses=agent_statuses,
            fundamental_freshness=fundamental_freshness,
            technical_evidence=technical_evidence,
            citations=citations,
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
        report = normalize_financial_amount_display(report)
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

        return AnalysisResponse(
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

    except Exception:
        logger.exception("分析 %s 时发生错误", symbol)

        raise HTTPException(
            status_code=500,
            detail="分析失败，请查看终端中的错误信息",
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

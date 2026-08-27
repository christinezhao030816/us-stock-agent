import logging
import sys
from pathlib import Path

from agents import Runner
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, field_validator


project_directory = Path(__file__).resolve().parent
agent_directory = project_directory / "agents"
static_directory = project_directory / "static"

sys.path.insert(0, str(agent_directory))

from strategist_agent import strategist_agent


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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


@app.get("/", include_in_schema=False)
def home_page() -> FileResponse:
    return FileResponse(str(static_directory / "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health_check() -> dict:
    return {
        "status": "ok",
        "service": "us-stock-agent",
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_stock(request: AnalysisRequest) -> AnalysisResponse:
    symbol = request.symbol

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
"""

    try:
        result = await Runner.run(strategist_agent, prompt)

        return AnalysisResponse(
            symbol=symbol,
            report=result.final_output,
        )

    except Exception:
        logger.exception("分析 %s 时发生错误", symbol)

        raise HTTPException(
            status_code=500,
            detail="分析失败，请查看终端中的错误信息",
        )

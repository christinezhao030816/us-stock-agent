import json
import os
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch


dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda: None
os.environ["SEC_USER_AGENT"] = "unit-test@example.com"
os.environ["ALPACA_API_KEY"] = "unit-test-key"
os.environ["ALPACA_SECRET_KEY"] = "unit-test-secret"

previous_dotenv = sys.modules.get("dotenv")
sys.modules["dotenv"] = dotenv_stub
try:
    import app as application
finally:
    if previous_dotenv is None:
        sys.modules.pop("dotenv", None)
    else:
        sys.modules["dotenv"] = previous_dotenv


def _status() -> dict:
    return {
        "status": "success",
        "core_data_present": True,
        "missing_items": [],
        "errors": [],
        "data_sources": ["test"],
        "source_symbols": {},
    }


def _agent_output(name: str) -> SimpleNamespace:
    payload = {"agent_status": _status()}

    if name == "technical":
        payload["technical_evidence"] = {
            "evidence_path": "/tmp/AAPL.json",
            "last_trading_date": "2026-09-02",
            "reproducible": True,
        }

    return SimpleNamespace(
        type="tool_call_output_item",
        tool_origin=SimpleNamespace(agent_tool_name=f"{name}_analysis"),
        output=json.dumps(payload),
        call_id=f"call-{name}",
    )


class AnalyzeEndpointCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def test_analyze_keeps_report_and_adds_structured_statuses(self) -> None:
        run_result = SimpleNamespace(
            final_output="原有报告字段内容",
            new_items=[
                _agent_output("technical"),
                _agent_output("fundamental"),
                _agent_output("news"),
            ],
        )

        with patch.object(
            application.Runner,
            "run",
            new=AsyncMock(return_value=run_result),
        ):
            response = await application.analyze_stock(
                application.AnalysisRequest(symbol="AAPL")
            )

        self.assertEqual(response.report, "原有报告字段内容")
        self.assertEqual(response.symbol, "AAPL")
        self.assertEqual(
            response.agent_statuses["technical"]["status"],
            "success",
        )
        self.assertEqual(
            response.agent_statuses["strategy"]["status"],
            "success",
        )
        self.assertEqual(
            response.technical_evidence["evidence_path"],
            "/tmp/AAPL.json",
        )


if __name__ == "__main__":
    unittest.main()

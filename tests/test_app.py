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
    from tools.output_language import cjk_paths
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
    async def test_english_mode_has_no_cjk_in_user_visible_response(self) -> None:
        claim = {
            "claim_id": "F1",
            "agent": "fundamental",
            "claim_type": "fundamental_metric",
            "metric_name": "revenue",
            "claim_text": "Alphabet 收入 = 127,577,000,000 美元",
            "value": 127.577,
            "unit": "USD billion",
            "period_or_as_of_date": "2026-06-30",
            "source_name": "SEC Company Facts API",
            "source_type": "sec_company_facts",
            "source_url": "https://data.sec.gov/example.json",
            "source_date": "2026-08-01",
            "source_anchor": "facts.us-gaap.Revenues.units.USD",
            "evidence_path": "https://data.sec.gov/example.json",
            "citation_status": "supported",
            "company_or_ticker": "GOOGL",
            "freshness_status": "fresh",
        }
        outputs = [
            _agent_output("technical"),
            _agent_output("news"),
            SimpleNamespace(
                type="tool_call_output_item",
                tool_origin=SimpleNamespace(agent_tool_name="fundamental_analysis"),
                output=json.dumps({"agent_status": _status(), "citations": [claim]}),
                call_id="call-fundamental",
            ),
        ]
        run_result = SimpleNamespace(
            final_output=(
                "## Fundamental Analysis\n"
                "Revenue was USD 127,577,000,000 [F1].\n\n"
                "## Overall Judgment\nNeutral."
            ),
            new_items=outputs,
        )

        with patch.object(
            application.Runner,
            "run",
            new=AsyncMock(return_value=run_result),
        ) as runner:
            response = await application.analyze_stock(
                application.AnalysisRequest(symbol="GOOGL", language="en")
            )

        prompt = runner.await_args.args[1]
        self.assertIn("ENGLISH OUTPUT CONTRACT", prompt)
        self.assertEqual(
            response.citations[0]["claim_text"],
            "GOOGL revenue = USD 127,577,000,000",
        )
        visible = response.model_dump(
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
        self.assertEqual(cjk_paths(visible), [])

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

    async def test_analyze_returns_claim_level_citations_and_keeps_report(self) -> None:
        claim = {
            "claim_id": "F1",
            "agent": "fundamental",
            "claim_type": "fundamental_metric",
            "metric_name": "revenue",
            "claim_text": "Revenue = 100 USD billion",
            "value": 100,
            "unit": "USD billion",
            "period_or_as_of_date": "2026-06-30",
            "source_name": "SEC Company Facts API",
            "source_type": "sec_company_facts",
            "source_url": "https://data.sec.gov/example.json",
            "source_date": "2026-08-01",
            "source_anchor": "facts.us-gaap.Revenues.units.USD",
            "evidence_path": "https://data.sec.gov/example.json",
            "citation_status": "supported",
        }
        outputs = [
            _agent_output("technical"),
            _agent_output("news"),
            SimpleNamespace(
                type="tool_call_output_item",
                tool_origin=SimpleNamespace(agent_tool_name="fundamental_analysis"),
                output=json.dumps(
                    {
                        "agent_status": _status(),
                        "citations": [claim],
                        "fundamental_freshness": {
                            "analysis_as_of": "2026-09-10",
                            "facts": [],
                            "summary": {
                                "fundamental_freshness_status": "fresh",
                                "fresh_fact_count": 1,
                                "caution_fact_count": 0,
                                "stale_fact_count": 0,
                                "unknown_freshness_count": 0,
                                "current_analysis_usable": True,
                            },
                        },
                    }
                ),
                call_id="call-fundamental",
            ),
        ]
        run_result = SimpleNamespace(final_output="原报告依据[F1]", new_items=outputs)

        with patch.object(
            application.Runner,
            "run",
            new=AsyncMock(return_value=run_result),
        ):
            response = await application.analyze_stock(
                application.AnalysisRequest(symbol="AAPL")
            )

        self.assertIn("原报告依据[F1]", response.report)
        self.assertIn("来源与证据", response.report)
        self.assertEqual(response.citations[0]["claim_id"], "F1")
        self.assertEqual(response.citation_summary["cited_claims"], 1)
        self.assertTrue(response.citation_summary["citation_integrity"])
        self.assertEqual(
            response.freshness_summary["fundamental_freshness_status"],
            "fresh",
        )
        self.assertIn("财务数据新鲜度", response.report)


if __name__ == "__main__":
    unittest.main()

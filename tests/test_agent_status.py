import json
import unittest
from types import SimpleNamespace

from tools.agent_status import (
    build_specialist_envelope,
    collect_tool_results,
    evaluate_fundamental_status,
    evaluate_news_status,
    evaluate_technical_status,
    extract_top_level_statuses,
)


def _technical_metrics(**overrides) -> dict:
    metrics = {
        "data_end": "2026-09-02",
        "trading_days": 300,
        "current_price_usd": 500,
        "return_20d_percent": 1.2,
        "sma20_usd": 495,
        "sma50_usd": 490,
        "sma200_usd": 450,
        "rsi14": 55,
        "latest_volume": 1_000_000,
        "support_20d_usd": 470,
        "resistance_20d_usd": 510,
        "price_source": "Yahoo Finance via yfinance",
        "technical_evidence": {
            "reproducible": True,
            "reproducibility_missing": [],
            "evidence_path": "/tmp/test-technical-evidence.json",
        },
        "symbol_mapping": {
            "canonical": "BRK.B",
            "yahoo": "BRK-B",
        },
    }
    metrics.update(overrides)
    return metrics


def _fundamental_metrics(**overrides) -> dict:
    amount = {
        "value": 1,
        "currency": "USD",
        "scale": "billion",
        "period": {"start": None, "end": "2026-06-30"},
        "statement_scope": "test",
        "display_zh": "1 十亿美元",
    }
    metrics = {
        "company": "Berkshire Hathaway Inc.",
        "form": "10-Q",
        "income_period_end": "2026-06-30",
        "filing_date": "2026-08-03",
        "financial_amounts": {
            "revenue": amount,
            "net_income": amount,
            "operating_cash_flow": amount,
            "capital_expenditure": amount,
            "free_cash_flow": amount,
            "cash": amount,
            "total_debt": amount,
            "assets": amount,
            "equity": amount,
        },
        "source": "SEC Company Facts API",
        "symbol_mapping": {
            "canonical": "BRK.B",
            "sec": "BRK-B",
        },
    }
    metrics.update(overrides)
    return metrics


def _tool_output(tool_name: str, output: object) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_call_output_item",
        tool_origin=SimpleNamespace(agent_tool_name=tool_name),
        output=output,
        call_id=f"call-{tool_name}",
    )


class SpecialistStatusTests(unittest.TestCase):
    def test_nested_tool_output_uses_call_name_not_enclosing_agent(self) -> None:
        origin = SimpleNamespace(agent_tool_name="technical_analysis")
        call = SimpleNamespace(
            type="tool_call_item",
            tool_origin=origin,
            _resolved_tool_name="get_technical_metrics",
            tool_name="get_technical_metrics",
            raw_item={
                "name": "get_technical_metrics",
                "call_id": "call-1",
            },
            call_id="call-1",
        )
        output = SimpleNamespace(
            type="tool_call_output_item",
            tool_origin=origin,
            raw_item={"call_id": "call-1"},
            call_id="call-1",
            output=_technical_metrics(),
        )

        payloads, errors = collect_tool_results(
            SimpleNamespace(new_items=[call, output])
        )

        self.assertFalse(errors)
        self.assertEqual(
            payloads["get_technical_metrics"][0]["current_price_usd"],
            500,
        )

    def test_no_professional_data_is_failed(self) -> None:
        status = evaluate_technical_status({}, ["行情查询失败"])
        self.assertEqual(status["status"], "failed")
        self.assertFalse(status["core_data_present"])

    def test_partial_professional_data_is_partial(self) -> None:
        metrics = _technical_metrics(sma200_usd=None)
        status = evaluate_technical_status(
            {"get_technical_metrics": [metrics]},
            [],
        )
        self.assertEqual(status["status"], "partial")
        self.assertIn("sma200_usd", status["missing_items"])

    def test_complete_technical_data_is_success(self) -> None:
        status = evaluate_technical_status(
            {"get_technical_metrics": [_technical_metrics()]},
            [],
        )
        self.assertEqual(status["status"], "success")

    def test_complete_fundamental_data_is_success(self) -> None:
        status = evaluate_fundamental_status(
            {"get_fundamental_metrics": [_fundamental_metrics()]},
            [],
        )
        self.assertEqual(status["status"], "success")

    def test_missing_fundamental_metric_is_partial(self) -> None:
        metrics = _fundamental_metrics()
        metrics["financial_amounts"] = {
            **metrics["financial_amounts"],
            "total_debt": None,
        }
        status = evaluate_fundamental_status(
            {"get_fundamental_metrics": [metrics]},
            [],
        )
        self.assertEqual(status["status"], "partial")
        self.assertIn("total_debt", status["missing_items"])

    def test_empty_news_results_are_honest_partial(self) -> None:
        status = evaluate_news_status(
            {
                "get_stock_news": [
                    {
                        "article_count": 0,
                        "source": "Alpaca News API",
                    }
                ],
                "get_recent_sec_filing_texts": [
                    {"filing_count": 0, "source": "SEC EDGAR"}
                ],
            },
            [],
        )
        self.assertEqual(status["status"], "partial")
        self.assertFalse(status["core_data_present"])

    def test_complete_news_data_is_success(self) -> None:
        status = evaluate_news_status(
            {
                "get_stock_news": [
                    {
                        "article_count": 2,
                        "source": "Alpaca News API",
                    }
                ],
                "get_recent_sec_filing_texts": [
                    {"filing_count": 1, "source": "SEC EDGAR"}
                ],
            },
            [],
        )
        self.assertEqual(status["status"], "success")

    def test_strategy_is_partial_when_specialist_is_not_success(self) -> None:
        envelopes = []
        for role, value in (
            ("technical", "success"),
            ("fundamental", "success"),
            ("news", "partial"),
        ):
            envelopes.append(
                _tool_output(
                    f"{role}_analysis",
                    json.dumps(
                        {
                            "agent_status": {
                                "status": value,
                                "core_data_present": value == "success",
                                "missing_items": [],
                                "errors": [],
                                "data_sources": [],
                                "source_symbols": {},
                            }
                        }
                    ),
                )
            )

        result = SimpleNamespace(
            new_items=envelopes,
            final_output="完整报告",
        )
        statuses = extract_top_level_statuses(result)
        self.assertEqual(statuses["news"]["status"], "partial")
        self.assertEqual(statuses["strategy"]["status"], "partial")

    def test_specialist_envelope_uses_tool_data_not_report_heading(self) -> None:
        result = SimpleNamespace(
            new_items=[],
            final_output="## 技术面摘要\n看起来像完整报告",
        )
        envelope = json.loads(
            build_specialist_envelope("technical", result)
        )
        self.assertEqual(envelope["agent_status"]["status"], "failed")

    def test_technical_envelope_carries_evidence_metadata(self) -> None:
        result = SimpleNamespace(
            new_items=[
                _tool_output(
                    "get_technical_metrics",
                    _technical_metrics(),
                )
            ],
            final_output="技术面报告",
        )
        envelope = json.loads(
            build_specialist_envelope("technical", result)
        )

        self.assertEqual(envelope["agent_status"]["status"], "success")
        self.assertTrue(envelope["technical_evidence"]["reproducible"])
        self.assertEqual(
            envelope["technical_evidence"]["evidence_path"],
            "/tmp/test-technical-evidence.json",
        )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation import batch_runner


class _Response:
    status_code = 200

    def __init__(self, data: dict) -> None:
        self._data = data

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._data


class _Client:
    def __init__(self, data: dict) -> None:
        self._data = data

    def post(self, *args, **kwargs) -> _Response:
        return _Response(self._data)


def _status(value: str) -> dict:
    return {
        "status": value,
        "core_data_present": value == "success",
        "missing_items": [],
        "errors": [],
        "data_sources": [],
        "source_symbols": {},
    }


class BatchRunnerStatusTests(unittest.TestCase):
    def test_old_response_without_evidence_remains_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = batch_runner.save_technical_evidence(
                {},
                "AAPL",
                Path(temporary_directory),
            )

        self.assertEqual(result["technical_evidence_path"], "")
        self.assertEqual(result["technical_price_date"], "")
        self.assertEqual(result["technical_price_basis"], "")
        self.assertEqual(result["technical_reproducible"], "")

    def test_old_response_without_citations_is_not_reported_as_full_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            result = batch_runner.save_citation_registry(
                {},
                "旧报告",
                "AAPL",
                Path(temporary_directory),
            )
        self.assertEqual(result["citation_coverage_pct"], "")
        self.assertFalse(result["citation_integrity"])

    def test_structured_status_has_priority_over_report_sections(self) -> None:
        report = "技术面摘要 基本面摘要 消息面摘要 综合判断"
        statuses, source = batch_runner.extract_agent_statuses(
            {
                "agent_statuses": {
                    "technical": _status("failed"),
                    "fundamental": _status("partial"),
                    "news": _status("failed"),
                    "strategy": _status("partial"),
                }
            },
            report,
        )
        self.assertEqual(source, "structured")
        self.assertEqual(statuses["technical"], "failed")
        self.assertEqual(statuses["fundamental"], "partial")
        self.assertEqual(batch_runner.overall_status(statuses), "partial")

    def test_analyze_one_records_structured_agent_statuses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            raw_directory = root / "raw"
            raw_directory.mkdir()
            source_evidence = root / "source.json"
            source_evidence.write_text(
                json.dumps({"raw_bars": [{"date": "2026-09-02"}]}),
                encoding="utf-8",
            )
            response_data = {
                "symbol": "BRK.B",
                "report": "技术面摘要 基本面摘要[F1] 消息面摘要 综合判断",
                "agent_statuses": {
                    "technical": _status("success"),
                    "fundamental": _status("success"),
                    "news": _status("partial"),
                    "strategy": _status("partial"),
                },
                "warnings": ["news=partial"],
                "technical_evidence": {
                    "evidence_path": str(source_evidence),
                    "last_trading_date": "2026-09-02",
                    "price_basis": {
                        "column": "Close",
                        "auto_adjust": False,
                    },
                    "reproducible": True,
                },
                "citations": [
                    {
                        "claim_id": "F1",
                        "agent": "fundamental",
                        "claim_type": "fundamental_metric",
                        "metric_name": "revenue",
                        "claim_text": "Revenue = 100",
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
                ],
                "source_registry": [
                    {
                        "source_id": "SRC1",
                        "source_name": "SEC Company Facts API",
                        "source_type": "sec_company_facts",
                        "source_url": "https://data.sec.gov/example.json",
                        "source_date": "2026-08-01",
                        "source_anchors": ["facts.us-gaap.Revenues.units.USD"],
                        "claim_ids": ["F1"],
                    }
                ],
            }

            with patch.object(batch_runner, "PROJECT_ROOT", root):
                result = batch_runner.analyze_one(
                    client=_Client(response_data),
                    base_url="http://test",
                    holding={"ticker": "BRK.B", "sector": "Financials"},
                    raw_directory=raw_directory,
                    agent_version="test",
                    model="test",
                )

            saved_evidence = root / result["technical_evidence_path"]
            self.assertTrue(saved_evidence.is_file())
            self.assertEqual(
                json.loads(saved_evidence.read_text())["raw_bars"][0][
                    "date"
                ],
                "2026-09-02",
            )

        self.assertEqual(result["technical_agent_status"], "success")
        self.assertTrue(result["technical_agent_success"])
        self.assertEqual(result["news_agent_status"], "partial")
        self.assertFalse(result["news_agent_success"])
        self.assertEqual(result["strategy_agent_status"], "partial")
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["technical_price_date"], "2026-09-02")
        self.assertEqual(
            result["technical_price_basis"],
            "Close; auto_adjust=false",
        )
        self.assertTrue(result["technical_reproducible"])
        self.assertEqual(result["citation_required_claims"], 1)
        self.assertEqual(result["cited_claims"], 1)
        self.assertEqual(result["citation_coverage_pct"], 100.0)
        self.assertTrue(result["citation_integrity"])


if __name__ == "__main__":
    unittest.main()

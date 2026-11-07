import json
import tempfile
import unittest
from pathlib import Path

from tools.citation_registry import (
    build_agent_citations,
    build_fundamental_citations,
    build_news_citations,
    build_source_registry,
    build_technical_citations,
    citation_summary,
    finalize_cited_report,
    inherit_strategy_citations,
    repair_cited_financial_values,
    save_source_registry,
)


def _supported_claim(claim_id: str = "F1") -> dict:
    return {
        "claim_id": claim_id,
        "agent": "fundamental",
        "claim_type": "fundamental_metric",
        "metric_name": "revenue",
        "claim_text": "Revenue was USD 100 billion",
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


class CitationMappingTests(unittest.TestCase):
    def test_strategy_restatement_inherits_supported_claim_id(self) -> None:
        claim = _supported_claim("F1")
        report = "基本面摘要：Revenue was 100 billion USD。"
        repaired = inherit_strategy_citations(report, [claim])
        self.assertIn("[F1]", repaired)

    def test_system_score_does_not_inherit_external_citation(self) -> None:
        claim = {**_supported_claim("F1"), "value": 78, "claim_text": "score 78"}
        repaired = inherit_strategy_citations("系统评分为78分。", [claim])
        self.assertNotIn("[F1]", repaired)

    def test_cited_financial_mismatch_is_repaired_from_structured_claim(self) -> None:
        claim = {
            **_supported_claim("F4"),
            "value": 1826.18,
            "unit": "TWD billion",
        }
        repaired = repair_cited_financial_values(
            "经营现金流：919,770,000,000 新台币。[F4]", [claim]
        )
        self.assertIn("1,826,180,000,000 新台币", repaired)

    def test_phase7a_abbv_multi_marker_line_uses_exact_f2_value(self) -> None:
        citations = [
            {
                **_supported_claim("F2"),
                "metric_name": "net_income",
                "value": 4.218,
                "unit": "USD billion",
                "claim_text": "AbbVie net income = 4,218,000,000 美元",
            },
            {
                **_supported_claim("F6"),
                "metric_name": "free_cash_flow",
                "value": 13.98,
                "unit": "USD billion",
            },
            {
                **_supported_claim("F16"),
                "metric_name": "net_margin",
                "value": 0.2424,
                "unit": "ratio",
            },
        ]
        report = (
            "净利润为4,220,000,000 美元，净利润率为24.24%。"
            "[F2][F6][F16]"
        )

        repaired = repair_cited_financial_values(report, citations)

        self.assertIn("4,218,000,000 美元", repaired)
        self.assertNotIn("4,220,000,000 美元", repaired)
        self.assertIn("[F2][F6][F16]", repaired)

    def test_phase7a_cat_f5_uses_exact_registry_value(self) -> None:
        claim = {
            **_supported_claim("F5"),
            "metric_name": "capital_expenditure",
            "value": 2.012,
            "unit": "USD billion",
        }

        repaired = repair_cited_financial_values(
            "同期资本支出为2,010,000,000 美元。[F5]",
            [claim],
        )

        self.assertIn("2,012,000,000 美元", repaired)
        self.assertNotIn("2,010,000,000 美元", repaired)
    def test_fundamental_metric_keeps_sec_concept_mapping(self) -> None:
        record = {
            "concept": "Revenues",
            "unit": "USD",
            "start": "2026-01-01",
            "end": "2026-06-30",
            "filed": "2026-08-01",
        }
        metrics = {
            "company": "Example Inc.",
            "income_period_end": "2026-06-30",
            "filing_date": "2026-08-01",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
            "financial_amounts": {
                "revenue": {
                    "value": 100,
                    "display_zh": "100 十亿美元",
                    "period": {"end": "2026-06-30"},
                }
            },
            "source_records": {"revenue": record},
            "latest_applicable_filing": {
                "form": "10-Q",
                "filing_date": "2026-08-01",
                "period_end": "2026-06-30",
                "accession_number": "0000000001-26-000001",
            },
            "latest_filing_source_url": (
                "https://data.sec.gov/submissions/CIK0000000001.json"
            ),
            "fundamental_freshness": {
                "facts": [
                    {
                        "metric_name": "revenue",
                        "freshness_status": "fresh",
                        "period_age_days": 72,
                    }
                ]
            },
        }

        claims = build_fundamental_citations(metrics)
        revenue = next(item for item in claims if item["metric_name"] == "revenue")
        self.assertEqual(revenue["citation_status"], "supported")
        self.assertTrue(
            revenue["source_anchor"].startswith(
                "facts.us-gaap.Revenues.units.USD period=2026-01-01..2026-06-30"
            )
        )
        self.assertEqual(revenue["source_date"], "2026-08-01")
        self.assertEqual(revenue["freshness_status"], "fresh")
        self.assertEqual(revenue["freshness"]["period_age_days"], 72)
        latest = next(
            item for item in claims
            if item["metric_name"] == "latest_filing_period"
        )
        self.assertEqual(latest["citation_status"], "supported")
        self.assertEqual(latest["period_or_as_of_date"], "2026-06-30")
        self.assertEqual(latest["freshness_status"], "fresh")
        self.assertIn(
            "0000000001-26-000001",
            latest["source_anchor"],
        )

    def test_technical_metrics_point_to_p2_json_paths(self) -> None:
        keys = (
            "current_price_usd",
            "previous_close_usd",
            "daily_change_percent",
            "return_5d_percent",
            "return_20d_percent",
            "return_60d_percent",
            "sma20_usd",
            "sma50_usd",
            "sma200_usd",
            "rsi14",
            "latest_volume",
            "average_volume_20d",
            "average_volume_60d",
            "volume_ratio_to_20d",
            "support_20d_usd",
            "resistance_20d_usd",
            "support_60d_usd",
            "resistance_60d_usd",
            "distance_from_support_20d_percent",
            "distance_from_resistance_20d_percent",
            "high_52_week_usd",
            "low_52_week_usd",
            "distance_from_52_week_high_percent",
            "distance_from_52_week_low_percent",
        )
        metrics = {key: index + 1 for index, key in enumerate(keys)}
        metrics["technical_evidence"] = {
            "source": "Yahoo Finance via yfinance",
            "source_url": "https://finance.yahoo.com/quote/BRK-B/history/",
            "canonical_symbol": "BRK.B",
            "vendor_symbol": "BRK-B",
            "retrieved_at_utc": "2026-09-10T12:00:00+00:00",
            "last_trading_date": "2026-09-09",
            "evidence_path": "/tmp/BRK_B.json",
            "indicator_evidence": {key: {"formula": "fixed"} for key in keys},
        }

        claims = build_technical_citations(metrics)
        self.assertTrue(all(item["citation_status"] == "supported" for item in claims))
        self.assertEqual(claims[0]["evidence_path"], "/tmp/BRK_B.json")
        self.assertEqual(
            claims[0]["source_anchor"],
            "technical_evidence.indicator_evidence.current_price_usd",
        )

    def test_news_preserves_title_date_publisher_url_and_match(self) -> None:
        claims = build_news_citations(
            {
                "get_stock_news": [
                    {
                        "symbol": "AAPL",
                        "articles": [
                            {
                                "headline": "Apple announces product",
                                "summary": "Apple described the launch.",
                                "source": "Reuters",
                                "created_at": "2026-09-09T10:00:00Z",
                                "url": "https://example.com/apple",
                                "symbols": ["AAPL"],
                            }
                        ],
                    }
                ]
            }
        )
        self.assertEqual(claims[0]["citation_status"], "supported")
        self.assertEqual(claims[0]["source_anchor"], "Apple announces product")
        self.assertEqual(claims[0]["source_name"], "Reuters")
        self.assertEqual(claims[0]["source_url"], "https://example.com/apple")
        self.assertEqual(claims[0]["evidence_depth"], "fully_supported")
        self.assertFalse(claims[0]["claim_scope_limited"])

    def test_headline_only_is_supported_only_with_limited_scope(self) -> None:
        claims = build_news_citations(
            {
                "get_stock_news": [
                    {
                        "symbol": "AAPL",
                        "articles": [
                            {
                                "headline": "Apple announces product",
                                "summary": None,
                                "source": "Example",
                                "created_at": "2026-09-09T10:00:00Z",
                                "url": "https://example.com/apple",
                                "symbols": ["AAPL"],
                            }
                        ],
                    }
                ]
            }
        )
        self.assertEqual(claims[0]["citation_status"], "supported")
        self.assertEqual(claims[0]["evidence_depth"], "headline_only")
        self.assertIn("未读取完整正文", claims[0]["claim_text"])

    def test_news_without_url_or_description_is_not_supported(self) -> None:
        claims = build_news_citations(
            {
                "get_stock_news": [
                    {
                        "symbol": "AAPL",
                        "articles": [
                            {
                                "headline": "Headline only",
                                "summary": None,
                                "source": "Example",
                                "created_at": "2026-09-09T10:00:00Z",
                                "url": None,
                                "symbols": ["AAPL"],
                            }
                        ],
                    }
                ]
            }
        )
        self.assertNotEqual(claims[0]["citation_status"], "supported")

    def test_sec_filing_keeps_accession_description_and_url(self) -> None:
        claims = build_news_citations(
            {
                "get_recent_sec_filing_texts": [
                    {
                        "symbol": "BRK.B",
                        "filings": [
                            {
                                "form": "8-K",
                                "filing_date": "2026-08-11",
                                "report_date": "2026-08-11",
                                "accession_number": "0001-26-000001",
                                "item_descriptions": ["Results of Operations"],
                                "filing_index_url": "https://www.sec.gov/example-index.html",
                                "primary_document": {
                                    "text": "Filed results text",
                                    "error": None,
                                },
                            }
                        ],
                    }
                ]
            }
        )
        self.assertEqual(claims[0]["citation_status"], "supported")
        self.assertIn("Results of Operations", claims[0]["claim_text"])
        self.assertEqual(claims[0]["company_or_ticker"], "BRK.B")
        self.assertEqual(
            claims[0]["source_anchor"],
            "accession:0001-26-000001; primary_document.text",
        )

    def test_foreign_fundamental_filing_keeps_p3_citation_and_freshness(self) -> None:
        claims = build_agent_citations(
            "fundamental",
            {
                "get_recent_sec_filing_texts": [
                    {
                        "symbol": "TSM",
                        "filings": [
                            {
                                "form": "6-K",
                                "filing_date": "2026-08-14",
                                "report_date": "2026-06-30",
                                "accession_number": "0001-26-000002",
                                "filing_index_url": "https://www.sec.gov/tsm-index.html",
                                "primary_document": {
                                    "text": "Quarterly financial results",
                                    "error": None,
                                },
                            }
                        ],
                    }
                ]
            },
        )
        self.assertEqual(claims[0]["claim_id"], "F1")
        self.assertEqual(claims[0]["citation_status"], "supported")
        self.assertIn(
            claims[0]["freshness_status"],
            {"fresh", "caution", "stale"},
        )


class CitationIntegrityTests(unittest.TestCase):
    def test_strategy_cannot_introduce_new_url_or_reference(self) -> None:
        claim = _supported_claim()
        report, registry, summary = finalize_cited_report(
            "依据[F1]。伪造[F99] https://fake.example/x\n\n"
            "## 15. 来源与证据\n模型自行生成的来源",
            [claim],
        )
        self.assertNotIn("https://fake.example/x", report)
        self.assertNotIn("[F99]", report)
        self.assertTrue(summary["citation_integrity"])
        self.assertEqual(len(registry), 1)
        self.assertEqual(report.count("## 来源与证据"), 1)
        self.assertNotIn("模型自行生成的来源", report)

    def test_ids_are_unique_refs_resolve_and_sources_are_used(self) -> None:
        claims = [_supported_claim("F1"), _supported_claim("F2")]
        report, registry, summary = finalize_cited_report("结论[F1][F2]", claims)
        self.assertEqual(len({item["claim_id"] for item in claims}), 2)
        self.assertFalse(summary["duplicate_claim_ids"])
        self.assertFalse(summary["unresolved_report_citations"])
        self.assertFalse(summary["unused_source_ids"])
        self.assertEqual(summary["cited_claims"], 2)
        self.assertIn("[F1]", report)
        self.assertIn("[F2]", report)
        self.assertEqual(len(registry), 1)

    def test_strategy_s_markers_are_mapped_only_to_real_sec_filing_claims(self) -> None:
        filings = []
        for index in (1, 2):
            filings.append(
                {
                    **_supported_claim(f"N{index}"),
                    "agent": "news",
                    "claim_type": "sec_filing_event",
                }
            )
        report, registry, summary = finalize_cited_report(
            "公司提交修订文件[S1]，随后公布季报[S2]；未知引用[S3]。",
            filings,
        )
        self.assertIn("[N1]", report)
        self.assertIn("[N2]", report)
        self.assertNotIn("[S1]", report)
        self.assertNotIn("[S2]", report)
        self.assertNotIn("[S3]", report)
        self.assertEqual({item for source in registry for item in source["claim_ids"]}, {"N1", "N2"})
        self.assertTrue(summary["citation_integrity"])

    def test_source_registry_excludes_claims_unused_by_report_body(self) -> None:
        claims = [_supported_claim("F1"), _supported_claim("F2")]
        report, registry, summary = finalize_cited_report("仅使用收入[F1]", claims)
        self.assertIn("[F1]", report)
        self.assertNotIn("[F2]", report)
        self.assertEqual(registry[0]["claim_ids"], ["F1"])
        self.assertEqual(summary["citation_required_claims"], 1)

    def test_duplicate_ids_fail_integrity(self) -> None:
        claims = [_supported_claim("F1"), _supported_claim("F1")]
        registry = build_source_registry(claims)
        summary = citation_summary(claims, "[F1]", registry)
        self.assertFalse(summary["citation_integrity"])
        self.assertEqual(summary["duplicate_claim_ids"], ["F1"])

    def test_registry_json_round_trip(self) -> None:
        claims = [_supported_claim()]
        registry = build_source_registry(claims)
        summary = citation_summary(claims, "[F1]", registry)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = save_source_registry(
                Path(temporary_directory) / "registry.json",
                citations=claims,
                source_registry=registry,
                summary=summary,
            )
            loaded = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["citations"][0]["claim_id"], "F1")
        self.assertTrue(loaded["citation_summary"]["citation_integrity"])


if __name__ == "__main__":
    unittest.main()

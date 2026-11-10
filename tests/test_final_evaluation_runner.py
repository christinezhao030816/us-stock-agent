import json
import tempfile
import unittest
from pathlib import Path

from evaluation.final_evaluation_runner import (
    CLAIM_INVENTORY_FIELDS,
    FACT_FIELDS,
    REPORT_FIELDS,
    _fundamental_verification,
    _technical_verification,
    build_claim_inventory,
    claim_inventory_summary,
    classify_extracted_item,
    citation_integrity_details,
    evaluate_completeness,
    evaluate_claim_inventory,
    evidence_classification_summary,
    evaluate_fact_claims,
    forbidden_financial_displays,
    independent_technical_ground_truth,
    load_rubric,
    metric_summary,
    parse_display_amount,
    reconcile_extraction_counts,
    split_report_body,
    validate_ticker_list,
)


def _bars(count: int = 260) -> list[dict]:
    bars = []
    for index in range(count):
        close = 100 + index * 0.25 + ((index % 7) - 3) * 0.4
        bars.append(
            {
                "date": f"2025-{index // 28 + 1:02d}-{index % 28 + 1:02d}",
                "open": close - 0.2,
                "high": close + 1.1,
                "low": close - 1.2,
                "close": close,
                "adjusted_close": close * 0.99,
                "volume": 1_000_000 + index * 100,
            }
        )
    return bars


def _technical_evidence() -> dict:
    return {
        "source_url": "https://finance.yahoo.com/quote/AAPL/history/",
        "raw_bars": _bars(),
    }


def _company_facts() -> dict:
    return {
        "cik": 123456,
        "facts": {
            "us-gaap": {
                "RevenueFromContractWithCustomerExcludingAssessedTax": {
                    "units": {
                        "USD": [
                            {
                                "val": 12_500_000_000,
                                "start": "2026-01-01",
                                "end": "2026-03-31",
                                "filed": "2026-05-01",
                                "form": "10-Q",
                                "accn": "0000123456-26-000001",
                            }
                        ]
                    }
                },
                "EarningsPerShareDiluted": {
                    "units": {
                        "USD/shares": [
                            {
                                "val": 2.34,
                                "start": "2026-01-01",
                                "end": "2026-03-31",
                                "filed": "2026-05-01",
                                "form": "10-Q",
                                "accn": "0000123456-26-000001",
                            }
                        ]
                    }
                },
            },
            "ifrs-full": {
                "Revenue": {
                    "units": {
                        "TWD": [
                            {
                                "val": 2_894_310_000_000,
                                "start": "2024-01-01",
                                "end": "2024-12-31",
                                "filed": "2025-04-17",
                                "form": "20-F",
                                "accn": "0001046179-25-000001",
                            }
                        ]
                    }
                }
            },
        },
    }


def _submissions() -> dict:
    return {
        "filings": {
            "recent": {
                "accessionNumber": ["0000123456-26-000001"],
                "form": ["10-Q"],
                "reportDate": ["2026-03-31"],
                "filingDate": ["2026-05-01"],
                "primaryDocument": ["example.htm"],
            }
        }
    }


class FinalEvaluationRubricTests(unittest.TestCase):
    def test_rubric_v3_freezes_all_metrics_and_output_fields(self) -> None:
        rubric = load_rubric()
        self.assertEqual(rubric["rubric_version"], "3.0")
        self.assertEqual(
            rubric["metrics"]["factual_accuracy"]["formula"],
            "correct / (correct + incorrect)",
        )
        self.assertEqual(len(rubric["metrics"]["completeness"]["modules"]), 8)
        self.assertTrue(set(rubric["output_contract"]["fact_required_fields"]) <= set(FACT_FIELDS))
        self.assertTrue(set(rubric["output_contract"]["report_required_fields"]) <= set(REPORT_FIELDS))
        self.assertTrue(
            set(rubric["output_contract"]["claim_inventory_required_fields"])
            <= set(CLAIM_INVENTORY_FIELDS)
        )

    def test_duplicate_tickers_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate tickers"):
            validate_ticker_list(["AAPL", "aapl"])


class IndependentTechnicalGroundTruthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rubric = load_rubric()
        self.evidence = _technical_evidence()
        self.truth = independent_technical_ground_truth(self.evidence)

    def test_recomputes_from_raw_bars_without_production_calculator(self) -> None:
        closes = [float(item["close"]) for item in self.evidence["raw_bars"]]
        expected_return = round((closes[-1] / closes[-21] - 1) * 100, 2)
        expected_sma200 = round(sum(closes[-200:]) / 200, 2)
        self.assertEqual(self.truth["return_20d"]["value"], expected_return)
        self.assertEqual(self.truth["sma200"]["value"], expected_sma200)
        self.assertEqual(self.truth["current_price"]["period"], self.evidence["raw_bars"][-1]["date"])

    def test_tampered_technical_value_is_incorrect(self) -> None:
        expected = self.truth["return_20d"]
        claim = {
            "metric_name": "return_20d",
            "value": expected["value"] + 0.02,
            "unit": "percent",
            "period_or_as_of_date": expected["period"],
            "source_anchor": "technical_evidence.indicator_evidence.return_20d_percent",
            "source_url": self.evidence["source_url"],
        }
        result = _technical_verification(claim, self.truth, self.evidence, self.rubric)
        self.assertEqual(result["accuracy"], "incorrect")
        self.assertEqual(result["numerical"], "incorrect")

    def test_price_tolerance_is_exactly_one_cent(self) -> None:
        expected = self.truth["current_price"]
        claim = {
            "metric_name": "current_price",
            "value": expected["value"] + 0.01,
            "unit": "USD",
            "period_or_as_of_date": expected["period"],
            "source_anchor": "technical_evidence.indicator_evidence.current_price_usd",
            "source_url": self.evidence["source_url"],
        }
        result = _technical_verification(claim, self.truth, self.evidence, self.rubric)
        self.assertEqual(result["accuracy"], "correct")


class IndependentFundamentalGroundTruthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.rubric = load_rubric()
        self.facts = _company_facts()
        self.submissions = _submissions()

    def test_us_gaap_amount_reconciles_value_currency_period_and_accession(self) -> None:
        claim = {
            "metric_name": "revenue",
            "value": 12.5,
            "unit": "USD billion",
            "period_or_as_of_date": "2026-03-31",
            "source_date": "2026-05-01",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000123456.json",
            "source_anchor": (
                "facts.us-gaap.RevenueFromContractWithCustomerExcludingAssessedTax.units.USD "
                "period=2026-01-01..2026-03-31 accession=0000123456-26-000001"
            ),
        }
        result = _fundamental_verification(claim, self.facts, self.submissions, self.rubric)
        self.assertEqual(result["accuracy"], "correct")
        self.assertEqual(result["numerical"], "correct")

    def test_full_display_does_not_change_numerical_ground_truth_value(self) -> None:
        display = parse_display_amount("收入为12,500,000,000 美元")
        self.assertEqual(display["value"] / 1_000_000_000, 12.5)
        claim = {
            "metric_name": "revenue",
            "value": 12.5,
            "unit": "USD billion",
            "period_or_as_of_date": "2026-03-31",
            "source_date": "2026-05-01",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000123456.json",
            "source_anchor": (
                "facts.us-gaap.RevenueFromContractWithCustomerExcludingAssessedTax.units.USD "
                "period=2026-01-01..2026-03-31 accession=0000123456-26-000001"
            ),
        }
        result = _fundamental_verification(claim, self.facts, self.submissions, self.rubric)
        self.assertEqual(result["numerical"], "correct")

    def test_ifrs_original_currency_is_preserved(self) -> None:
        claim = {
            "metric_name": "revenue",
            "value": 2894.31,
            "unit": "TWD billion",
            "period_or_as_of_date": "2024-12-31",
            "source_date": "2025-04-17",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000123456.json",
            "source_anchor": (
                "facts.ifrs-full.Revenue.units.TWD period=2024-01-01..2024-12-31 "
                "accession=0001046179-25-000001"
            ),
        }
        result = _fundamental_verification(claim, self.facts, self.submissions, self.rubric)
        self.assertEqual(result["accuracy"], "correct")
        wrong_currency = {**claim, "unit": "USD billion"}
        wrong_result = _fundamental_verification(
            wrong_currency, self.facts, self.submissions, self.rubric
        )
        self.assertEqual(wrong_result["accuracy"], "incorrect")
        self.assertEqual(wrong_result["numerical"], "incorrect")

    def test_period_mismatch_is_numerically_incorrect(self) -> None:
        claim = {
            "metric_name": "revenue",
            "value": 12.5,
            "unit": "USD billion",
            "period_or_as_of_date": "2025-12-31",
            "source_date": "2026-05-01",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000123456.json",
            "source_anchor": (
                "facts.us-gaap.RevenueFromContractWithCustomerExcludingAssessedTax.units.USD "
                "period=2026-01-01..2026-03-31 accession=0000123456-26-000001"
            ),
        }
        result = _fundamental_verification(claim, self.facts, self.submissions, self.rubric)
        self.assertEqual(result["accuracy"], "incorrect")
        self.assertEqual(result["numerical"], "incorrect")

    def test_eps_share_unit_and_date_pseudo_unit_are_valid(self) -> None:
        eps_claim = {
            "metric_name": "EPS",
            "value": 2.34,
            "unit": "USD/share",
            "period_or_as_of_date": "2026-03-31",
            "source_date": "2026-05-01",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000123456.json",
            "source_anchor": (
                "facts.us-gaap.EarningsPerShareDiluted.units.USD/shares "
                "period=2026-01-01..2026-03-31 accession=0000123456-26-000001"
            ),
        }
        period_claim = {
            "metric_name": "reporting_period",
            "value": "2026-03-31",
            "unit": "date",
            "period_or_as_of_date": "2026-03-31",
            "source_date": "2026-05-01",
            "source_url": "https://data.sec.gov/api/xbrl/companyfacts/CIK0000123456.json",
            "source_anchor": (
                "facts.us-gaap.RevenueFromContractWithCustomerExcludingAssessedTax.units.USD "
                "period=2026-01-01..2026-03-31 accession=0000123456-26-000001"
            ),
        }
        self.assertEqual(
            _fundamental_verification(
                eps_claim, self.facts, self.submissions, self.rubric
            )["accuracy"],
            "correct",
        )
        self.assertEqual(
            _fundamental_verification(
                period_claim, self.facts, self.submissions, self.rubric
            )["accuracy"],
            "correct",
        )


class CitationAndCompletenessTests(unittest.TestCase):
    def test_claim_inventory_starts_from_body_and_detects_missing_invalid_claims(self) -> None:
        citations = [
            {
                "claim_id": "F1",
                "metric_name": "revenue",
                "claim_type": "fundamental_metric",
                "claim_text": "收入为100美元",
                "value": 100,
                "unit": "USD",
            },
            {
                "claim_id": "N1",
                "agent": "news",
                "metric_name": "filing_event",
                "claim_type": "sec_filing_event",
                "claim_text": "提交日期为2026-08-01",
                "value": "2026-08-01",
                "unit": "date",
            },
        ]
        report = """
# 合成报告
错误数字：收入为999美元。
未引用事实：收入为100美元。
无效引用：提交日期为2026-08-01 [S1]。
我认为公司前景很好。
"""
        inventory = build_claim_inventory("TEST", report, citations)
        self.assertEqual(len(inventory), 4)
        self.assertTrue(any(item["registry_claim_id"] == "" for item in inventory))
        uncited = next(item for item in inventory if "100美元" in item["sentence_text"])
        self.assertEqual(uncited["registry_claim_id"], "F1")
        self.assertFalse(uncited["citation_valid"])
        invalid = next(item for item in inventory if "[S1]" in item["sentence_text"])
        self.assertEqual(invalid["registry_claim_id"], "N1")
        self.assertEqual(invalid["extraction_method"], "body_invalid_citation")
        opinion = next(item for item in inventory if "前景很好" in item["sentence_text"])
        self.assertEqual(opinion["item_category"], "analytical_judgment")
        summary = claim_inventory_summary(inventory, citations)
        self.assertEqual(summary["total_body_factual_claims"], 3)
        self.assertEqual(summary["registry_matched_claims"], 2)
        self.assertEqual(summary["unmatched_body_claims"], 1)
        self.assertEqual(summary["uncited_body_claims"], 2)
        self.assertEqual(summary["invalid_citation_claims"], 1)

    def test_system_claims_use_internal_provenance_not_external_denominator(self) -> None:
        provenance = [
            {
                "provenance_id": "I1",
                "claim_category": "system_derived",
                "metric_name": "confidence_score",
                "input_fields": {"technical_metrics_present": 4},
                "formula_or_rule": "score/v1",
                "result": {"final_score": 78},
                "calculation_timestamp": "2026-09-21T00:00:00Z",
                "provenance_path": "tool_outputs.calculate_confidence_score[-1]",
                "deterministic": True,
            }
        ]
        inventory = build_claim_inventory(
            "TEST", "## 综合置信度评分\n系统评分为78分。", [], provenance
        )
        summary = evidence_classification_summary(inventory)
        self.assertEqual(summary["external_claim_occurrences"], 0)
        self.assertEqual(summary["internal_system_claims"], 1)
        self.assertEqual(summary["internal_traceability_coverage_pct"], 100.0)

    def test_missing_internal_provenance_is_visible(self) -> None:
        inventory = build_claim_inventory(
            "TEST", "## 综合置信度评分\n系统评分为78分。", [], []
        )
        summary = evidence_classification_summary(inventory)
        self.assertEqual(summary["internal_traceability_coverage_pct"], 0.0)

    def test_number_or_date_cannot_be_relabelled_as_opinion(self) -> None:
        self.assertEqual(
            classify_extracted_item("我认为收入为9,180,000,000美元。", "综合判断"),
            "external_fact",
        )
        self.assertEqual(
            classify_extracted_item("我认为公司前景偏积极。", "综合判断"),
            "analytical_judgment",
        )

    def test_duplicate_fact_only_counts_once_for_factual_accuracy(self) -> None:
        rows = [
            {
                "claim_id": "T1",
                "unique_claim_key": "T1",
                "occurrence_id": "1",
                "accuracy_label": "correct",
                "support_label": "supported",
                "numerical_label": "correct",
                "citation_label": "correct",
            },
            {
                "claim_id": "T1",
                "unique_claim_key": "T1",
                "occurrence_id": "2",
                "accuracy_label": "correct",
                "support_label": "supported",
                "numerical_label": "correct",
                "citation_label": "not_cited",
            },
        ]
        summary = metric_summary(rows)
        self.assertEqual(summary["factual_correct"], 1)
        self.assertEqual(summary["unique_external_claims"], 1)
        self.assertEqual(summary["external_claim_occurrences"], 2)
        self.assertEqual(summary["citation_required_claims"], 2)
        self.assertEqual(summary["citation_coverage_pct"], 50.0)

    def test_reconciliation_closes_for_all_extracted_items(self) -> None:
        report = """
## 技术面
收盘价为100美元[T1]。
## 综合判断
我认为前景偏积极[T1]。
## 系统评分
系统评分为78分。
## 数据限制
新闻数据不可用，状态为partial。
普通说明文字。
"""
        citation = {"claim_id": "T1", "metric_name": "current_price", "value": 100}
        summary = evidence_classification_summary(
            build_claim_inventory("TEST", report, [citation], [])
        )
        self.assertTrue(summary["reconciliation_closed"])
        self.assertEqual(summary["total_extracted_items"], summary["reconciliation_total"])

    def test_legacy_879_868_reconciliation_is_exact(self) -> None:
        result = reconcile_extraction_counts(
            total_extracted_items=879,
            body_not_claimed_markers=11,
            external_factual_denominator=868,
            catalog_only_not_claimed=1,
            total_not_claimed_rows=12,
        )
        self.assertTrue(result["closed"])
        self.assertEqual(result["unexplained_difference"], 0)

    def test_full_amount_parser_supports_currency_parentheses_and_per_share(self) -> None:
        self.assertEqual(
            parse_display_amount("亏损为(250,000,000 美元)"),
            {"value": -250000000.0, "currency": "USD", "unit": "USD"},
        )
        self.assertEqual(
            parse_display_amount("收入582,000,000,000 新台币"),
            {"value": 582000000000.0, "currency": "TWD", "unit": "TWD"},
        )
        self.assertEqual(
            parse_display_amount("EPS为1.43 美元/股"),
            {"value": 1.43, "currency": "USD", "unit": "USD/share"},
        )
        self.assertEqual(
            forbidden_financial_displays(
                "收入9,180,000,000 美元，EPS 1.43 美元/股，RSI 63.38"
            ),
            [],
        )
        self.assertEqual(
            forbidden_financial_displays("收入9.18十亿美元或9.18 billion USD"),
            ["十亿美元", "billion"],
        )

    def test_body_value_cannot_be_validated_by_a_different_registry_value(self) -> None:
        citation = {
            "claim_id": "F1",
            "metric_name": "revenue",
            "value": 100,
            "unit": "USD",
        }
        inventory = build_claim_inventory(
            "TEST", "收入为999美元[F1]。", [citation]
        )
        registry_row = {
            "ticker": "TEST",
            "claim_id": "F1",
            "category": "fundamental",
            "claim_type": "fundamental_metric",
            "metric_name": "revenue",
            "agent_claim": "收入为100美元",
            "ground_truth": "{}",
            "accuracy_label": "correct",
            "support_label": "supported",
            "numerical_label": "correct",
            "citation_label": "correct",
            "evidence_source": "SEC",
            "evidence_url": "https://example.com",
            "notes": "",
        }
        rows, _ = evaluate_claim_inventory(inventory, [registry_row], [citation])
        self.assertEqual(rows[0]["accuracy_label"], "incorrect")
        self.assertEqual(rows[0]["support_label"], "unsupported")
        self.assertEqual(rows[0]["numerical_label"], "incorrect")
        self.assertEqual(rows[0]["citation_label"], "incorrect")

    def test_sources_appendix_does_not_create_narrative_coverage(self) -> None:
        report = "# Analysis\nNo claim marker here.\n\n## 来源与证据\n- [T1] appended"
        self.assertNotIn("[T1]", split_report_body(report))
        citations = [{"claim_id": "T1", "source_url": "https://example.com/a"}]
        registry = [
            {
                "source_id": "SRC1",
                "source_url": "https://example.com/a",
                "claim_ids": ["T1"],
            }
        ]
        result = citation_integrity_details(citations, report, registry)
        self.assertFalse(result["citation_integrity"])
        self.assertEqual(result["unused_source_ids"], ["SRC1"])

    def test_integrity_detects_duplicate_unknown_and_unregistered_url(self) -> None:
        citations = [
            {"claim_id": "T1", "source_url": "https://example.com/a"},
            {"claim_id": "T1", "source_url": "https://example.com/a"},
        ]
        report = "Claim [T1] and unknown [F9] [S1] https://bad.example/item"
        registry = [
            {
                "source_id": "SRC1",
                "source_url": "https://example.com/a",
                "claim_ids": ["T1"],
            }
        ]
        result = citation_integrity_details(citations, report, registry)
        self.assertEqual(result["duplicate_claim_ids"], ["T1"])
        self.assertEqual(result["unknown_claim_ids"], ["F9", "S1"])
        self.assertEqual(result["unregistered_urls"], ["https://bad.example/item"])

    def test_empty_heading_fails_but_complete_missing_disclosure_passes(self) -> None:
        report = """
# Report
## 技术面摘要

## 消息面摘要
新闻数据不可用，因为接口未返回文章；这限制事件判断并降低置信度，消息面状态为 partial。
"""
        result = evaluate_completeness(report)
        self.assertFalse(result["technical"]["completed"])
        self.assertTrue(result["news_events"]["completed"])

    def test_parent_module_includes_nested_subheading_content(self) -> None:
        report = """
# Report
## 基本面摘要
### 盈利与增长
收入和利润数据来自结构化申报，并明确说明期间、币种和数据限制，足以形成可审查的基本面摘要。
## 看空与风险因素
如果收入放缓且现金流转弱，下行风险会显著增加，因此这是明确的看空情景而非空标题。
"""
        result = evaluate_completeness(report)
        self.assertTrue(result["fundamental"]["completed"])
        self.assertTrue(result["risks"]["completed"])
        self.assertTrue(result["bear_case"]["completed"])

    def test_not_claimed_does_not_inflate_unsupported_rate(self) -> None:
        rows = [
            {
                "accuracy_label": "correct",
                "support_label": "supported",
                "numerical_label": "correct",
                "citation_label": "correct",
            },
            {
                "accuracy_label": "not_claimed",
                "support_label": "not_claimed",
                "numerical_label": "not_claimed",
                "citation_label": "not_evaluated",
            },
            {
                "accuracy_label": "unverifiable",
                "support_label": "unsupported",
                "numerical_label": "unsupported",
                "citation_label": "incorrect",
            },
        ]
        summary = metric_summary(rows)
        self.assertEqual(summary["unsupported_claims"], 1)
        self.assertEqual(summary["total_factual_claims"], 2)
        self.assertEqual(summary["unsupported_claim_rate_pct"], 50.0)
        self.assertEqual(summary["factual_not_claimed"], 1)

    def test_fact_evaluator_checks_news_structured_fields(self) -> None:
        response = {
            "report": "Event [N1]",
            "citations": [
                {
                    "claim_id": "N1",
                    "agent": "news",
                    "claim_type": "news_event",
                    "metric_name": "event",
                    "claim_text": "Acme reports results",
                    "value": {
                        "title": "Acme reports results",
                        "description": "Revenue increased.",
                        "company_match": True,
                    },
                    "source_name": "Reuters",
                    "source_url": "https://example.com/news",
                    "source_date": "2026-09-20T01:00:00Z",
                    "period_or_as_of_date": "2026-09-20T01:00:00Z",
                    "source_anchor": "Acme reports results",
                    "citation_status": "supported",
                    "company_or_ticker": "AAPL",
                }
            ],
            "source_registry": [
                {
                    "source_id": "SRC1",
                    "source_url": "https://example.com/news",
                    "claim_ids": ["N1"],
                }
            ],
        }
        news_snapshot = {
            "news": {
                "articles": [
                    {
                        "headline": "Acme reports results",
                        "summary": "Revenue increased.",
                        "source": "Reuters",
                        "created_at": "2026-09-20T01:00:00Z",
                        "url": "https://example.com/news",
                        "symbols": ["AAPL"],
                    }
                ]
            },
            "filings": {"filings": []},
        }
        rows, integrity = evaluate_fact_claims(
            ticker="AAPL",
            response=response,
            technical_evidence=None,
            company_facts=None,
            submissions=None,
            news_snapshot=news_snapshot,
            rubric=load_rubric(),
        )
        news_row = next(row for row in rows if row["claim_id"] == "N1")
        self.assertEqual(news_row["accuracy_label"], "correct")
        self.assertEqual(news_row["citation_label"], "correct")
        self.assertTrue(integrity["citation_integrity"])

    def test_missing_body_marker_affects_coverage_not_citation_accuracy(self) -> None:
        response = {
            "report": "Event without an inline marker.",
            "citations": [
                {
                    "claim_id": "N1",
                    "agent": "news",
                    "claim_type": "news_event",
                    "metric_name": "event",
                    "claim_text": "Acme reports results",
                    "value": {
                        "title": "Acme reports results",
                        "description": "Revenue increased.",
                        "company_match": True,
                    },
                    "source_name": "Reuters",
                    "source_url": "https://example.com/news",
                    "source_date": "2026-09-20T01:00:00Z",
                    "period_or_as_of_date": "2026-09-20T01:00:00Z",
                    "source_anchor": "Acme reports results",
                    "citation_status": "supported",
                    "company_or_ticker": "AAPL",
                }
            ],
            "source_registry": [
                {
                    "source_id": "SRC1",
                    "source_url": "https://example.com/news",
                    "claim_ids": ["N1"],
                }
            ],
        }
        news_snapshot = {
            "news": {
                "articles": [
                    {
                        "headline": "Acme reports results",
                        "summary": "Revenue increased.",
                        "source": "Reuters",
                        "created_at": "2026-09-20T01:00:00Z",
                        "url": "https://example.com/news",
                        "symbols": ["AAPL"],
                    }
                ]
            },
            "filings": {"filings": []},
        }
        rows, _ = evaluate_fact_claims(
            ticker="AAPL",
            response=response,
            technical_evidence=None,
            company_facts=None,
            submissions=None,
            news_snapshot=news_snapshot,
            rubric=load_rubric(),
        )
        news_row = next(row for row in rows if row["claim_id"] == "N1")
        summary = metric_summary(rows)
        self.assertEqual(news_row["accuracy_label"], "correct")
        self.assertEqual(news_row["citation_label"], "not_cited")
        self.assertEqual(summary["citation_accuracy_denominator"], 0)
        self.assertEqual(summary["citation_coverage_pct"], 0.0)

    def test_output_schema_round_trips_as_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            target = Path(temporary_directory) / "schema.json"
            target.write_text(
                json.dumps(
                    {
                        "fact": FACT_FIELDS,
                        "report": REPORT_FIELDS,
                        "claim_inventory": CLAIM_INVENTORY_FIELDS,
                    }
                ),
                encoding="utf-8",
            )
            restored = json.loads(target.read_text(encoding="utf-8"))
        self.assertIn("ground_truth", restored["fact"])
        self.assertIn("citation_integrity", restored["report"])
        self.assertIn("sentence_text", restored["claim_inventory"])


if __name__ == "__main__":
    unittest.main()
    build_claim_inventory,
    claim_inventory_summary,

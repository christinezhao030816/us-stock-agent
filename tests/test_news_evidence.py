import unittest

from evaluation.final_evaluation_runner import (
    build_claim_inventory,
    evaluate_completeness,
)
from tools.citation_registry import (
    build_news_citations,
    enforce_news_evidence_boundaries,
)
from tools.news_evidence import (
    classify_article_evidence,
    classify_filing_evidence,
)


def _article(headline: str, summary=None, url="https://example.com/item") -> dict:
    return {
        "headline": headline,
        "summary": summary,
        "source": "Example News",
        "created_at": "2026-11-05T11:00:00Z",
        "url": url,
        "symbols": ["TEST"],
    }


PHASE7A_HEADLINE_FIXTURES = (
    "Caterpillar Acquires John Fabick Tractor; Financial Terms Not Disclosed",
    "Costco Wholesale Reported Net Sales Of $24.50B For The Retail Month Of October, "
    "An Increase Of 12.6 Percent From $21.75B Last Year; Net Sales For The First "
    "Nine Weeks Were $54.52B, An Increase Of 12.8 Percent From $48.33B Last Year",
    "Reported Sunday, Johnson & Johnson Says Impella Outcomes Depend On Patient "
    "Selection, Timing And Post-Procedural Care, Citing Lower Mortality In Three "
    "TCT Analyses Versus CHIP-BCIS3; PROTECT IV Results Expected 2027",
    "Watching Nvidia; Hunterbrook Media Issues Report Titled, “The Comprehensive "
    "Guide to the Bank of Nvidia”",
    "Toyota Motor Raises FY2027 GAAP EPS Guidance from $17.08 to $18.84 vs $21.80 "
    "Est; Lowers FY2027 Sales Guidance from $338.850B to $336.087B vs $337.600B Est",
    "Toyota Interim Dividend ¥50/Share vs ¥45 YoY; Maintains FY Dividend Forecast At ¥100",
)


class NewsEvidenceBoundaryTests(unittest.TestCase):
    def test_english_boundary_rendering_does_not_insert_chinese_labels(self) -> None:
        claims = build_news_citations(
            {
                "get_stock_news": [
                    {
                        "symbol": "TEST",
                        "articles": [_article("Verified headline")],
                    }
                ]
            }
        )
        repaired = enforce_news_evidence_boundaries(
            "The report expanded beyond the headline. [N1]",
            claims,
            language="en",
        )
        self.assertIn("Headline-only evidence", repaired)
        self.assertIn("Verified headline", repaired)
        self.assertNotIn("标题级信息", repaired)

    def test_phase7a_headline_claims_remain_atomic_and_keep_title_values(self) -> None:
        for title in PHASE7A_HEADLINE_FIXTURES:
            with self.subTest(title=title):
                claims = build_news_citations(
                    {
                        "get_stock_news": [
                            {"symbol": "TEST", "articles": [_article(title)]}
                        ]
                    }
                )
                repaired = enforce_news_evidence_boundaries(
                    "管理层因此能够保证长期利润增长。[N1]",
                    claims,
                )
                inventory = build_claim_inventory("TEST", repaired, claims, [])
                occurrences = [
                    item
                    for item in inventory
                    if item["registry_claim_id"] == "N1"
                ]
                self.assertEqual(len(occurrences), 1)
                self.assertIn("标题级信息，未读取完整正文", repaired)
                self.assertNotIn("保证长期利润增长", repaired)
                for token in __import__("re").findall(
                    r"(?:[$¥]\d[\d.]*B?|\d+(?:\.\d+)?\s*Percent|FY\d{4}|2027)",
                    title,
                    flags=__import__("re").IGNORECASE,
                ):
                    self.assertIn(token, repaired)

    def test_phase7a_summary_without_full_text_is_not_strategy_eligible(self) -> None:
        claims = build_news_citations(
            {
                "get_stock_news": [
                    {
                        "symbol": "XOM",
                        "articles": [
                            {
                                **_article(
                                    "These Analysts Boost Their Forecasts On ExxonMobil",
                                    summary=(
                                        "ExxonMobil beat estimates; shares fell 2.9%."
                                    ),
                                ),
                                "symbols": ["XOM"],
                            }
                        ],
                    }
                ]
            }
        )

        self.assertEqual(claims[0]["evidence_depth"], "fully_supported")
        self.assertFalse(claims[0]["full_text_available"])
        self.assertFalse(claims[0]["strategy_eligible"])
        self.assertEqual(
            claims[0]["strategy_limitation_reason"],
            "full_text_unavailable",
        )

    def test_phase7a_ineligible_news_cannot_remain_in_core_strategy_sections(self) -> None:
        headline = build_news_citations(
            {
                "get_stock_news": [
                    {
                        "symbol": "XOM",
                        "articles": [
                            {
                                **_article(
                                    "UBS Maintains Buy on ExxonMobil, Raises Price Target to $213"
                                ),
                                "symbols": ["XOM"],
                            }
                        ],
                    }
                ]
            }
        )[0]
        summary_only = build_news_citations(
            {
                "get_stock_news": [
                    {
                        "symbol": "SONY",
                        "articles": [
                            {
                                **_article(
                                    "Sony Boosts Guidance as Profit Jumps",
                                    summary="Quarterly net profit increased 44%.",
                                ),
                                "symbols": ["SONY"],
                            }
                        ],
                    }
                ]
            }
        )[0]
        summary_only["claim_id"] = "N2"
        report = """## 7. 看多因素
分析师目标价上调，构成核心看多催化剂。[N1]
Sony利润增长证明长期增长确定。[N2]

## 4. 消息面摘要
消息模块保留受限信息。[N1][N2]

## 9. 综合判断
上述新闻支持买入。[N1][N2]
"""

        repaired = enforce_news_evidence_boundaries(
            report,
            [headline, summary_only],
        )
        core_bull = repaired.split("## 7. 看多因素", 1)[1].split(
            "## 4. 消息面摘要", 1
        )[0]
        core_final = repaired.split("## 9. 综合判断", 1)[1].split(
            "### 标题级与待核实信息", 1
        )[0]

        self.assertNotIn("[N1]", core_bull)
        self.assertNotIn("[N2]", core_bull)
        self.assertNotIn("[N1]", core_final)
        self.assertNotIn("[N2]", core_final)
        self.assertIn("## 4. 消息面摘要", repaired)
        self.assertIn("标题级与待核实信息", repaired)
        self.assertIn("不作为核心投资结论依据", repaired)

    def test_phase7a_truncated_source_is_gated_out_of_core_conclusion(self) -> None:
        claim = {
            "claim_id": "N1",
            "agent": "news",
            "citation_status": "partial",
            "evidence_depth": "truncated_source",
            "claim_text": "附件内容不完整，仅确认已读取片段：SEC 8-K filing",
            "strategy_eligible": False,
            "strategy_limitation_reason": "truncated_source",
            "source_content": {
                "form": "8-K",
                "accession_number": "0001-26-000001",
            },
        }
        repaired = enforce_news_evidence_boundaries(
            "## 综合判断\n附件证明全年利润确定增长。[N1]",
            [claim],
        )

        core = repaired.split("## 综合判断", 1)[1].split(
            "### 标题级与待核实信息", 1
        )[0]
        self.assertNotIn("[N1]", core)
        self.assertNotIn("全年利润确定增长", repaired)
        self.assertIn("附件内容不完整", repaired)
    def test_headline_only_cannot_be_expanded(self) -> None:
        claims = build_news_citations(
            {"get_stock_news": [{"symbol": "TEST", "articles": [_article("Company raises guidance")]}]}
        )
        claim = claims[0]
        self.assertEqual(claim["evidence_depth"], "headline_only")
        self.assertTrue(claim["claim_scope_limited"])
        report = "该事项将显著提高长期利润率并推动股价上涨。[N1]"
        repaired = enforce_news_evidence_boundaries(report, claims)
        self.assertNotIn("长期利润率", repaired)
        self.assertIn("标题级信息，未读取完整正文：Company raises guidance [N1]", repaired)

    def test_truncated_sec_attachment_cannot_be_completed(self) -> None:
        filing = {
            "form": "8-K",
            "filing_date": "2026-11-05",
            "report_date": "2026-09-30",
            "accession_number": "0001-26-000001",
            "filing_index_url": "https://www.sec.gov/example-index.html",
            "item_descriptions": ["Results of Operations"],
            "primary_document": {"text": "8-K cover", "truncated": False, "error": None},
            "exhibits": [
                {
                    "document_type": "EX-99.1",
                    "text": "Actual EPS was 1.99 [truncated]",
                    "truncated": True,
                    "error": None,
                }
            ],
        }
        boundary = classify_filing_evidence(filing)
        self.assertEqual(boundary["evidence_depth"], "truncated_source")
        claims = build_news_citations(
            {"get_recent_sec_filing_texts": [{"symbol": "TEST", "filings": [filing]}]}
        )
        repaired = enforce_news_evidence_boundaries(
            "附件完整证明公司全年利润会增长。[N1]", claims
        )
        self.assertNotIn("全年利润会增长", repaired)
        self.assertIn("附件内容不完整，仅确认已读取片段", repaired)

    def test_unavailable_news_is_operational_not_in_conclusion(self) -> None:
        claim = {
            "claim_id": "N1",
            "agent": "news",
            "citation_status": "unsupported",
            "evidence_depth": "unavailable",
            "claim_text": "待核实信息（正文不可用）：Unknown event",
            "source_content": {"title": "Unknown event"},
        }
        repaired = enforce_news_evidence_boundaries(
            "该事件是确定性看多催化剂。[N1]", [claim]
        )
        self.assertNotIn("确定性看多催化剂", repaired)
        self.assertIn("不作为已确认新闻事实或投资结论依据", repaired)
        inventory = build_claim_inventory("TEST", repaired, [claim], [])
        self.assertTrue(
            all(item["item_category"] != "external_fact" for item in inventory)
        )

    def test_limited_strategy_occurrence_keeps_depth_and_disclosure(self) -> None:
        article = _article("Company reports EPS 1.99")
        claims = build_news_citations(
            {"get_stock_news": [{"symbol": "TEST", "articles": [article]}]}
        )
        repaired = enforce_news_evidence_boundaries(
            "EPS beat expectations and guarantees growth.[N1]", claims
        )
        inventory = build_claim_inventory("TEST", repaired, claims, [])
        occurrence = next(item for item in inventory if item["registry_claim_id"] == "N1")
        self.assertEqual(occurrence["evidence_depth"], "headline_only")
        self.assertTrue(occurrence["claim_scope_limited"])
        self.assertIn("未读取完整正文", occurrence["sentence_text"])

    def test_news_module_cannot_pass_by_deletion(self) -> None:
        empty = evaluate_completeness("## 消息面摘要\n\n## 综合判断\n内容足够长但没有新闻。")
        self.assertFalse(empty["news_events"]["completed"])
        disclosed = evaluate_completeness(
            "## 消息面摘要\n待核实信息正文不可用，因此消息模块为partial；这限制事件判断并降低置信度。"
        )
        self.assertTrue(disclosed["news_events"]["completed"])

    def test_aapl_and_duk_headline_fixtures_use_same_generic_rule(self) -> None:
        fixtures = [
            "Apple Cuts Internal AI Usage Budgets in Move to Reduce Spending",
            "Duke Energy Q3 Adj. EPS $1.99 Beats $1.86 Estimate",
            "Duke Energy Raises FY2026 Adj EPS Guidance to $6.70-$6.80",
        ]
        for title in fixtures:
            with self.subTest(title=title):
                boundary = classify_article_evidence(
                    _article(title), company_match=True
                )
                self.assertEqual(boundary["evidence_depth"], "headline_only")
                self.assertFalse(boundary["full_text_available"])


if __name__ == "__main__":
    unittest.main()

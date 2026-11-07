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


class NewsEvidenceBoundaryTests(unittest.TestCase):
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

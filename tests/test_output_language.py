import unittest

from tools.citation_registry import citations_for_language
from tools.financial_freshness import evaluate_fact_freshness
from tools.output_language import (
    assert_english_output_contract,
    cjk_paths,
)


class EnglishOutputContractTests(unittest.TestCase):
    def test_citation_localization_preserves_evidence_identity(self) -> None:
        original = {
            "claim_id": "F2",
            "agent": "fundamental",
            "claim_type": "fundamental_metric",
            "metric_name": "net_income",
            "claim_text": "净利润为1,141,600,000,000日元",
            "value": 1141.6,
            "unit": "JPY billion",
            "currency": "JPY",
            "period_or_as_of_date": "2026-03-31",
            "source_name": "SEC Company Facts API",
            "source_type": "sec_company_facts",
            "source_url": "https://data.sec.gov/example.json",
            "source_date": "2026-06-20",
            "source_anchor": "facts.ifrs-full.ProfitLoss.units.JPY accession=abc-123",
            "evidence_path": "https://data.sec.gov/example.json",
            "citation_status": "supported",
            "company_or_ticker": "SONY",
            "source_content": {
                "description": "附件内容不完整",
                "accession_number": "abc-123",
            },
        }

        localized = citations_for_language([original], "en")[0]

        self.assertEqual(
            localized["claim_text"],
            "SONY net_income = JPY 1,141,600,000,000",
        )
        for key in (
            "claim_id",
            "value",
            "unit",
            "currency",
            "period_or_as_of_date",
            "source_url",
            "source_date",
            "source_anchor",
        ):
            self.assertEqual(localized[key], original[key])
        self.assertEqual(cjk_paths(localized), [])
        self.assertEqual(
            localized["source_content"]["accession_number"],
            "abc-123",
        )

    def test_freshness_reason_is_english(self) -> None:
        fact = evaluate_fact_freshness(
            metric_name="revenue",
            value=100,
            unit="USD",
            form="10-Q",
            fiscal_year=2026,
            fiscal_period="Q2",
            period_start="2026-04-01",
            period_end="2026-06-30",
            filed_date="2026-08-01",
            accession="abc-123",
            analysis_as_of="2026-09-11",
        )
        self.assertEqual(cjk_paths(fact), [])

    def test_contract_reports_nested_cjk_path(self) -> None:
        with self.assertRaisesRegex(ValueError, r"response\.report"):
            assert_english_output_contract({"report": "中英文 mixed"})


if __name__ == "__main__":
    unittest.main()

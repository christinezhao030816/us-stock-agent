import unittest
from inspect import signature

from tools.agent_status import evaluate_fundamental_status
from tools.citation_registry import build_fundamental_citations
from tools.ifrs_fundamental import (
    IFRS_CONCEPT_MAP,
    build_ifrs_fundamental_metrics,
    flow_period_granularity,
    latest_foreign_filings,
)
from tools.fundamental_agent_tool import (
    get_fundamental_metrics as agent_fundamental_tool,
)


def _flow(
    value: float,
    *,
    start: str = "2025-04-01",
    end: str = "2026-03-31",
    filed: str = "2026-06-18",
    form: str = "20-F",
    accession: str = "current-accession",
    fiscal_year: int = 2026,
    fiscal_period: str = "FY",
) -> dict:
    return {
        "val": value,
        "start": start,
        "end": end,
        "filed": filed,
        "form": form,
        "accn": accession,
        "fy": fiscal_year,
        "fp": fiscal_period,
    }


def _instant(value: float, **overrides) -> dict:
    record = _flow(value, **overrides)
    record.pop("start")
    return record


def _company_facts(
    currency: str = "JPY",
    *,
    include_capex: bool = True,
    records: dict | None = None,
) -> dict:
    unit_per_share = f"{currency}/shares"
    facts = {
        "Revenue": {"units": {currency: [_flow(100_000_000_000)]}},
        "ProfitLossAttributableToOwnersOfParent": {
            "units": {currency: [_flow(10_000_000_000)]}
        },
        "DilutedEarningsLossPerShare": {
            "units": {unit_per_share: [_flow(123.45)]}
        },
        "CashFlowsFromUsedInOperatingActivities": {
            "units": {currency: [_flow(20_000_000_000)]}
        },
        "Assets": {"units": {currency: [_instant(300_000_000_000)]}},
        "Liabilities": {
            "units": {currency: [_instant(180_000_000_000)]}
        },
        "Equity": {"units": {currency: [_instant(120_000_000_000)]}},
        "CashAndCashEquivalents": {
            "units": {currency: [_instant(50_000_000_000)]}
        },
        "Borrowings": {
            "units": {currency: [_instant(40_000_000_000)]}
        },
    }
    if include_capex:
        facts[
            "PurchaseOfPropertyPlantAndEquipmentClassifiedAsInvestingActivities"
        ] = {"units": {currency: [_flow(5_000_000_000)]}}
    if records:
        facts.update(records)
    return {
        "entityName": "Pilot Foreign Issuer",
        "facts": {
            "ifrs-full": facts,
            "dei": {},
        },
    }


def _submissions(
    *,
    twenty_f_end: str = "2026-03-31",
    twenty_f_filed: str = "2026-06-18",
    twenty_f_accession: str = "current-accession",
    six_k: bool = False,
) -> dict:
    forms = ["20-F"]
    filed = [twenty_f_filed]
    ends = [twenty_f_end]
    accessions = [twenty_f_accession]
    documents = ["annual.htm"]
    if six_k:
        forms.insert(0, "6-K")
        filed.insert(0, "2026-08-10")
        ends.insert(0, "2026-06-30")
        accessions.insert(0, "interim-accession")
        documents.insert(0, "interim.htm")
    return {
        "filings": {
            "recent": {
                "form": forms,
                "filingDate": filed,
                "reportDate": ends,
                "accessionNumber": accessions,
                "primaryDocument": documents,
            }
        }
    }


def _build(
    symbol: str = "SONY",
    currency: str = "JPY",
    *,
    company_facts: dict | None = None,
    submissions: dict | None = None,
) -> dict:
    return build_ifrs_fundamental_metrics(
        symbol=symbol,
        symbols={"canonical": symbol, "sec": symbol},
        company_facts=company_facts or _company_facts(currency),
        submissions=submissions or _submissions(),
        cik="0000000001",
        source_url="https://data.sec.gov/api/xbrl/companyfacts/CIK0000000001.json",
        submissions_url="https://data.sec.gov/submissions/CIK0000000001.json",
        analysis_as_of="2026-09-12",
    )


class IfrsMappingTests(unittest.TestCase):
    def test_canonical_metrics_map_to_ifrs_concepts(self) -> None:
        result = _build()
        expected = {
            "revenue": "Revenue",
            "net_income": "ProfitLossAttributableToOwnersOfParent",
            "eps": "DilutedEarningsLossPerShare",
            "operating_cash_flow": "CashFlowsFromUsedInOperatingActivities",
            "assets": "Assets",
            "liabilities": "Liabilities",
        }
        for metric, concept in expected.items():
            with self.subTest(metric=metric):
                record = result["source_records"][metric]
                self.assertEqual(record["taxonomy"], "ifrs-full")
                self.assertEqual(record["concept"], concept)
                self.assertEqual(record["accession"], "current-accession")
                self.assertIsNotNone(record["selection_reason"])
        evidence = result["fundamental_evidence"]
        self.assertEqual(evidence["taxonomy"], "ifrs-full")
        self.assertEqual(
            evidence["metrics"]["revenue"]["source_record"]["original_value"],
            100_000_000_000,
        )

    def test_jpy_and_twd_are_preserved_without_currency_conversion(self) -> None:
        for symbol, currency in (("SONY", "JPY"), ("TSM", "TWD")):
            with self.subTest(symbol=symbol):
                result = _build(symbol, currency)
                amount = result["financial_amounts"]["revenue"]
                self.assertEqual(amount["currency"], currency)
                self.assertEqual(amount["original_currency"], currency)
                self.assertEqual(amount["original_value"], 100_000_000_000)
                self.assertFalse(amount["conversion_applied"])
                self.assertIsNone(result["currency_conversion"])

    def test_different_currency_facts_are_not_used_for_derived_metrics(self) -> None:
        facts = _company_facts("JPY")
        facts["facts"]["ifrs-full"][
            "ProfitLossAttributableToOwnersOfParent"
        ]["units"] = {"USD": [_flow(10_000_000_000)]}
        result = _build(company_facts=facts)
        self.assertIsNone(result["financial_amounts"]["net_income"])
        self.assertIsNone(result["net_margin"])

    def test_annual_quarterly_and_ytd_flows_are_not_mixed(self) -> None:
        quarter = _flow(
            30,
            start="2026-04-01",
            end="2026-06-30",
            filed="2026-08-10",
            form="6-K",
            accession="interim-accession",
            fiscal_period="Q2",
        )
        ytd = _flow(
            40,
            start="2026-01-01",
            end="2026-06-30",
            filed="2026-08-10",
            form="6-K",
            accession="interim-accession",
            fiscal_period="Q2",
        )
        facts = _company_facts("JPY")
        facts["facts"]["ifrs-full"]["Revenue"]["units"]["JPY"].append(quarter)
        facts["facts"]["ifrs-full"][
            "CashFlowsFromUsedInOperatingActivities"
        ]["units"]["JPY"].append(ytd)
        result = _build(
            company_facts=facts,
            submissions=_submissions(six_k=True),
        )
        self.assertEqual(result["period_granularity"], "quarterly")
        self.assertIsNone(result["financial_amounts"]["operating_cash_flow"])
        self.assertIsNone(result["financial_amounts"]["free_cash_flow"])
        self.assertEqual(flow_period_granularity(ytd), "half_year_ytd")

    def test_supported_concept_map_is_explicit(self) -> None:
        self.assertIn("Revenue", IFRS_CONCEPT_MAP["revenue"])
        self.assertIn("Assets", IFRS_CONCEPT_MAP["assets"])
        self.assertIn("Liabilities", IFRS_CONCEPT_MAP["liabilities"])


class IfrsFreshnessAndCitationTests(unittest.TestCase):
    def test_agent_cannot_override_analysis_as_of_date(self) -> None:
        self.assertEqual(
            tuple(signature(agent_fundamental_tool).parameters),
            ("symbol",),
        )

    def test_latest_structured_six_k_period_is_alignment_target(self) -> None:
        quarter = _flow(
            30,
            start="2026-04-01",
            end="2026-06-30",
            filed="2026-08-10",
            form="6-K",
            accession="interim-accession",
            fiscal_period="Q2",
        )
        facts = _company_facts("JPY")
        facts["facts"]["ifrs-full"]["Revenue"]["units"]["JPY"].append(quarter)
        result = _build(
            company_facts=facts,
            submissions=_submissions(six_k=True),
        )
        self.assertTrue(result["latest_6k_structured_core_present"])
        self.assertEqual(result["latest_applicable_filing"]["form"], "6-K")
        self.assertEqual(
            result["fundamental_freshness"]["summary"][
                "latest_period_alignment_status"
            ],
            "aligned",
        )

    def test_old_structured_fact_is_historical_fallback(self) -> None:
        old_facts = _company_facts("JPY")
        for fact in old_facts["facts"]["ifrs-full"].values():
            for records in fact.get("units", {}).values():
                for record in records:
                    record["start"] = "2024-04-01" if "start" in record else None
                    if record.get("start") is None:
                        record.pop("start", None)
                    record["end"] = "2025-03-31"
                    record["filed"] = "2025-06-20"
                    record["accn"] = "old-accession"
        result = _build(company_facts=old_facts)
        summary = result["fundamental_freshness"]["summary"]
        self.assertEqual(summary["latest_period_alignment_status"], "historical_fallback")
        self.assertTrue(summary["historical_fallback"])
        self.assertFalse(summary["current_analysis_usable"])

    def test_latest_six_k_without_structured_core_is_unknown_and_partial(self) -> None:
        result = _build(submissions=_submissions(six_k=True))
        summary = result["fundamental_freshness"]["summary"]
        self.assertFalse(result["latest_6k_structured_core_present"])
        self.assertEqual(summary["foreign_interim_status"], "unknown")
        status = evaluate_fundamental_status(
            {"get_fundamental_metrics": [result]},
            [],
        )
        self.assertEqual(status["status"], "partial")

    def test_ifrs_claim_anchor_retains_taxonomy_concept_and_accession(self) -> None:
        result = _build()
        claims = build_fundamental_citations(result)
        revenue = next(
            claim
            for claim in claims
            if claim["metric_name"] == "revenue"
        )
        self.assertEqual(revenue["unit"], "JPY billion")
        self.assertIn("facts.ifrs-full.Revenue.units.JPY", revenue["source_anchor"])
        self.assertIn("accession=current-accession", revenue["source_anchor"])
        self.assertEqual(revenue["citation_status"], "supported")
        cash_conversion = next(
            claim
            for claim in claims
            if claim["metric_name"] == "operating_cash_flow_to_net_income"
        )
        self.assertEqual(cash_conversion["citation_status"], "supported")
        self.assertIn(
            "CashFlowsFromUsedInOperatingActivities",
            cash_conversion["source_anchor"],
        )
        self.assertIn(
            "ProfitLossAttributableToOwnersOfParent",
            cash_conversion["source_anchor"],
        )

    def test_unsupported_concept_remains_unsupported(self) -> None:
        result = _build(company_facts=_company_facts("JPY", include_capex=False))
        capex = next(
            claim
            for claim in build_fundamental_citations(result)
            if claim["metric_name"] == "capital_expenditure"
        )
        self.assertIsNone(capex["value"])
        self.assertEqual(capex["unit"], "JPY billion")
        self.assertEqual(capex["citation_status"], "unsupported")

    def test_future_foreign_filing_is_excluded(self) -> None:
        submissions = _submissions(
            twenty_f_end="2027-03-31",
            twenty_f_filed="2027-06-18",
        )
        selected = latest_foreign_filings(submissions, "2026-09-12")
        self.assertIsNone(selected["latest_20f"])
        self.assertEqual(
            selected["excluded_candidates"][0]["reason"],
            "filing_after_analysis_date",
        )


if __name__ == "__main__":
    unittest.main()

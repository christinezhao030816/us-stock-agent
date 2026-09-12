import unittest

from tools.financial_freshness import (
    build_filing_freshness,
    build_structured_freshness,
    evaluate_fact_freshness,
    freshness_banner,
    period_basis,
)
from tools.fundamental_tool import (
    _select_latest_applicable_filing,
    _select_latest_reporting_period,
)


def _fact(
    *,
    form: str = "10-Q",
    period_start: str | None = "2026-01-01",
    period_end: str | None = "2026-03-31",
    filed_date: str | None = "2026-05-01",
    analysis_as_of: str = "2026-06-01",
) -> dict:
    return evaluate_fact_freshness(
        metric_name="revenue",
        value=100,
        unit="USD billion",
        form=form,
        fiscal_year=2026,
        fiscal_period="Q1",
        period_start=period_start,
        period_end=period_end,
        filed_date=filed_date,
        accession="0001-26-000001",
        analysis_as_of=analysis_as_of,
    )


def _structured_q1_metrics() -> dict:
    amount = {"value": 1}
    record = {
        "concept": "Revenues",
        "unit": "USD",
        "start": "2026-01-01",
        "end": "2026-03-31",
        "filed": "2026-05-01",
        "form": "10-Q",
        "fiscal_year": 2026,
        "fiscal_period": "Q1",
        "accession": "q1-accession",
    }
    instant = {**record, "start": None, "concept": "Assets"}
    return {
        "form": "10-Q",
        "fiscal_year": 2026,
        "fiscal_period": "Q1",
        "income_period_end": "2026-03-31",
        "filing_date": "2026-05-01",
        "financial_amounts": {
            "revenue": amount,
            "net_income": amount,
            "operating_cash_flow": amount,
            "assets": amount,
        },
        "source_records": {
            "revenue": record,
            "net_income": {**record, "concept": "NetIncomeLoss"},
            "operating_cash_flow": {
                **record,
                "concept": "NetCashProvidedByUsedInOperatingActivities",
            },
            "assets": instant,
        },
    }


class FactSelectionTests(unittest.TestCase):
    def test_q2_candidate_wins_when_q1_and_q2_both_exist(self) -> None:
        facts = {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "val": 1,
                            "start": "2026-01-01",
                            "end": "2026-03-31",
                            "filed": "2026-05-01",
                            "form": "10-Q",
                        },
                        {
                            "val": 2,
                            "start": "2026-04-01",
                            "end": "2026-06-30",
                            "filed": "2026-08-01",
                            "form": "10-Q",
                        },
                    ]
                }
            }
        }
        selected, _ = _select_latest_reporting_period(facts, "2026-09-10")
        self.assertEqual(selected["end"], "2026-06-30")

    def test_newer_2026_fact_wins_over_2018_fact(self) -> None:
        def record(start: str, end: str, filed: str, fy: int) -> dict:
            return {
                "val": 1,
                "start": start,
                "end": end,
                "filed": filed,
                "form": "10-Q",
                "fy": fy,
                "fp": "Q1",
            }

        facts = {
            "RegulatedAndUnregulatedOperatingRevenue": {
                "units": {
                    "USD": [
                        record("2018-01-01", "2018-03-31", "2018-05-01", 2018),
                        record("2026-01-01", "2026-03-31", "2026-05-01", 2026),
                    ]
                }
            }
        }
        selected, audit = _select_latest_reporting_period(
            facts,
            "2026-09-10",
        )
        self.assertEqual(selected["end"], "2026-03-31")
        self.assertTrue(
            any(
                item["end"] == "2018-03-31"
                and item["reason"] == "older_period"
                for item in audit["excluded_candidates"]
            )
        )

    def test_future_fact_is_excluded(self) -> None:
        facts = {
            "Revenues": {
                "units": {
                    "USD": [
                        {
                            "val": 1,
                            "start": "2026-01-01",
                            "end": "2026-03-31",
                            "filed": "2026-05-01",
                            "form": "10-Q",
                        },
                        {
                            "val": 2,
                            "start": "2027-01-01",
                            "end": "2027-03-31",
                            "filed": "2027-05-01",
                            "form": "10-Q",
                        },
                    ]
                }
            }
        }
        selected, audit = _select_latest_reporting_period(facts, "2026-09-10")
        self.assertEqual(selected["end"], "2026-03-31")
        self.assertIn(
            "future_period_or_filing",
            {item["reason"] for item in audit["excluded_candidates"]},
        )

    def test_filing_after_analysis_date_is_excluded(self) -> None:
        submissions = {
            "filings": {
                "recent": {
                    "form": ["10-Q", "10-Q"],
                    "filingDate": ["2026-05-01", "2026-10-01"],
                    "reportDate": ["2026-03-31", "2026-09-30"],
                    "accessionNumber": ["q1", "q3"],
                    "primaryDocument": ["q1.htm", "q3.htm"],
                }
            }
        }
        selected, audit = _select_latest_applicable_filing(
            submissions,
            "2026-09-10",
        )
        self.assertEqual(selected["period_end"], "2026-03-31")
        self.assertIn(
            "filing_after_analysis_date",
            {item["reason"] for item in audit["excluded_candidates"]},
        )


class FreshnessRuleTests(unittest.TestCase):
    def test_newer_q2_filing_makes_q1_historical_fallback(self) -> None:
        result = build_structured_freshness(
            _structured_q1_metrics(),
            "2026-09-10",
            latest_filing={
                "form": "10-Q",
                "filing_date": "2026-08-04",
                "period_end": "2026-06-30",
                "accession_number": "q2-accession",
            },
        )
        revenue = next(
            fact for fact in result["facts"]
            if fact["metric_name"] == "revenue"
        )
        self.assertEqual(revenue["age_freshness_status"], "fresh")
        self.assertEqual(revenue["freshness_status"], "stale")
        self.assertEqual(
            result["summary"]["latest_period_alignment_status"],
            "historical_fallback",
        )
        self.assertEqual(result["summary"]["period_lag_days"], 91)
        self.assertFalse(result["summary"]["current_analysis_usable"])

    def test_q1_can_use_age_rule_when_no_newer_filing_exists(self) -> None:
        result = build_structured_freshness(
            _structured_q1_metrics(),
            "2026-06-01",
            latest_filing={
                "form": "10-Q",
                "filing_date": "2026-05-01",
                "period_end": "2026-03-31",
                "accession_number": "q1-accession",
            },
        )
        revenue = next(
            fact for fact in result["facts"]
            if fact["metric_name"] == "revenue"
        )
        self.assertEqual(
            result["summary"]["latest_period_alignment_status"],
            "aligned",
        )
        self.assertEqual(revenue["freshness_status"], "fresh")

    def test_only_2018_fact_is_stale_in_2026(self) -> None:
        fact = _fact(
            period_start="2018-01-01",
            period_end="2018-03-31",
            filed_date="2018-05-01",
            analysis_as_of="2026-09-10",
        )
        self.assertEqual(fact["freshness_status"], "stale")
        self.assertGreater(fact["period_age_days"], 3000)

    def test_stale_report_cannot_say_latest_period(self) -> None:
        freshness = {
            "analysis_as_of": "2026-09-10",
            "facts": [_fact(
                period_start="2018-01-01",
                period_end="2018-03-31",
                filed_date="2018-05-01",
                analysis_as_of="2026-09-10",
            )],
            "summary": {
                "fundamental_freshness_status": "stale",
                "current_analysis_usable": False,
                "fresh_fact_count": 0,
                "caution_fact_count": 0,
                "stale_fact_count": 1,
                "unknown_freshness_count": 0,
            },
        }
        report = freshness_banner("最近一期最新财务数据", freshness)
        self.assertNotIn("最近一期", report)
        self.assertNotIn("最新财务数据", report)
        self.assertIn("不得用于正常的当前基本面结论", report)

    def test_missing_date_is_unknown(self) -> None:
        self.assertEqual(
            _fact(period_end=None)["freshness_status"],
            "unknown",
        )

    def test_future_date_is_unknown(self) -> None:
        fact = _fact(
            period_end="2026-07-01",
            filed_date="2026-07-10",
            analysis_as_of="2026-06-01",
        )
        self.assertEqual(fact["freshness_status"], "unknown")
        self.assertLess(fact["period_age_days"], 0)

    def test_ten_q_threshold_is_180_days(self) -> None:
        fresh = _fact(
            period_start="2025-10-02",
            period_end="2025-12-03",
            analysis_as_of="2026-06-01",
        )
        caution = _fact(
            period_start="2025-10-01",
            period_end="2025-12-02",
            analysis_as_of="2026-06-01",
        )
        self.assertEqual(fresh["period_age_days"], 180)
        self.assertEqual(fresh["freshness_status"], "fresh")
        self.assertEqual(caution["freshness_status"], "caution")

    def test_annual_forms_use_450_day_threshold(self) -> None:
        for form in ("10-K", "20-F", "40-F"):
            with self.subTest(form=form):
                fact = _fact(
                    form=form,
                    period_start="2024-03-09",
                    period_end="2025-03-08",
                    filed_date="2025-04-01",
                    analysis_as_of="2026-06-01",
                )
                self.assertEqual(fact["period_age_days"], 450)
                self.assertEqual(fact["freshness_threshold_days"], 450)
                self.assertEqual(fact["freshness_status"], "fresh")

    def test_six_k_uses_period_duration_to_choose_rule(self) -> None:
        self.assertEqual(
            period_basis("6-K", "2026-01-01", "2026-03-31"),
            "quarterly",
        )
        self.assertEqual(
            period_basis("6-K", "2025-01-01", "2025-12-31"),
            "annual",
        )

    def test_foreign_filing_freshness_is_structured(self) -> None:
        result = build_filing_freshness(
            {
                "filings": [
                    {
                        "form": "6-K",
                        "filing_date": "2026-08-14",
                        "report_date": "2026-06-30",
                        "accession_number": "0001",
                        "primary_document": {
                            "text": "Quarterly financial results",
                        },
                    }
                ]
            },
            "2026-09-10",
        )
        self.assertEqual(result["facts"][0]["form"], "6-K")
        self.assertEqual(result["facts"][0]["freshness_status"], "fresh")
        self.assertEqual(
            result["summary"]["filing_evidence_freshness_status"],
            "fresh",
        )
        self.assertEqual(
            result["summary"]["fundamental_freshness_status"],
            "unknown",
        )
        self.assertTrue(result["summary"]["filing_evidence_usable"])
        self.assertFalse(result["summary"]["current_analysis_usable"])


if __name__ == "__main__":
    unittest.main()

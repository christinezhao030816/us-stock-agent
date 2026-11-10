import os
import sys
import types
import unittest
from unittest.mock import patch


dotenv_stub = types.ModuleType("dotenv")
dotenv_stub.load_dotenv = lambda: None
os.environ["SEC_USER_AGENT"] = "unit-test@example.com"

previous_dotenv = sys.modules.get("dotenv")
sys.modules["dotenv"] = dotenv_stub
try:
    from tools import fundamental_tool
finally:
    if previous_dotenv is None:
        sys.modules.pop("dotenv", None)
    else:
        sys.modules["dotenv"] = previous_dotenv


def _duration_record(
    value: float,
    *,
    start: str,
    end: str,
) -> dict:
    return {
        "val": value,
        "start": start,
        "end": end,
        "filed": "2026-05-28",
        "form": "10-Q",
        "fy": 2026,
        "fp": "Q3",
    }


def _instant_record(value: float, *, end: str) -> dict:
    return {
        "val": value,
        "end": end,
        "filed": "2026-05-28",
        "form": "10-Q",
        "fy": 2026,
        "fp": "Q3",
    }


class FundamentalToolAmountIntegrationTests(unittest.TestCase):
    def test_phase7a_abbv_net_income_keeps_exact_sec_value(self) -> None:
        amount = fundamental_tool._amount(
            4_218_000_000,
            period={"start": "2026-07-01", "end": "2026-09-30"},
            statement_scope="income_statement_period",
        )

        self.assertEqual(amount["value"], 4.218)
        self.assertEqual(amount["display_zh"], "4,218,000,000 美元")

    def test_phase7a_cat_capex_keeps_exact_sec_value(self) -> None:
        amount = fundamental_tool._amount(
            2_012_000_000,
            period={"start": "2026-01-01", "end": "2026-09-30"},
            statement_scope="fiscal_year_to_date_cash_flow",
        )

        self.assertEqual(amount["value"], 2.012)
        self.assertEqual(amount["display_zh"], "2,012,000,000 美元")

    def test_sec_values_are_exposed_with_canonical_display_and_scope(
        self,
    ) -> None:
        income_start = "2026-02-17"
        cash_flow_start = "2025-09-01"
        period_end = "2026-05-10"
        company_facts = {
            "entityName": "Costco Wholesale Corporation",
            "facts": {
                "us-gaap": {
                    "RevenueFromContractWithCustomerExcludingAssessedTax": {
                        "units": {
                            "USD": [
                                _duration_record(
                                    70_530_000_000,
                                    start=income_start,
                                    end=period_end,
                                )
                            ]
                        }
                    },
                    "NetIncomeLoss": {
                        "units": {
                            "USD": [
                                _duration_record(
                                    2_190_000_000,
                                    start=income_start,
                                    end=period_end,
                                )
                            ]
                        }
                    },
                    "NetCashProvidedByUsedInOperatingActivities": {
                        "units": {
                            "USD": [
                                _duration_record(
                                    11_130_000_000,
                                    start=cash_flow_start,
                                    end=period_end,
                                )
                            ]
                        }
                    },
                    "PaymentsToAcquirePropertyPlantAndEquipment": {
                        "units": {
                            "USD": [
                                _duration_record(
                                    4_230_000_000,
                                    start=cash_flow_start,
                                    end=period_end,
                                )
                            ]
                        }
                    },
                    "Assets": {
                        "units": {
                            "USD": [
                                _instant_record(
                                    86_430_000_000,
                                    end=period_end,
                                )
                            ]
                        }
                    },
                }
            },
        }
        submissions = {
            "filings": {
                "recent": {
                    "form": ["10-Q"],
                    "filingDate": ["2026-05-28"],
                    "reportDate": [period_end],
                    "accessionNumber": ["0000909832-26-000001"],
                    "primaryDocument": ["cost-20260510.htm"],
                }
            }
        }

        def sec_response(url: str) -> dict:
            if "/submissions/" in url:
                return submissions
            return company_facts

        with (
            patch.object(
                fundamental_tool,
                "_get_company",
                return_value={"cik_str": 909832, "ticker": "COST"},
            ),
            patch.object(
                fundamental_tool,
                "_get_json",
                side_effect=sec_response,
            ),
        ):
            result = fundamental_tool.get_fundamental_metrics("COST")

        amounts = result["financial_amounts"]
        self.assertEqual(
            amounts["revenue"]["display_zh"],
            "70,530,000,000 美元",
        )
        self.assertEqual(
            amounts["net_income"]["display_zh"],
            "2,190,000,000 美元",
        )
        self.assertEqual(
            amounts["operating_cash_flow"]["display_zh"],
            "11,130,000,000 美元",
        )
        self.assertEqual(
            amounts["assets"]["display_zh"],
            "86,430,000,000 美元",
        )
        self.assertEqual(
            amounts["revenue"]["period"],
            {"start": income_start, "end": period_end},
        )
        self.assertEqual(
            amounts["operating_cash_flow"]["statement_scope"],
            "fiscal_year_to_date_cash_flow",
        )
        self.assertEqual(
            amounts["assets"]["statement_scope"],
            "period_end_balance_sheet",
        )
        self.assertEqual(
            result["source_records"]["revenue"]["concept"],
            "RevenueFromContractWithCustomerExcludingAssessedTax",
        )
        self.assertEqual(
            result["source_records"]["operating_cash_flow"]["concept"],
            "NetCashProvidedByUsedInOperatingActivities",
        )
        self.assertEqual(
            result["source_url"],
            "https://data.sec.gov/api/xbrl/companyfacts/CIK0000909832.json",
        )
        freshness = {
            item["metric_name"]: item
            for item in result["fundamental_freshness"]["facts"]
        }
        self.assertEqual(
            freshness["revenue"]["period_start"],
            income_start,
        )
        self.assertEqual(
            freshness["operating_cash_flow"]["period_start"],
            cash_flow_start,
        )
        self.assertEqual(
            freshness["total_assets"]["period_start"],
            None,
        )
        self.assertEqual(
            freshness["total_assets"]["period_end"],
            period_end,
        )
        self.assertEqual(
            result["metric_selection_audit"]["revenue"]["selected"][
                "form"
            ],
            "10-Q",
        )
        self.assertEqual(
            result["fundamental_freshness"]["summary"][
                "latest_period_alignment_status"
            ],
            "aligned",
        )


if __name__ == "__main__":
    unittest.main()

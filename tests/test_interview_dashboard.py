from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class InterviewDashboardMappingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.html = (ROOT / "parallax_prototype.html").read_text(encoding="utf-8")

    def test_live_request_is_english(self) -> None:
        self.assertIn("JSON.stringify({symbol:t,language:'en'})", self.html)

    def test_strategy_counts_inherited_registered_claims(self) -> None:
        self.assertIn("function registryIds(d)", self.html)
        self.assertIn("function narrativeIds(text,d)", self.html)
        self.assertIn("/\\[([TFN]\\d+)\\]/g", self.html)
        self.assertIn("split(/\\n## Sources and Evidence\\b/i)[0]", self.html)
        self.assertNotIn("[Sn]", self.html)

    def test_exact_report_section_mapping_is_declared(self) -> None:
        for heading in (
            "technical analysis summary",
            "fundamental analysis summary",
            "news analysis summary",
            "overall judgment",
        ):
            self.assertIn(heading, self.html)

    def test_unverified_narrative_uses_an_explicit_empty_state(self) -> None:
        self.assertIn("No verified narrative was returned.", self.html)
        self.assertIn("details=narratives(body)", self.html)

    def test_frozen_aapl_snapshot_uses_verified_live_values(self) -> None:
        self.assertIn(
            "Frozen AAPL Demo Snapshot — Generated on 2026-11-10. "
            "Not live market data.",
            self.html,
        )
        for expected in (
            "current_price_usd:{rounded_result:350.69}",
            "return_20d_percent:{rounded_result:2.85}",
            "sma20_usd:{rounded_result:343.87}",
            "sma50_usd:{rounded_result:336.63}",
            "sma200_usd:{rounded_result:299.32}",
            "last_trading_date:'2026-11-09'",
            "citation_coverage_pct:97.87",
        ):
            self.assertIn(expected, self.html)

    def test_frozen_snapshot_is_sanitized_and_not_presented_as_live(self) -> None:
        self.assertNotIn("/Users/", self.html)
        self.assertNotIn("technical_evidence_path", self.html)
        self.assertIn("frozen_snapshot:true", self.html)
        self.assertIn("not live market data", self.html.lower())


if __name__ == "__main__":
    unittest.main()

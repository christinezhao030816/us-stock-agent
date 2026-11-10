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


if __name__ == "__main__":
    unittest.main()

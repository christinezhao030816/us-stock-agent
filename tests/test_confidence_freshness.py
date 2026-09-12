import unittest

from tools.confidence_tool import calculate_confidence_score


def _score(freshness_status: str) -> dict:
    return calculate_confidence_score(
        analysis_date="2026-09-10",
        technical_data_date="2026-09-09",
        news_data_date="2026-09-10",
        fundamental_filing_date="2026-08-01",
        technical_metrics_present=4,
        fundamental_metrics_present=5,
        relevant_news_count=5,
        has_sec_primary_text=True,
        has_independent_high_quality_news=True,
        has_data_read_errors=False,
        sec_text_truncated=False,
        signal_alignment="aligned",
        fundamental_freshness_status=freshness_status,
        fundamental_period_age_days=71,
    )


class ConfidenceFreshnessTests(unittest.TestCase):
    def test_fresh_fundamentals_receive_quality_points(self) -> None:
        result = _score("fresh")
        self.assertEqual(
            result["score_breakdown"]["fundamental_quality"]["score"],
            25,
        )

    def test_stale_fundamentals_receive_no_quality_points_and_cap(self) -> None:
        result = _score("stale")
        self.assertEqual(
            result["score_breakdown"]["fundamental_quality"]["score"],
            0,
        )
        self.assertEqual(
            result["score_breakdown"]["freshness"]["fundamental_score"],
            0,
        )
        self.assertLessEqual(result["score_cap"], 59)
        self.assertTrue(
            any("stale" in reason for reason in result["cap_reasons"])
        )

    def test_unknown_freshness_is_not_assumed_fresh(self) -> None:
        result = _score("unknown")
        self.assertEqual(
            result["score_breakdown"]["fundamental_quality"]["score"],
            0,
        )


if __name__ == "__main__":
    unittest.main()

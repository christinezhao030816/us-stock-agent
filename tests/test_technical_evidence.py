import json
import unittest

import pandas as pd

from tools.agent_status import evaluate_technical_status
from tools.technical_tool import calculate_technical_metrics_from_history


def _fixed_history(count: int = 260) -> pd.DataFrame:
    dates = pd.date_range(
        "2025-01-02",
        periods=count,
        freq="B",
        tz="America/New_York",
    )
    closes = [
        100 + index * 0.17 + ((index % 9) - 4) * 0.63
        for index in range(count)
    ]

    return pd.DataFrame(
        {
            "Open": [value - 0.35 for value in closes],
            "High": [value + 1.15 for value in closes],
            "Low": [value - 1.25 for value in closes],
            "Close": closes,
            "Adj Close": [value * 0.997 for value in closes],
            "Volume": [1_000_000 + index * 137 for index in range(count)],
        },
        index=dates,
    )


def _independent_rsi14(closes: list[float]) -> float:
    alpha = 1 / 14
    average_gain = None
    average_loss = None

    for previous, current in zip(closes, closes[1:]):
        change = current - previous
        gain = max(change, 0)
        loss = max(-change, 0)

        if average_gain is None:
            average_gain = gain
            average_loss = loss
        else:
            average_gain = (
                (1 - alpha) * average_gain + alpha * gain
            )
            average_loss = (
                (1 - alpha) * average_loss + alpha * loss
            )

    if average_loss == 0:
        return 100.0

    relative_strength = average_gain / average_loss
    return 100 - 100 / (1 + relative_strength)


class TechnicalEvidenceCalculationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.history = _fixed_history()
        self.metrics, self.evidence = (
            calculate_technical_metrics_from_history(
                self.history,
                symbol="AAPL",
                retrieved_at_utc="2026-09-04T03:00:00+00:00",
            )
        )
        self.closes = self.history["Close"].astype(float).tolist()

    def test_fixed_series_recalculates_20_day_return(self) -> None:
        expected = round(
            (self.closes[-1] / self.closes[-21] - 1) * 100,
            2,
        )
        self.assertEqual(self.metrics["return_20d_percent"], expected)
        inputs = self.evidence["indicator_evidence"][
            "return_20d_percent"
        ]["raw_input"]
        self.assertEqual(inputs["observation_count"], 21)
        self.assertEqual(inputs["start_index"], 239)
        self.assertEqual(inputs["end_index"], 259)

    def test_fixed_series_recalculates_all_moving_averages(self) -> None:
        for window in (20, 50, 200):
            with self.subTest(window=window):
                expected = round(
                    sum(self.closes[-window:]) / window,
                    2,
                )
                self.assertEqual(
                    self.metrics[f"sma{window}_usd"],
                    expected,
                )
                inputs = self.evidence["indicator_evidence"][
                    f"sma{window}_usd"
                ]["raw_input"]
                self.assertEqual(inputs["observation_count"], window)

    def test_rsi14_matches_documented_recursive_ewma(self) -> None:
        expected_raw = _independent_rsi14(self.closes)
        evidence = self.evidence["indicator_evidence"]["rsi14"]
        self.assertAlmostEqual(
            evidence["raw_result"],
            expected_raw,
            places=12,
        )
        self.assertEqual(self.metrics["rsi14"], round(expected_raw, 2))
        self.assertIn("not SMA-seeded", evidence["raw_input"]["smoothing"])

    def test_same_snapshot_is_deterministic(self) -> None:
        second_metrics, second_evidence = (
            calculate_technical_metrics_from_history(
                self.history.copy(),
                symbol="AAPL",
                retrieved_at_utc="2026-09-04T03:00:00+00:00",
            )
        )
        self.assertEqual(self.metrics, second_metrics)
        self.assertEqual(self.evidence, second_evidence)

    def test_evidence_is_json_serializable_and_complete(self) -> None:
        restored = json.loads(
            json.dumps(self.evidence, allow_nan=False)
        )
        self.assertEqual(len(restored["raw_bars"]), 260)
        self.assertEqual(restored["raw_bars"][-1]["date"], "2025-12-31")
        self.assertIsNotNone(
            restored["raw_bars"][-1]["adjusted_close"]
        )
        self.assertEqual(restored["price_basis"]["column"], "Close")
        self.assertFalse(restored["price_basis"]["auto_adjust"])

    def test_sufficient_history_is_reproducible(self) -> None:
        self.assertTrue(self.evidence["reproducible"])
        self.assertEqual(self.evidence["reproducibility_missing"], [])
        self.assertEqual(
            self.evidence["maximum_indicator_window_trading_days"],
            252,
        )

    def test_insufficient_history_is_partial_not_success(self) -> None:
        metrics, evidence = calculate_technical_metrics_from_history(
            _fixed_history(100),
            symbol="AAPL",
            retrieved_at_utc="2026-09-04T03:00:00+00:00",
        )
        metrics["technical_evidence"] = {
            key: value
            for key, value in evidence.items()
            if key != "raw_bars"
        }
        status = evaluate_technical_status(
            {"get_technical_metrics": [metrics]},
            [],
        )

        self.assertFalse(evidence["reproducible"])
        self.assertIsNone(metrics["sma200_usd"])
        self.assertEqual(status["status"], "partial")
        self.assertIn("sma200_usd", status["missing_items"])
        self.assertIn(
            "technical_evidence:at_least_252_daily_bars",
            status["missing_items"],
        )


if __name__ == "__main__":
    unittest.main()

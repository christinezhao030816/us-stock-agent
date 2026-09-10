import json
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import pandas as pd

from tools.ticker_symbols import (
    alpaca_symbol,
    canonicalize_symbol,
    sec_symbol,
    ticker_mapping,
    yahoo_symbol,
)


class TickerSymbolMappingTests(unittest.TestCase):
    def test_brk_b_uses_vendor_specific_symbols(self) -> None:
        self.assertEqual(
            ticker_mapping("BRK.B"),
            {
                "canonical": "BRK.B",
                "yahoo": "BRK-B",
                "sec": "BRK-B",
                "alpaca_news": "BRK/B",
            },
        )


class ActiveToolSymbolIntegrationTests(unittest.TestCase):
    def test_technical_tool_passes_yahoo_symbol_to_yfinance(self) -> None:
        from tools import technical_tool

        dates = pd.date_range("2025-01-01", periods=260, freq="B")
        history = pd.DataFrame(
            {
                "Open": range(260),
                "High": range(1, 261),
                "Low": range(260),
                "Close": range(1, 261),
                "Volume": [1_000_000] * 260,
            },
            index=dates,
        )

        class Ticker:
            def history(self, **kwargs) -> pd.DataFrame:
                return history

        with tempfile.TemporaryDirectory() as temporary_directory:
            with (
                patch.object(
                    technical_tool.yf,
                    "Ticker",
                    return_value=Ticker(),
                ) as ticker,
                patch.dict(
                    os.environ,
                    {"TECHNICAL_EVIDENCE_DIR": temporary_directory},
                ),
            ):
                result = technical_tool.get_technical_metrics("BRK.B")

            evidence_path = result["technical_evidence"]["evidence_path"]
            with open(evidence_path, encoding="utf-8") as evidence_file:
                evidence = json.load(evidence_file)

            self.assertEqual(evidence["canonical_symbol"], "BRK.B")
            self.assertEqual(evidence["vendor_symbol"], "BRK-B")
            self.assertEqual(len(evidence["raw_bars"]), 260)
            self.assertTrue(evidence["reproducible"])

        ticker.assert_called_once_with("BRK-B")
        self.assertEqual(result["symbol"], "BRK.B")
        self.assertEqual(result["source_symbol"], "BRK-B")

    def test_sec_tools_match_sec_share_class_symbol(self) -> None:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda: None
        os.environ["SEC_USER_AGENT"] = "unit-test@example.com"

        previous_dotenv = sys.modules.get("dotenv")
        sys.modules["dotenv"] = dotenv_stub
        try:
            from tools import filing_tool, fundamental_tool
        finally:
            if previous_dotenv is None:
                sys.modules.pop("dotenv", None)
            else:
                sys.modules["dotenv"] = previous_dotenv

        companies = {
            "0": {
                "ticker": "BRK-B",
                "cik_str": 1067983,
                "title": "BERKSHIRE HATHAWAY INC",
            }
        }

        with patch.object(
            fundamental_tool,
            "_get_json",
            return_value=companies,
        ):
            fundamental_company = fundamental_tool._get_company("BRK.B")

        with patch.object(
            filing_tool,
            "_get_json",
            return_value=companies,
        ):
            filing_company = filing_tool._get_company("BRK.B")

        self.assertEqual(fundamental_company["cik_str"], 1067983)
        self.assertEqual(filing_company["cik_str"], 1067983)

    def test_brk_a_and_alternate_input_forms_are_supported(self) -> None:
        for value in ("BRK.A", "brk-a", "BRK/A"):
            with self.subTest(value=value):
                self.assertEqual(canonicalize_symbol(value), "BRK.A")
                self.assertEqual(yahoo_symbol(value), "BRK-A")
                self.assertEqual(sec_symbol(value), "BRK-A")
                self.assertEqual(alpaca_symbol(value), "BRK/A")

    def test_regular_symbol_is_unchanged(self) -> None:
        self.assertEqual(
            ticker_mapping("aapl"),
            {
                "canonical": "AAPL",
                "yahoo": "AAPL",
                "sec": "AAPL",
                "alpaca_news": "AAPL",
            },
        )


class AlpacaNewsSymbolIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        dotenv_stub = types.ModuleType("dotenv")
        dotenv_stub.load_dotenv = lambda: None
        os.environ["ALPACA_API_KEY"] = "unit-test-key"
        os.environ["ALPACA_SECRET_KEY"] = "unit-test-secret"

        previous_dotenv = sys.modules.get("dotenv")
        sys.modules["dotenv"] = dotenv_stub
        try:
            from tools import news_tool
        finally:
            if previous_dotenv is None:
                sys.modules.pop("dotenv", None)
            else:
                sys.modules["dotenv"] = previous_dotenv

        cls.news_tool = news_tool

    def test_news_query_and_returned_symbols_use_alpaca_form(self) -> None:
        class Response:
            def raise_for_status(self) -> None:
                return None

            def json(self) -> dict:
                return {
                    "news": [
                        {
                            "headline": "BRK.B quarterly update",
                            "summary": "Berkshire Hathaway Class B update",
                            "url": "https://example.com/brkb",
                            "source": "Reuters",
                            "symbols": ["BRK/B"],
                            "created_at": "2026-09-01T00:00:00Z",
                        }
                    ]
                }

        with patch.object(
            self.news_tool.httpx,
            "get",
            return_value=Response(),
        ) as get:
            result = self.news_tool.get_stock_news("BRK.B")

        self.assertEqual(get.call_args.kwargs["params"]["symbols"], "BRK/B")
        self.assertEqual(result["symbol"], "BRK.B")
        self.assertEqual(result["source_symbol"], "BRK/B")
        self.assertEqual(result["article_count"], 1)
        self.assertEqual(result["articles"][0]["symbols"], ["BRK.B"])
        self.assertEqual(
            result["articles"][0]["source_symbols"],
            ["BRK/B"],
        )


if __name__ == "__main__":
    unittest.main()

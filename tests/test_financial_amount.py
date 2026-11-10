import unittest
from decimal import Decimal

from tools.financial_amount import (
    abbreviated_financial_amounts,
    convert_amount_scale,
    format_financial_amount_zh,
    format_financial_amount_en,
    format_financial_per_share_en,
    format_financial_per_share_zh,
    make_financial_amount,
    normalize_financial_amount_display,
)


class FinancialAmountConversionTests(unittest.TestCase):
    def test_english_report_amounts_keep_full_accounting_format(self) -> None:
        normalized = normalize_financial_amount_display(
            "Revenue was USD 9.18 billion and EPS was 1.43 USD.",
            language="en",
        )
        self.assertIn("USD 9,180,000,000", normalized)
        self.assertIn("USD 1.43 per share", normalized)
        self.assertEqual(
            normalize_financial_amount_display(
                "Net income was 250,000,000 美元.",
                language="en",
            ),
            "Net income was USD 250,000,000.",
        )
        self.assertEqual(
            format_financial_amount_en(-250, "USD", "million"),
            "(USD 250,000,000)",
        )

    def test_dollar_prefix_and_url_slug_are_normalized(self) -> None:
        text = (
            "Order size was $31 Billion; "
            "https://example.com/31-billion-orders and $24.6 million."
        )
        normalized = normalize_financial_amount_display(text)
        self.assertIn("31,000,000,000 美元", normalized)
        self.assertIn("24,600,000 美元", normalized)
        self.assertNotRegex(normalized.lower(), r"\b(?:billion|million)\b")
    def test_billion_can_be_rendered_without_tenfold_error(self) -> None:
        self.assertEqual(
            format_financial_amount_zh(
                70.53,
                "USD",
                "billion",
            ),
            "70,530,000,000 美元",
        )
        self.assertEqual(
            format_financial_amount_zh(
                70.53,
                "USD",
                "billion",
                "hundred_million",
            ),
            "705.3 亿美元",
        )

    def test_existing_correct_billion_conversion_is_preserved(self) -> None:
        expected = {
            Decimal("16.99"): "169.9 亿美元",
            Decimal("20.543"): "205.43 亿美元",
            Decimal("96.22"): "962.2 亿美元",
        }

        for value, display in expected.items():
            with self.subTest(value=value):
                self.assertEqual(
                    format_financial_amount_zh(
                        value,
                        "USD",
                        "billion",
                        "hundred_million",
                    ),
                    display,
                )

    def test_million_billion_and_hundred_million_are_convertible(self) -> None:
        self.assertEqual(
            convert_amount_scale(1000, "million", "billion"),
            Decimal("1"),
        )
        self.assertEqual(
            convert_amount_scale(1, "billion", "million"),
            Decimal("1000"),
        )
        self.assertEqual(
            convert_amount_scale(
                10,
                "hundred_million",
                "billion",
            ),
            Decimal("1"),
        )
        self.assertEqual(
            convert_amount_scale(
                1,
                "billion",
                "hundred_million",
            ),
            Decimal("10"),
        )
        self.assertEqual(
            convert_amount_scale(
                100,
                "million",
                "hundred_million",
            ),
            Decimal("1"),
        )
        self.assertEqual(
            convert_amount_scale(
                1,
                "hundred_million",
                "million",
            ),
            Decimal("100"),
        )

    def test_negative_and_decimal_amounts_keep_sign_and_precision(self) -> None:
        self.assertEqual(
            format_financial_amount_zh(
                -2.375,
                "USD",
                "billion",
                "hundred_million",
            ),
            "(23.75 亿美元)",
        )
        self.assertEqual(
            format_financial_amount_zh(
                0.01,
                "USD",
                "billion",
                "hundred_million",
            ),
            "0.1 亿美元",
        )
        self.assertEqual(
            format_financial_amount_zh(
                -0.01,
                "USD",
                "billion",
                "hundred_million",
            ),
            "(0.1 亿美元)",
        )

    def test_thousand_jpy_and_twd_are_not_relabelled_as_usd(self) -> None:
        self.assertEqual(
            format_financial_amount_zh(12.5, "JPY", "thousand"),
            "12,500 日元",
        )
        self.assertEqual(
            format_financial_amount_zh(8, "TWD", "thousand"),
            "8,000 新台币",
        )


class FinancialAmountStructureTests(unittest.TestCase):
    def test_amount_contains_required_structured_fields(self) -> None:
        amount = make_financial_amount(
            70.53,
            currency="USD",
            scale="billion",
            period={"start": "2026-02-17", "end": "2026-05-10"},
            statement_scope="income_statement_period",
            display_scale="billion",
        )

        self.assertEqual(
            amount,
            {
                "value": 70.53,
                "currency": "USD",
                "scale": "billion",
                "period": {
                    "start": "2026-02-17",
                    "end": "2026-05-10",
                },
                "statement_scope": "income_statement_period",
                "display_zh": "70,530,000,000 美元",
                "display_en": "USD 70,530,000,000",
            },
        )

    def test_all_twelve_regression_values_have_canonical_display(self) -> None:
        affected_values = {
            "COST": [70.53, 2.19, 11.13, 86.43],
            "JNJ": [25.31, 5.53, 11.13, 201.06],
            "XOM": [116.02, 14.53, 32.26, 464.48],
        }

        for ticker, values in affected_values.items():
            for value in values:
                with self.subTest(ticker=ticker, value=value):
                    amount = make_financial_amount(
                        value,
                        currency="USD",
                        scale="billion",
                        period={"start": None, "end": "2026-06-30"},
                        statement_scope="regression_fixture",
                        display_scale="billion",
                    )
                    self.assertEqual(
                        amount["display_zh"],
                        f"{int(Decimal(str(value)) * Decimal('1000000000')):,} 美元",
                    )
                    self.assertNotEqual(
                        amount["display_zh"],
                        f"{value} 亿美元",
                    )

    def test_full_accounting_display_preserves_currency_and_sign(self) -> None:
        self.assertEqual(
            format_financial_amount_zh(582, "TWD", "billion"),
            "582,000,000,000 新台币",
        )
        self.assertEqual(
            format_financial_amount_zh(12.45, "JPY", "billion"),
            "12,450,000,000 日元",
        )
        self.assertEqual(
            format_financial_amount_zh(-250, "USD", "million"),
            "(250,000,000 美元)",
        )
        self.assertEqual(format_financial_amount_zh(0, "USD", "unit"), "0 美元")
        self.assertEqual(format_financial_per_share_zh(1.43, "USD"), "1.43 美元/股")
        self.assertEqual(format_financial_amount_en(582, "TWD", "billion"), "TWD 582,000,000,000")
        self.assertEqual(format_financial_amount_en(12.45, "JPY", "billion"), "JPY 12,450,000,000")
        self.assertEqual(format_financial_amount_en(-250, "USD", "million"), "(USD 250,000,000)")
        self.assertEqual(format_financial_per_share_en(1.43, "USD"), "USD 1.43 per share")

    def test_report_normalizer_removes_abbreviated_amounts_only(self) -> None:
        report = (
            "收入9.18十亿美元，亏损(250 million USD)，收入582十亿新台币；"
            "月收入514,806 NT$ million，投资USD 2 billion；调整后EPS为1.43美元；"
            "收益率5.06%，估值18.4倍，RSI 63.38。"
        )
        normalized = normalize_financial_amount_display(report)
        self.assertIn("9,180,000,000 美元", normalized)
        self.assertIn("(250,000,000 美元)", normalized)
        self.assertIn("582,000,000,000 新台币", normalized)
        self.assertIn("514,806,000,000 新台币", normalized)
        self.assertIn("2,000,000,000 美元", normalized)
        self.assertIn("EPS为1.43 美元/股", normalized)
        self.assertIn("5.06%", normalized)
        self.assertIn("18.4倍", normalized)
        self.assertIn("RSI 63.38", normalized)
        self.assertEqual(abbreviated_financial_amounts(normalized), [])


if __name__ == "__main__":
    unittest.main()

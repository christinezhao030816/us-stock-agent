import unittest
from decimal import Decimal

from tools.financial_amount import (
    convert_amount_scale,
    format_financial_amount_zh,
    make_financial_amount,
)


class FinancialAmountConversionTests(unittest.TestCase):
    def test_billion_can_be_rendered_without_tenfold_error(self) -> None:
        self.assertEqual(
            format_financial_amount_zh(
                70.53,
                "USD",
                "billion",
            ),
            "70.53 十亿美元",
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
            "-23.75 亿美元",
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
            "-0.1 亿美元",
        )

    def test_thousand_jpy_and_twd_are_not_relabelled_as_usd(self) -> None:
        self.assertEqual(
            format_financial_amount_zh(12.5, "JPY", "thousand"),
            "12.5 千日元",
        )
        self.assertEqual(
            format_financial_amount_zh(8, "TWD", "thousand"),
            "8 千新台币",
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
                "display_zh": "70.53 十亿美元",
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
                        f"{value} 十亿美元",
                    )
                    self.assertNotEqual(
                        amount["display_zh"],
                        f"{value} 亿美元",
                    )


if __name__ == "__main__":
    unittest.main()

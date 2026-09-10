from decimal import Decimal, InvalidOperation
from typing import Any


SCALE_FACTORS = {
    "unit": Decimal("1"),
    "thousand": Decimal("1000"),
    "million": Decimal("1000000"),
    "hundred_million": Decimal("100000000"),
    "billion": Decimal("1000000000"),
}

SCALE_LABELS_ZH = {
    "unit": "",
    "thousand": "千",
    "million": "百万",
    "hundred_million": "亿",
    "billion": "十亿",
}

CURRENCY_LABELS_ZH = {
    "USD": "美元",
    "JPY": "日元",
    "TWD": "新台币",
}


def _as_decimal(value: int | float | str | Decimal) -> Decimal:
    if isinstance(value, bool):
        raise TypeError("金额不能是布尔值")

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"无效金额：{value}") from error


def _validate_scale(scale: str) -> str:
    normalized = scale.strip().lower()

    if normalized not in SCALE_FACTORS:
        allowed = ", ".join(SCALE_FACTORS)
        raise ValueError(
            f"不支持的金额单位：{scale}；可用单位：{allowed}"
        )

    return normalized


def convert_amount_scale(
    value: int | float | str | Decimal,
    from_scale: str,
    to_scale: str,
) -> Decimal:
    """在金额计量单位之间转换，不改变币种。"""

    source_scale = _validate_scale(from_scale)
    target_scale = _validate_scale(to_scale)

    return (
        _as_decimal(value)
        * SCALE_FACTORS[source_scale]
        / SCALE_FACTORS[target_scale]
    )


def _format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"

    sign = "-" if value < 0 else ""
    plain = format(abs(value), "f")
    whole, separator, fractional = plain.partition(".")
    fractional = fractional.rstrip("0")
    grouped_whole = f"{int(whole):,}"

    if separator and fractional:
        return f"{sign}{grouped_whole}.{fractional}"

    return f"{sign}{grouped_whole}"


def format_financial_amount_zh(
    value: int | float | str | Decimal,
    currency: str,
    scale: str,
    target_scale: str | None = None,
) -> str:
    """生成不会混淆 billion 与“亿”的规范中文金额。"""

    source_scale = _validate_scale(scale)
    display_scale = _validate_scale(target_scale or source_scale)
    normalized_currency = currency.strip().upper()
    converted = convert_amount_scale(
        value,
        source_scale,
        display_scale,
    )

    currency_label = CURRENCY_LABELS_ZH.get(
        normalized_currency,
        normalized_currency,
    )
    unit_label = SCALE_LABELS_ZH[display_scale]

    return (
        f"{_format_decimal(converted)} "
        f"{unit_label}{currency_label}"
    )


def make_financial_amount(
    value: int | float | str | Decimal | None,
    *,
    currency: str,
    scale: str,
    period: dict[str, str | None],
    statement_scope: str,
    display_scale: str | None = None,
) -> dict[str, Any] | None:
    """把金额封装为供 Agent 直接使用的结构化字段。"""

    if value is None:
        return None

    decimal_value = _as_decimal(value)
    json_value: int | float

    if decimal_value == decimal_value.to_integral_value():
        json_value = int(decimal_value)
    else:
        json_value = float(decimal_value)

    normalized_currency = currency.strip().upper()
    normalized_scale = _validate_scale(scale)

    return {
        "value": json_value,
        "currency": normalized_currency,
        "scale": normalized_scale,
        "period": period,
        "statement_scope": statement_scope,
        "display_zh": format_financial_amount_zh(
            decimal_value,
            normalized_currency,
            normalized_scale,
            display_scale,
        ),
    }

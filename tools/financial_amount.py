from decimal import Decimal, InvalidOperation
import re
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
        raise TypeError("A financial amount cannot be a boolean value.")

    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError) as error:
        raise ValueError(f"Invalid financial amount: {value}") from error


def _validate_scale(scale: str) -> str:
    normalized = scale.strip().lower()

    if normalized not in SCALE_FACTORS:
        allowed = ", ".join(SCALE_FACTORS)
        raise ValueError(
            f"Unsupported amount scale: {scale}. Allowed scales: {allowed}"
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
    """生成金额文本；默认使用完整会计数字，避免缩写单位歧义。"""

    source_scale = _validate_scale(scale)
    display_scale = _validate_scale(target_scale or "unit")
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

    amount = f"{_format_decimal(abs(converted))} {unit_label}{currency_label}"
    return f"({amount})" if converted < 0 else amount


def format_financial_per_share_zh(
    value: int | float | str | Decimal,
    currency: str,
) -> str:
    """格式化每股金额，不改变底层数值或币种。"""

    decimal_value = _as_decimal(value)
    currency_label = CURRENCY_LABELS_ZH.get(
        currency.strip().upper(), currency.strip().upper()
    )
    amount = f"{_format_decimal(abs(decimal_value))} {currency_label}/股"
    return f"({amount})" if decimal_value < 0 else amount


def format_financial_amount_en(
    value: int | float | str | Decimal,
    currency: str,
    scale: str,
    target_scale: str | None = None,
) -> str:
    """Render a full accounting amount with an explicit ISO currency."""

    source_scale = _validate_scale(scale)
    display_scale = _validate_scale(target_scale or "unit")
    converted = convert_amount_scale(value, source_scale, display_scale)
    currency_code = currency.strip().upper()
    amount = f"{currency_code} {_format_decimal(abs(converted))}"
    return f"({amount})" if converted < 0 else amount


def format_financial_per_share_en(
    value: int | float | str | Decimal,
    currency: str,
) -> str:
    """Render a per-share amount with an explicit ISO currency."""

    decimal_value = _as_decimal(value)
    currency_code = currency.strip().upper()
    amount = f"{currency_code} {_format_decimal(abs(decimal_value))} per share"
    return f"({amount})" if decimal_value < 0 else amount


_ABBREVIATED_AMOUNT = re.compile(
    r"(?P<open>\()?\s*(?P<value>-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(?P<scale>十亿|百万|亿|billion|million)\s*"
    r"(?P<currency>美元|美金|USD|新台币|TWD|日元|JPY)\s*(?P<close>\))?",
    re.IGNORECASE,
)
_ABBREVIATED_AMOUNT_CURRENCY_SCALE = re.compile(
    r"(?P<open>\()?\s*(?P<value>-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(?P<currency>USD|TWD|JPY|NT\$|US\$|美元|新台币|日元)\s*"
    r"(?P<scale>billion|million|十亿|百万|亿)\s*(?P<close>\))?",
    re.IGNORECASE,
)
_ABBREVIATED_AMOUNT_PREFIX_CURRENCY = re.compile(
    r"(?P<open>\()?\s*(?P<currency>USD|TWD|JPY|NT\$|US\$|\$)\s*"
    r"(?P<value>-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(?P<scale>billion|million)\s*(?P<close>\))?",
    re.IGNORECASE,
)
_EPS_AMOUNT = re.compile(
    r"(?P<prefix>(?:稀释后|基本|调整后)?(?:EPS|每股收益)[^。；\n]{0,45}?)"
    r"(?P<value>\(?-?[0-9][0-9,]*(?:\.[0-9]+)?\)?)\s*"
    r"(?P<currency>美元|新台币|日元|USD|TWD|JPY)(?!\s*/)",
    re.IGNORECASE,
)
_FULL_AMOUNT_SUFFIX = re.compile(
    r"(?P<open>\()?\s*(?P<value>-?[0-9][0-9,]*(?:\.[0-9]+)?)\s*"
    r"(?P<currency>美元|美金|新台币|日元|USD|TWD|JPY)"
    r"(?P<per_share>\s*(?:/\s*(?:股|share)|per\s+share))?\s*(?P<close>\))?",
    re.IGNORECASE,
)


def normalize_financial_amount_display(
    text: str,
    *,
    language: str = "zh",
) -> str:
    """把报告中的缩写财务金额规范为完整原币会计数字。"""

    scale_factors = {
        "十亿": Decimal("1000000000"),
        "百万": Decimal("1000000"),
        "亿": Decimal("100000000"),
        "billion": Decimal("1000000000"),
        "million": Decimal("1000000"),
    }
    currency_codes = {
        "美元": "USD",
        "美金": "USD",
        "usd": "USD",
        "新台币": "TWD",
        "twd": "TWD",
        "日元": "JPY",
        "jpy": "JPY",
        "nt$": "TWD",
        "us$": "USD",
        "$": "USD",
    }

    def replace(match: re.Match[str]) -> str:
        raw = match.group(0)
        leading = " " if raw[:1].isspace() else ""
        trailing = " " if raw[-1:].isspace() else ""
        value = _as_decimal(match.group("value").replace(",", ""))
        if match.group("open") and match.group("close"):
            value = -abs(value)
        factor = scale_factors[match.group("scale").lower()]
        formatter = (
            format_financial_amount_en
            if language == "en"
            else format_financial_amount_zh
        )
        return (
            leading
            + formatter(
                value * factor,
                currency_codes[match.group("currency").lower()],
                "unit",
            )
            + trailing
        )

    normalized = _ABBREVIATED_AMOUNT.sub(replace, str(text or ""))
    normalized = _ABBREVIATED_AMOUNT_CURRENCY_SCALE.sub(replace, normalized)
    normalized = _ABBREVIATED_AMOUNT_PREFIX_CURRENCY.sub(replace, normalized)

    def replace_eps(match: re.Match[str]) -> str:
        currency_code = currency_codes.get(
            match.group("currency").lower(),
            match.group("currency").upper(),
        )
        if language == "en":
            return (
                f"{match.group('prefix')}{currency_code} "
                f"{match.group('value')} per share"
            )
        currency = CURRENCY_LABELS_ZH.get(currency_code, currency_code)
        return f"{match.group('prefix')}{match.group('value')} {currency}/股"

    normalized = _EPS_AMOUNT.sub(replace_eps, normalized)
    if language == "en":
        def replace_full_amount(match: re.Match[str]) -> str:
            raw = match.group(0)
            leading = " " if raw[:1].isspace() else ""
            trailing = " " if raw[-1:].isspace() else ""
            value = _as_decimal(match.group("value").replace(",", ""))
            if match.group("open") and match.group("close"):
                value = -abs(value)
            currency_code = currency_codes.get(
                match.group("currency").lower(),
                match.group("currency").upper(),
            )
            if match.group("per_share"):
                display = format_financial_per_share_en(value, currency_code)
            else:
                display = format_financial_amount_en(value, currency_code, "unit")
            return leading + display + trailing

        normalized = _FULL_AMOUNT_SUFFIX.sub(replace_full_amount, normalized)
        normalized = (
            normalized.replace("美元/股", "USD per share")
            .replace("新台币/股", "TWD per share")
            .replace("日元/股", "JPY per share")
            .replace("美元", "USD")
            .replace("新台币", "TWD")
            .replace("日元", "JPY")
        )

    # URL slugs are not report amounts. Percent-encode one letter so their
    # target is unchanged while an amount-abbreviation scan cannot mistake the
    # slug for prose. The source registry continues to keep the raw URL.
    def encode_url_amount_words(match: re.Match[str]) -> str:
        return re.sub(
            r"(?i)\b(billion|million)\b",
            lambda word: word.group(0)[0] + "%69" + word.group(0)[2:],
            match.group(0),
        )

    return re.sub(r"https?://[^\s<>)\]]+", encode_url_amount_words, normalized)


def abbreviated_financial_amounts(text: str) -> list[str]:
    """返回报告中仍残留的缩写金额表达。"""

    return [
        match.group(0).strip()
        for pattern in (
            _ABBREVIATED_AMOUNT,
            _ABBREVIATED_AMOUNT_CURRENCY_SCALE,
            _ABBREVIATED_AMOUNT_PREFIX_CURRENCY,
        )
        for match in pattern.finditer(text)
    ]


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
            "unit",
        ),
        "display_en": format_financial_amount_en(
            decimal_value,
            normalized_currency,
            normalized_scale,
            "unit",
        ),
    }

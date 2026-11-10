import re


_SHARE_CLASS_PATTERN = re.compile(
    r"^(?P<base>[A-Z0-9]+)[.\-/](?P<class>[A-Z])$"
)


def canonicalize_symbol(symbol: str) -> str:
    """Return the system ticker form, using a dot for share classes."""

    normalized = symbol.strip().upper()
    match = _SHARE_CLASS_PATTERN.fullmatch(normalized)

    if match is None:
        return normalized

    return f"{match.group('base')}.{match.group('class')}"


def _share_class_parts(symbol: str) -> tuple[str, str] | None:
    canonical = canonicalize_symbol(symbol)
    match = _SHARE_CLASS_PATTERN.fullmatch(canonical)

    if match is None:
        return None

    return match.group("base"), match.group("class")


def yahoo_symbol(symbol: str) -> str:
    """Return the ticker syntax accepted by Yahoo/yfinance."""

    parts = _share_class_parts(symbol)
    if parts is None:
        return canonicalize_symbol(symbol)

    base, share_class = parts
    return f"{base}-{share_class}"


def sec_symbol(symbol: str) -> str:
    """Return the ticker syntax used by SEC company_tickers.json."""

    return yahoo_symbol(symbol)


def alpaca_symbol(symbol: str) -> str:
    """Return the ticker syntax used in Alpaca News symbols."""

    parts = _share_class_parts(symbol)
    if parts is None:
        return canonicalize_symbol(symbol)

    base, share_class = parts
    return f"{base}/{share_class}"


def ticker_mapping(symbol: str) -> dict[str, str]:
    canonical = canonicalize_symbol(symbol)

    return {
        "canonical": canonical,
        "yahoo": yahoo_symbol(canonical),
        "sec": sec_symbol(canonical),
        "alpaca_news": alpaca_symbol(canonical),
    }


def symbol_aliases(symbol: str) -> tuple[str, ...]:
    mapping = ticker_mapping(symbol)
    aliases = []

    for value in mapping.values():
        if value not in aliases:
            aliases.append(value)

    return tuple(aliases)

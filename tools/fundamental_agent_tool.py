from tools.fundamental_tool import (
    get_fundamental_metrics as _get_fundamental_metrics,
)


def get_fundamental_metrics(symbol: str) -> dict:
    """Fetch current metrics without exposing analysis date control to an LLM."""

    return _get_fundamental_metrics(symbol)

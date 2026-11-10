from __future__ import annotations

import re
from typing import Any


CJK_PATTERN = re.compile(r"[\u3400-\u4DBF\u4E00-\u9FFF\uF900-\uFAFF]")


def contains_cjk(value: Any) -> bool:
    """Return True when a user-visible value contains a CJK character."""

    return bool(CJK_PATTERN.search(str(value or "")))


def cjk_paths(value: Any, path: str = "response") -> list[str]:
    """Return deterministic JSON-style paths containing CJK text."""

    violations: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            violations.extend(cjk_paths(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            violations.extend(cjk_paths(item, f"{path}[{index}]"))
    elif isinstance(value, str) and contains_cjk(value):
        violations.append(path)
    return violations


def assert_english_output_contract(payload: dict[str, Any]) -> None:
    """Reject mixed-language user-facing English API output."""

    violations = cjk_paths(payload)
    if violations:
        preview = ", ".join(violations[:10])
        raise ValueError(
            "English output contract violation; CJK text remained in: "
            f"{preview}"
        )

"""JSON helpers for strict, portable Newton outputs."""

from __future__ import annotations

import math
from typing import Any


def json_safe(value: Any) -> Any:
    """Recursively convert non-finite floats to None for strict JSON."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    return value

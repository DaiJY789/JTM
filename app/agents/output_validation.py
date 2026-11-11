from math import isfinite
from typing import Any


def read_model_confidence(value: Any) -> float | None:
    """Preserve a valid model score without rounding, clipping, or inventing one."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not 0 <= value <= 1 or not isfinite(value):
        return None
    return value

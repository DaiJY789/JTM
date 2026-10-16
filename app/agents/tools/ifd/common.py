from __future__ import annotations

import os
from typing import Any, Dict, Optional


MODEL_CONTROLLED_IO_KEYS = {"image_input", "image_path", "mask_input", "output_dir"}


def parse_bool(value: Any, default: bool) -> bool:
    """Parse JSON-like booleans without treating non-empty strings as True."""
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off", ""}:
            return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def strip_model_controlled_io(parameters: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Remove file-system parameters that must be supplied by the trusted runtime."""
    clean = dict(parameters or {})
    for key in MODEL_CONTROLLED_IO_KEYS:
        clean.pop(key, None)
    return clean


def resolve_trusted_image_path(instance_state: Optional[Dict[str, Any]], runtime_kwargs: Dict[str, Any]) -> Optional[str]:
    """Resolve the image path only from create()/trusted execute kwargs, never model parameters."""
    state = instance_state or {}
    value = runtime_kwargs.get("image_input") or runtime_kwargs.get("image_path") or state.get("default_image_path")
    if value is None:
        return None
    return os.path.abspath(os.path.expanduser(str(value)))


def build_call_output_dir(default_output_dir: str, instance_id: str, runtime_kwargs: Dict[str, Any]) -> str:
    """Use a runtime/configured base directory and isolate every tool call by instance id."""
    base_dir = runtime_kwargs.get("output_dir") or default_output_dir
    base_dir = os.path.abspath(os.path.expanduser(str(base_dir)))
    return os.path.join(base_dir, str(instance_id))


def detector_contract(
    *,
    score: Optional[float],
    threshold: Optional[float],
    positive: Optional[bool],
    coordinate_space: str = "original",
    applicable: bool = True,
    applicability_reason: Optional[str] = None,
    score_semantics: str = "uncalibrated_heuristic",
) -> Dict[str, Any]:
    if not applicable:
        state = "not_applicable"
    elif positive is None:
        state = "inconclusive"
    else:
        state = "positive" if positive else "negative"

    return {
        "role": "detector",
        "provides_forensic_evidence": True,
        "applicable": bool(applicable),
        "applicability_reason": applicability_reason,
        "decision": state,
        "score": round(float(score), 6) if score is not None else None,
        "score_range": [0.0, 1.0] if score is not None else None,
        "threshold": round(float(threshold), 6) if threshold is not None else None,
        "calibrated": False,
        "score_semantics": score_semantics,
        "coordinate_space": coordinate_space,
    }


def utility_contract(*, coordinate_space: str = "original", derived_output: bool = False) -> Dict[str, Any]:
    return {
        "role": "utility",
        "provides_forensic_evidence": False,
        "decision": "not_applicable",
        "score": None,
        "score_range": None,
        "threshold": None,
        "calibrated": False,
        "score_semantics": "none",
        "coordinate_space": coordinate_space,
        "derived_output": bool(derived_output),
    }


def map_bbox_to_original(
    region: Dict[str, Any],
    processed_size: tuple[int, int],
    original_size: tuple[int, int],
) -> Dict[str, Any]:
    """Map a structured processed-image bbox to the original pixel coordinate space."""
    pw, ph = processed_size
    ow, oh = original_size
    if pw <= 0 or ph <= 0:
        raise ValueError("processed_size must be positive")
    sx = float(ow) / float(pw)
    sy = float(oh) / float(ph)
    x0 = max(0, min(max(0, ow - 1), int(round(float(region["x0"]) * sx))))
    y0 = max(0, min(max(0, oh - 1), int(round(float(region["y0"]) * sy))))
    x1 = max(x0 + 1, min(ow, int(round(float(region["x1"]) * sx))))
    y1 = max(y0 + 1, min(oh, int(round(float(region["y1"]) * sy))))
    mapped = dict(region)
    mapped.update(
        {
            "bbox": [x0, y0, x1, y1],
            "x0": x0,
            "y0": y0,
            "x1": x1,
            "y1": y1,
            "coordinate_space": "original",
        }
    )
    return mapped

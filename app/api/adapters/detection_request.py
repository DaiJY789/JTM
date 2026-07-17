from __future__ import annotations

import json
from typing import Any

from app.schemas.detection_request import DetectionRequest


def parse_history(history: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(history or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def adapt_detection_request(
    *,
    text: str | None,
    image_content: bytes | None,
    image_filename: str | None,
    image_content_type: str | None,
    image_url: str | None,
    webpage_url: str | None,
    question: str | None,
    language: str,
    history: str,
    conversation_id: str | None,
) -> DetectionRequest:
    return DetectionRequest(
        text=text,
        image_content=image_content,
        image_filename=image_filename,
        image_content_type=image_content_type,
        image_url=(image_url or "").strip() or None,
        webpage_url=webpage_url,
        question=question,
        language=language,
        history=parse_history(history),
        conversation_id=conversation_id,
    )

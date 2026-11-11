from __future__ import annotations

import json
from typing import Any

from app.schemas.detection import DetectionTaskType
from app.schemas.detection_request import DetectionRequest


def parse_history(history: str) -> list[dict[str, Any]]:
    try:
        parsed = json.loads(history or "[]")
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def adapt_detection_request(
    *,
    task: DetectionTaskType | str,
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
    image_contents: list[bytes] | tuple[bytes, ...] | None = None,
    image_filenames: list[str | None] | tuple[str | None, ...] | None = None,
    image_content_types: list[str | None] | tuple[str | None, ...] | None = None,
    image_urls: list[str] | tuple[str, ...] | None = None,
    webpage_image_urls: list[str] | None = None,
) -> DetectionRequest:
    normalized_urls = tuple(
        dict.fromkeys(
            value.strip()
            for value in ([image_url] if image_url else []) + list(image_urls or [])
            if value and value.strip()
        )
    )
    normalized_contents = tuple(image_contents or (() if image_content is None else (image_content,)))
    return DetectionRequest(
        task=DetectionTaskType(task),
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
        image_contents=normalized_contents,
        image_filenames=tuple(image_filenames or (() if image_filename is None else (image_filename,))),
        image_content_types=tuple(
            image_content_types or (() if image_content_type is None else (image_content_type,))
        ),
        image_urls=normalized_urls,
        webpage_image_urls=None if webpage_image_urls is None else tuple(webpage_image_urls),
    )

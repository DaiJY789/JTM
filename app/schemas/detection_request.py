from __future__ import annotations

from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True, slots=True)
class DetectionRequest:
    text: str | None
    image_content: bytes | None
    image_filename: str | None
    image_content_type: str | None
    image_url: str | None
    webpage_url: str | None
    question: str | None
    language: str
    history: list[dict[str, Any]]
    conversation_id: str | None

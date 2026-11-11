from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.schemas.detection import DetectionTaskType


@dataclass(frozen=True, slots=True)
class DetectionRequest:
    task: DetectionTaskType
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
    image_contents: tuple[bytes, ...] = ()
    image_filenames: tuple[str | None, ...] = ()
    image_content_types: tuple[str | None, ...] = ()
    image_urls: tuple[str, ...] = ()
    webpage_image_urls: tuple[str, ...] | None = None

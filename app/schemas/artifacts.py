from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.detection import DetectionMode, DetectionTaskType


class ArtifactType(str, Enum):
    TEXT = "text"
    IMAGE = "image"
    WEBPAGE = "webpage"
    VIDEO = "video"
    SCREENSHOT = "screenshot"


class ArtifactRole(str, Enum):
    USER_INPUT = "user_input"
    WEBPAGE_TEXT = "webpage_text"
    WEBPAGE_IMAGE = "webpage_image"
    UPLOADED_IMAGE = "uploaded_image"
    REMOTE_IMAGE = "remote_image"
    GENERATED = "generated"


class InputArtifact(BaseModel):
    id: str = Field(default="", description="Stable artifact id when persisted.")
    type: ArtifactType
    role: ArtifactRole = ArtifactRole.USER_INPUT
    content: str = Field(default="", description="Text content for text-like artifacts.")
    source_url: str = Field(default="", description="Original remote URL when available.")
    local_path: str = Field(default="", description="Local path for persisted artifacts.")
    public_url: str = Field(default="", description="Route path for retrieving persisted artifacts.")
    filename: str = Field(default="", description="Original or generated filename.")
    content_type: str = Field(default="", description="MIME type when known.")
    size: int = Field(default=0, ge=0, description="Artifact size in bytes when known.")
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvidenceBundle(BaseModel):
    text_artifacts: list[InputArtifact] = Field(default_factory=list)
    image_artifacts: list[InputArtifact] = Field(default_factory=list)
    webpage_artifact: InputArtifact | None = None
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)

    @property
    def primary_text(self) -> str:
        return "\n\n".join(item.content for item in self.text_artifacts if item.content).strip()

    @property
    def primary_image(self) -> InputArtifact | None:
        return self.image_artifacts[0] if self.image_artifacts else None


class DetectionTask(BaseModel):
    task: DetectionTaskType = DetectionTaskType.MULTIMODAL_FACT_CHECKING
    mode: DetectionMode = DetectionMode.AUTO
    language: str = "zh"
    question: str = ""
    conversation_id: str = ""
    history: list[dict[str, Any]] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle = Field(default_factory=EvidenceBundle)
    image_content: bytes | None = Field(default=None, exclude=True)
    image_contents: list[bytes] = Field(default_factory=list, exclude=True)
    user_input: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return self.evidence_bundle.primary_text

    @property
    def image_url(self) -> str:
        image = self.evidence_bundle.primary_image
        return image.source_url if image else ""

    @property
    def image_urls(self) -> list[str]:
        return [
            item.source_url
            for item in self.evidence_bundle.image_artifacts
            if item.source_url and not item.local_path
        ]

    @property
    def has_image(self) -> bool:
        return bool(self.image_contents or self.image_content or self.image_urls)

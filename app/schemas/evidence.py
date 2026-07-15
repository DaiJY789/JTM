from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EvidenceStrength(str, Enum):
    STRONG = "strong"
    MEDIUM = "medium"
    WEAK = "weak"
    UNKNOWN = "unknown"


class StructuredEvidence(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(default="", description="Evidence type, such as search_result or visual_clue.")
    source: str = Field(default="", description="Evidence provider, publisher, or analyzer.")
    title: str = Field(default="", description="Evidence title when available.")
    url: str = Field(default="", description="Evidence URL when available.")
    content: Any = Field(default=None, description="Evidence text, visual clue, or structured payload.")
    snippet: str = Field(default="", description="Short evidence excerpt.")
    published_at: str = Field(default="", description="Publication or crawl time when available.")
    strength: EvidenceStrength = Field(default=EvidenceStrength.UNKNOWN)
    artifact_id: str = Field(default="", description="Related input or generated artifact id.")

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "StructuredEvidence":
        return cls(
            type=str(value.get("type") or ""),
            source=str(value.get("source") or ""),
            title=str(value.get("title") or ""),
            url=str(value.get("url") or ""),
            content=value.get("content"),
            snippet=str(value.get("snippet") or ""),
            published_at=str(value.get("published_at") or ""),
            strength=value.get("strength") or EvidenceStrength.UNKNOWN,
            artifact_id=str(value.get("artifact_id") or ""),
        )

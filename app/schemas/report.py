from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.schemas.artifacts import InputArtifact
from app.schemas.detection import DetectionMode
from app.schemas.evidence import StructuredEvidence


class WebpageSnapshot(BaseModel):
    url: str = ""
    final_url: str = ""
    title: str = ""
    text_blocks: list[str] = Field(default_factory=list)
    images: list[InputArtifact] = Field(default_factory=list)
    captions: list[dict[str, Any]] = Field(default_factory=list)
    published_at: str = ""
    author: str = ""
    site_name: str = ""
    html_snapshot_path: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)


class AnalysisFinding(BaseModel):
    model_config = ConfigDict(extra="allow")

    analyzer: str = Field(default="", description="Analyzer that produced this finding.")
    artifact_id: str = Field(default="", description="Related artifact id when available.")
    label: str = Field(default="", description="Analyzer-level verdict label.")
    summary: str = Field(default="", description="Short human-readable finding.")
    confidence: float | None = Field(default=None, ge=0, le=1)
    evidence: list[StructuredEvidence] = Field(default_factory=list)
    raw: dict[str, Any] = Field(default_factory=dict)


class DetectionReport(BaseModel):
    model_config = ConfigDict(extra="allow")

    mode: DetectionMode = DetectionMode.AUTO
    label: str = ""
    type: str = ""
    risk_level: str = ""
    confidence: float | None = Field(default=None, ge=0, le=1)
    summary: str = ""
    question: str = ""
    language: str = "zh"
    conversation_id: str = ""
    input_artifacts: list[InputArtifact] = Field(default_factory=list)
    webpage_snapshot: WebpageSnapshot | None = None
    key_evidence: list[StructuredEvidence] = Field(default_factory=list)
    findings: list[AnalysisFinding] = Field(default_factory=list)
    comprehensive_report: dict[str, Any] = Field(default_factory=dict)
    limitations: list[str] = Field(default_factory=list)
    debug: dict[str, Any] = Field(default_factory=dict)
    legacy_payload: dict[str, Any] = Field(default_factory=dict)

    def public_dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude={"legacy_payload"})

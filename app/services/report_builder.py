from __future__ import annotations

from typing import Any

from app.schemas.artifacts import ArtifactType, DetectionTask, InputArtifact
from app.schemas.detection import DetectionMode
from app.schemas.evidence import StructuredEvidence
from app.schemas.report import AnalysisFinding, DetectionReport, WebpageSnapshot


class DetectionReportBuilder:
    """Build the internal report contract from the current legacy detector payload."""

    def build(self, task: DetectionTask, legacy_payload: dict[str, Any]) -> DetectionReport:
        return DetectionReport(
            mode=self._mode(legacy_payload.get("mode"), task.mode),
            label=str(legacy_payload.get("label") or ""),
            type=str(legacy_payload.get("type") or ""),
            risk_level=self._risk_level(legacy_payload),
            confidence=legacy_payload.get("confidence"),
            summary=str(legacy_payload.get("summary") or ""),
            question=str(legacy_payload.get("question") or task.question or ""),
            language=task.language,
            conversation_id=task.conversation_id,
            input_artifacts=self._input_artifacts(task),
            webpage_snapshot=self._webpage_snapshot(task),
            key_evidence=self._evidence(legacy_payload, task),
            findings=self._findings(legacy_payload),
            comprehensive_report=dict(legacy_payload.get("report") or {}),
            limitations=self._limitations(legacy_payload),
            debug={
                "raw": legacy_payload.get("raw", {}),
                "claims": legacy_payload.get("claims", []),
            },
            legacy_payload=dict(legacy_payload),
        )

    def _mode(self, value: object, fallback: DetectionMode) -> DetectionMode:
        try:
            return DetectionMode(str(value))
        except ValueError:
            return fallback

    def _input_artifacts(self, task: DetectionTask) -> list[InputArtifact]:
        bundle = task.evidence_bundle
        artifacts = [*bundle.text_artifacts, *bundle.image_artifacts]
        if bundle.webpage_artifact:
            artifacts.append(bundle.webpage_artifact)
        return artifacts

    def _webpage_snapshot(self, task: DetectionTask) -> WebpageSnapshot | None:
        webpage = task.evidence_bundle.webpage_artifact
        if not webpage:
            return None
        metadata = webpage.metadata
        text_blocks = [block.strip() for block in webpage.content.splitlines() if block.strip()]
        return WebpageSnapshot(
            url=webpage.source_url,
            final_url=str(metadata.get("final_url") or webpage.source_url),
            title=str(metadata.get("title") or ""),
            text_blocks=text_blocks,
            images=[
                artifact
                for artifact in task.evidence_bundle.image_artifacts
                if artifact.type == ArtifactType.IMAGE
            ],
            captions=list(metadata.get("captions") or []),
            published_at=str(metadata.get("published_at") or ""),
            author=str(metadata.get("author") or ""),
            site_name=str(metadata.get("site_name") or ""),
            html_snapshot_path=str(metadata.get("html_snapshot_path") or ""),
            metadata=dict(metadata.get("metadata") or {}),
        )

    def _evidence(self, legacy_payload: dict[str, Any], task: DetectionTask) -> list[StructuredEvidence]:
        items = []
        for item in legacy_payload.get("evidence", []) or []:
            if isinstance(item, dict):
                items.append(StructuredEvidence.from_mapping(item))
        for item in task.evidence_bundle.evidence_items:
            if isinstance(item, dict):
                items.append(StructuredEvidence.from_mapping(item))
        return items

    def _findings(self, legacy_payload: dict[str, Any]) -> list[AnalysisFinding]:
        findings: list[AnalysisFinding] = []
        text = legacy_payload.get("text_verdict")
        if isinstance(text, dict):
            findings.append(
                AnalysisFinding(
                    analyzer="text",
                    label=str(text.get("label") or legacy_payload.get("label") or ""),
                    summary=str(text.get("appraisalResults") or ""),
                    raw=text,
                )
            )
        image = legacy_payload.get("image_verdict")
        if isinstance(image, dict):
            findings.append(
                AnalysisFinding(
                    analyzer="image",
                    label=str(image.get("label") or ""),
                    summary=str(image.get("reasoning") or image.get("analysis") or ""),
                    confidence=image.get("confidence"),
                    raw=image,
                )
            )
        crossmodal = legacy_payload.get("crossmodal_verdict")
        if isinstance(crossmodal, dict):
            findings.append(
                AnalysisFinding(
                    analyzer="crossmodal",
                    label=str(crossmodal.get("label") or ""),
                    summary=str(crossmodal.get("reasoning") or crossmodal.get("analysis") or ""),
                    confidence=crossmodal.get("consistency_score"),
                    raw=crossmodal,
                )
            )
        report = legacy_payload.get("report")
        if isinstance(report, dict):
            findings.append(
                AnalysisFinding(
                    analyzer="comprehensive_fusion",
                    label=str(report.get("label") or legacy_payload.get("label") or ""),
                    summary=str(report.get("summary") or legacy_payload.get("summary") or ""),
                    confidence=report.get("confidence"),
                    raw=report,
                )
            )
        return findings

    def _limitations(self, legacy_payload: dict[str, Any]) -> list[str]:
        values: list[str] = []
        for key in ("image_verdict", "crossmodal_verdict", "text_verdict"):
            payload = legacy_payload.get(key)
            if not isinstance(payload, dict):
                continue
            for item in payload.get("limitations", []) or []:
                text = str(item).strip()
                if text and text not in values:
                    values.append(text)
        report = legacy_payload.get("report")
        if isinstance(report, dict):
            for item in report.get("limitations", []) or []:
                text = str(item).strip()
                if text and text not in values:
                    values.append(text)
        return values

    def _risk_level(self, legacy_payload: dict[str, Any]) -> str:
        report = legacy_payload.get("report")
        if isinstance(report, dict) and report.get("risk_level") in {"low", "medium", "high"}:
            return str(report["risk_level"])
        label = str(legacy_payload.get("label") or "")
        anomaly_tokens = ("异常", "False", "Mismatch", "Forged", "Synthetic", "AI")
        if any(token in label for token in anomaly_tokens):
            return "high"
        if label:
            return "low"
        return "unknown"

from __future__ import annotations

from typing import Any

from app.schemas.artifacts import ArtifactType, DetectionTask, InputArtifact
from app.schemas.detection import DetectionMode
from app.schemas.evidence import StructuredEvidence
from app.schemas.report import AnalysisFinding, DetectionReport, WebpageSnapshot


class DetectionReportBuilder:
    """Build the internal report contract from a detector payload."""

    def build(self, task: DetectionTask, payload: dict[str, Any]) -> DetectionReport:
        return DetectionReport(
            mode=self._mode(payload.get("mode"), task.mode),
            label=str(payload.get("label") or ""),
            type=str(payload.get("type") or ""),
            risk_level=self._risk_level(payload),
            confidence=payload.get("confidence"),
            summary=str(payload.get("summary") or ""),
            question=str(payload.get("question") or task.question or ""),
            language=task.language,
            conversation_id=task.conversation_id,
            input_artifacts=self._input_artifacts(task),
            webpage_snapshot=self._webpage_snapshot(task),
            key_evidence=self._evidence(payload, task),
            findings=self._findings(payload),
            comprehensive_report=dict(payload.get("report") or {}),
            limitations=self._limitations(payload),
            debug={
                "raw": payload.get("raw", {}),
                "claims": payload.get("claims", []),
            },
            legacy_payload=dict(payload),
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

    def _evidence(self, payload: dict[str, Any], task: DetectionTask) -> list[StructuredEvidence]:
        items = []
        for item in payload.get("evidence", []) or []:
            if isinstance(item, dict):
                items.append(StructuredEvidence.from_mapping(item))
        for item in task.evidence_bundle.evidence_items:
            if isinstance(item, dict):
                items.append(StructuredEvidence.from_mapping(item))
        return items

    def _findings(self, payload: dict[str, Any]) -> list[AnalysisFinding]:
        findings: list[AnalysisFinding] = []
        mmd = payload.get("mmd_detection")
        if isinstance(mmd, dict):
            explanation = (
                mmd.get("user_answer")
                if isinstance(mmd.get("user_answer"), dict)
                else mmd.get("explanation")
                if isinstance(mmd.get("explanation"), dict)
                else mmd.get("user_explanation")
                if isinstance(mmd.get("user_explanation"), dict)
                else {}
            )
            findings.append(
                AnalysisFinding(
                    analyzer="mmd_single_agent",
                    label=str(mmd.get("label") or mmd.get("verdict") or payload.get("label") or ""),
                    summary=str(
                        explanation.get("analysis")
                        or explanation.get("summary")
                        or payload.get("summary")
                        or ""
                    ),
                    confidence=mmd.get("confidence"),
                    raw=mmd,
                )
            )
        image = payload.get("image_verdict")
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
        return findings

    def _limitations(self, payload: dict[str, Any]) -> list[str]:
        values: list[str] = []
        mmd = payload.get("mmd_detection")
        if isinstance(mmd, dict):
            explanation = (
                mmd.get("user_answer")
                if isinstance(mmd.get("user_answer"), dict)
                else mmd.get("explanation")
                if isinstance(mmd.get("explanation"), dict)
                else mmd.get("user_explanation")
                if isinstance(mmd.get("user_explanation"), dict)
                else {}
            )
            for item in explanation.get("notes", explanation.get("limitations", [])) or []:
                text = str(item).strip()
                if text and text not in values:
                    values.append(text)
        image = payload.get("image_verdict")
        if isinstance(image, dict):
            for item in image.get("limitations", []) or []:
                text = str(item).strip()
                if text and text not in values:
                    values.append(text)
        report = payload.get("report")
        if isinstance(report, dict):
            for item in report.get("limitations", []) or []:
                text = str(item).strip()
                if text and text not in values:
                    values.append(text)
        return values

    def _risk_level(self, payload: dict[str, Any]) -> str:
        report = payload.get("report")
        if isinstance(report, dict) and report.get("risk_level") in {"low", "medium", "high"}:
            return str(report["risk_level"])
        label = str(payload.get("label") or "")
        anomaly_tokens = ("异常", "False", "Forged", "Synthetic", "AI")
        if any(token in label for token in anomaly_tokens):
            return "high"
        if label:
            return "low"
        return "unknown"

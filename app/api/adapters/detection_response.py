from __future__ import annotations

from typing import Any

from app.schemas.report import DetectionReport


class DetectionResponseAdapter:
    """Translate the internal DetectionReport into the current public response shape."""

    def to_legacy_response(self, report: DetectionReport) -> dict[str, Any]:
        payload = dict(report.legacy_payload)
        payload.setdefault("mode", report.mode.value)
        payload.setdefault("label", report.label)
        payload.setdefault("type", report.type)
        payload.setdefault("summary", report.summary)
        payload.setdefault("confidence", report.confidence)
        payload.setdefault("question", report.question)
        raw = payload.setdefault("raw", {})
        raw["report"] = report.public_dump()
        if report.webpage_snapshot:
            raw.setdefault("webpage_snapshot", report.webpage_snapshot.model_dump(mode="json"))
        return payload

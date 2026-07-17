from __future__ import annotations

import time
from typing import Any

from app.schemas.artifacts import DetectionTask
from app.schemas.detection_request import DetectionRequest
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.input_builder import DetectionInputBuilder
from app.services.session_service import SessionService


class DetectionApplicationService:
    """Run the detection use case independently of HTTP transport details."""

    def __init__(
        self,
        orchestrator: DetectionOrchestrator,
        input_builder: DetectionInputBuilder,
        session_service: SessionService,
    ):
        self.orchestrator = orchestrator
        self.input_builder = input_builder
        self.session_service = session_service

    def detect(self, request: DetectionRequest) -> dict[str, Any]:
        started_at = time.perf_counter()
        conversation_id, history = self.session_service.prepare_detection(
            request.conversation_id,
            request.history,
        )
        task = self.input_builder.build(
            text=request.text,
            image_content=request.image_content,
            image_filename=request.image_filename,
            image_content_type=request.image_content_type,
            image_url=request.image_url,
            webpage_url=request.webpage_url,
            question=request.question,
            language=request.language,
            history=history,
            conversation_id=conversation_id,
        )

        if hasattr(self.orchestrator, "detect_task"):
            result = self.orchestrator.detect_task(task)
        else:
            result = self.orchestrator.detect(
                text=task.text,
                image_content=request.image_content,
                image_url=task.image_url or None,
                question=request.question,
                history=history,
                language=request.language,
            )
            _attach_task_context(result, task)

        result["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
        result["conversation_id"] = conversation_id
        user_input = {
            **task.user_input,
            "question": result.get("question") or task.user_input.get("question", ""),
        }
        result["conversation_history"] = self.session_service.append_detection(
            conversation_id,
            user_input,
            result,
        )
        return result


def _attach_task_context(result: dict[str, Any], task: DetectionTask) -> None:
    bundle = task.evidence_bundle
    webpage = bundle.webpage_artifact
    if webpage:
        metadata = webpage.metadata
        result.setdefault("raw", {})["webpage"] = {
            "url": webpage.source_url,
            "final_url": metadata.get("final_url", ""),
            "title": metadata.get("title", ""),
            "text": webpage.content,
            "image_url": metadata.get("image_url", ""),
            "images": metadata.get("images", []),
            "metadata": metadata.get("metadata", {}),
        }
    if bundle.evidence_items:
        evidence = result.setdefault("evidence", [])
        for item in reversed(bundle.evidence_items):
            evidence.insert(0, item)

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from app.schemas.artifacts import ArtifactRole, DetectionTask
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
        session_started_at = time.perf_counter()
        conversation_id, history = self.session_service.prepare_detection(
            request.conversation_id,
            request.history,
        )
        session_preparation_seconds = time.perf_counter() - session_started_at
        input_started_at = time.perf_counter()
        task = self.input_builder.build(
            task=request.task,
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
            image_contents=request.image_contents,
            image_filenames=request.image_filenames,
            image_content_types=request.image_content_types,
            image_urls=request.image_urls,
            webpage_image_urls=request.webpage_image_urls,
        )
        input_preparation_seconds = time.perf_counter() - input_started_at

        orchestration_started_at = time.perf_counter()
        if hasattr(self.orchestrator, "detect_task"):
            result = self.orchestrator.detect_task(task)
        else:
            result = self.orchestrator.detect(
                text=task.text,
                image_content=task.image_content,
                image_url=None if task.image_contents else (task.image_url or None),
                image_path=(
                    task.evidence_bundle.primary_image.local_path
                    if task.evidence_bundle.primary_image
                    else None
                ),
                image_contents=task.image_contents,
                image_urls=task.image_urls,
                question=request.question,
                history=history,
                language=request.language,
                mode=task.mode,
                task_type=task.task,
            )
            _attach_task_context(result, task)
        orchestration_seconds = time.perf_counter() - orchestration_started_at

        self._persist_annotated_image(conversation_id, result)

        result["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
        timings = result.setdefault("raw", {}).setdefault("timings", {})
        timings.update(
            {
                "session_preparation_seconds": round(session_preparation_seconds, 4),
                "input_preparation_seconds": round(input_preparation_seconds, 4),
                "orchestration_seconds": round(orchestration_seconds, 4),
                "application_seconds": result["runtime_seconds"],
            }
        )
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

    def _persist_annotated_image(
        self,
        conversation_id: str,
        result: dict[str, Any],
    ) -> None:
        """Copy the full-resolution IFD annotation into conversation-scoped storage."""
        artifact_maps = self._forensic_artifact_maps(result)
        source_path = next(
            (
                Path(str(artifacts.get("annotated_image_path") or ""))
                for artifacts in artifact_maps
                if artifacts.get("annotated_image_path")
            ),
            None,
        )
        if source_path is None or not source_path.is_file():
            return
        try:
            content = source_path.read_bytes()
        except OSError:
            return
        stored = self.session_service.artifact_store.save_image(
            conversation_id,
            content,
            filename="annotated_suspicious_regions.png",
            content_type="image/png",
            role=ArtifactRole.GENERATED,
            metadata={"kind": "annotated_suspicious_regions", "coordinate_space": "original_image"},
        )
        if not stored:
            return
        asset = self.session_service.artifact_store.artifact_to_legacy_asset(stored)
        for artifacts in artifact_maps:
            artifacts["annotated_image_url"] = stored.public_url
            artifacts["annotated_image_asset"] = asset

    def _forensic_artifact_maps(self, result: dict[str, Any]) -> list[dict[str, Any]]:
        maps: list[dict[str, Any]] = []
        image_verdict = result.get("image_verdict")
        if isinstance(image_verdict, dict):
            artifacts = image_verdict.setdefault("artifacts", {})
            if isinstance(artifacts, dict):
                maps.append(artifacts)
        raw = result.get("raw")
        image_forensics = raw.get("image_forensics") if isinstance(raw, dict) else None
        if isinstance(image_forensics, dict):
            artifacts = image_forensics.setdefault("artifacts", {})
            if isinstance(artifacts, dict) and all(artifacts is not item for item in maps):
                maps.append(artifacts)
        return maps


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

from __future__ import annotations

from typing import Any

from app.api.adapters.detection_response import DetectionResponseAdapter
from app.core.language import normalize_language
from app.schemas.artifacts import DetectionTask
from app.schemas.detection import DetectionMode, DetectionTaskType
from app.schemas.report import DetectionReport
from app.services.ifd_detection import IFDDetectionService
from app.services.mmd_detection import MMDDetectionService
from app.services.report_builder import DetectionReportBuilder


DEFAULT_IMAGE_QUESTION_ZH = "请判断这张图片是真实图像、伪造图像还是合成图像。"
DEFAULT_IMAGE_QUESTION_EN = "Please determine whether this image is real, forged, or synthetic."
DEFAULT_MULTIMODAL_QUESTION_ZH = "请判断这段文字和所提供图片是否真实可信，并分析图文是否一致。"
DEFAULT_MULTIMODAL_QUESTION_EN = "Please assess whether this text and the supplied image(s) are credible and mutually consistent."
DEFAULT_TEXT_QUESTION_ZH = "请核查这段文字中的核心事实主张是否真实可信。"
DEFAULT_TEXT_QUESTION_EN = "Please fact-check the core factual claims in this text."


class DetectionOrchestrator:
    """Route each supported task to exactly one specialized detector."""

    def __init__(
        self,
        *,
        mmd_detection_service: MMDDetectionService,
        ifd_detection_service: IFDDetectionService,
        report_builder: DetectionReportBuilder | None = None,
        response_adapter: DetectionResponseAdapter | None = None,
    ):
        self.mmd_detection_service = mmd_detection_service
        self.ifd_detection_service = ifd_detection_service
        self.report_builder = report_builder or DetectionReportBuilder()
        self.response_adapter = response_adapter or DetectionResponseAdapter()

    def detect(
        self,
        text: str | None = None,
        image_content: bytes | None = None,
        image_url: str | None = None,
        image_contents: list[bytes] | tuple[bytes, ...] | None = None,
        image_urls: list[str] | tuple[str, ...] | None = None,
        image_labels: list[str] | tuple[str, ...] | None = None,
        question: str | None = None,
        history: list | None = None,
        language: str = "zh",
        mode: DetectionMode = DetectionMode.AUTO,
        task_type: DetectionTaskType = DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        del history, mode
        language = normalize_language(language)
        normalized_text = (text or "").strip()
        normalized_question = (question or "").strip()
        normalized_image_contents = list(
            image_contents or (() if image_content is None else (image_content,))
        )
        normalized_image_urls = list(
            dict.fromkeys(
                value.strip()
                for value in ([image_url] if image_url else []) + list(image_urls or [])
                if value and value.strip()
            )
        )
        normalized_image_path = str(image_path or "").strip()
        has_image = bool(normalized_image_path or normalized_image_contents or normalized_image_urls)

        if task_type == DetectionTaskType.AIGC_DETECTION:
            raise ValueError("task=aigc_detection 暂未实现。")

        if task_type == DetectionTaskType.DEEPFAKE_DETECTION:
            if not has_image:
                raise ValueError("task=deepfake_detection 需要提供图片。")
            if len(normalized_image_contents) + len(normalized_image_urls) > 1:
                raise ValueError("task=deepfake_detection 当前只支持一张图片。")
            resolved_question = normalized_question or (
                DEFAULT_IMAGE_QUESTION_ZH if language == "zh" else DEFAULT_IMAGE_QUESTION_EN
            )
            result = self.ifd_detection_service.detect(
                image_content=normalized_image_contents[0] if normalized_image_contents else None,
                image_url=normalized_image_urls[0] if normalized_image_urls else None,
                image_path=normalized_image_path or None,
                question=resolved_question,
                language=language,
            )
            self._attach_single_agent_trace(result, "deepfake_detection", "IFD_Agent")
            result["question"] = resolved_question
            return result

        if not normalized_text:
            raise ValueError("task=multimodal_fact_checking 需要提供待核查的文本。")
        if normalized_question:
            resolved_question = normalized_question
        elif has_image:
            resolved_question = (
                DEFAULT_MULTIMODAL_QUESTION_ZH if language == "zh" else DEFAULT_MULTIMODAL_QUESTION_EN
            )
        else:
            resolved_question = DEFAULT_TEXT_QUESTION_ZH if language == "zh" else DEFAULT_TEXT_QUESTION_EN
        result = self.mmd_detection_service.detect(
            text=normalized_text,
            image_content=normalized_image_contents[0] if normalized_image_contents else None,
            image_url=normalized_image_urls[0] if normalized_image_urls else None,
            image_contents=normalized_image_contents,
            image_urls=normalized_image_urls,
            image_labels=image_labels,
            question=resolved_question,
            language=language,
        )
        result["question"] = resolved_question
        return result

    def detect_task(self, task: DetectionTask) -> dict[str, Any]:
        report = self.detect_report(task)
        return self.response_adapter.to_legacy_response(report)

    def detect_report(self, task: DetectionTask) -> DetectionReport:
        image_urls = task.image_urls
        primary_image = task.evidence_bundle.primary_image
        result = self.detect(
            text=task.text,
            image_contents=task.image_contents,
            image_urls=image_urls,
            image_path=primary_image.local_path if primary_image else None,
            question=task.question,
            history=task.history,
            language=task.language,
            mode=task.mode,
            task_type=task.task,
        )
        self._attach_task_context(result, task)
        return self.report_builder.build(task, result)

    def resolve_mode(
        self,
        text: str,
        has_image: bool,
        question: str = "",
    ) -> DetectionMode:
        del question
        if text.strip() and has_image:
            return DetectionMode.MULTIMODAL
        if has_image:
            return DetectionMode.IMAGE_ONLY
        return DetectionMode.TEXT_ONLY

    def _attach_single_agent_trace(self, result: dict[str, Any], name: str, agent: str) -> None:
        raw = result.setdefault("raw", {})
        raw["workflow"] = {
            "name": name,
            "routing": "single_agent",
            "steps": [agent],
            "dependencies": [],
        }

    def _attach_task_context(self, result: dict[str, Any], task: DetectionTask) -> None:
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
        result.setdefault("raw", {})["input_artifacts"] = {
            "text": [item.model_dump(mode="json", exclude={"content"}) for item in bundle.text_artifacts],
            "images": [item.model_dump(mode="json", exclude={"content"}) for item in bundle.image_artifacts],
            "webpage": webpage.model_dump(mode="json", exclude={"content"}) if webpage else None,
        }

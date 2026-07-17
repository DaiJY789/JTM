from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.api.adapters.detection_response import DetectionResponseAdapter
from app.core.language import normalize_language
from app.schemas.artifacts import DetectionTask
from app.schemas.detection import DetectionMode
from app.schemas.report import DetectionReport
from app.services.comprehensive_fusion import ComprehensiveFusionService
from app.services.crossmodal import CrossmodalConsistencyPipeline
from app.services.report_builder import DetectionReportBuilder
from app.services.textual.T_pipeline import TextualDetectionPipeline
from app.services.visual.visual_verdict import VisualVerdictService


DEFAULT_TEXT_QUESTION_ZH = "请判断这段文字的真实性。"
DEFAULT_TEXT_QUESTION_EN = "Please determine the authenticity of this text."
DEFAULT_IMAGE_QUESTION_ZH = "请判断这张图片是真实图像，伪造图像还是合成图像。"
DEFAULT_IMAGE_QUESTION_EN = "Please determine whether this image is a real image, a forged image, or a synthetic image."
DEFAULT_MULTIMODAL_QUESTION_ZH = "请判断这段文字和这张图片是否真实可信，并分析图文是否一致。"
DEFAULT_MULTIMODAL_QUESTION_EN = "Please assess whether this text and image are credible, and analyze whether the image and text are consistent."


class DetectionOrchestrator:
    GENERIC_IMAGE_QUESTIONS = {
        "这张图片是真的假的",
        "这张图片是真的假的？",
        "图片是真的假的",
        "图片是真的吗",
        "这是真的吗",
        "真的假的",
        "is this image real",
        "is this real",
        "real or fake",
        "fake or real",
    }

    def __init__(
        self,
        textual_pipeline: TextualDetectionPipeline,
        visual_verdict_service: VisualVerdictService,
        crossmodal_pipeline: CrossmodalConsistencyPipeline,
        comprehensive_fusion_service: ComprehensiveFusionService,
        report_builder: DetectionReportBuilder | None = None,
        response_adapter: DetectionResponseAdapter | None = None,
    ):
        self.textual_pipeline = textual_pipeline
        self.visual_verdict_service = visual_verdict_service
        self.crossmodal_pipeline = crossmodal_pipeline
        self.comprehensive_fusion_service = comprehensive_fusion_service
        self.report_builder = report_builder or DetectionReportBuilder()
        self.response_adapter = response_adapter or DetectionResponseAdapter()

    def detect(
        self,
        text: str | None = None,
        image_content: bytes | None = None,
        image_url: str | None = None,
        question: str | None = None,
        history: list | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        normalized_text = (text or "").strip()
        normalized_question = (question or "").strip()
        resolved_mode = self.resolve_mode(normalized_text, bool(image_content or image_url), normalized_question)
        resolved_question = self._resolve_question(normalized_question, resolved_mode, language)

        if resolved_mode == DetectionMode.TEXT_ONLY:
            result = self.textual_pipeline.detect(normalized_text or normalized_question, history, language)
            result["question"] = resolved_question
            return result
        if resolved_mode == DetectionMode.IMAGE_ONLY:
            result = self.visual_verdict_service.detect(image_content, image_url, resolved_question or normalized_text, language)
            result["question"] = resolved_question
            return result
        with ThreadPoolExecutor(max_workers=3) as executor:
            textual_future = executor.submit(self.textual_pipeline.detect, normalized_text, history, language)
            visual_future = executor.submit(
                self.visual_verdict_service.detect,
                image_content,
                image_url,
                resolved_question,
                language,
            )
            crossmodal_future = executor.submit(
                self.crossmodal_pipeline.detect,
                text=normalized_text,
                image_content=image_content,
                image_url=image_url,
                question=resolved_question,
                language=language,
            )
            textual_result = textual_future.result()
            visual_result = visual_future.result()
            crossmodal_result = crossmodal_future.result()

        report = self.comprehensive_fusion_service.generate_report(
            text=normalized_text,
            image_content=image_content,
            image_url=image_url,
            question=resolved_question,
            textual_result=textual_result,
            visual_result=visual_result,
            crossmodal_result=crossmodal_result,
            language=language,
        )
        result = {
            "mode": "multimodal",
            "label": report["label"],
            "type": report["type"],
            "summary": report["summary"],
            "confidence": report["confidence"],
            "text_verdict": textual_result.get("text_verdict"),
            "image_verdict": visual_result.get("image_verdict"),
            "crossmodal_verdict": crossmodal_result,
            "evidence": self._merge_evidence(textual_result, visual_result, crossmodal_result),
            "claims": textual_result.get("claims", []),
            "report": report,
            "raw": {
                "textual_detection": textual_result,
                "visual_detection": visual_result,
                "crossmodal_consistency": crossmodal_result,
                "comprehensive_fusion": report,
            },
            "user_answer": report["user_answer"],
        }
        result["question"] = resolved_question
        return result

    def _merge_evidence(
        self,
        textual_result: dict[str, Any],
        visual_result: dict[str, Any],
        crossmodal_result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        evidence = list(textual_result.get("evidence", []) or [])
        evidence.extend(visual_result.get("evidence", []) or [])
        if crossmodal_result.get("matched_elements") or crossmodal_result.get("mismatched_elements"):
            evidence.append(
                {
                    "type": "crossmodal_consistency",
                    "source": "CrossmodalConsistencyPipeline",
                    "content": {
                        "label": crossmodal_result.get("label"),
                        "matched_elements": crossmodal_result.get("matched_elements", []),
                        "mismatched_elements": crossmodal_result.get("mismatched_elements", []),
                        "missing_elements": crossmodal_result.get("missing_elements", []),
                        "reasoning": crossmodal_result.get("reasoning", ""),
                    },
                }
            )
        return evidence

    def detect_task(self, task: DetectionTask) -> dict[str, Any]:
        report = self.detect_report(task)
        return self.response_adapter.to_legacy_response(report)

    def detect_report(self, task: DetectionTask) -> DetectionReport:
        primary_image = task.evidence_bundle.primary_image
        result = self.detect(
            text=task.text,
            image_content=task.image_content,
            image_url=primary_image.source_url if primary_image else None,
            question=task.question,
            history=task.history,
            language=task.language,
        )
        self._attach_task_context(result, task)
        return self.report_builder.build(task, result)

    def resolve_mode(
        self,
        text: str,
        has_image: bool,
        question: str,
    ) -> DetectionMode:
        if has_image and self._is_image_only_text(text, question):
            return DetectionMode.IMAGE_ONLY
        if has_image and text:
            return DetectionMode.MULTIMODAL
        if has_image:
            return DetectionMode.IMAGE_ONLY
        return DetectionMode.TEXT_ONLY

    def _is_image_only_text(self, text: str, question: str) -> bool:
        candidate = (question or text).strip().lower().replace("?", "").replace("？", "")
        if not candidate:
            return True
        normalized_generic = {item.lower().replace("?", "").replace("？", "") for item in self.GENERIC_IMAGE_QUESTIONS}
        return candidate in normalized_generic

    def _resolve_question(self, question: str, mode: DetectionMode, language: str) -> str:
        if question:
            return question
        if mode == DetectionMode.TEXT_ONLY:
            return DEFAULT_TEXT_QUESTION_ZH if language == "zh" else DEFAULT_TEXT_QUESTION_EN
        if mode == DetectionMode.IMAGE_ONLY:
            return DEFAULT_IMAGE_QUESTION_ZH if language == "zh" else DEFAULT_IMAGE_QUESTION_EN
        if mode == DetectionMode.MULTIMODAL:
            return DEFAULT_MULTIMODAL_QUESTION_ZH if language == "zh" else DEFAULT_MULTIMODAL_QUESTION_EN
        return ""

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

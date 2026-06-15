from typing import Any

from app.core.language import normalize_language
from app.schemas.detection import DetectionMode
from app.services.vision.image import ImageDetectionService
from app.services.multimodal import MultimodalDetectionService
from app.services.text.text_detection import TextDetectionService


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
        text_detection_service: TextDetectionService,
        image_detection_service: ImageDetectionService,
        multimodal_detection_service: MultimodalDetectionService,
    ):
        self.text_detection_service = text_detection_service
        self.image_detection_service = image_detection_service
        self.multimodal_detection_service = multimodal_detection_service

    def detect(
        self,
        text: str | None = None,
        image_content: bytes | None = None,
        image_url: str | None = None,
        question: str | None = None,
        mode: str | DetectionMode = DetectionMode.AUTO,
        history: list | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        normalized_text = (text or "").strip()
        normalized_question = (question or "").strip()
        resolved_mode = self.resolve_mode(normalized_text, bool(image_content or image_url), normalized_question, mode)
        resolved_question = self._resolve_question(normalized_question, resolved_mode, language)

        if resolved_mode == DetectionMode.TEXT_ONLY:
            result = self.text_detection_service.detect(normalized_text or normalized_question, history, language)
            result["question"] = resolved_question
            return result
        if resolved_mode == DetectionMode.IMAGE_ONLY:
            result = self.image_detection_service.detect(image_content, image_url, resolved_question or normalized_text, language)
            result["question"] = resolved_question
            return result
        result = self.multimodal_detection_service.detect(
            text=normalized_text,
            image_content=image_content,
            image_url=image_url,
            question=resolved_question,
            history=history,
            language=language,
        )
        result["question"] = resolved_question
        return result

    def resolve_mode(
        self,
        text: str,
        has_image: bool,
        question: str,
        mode: str | DetectionMode = DetectionMode.AUTO,
    ) -> DetectionMode:
        requested_mode = DetectionMode(mode)
        if requested_mode != DetectionMode.AUTO:
            return requested_mode

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

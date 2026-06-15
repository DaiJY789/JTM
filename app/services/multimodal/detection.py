from typing import Any
from concurrent.futures import ThreadPoolExecutor

from app.core.language import normalize_language
from app.services.multimodal.alignment import MultimodalAlignmentService
from app.services.multimodal.verdict import MultimodalVerdictService
from app.services.text.text_detection import TextDetectionService
from app.services.vision.image import ImageDetectionService
from app.services.vision.image_io import ImageService


class MultimodalDetectionService:
    def __init__(
        self,
        text_detection_service: TextDetectionService,
        image_detection_service: ImageDetectionService,
        image_service: ImageService,
        alignment_service: MultimodalAlignmentService | None = None,
        verdict_service: MultimodalVerdictService | None = None,
    ):
        self.text_detection_service = text_detection_service
        self.image_detection_service = image_detection_service
        self.image_service = image_service
        self.alignment_service = alignment_service or MultimodalAlignmentService(image_service)
        self.verdict_service = verdict_service or MultimodalVerdictService()

    def detect(
        self,
        text: str,
        image_content: bytes | None = None,
        image_url: str | None = None,
        question: str | None = None,
        history: list | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        with ThreadPoolExecutor(max_workers=3) as executor:
            text_future = executor.submit(self.text_detection_service.detect, text, history, language)
            image_future = executor.submit(
                self.image_detection_service.detect,
                image_content,
                image_url,
                question,
                language,
            )
            alignment_future = executor.submit(
                self.alignment_service.evaluate,
                text=text,
                image_content=image_content,
                image_url=image_url,
                question=question,
                language=language,
            )
            text_result = text_future.result()
            image_result = image_future.result()
            alignment = alignment_future.result()
        final = self.verdict_service.evaluate(text_result, image_result, alignment, language)

        return {
            "mode": "multimodal",
            "label": final["label"],
            "type": final["type"],
            "summary": final["summary"],
            "confidence": final["confidence"],
            "text_verdict": text_result.get("text_verdict"),
            "image_verdict": image_result.get("image_verdict"),
            "alignment_verdict": alignment,
            "evidence": self._merge_evidence(text_result, image_result, alignment),
            "claims": text_result.get("claims", []),
            "raw": {
                "text_detection": text_result,
                "image_detection": image_result,
                "alignment_detection": alignment,
                "fusion": final.get("decision_trace", {}),
            },
            "user_answer": final["user_answer"],
        }

    def _merge_evidence(
        self,
        text_result: dict[str, Any],
        image_result: dict[str, Any],
        alignment: dict[str, Any],
    ) -> list[dict[str, Any]]:
        evidence = list(text_result.get("evidence", []) or [])
        evidence.extend(image_result.get("evidence", []) or [])
        if alignment.get("matched_elements") or alignment.get("mismatched_elements"):
            evidence.append(
                {
                    "type": "image_text_alignment",
                    "source": "MultimodalAlignmentService",
                    "content": {
                        "label": alignment.get("label"),
                        "matched_elements": alignment.get("matched_elements", []),
                        "mismatched_elements": alignment.get("mismatched_elements", []),
                        "missing_elements": alignment.get("missing_elements", []),
                        "reasoning": alignment.get("reasoning", ""),
                    },
                }
            )
        return evidence

    def _detect_alignment(
        self,
        text: str,
        image_content: bytes | None,
        image_url: str | None,
        question: str | None,
        language: str,
    ) -> dict[str, Any]:
        return self.alignment_service.evaluate(text, image_content, image_url, question, language=language)

    def _extract_alignment_label(self, analysis: str) -> str:
        return self.alignment_service._extract_alignment_label(analysis)

    def _extract_block(self, analysis: str, field_name: str) -> str:
        return self.alignment_service._extract_block(analysis, field_name)

    def _extract_list_field(self, analysis: str, field_name: str) -> list[str]:
        return self.alignment_service._extract_list_field(analysis, field_name)

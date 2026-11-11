from __future__ import annotations

import json
import re
from collections.abc import Callable
from time import perf_counter
from typing import Any

from app.agents.models import ModelFactory
from app.agents.tools.foundation import (
    BaseTool,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from app.core.config import Settings, settings
from app.core.language import is_zh, normalize_language
from app.services.visual.image_io import ImageService


OCREvaluator = Callable[[str, bytes | None, str | None, str], str]
ENTITY_TYPES = {
    "person",
    "organization",
    "location",
    "date",
    "time",
    "event",
    "product",
    "document",
    "other",
}


def _tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="extract_image_ocr_entities",
            description=(
                "Extract visible text and named/visual entities from the current trusted image. "
                "Use it when image text, people, organizations, locations, dates, events, or objects "
                "may affect multimodal fact checking."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "focus": OpenAIFunctionPropertySchema(
                        type="string",
                        description="Optional short instruction describing which visible details matter most.",
                    ),
                },
                required=[],
            ),
            strict=False,
        ),
    )


class ImageOCREntityTool(BaseTool):
    """Vision-model-backed OCR and entity extraction for Chinese news images."""

    name = "extract_image_ocr_entities"

    def __init__(
        self,
        image_service: ImageService,
        config: Settings = settings,
        model_factory: ModelFactory | None = None,
        evaluator: OCREvaluator | None = None,
    ):
        super().__init__(tool_schema=_tool_schema())
        self.image_service = image_service
        self.config = config
        self.model_factory = model_factory or image_service.model_factory
        self.evaluator = evaluator

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def run(
        self,
        *,
        image_content: bytes | None = None,
        image_url: str | None = None,
        language: str = "zh",
        focus: str = "",
    ) -> dict[str, Any]:
        started_at = perf_counter()
        language = normalize_language(language)
        builder = getattr(self.image_service, "build_ocr_image_content_part", None)
        if not callable(builder):
            builder = getattr(self.image_service, "build_mfc_image_content_part", None)
        image_part = builder(image_content, image_url) if callable(builder) else None
        if image_part is None:
            return self._error_result(
                "缺少有效图片。" if is_zh(language) else "A valid image is required.",
                started_at,
            )

        prompt = self._prompt(language, focus)
        try:
            raw_output = self._call_model(
                prompt,
                image_part,
                image_content,
                image_url,
                language,
            )
            parsed = self._parse_json(raw_output)
            result = self._normalize_result(parsed)
            result["status"] = "success"
            result["timings"] = {
                "total_seconds": round(perf_counter() - started_at, 4),
                "model_calls": 1,
                "model_output_characters": len(raw_output),
            }
            return result
        except Exception as exc:
            result = self._error_result(
                "OCR与实体识别失败。" if is_zh(language) else "OCR and entity extraction failed.",
                started_at,
                model_calls=1,
            )
            result["error"] = str(exc)
            return result

    def _call_model(
        self,
        prompt: str,
        image_part: dict[str, Any],
        image_content: bytes | None,
        image_url: str | None,
        language: str,
    ) -> str:
        if self.evaluator:
            return self.evaluator(prompt, image_content, image_url, language)
        client = self.model_factory.qwen_vision_client(
            timeout=self.config.mfc_ocr_timeout_seconds,
            max_retries=0,
        )
        completion = client.chat.completions.create(
            model=self.config.qwen_vision_model,
            messages=[
                {
                    "role": "system",
                    "content": "Extract only directly visible image text and entities. Output JSON only.",
                },
                {"role": "user", "content": [image_part, {"type": "text", "text": prompt}]},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=self.config.mfc_ocr_max_tokens,
        )
        return completion.choices[0].message.content or ""

    def _prompt(self, language: str, focus: str) -> str:
        output_language = "Chinese" if is_zh(language) else "English"
        return f"""
Inspect the supplied image and return OCR plus entity observations in {output_language}.
Only report content directly visible in the image. Do not infer identities from facial appearance,
do not use outside knowledge, and do not perform fact checking.

OCR rules:
- Preserve the reading order and original visible wording.
- Use bbox coordinates normalized to 0..1000 as [x0,y0,x1,y1].
- confidence is optional; omit it rather than inventing a score.

Entity rules:
- Allowed types: person, organization, location, date, time, event, product, document, other.
- A person may be described generically unless their name is visibly written.
- evidence must quote visible text or describe the directly visible cue.
- Deduplicate repeated entities.

Focus: {str(focus or '').strip()}

Return exactly one JSON object:
{{
  "ocr_text": "all visible text in reading order",
  "text_blocks": [{{"text": "visible text", "bbox": [0,0,1000,1000], "confidence": 0.0}}],
  "entities": [{{"name": "entity", "type": "location", "evidence": "visible basis", "bbox": [0,0,1000,1000]}}],
  "visual_objects": ["directly visible object or scene"],
  "limitations": ["optional evidence limitation"]
}}
""".strip()

    def _parse_json(self, raw_output: str) -> dict[str, Any]:
        text = str(raw_output or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text)
        try:
            value = json.loads(text)
        except json.JSONDecodeError as exc:
            start = text.find("{")
            if start < 0:
                raise ValueError("OCR tool did not return valid JSON") from exc
            try:
                value, _ = json.JSONDecoder().raw_decode(text[start:])
            except json.JSONDecodeError as nested_exc:
                raise ValueError("OCR tool did not return valid JSON") from nested_exc
        if not isinstance(value, dict):
            raise ValueError("OCR tool output must be a JSON object")
        return value

    def _normalize_result(self, value: dict[str, Any]) -> dict[str, Any]:
        blocks = []
        for item in value.get("text_blocks", []) if isinstance(value.get("text_blocks"), list) else []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            block: dict[str, Any] = {"text": text, "bbox": self._bbox(item.get("bbox"))}
            confidence = self._confidence(item.get("confidence"))
            if confidence is not None:
                block["confidence"] = confidence
            blocks.append(block)

        entities = []
        seen_entities = set()
        for item in value.get("entities", []) if isinstance(value.get("entities"), list) else []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            entity_type = str(item.get("type") or "other").strip().lower()
            if entity_type not in ENTITY_TYPES:
                entity_type = "other"
            dedupe_key = (name.casefold(), entity_type)
            if dedupe_key in seen_entities:
                continue
            seen_entities.add(dedupe_key)
            entity = {
                "name": name,
                "type": entity_type,
                "evidence": str(item.get("evidence") or "").strip(),
            }
            if item.get("bbox") is not None:
                entity["bbox"] = self._bbox(item.get("bbox"))
            entities.append(entity)

        ocr_text = str(value.get("ocr_text") or "").strip()
        if not ocr_text and blocks:
            ocr_text = "\n".join(item["text"] for item in blocks)
        return {
            "ocr_text": ocr_text,
            "text_blocks": blocks[:100],
            "entities": entities[:50],
            "visual_objects": self._strings(value.get("visual_objects"), limit=30),
            "limitations": self._strings(value.get("limitations"), limit=5),
        }

    def _bbox(self, value: Any) -> list[int]:
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            return [0, 0, 0, 0]
        coordinates = []
        for item in value:
            try:
                coordinates.append(max(0, min(1000, int(round(float(item))))))
            except (TypeError, ValueError):
                coordinates.append(0)
        x0, y0, x1, y1 = coordinates
        return [min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)]

    def _confidence(self, value: Any) -> float | None:
        if value is None:
            return None
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return None

    def _strings(self, value: Any, *, limit: int) -> list[str]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        return result[:limit]

    def _error_result(
        self,
        message: str,
        started_at: float,
        *,
        model_calls: int = 0,
    ) -> dict[str, Any]:
        return {
            "status": "error",
            "ocr_text": "",
            "text_blocks": [],
            "entities": [],
            "visual_objects": [],
            "limitations": [message],
            "timings": {
                "total_seconds": round(perf_counter() - started_at, 4),
                "model_calls": model_calls,
                "model_output_characters": 0,
            },
        }


__all__ = ["ImageOCREntityTool"]

import base64
import json
import mimetypes
import re
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image

from app.core.language import is_zh, normalize_language
from app.prompts.multimodal import MULTIMODAL_ALIGNMENT_PROMPT_EN, MULTIMODAL_ALIGNMENT_PROMPT_ZH
from app.services.vision.image_io import ImageService


class MultimodalAlignmentService:
    LABELS = {"Consistent", "Partially_Consistent", "Inconsistent"}
    LABEL_ALIASES = {
        "Mismatch": "Inconsistent",
        "Image_Out_Of_Context": "Inconsistent",
        "Weakly_Related": "Partially_Consistent",
        "Cannot_Determine": "Partially_Consistent",
        "图文一致": "Consistent",
        "图文部分一致": "Partially_Consistent",
        "部分一致": "Partially_Consistent",
        "图文不一致": "Inconsistent",
        "不一致": "Inconsistent",
    }

    def __init__(self, image_service: ImageService):
        self.image_service = image_service

    def evaluate(
        self,
        text: str,
        image_content: bytes | None = None,
        image_url: str | None = None,
        question: str | None = None,
        text_result: dict[str, Any] | None = None,
        image_result: dict[str, Any] | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        prompt = self._build_prompt(text, question, text_result or {}, image_result or {}, language)
        raw = self._call_vision_model(prompt, image_content, image_url, language)
        parsed = self._parse_json(raw) or self._parse_legacy_text(raw)
        return self._normalize_result(parsed, raw, language)

    def _build_prompt(
        self,
        text: str,
        question: str | None,
        text_result: dict[str, Any],
        image_result: dict[str, Any],
        language: str,
    ) -> str:
        base_prompt = MULTIMODAL_ALIGNMENT_PROMPT_ZH if is_zh(language) else MULTIMODAL_ALIGNMENT_PROMPT_EN
        context = {
            "news_text": text,
            "user_question": question or "",
            "text_verdict": {
                "label": text_result.get("label"),
                "summary": text_result.get("summary"),
                "claims": text_result.get("claims", [])[:3],
            },
            "image_verdict": {
                "label": image_result.get("label"),
                "summary": image_result.get("summary"),
                "suspicious_regions": image_result.get("image_verdict", {}).get("suspicious_regions", []),
            },
        }
        return f"{base_prompt}\n\nINPUT_JSON:\n{json.dumps(context, ensure_ascii=False)}"

    def _call_vision_model(
        self,
        prompt: str,
        image_content: bytes | None,
        image_url: str | None,
        language: str,
    ) -> str:
        image_part = self._image_content_part(image_content, image_url)
        if not image_part:
            return ""
        client = self.image_service.model_factory.qwen_vision_client(
            timeout=getattr(self.image_service.config, "vision_verdict_timeout_seconds", None)
        )
        system_prompt = "你是惊堂木图文一致性分析模块，只输出 JSON。" if is_zh(language) else (
            "You are Jingtangmu's image-text consistency module. Output JSON only."
        )
        completion = client.chat.completions.create(
            model=self.image_service.config.qwen_vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": [image_part, {"type": "text", "text": prompt}]},
            ],
            temperature=0,
            max_tokens=1800,
        )
        return completion.choices[0].message.content or ""

    def _image_content_part(self, image_content: bytes | None, image_url: str | None) -> dict[str, Any] | None:
        if image_content:
            data_url = self._bytes_to_data_url(image_content)
            return {"type": "image_url", "image_url": {"url": data_url}} if data_url else None
        if image_url:
            candidate = Path(image_url)
            if candidate.exists():
                data_url = self._path_to_data_url(candidate)
                return {"type": "image_url", "image_url": {"url": data_url}} if data_url else None
            return {"type": "image_url", "image_url": {"url": image_url}}
        return None

    def _bytes_to_data_url(self, content: bytes) -> str:
        try:
            with Image.open(BytesIO(content)) as image:
                image_format = image.format or "PNG"
        except Exception:
            return ""
        mime_type = Image.MIME.get(image_format.upper(), "image/png")
        encoded = base64.b64encode(content).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _path_to_data_url(self, path: Path) -> str:
        try:
            content = path.read_bytes()
        except OSError:
            return ""
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        encoded = base64.b64encode(content).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _parse_json(self, raw_output: str) -> dict[str, Any]:
        text = (raw_output or "").strip()
        candidates = [text]
        candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S))
        decoder = json.JSONDecoder()
        for candidate in candidates:
            for idx, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[idx:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return {}

    def _parse_legacy_text(self, analysis: str) -> dict[str, Any]:
        return {
            "label": self._extract_alignment_label(analysis),
            "matched_elements": self._extract_list_field(analysis, "Matched_Elements"),
            "mismatched_elements": self._extract_list_field(analysis, "Mismatched_Elements"),
            "reasoning": self._extract_block(analysis, "Reasoning"),
        }

    def _normalize_result(self, parsed: dict[str, Any], raw: str, language: str) -> dict[str, Any]:
        label = str(parsed.get("label") or "Cannot_Determine").strip()
        if label not in self.LABELS:
            label = self.LABEL_ALIASES.get(label, self._extract_alignment_label(label))
        consistency_score = self._bounded_float(parsed.get("consistency_score"), self._default_score(label))
        reasoning = str(parsed.get("reasoning") or raw or self._fallback_reasoning(label, language)).strip()
        return {
            "label": label,
            "analysis": raw or reasoning,
            "consistency_score": consistency_score,
            "text_elements": parsed.get("text_elements") if isinstance(parsed.get("text_elements"), dict) else {},
            "visual_elements": parsed.get("visual_elements") if isinstance(parsed.get("visual_elements"), dict) else {},
            "matched_elements": self._string_list(parsed.get("matched_elements")),
            "mismatched_elements": self._string_list(parsed.get("mismatched_elements")),
            "missing_elements": self._string_list(parsed.get("missing_elements")),
            "reasoning": reasoning,
            "limitations": self._string_list(parsed.get("limitations")),
        }

    def _extract_alignment_label(self, analysis: str) -> str:
        labels = [
            "Partially_Consistent",
            "Inconsistent",
            "Consistent",
            "Image_Out_Of_Context",
            "Cannot_Determine",
            "Weakly_Related",
            "Mismatch",
            "图文部分一致",
            "图文不一致",
            "图文一致",
        ]
        for label in labels:
            if label in str(analysis or ""):
                return self.LABEL_ALIASES.get(label, label)
        match = re.search(r"Label\s*[:：]\s*([A-Za-z_]+)", str(analysis or ""))
        candidate = match.group(1) if match else ""
        if candidate in self.LABELS:
            return candidate
        return self.LABEL_ALIASES.get(candidate, "Partially_Consistent")

    def _extract_block(self, analysis: str, field_name: str) -> str:
        known_fields = ["Label", "Matched_Elements", "Mismatched_Elements", "Missing_Elements", "Reasoning"]
        next_fields = "|".join([re.escape(field) for field in known_fields if field != field_name])
        pattern = rf"{field_name}\s*[:：]\s*(.*?)(?=\n\s*(?:{next_fields})\s*[:：]|\Z)"
        match = re.search(pattern, str(analysis or ""), re.S)
        return re.sub(r"\s+\n", "\n", match.group(1)).strip() if match else ""

    def _extract_list_field(self, analysis: str, field_name: str) -> list[str]:
        block = self._extract_block(analysis, field_name)
        if not block:
            return []
        items = []
        for line in block.splitlines():
            cleaned = re.sub(r"^\s*[-*•\d.、)）]+\s*", "", line).strip()
            if cleaned:
                items.append(cleaned)
        if len(items) <= 1 and ("；" in block or ";" in block):
            items = [item.strip() for item in re.split(r"[；;]", block) if item.strip()]
        return items

    def _string_list(self, value: Any) -> list[str]:
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, str) and value.strip():
            return [value.strip()]
        return []

    def _bounded_float(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return round(max(0.0, min(1.0, number)), 4)

    def _default_score(self, label: str) -> float:
        return {
            "Consistent": 0.82,
            "Partially_Consistent": 0.5,
            "Inconsistent": 0.22,
        }.get(label, 0.0)

    def _fallback_reasoning(self, label: str, language: str) -> str:
        if is_zh(language):
            return "已根据图文要素比较生成判断。"
        return "The judgment is based on comparing text and visual elements."

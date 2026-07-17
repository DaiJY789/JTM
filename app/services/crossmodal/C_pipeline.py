import json
import re
from typing import Any

from app.core.language import is_zh, normalize_language
from app.agents.prompts.crossmodal import CROSSMODAL_CONSISTENCY_PROMPT_EN, CROSSMODAL_CONSISTENCY_PROMPT_ZH
from app.services.visual.image_io import ImageService


class CrossmodalConsistencyPipeline:
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

    def detect(
        self,
        text: str,
        image_content: bytes | None = None,
        image_url: str | None = None,
        question: str | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        prompt = self._build_prompt(text, question, language)
        try:
            raw = self._call_vision_model(prompt, image_content, image_url, language)
        except Exception as exc:
            return self._fallback_unavailable(exc, language)
        parsed = self._parse_json(raw) or self._parse_legacy_text(raw)
        return self._normalize_result(parsed, raw, language)

    def _build_prompt(
        self,
        text: str,
        question: str | None,
        language: str,
    ) -> str:
        base_prompt = CROSSMODAL_CONSISTENCY_PROMPT_ZH if is_zh(language) else CROSSMODAL_CONSISTENCY_PROMPT_EN
        context = {
            "news_text": text,
            "user_question": question or "",
        }
        return f"{base_prompt}\n\nINPUT_JSON:\n{json.dumps(context, ensure_ascii=False)}"

    def _call_vision_model(
        self,
        prompt: str,
        image_content: bytes | None,
        image_url: str | None,
        language: str,
    ) -> str:
        image_part = self.image_service.build_image_content_part(image_content, image_url)
        if not image_part:
            return ""
        client = self.image_service.model_factory.qwen_vision_client(
            timeout=getattr(self.image_service.config, "vision_verdict_timeout_seconds", None)
        )
        system_prompt = "你是惊堂木跨模态一致性检查模块，只输出 JSON。" if is_zh(language) else (
            "You are Jingtangmu's cross-modal consistency checker. Output JSON only."
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

    def _fallback_unavailable(self, exc: Exception, language: str) -> dict[str, Any]:
        message = (
            "图文一致性模型调用失败，已保留文本和图像单模态检测结果。"
            if is_zh(language)
            else "The image-text consistency model call failed; text and image single-modal results were kept."
        )
        detail = str(exc).strip()
        if detail:
            message = f"{message} {detail}"
        return self._normalize_result(
            {
                "label": "Cannot_Determine",
                "consistency_score": 0.5,
                "reasoning": message,
                "limitations": [message],
            },
            "",
            language,
        )

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
            "label": self._extract_crossmodal_label(analysis),
            "matched_elements": self._extract_list_field(analysis, "Matched_Elements"),
            "mismatched_elements": self._extract_list_field(analysis, "Mismatched_Elements"),
            "reasoning": self._extract_block(analysis, "Reasoning"),
        }

    def _normalize_result(self, parsed: dict[str, Any], raw: str, language: str) -> dict[str, Any]:
        label = str(parsed.get("label") or "Cannot_Determine").strip()
        if label not in self.LABELS:
            label = self.LABEL_ALIASES.get(label, self._extract_crossmodal_label(label))
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

    def _extract_crossmodal_label(self, analysis: str) -> str:
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

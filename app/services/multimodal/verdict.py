import json
import re
from collections.abc import Callable
from typing import Any

from app.agents.models import ModelFactory
from app.core.config import Settings, settings
from app.core.language import is_zh, normalize_language

FusionEvaluator = Callable[[str, float | None], str]


class MultimodalVerdictService:
    TEXT_ANOMALY_LABELS = {
        "Completely_False",
        "Mostly_False",
        "Likely_False",
        "完全虚假",
        "大部分虚假",
        "可能虚假",
    }
    TEXT_NORMAL_LABELS = {
        "Completely_True",
        "Mostly_True",
        "Likely_True",
        "完全真实",
        "大部分真实",
        "可能真实",
    }
    IMAGE_ANOMALY_LABELS = {
        "伪造图像",
        "合成图像",
        "Forged_Image",
        "Synthetic_Image",
        "存在伪造",
        "完全AI生成",
        "Forgery_Detected",
        "Completely_AI_Generated",
        "局部篡改",
        "AI操纵",
        "AI生成",
        "Manipulated",
        "AI_Generated",
        "Deepfake_Suspected",
    }
    IMAGE_NORMAL_LABELS = {"真实图像", "Real_Image", "完全真实", "Completely_Real", "Real", "Likely_Real"}

    LABEL_ZH = {
        "Text_Anomaly": "文本异常",
        "Image_Anomaly": "图像异常",
        "No_Anomaly": "均无异常",
        "All_Anomaly": "全部异常",
    }
    LABEL_EN = {
        "Text_Anomaly": "Text anomaly",
        "Image_Anomaly": "Image anomaly",
        "No_Anomaly": "No anomaly",
        "All_Anomaly": "All anomaly",
    }
    TYPE_ZH = {
        "Consistent": "图文一致",
        "Partially_Consistent": "图文部分一致",
        "Inconsistent": "图文不一致",
    }
    TYPE_EN = {
        "Consistent": "Image-text consistent",
        "Partially_Consistent": "Image-text partially consistent",
        "Inconsistent": "Image-text inconsistent",
    }

    def __init__(
        self,
        config: Settings = settings,
        model_factory: ModelFactory | None = None,
        evaluator: FusionEvaluator | None = None,
    ):
        self.config = config
        self.model_factory = model_factory or ModelFactory(config)
        self.evaluator = evaluator

    def evaluate(
        self,
        text_result: dict[str, Any],
        image_result: dict[str, Any],
        alignment_result: dict[str, Any],
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        text_label = str(text_result.get("label") or "")
        image_label = str(image_result.get("label") or "")
        alignment_label = self._normalize_alignment_label(alignment_result.get("label"))

        text_anomaly = self._is_text_anomaly(text_label)
        image_anomaly = self._is_image_anomaly(image_label)
        label_key = self._fusion_label_key(text_anomaly, image_anomaly)
        label = self._display_label(label_key, language)
        verdict_type = self._display_type(alignment_label, language)
        confidence = self._confidence(label_key, text_result, image_result, alignment_result)

        model_answer = self._model_user_answer(
            label=label,
            verdict_type=verdict_type,
            text_result=text_result,
            image_result=image_result,
            alignment_result=alignment_result,
            confidence=confidence,
            language=language,
        )
        user_answer = model_answer or self._fallback_user_answer(
            label=label,
            verdict_type=verdict_type,
            text_result=text_result,
            image_result=image_result,
            alignment_result=alignment_result,
            language=language,
        )

        return {
            "label": label,
            "type": verdict_type,
            "confidence": confidence,
            "summary": user_answer.get("judgment") or user_answer.get("conclusion") or "",
            "user_answer": user_answer,
            "decision_trace": {
                "text_label": text_label,
                "image_label": image_label,
                "alignment_label": alignment_label,
                "text_anomaly": text_anomaly,
                "image_anomaly": image_anomaly,
                "fusion_label": label_key,
            },
        }

    def _is_text_anomaly(self, label: str) -> bool:
        return label in self.TEXT_ANOMALY_LABELS

    def _is_image_anomaly(self, label: str) -> bool:
        return label in self.IMAGE_ANOMALY_LABELS

    def _fusion_label_key(self, text_anomaly: bool, image_anomaly: bool) -> str:
        if text_anomaly and image_anomaly:
            return "All_Anomaly"
        if text_anomaly:
            return "Text_Anomaly"
        if image_anomaly:
            return "Image_Anomaly"
        return "No_Anomaly"

    def _normalize_alignment_label(self, label: Any) -> str:
        text = str(label or "").strip()
        aliases = {
            "Consistent": "Consistent",
            "Partially_Consistent": "Partially_Consistent",
            "Inconsistent": "Inconsistent",
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
        return aliases.get(text, "Partially_Consistent")

    def _display_label(self, label_key: str, language: str) -> str:
        return (self.LABEL_ZH if is_zh(language) else self.LABEL_EN).get(label_key, label_key)

    def _display_type(self, alignment_label: str, language: str) -> str:
        return (self.TYPE_ZH if is_zh(language) else self.TYPE_EN).get(alignment_label, alignment_label)

    def _confidence(
        self,
        label_key: str,
        text_result: dict[str, Any],
        image_result: dict[str, Any],
        alignment_result: dict[str, Any],
    ) -> float:
        values = [
            self._bounded_float(text_result.get("confidence"), 0.62),
            self._bounded_float(image_result.get("confidence"), 0.62),
            self._bounded_float(alignment_result.get("consistency_score"), 0.5),
        ]
        base = sum(values) / len(values)
        if label_key in {"Text_Anomaly", "Image_Anomaly", "All_Anomaly"}:
            base = max(base, 0.66)
        return round(max(0.0, min(1.0, base)), 4)

    def _model_user_answer(
        self,
        label: str,
        verdict_type: str,
        text_result: dict[str, Any],
        image_result: dict[str, Any],
        alignment_result: dict[str, Any],
        confidence: float,
        language: str,
    ) -> dict[str, Any]:
        prompt = self._build_prompt(label, verdict_type, text_result, image_result, alignment_result, confidence, language)
        raw = self._call_model(prompt, language)
        parsed = self._parse_json(raw)
        if not parsed:
            return {}
        judgment = str(parsed.get("judgment") or parsed.get("conclusion") or "").strip()
        analysis = str(parsed.get("reason_analysis") or parsed.get("analysis") or "").strip()
        notes = parsed.get("notes") if isinstance(parsed.get("notes"), list) else []
        if not judgment and not analysis:
            return {}
        return {
            "judgment": judgment,
            "reason_analysis": analysis,
            "notes": [str(item).strip() for item in notes if str(item).strip()],
        }

    def _build_prompt(
        self,
        label: str,
        verdict_type: str,
        text_result: dict[str, Any],
        image_result: dict[str, Any],
        alignment_result: dict[str, Any],
        confidence: float,
        language: str,
    ) -> str:
        context = {
            "final_label": label,
            "image_text_consistency_type": verdict_type,
            "confidence": confidence,
            "text_detection": self._compact_text_result(text_result),
            "image_detection": self._compact_image_result(image_result),
            "alignment_detection": alignment_result,
        }
        if is_zh(language):
            instruction = (
                "你是惊堂木多模态检测结果综合分析模块。请基于文本检测、图像检测和图文一致性检测结果，"
                "给出面向用户的最终判断和原因分析。最终判断必须围绕 final_label 和 image_text_consistency_type，"
                "不要引入未在输入证据中出现的新事实。只输出 JSON。"
                "JSON 字段：judgment（一句话结论）、reason_analysis（一段直觉化原因分析）、notes（数组，可为空）。"
            )
        else:
            instruction = (
                "You are Jingtangmu's multimodal fusion analysis module. Based on text detection, image detection, "
                "and image-text consistency results, produce a user-facing final judgment and concise reasoning. "
                "Base the judgment on final_label and image_text_consistency_type. Do not invent facts not present in the evidence. "
                "Output JSON only with: judgment, reason_analysis, notes."
            )
        return f"{instruction}\n\nINPUT_JSON:\n{json.dumps(context, ensure_ascii=False)}"

    def _call_model(self, prompt: str, language: str) -> str:
        if self.evaluator:
            return self.evaluator(prompt, getattr(self.config, "text_verdict_timeout_seconds", None))
        try:
            client = self.model_factory.deepseek_client(timeout=getattr(self.config, "text_verdict_timeout_seconds", None))
            system_prompt = "你是严格输出 JSON 的多模态检测综合分析器。" if is_zh(language) else (
                "You are a strict JSON-only multimodal fusion evaluator."
            )
            completion = client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=900,
            )
            return completion.choices[0].message.content or ""
        except Exception:
            return ""

    def _fallback_user_answer(
        self,
        label: str,
        verdict_type: str,
        text_result: dict[str, Any],
        image_result: dict[str, Any],
        alignment_result: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        text_label = text_result.get("label") or ""
        image_label = image_result.get("label") or ""
        alignment_reason = alignment_result.get("reasoning") or alignment_result.get("analysis") or ""
        if is_zh(language):
            judgment = f"综合判断为{label}，图文一致性为{verdict_type}。"
            analysis = (
                f"文本检测结果为{text_label or '未给出'}，图像检测结果为{image_label or '未给出'}。"
                f"图文关系判断为{verdict_type}"
                f"{'，依据是' + self._short_text(alignment_reason) if alignment_reason else '。'}"
            )
        else:
            judgment = f"The fused judgment is {label}, with {verdict_type}."
            analysis = (
                f"Text detection: {text_label or 'not provided'}. Image detection: {image_label or 'not provided'}. "
                f"Image-text relation: {verdict_type}"
                f"{'; reason: ' + self._short_text(alignment_reason) if alignment_reason else '.'}"
            )
        return {"judgment": judgment, "reason_analysis": analysis, "notes": []}

    def _compact_text_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "label": result.get("label"),
            "type": result.get("type"),
            "summary": result.get("summary"),
            "claims": result.get("claims", [])[:3],
            "user_answer": result.get("user_answer") or result.get("text_verdict", {}).get("user_answer", {}),
        }

    def _compact_image_result(self, result: dict[str, Any]) -> dict[str, Any]:
        verdict = result.get("image_verdict", {}) or {}
        return {
            "label": result.get("label"),
            "summary": result.get("summary"),
            "forgery_types": result.get("forgery_types") or verdict.get("forgery_types", []),
            "suspicious_regions": verdict.get("suspicious_regions", []),
            "user_answer": result.get("user_answer") or verdict.get("user_answer", {}),
        }

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

    def _short_text(self, value: Any) -> str:
        text = str(value or "").strip()
        return " ".join(text.split())

    def _bounded_float(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return max(0.0, min(1.0, number))

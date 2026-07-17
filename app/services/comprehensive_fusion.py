import json
import re
from collections.abc import Callable
from typing import Any

from app.agents.prompts.comprehensive_fusion import (
    COMPREHENSIVE_FUSION_REPORT_PROMPT_EN,
    COMPREHENSIVE_FUSION_REPORT_PROMPT_ZH,
)
from app.core.language import is_zh, normalize_language
from app.services.visual.image_io import ImageService


FusionEvaluator = Callable[..., str]


class ComprehensiveFusionService:
    """Use a multimodal model to synthesize all detector outputs into the final report."""

    TEXT_ANOMALY_LABELS = {
        "Completely_False",
        "Mostly_False",
        "Likely_False",
        "完全虚假",
        "大部分虚假",
        "可能虚假",
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
    LABELS_ZH = {"文本异常", "图像异常", "均无异常", "全部异常"}
    LABELS_EN = {"Text anomaly", "Image anomaly", "No anomaly", "All anomaly"}
    TYPES_ZH = {"图文一致", "图文部分一致", "图文不一致"}
    TYPES_EN = {
        "Image-text consistent",
        "Image-text partially consistent",
        "Image-text inconsistent",
    }

    def __init__(self, image_service: ImageService, evaluator: FusionEvaluator | None = None):
        self.image_service = image_service
        self.config = image_service.config
        self.model_factory = getattr(image_service, "model_factory", None)
        self.evaluator = evaluator

    def generate_report(
        self,
        *,
        text: str,
        image_content: bytes | None,
        image_url: str | None,
        question: str | None,
        textual_result: dict[str, Any],
        visual_result: dict[str, Any],
        crossmodal_result: dict[str, Any],
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        fallback = self._fallback_report(textual_result, visual_result, crossmodal_result, language)
        prompt = self._build_prompt(
            text=text,
            question=question,
            textual_result=textual_result,
            visual_result=visual_result,
            crossmodal_result=crossmodal_result,
            fallback=fallback,
            language=language,
        )
        raw = self._call_multimodal_model(prompt, image_content, image_url, language)
        parsed = self._parse_json(raw)
        report = self._normalize_report(parsed, fallback, language)
        report["decision_trace"] = {
            "textual_label": textual_result.get("label"),
            "visual_label": visual_result.get("label"),
            "crossmodal_label": crossmodal_result.get("label"),
            "model_report_used": bool(parsed),
        }
        report["raw_model_report"] = raw
        return report

    def _build_prompt(
        self,
        *,
        text: str,
        question: str | None,
        textual_result: dict[str, Any],
        visual_result: dict[str, Any],
        crossmodal_result: dict[str, Any],
        fallback: dict[str, Any],
        language: str,
    ) -> str:
        base = COMPREHENSIVE_FUSION_REPORT_PROMPT_ZH if is_zh(language) else COMPREHENSIVE_FUSION_REPORT_PROMPT_EN
        context = {
            "news_text": text,
            "user_question": question or "",
            "textual_detection": self._compact_textual_result(textual_result),
            "visual_detection": self._compact_visual_result(visual_result),
            "crossmodal_consistency": crossmodal_result,
            "rule_based_fallback": {
                "label": fallback["label"],
                "type": fallback["type"],
                "confidence": fallback["confidence"],
            },
        }
        return f"{base}\n\nINPUT_JSON:\n{json.dumps(context, ensure_ascii=False)}"

    def _call_multimodal_model(
        self,
        prompt: str,
        image_content: bytes | None,
        image_url: str | None,
        language: str,
    ) -> str:
        if self.evaluator:
            try:
                return self.evaluator(prompt, image_content, image_url, language)
            except TypeError:
                return self.evaluator(prompt, getattr(self.config, "vision_verdict_timeout_seconds", None))
        if self.model_factory is None:
            return ""
        image_part = self.image_service.build_image_content_part(image_content, image_url)
        if not image_part:
            return ""
        try:
            client = self.model_factory.qwen_vision_client(
                timeout=getattr(self.config, "vision_verdict_timeout_seconds", None)
            )
            system_prompt = (
                "你是惊堂木综合融合结论模块。直接观察原图并基于输入证据生成严格 JSON 报告。"
                if is_zh(language)
                else "You are Jingtangmu's comprehensive fusion module. Inspect the image and output a strict JSON report."
            )
            completion = client.chat.completions.create(
                model=self.config.qwen_vision_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": [image_part, {"type": "text", "text": prompt}]},
                ],
                temperature=0,
                max_tokens=1800,
            )
            return completion.choices[0].message.content or ""
        except Exception:
            return ""

    def _fallback_report(
        self,
        textual_result: dict[str, Any],
        visual_result: dict[str, Any],
        crossmodal_result: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        text_label = str(textual_result.get("label") or "")
        visual_label = str(visual_result.get("label") or "")
        text_anomaly = text_label in self.TEXT_ANOMALY_LABELS
        visual_anomaly = visual_label in self.IMAGE_ANOMALY_LABELS
        if text_anomaly and visual_anomaly:
            label = "全部异常" if is_zh(language) else "All anomaly"
        elif text_anomaly:
            label = "文本异常" if is_zh(language) else "Text anomaly"
        elif visual_anomaly:
            label = "图像异常" if is_zh(language) else "Image anomaly"
        else:
            label = "均无异常" if is_zh(language) else "No anomaly"

        relation = self._display_type(crossmodal_result.get("label"), language)
        confidence_values = [
            self._bounded_float(textual_result.get("confidence"), 0.62),
            self._bounded_float(visual_result.get("confidence"), 0.62),
            self._bounded_float(crossmodal_result.get("consistency_score"), 0.5),
        ]
        confidence = round(sum(confidence_values) / len(confidence_values), 4)
        if text_anomaly or visual_anomaly:
            confidence = max(confidence, 0.66)
        crossmodal_reason = str(crossmodal_result.get("reasoning") or "").strip()
        if is_zh(language):
            judgment = f"综合判断为{label}，跨模态关系为{relation}。"
            analysis = (
                f"文本检测结果为{text_label or '未给出'}，图像检测结果为{visual_label or '未给出'}。"
                f"跨模态一致性为{relation}{'，依据是' + crossmodal_reason if crossmodal_reason else '。'}"
            )
        else:
            judgment = f"The fused judgment is {label}, with {relation}."
            analysis = (
                f"Textual detection: {text_label or 'not provided'}. Visual detection: {visual_label or 'not provided'}. "
                f"Cross-modal consistency: {relation}{'; reason: ' + crossmodal_reason if crossmodal_reason else '.'}"
            )
        return {
            "label": label,
            "type": relation,
            "confidence": confidence,
            "summary": judgment,
            "risk_level": "high" if text_anomaly or visual_anomaly else "low",
            "user_answer": {
                "judgment": judgment,
                "reason_analysis": analysis,
                "text_analysis": str(textual_result.get("summary") or ""),
                "visual_analysis": str(visual_result.get("summary") or ""),
                "crossmodal_analysis": crossmodal_reason,
                "notes": [],
            },
            "limitations": [],
        }

    def _normalize_report(
        self,
        parsed: dict[str, Any],
        fallback: dict[str, Any],
        language: str,
    ) -> dict[str, Any]:
        if not parsed:
            return fallback
        allowed_labels = self.LABELS_ZH if is_zh(language) else self.LABELS_EN
        allowed_types = self.TYPES_ZH if is_zh(language) else self.TYPES_EN
        label = str(parsed.get("label") or "").strip()
        relation = str(parsed.get("type") or "").strip()
        if label not in allowed_labels:
            label = fallback["label"]
        if relation not in allowed_types:
            relation = fallback["type"]
        answer = parsed.get("user_answer") if isinstance(parsed.get("user_answer"), dict) else {}
        fallback_answer = fallback["user_answer"]
        normalized_answer = {
            key: str(answer.get(key) or fallback_answer.get(key) or "").strip()
            for key in ("judgment", "reason_analysis", "text_analysis", "visual_analysis", "crossmodal_analysis")
        }
        notes = answer.get("notes") if isinstance(answer.get("notes"), list) else fallback_answer.get("notes", [])
        normalized_answer["notes"] = [str(item).strip() for item in notes if str(item).strip()]
        summary = str(parsed.get("summary") or normalized_answer["judgment"] or fallback["summary"]).strip()
        limitations = parsed.get("limitations") if isinstance(parsed.get("limitations"), list) else []
        risk_level = str(parsed.get("risk_level") or fallback["risk_level"]).lower()
        if risk_level not in {"low", "medium", "high"}:
            risk_level = fallback["risk_level"]
        return {
            "label": label,
            "type": relation,
            "confidence": self._bounded_float(parsed.get("confidence"), fallback["confidence"]),
            "summary": summary,
            "risk_level": risk_level,
            "user_answer": normalized_answer,
            "limitations": [str(item).strip() for item in limitations if str(item).strip()],
        }

    def _display_type(self, label: Any, language: str) -> str:
        normalized = {
            "Consistent": "Consistent",
            "Partially_Consistent": "Partially_Consistent",
            "Inconsistent": "Inconsistent",
            "Mismatch": "Inconsistent",
            "Image_Out_Of_Context": "Inconsistent",
            "Weakly_Related": "Partially_Consistent",
            "Cannot_Determine": "Partially_Consistent",
        }.get(str(label or ""), "Partially_Consistent")
        if is_zh(language):
            return {
                "Consistent": "图文一致",
                "Partially_Consistent": "图文部分一致",
                "Inconsistent": "图文不一致",
            }[normalized]
        return {
            "Consistent": "Image-text consistent",
            "Partially_Consistent": "Image-text partially consistent",
            "Inconsistent": "Image-text inconsistent",
        }[normalized]

    def _compact_textual_result(self, result: dict[str, Any]) -> dict[str, Any]:
        return {
            "label": result.get("label"),
            "type": result.get("type"),
            "summary": result.get("summary"),
            "confidence": result.get("confidence"),
            "claims": result.get("claims", [])[:3],
            "evidence": result.get("evidence", [])[:6],
            "user_answer": result.get("user_answer") or result.get("text_verdict", {}).get("user_answer", {}),
        }

    def _compact_visual_result(self, result: dict[str, Any]) -> dict[str, Any]:
        verdict = result.get("image_verdict", {}) or {}
        return {
            "label": result.get("label"),
            "summary": result.get("summary"),
            "confidence": result.get("confidence"),
            "forgery_types": verdict.get("forgery_types", []),
            "suspicious_regions": verdict.get("suspicious_regions", []),
            "tool_evidence": verdict.get("tool_evidence", [])[:8],
            "user_answer": result.get("user_answer") or verdict.get("user_answer", {}),
        }

    def _parse_json(self, raw_output: str) -> dict[str, Any]:
        text = str(raw_output or "").strip()
        candidates = [text]
        candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S))
        decoder = json.JSONDecoder()
        for candidate in candidates:
            for index, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[index:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return {}

    def _bounded_float(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return round(max(0.0, min(1.0, number)), 4)

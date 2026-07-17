import json
import re
from copy import deepcopy
from typing import Any

from app.agents.models import ModelFactory
from app.core.config import Settings, settings
from app.core.language import is_zh, normalize_language


class FollowupAnalysisService:
    """Answer follow-up questions from the latest detection result without rerunning detectors."""

    def __init__(self, config: Settings = settings, model_factory: ModelFactory | None = None):
        self.config = config
        self.model_factory = model_factory or ModelFactory(config)

    def answer(
        self,
        question: str,
        latest_input: dict[str, Any],
        latest_result: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        base_result = deepcopy(latest_result)
        prompt = self._build_prompt(question, latest_input, base_result, history or [], language)
        raw_output = self._call_model(prompt, language)
        parsed = self._parse_json(raw_output)
        return self._merge_result(base_result, parsed, question, raw_output, language)

    def explain(
        self,
        question: str,
        latest_input: dict[str, Any],
        latest_result: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        prompt = self._build_prompt(question, latest_input, deepcopy(latest_result), history or [], language)
        raw_output = self._call_model(prompt, language)
        parsed = self._parse_json(raw_output)
        answer = self._normalize_user_answer(parsed.get("user_answer"), deepcopy(latest_result), question, language)
        return {
            "question": question,
            "answer": answer,
            "content": answer.get("reason_analysis") or answer.get("judgment") or "",
            "raw_output": raw_output,
            "updated_fields": parsed.get("updated_fields", {}),
        }

    def _call_model(self, prompt: str, language: str) -> str:
        try:
            client = self.model_factory.deepseek_client(timeout=getattr(self.config, "text_verdict_timeout_seconds", None))
            system_prompt = (
                "你是追问分析助手，只基于已有检测结果回答和谨慎修正，必须输出 JSON。"
                if is_zh(language)
                else "You are a follow-up analysis assistant. Use only the existing detection result and output JSON."
            )
            completion = client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1000,
            )
            return completion.choices[0].message.content or ""
        except Exception as exc:
            answer = {
                "judgment": "无法完成追问分析。" if is_zh(language) else "Unable to complete follow-up analysis.",
                "reason_analysis": (
                    f"追问分析模型调用失败，当前仍保留上一次检测结果。错误信息：{exc}"
                    if is_zh(language)
                    else f"The follow-up model call failed, so the previous detection result is kept. Error: {exc}"
                ),
                "notes": [],
            }
            return json.dumps({"user_answer": answer, "updated_fields": {"model_call_failed": True}}, ensure_ascii=False)

    def _build_prompt(
        self,
        question: str,
        latest_input: dict[str, Any],
        latest_result: dict[str, Any],
        history: list[dict[str, Any]],
        language: str,
    ) -> str:
        if is_zh(language):
            instructions = """
你正在回答用户对上一次检测结果的追问。
不要重新执行检测流程，不要编造新的工具调用结果，不要声称重新检索或重新分析了图片工具。
只能基于已有输入、已有检测结果、已有文本声明、已有检索证据、已有图像工具证据和对话历史进行解释、补充或谨慎修正。
如果追问指出了已有结果中的矛盾，可以修正 label、confidence、user_answer、forgery_types 或 forgery_regions。
forgery_types 只能从以下列表选择一个值：无、拼接、复制移动、内容移除、内容替换、人脸篡改、局部重绘、全局编辑、未知。
输出必须是 JSON，不要输出 Markdown。
JSON 格式：
{
  "label": "可选，保持或修正后的标签",
  "confidence": 0.0,
  "forgery_types": ["可选，最多一个值"],
  "user_answer": {
    "judgment": "更新后的最终判断或对追问的直接回答",
    "reason_analysis": "展示在对话流中的解释说明",
    "notes": ["相关说明"],
    "forgery_regions": [{"region": "x0,y0,x1,y1", "score": 0.0, "reason": "可选"}]
  },
  "updated_fields": {
    "label_changed": false,
    "confidence_changed": false,
    "regions_changed": false
  }
}
"""
        else:
            instructions = """
You are answering a follow-up question about the previous detection result.
Do not rerun the detection pipeline, invent new tool outputs, or claim that tools/search were rerun.
Use only the existing input, existing detection result, existing text claims, existing retrieved evidence, existing image-tool evidence, and conversation history to explain, supplement, or cautiously revise the result.
If the follow-up identifies a real contradiction, you may revise label, confidence, user_answer, forgery_types, or forgery_regions.
forgery_types must contain at most one value selected from: none, splicing, copy_move, content_removal, content_replacement, face_tampering, local_repainting, global_editing, unknown.
Output JSON only. Do not output Markdown.
JSON format:
{
  "label": "optional kept or revised label",
  "confidence": 0.0,
  "forgery_types": ["optional, at most one value"],
  "user_answer": {
    "judgment": "updated final judgment or direct answer",
    "reason_analysis": "explanation shown in the chat stream",
    "notes": ["relevant notes"],
    "forgery_regions": [{"region": "x0,y0,x1,y1", "score": 0.0, "reason": "optional"}]
  },
  "updated_fields": {
    "label_changed": false,
    "confidence_changed": false,
    "regions_changed": false
  }
}
"""
        payload = {
            "language": language,
            "followup_question": question,
            "latest_input": latest_input,
            "latest_result": self._compact_result(latest_result),
            "recent_history": history[-8:],
        }
        return f"{instructions}\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"

    def _compact_result(self, result: dict[str, Any]) -> dict[str, Any]:
        image_forensics = result.get("raw", {}).get("image_forensics", {}) if isinstance(result.get("raw"), dict) else {}
        return {
            "mode": result.get("mode"),
            "label": result.get("label"),
            "summary": result.get("summary"),
            "confidence": result.get("confidence"),
            "question": result.get("question"),
            "user_answer": result.get("user_answer"),
            "claims": result.get("claims", []),
            "text_verdict": result.get("text_verdict"),
            "image_verdict": result.get("image_verdict"),
            "crossmodal_verdict": result.get("crossmodal_verdict"),
            "report": result.get("report"),
            "tool_evidence": image_forensics.get("tool_evidence", []),
            "suspicious_regions": image_forensics.get("suspicious_regions") or result.get("image_verdict", {}).get("suspicious_regions", []),
            "forgery_types": image_forensics.get("forgery_types") or result.get("image_verdict", {}).get("forgery_types", []),
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
                return parsed if isinstance(parsed, dict) else {}
        return {}

    def _merge_result(
        self,
        base_result: dict[str, Any],
        parsed: dict[str, Any],
        question: str,
        raw_output: str,
        language: str,
    ) -> dict[str, Any]:
        result = deepcopy(base_result)
        if not result:
            result = {"mode": "followup", "label": "Cannot_Determine", "summary": ""}

        previous_label = result.get("label")
        if parsed.get("label"):
            result["label"] = str(parsed.get("label")).strip()
        if parsed.get("confidence") is not None:
            result["confidence"] = self._bounded_float(parsed.get("confidence"), result.get("confidence"))

        answer = self._normalize_user_answer(parsed.get("user_answer"), result, question, language)
        result["user_answer"] = answer
        result["summary"] = answer.get("judgment") or result.get("summary") or ""
        result["question"] = question

        if isinstance(parsed.get("forgery_types"), list):
            result["forgery_types"] = [
                self._canonical_forgery_type(item, language) for item in parsed["forgery_types"][:1] if str(item or "").strip()
            ]
        if answer.get("forgery_regions"):
            result["suspicious_regions"] = answer["forgery_regions"]

        image_verdict = result.get("image_verdict")
        if isinstance(image_verdict, dict):
            image_verdict["label"] = result.get("label", image_verdict.get("label"))
            image_verdict["confidence"] = result.get("confidence", image_verdict.get("confidence"))
            image_verdict["user_answer"] = answer
            image_verdict["reasoning"] = answer.get("judgment", image_verdict.get("reasoning", ""))
            if result.get("forgery_types"):
                image_verdict["forgery_types"] = result["forgery_types"]
            if answer.get("forgery_regions"):
                image_verdict["suspicious_regions"] = answer["forgery_regions"]

        result["followup"] = {
            "question": question,
            "updated_fields": parsed.get("updated_fields", {}),
            "previous_label": previous_label,
            "raw_output": raw_output,
        }
        return result

    def _normalize_user_answer(
        self,
        value: Any,
        result: dict[str, Any],
        question: str,
        language: str,
    ) -> dict[str, Any]:
        current = result.get("user_answer") if isinstance(result.get("user_answer"), dict) else {}
        if isinstance(value, dict):
            judgment = str(value.get("judgment") or value.get("conclusion") or "").strip()
            reason = str(value.get("reason_analysis") or value.get("analysis") or "").strip()
            notes = value.get("notes") if isinstance(value.get("notes"), list) else []
            regions = value.get("forgery_regions") if isinstance(value.get("forgery_regions"), list) else []
        else:
            judgment = ""
            reason = ""
            notes = []
            regions = []

        if not judgment:
            judgment = str(current.get("judgment") or current.get("conclusion") or result.get("summary") or "").strip()
        if not reason:
            reason = (
                f"关于“{question}”，当前只能基于上一次检测结果进行解释；原结论暂未改变。"
                if is_zh(language)
                else f"Regarding '{question}', this answer is based only on the previous detection result; the prior conclusion is unchanged."
            )
        clean_notes = [str(item).strip() for item in notes if str(item or "").strip()]
        clean_regions = [item for item in regions if isinstance(item, dict)]
        return {
            "judgment": judgment,
            "reason_analysis": reason,
            "notes": clean_notes,
            "forgery_regions": clean_regions,
            "conclusion": judgment,
            "analysis": reason,
        }

    def _canonical_forgery_type(self, value: Any, language: str) -> str:
        text = str(value or "").strip()
        key = text.lower().replace("-", "_").replace(" ", "_")
        zh_aliases = {
            "无": "无",
            "拼接": "拼接",
            "复制移动": "复制移动",
            "内容移除": "内容移除",
            "内容删除": "内容移除",
            "局部修复": "内容移除",
            "内容替换": "内容替换",
            "人脸篡改": "人脸篡改",
            "换脸": "人脸篡改",
            "局部重绘": "局部重绘",
            "重绘": "局部重绘",
            "ai辅助编辑": "局部重绘",
            "全局编辑": "全局编辑",
            "修饰改动": "全局编辑",
            "未知": "未知",
        }
        en_to_zh = {
            "none": "无",
            "splicing": "拼接",
            "copy_move": "复制移动",
            "inpainting": "内容移除",
            "content_removal": "内容移除",
            "content_deletion": "内容移除",
            "content_replacement": "内容替换",
            "face_swap": "人脸篡改",
            "face_tampering": "人脸篡改",
            "deepfake": "人脸篡改",
            "repainting": "局部重绘",
            "local_repainting": "局部重绘",
            "ai_edited": "局部重绘",
            "ai_generated": "局部重绘",
            "retouching": "全局编辑",
            "global_editing": "全局编辑",
            "unknown": "未知",
            "unknown_forgery": "未知",
        }
        zh_value = zh_aliases.get(text) or zh_aliases.get(text.lower()) or en_to_zh.get(key) or "未知"
        if is_zh(language):
            return zh_value
        return {
            "无": "none",
            "拼接": "splicing",
            "复制移动": "copy_move",
            "内容移除": "content_removal",
            "内容替换": "content_replacement",
            "人脸篡改": "face_tampering",
            "局部重绘": "local_repainting",
            "全局编辑": "global_editing",
            "未知": "unknown",
        }.get(zh_value, "unknown")

    def _bounded_float(self, value: Any, default: Any) -> float | None:
        try:
            number = float(value)
        except (TypeError, ValueError):
            try:
                return float(default) if default is not None else None
            except (TypeError, ValueError):
                return None
        return max(0.0, min(1.0, number))

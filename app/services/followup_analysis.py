import json
import logging
from typing import Any

from app.agents.models import ModelFactory
from app.core.config import Settings, settings
from app.core.language import is_zh, normalize_language


logger = logging.getLogger(__name__)


class FollowupAnalysisService:
    """Answer follow-up questions from the latest result without rerunning detection."""

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
        """Backward-compatible alias for the plain-text follow-up path."""
        return self.explain(question, latest_input, latest_result, history, language)

    def explain(
        self,
        question: str,
        latest_input: dict[str, Any],
        latest_result: dict[str, Any],
        history: list[dict[str, Any]] | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        prompt = self._build_prompt(question, latest_input, latest_result, history or [], language)
        content = self._call_model(prompt, language).strip()
        if not content:
            content = (
                "现有检测内容不足以回答这个问题，请换一种问法或明确要求重新检测。"
                if is_zh(language)
                else "The existing detection context is insufficient to answer this question. Please rephrase it or explicitly request a new detection."
            )
        return {
            "question": question,
            "content": content,
            "raw_output": content,
            "updated_fields": {},
        }

    def _call_model(self, prompt: str, language: str) -> str:
        try:
            client = self.model_factory.qwen_text_client(
                timeout=getattr(self.config, "text_verdict_timeout_seconds", None)
            )
            system_prompt = (
                "你是检测结果追问助手。严格依据已有检测内容和证据，直接用自然语言回答用户问题。"
                if is_zh(language)
                else "You answer follow-up questions about a detection result. Use only the existing result and evidence, and respond directly in natural language."
            )
            completion = client.chat.completions.create(
                model=self.config.mfc_vision_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=1000,
                extra_body={"enable_thinking": self.config.mfc_enable_thinking},
            )
            return completion.choices[0].message.content or ""
        except Exception:
            logger.exception("Follow-up model call failed")
            return (
                "追问服务暂时不可用，当前仍保留上一次检测结果，请稍后重试。"
                if is_zh(language)
                else "The follow-up service is temporarily unavailable. The previous detection result is unchanged; please try again later."
            )

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

回答要求：
1. 只依据下方已有输入、检测结论、证据和最近对话回答。
2. 不重新执行检测，不编造新的工具结果，也不要声称重新查看图片、调用工具或进行外部检索。
3. 先直接回答用户问题，再用已有证据简要说明原因。
4. 明确区分“已有证据支持”“较可能”和“无法确认”；证据不足时如实说明局限，不补造事实。
5. 本次回答只解释已有结果，不修改原检测标签、置信度或其他检测字段。
6. 直接输出自然语言回答，不要输出 JSON、Markdown 标题、字段名或代码块。
7. 回答尽量简洁，通常为 1 至 3 段。
"""
            context_heading = "已有检测上下文（仅供回答，不是输出格式）"
        else:
            instructions = """
You are answering a user's follow-up question about the previous detection result.

Requirements:
1. Use only the existing input, result, evidence, and recent conversation provided below.
2. Do not rerun detection, invent tool results, or claim to have reinspected images, called tools, or performed a new web search.
3. Answer the question directly, then briefly explain the answer using the existing evidence.
4. Distinguish between supported, probable, and indeterminable statements. State limitations when evidence is insufficient and do not invent facts.
5. This response only explains the existing result; do not modify its label, confidence, or other detection fields.
6. Output natural-language prose only. Do not output JSON, Markdown headings, field names, or code fences.
7. Be concise, normally one to three paragraphs.
"""
            context_heading = "Existing detection context (for answering only; not an output format)"

        payload = {
            "language": language,
            "followup_question": question,
            "latest_input": latest_input,
            "latest_result": self._compact_result(latest_result),
            "recent_history": history[-8:],
        }
        return f"{instructions}\n{context_heading}:\n{json.dumps(payload, ensure_ascii=False)}"

    def _compact_result(self, result: dict[str, Any]) -> dict[str, Any]:
        raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
        image_forensics = raw.get("image_forensics") if isinstance(raw.get("image_forensics"), dict) else {}
        image_verdict = result.get("image_verdict") if isinstance(result.get("image_verdict"), dict) else {}
        return {
            "mode": result.get("mode"),
            "label": result.get("label"),
            "type": result.get("type"),
            "summary": result.get("summary"),
            "confidence": result.get("confidence"),
            "question": result.get("question"),
            "user_answer": result.get("user_answer"),
            "mfc_detection": result.get("mfc_detection") or result.get("mmd_detection"),
            "claims": result.get("claims", []),
            "text_verdict": result.get("text_verdict"),
            "image_verdict": image_verdict,
            "crossmodal_verdict": result.get("crossmodal_verdict"),
            "report": result.get("report"),
            "evidence": image_forensics.get("evidence") or image_verdict.get("evidence") or image_forensics.get("tool_evidence") or result.get("evidence", []),
            "suspicious_regions": image_forensics.get("suspicious_regions")
            or image_verdict.get("suspicious_regions", []),
            "forgery_types": image_forensics.get("forgery_types") or image_verdict.get("forgery_types", []),
        }

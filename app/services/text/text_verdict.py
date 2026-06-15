import json
import re
from collections.abc import Callable
from typing import Any

from app.agents.models import ModelFactory
from app.core.config import Settings, settings
from app.core.language import is_zh, normalize_language
from app.prompts.text import TEXT_VERDICT_JSON_PROMPT_EN, TEXT_VERDICT_JSON_PROMPT_ZH

VerdictEvaluator = Callable[[str, float | None], str]


class TextVerdictService:
    """Produce a structured final text-level verdict from text, claims, and evidence."""

    LABELS = {
        "Completely_False",
        "Mostly_False",
        "Mixed",
        "Mostly_True",
        "Completely_True",
        "Likely_False",
        "Likely_True",
        "Cannot_Determine",
    }
    LABEL_ALIASES = {
        "完全虚假": "Completely_False",
        "大部分虚假": "Mostly_False",
        "真假参半": "Mixed",
        "大部分真实": "Mostly_True",
        "完全真实": "Completely_True",
        "可能虚假": "Likely_False",
        "可能真实": "Likely_True",
        "无法判断": "Cannot_Determine",
        "无法鉴定": "Cannot_Determine",
    }
    LABELS_ZH = {
        "Completely_False": "完全虚假",
        "Mostly_False": "大部分虚假",
        "Mixed": "真假参半",
        "Mostly_True": "大部分真实",
        "Completely_True": "完全真实",
        "Likely_False": "可能虚假",
        "Likely_True": "可能真实",
        "Cannot_Determine": "无法判断",
    }

    def __init__(
        self,
        config: Settings = settings,
        model_factory: ModelFactory | None = None,
        evaluator: VerdictEvaluator | None = None,
    ):
        if hasattr(config, "config") and hasattr(config, "model_factory") and model_factory is None:
            legacy_fact_checker = config
            config = legacy_fact_checker.config
            model_factory = legacy_fact_checker.model_factory
        self.config = config
        self.model_factory = model_factory or ModelFactory(config)
        self.evaluator = evaluator

    def evaluate(
        self,
        text: str,
        claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]] | None = None,
        history: list | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        evidence = evidence or []
        prompt = self._build_prompt(text, claims, evidence, history or [], language)
        raw_output = self._call_model(prompt, language)
        parsed = self._parse_json(raw_output)
        return self._normalize_result(parsed, raw_output, claims, evidence, language)

    def normalize_label(self, label: Any) -> str:
        label_text = str(label or "").strip()
        if label_text in self.LABELS:
            return label_text
        return self.LABEL_ALIASES.get(label_text, "Cannot_Determine")

    def _call_model(self, prompt: str, language: str) -> str:
        if self.evaluator:
            return self.evaluator(prompt, self.config.text_verdict_timeout_seconds)

        try:
            client = self.model_factory.deepseek_client(timeout=self.config.text_verdict_timeout_seconds)
            system_prompt = (
                "你是严格输出 JSON 的虚假信息判定器。中文请求必须输出中文。"
                if is_zh(language)
                else "You are a strict JSON-only misinformation verdict evaluator."
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
            if is_zh(language):
                return json.dumps({
                    "label": "无法判断",
                    "confidence": 0.0,
                    "summary": "文本判定模型调用失败。",
                    "reasoning": str(exc),
                    "evidence_assessment": "由于模型或网络异常，无法评估检索证据。",
                    "user_answer": {
                        "analysis": "系统已完成声明和证据输入整理，但最终判定模型调用失败，因此无法给出可靠分析。",
                        "conclusion": "最终标签：无法判断。置信度：0.00。",
                    },
                    "claim_assessments": [],
                }, ensure_ascii=False)
            return json.dumps({
                "label": "Cannot_Determine",
                "confidence": 0.0,
                "summary": "Text verdict model call failed.",
                "reasoning": str(exc),
                "evidence_assessment": "Unable to evaluate evidence due to model or network failure.",
                "user_answer": {
                    "analysis": "The final verdict model call failed after preparing the claims and evidence.",
                    "conclusion": "Final label: Cannot Determine. Confidence: 0.00.",
                },
                "claim_assessments": [],
            }, ensure_ascii=False)

    def _build_prompt(
        self,
        text: str,
        claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        history: list[dict[str, Any]],
        language: str,
    ) -> str:
        base_prompt = TEXT_VERDICT_JSON_PROMPT_ZH if is_zh(language) else TEXT_VERDICT_JSON_PROMPT_EN
        payload = {
            "language": language,
            "text": text,
            "claims": claims,
            "evidence": evidence,
            "history": history[-5:],
        }
        return f"{base_prompt}\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"

    def _parse_json(self, raw_output: str) -> dict[str, Any]:
        payload = self._extract_json_payload(raw_output)
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _extract_json_payload(self, raw_output: str) -> str:
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
                    return json.dumps(parsed, ensure_ascii=False)
        return text

    def _normalize_result(
        self,
        parsed: dict[str, Any],
        raw_output: str,
        claims: list[dict[str, Any]],
        evidence: list[dict[str, Any]],
        language: str,
    ) -> dict[str, Any]:
        label = self.normalize_label(parsed.get("label"))
        confidence = self._normalize_confidence(parsed.get("confidence"))
        verdict_type = self._verdict_type(label, language)
        summary = str(parsed.get("summary") or "").strip()
        reasoning = str(parsed.get("reasoning") or "").strip()
        evidence_assessment = str(parsed.get("evidence_assessment") or "").strip()
        user_answer = self._normalize_user_answer(
            parsed.get("user_answer"),
            summary,
            reasoning,
            label,
            confidence,
            language,
        )
        claim_assessments = parsed.get("claim_assessments", [])
        if not isinstance(claim_assessments, list):
            claim_assessments = []

        appraisal = summary or reasoning or (
            "无法生成可靠的文本判定结果。"
            if is_zh(language)
            else "Unable to determine a reliable text verdict."
        )
        explain = {
            "text": reasoning,
            "evidence_assessment": evidence_assessment,
            "claim_assessments": claim_assessments,
            "news": self._flatten_evidence(evidence),
        }
        return {
            "type": verdict_type,
            "label": label,
            "confidence": confidence,
            "appraisalResults": appraisal,
            "explainInDetail": explain,
            "claim_assessments": claim_assessments,
            "evidence_assessment": evidence_assessment,
            "user_answer": user_answer,
            "label_policy": {
                "model_judged_evidence_sufficiency": True,
                "claim_count": len(claims),
                "evidence_bundle_count": len(evidence),
            },
            "raw_output": raw_output,
        }

    def _verdict_type(self, label: str, language: str) -> str:
        fact_labels = {"Completely_False", "Mostly_False", "Mixed", "Mostly_True", "Completely_True"}
        if label in fact_labels:
            return "事实核查" if is_zh(language) else "fact_checking"
        return "谣言检测" if is_zh(language) else "rumor_detection"

    def _normalize_confidence(self, value: Any) -> float | None:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        return max(0.0, min(1.0, confidence))

    def _normalize_user_answer(
        self,
        value: Any,
        summary: str,
        reasoning: str,
        label: str,
        confidence: float | None,
        language: str,
    ) -> dict[str, str]:
        if isinstance(value, dict):
            analysis = str(value.get("analysis") or "").strip()
            conclusion = str(value.get("conclusion") or "").strip()
        else:
            analysis = ""
            conclusion = ""

        if not analysis:
            analysis = reasoning or summary or (
                "未生成详细分析。"
                if is_zh(language)
                else "No detailed analysis was generated."
            )
        if not conclusion:
            confidence_text = f"{confidence:.2f}" if isinstance(confidence, float) else "unknown"
            if is_zh(language):
                label_text = self.LABELS_ZH.get(label, label)
                confidence_text = f"{confidence:.2f}" if isinstance(confidence, float) else "未知"
                conclusion = f"最终标签：{label_text}。置信度：{confidence_text}。"
            else:
                conclusion = f"Final label: {label}. Confidence: {confidence_text}."
        return {"analysis": analysis, "conclusion": conclusion}

    def _flatten_evidence(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flattened = []
        for bundle in evidence:
            if not isinstance(bundle, dict):
                continue
            for item in bundle.get("items", []):
                if isinstance(item, dict):
                    flattened.append(item)
        return flattened

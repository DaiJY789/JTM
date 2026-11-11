from __future__ import annotations

from typing import Any

from app.agents.MFC_Agent import MFCDetectionAgent
from app.core.language import is_zh, normalize_language


class MFCDetectionService:
    """Adapt the single-agent MFC result to the application's public response."""

    def __init__(self, agent: MFCDetectionAgent):
        self.agent = agent

    def detect(
        self,
        *,
        text: str,
        image_content: bytes | None,
        image_url: str | None,
        image_contents: list[bytes] | tuple[bytes, ...] | None = None,
        image_urls: list[str] | tuple[str, ...] | None = None,
        image_labels: list[str] | tuple[str, ...] | None = None,
        question: str,
        language: str,
    ) -> dict[str, Any]:
        language = normalize_language(language)
        has_image = bool(image_content or image_url or image_contents or image_urls)
        detection = self.agent.detect(
            text=text,
            image_content=image_content,
            image_url=image_url,
            image_contents=image_contents,
            image_urls=image_urls,
            image_labels=image_labels,
            question=question,
            language=language,
        )
        if detection.get("status") != "success" or not detection.get("label"):
            return self._error_response(detection, question, language, has_image=has_image)

        verdict = str(detection["label"])
        public_detection = self._public_detection(detection)
        explanation = dict(detection.get("user_answer") or {})
        limitations = self._strings(explanation.get("notes"))
        crossmodal = self._crossmodal_verdict(detection, language) if has_image else None
        user_answer = self._user_answer(detection)
        evidence = self._evidence(detection)
        tool_dependencies = self._tool_dependencies(detection)
        workflow = {
            "name": "multimodal_fact_checking" if has_image else "text_fact_checking",
            "routing": "single_agent",
            "steps": ["MFC_Agent"],
            "dependencies": tool_dependencies,
        }
        report = {
            "label": verdict,
            "type": detection.get("type") or "",
            "summary": str(explanation.get("analysis") or explanation.get("judgment") or "").strip(),
            "confidence": detection.get("confidence"),
            "risk_level": self._risk_level(verdict),
            "decision_status": detection.get("decision_status"),
            "limitations": limitations,
            "user_answer": user_answer,
            "agent_trace": workflow,
        }
        return {
            "mode": "multimodal" if has_image else "text_only",
            "label": verdict,
            "type": detection.get("type") or "",
            "summary": report["summary"],
            "confidence": detection.get("confidence"),
            "decision_status": detection.get("decision_status"),
            "mfc_detection": public_detection,
            "text_verdict": {
                "type": detection.get("type") or "",
                "label": verdict,
                "appraisalResults": report["summary"],
                "explainInDetail": {"fine_grained_checks": detection.get("fine_grained_checks", {})},
                "user_answer": user_answer,
                "limitations": limitations,
            },
            "image_verdict": None,
            "crossmodal_verdict": crossmodal,
            "evidence": evidence,
            "claims": [],
            "report": report,
            "user_answer": user_answer,
            "raw": {
                "workflow": workflow,
                "mfc_diagnostics": detection.get("diagnostics", {}),
                "mfc_tool_evidence": detection.get("tool_context", {}),
                "timings": {
                    "agents": {"MFC_Agent": detection.get("timings", {}).get("total_seconds", 0.0)},
                    "critical_path_seconds": detection.get("timings", {}).get("total_seconds", 0.0),
                    "parallel_groups": self._parallel_groups(detection),
                    "model_calls": detection.get("timings", {}).get("model_calls", 1),
                    "image_preprocess_seconds": detection.get("timings", {}).get("image_preprocess_seconds", 0.0),
                    "enrichment_seconds": detection.get("timings", {}).get("enrichment_seconds", 0.0),
                    "tool_seconds": detection.get("timings", {}).get("tool_seconds", 0.0),
                    "tool_rounds": detection.get("timings", {}).get("tool_rounds", 0),
                    "search_calls": detection.get("timings", {}).get("search_calls", 0),
                    "ocr_seconds": detection.get("timings", {}).get("ocr_seconds", 0.0),
                    "search_seconds": detection.get("timings", {}).get("search_seconds", 0.0),
                    "model_seconds": detection.get("timings", {}).get("model_seconds", 0.0),
                    "model_output_characters": detection.get("timings", {}).get("model_output_characters", 0),
                    "input_image_count": detection.get("timings", {}).get("input_image_count", 1),
                    "thinking_enabled": detection.get("timings", {}).get("thinking_enabled", False),
                    "thinking_characters": detection.get("timings", {}).get("thinking_characters", 0),
                    "finish_reason": detection.get("timings", {}).get("finish_reason", ""),
                    "prompt_tokens": detection.get("timings", {}).get("prompt_tokens", 0),
                    "completion_tokens": detection.get("timings", {}).get("completion_tokens", 0),
                    "reasoning_tokens": detection.get("timings", {}).get("reasoning_tokens", 0),
                    "invalid_output_retries": detection.get("timings", {}).get("invalid_output_retries", 0),
                    "total_model_output_characters": detection.get("timings", {}).get("total_model_output_characters", 0),
                    "model_attempts": detection.get("timings", {}).get("model_attempts", []),
                },
            },
            "question": question,
        }

    def _user_answer(self, detection: dict[str, Any]) -> dict[str, Any]:
        answer = dict(detection.get("user_answer") or {})
        return {
            "judgment": str(answer.get("judgment") or "").strip(),
            "analysis": str(answer.get("analysis") or "").strip(),
            "notes": self._strings(answer.get("notes")),
        }

    def _signal_summary(self, signals: dict[str, Any], prefix: str, language: str) -> str:
        parts = []
        unverifiable = []
        for key, value in signals.items():
            if not key.startswith(prefix) or not isinstance(value, dict):
                continue
            evidence = str(value.get("evidence") or "").strip()
            if value.get("assessment") in {"confirmed", "suspected"} and evidence:
                parts.append(evidence)
            elif value.get("assessment") == "not_verifiable" and evidence:
                unverifiable.append(evidence)
        if parts:
            return "；".join(parts)
        if unverifiable:
            prefix_text = "当前无法验证：" if is_zh(language) else "Currently unverifiable: "
            return f"{prefix_text}{'；'.join(unverifiable[:2])}"
        defaults = {
            "T": ("未发现明确的文字虚假线索。", "No explicit textual misinformation signal was found."),
            "V": ("未发现明确的视觉生成或编辑线索。", "No explicit visual generation or editing signal was found."),
            "C": ("未发现明确的图文冲突。", "No explicit image-text conflict was found."),
        }
        return defaults[prefix][0 if is_zh(language) else 1]

    def _crossmodal_verdict(self, detection: dict[str, Any], language: str) -> dict[str, Any]:
        signals = (
            detection.get("fine_grained_checks")
            if isinstance(detection.get("fine_grained_checks"), dict)
            else {}
        )
        assessment = (
            detection.get("crossmodal_assessment")
            if isinstance(detection.get("crossmodal_assessment"), dict)
            else {}
        )
        explicit_label = str(assessment.get("label") or "").strip()
        if explicit_label in {"Consistent", "Partially_Consistent", "Inconsistent", "Cannot_Determine"}:
            reason = str(assessment.get("reason") or "").strip()
            if not reason:
                reason = (
                    "当前单张配图不足以确认核心图文关系。"
                    if is_zh(language)
                    else "The selected image is insufficient to confirm the core image-text relation."
                )
            matched = [reason] if explicit_label == "Consistent" else []
            mismatched = [reason] if explicit_label in {"Inconsistent", "Partially_Consistent"} else []
            missing = []
            if explicit_label == "Cannot_Determine":
                missing = [
                    str(item.get("evidence"))
                    for key, item in signals.items()
                    if key.startswith("C")
                    and isinstance(item, dict)
                    and item.get("assessment") == "not_verifiable"
                    and item.get("evidence")
                ] or [reason]
            scores = {
                "Consistent": 1.0,
                "Partially_Consistent": 0.5,
                "Inconsistent": 0.0,
                "Cannot_Determine": None,
            }
            return {
                "label": explicit_label,
                "analysis": reason,
                "matched_elements": matched,
                "mismatched_elements": mismatched,
                "missing_elements": missing,
                "reasoning": reason,
                "limitations": self._strings(detection.get("user_answer", {}).get("notes")),
                "consistency_score": scores[explicit_label],
            }

        crossmodal = [value for key, value in signals.items() if key.startswith("C") and isinstance(value, dict)]
        confirmed = [
            item for item in crossmodal
            if item.get("assessment") == "confirmed" and int(item.get("severity") or 0) >= 2
        ]
        suspected = [
            item for item in crossmodal
            if item.get("assessment") == "suspected"
            or (item.get("assessment") == "confirmed" and int(item.get("severity") or 0) == 1)
        ]
        unverifiable = [item for item in crossmodal if item.get("assessment") == "not_verifiable"]
        absent = [item for item in crossmodal if item.get("assessment") == "absent"]
        if confirmed:
            label = "Inconsistent"
        elif suspected:
            label = "Partially_Consistent"
        elif unverifiable:
            label = "Cannot_Determine"
        elif absent:
            label = "Consistent"
        else:
            label = "Cannot_Determine"
        mismatched = [str(item.get("evidence")) for item in confirmed if item.get("evidence")]
        missing = [str(item.get("evidence")) for item in unverifiable if item.get("evidence")]
        matched = [str(item.get("evidence")) for item in absent if item.get("evidence")]
        reasoning = "；".join(mismatched)
        if not reasoning and suspected:
            reasoning = "；".join(str(item.get("evidence")) for item in suspected if item.get("evidence"))
        if not reasoning and missing:
            reasoning = (
                f"当前输入无法验证部分图文关系：{'；'.join(missing[:2])}"
                if is_zh(language)
                else f"The current input cannot verify part of the image-text relation: {'; '.join(missing[:2])}"
            )
        if not reasoning:
            reasoning = (
                "当前单张配图不足以确认核心图文关系。"
                if label == "Cannot_Determine" and is_zh(language)
                else "The selected image is insufficient to confirm the core image-text relation."
                if label == "Cannot_Determine"
                else "未发现明确的图文冲突。"
                if is_zh(language)
                else "No explicit image-text conflict was found."
            )
        if label == "Inconsistent":
            consistency_score = round(1.0 - max(int(item.get("severity") or 0) for item in confirmed) / 3.0, 4)
        elif label == "Partially_Consistent":
            consistency_score = 0.5
        elif label == "Consistent":
            consistency_score = 1.0
        else:
            consistency_score = None
        return {
            "label": label,
            "analysis": reasoning,
            "matched_elements": matched,
            "mismatched_elements": mismatched,
            "missing_elements": missing,
            "reasoning": reasoning,
            "limitations": self._strings(detection.get("user_answer", {}).get("notes")),
            "consistency_score": consistency_score,
        }

    def _evidence(self, detection: dict[str, Any]) -> list[dict[str, Any]]:
        evidence = []
        tool_context = detection.get("tool_context") if isinstance(detection.get("tool_context"), dict) else {}
        ocr = tool_context.get("ocr") if isinstance(tool_context.get("ocr"), dict) else {}
        if ocr.get("status") == "success" and (
            ocr.get("ocr_text") or ocr.get("entities") or ocr.get("visual_objects")
        ):
            evidence.append(
                {
                    "type": "mfc_ocr_observation",
                    "source": "ImageOCREntityTool",
                    "title": "输入图片 OCR 与实体识别",
                    "content": {
                        "ocr_text": ocr.get("ocr_text", ""),
                        "entities": ocr.get("entities", []),
                        "visual_objects": ocr.get("visual_objects", []),
                        "limitations": ocr.get("limitations", []),
                    },
                }
            )
        search = tool_context.get("search") if isinstance(tool_context.get("search"), dict) else {}
        for item in search.get("results", []) if isinstance(search.get("results"), list) else []:
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "type": "external_search_result",
                    "source": str(item.get("site_name") or item.get("domain") or "BoCha"),
                    "title": str(item.get("title") or ""),
                    "url": str(item.get("url") or ""),
                    "snippet": str(item.get("snippet") or ""),
                    "content": {
                        "evidence_id": item.get("evidence_id"),
                        "query_id": item.get("query_id"),
                        "domain": item.get("domain"),
                        "date_last_crawled": item.get("date_last_crawled"),
                        "date_note": "date_last_crawled is crawl time, not publication time",
                    },
                }
            )
        for item in detection.get("atomic_points", []) if isinstance(detection.get("atomic_points"), list) else []:
            if not isinstance(item, dict):
                continue
            evidence.append(
                {
                    "type": "mfc_atomic_point",
                    "source": "MFC_Agent",
                    "title": str(item.get("claim") or ""),
                    "content": {
                        "point_type": item.get("point_type"),
                        "visual_observation": item.get("visual_observation"),
                        "search_query": item.get("search_query"),
                    },
                }
            )
        signals = (
            detection.get("fine_grained_checks")
            if isinstance(detection.get("fine_grained_checks"), dict)
            else {}
        )
        for key, value in signals.items():
            if (
                not isinstance(value, dict)
                or value.get("assessment") not in {"confirmed", "suspected"}
                or not value.get("evidence")
            ):
                continue
            evidence.append(
                {
                    "type": "mfc_fine_grained_check",
                    "source": "MFC_Agent",
                    "title": key,
                    "content": {
                        "severity": value.get("severity", 0),
                        "assessment": value.get("assessment"),
                        "evidence_basis": value.get("evidence_basis"),
                        "observation": value.get("analysis", ""),
                        "regions": value.get("regions", []),
                    },
                }
            )
        return evidence

    def _tool_dependencies(self, detection: dict[str, Any]) -> list[str]:
        context = detection.get("tool_context") if isinstance(detection.get("tool_context"), dict) else {}
        if not context.get("enabled"):
            return []
        dependencies = []
        if isinstance(context.get("ocr"), dict) and context["ocr"].get("status") != "not_run":
            dependencies.append("ImageOCREntityTool")
        search = context.get("search") if isinstance(context.get("search"), dict) else {}
        if search.get("queries"):
            dependencies.append("BoChaSearchTool")
        return dependencies

    def _parallel_groups(self, detection: dict[str, Any]) -> list[list[str]]:
        context = detection.get("tool_context") if isinstance(detection.get("tool_context"), dict) else {}
        if not context.get("enabled"):
            return []
        ocr = context.get("ocr") if isinstance(context.get("ocr"), dict) else {}
        search = context.get("search") if isinstance(context.get("search"), dict) else {}
        if any("round" in item for item in search.get("queries", []) if isinstance(item, dict)):
            rounds: dict[int, list[str]] = {}
            for item in search["queries"]:
                if isinstance(item, dict) and item.get("query_id"):
                    rounds.setdefault(int(item.get("round", 1)), []).append(f"BoChaSearchTool:{item['query_id']}")
            return [group for group in rounds.values() if len(group) > 1]
        query_ids = [
            str(item.get("query_id") or "")
            for item in search.get("queries", [])
            if isinstance(item, dict) and item.get("query_id")
        ]
        group = []
        if ocr.get("status") != "not_run":
            group.append("ImageOCREntityTool")
            query_ids = [query_id for query_id in query_ids if query_id in {"Q1", "Q2"}]
        group.extend(f"BoChaSearchTool:{query_id}" for query_id in query_ids)
        return [group] if len(group) > 1 else []

    def _error_response(
        self,
        detection: dict[str, Any],
        question: str,
        language: str,
        *,
        has_image: bool,
    ) -> dict[str, Any]:
        verdict = None
        confidence = None
        explanation = dict(detection.get("user_answer") or {})
        model_attempted = int(detection.get("timings", {}).get("model_calls") or 0) > 0
        message = (
            "服务器繁忙，检测暂时失败，请稍后重试。"
            if is_zh(language)
            else "The server is busy and detection failed. Please try again later."
        ) if model_attempted else str(explanation.get("analysis") or "").strip() or (
            "检测服务暂时不可用。" if is_zh(language) else "The detection service is temporarily unavailable."
        )
        limitations = self._strings(explanation.get("notes")) or [message]
        user_answer = {
            "judgment": message,
            "analysis": message,
            "notes": limitations,
        }
        tool_dependencies = self._tool_dependencies(detection)
        return {
            "mode": "multimodal" if has_image else "text_only",
            "label": verdict,
            "type": detection.get("type") or "",
            "summary": message,
            "confidence": confidence,
            "decision_status": "technical_error",
            "mfc_detection": self._public_detection(detection),
            "text_verdict": None,
            "image_verdict": None,
            "crossmodal_verdict": None,
            "evidence": [],
            "claims": [],
            "report": {
                "label": verdict,
                "summary": message,
                "confidence": confidence,
                "decision_status": "technical_error",
                "limitations": limitations,
            },
            "user_answer": user_answer,
            "raw": {
                "workflow": {
                    "name": "multimodal_fact_checking" if has_image else "text_fact_checking",
                    "routing": "single_agent",
                    "steps": ["MFC_Agent"],
                    "dependencies": tool_dependencies,
                },
                "mfc_tool_evidence": detection.get("tool_context", {}),
                "mfc_diagnostics": detection.get("diagnostics", {}),
                "timings": {
                    "agents": {"MFC_Agent": detection.get("timings", {}).get("total_seconds", 0.0)},
                    "critical_path_seconds": detection.get("timings", {}).get("total_seconds", 0.0),
                    "parallel_groups": self._parallel_groups(detection),
                    "model_calls": detection.get("timings", {}).get("model_calls", 0),
                    "image_preprocess_seconds": detection.get("timings", {}).get("image_preprocess_seconds", 0.0),
                    "enrichment_seconds": detection.get("timings", {}).get("enrichment_seconds", 0.0),
                    "tool_seconds": detection.get("timings", {}).get("tool_seconds", 0.0),
                    "tool_rounds": detection.get("timings", {}).get("tool_rounds", 0),
                    "search_calls": detection.get("timings", {}).get("search_calls", 0),
                    "ocr_seconds": detection.get("timings", {}).get("ocr_seconds", 0.0),
                    "search_seconds": detection.get("timings", {}).get("search_seconds", 0.0),
                    "model_seconds": detection.get("timings", {}).get("model_seconds", 0.0),
                    "model_output_characters": detection.get("timings", {}).get("model_output_characters", 0),
                    "input_image_count": detection.get("timings", {}).get("input_image_count", 0),
                    "thinking_enabled": detection.get("timings", {}).get("thinking_enabled", False),
                    "thinking_characters": detection.get("timings", {}).get("thinking_characters", 0),
                    "finish_reason": detection.get("timings", {}).get("finish_reason", ""),
                    "prompt_tokens": detection.get("timings", {}).get("prompt_tokens", 0),
                    "completion_tokens": detection.get("timings", {}).get("completion_tokens", 0),
                    "reasoning_tokens": detection.get("timings", {}).get("reasoning_tokens", 0),
                    "invalid_output_retries": detection.get("timings", {}).get("invalid_output_retries", 0),
                    "total_model_output_characters": detection.get("timings", {}).get("total_model_output_characters", 0),
                    "model_attempts": detection.get("timings", {}).get("model_attempts", []),
                },
            },
            "question": question,
        }

    def _public_detection(self, detection: dict[str, Any]) -> dict[str, Any]:
        """Expose only fields needed by API consumers and the user-facing UI."""
        public_checks = {}
        source_checks = detection.get("fine_grained_checks") or {}
        for key, value in source_checks.items():
            if not isinstance(value, dict):
                continue
            check = {
                "assessment": value.get("assessment"),
                "severity": value.get("severity", 0),
                "evidence_basis": value.get("evidence_basis"),
                "analysis": value.get("analysis", ""),
            }
            if value.get("regions"):
                check["regions"] = value["regions"]
            public_checks[key] = check
        result = {
            "status": detection.get("status"),
            "label": detection.get("label"),
            "type": detection.get("type"),
            "confidence": detection.get("confidence"),
            "atomic_points": detection.get("atomic_points", []),
            "fine_grained_checks": public_checks,
            "user_answer": detection.get("user_answer", {}),
            "evidence": detection.get("evidence", []),
        }
        if detection.get("error"):
            result["error"] = detection["error"]
        return result

    def _risk_level(self, verdict: str) -> str:
        if verdict in {"False", "Mostly False"}:
            return "high"
        if verdict == "Mixture":
            return "medium"
        return "low"

    def _strings(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


__all__ = ["MFCDetectionService"]

from __future__ import annotations

import time
from typing import Any

from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.followup_analysis import FollowupAnalysisService
from app.services.session_service import SessionService


class FollowupContextError(ValueError):
    pass


class FollowupApplicationService:
    """Answer from session state, rerunning detection only on explicit intent."""

    def __init__(self, session_service: SessionService, analysis_service: FollowupAnalysisService):
        self.session_service = session_service
        self.analysis_service = analysis_service

    def followup(
        self,
        *,
        question: str,
        conversation_id: str | None,
        language: str,
        orchestrator: DetectionOrchestrator,
    ) -> dict[str, Any]:
        started_at = time.perf_counter()
        resolved_id, context = self.session_service.latest_context(conversation_id)
        latest_result = context.get("latest_result") or {}
        if not latest_result:
            raise FollowupContextError("当前对话还没有可追问的检测结果。")

        latest_input = context.get("latest_input", {}) or {}
        history = context.get("conversation_history", [])
        if should_rerun_detection(question):
            return self._rerun(
                question=question,
                conversation_id=resolved_id,
                language=language,
                latest_input=latest_input,
                latest_result=latest_result,
                history=history,
                orchestrator=orchestrator,
                started_at=started_at,
            )

        explanation = self.analysis_service.explain(
            question=question,
            latest_input=latest_input,
            latest_result=latest_result,
            history=history,
            language=language,
        )
        result = dict(latest_result)
        result["conversation_id"] = resolved_id
        result["conversation_history"] = self.session_service.append_followup(
            resolved_id,
            question,
            result,
            language,
            assistant_content=explanation.get("content"),
            update_latest_result=False,
        )
        return result

    def _rerun(
        self,
        *,
        question: str,
        conversation_id: str,
        language: str,
        latest_input: dict[str, Any],
        latest_result: dict[str, Any],
        history: list[dict[str, Any]],
        orchestrator: DetectionOrchestrator,
        started_at: float,
    ) -> dict[str, Any]:
        image_content = self.session_service.load_latest_image(conversation_id, latest_input)
        result = orchestrator.detect(
            text=latest_input.get("text") or None,
            image_content=image_content,
            image_url=latest_input.get("image_url") or None,
            question=question,
            history=history,
            language=language,
        )
        result["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
        result["conversation_id"] = conversation_id
        next_input = {
            **latest_input,
            "mode": "auto",
            "language": language,
            "question": result.get("question") or question,
            "has_image": bool(image_content) or bool(latest_input.get("image_url")),
        }
        result["conversation_history"] = self.session_service.append_detection(
            conversation_id,
            next_input,
            result,
        )
        result["followup"] = {"question": question, "rerun_detection": True}
        return result

def should_rerun_detection(question: str) -> bool:
    text = " ".join(str(question or "").lower().split())
    if not text:
        return False
    negative_patterns = [
        "不要重新检测",
        "不用重新检测",
        "无需重新检测",
        "不要重跑",
        "不用重跑",
        "无需重跑",
        "不要重新跑",
        "不用重新跑",
        "do not rerun",
        "don't rerun",
        "no rerun",
        "without rerunning",
    ]
    if any(pattern in text for pattern in negative_patterns):
        return False
    rerun_patterns = [
        "重新检测",
        "重新分析",
        "重新判断",
        "重新评估",
        "重新鉴定",
        "重新识别",
        "重新跑",
        "重跑",
        "再检测",
        "再分析",
        "再判断",
        "再评估",
        "再跑一遍",
        "重新调用工具",
        "重新使用工具",
        "重跑工具",
        "跑一下工具",
        "重新走检测",
        "完整检测",
        "从头检测",
        "从头分析",
        "不依赖已有",
        "不要依赖已有",
        "rerun",
        "re-run",
        "detect again",
        "reanalyze",
        "re-analyze",
        "run the tools again",
        "run tools again",
        "from scratch",
        "full detection",
    ]
    return any(pattern in text for pattern in rerun_patterns)

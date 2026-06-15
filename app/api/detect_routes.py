import json
import time
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.dependencies import get_detection_orchestrator
from app.schemas.detection import DetectionMode, DetectionResponse
from app.services.conversation_history import ConversationHistoryStore
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.followup_analysis import FollowupAnalysisService

detect_router = APIRouter()
conversation_store = ConversationHistoryStore()
followup_service = FollowupAnalysisService()


def parse_history(history: str) -> list:
    try:
        parsed = json.loads(history or "[]")
        return parsed if isinstance(parsed, list) else []
    except json.JSONDecodeError:
        return []


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


def mode_from_value(value: object) -> DetectionMode:
    try:
        return DetectionMode(str(value or DetectionMode.AUTO.value))
    except ValueError:
        return DetectionMode.AUTO


def load_stored_image_content(conversation_id: str, latest_input: dict) -> bytes | None:
    image_asset = latest_input.get("image_asset") if isinstance(latest_input, dict) else {}
    if not isinstance(image_asset, dict):
        return None
    stored_filename = image_asset.get("stored_filename")
    if stored_filename:
        path = conversation_store.asset_path(conversation_id, f"images/{stored_filename}")
        if path:
            return path.read_bytes()
    raw_path = image_asset.get("path")
    if raw_path:
        path = Path(raw_path)
        if path.is_file():
            return path.read_bytes()
    return None


@detect_router.get("/conversations")
async def list_conversations_api():
    return {"conversations": conversation_store.list_conversations()}


@detect_router.get("/conversations/{conversation_id}")
async def get_conversation_api(conversation_id: str):
    resolved_conversation_id = conversation_store.ensure_id(conversation_id)
    return conversation_store.snapshot(resolved_conversation_id)


@detect_router.delete("/conversations/{conversation_id}")
async def delete_conversation_api(conversation_id: str):
    deleted = conversation_store.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到该对话。")
    return {"deleted": True, "conversation_id": conversation_id}


@detect_router.get("/conversations/{conversation_id}/assets/{asset_path:path}")
async def get_conversation_asset_api(conversation_id: str, asset_path: str):
    path = conversation_store.asset_path(conversation_id, asset_path)
    if not path:
        raise HTTPException(status_code=404, detail="未找到该资源。")
    return FileResponse(path)


@detect_router.post("/followup", response_model=DetectionResponse)
async def followup_api(
    question: str = Form(..., description="Follow-up question about the latest detection result."),
    conversation_id: Optional[str] = Form(None, description="Conversation id with an existing detection result."),
    language: str = Form("zh", description="Output language: zh or en."),
    orchestrator: DetectionOrchestrator = Depends(get_detection_orchestrator),
):
    started_at = time.perf_counter()
    resolved_conversation_id = conversation_store.ensure_id(conversation_id)
    context = conversation_store.latest_context(resolved_conversation_id)
    latest_result = context.get("latest_result") or {}
    if not latest_result:
        raise HTTPException(status_code=400, detail="当前对话还没有可追问的检测结果。")

    latest_input = context.get("latest_input", {}) or {}
    history = context.get("conversation_history", [])
    rerun_required = should_rerun_detection(question)
    if rerun_required:
        image_content = load_stored_image_content(resolved_conversation_id, latest_input)
        mode = mode_from_value(latest_input.get("mode") or latest_result.get("mode"))
        result = orchestrator.detect(
            text=latest_input.get("text") or None,
            image_content=image_content,
            image_url=latest_input.get("image_url") or None,
            question=question,
            mode=mode,
            history=history,
            language=language,
        )
        result["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
        result["conversation_id"] = resolved_conversation_id
        next_input = {
            **latest_input,
            "mode": mode.value,
            "language": language,
            "question": result.get("question") or question,
            "has_image": bool(image_content) or bool(latest_input.get("image_url")),
        }
        result["conversation_history"] = conversation_store.append_turn(
            resolved_conversation_id,
            next_input,
            result,
        )
        result["followup"] = {"question": question, "rerun_detection": True}
        return DetectionResponse.model_validate(result)

    explanation = followup_service.explain(
        question=question,
        latest_input=latest_input,
        latest_result=latest_result,
        history=history,
        language=language,
    )
    result = dict(latest_result)
    result["conversation_id"] = resolved_conversation_id
    result["conversation_history"] = conversation_store.append_followup_turn(
        resolved_conversation_id,
        question,
        result,
        language,
        assistant_content=explanation.get("content"),
        update_latest_result=False,
    )
    return DetectionResponse.model_validate(result)


@detect_router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Run unified misinformation detection",
    description=(
        "Unified endpoint for text-only fact checking, image-only authenticity analysis, "
        "and multimodal image-text consistency detection. `mode=auto` selects the most "
        "appropriate pipeline from the provided inputs."
    ),
    responses={
        400: {
            "description": "Invalid input combination.",
            "content": {
                "application/json": {
                    "example": {"detail": "multimodal 模式需要提供 text 或 question 作为图文对齐文本。"}
                }
            },
        }
    },
    openapi_extra={
        "requestBody": {
            "content": {
                "multipart/form-data": {
                    "examples": {
                        "text_only": {
                            "summary": "Text-only fact checking",
                            "value": {
                                "mode": "text_only",
                                "language": "zh",
                                "text": "某新闻声称某地发生了一起重大公共事件。",
                            },
                        },
                        "image_only": {
                            "summary": "Image-only authenticity detection",
                            "value": {
                                "mode": "image_only",
                                "language": "zh",
                                "question": "这张图片是真的假的？",
                                "image_url": "https://example.com/news-image.jpg",
                            },
                        },
                        "multimodal": {
                            "summary": "Image-text consistency detection",
                            "value": {
                                "mode": "multimodal",
                                "language": "en",
                                "text": "The article claims the photo shows the event described in the news.",
                                "image_url": "https://example.com/event-photo.jpg",
                                "question": "Does the image support the news text?",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def detect_api(
    text: Optional[str] = Form(
        None,
        description="News text, claim, title, or article body to analyze.",
    ),
    image: Optional[UploadFile] = File(
        None,
        description="Uploaded image file. Used for image_only and multimodal detection.",
    ),
    image_url: Optional[str] = Form(
        None,
        description="Remote image URL. Use this instead of image upload when the image is already online.",
    ),
    question: Optional[str] = Form(
        None,
        description="Optional short user question that focuses the detection task.",
    ),
    mode: DetectionMode = Form(
        DetectionMode.AUTO,
        description="Detection mode: auto, text_only, image_only, or multimodal.",
    ),
    language: str = Form(
        "zh",
        description="Output and prompt language. Supported values: zh for Chinese, en for English. Default: zh.",
    ),
    history: str = Form(
        "[]",
        description="Optional JSON array of previous chat messages for contextual text fact-checking.",
    ),
    conversation_id: Optional[str] = Form(
        None,
        description="Conversation id used by the backend to maintain per-conversation history.",
    ),
    orchestrator: DetectionOrchestrator = Depends(get_detection_orchestrator),
):
    """Run text, image, or multimodal misinformation detection."""
    started_at = time.perf_counter()
    if not (text or question or image or image_url):
        raise HTTPException(status_code=400, detail="text、question、image 或 image_url 至少需要提供一个。")
    if mode in {DetectionMode.IMAGE_ONLY, DetectionMode.MULTIMODAL} and not (image or image_url):
        raise HTTPException(status_code=400, detail=f"{mode.value} 模式需要提供 image 或 image_url。")
    if mode == DetectionMode.TEXT_ONLY and not (text or question):
        raise HTTPException(status_code=400, detail="text_only 模式需要提供 text 或 question。")
    if mode == DetectionMode.MULTIMODAL and not (text or question):
        raise HTTPException(status_code=400, detail="multimodal 模式需要提供 text 或 question 作为图文对齐文本。")

    image_content = await image.read() if image else None
    resolved_conversation_id = conversation_store.ensure_id(conversation_id)
    image_asset = conversation_store.save_uploaded_image(
        resolved_conversation_id,
        image_content,
        filename=image.filename if image else None,
        content_type=image.content_type if image else None,
    )
    request_history = parse_history(history)
    backend_history = conversation_store.get(resolved_conversation_id)
    combined_history = [*backend_history, *request_history]
    result = orchestrator.detect(
        text=text,
        image_content=image_content,
        image_url=image_url,
        question=question,
        mode=mode,
        history=combined_history,
        language=language,
    )
    result["runtime_seconds"] = round(time.perf_counter() - started_at, 3)
    result["conversation_id"] = resolved_conversation_id
    result["conversation_history"] = conversation_store.append_turn(
        resolved_conversation_id,
        {
            "mode": str(mode.value if isinstance(mode, DetectionMode) else mode),
            "language": language,
            "text": text or "",
            "question": result.get("question") or question or "",
            "image_url": image_url or "",
            "has_image": bool(image_content),
            "image_asset": image_asset,
        },
        result,
    )
    return DetectionResponse.model_validate(result)

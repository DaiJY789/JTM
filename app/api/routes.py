from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.adapters.detection_request import adapt_detection_request
from app.api.dependencies import get_detection_orchestrator
from app.schemas.detection import DetectionResponse
from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore
from app.services.detection_application import DetectionApplicationService
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.followup_application import (
    FollowupApplicationService,
    FollowupContextError,
)
from app.services.followup_analysis import FollowupAnalysisService
from app.services.input_builder import DetectionInputBuilder, DetectionInputError
from app.services.session_service import SessionService
from app.services.webpage_extraction import WebpageExtractionService

detect_router = APIRouter()
conversation_store = ConversationHistoryStore()
followup_service = FollowupAnalysisService()
webpage_extraction_service = WebpageExtractionService()
artifact_store = ArtifactStore(conversation_store.log_dir)
session_service = SessionService(conversation_store, artifact_store)
followup_application = FollowupApplicationService(session_service, followup_service)


# Conversation routes

@detect_router.get("/conversations")
async def list_conversations():
    return {"conversations": session_service.list_conversations()}


@detect_router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    return session_service.snapshot(conversation_id)


@detect_router.delete("/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str):
    deleted = session_service.delete(conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="未找到该对话。")
    return {"deleted": True, "conversation_id": conversation_id}


@detect_router.get("/conversations/{conversation_id}/assets/{asset_path:path}")
async def get_conversation_asset(conversation_id: str, asset_path: str):
    path = session_service.resolve_asset(conversation_id, asset_path)
    if not path:
        raise HTTPException(status_code=404, detail="未找到该资源。")
    return FileResponse(path)


# Follow-up route

@detect_router.post("/followup", response_model=DetectionResponse)
async def followup(
    question: str = Form(..., description="Follow-up question about the latest detection result."),
    conversation_id: Optional[str] = Form(None, description="Conversation id with an existing detection result."),
    language: str = Form("zh", description="Output language: zh or en."),
    orchestrator: DetectionOrchestrator = Depends(get_detection_orchestrator),
):
    try:
        result = followup_application.followup(
            question=question,
            conversation_id=conversation_id,
            language=language,
            orchestrator=orchestrator,
        )
    except FollowupContextError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DetectionResponse.model_validate(result)


# Detection route

@detect_router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Run unified misinformation detection",
    description=(
        "Unified endpoint for textual fact checking, visual authenticity analysis, "
        "cross-modal consistency checking, and multimodal comprehensive report generation. The backend automatically "
        "selects the appropriate pipeline from the provided inputs."
    ),
    responses={
        400: {
            "description": "Invalid input combination.",
            "content": {
                "application/json": {
                    "example": {"detail": "text、question、image、image_url 或 webpage_url 至少需要提供一个。"}
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
                                "language": "zh",
                                "text": "某新闻声称某地发生了一起重大公共事件。",
                            },
                        },
                        "image_only": {
                            "summary": "Image-only authenticity detection",
                            "value": {
                                "language": "zh",
                                "question": "这张图片是真的假的？",
                                "image_url": "https://example.com/news-image.jpg",
                            },
                        },
                        "multimodal": {
                            "summary": "Cross-modal consistency and comprehensive fusion report",
                            "value": {
                                "language": "en",
                                "text": "The article claims the photo shows the event described in the news.",
                                "image_url": "https://example.com/event-photo.jpg",
                                "question": "Does the image support the news text?",
                            },
                        },
                        "webpage": {
                            "summary": "Webpage multimodal analysis",
                            "value": {
                                "language": "zh",
                                "webpage_url": "https://example.com/news-page",
                                "question": "分析这个网页中图文信息是否真实可信。",
                            },
                        },
                    }
                }
            }
        }
    },
)
async def detect(
    text: Optional[str] = Form(
        None,
        description="News text, claim, title, or article body to analyze.",
    ),
    image: Optional[UploadFile] = File(
        None,
        description="Uploaded image file. The backend selects visual or multimodal analysis automatically.",
    ),
    image_url: Optional[str] = Form(
        None,
        description="Remote image URL. Use this instead of image upload when the image is already online.",
    ),
    webpage_url: Optional[str] = Form(
        None,
        description="Webpage URL. The backend extracts page text and representative images for multimodal analysis.",
    ),
    question: Optional[str] = Form(
        None,
        description="Optional short user question that focuses the detection task.",
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
    image_content = await image.read() if image else None
    request = adapt_detection_request(
        text=text,
        image_content=image_content,
        image_filename=image.filename if image else None,
        image_content_type=image.content_type if image else None,
        image_url=image_url,
        webpage_url=webpage_url,
        question=question,
        language=language,
        history=history,
        conversation_id=conversation_id,
    )
    builder = DetectionInputBuilder(webpage_extraction_service, artifact_store)
    application = DetectionApplicationService(orchestrator, builder, session_service)
    try:
        result = application.detect(request)
    except DetectionInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return DetectionResponse.model_validate(result)

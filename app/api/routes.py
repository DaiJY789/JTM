import hashlib
import json
from time import perf_counter
from typing import Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from app.api.adapters.detection_request import adapt_detection_request
from app.api.dependencies import get_detection_orchestrator
from app.core.logging import log_io_event, new_request_id
from app.schemas.detection import DetectionResponse, DetectionTaskType
from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore
from app.services.conversation_image_preview import (
    ConversationImagePreviewError,
    ConversationImagePreviewService,
)
from app.services.detection_application import DetectionApplicationService
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.followup_application import (
    FollowupApplicationService,
    FollowupContextError,
)
from app.services.followup_analysis import FollowupAnalysisService
from app.services.input_builder import DetectionInputBuilder, DetectionInputError
from app.services.remote_image_downloader import RemoteImageDownloader
from app.services.session_service import SessionService
from app.services.webpage_extraction import WebpageExtractionError, WebpageExtractionService

detect_router = APIRouter()
conversation_store = ConversationHistoryStore()
followup_service = FollowupAnalysisService()
webpage_extraction_service = WebpageExtractionService()
remote_image_downloader = RemoteImageDownloader()
artifact_store = ArtifactStore(conversation_store.log_dir)
session_service = SessionService(conversation_store, artifact_store)
followup_application = FollowupApplicationService(session_service, followup_service)


@detect_router.post(
    "/webpages/extract",
    summary="Extract news text and images from a webpage",
)
async def extract_webpage(
    webpage_url: str = Form(..., description="Public webpage URL to parse."),
):
    """Parse a webpage for frontend preview without starting a detection run."""
    request_id = new_request_id()
    started_at = perf_counter()
    endpoint = "/webpages/extract"
    log_io_event("request_input", request_id, endpoint, {"webpage_url": webpage_url})
    try:
        result = webpage_extraction_service.extract(webpage_url)
    except WebpageExtractionError as exc:
        log_io_event("request_error", request_id, endpoint, {
            "duration_seconds": round(perf_counter() - started_at, 4),
            "status_code": 400,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "response": {"detail": str(exc)},
        })
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    payload = {
        "url": result.url,
        "final_url": result.final_url,
        "title": result.title,
        "text": result.text,
        "image_url": result.image_url,
        "images": result.images,
        "metadata": result.metadata,
    }
    log_io_event("request_output", request_id, endpoint, {
        "duration_seconds": round(perf_counter() - started_at, 4),
        "status_code": 200,
        "response": payload,
    })
    return payload


# Conversation routes

@detect_router.get("/conversations")
async def list_conversations():
    return {"conversations": session_service.list_conversations()}


@detect_router.get("/conversations/{conversation_id}")
async def get_conversation(conversation_id: str):
    return session_service.snapshot(conversation_id)


@detect_router.get("/conversations/{conversation_id}/images/{image_index}")
def get_conversation_image(conversation_id: str, image_index: int):
    try:
        preview = ConversationImagePreviewService(session_service, artifact_store).get(
            conversation_id,
            image_index,
        )
    except ConversationImagePreviewError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        preview.path,
        media_type=preview.media_type,
        headers={"Cache-Control": "private, max-age=86400"},
    )


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

@detect_router.post(
    "/conversations/{conversation_id}/followup",
    response_model=DetectionResponse,
    summary="Ask a follow-up question about a conversation",
)
async def conversation_followup(
    conversation_id: str,
    question: str = Form(..., description="Follow-up question about the latest detection result."),
    language: str = Form("zh", description="Output language: zh or en."),
    orchestrator: DetectionOrchestrator = Depends(get_detection_orchestrator),
):
    return _run_followup_request(
        endpoint=f"/conversations/{conversation_id}/followup",
        question=question,
        conversation_id=conversation_id,
        language=language,
        orchestrator=orchestrator,
    )


@detect_router.post("/followup", response_model=DetectionResponse)
async def followup(
    question: str = Form(..., description="Follow-up question about the latest detection result."),
    conversation_id: Optional[str] = Form(None, description="Conversation id with an existing detection result."),
    language: str = Form("zh", description="Output language: zh or en."),
    orchestrator: DetectionOrchestrator = Depends(get_detection_orchestrator),
):
    """Legacy follow-up endpoint kept for backward compatibility."""
    return _run_followup_request(
        endpoint="/followup",
        question=question,
        conversation_id=conversation_id,
        language=language,
        orchestrator=orchestrator,
    )


def _run_followup_request(
    *,
    endpoint: str,
    question: str,
    conversation_id: str | None,
    language: str,
    orchestrator: DetectionOrchestrator,
) -> DetectionResponse:
    request_id = new_request_id()
    started_at = perf_counter()
    log_io_event(
        "request_input", request_id, endpoint,
        {"question": question, "conversation_id": conversation_id, "language": language},
    )
    try:
        result = followup_application.followup(
            question=question,
            conversation_id=conversation_id,
            language=language,
            orchestrator=orchestrator,
        )
    except FollowupContextError as exc:
        log_io_event("request_error", request_id, endpoint, {
            "duration_seconds": round(perf_counter() - started_at, 4),
            "status_code": 400,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "response": {"detail": str(exc)},
        })
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log_io_event("request_error", request_id, endpoint, {
            "duration_seconds": round(perf_counter() - started_at, 4),
            "status_code": 500,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        raise
    try:
        response = DetectionResponse.model_validate(result)
    except Exception as exc:
        log_io_event("request_error", request_id, endpoint, {
            "duration_seconds": round(perf_counter() - started_at, 4),
            "status_code": 500,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        raise
    log_io_event("request_output", request_id, endpoint, {
        "duration_seconds": round(perf_counter() - started_at, 4),
        "status_code": 200,
        "response": response.model_dump(mode="json"),
    })
    return response


# Detection route

@detect_router.post(
    "/detect",
    response_model=DetectionResponse,
    summary="Run unified misinformation detection",
    description=(
        "Unified endpoint for single-agent multimodal misinformation detection and "
        "deepfake image detection. Routing follows the explicit task field."
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
                        "image_only": {
                            "summary": "Image-only authenticity detection",
                            "value": {
                                "task": "deepfake_detection",
                                "language": "zh",
                                "question": "这张图片是真的假的？",
                                "image_url": "https://example.com/news-image.jpg",
                            },
                        },
                        "multimodal": {
                            "summary": "Single-agent multimodal misinformation detection",
                            "value": {
                                "task": "multimodal_fact_checking",
                                "language": "en",
                                "text": "The article claims the photo shows the event described in the news.",
                                "image_url": "https://example.com/event-photo.jpg",
                                "question": "Does the image support the news text?",
                            },
                        },
                        "webpage": {
                            "summary": "Webpage multimodal analysis",
                            "value": {
                                "task": "multimodal_fact_checking",
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
    task: DetectionTaskType = Form(
        DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        description=(
            "Detection task: multimodal_fact_checking, deepfake_detection, or aigc_detection. "
            "aigc_detection is reserved and not implemented yet."
        ),
    ),
    text: Optional[str] = Form(
        None,
        description="News text, claim, title, or article body. Required for multimodal_fact_checking.",
    ),
    image: Optional[UploadFile] = File(
        None,
        description="Legacy single image upload. Multimodal requests may alternatively use repeated images fields.",
    ),
    images: Optional[list[UploadFile]] = File(
        None,
        description="Multiple ordered images for multimodal fact checking. Deepfake detection accepts one image only.",
    ),
    image_url: Optional[str] = Form(
        None,
        description="Remote image URL. Use this instead of image upload when the image is already online.",
    ),
    image_urls: Optional[list[str]] = Form(
        None,
        description="Multiple ordered remote image URLs for multimodal fact checking.",
    ),
    webpage_url: Optional[str] = Form(
        None,
        description="Webpage URL. The backend extracts page text and representative images for multimodal analysis.",
    ),
    webpage_image_urls: Optional[str] = Form(None, description="JSON array of selected webpage image URLs. An empty array excludes all webpage images."),
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
    """Legacy task-routed detection endpoint."""
    return await _run_detection_request(
        endpoint="/detect", task=task, text=text, image=image, images=images,
        image_url=image_url, image_urls=image_urls, webpage_url=webpage_url,
        webpage_image_urls=webpage_image_urls,
        question=question, language=language, history=history,
        conversation_id=conversation_id, orchestrator=orchestrator,
    )


@detect_router.post(
    "/detect/mmd",
    response_model=DetectionResponse,
    summary="Detect multimodal misinformation",
    description="Fact-check news text with optional images (up to four) or a webpage. Without images, the endpoint performs text-only fact checking.",
)
async def detect_mmd(
    text: Optional[str] = Form(None, description="News text, claim, title, or article body."),
    image: Optional[UploadFile] = File(None, description="Single-image compatibility field."),
    images: Optional[list[UploadFile]] = File(None, description="Ordered images; up to four total."),
    image_url: Optional[str] = Form(None, description="Single remote image URL."),
    image_urls: Optional[list[str]] = Form(None, description="Ordered remote image URLs; up to four total."),
    webpage_url: Optional[str] = Form(None, description="Webpage URL for text and image extraction."),
    webpage_image_urls: Optional[str] = Form(None, description="JSON array of selected webpage image URLs. An empty array excludes all webpage images."),
    question: Optional[str] = Form(None),
    language: str = Form("zh"),
    history: str = Form("[]"),
    conversation_id: Optional[str] = Form(None),
    orchestrator: DetectionOrchestrator = Depends(get_detection_orchestrator),
):
    return await _run_detection_request(
        endpoint="/detect/mmd", task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        text=text, image=image, images=images, image_url=image_url,
        image_urls=image_urls, webpage_url=webpage_url, question=question,
        webpage_image_urls=webpage_image_urls,
        language=language, history=history, conversation_id=conversation_id,
        orchestrator=orchestrator,
    )


@detect_router.post(
    "/detect/deepfake",
    response_model=DetectionResponse,
    summary="Detect deepfake images",
    description="Inspect one uploaded image, one image URL, or an image extracted from a webpage.",
)
async def detect_deepfake(
    image: Optional[UploadFile] = File(None, description="One image upload."),
    image_url: Optional[str] = Form(None, description="One remote image URL."),
    webpage_url: Optional[str] = Form(None, description="Webpage URL for image extraction."),
    webpage_image_urls: Optional[str] = Form(None, description="JSON array of selected webpage image URLs. An empty array excludes all webpage images."),
    question: Optional[str] = Form(None),
    language: str = Form("zh"),
    history: str = Form("[]"),
    conversation_id: Optional[str] = Form(None),
    orchestrator: DetectionOrchestrator = Depends(get_detection_orchestrator),
):
    return await _run_detection_request(
        endpoint="/detect/deepfake", task=DetectionTaskType.DEEPFAKE_DETECTION,
        text=None, image=image, images=None, image_url=image_url,
        image_urls=None, webpage_url=webpage_url, question=question,
        webpage_image_urls=webpage_image_urls,
        language=language, history=history, conversation_id=conversation_id,
        orchestrator=orchestrator,
    )


async def _run_detection_request(
    *,
    endpoint: str,
    task: DetectionTaskType,
    text: str | None,
    image: UploadFile | None,
    images: list[UploadFile] | None,
    image_url: str | None,
    image_urls: list[str] | None,
    webpage_url: str | None,
    webpage_image_urls: str | None,
    question: str | None,
    language: str,
    history: str,
    conversation_id: str | None,
    orchestrator: DetectionOrchestrator,
) -> DetectionResponse:
    request_id = new_request_id()
    started_at = perf_counter()
    uploaded_images = ([image] if image else []) + list(images or [])
    image_contents = [await item.read() for item in uploaded_images]
    image_content = image_contents[0] if image_contents else None
    log_io_event("request_input", request_id, endpoint, {
        "task": task.value,
        "text": text,
        "question": question,
        "language": language,
        "conversation_id": conversation_id,
        "history": history,
        "image_url": image_url,
        "image_urls": image_urls or [],
        "webpage_url": webpage_url,
        "webpage_image_urls": webpage_image_urls,
        "images": [
            {
                "filename": upload.filename,
                "content_type": upload.content_type,
                "size_bytes": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for upload, content in zip(uploaded_images, image_contents)
        ],
    })
    try:
        selected_webpage_images = None
        if webpage_image_urls is not None:
            try:
                selected_webpage_images = json.loads(webpage_image_urls)
            except json.JSONDecodeError as exc:
                raise DetectionInputError("网页图片清单必须是 JSON 链接数组。") from exc
            if not isinstance(selected_webpage_images, list) or not all(isinstance(url, str) for url in selected_webpage_images):
                raise DetectionInputError("网页图片清单必须是 JSON 链接数组。")
        request = adapt_detection_request(
            task=task,
            text=text,
            image_content=image_content,
            image_filename=uploaded_images[0].filename if uploaded_images else None,
            image_content_type=uploaded_images[0].content_type if uploaded_images else None,
            image_url=image_url,
            webpage_url=webpage_url,
            question=question,
            language=language,
            history=history,
            conversation_id=conversation_id,
            image_contents=image_contents,
            image_filenames=[item.filename for item in uploaded_images],
            image_content_types=[item.content_type for item in uploaded_images],
            image_urls=image_urls,
            webpage_image_urls=selected_webpage_images,
        )
        builder = DetectionInputBuilder(
            webpage_extraction_service,
            artifact_store,
            remote_image_downloader=remote_image_downloader,
        )
        application = DetectionApplicationService(orchestrator, builder, session_service)
        result = application.detect(request)
    except DetectionInputError as exc:
        log_io_event("request_error", request_id, endpoint, {
            "duration_seconds": round(perf_counter() - started_at, 4),
            "status_code": 400,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "response": {"detail": str(exc)},
        })
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log_io_event("request_error", request_id, endpoint, {
            "duration_seconds": round(perf_counter() - started_at, 4),
            "status_code": 500,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        raise
    try:
        response = DetectionResponse.model_validate(result)
    except Exception as exc:
        log_io_event("request_error", request_id, endpoint, {
            "duration_seconds": round(perf_counter() - started_at, 4),
            "status_code": 500,
            "error_type": type(exc).__name__,
            "message": str(exc),
        })
        raise
    log_io_event("request_output", request_id, endpoint, {
        "duration_seconds": round(perf_counter() - started_at, 4),
        "status_code": 200,
        "response": response.model_dump(mode="json"),
    })
    return response

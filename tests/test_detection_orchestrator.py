import pytest

from app.schemas.artifacts import ArtifactRole, ArtifactType, DetectionTask, EvidenceBundle, InputArtifact
from app.schemas.detection import DetectionMode, DetectionTaskType
from app.services.detection_orchestrator import DetectionOrchestrator


class StubMFCService:
    def __init__(self):
        self.calls = []

    def detect(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "mode": "multimodal",
            "label": "True",
            "type": "multimodal_misinformation",
            "summary": "图文可信。",
            "confidence": 0.8,
            "mfc_detection": {
                "label": "True",
                "confidence": 0.8,
                "user_answer": {"judgment": "图文可信。", "analysis": "图文可信。", "notes": []},
            },
            "evidence": [],
            "claims": [],
            "raw": {"workflow": {"routing": "single_agent", "steps": ["MFC_Agent"]}},
        }


class StubIFDService:
    def __init__(self):
        self.calls = []

    def detect(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "mode": "image_only",
            "label": "真实图像",
            "type": "deepfake_detection",
            "summary": "未发现明确伪造。",
            "confidence": 0.8,
            "image_verdict": {"label": "真实图像", "analysis": "未发现明确伪造。"},
            "evidence": [],
            "claims": [],
            "raw": {},
        }


def make_orchestrator():
    return DetectionOrchestrator(
        mfc_detection_service=StubMFCService(),
        ifd_detection_service=StubIFDService(),
    )


def test_multimodal_task_uses_only_mfc_agent():
    orchestrator = make_orchestrator()

    result = orchestrator.detect(
        text="新闻文本",
        image_content=b"image",
        task_type=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
    )

    assert result["question"] == "请判断这段文字和所提供图片是否真实可信，并分析图文是否一致。"
    assert result["raw"]["workflow"]["steps"] == ["MFC_Agent"]
    assert len(orchestrator.mfc_detection_service.calls) == 1
    assert orchestrator.ifd_detection_service.calls == []


def test_deepfake_task_uses_only_deepfake_agent():
    orchestrator = make_orchestrator()

    result = orchestrator.detect(
        image_content=b"image",
        task_type=DetectionTaskType.DEEPFAKE_DETECTION,
    )

    assert result["question"] == "请判断这张图片是真实图像、伪造图像还是合成图像。"
    assert result["raw"]["workflow"]["steps"] == ["IFD_Agent"]
    assert len(orchestrator.ifd_detection_service.calls) == 1
    assert orchestrator.mfc_detection_service.calls == []


def test_multimodal_task_accepts_text_without_image():
    orchestrator = make_orchestrator()

    result = orchestrator.detect(
        text="新闻文本",
        task_type=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
    )

    assert result["question"] == "请核查这段文字中的核心事实主张是否真实可信。"
    call = orchestrator.mfc_detection_service.calls[0]
    assert call["image_contents"] == []
    assert call["image_urls"] == []


def test_multimodal_task_still_requires_text():
    with pytest.raises(ValueError, match="需要提供待核查的文本"):
        make_orchestrator().detect(image_content=b"image")


def test_deepfake_task_requires_image():
    with pytest.raises(ValueError, match="需要提供图片"):
        make_orchestrator().detect(task_type=DetectionTaskType.DEEPFAKE_DETECTION)


def test_detect_task_uses_normalized_artifacts_and_builds_report():
    orchestrator = make_orchestrator()
    task = DetectionTask(
        task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        language="zh",
        question="判断图文是否一致",
        conversation_id="conv-report",
        evidence_bundle=EvidenceBundle(
            text_artifacts=[
                InputArtifact(type=ArtifactType.TEXT, role=ArtifactRole.USER_INPUT, content="新闻文本")
            ],
            image_artifacts=[
                InputArtifact(
                    type=ArtifactType.IMAGE,
                    role=ArtifactRole.REMOTE_IMAGE,
                    source_url="https://example.com/image.jpg",
                )
            ],
            evidence_items=[{"type": "manual_context", "source": "test", "content": "context"}],
        ),
    )

    result = orchestrator.detect_task(task)

    assert result["question"] == "判断图文是否一致"
    assert orchestrator.mfc_detection_service.calls[0]["image_url"] == "https://example.com/image.jpg"
    assert result["raw"]["input_artifacts"]["images"][0]["source_url"] == "https://example.com/image.jpg"
    assert result["raw"]["report"]["conversation_id"] == "conv-report"


def test_detect_task_passes_all_mfc_images_to_single_agent():
    orchestrator = make_orchestrator()
    task = DetectionTask(
        task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        evidence_bundle=EvidenceBundle(
            text_artifacts=[InputArtifact(type=ArtifactType.TEXT, content="多图新闻文本")],
            image_artifacts=[
                InputArtifact(type=ArtifactType.IMAGE, source_url="https://example.com/2.jpg"),
                InputArtifact(type=ArtifactType.IMAGE, source_url="https://example.com/3.jpg"),
            ],
        ),
        image_content=b"image-1",
        image_contents=[b"image-1"],
    )

    orchestrator.detect_task(task)

    call = orchestrator.mfc_detection_service.calls[0]
    assert call["image_contents"] == [b"image-1"]
    assert call["image_urls"] == [
        "https://example.com/2.jpg",
        "https://example.com/3.jpg",
    ]


def test_detect_task_passes_persisted_path_to_deepfake_service(tmp_path):
    image_path = tmp_path / "conversations" / "conv" / "images" / "input.png"
    image_path.parent.mkdir(parents=True)
    image_path.write_bytes(b"image")
    orchestrator = make_orchestrator()
    task = DetectionTask(
        task=DetectionTaskType.DEEPFAKE_DETECTION,
        evidence_bundle=EvidenceBundle(
            image_artifacts=[
                InputArtifact(
                    type=ArtifactType.IMAGE,
                    role=ArtifactRole.UPLOADED_IMAGE,
                    local_path=str(image_path),
                )
            ],
        ),
        image_content=b"image",
        image_contents=[b"image"],
    )

    orchestrator.detect_task(task)

    call = orchestrator.ifd_detection_service.calls[0]
    assert call["image_path"] == str(image_path)
    assert call["image_url"] is None


def test_resolve_mode_is_shape_based_only():
    orchestrator = make_orchestrator()
    assert orchestrator.resolve_mode("新闻文本", True) == DetectionMode.MULTIMODAL
    assert orchestrator.resolve_mode("", True) == DetectionMode.IMAGE_ONLY
    assert orchestrator.resolve_mode("新闻文本", False) == DetectionMode.TEXT_ONLY

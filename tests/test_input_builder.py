from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from app.services.artifact_store import ArtifactStore
from app.schemas.detection import DetectionMode, DetectionTaskType
from app.services.input_builder import DetectionInputBuilder, DetectionInputError
from app.services.remote_image_downloader import DownloadedRemoteImage
from app.services.webpage_extraction import WebpageExtractionResult


class StubWebpageExtractor:
    def extract(self, url):
        return WebpageExtractionResult(
            url=url,
            final_url="https://example.com/news",
            title="网页标题",
            text="正文第一段。",
            image_url="https://example.com/a.jpg",
            images=["https://example.com/a.jpg", "https://example.com/b.jpg"],
            metadata={"image_count": 2},
        )


def valid_png_bytes(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (2, 2), color=color).save(output, format="PNG")
    return output.getvalue()


class StubRemoteImageDownloader:
    def download(self, url):
        return DownloadedRemoteImage(
            content=valid_png_bytes(),
            suffix=".png",
            media_type="image/png",
            final_url=url,
        )


def test_input_builder_normalizes_webpage_into_task(tmp_path):
    builder = DetectionInputBuilder(
        StubWebpageExtractor(),
        ArtifactStore(tmp_path),
        remote_image_downloader=StubRemoteImageDownloader(),
    )

    task = builder.build(
        task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        text=None,
        image_content=None,
        image_filename=None,
        image_content_type=None,
        image_url=None,
        webpage_url="https://example.com/news",
        question="判断网页图文是否可信",
        language="zh",
        history=[],
        conversation_id="conv",
    )

    assert task.text == "网页标题\n\n正文第一段。"
    assert task.task == DetectionTaskType.MULTIMODAL_FACT_CHECKING
    assert task.image_url == "https://example.com/a.jpg"
    assert task.image_urls == []
    assert len(task.image_contents) == 2
    assert len(task.evidence_bundle.image_artifacts) == 2
    assert all(item.local_path for item in task.evidence_bundle.image_artifacts)
    assert all(Path(item.local_path).is_file() for item in task.evidence_bundle.image_artifacts)
    assert task.evidence_bundle.evidence_items[0]["type"] == "webpage_source"
    assert task.user_input["webpage"]["image_count"] == 2


def test_input_builder_persists_uploaded_image(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    task = builder.build(
        task=DetectionTaskType.DEEPFAKE_DETECTION,
        text=None,
        image_content=valid_png_bytes(),
        image_filename="input.png",
        image_content_type="image/png",
        image_url=None,
        webpage_url=None,
        question="这张图是真的吗？",
        language="zh",
        history=[],
        conversation_id="conv",
    )

    assert task.has_image
    assert task.mode == DetectionMode.IMAGE_ONLY
    assert task.user_input["task"] == "deepfake_detection"
    assert task.user_input["image_asset"]["stored_filename"].endswith(".png")
    assert task.evidence_bundle.image_artifacts[0].local_path


def test_input_builder_reads_detection_bytes_from_persisted_copy(tmp_path):
    replacement = valid_png_bytes("black")

    class RewritingArtifactStore(ArtifactStore):
        def save_image(self, *args, **kwargs):
            artifact = super().save_image(*args, **kwargs)
            Path(artifact.local_path).write_bytes(replacement)
            return artifact

    original = valid_png_bytes("white")
    builder = DetectionInputBuilder(
        StubWebpageExtractor(),
        RewritingArtifactStore(tmp_path),
    )

    task = builder.build(
        task=DetectionTaskType.DEEPFAKE_DETECTION,
        text=None,
        image_content=original,
        image_filename="input.png",
        image_content_type="image/png",
        image_url=None,
        webpage_url=None,
        question=None,
        language="zh",
        history=[],
        conversation_id="persisted-source",
    )

    assert task.image_contents == [replacement]
    assert task.image_contents[0] != original


def test_input_builder_preserves_order_and_limits_multiple_mmd_images(tmp_path):
    builder = DetectionInputBuilder(
        StubWebpageExtractor(),
        ArtifactStore(tmp_path),
        max_mmd_images=2,
    )
    contents = [valid_png_bytes(), valid_png_bytes(), valid_png_bytes()]

    task = builder.build(
        task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        text="多图新闻文本",
        image_content=contents[0],
        image_filename="one.png",
        image_content_type="image/png",
        image_url=None,
        webpage_url=None,
        question=None,
        language="zh",
        history=[],
        conversation_id="multi-image",
        image_contents=contents,
        image_filenames=["one.png", "two.png", "three.png"],
        image_content_types=["image/png"] * 3,
    )

    assert len(task.image_contents) == 2
    assert [item.filename for item in task.evidence_bundle.image_artifacts] == [
        "one.png",
        "two.png",
    ]
    assert task.user_input["image_count"] == 2
    assert task.user_input["images_truncated"] is True
    assert len(task.user_input["image_assets"]) == 2
    assert task.user_input["image_urls"] == []


def test_input_builder_exposes_all_selected_remote_image_urls(tmp_path):
    builder = DetectionInputBuilder(
        StubWebpageExtractor(),
        ArtifactStore(tmp_path),
        max_mmd_images=4,
        remote_image_downloader=StubRemoteImageDownloader(),
    )

    task = builder.build(
        task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        text="多图新闻文本",
        image_content=None,
        image_filename=None,
        image_content_type=None,
        image_url=None,
        webpage_url=None,
        question=None,
        language="zh",
        history=[],
        conversation_id="multi-url",
        image_urls=[
            "https://example.com/one.jpg",
            "https://example.com/two.jpg",
        ],
    )

    assert task.image_urls == []
    assert task.user_input["image_urls"] == [
        "https://example.com/one.jpg",
        "https://example.com/two.jpg",
    ]
    assert len(task.image_contents) == 2
    assert len(task.user_input["image_assets"]) == 2
    assert all(Path(item.local_path).is_file() for item in task.evidence_bundle.image_artifacts)
    assert task.user_input["image_count"] == 2


def test_input_builder_rejects_multiple_deepfake_images(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    with pytest.raises(DetectionInputError, match="只支持一张图片"):
        builder.build(
            task=DetectionTaskType.DEEPFAKE_DETECTION,
            text=None,
            image_content=valid_png_bytes(),
            image_filename="one.png",
            image_content_type="image/png",
            image_url=None,
            webpage_url=None,
            question=None,
            language="zh",
            history=[],
            conversation_id="deepfake-multiple",
            image_contents=[valid_png_bytes(), valid_png_bytes()],
            image_filenames=["one.png", "two.png"],
            image_content_types=["image/png", "image/png"],
        )


def test_input_builder_rejects_empty_input(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    try:
        builder.build(
            task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
            text=None,
            image_content=None,
            image_filename=None,
            image_content_type=None,
            image_url=None,
            webpage_url=None,
            question=None,
            language="zh",
            history=[],
            conversation_id="conv",
        )
    except DetectionInputError as exc:
        assert "至少需要提供一个" in str(exc)
    else:
        raise AssertionError("DetectionInputError was not raised")


def test_input_builder_rejects_deepfake_task_without_image(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    try:
        builder.build(
            task=DetectionTaskType.DEEPFAKE_DETECTION,
            text="只有文本",
            image_content=None,
            image_filename=None,
            image_content_type=None,
            image_url=None,
            webpage_url=None,
            question=None,
            language="zh",
            history=[],
            conversation_id="conv",
        )
    except DetectionInputError as exc:
        assert "deepfake_detection" in str(exc)
    else:
        raise AssertionError("DetectionInputError was not raised")


def test_input_builder_rejects_fact_checking_without_text(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    try:
        builder.build(
            task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
            text=None,
            image_content=valid_png_bytes(),
            image_filename="input.png",
            image_content_type="image/png",
            image_url=None,
            webpage_url=None,
            question="这条消息是真的吗？",
            language="zh",
            history=[],
            conversation_id="conv",
        )
    except DetectionInputError as exc:
        assert "multimodal_fact_checking" in str(exc)
    else:
        raise AssertionError("DetectionInputError was not raised")


def test_input_builder_accepts_text_only_fact_checking(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    task = builder.build(
        task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        text="待核查声明",
        image_content=None,
        image_filename=None,
        image_content_type=None,
        image_url=None,
        webpage_url=None,
        question=None,
        language="zh",
        history=[],
        conversation_id="conv",
    )

    assert task.mode == DetectionMode.TEXT_ONLY
    assert task.text == "待核查声明"
    assert task.image_contents == []
    assert task.evidence_bundle.image_artifacts == []
    assert task.user_input["mode"] == "text_only"
    assert task.user_input["has_image"] is False


def test_input_builder_rejects_unimplemented_aigc_task(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    try:
        builder.build(
            task=DetectionTaskType.AIGC_DETECTION,
            text="content",
            image_content=None,
            image_filename=None,
            image_content_type=None,
            image_url=None,
            webpage_url=None,
            question=None,
            language="zh",
            history=[],
            conversation_id="conv",
        )
    except DetectionInputError as exc:
        assert "暂未实现" in str(exc)
    else:
        raise AssertionError("DetectionInputError was not raised")


def test_input_builder_rejects_invalid_uploaded_image_before_persisting(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    try:
        builder.build(
            task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
            text="待核查声明",
            image_content=b"not an image",
            image_filename="input.png",
            image_content_type="image/png",
            image_url=None,
            webpage_url=None,
            question=None,
            language="zh",
            history=[],
            conversation_id="conv",
        )
    except DetectionInputError as exc:
        assert "有效图片" in str(exc)
    else:
        raise AssertionError("DetectionInputError was not raised")

    assert not list(tmp_path.rglob("input.png"))
    assert not list(tmp_path.rglob("*.png"))


def test_input_builder_rejects_invalid_image_url(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    try:
        builder.build(
            task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
            text="待核查声明",
            image_content=None,
            image_filename=None,
            image_content_type=None,
            image_url="blob:http://localhost/not-a-real-image",
            webpage_url=None,
            question=None,
            language="zh",
            history=[],
            conversation_id="conv",
        )
    except DetectionInputError as exc:
        assert "image_url" in str(exc)
    else:
        raise AssertionError("DetectionInputError was not raised")

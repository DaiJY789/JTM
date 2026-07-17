from app.services.artifact_store import ArtifactStore
from app.services.input_builder import DetectionInputBuilder, DetectionInputError
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


def test_input_builder_normalizes_webpage_into_task(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    task = builder.build(
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
    assert task.image_url == "https://example.com/a.jpg"
    assert len(task.evidence_bundle.image_artifacts) == 2
    assert task.evidence_bundle.evidence_items[0]["type"] == "webpage_source"
    assert task.user_input["webpage"]["image_count"] == 2


def test_input_builder_persists_uploaded_image(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    task = builder.build(
        text=None,
        image_content=b"image bytes",
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
    assert task.user_input["image_asset"]["stored_filename"].endswith(".png")
    assert task.evidence_bundle.image_artifacts[0].local_path


def test_input_builder_rejects_empty_input(tmp_path):
    builder = DetectionInputBuilder(StubWebpageExtractor(), ArtifactStore(tmp_path))

    try:
        builder.build(
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

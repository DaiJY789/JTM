import json

from app.agents.tools.mfc import BoChaSearchTool, ImageOCREntityTool, MFCSearchQueryBuilder
from app.core.config import Settings


class FakeImageService:
    model_factory = None

    def build_mfc_image_content_part(self, image_content=None, image_url=None):
        if not image_content and not image_url:
            return None
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


def test_ocr_entity_tool_normalizes_visible_text_entities_and_boxes():
    payload = {
        "ocr_text": "北京暴雨\n2026-09-15",
        "text_blocks": [
            {"text": "北京暴雨", "bbox": [-10, 20, 1200, 300], "confidence": 1.2},
            {"text": "2026-09-15", "bbox": [100, 500, 400, 600]},
        ],
        "entities": [
            {"name": "北京", "type": "location", "evidence": "画面文字“北京暴雨”"},
            {"name": "北京", "type": "location", "evidence": "重复项"},
            {"name": "2026-09-15", "type": "date", "evidence": "画面日期"},
        ],
        "visual_objects": ["积水道路", "车辆"],
        "limitations": ["人物身份不可见"],
    }
    calls = []

    def evaluator(prompt, image_content, image_url, language):
        calls.append((prompt, image_content, image_url, language))
        return json.dumps(payload, ensure_ascii=False)

    tool = ImageOCREntityTool(FakeImageService(), Settings(), evaluator=evaluator)
    result = tool.run(image_content=b"image", language="zh", focus="检查地点和日期")

    assert result["status"] == "success"
    assert result["ocr_text"] == "北京暴雨\n2026-09-15"
    assert result["text_blocks"][0]["bbox"] == [0, 20, 1000, 300]
    assert result["text_blocks"][0]["confidence"] == 1.0
    assert [item["type"] for item in result["entities"]] == ["location", "date"]
    assert len(calls) == 1
    assert "检查地点和日期" in calls[0][0]
    assert tool.get_openai_tool_schema().function.name == "extract_image_ocr_entities"


def test_ocr_entity_tool_fails_closed_on_invalid_json():
    tool = ImageOCREntityTool(
        FakeImageService(),
        Settings(),
        evaluator=lambda *_: "not-json",
    )

    result = tool.run(image_url="https://example.com/image.jpg", language="zh")

    assert result["status"] == "error"
    assert result["ocr_text"] == ""
    assert result["entities"] == []
    assert result["timings"]["model_calls"] == 1
    assert "valid JSON" in result["error"]


def test_bocha_search_honors_the_remaining_agent_time_budget():
    calls = []
    def requester(*args, **kwargs):
        calls.append(kwargs)
        return StubResponse({"data": {"webPages": {"value": []}}})
    tool = BoChaSearchTool(Settings(bocha_api_key="test", search_timeout_seconds=15), requester=requester)
    assert tool.run("query", timeout_seconds=2)["status"] == "success"
    assert calls[0]["timeout"] == 2
    assert tool.run("query", timeout_seconds=0)["status"] == "error"
    assert len(calls) == 1


class StubResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_bocha_search_normalizes_and_deduplicates_results():
    captured = {}
    payload = {
        "data": {
            "webPages": {
                "value": [
                    {
                        "name": "官方通报",
                        "url": "https://example.gov/report",
                        "siteName": "示例政府",
                        "snippet": "官方发布了事件说明。",
                        "dateLastCrawled": "2026-09-15",
                    },
                    {"name": "重复", "url": "https://example.gov/report"},
                    {
                        "name": "媒体报道",
                        "url": "https://news.example/article",
                        "siteName": "示例新闻",
                        "snippet": "报道引用官方说明。",
                    },
                ]
            },
            "images": {
                "value": [
                    {
                        "name": "现场图片",
                        "contentUrl": "https://images.example/photo.jpg",
                        "thumbnailUrl": "https://images.example/thumb.jpg",
                    }
                ]
            },
        }
    }

    def requester(url, **kwargs):
        captured["url"] = url
        captured.update(kwargs)
        return StubResponse(payload)

    config = Settings(
        bocha_api_key="test-key",
        bocha_search_url="https://bocha.example/search",
        search_timeout_seconds=7,
    )
    tool = BoChaSearchTool(config, requester=requester)
    result = tool.run("  北京   暴雨 官方通报  ", count=5)

    assert result["status"] == "success"
    assert result["query"] == "北京 暴雨 官方通报"
    assert result["result_count"] == 2
    assert result["results"][0]["title"] == "官方通报"
    assert result["images"][0]["url"] == "https://images.example/photo.jpg"
    assert captured["url"] == "https://bocha.example/search"
    assert captured["json"] == {"query": "北京 暴雨 官方通报", "count": 5, "summary": True}
    assert captured["timeout"] == 7
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert tool.get_openai_tool_schema().function.name == "bocha_web_search"


def test_bocha_search_accepts_null_images_from_real_api_shape():
    payload = {
        "code": 200,
        "msg": None,
        "data": {
            "webPages": {
                "value": [
                    {
                        "name": "中国政府网",
                        "url": "https://www.gov.cn/",
                        "siteName": "中国政府网",
                        "snippet": "国务院官方网站。",
                    }
                ]
            },
            "images": None,
            "videos": None,
        },
    }

    tool = BoChaSearchTool(
        Settings(bocha_api_key="test-key"),
        requester=lambda *_args, **_kwargs: StubResponse(payload),
    )

    result = tool.run("国务院 官方网站", count=3)

    assert result["status"] == "success"
    assert result["result_count"] == 1
    assert result["results"][0]["url"] == "https://www.gov.cn/"
    assert result["images"] == []


def test_bocha_search_fails_without_api_key_or_query():
    tool = BoChaSearchTool(Settings(bocha_api_key=""), requester=lambda *_args, **_kwargs: None)

    missing_key = tool.run("待核查事件")
    empty_query = tool.run("   ")

    assert missing_key["status"] == "error"
    assert "BOCHA_API_KEY" in missing_key["limitations"][0]
    assert empty_query["status"] == "error"
    assert empty_query["query"] == ""


def test_bocha_search_returns_structured_error_for_invalid_count():
    tool = BoChaSearchTool(
        Settings(bocha_api_key="test-key"),
        requester=lambda *_args, **_kwargs: None,
    )

    result = tool.run("待核查事件", count="invalid")

    assert result["status"] == "error"
    assert result["results"] == []
    assert "invalid literal" in result["error"]


def test_mfc_search_query_builder_uses_whole_post_detail_and_ocr_views():
    builder = MFCSearchQueryBuilder(max_query_characters=120)
    text_queries = builder.build_text_queries(
        "网传某地发生严重洪水。官方已经宣布全市停课，预计影响20万人。"
    )
    crossmodal_query = builder.build_crossmodal_query(
        "网传某地发生严重洪水。官方已经宣布全市停课，预计影响20万人。",
        {
            "ocr_text": "2021年7月 河南郑州",
            "entities": [
                {"name": "河南郑州", "type": "location"},
                {"name": "2021年7月", "type": "date"},
            ],
        },
    )

    assert [item["query_id"] for item in text_queries] == ["Q1", "Q2"]
    assert text_queries[0]["strategy"] == "whole_post"
    assert "20万人" in text_queries[1]["query"]
    assert crossmodal_query["query_id"] == "Q3"
    assert crossmodal_query["strategy"] == "crossmodal_ocr"
    assert "河南郑州" in crossmodal_query["query"]
    assert "2021年7月" in crossmodal_query["query"]


def test_mfc_search_query_builder_uses_language_neutral_source_terms_for_non_han_text():
    builder = MFCSearchQueryBuilder(max_query_characters=160)
    query = builder.build_crossmodal_query(
        "Officials confirmed that only one airport terminal closed on April 21.",
        None,
    )

    assert query["strategy"] == "source_verification"
    assert "original source official statement fact check" in query["query"]
    assert "事实核查" not in query["query"]

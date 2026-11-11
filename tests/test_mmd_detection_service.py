import json

from app.agents.MMD_Agent import MMDDetectionAgent
from app.core.config import Settings
from app.schemas.detection import DetectionResponse, DetectionTaskType
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.mmd_detection import MMDDetectionService


class FakeImageService:
    model_factory = None

    def build_image_content_part(self, image_content=None, image_url=None):
        if not image_content and not image_url:
            return None
        return {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}}


def payload():
    return {
        "label": "Mostly False",
        "confidence": 0.82,
        "forgery_type": "C2_contextual_inconsistency",
        "atomic_points": [
            {
                "claim": "图片展示了文字所述事件。",
                "visual_observation": "图片中的背景与文字冲突。",
                "search_query": "事件 地点 现场图片",
                "point_type": "crossmodal_claim",
            }
        ],
        "fine_grained_checks": {
            "T2_distortion": {"assessment": "confirmed", "severity": 2, "evidence_basis": "text_internal", "analysis": "文字扩大了图片可支持的事实。"},
            "C2_contextual_inconsistency": {"assessment": "confirmed", "severity": 3, "evidence_basis": "visual_input", "analysis": "图片语境不匹配。"},
        },
        "user_answer": {
            "judgment": "该内容整体上基本虚假。",
            "analysis": "图片可能真实，但其语境不能支持文字中的核心事件。",
            "notes": ["未使用外部新闻来源核验。"],
        },
    }


def make_service(calls):
    def evaluator(*args):
        calls.append(args)
        return json.dumps(payload(), ensure_ascii=False)

    return MMDDetectionService(MMDDetectionAgent(FakeImageService(), Settings(), evaluator=evaluator))


def test_mmd_service_exposes_machine_label_and_user_explanation():
    calls = []
    result = make_service(calls).detect(
        text="新闻文字",
        image_content=b"image",
        image_url=None,
        question="判断真实性",
        language="zh",
    )

    assert len(calls) == 1
    assert result["label"] == "Mostly False"
    assert result["type"] == "multimodal_misinformation"
    assert result["mmd_detection"]["atomic_points"][0]["point_type"] == "crossmodal_claim"
    assert set(result["mmd_detection"]) == {
        "status",
        "label",
        "confidence",
        "forgery_type",
        "atomic_points",
        "fine_grained_checks",
        "user_answer",
    }
    assert result["user_answer"]["judgment"] == "该内容整体判断为基本虚假。"
    assert result["raw"]["workflow"]["steps"] == ["MMD_Agent"]
    assert "mmd_detection" not in result["raw"]
    assert result["raw"]["timings"]["model_calls"] == 1
    assert result["crossmodal_verdict"]["label"] == "Inconsistent"
    serialized = DetectionResponse.model_validate(result).model_dump(mode="json")
    assert serialized["mmd_detection"]["label"] == serialized["label"]
    assert serialized["forgery_type"] == "C2_contextual_inconsistency"
    assert serialized["mmd_detection"]["forgery_type"] == serialized["forgery_type"]
    assert serialized["mmd_detection"]["user_answer"] == serialized["user_answer"]
    assert set(serialized["user_answer"]) == {"judgment", "analysis", "notes"}
    assert "timings" not in serialized["mmd_detection"]
    assert len(serialized["mmd_detection"]["fine_grained_checks"]) == 8
    assert "present" not in serialized["mmd_detection"]["fine_grained_checks"]["C2_contextual_inconsistency"]


def test_mmd_service_marks_text_only_mode_without_crossmodal_result():
    calls = []
    result = make_service(calls).detect(
        text="待核查文本",
        image_content=None,
        image_url=None,
        question="核查核心主张",
        language="zh",
    )

    assert len(calls) == 1
    assert result["mode"] == "text_only"
    assert result["crossmodal_verdict"] is None
    assert result["image_verdict"] is None
    assert result["raw"]["workflow"]["name"] == "text_fact_checking"
    assert set(result["user_answer"]) == {"judgment", "analysis", "notes"}


def test_mmd_service_reports_busy_after_three_failed_model_attempts():
    calls = []

    def evaluator(*_):
        calls.append(1)
        raise TimeoutError("Temporary timeout")

    service = MMDDetectionService(MMDDetectionAgent(FakeImageService(), Settings(), evaluator=evaluator))
    result = service.detect(
        text="新闻文字", image_content=b"image", image_url=None,
        question="判断真实性", language="zh",
    )

    assert len(calls) == 3
    assert result["label"] is None
    assert result["decision_status"] == "technical_error"
    assert "服务器繁忙" in result["summary"]
    assert DetectionResponse.model_validate(result).label is None


def test_orchestrator_routes_multimodal_task_to_single_agent():
    calls = []

    class ForbiddenIFDService:
        def detect(self, **kwargs):
            raise AssertionError("deepfake detector must not run for an MMD task")

    orchestrator = DetectionOrchestrator(
        mmd_detection_service=make_service(calls),
        ifd_detection_service=ForbiddenIFDService(),
    )

    result = orchestrator.detect(
        text="新闻文字",
        image_content=b"image",
        task_type=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        language="zh",
    )

    assert len(calls) == 1
    assert result["label"] == "Mostly False"
    assert result["raw"]["workflow"]["routing"] == "single_agent"


def test_mmd_service_exposes_null_label_on_timeout():
    def timed_out(*_):
        raise TimeoutError("Request timed out.")

    service = MMDDetectionService(
        MMDDetectionAgent(FakeImageService(), Settings(), evaluator=timed_out)
    )

    result = service.detect(
        text="新闻文字",
        image_content=b"image",
        image_url=None,
        question="判断真实性",
        language="zh",
    )

    assert result["label"] is None
    assert result["confidence"] is None
    assert result["mmd_detection"]["status"] == "error"
    assert result["mmd_detection"]["label"] is None
    assert result["forgery_type"] is None
    assert result["mmd_detection"]["forgery_type"] is None
    assert result["mmd_detection"]["error"] == "Request timed out."
    assert DetectionResponse.model_validate(result).label is None


def test_mmd_service_does_not_treat_unverifiable_image_relation_as_conflict():
    data = payload()
    data["label"] = "Mixture"
    data["claim_assessments"] = [
        {
            "claim": "图片展示具体事件。",
            "importance": "core",
            "status": "not_verifiable",
            "evidence_basis": "none",
            "evidence": "图片缺少事件标识。",
        }
    ]
    data["fine_grained_checks"] = {
        "C1_semantic_inconsistency": {
            "assessment": "not_verifiable",
            "severity": 0,
            "evidence_basis": "none",
            "analysis": "图片没有足够信息验证具体事件。",
        },
        "C2_contextual_inconsistency": {
            "assessment": "not_verifiable",
            "severity": 0,
            "evidence_basis": "none",
            "analysis": "图片没有日期和地点标识。",
        },
        "C3_pragmatic_inconsistency": {
            "assessment": "absent",
            "severity": 0,
            "evidence_basis": "none",
            "analysis": "未发现诱导性组合。",
        },
    }
    service = MMDDetectionService(
        MMDDetectionAgent(
            FakeImageService(), Settings(), evaluator=lambda *_: json.dumps(data, ensure_ascii=False)
        )
    )

    result = service.detect(
        text="新闻文字",
        image_content=b"image",
        image_url=None,
        question="判断真实性",
        language="zh",
    )

    assert result["crossmodal_verdict"]["label"] == "Cannot_Determine"
    assert result["crossmodal_verdict"]["mismatched_elements"] == []
    assert len(result["crossmodal_verdict"]["missing_elements"]) == 2


def test_mmd_service_does_not_map_empty_signals_to_consistent():
    data = payload()
    data["label"] = "True"
    data["claim_assessments"] = []
    data["fine_grained_checks"] = {}
    data.pop("crossmodal_assessment", None)
    service = MMDDetectionService(
        MMDDetectionAgent(
            FakeImageService(), Settings(), evaluator=lambda *_: json.dumps(data, ensure_ascii=False)
        )
    )

    result = service.detect(
        text="新闻文字",
        image_content=b"image",
        image_url=None,
        question="判断真实性",
        language="zh",
    )

    assert result["label"] == "True"
    assert result["decision_status"] == "insufficient_evidence"
    assert result["crossmodal_verdict"]["label"] == "Cannot_Determine"
    assert result["crossmodal_verdict"]["consistency_score"] is None


def test_mmd_service_exposes_enrichment_as_evidence_and_raw_tool_context():
    class EnrichedAgent:
        def detect(self, **_kwargs):
            data = payload()
            return {
                **data,
                "status": "success",
                "decision_status": "confident",
                "claim_assessments": [],
                "crossmodal_assessment": {
                    "label": "Inconsistent",
                    "evidence_basis": "ocr_output",
                    "reason": "OCR地点与帖子地点不一致。",
                },
                "tool_context": {
                    "enabled": True,
                    "ocr": {
                        "status": "success",
                        "ocr_text": "2021年7月 河南郑州",
                        "entities": [{"name": "河南郑州", "type": "location"}],
                        "visual_objects": ["积水道路"],
                        "limitations": [],
                    },
                    "search": {
                        "status": "success",
                        "queries": [
                            {"query_id": "Q1", "strategy": "whole_post", "query": "洪水 停课", "status": "success"},
                            {"query_id": "Q2", "strategy": "key_detail", "query": "停课 20万人", "status": "success"},
                            {"query_id": "Q3", "strategy": "crossmodal_ocr", "query": "洪水 河南郑州", "status": "success"},
                        ],
                        "results": [
                            {
                                "evidence_id": "S1",
                                "query_id": "Q1",
                                "title": "官方通报",
                                "url": "https://example.gov/report",
                                "domain": "example.gov",
                                "site_name": "示例政府",
                                "snippet": "官方发布情况说明。",
                                "date_last_crawled": "2026-09-16T00:00:00Z",
                            }
                        ],
                        "images": [],
                        "limitations": [],
                    },
                    "timings": {"model_calls": 1},
                },
                "timings": {
                    "model_calls": 2,
                    "total_seconds": 3.0,
                    "enrichment_seconds": 1.0,
                    "ocr_seconds": 0.8,
                    "search_seconds": 0.3,
                    "model_seconds": 2.0,
                },
            }

    result = MMDDetectionService(EnrichedAgent()).detect(
        text="新闻文字",
        image_content=b"image",
        image_url=None,
        question="判断真实性",
        language="zh",
    )

    evidence_types = [item["type"] for item in result["evidence"]]
    assert "mmd_ocr_observation" in evidence_types
    assert "external_search_result" in evidence_types
    assert result["raw"]["workflow"]["dependencies"] == [
        "ImageOCREntityTool",
        "BoChaSearchTool",
    ]
    assert result["raw"]["mmd_tool_evidence"]["search"]["queries"][2]["query_id"] == "Q3"
    assert result["raw"]["timings"]["model_calls"] == 2
    search_evidence = next(item for item in result["evidence"] if item["type"] == "external_search_result")
    assert search_evidence.get("published_at") is None
    assert search_evidence["content"]["date_last_crawled"] == "2026-09-16T00:00:00Z"

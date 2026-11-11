import json

from app.agents.MFC_Agent import MFCDetectionAgent
from app.core.config import Settings
from app.schemas.detection import DetectionResponse, DetectionTaskType
from app.services.detection_orchestrator import DetectionOrchestrator
from app.services.mfc_detection import MFCDetectionService


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
        "type": "C2_contextual_inconsistency",
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

    return MFCDetectionService(MFCDetectionAgent(FakeImageService(), Settings(), evaluator=evaluator))


def test_mfc_service_exposes_machine_label_and_user_explanation():
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
    assert result["type"] == "C2_contextual_inconsistency"
    assert result["mfc_detection"]["atomic_points"][0]["point_type"] == "crossmodal_claim"
    assert set(result["mfc_detection"]) == {
        "status",
        "label",
        "confidence",
        "type",
        "atomic_points",
        "fine_grained_checks",
        "user_answer",
        "evidence",
    }
    assert result["user_answer"]["judgment"] == "该内容整体判断为基本虚假。"
    assert result["raw"]["workflow"]["steps"] == ["MFC_Agent"]
    assert "mfc_detection" not in result["raw"]
    assert result["raw"]["timings"]["model_calls"] == 1
    assert result["crossmodal_verdict"]["label"] == "Inconsistent"
    serialized = DetectionResponse.model_validate(result).model_dump(mode="json")
    assert serialized["mfc_detection"]["label"] == serialized["label"]
    assert serialized["confidence"] == payload()["confidence"]
    assert serialized["mfc_detection"]["confidence"] == serialized["confidence"]
    assert serialized["type"] == "C2_contextual_inconsistency"
    assert serialized["mfc_detection"]["type"] == serialized["type"]
    assert serialized["mfc_detection"]["user_answer"] == serialized["user_answer"]
    assert set(serialized["user_answer"]) == {"judgment", "analysis", "notes"}
    assert "timings" not in serialized["mfc_detection"]
    assert len(serialized["mfc_detection"]["fine_grained_checks"]) == 8
    assert "present" not in serialized["mfc_detection"]["fine_grained_checks"]["C2_contextual_inconsistency"]


def test_mfc_service_marks_text_only_mode_without_crossmodal_result():
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
    assert "纯文本事实核查智能体" in calls[0][0]
    assert "input_scope" not in calls[0][0]
    assert result["crossmodal_verdict"] is None
    assert result["image_verdict"] is None
    assert result["raw"]["workflow"]["name"] == "text_fact_checking"
    assert set(result["user_answer"]) == {"judgment", "analysis", "notes"}


def test_mfc_service_reports_busy_after_three_failed_model_attempts():
    calls = []

    def evaluator(*_):
        calls.append(1)
        raise TimeoutError("Temporary timeout")

    service = MFCDetectionService(MFCDetectionAgent(FakeImageService(), Settings(), evaluator=evaluator))
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
            raise AssertionError("deepfake detector must not run for an MFC task")

    orchestrator = DetectionOrchestrator(
        mfc_detection_service=make_service(calls),
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


def test_mfc_service_exposes_null_label_on_timeout():
    def timed_out(*_):
        raise TimeoutError("Request timed out.")

    service = MFCDetectionService(
        MFCDetectionAgent(FakeImageService(), Settings(), evaluator=timed_out)
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
    assert result["mfc_detection"]["status"] == "error"
    assert result["mfc_detection"]["label"] is None
    assert result["type"] == ""
    assert result["mfc_detection"]["type"] is None
    assert result["mfc_detection"]["error"] == "Request timed out."
    assert DetectionResponse.model_validate(result).label is None


def test_mfc_service_does_not_treat_unverifiable_image_relation_as_conflict():
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
    service = MFCDetectionService(
        MFCDetectionAgent(
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


def test_mfc_service_does_not_map_empty_signals_to_consistent():
    data = payload()
    data["label"] = "True"
    data["claim_assessments"] = []
    data["fine_grained_checks"] = {}
    data.pop("crossmodal_assessment", None)
    service = MFCDetectionService(
        MFCDetectionAgent(
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
    assert result["decision_status"] == "confident"
    assert result["confidence"] == data["confidence"]
    assert result["crossmodal_verdict"]["label"] == "Cannot_Determine"
    assert result["crossmodal_verdict"]["consistency_score"] is None


def test_mfc_service_exposes_actual_search_evidence_in_public_json():
    class SearchTool:
        def run(self, query, count=3, *, timeout_seconds=None):
            return {
                "status": "success",
                "results": [{
                    "title": "官方通报", "url": "https://example.gov/report",
                    "site_name": "示例政府", "snippet": "官方发布情况说明。",
                    "date_last_crawled": "2026-11-11T00:00:00Z",
                }],
            }

    calls = []
    def evaluator(prompt, *_):
        calls.append(prompt)
        if len(calls) == 1:
            return json.dumps({"action": "call_tools", "tool_calls": [
                {"tool": "bocha_web_search", "arguments": {"query": "事件 官方通报"}}
            ]})
        return json.dumps(payload(), ensure_ascii=False)

    agent = MFCDetectionAgent(FakeImageService(), Settings(), search_tool=SearchTool(), evaluator=evaluator)
    result = MFCDetectionService(agent).detect(
        text="新闻文字", image_content=b"image", image_url=None,
        question="判断真实性", language="zh",
    )
    serialized = DetectionResponse.model_validate(result).model_dump(mode="json")
    evidence = serialized["mfc_detection"]["evidence"]
    assert len(evidence) == 1
    assert evidence[0]["evidence_id"] == "S1"
    assert evidence[0]["url"] == "https://example.gov/report"
    assert evidence[0]["date_last_crawled"] == "2026-11-11T00:00:00Z"
    assert result["raw"]["workflow"]["dependencies"] == ["BoChaSearchTool"]
    assert result["raw"]["timings"]["model_calls"] == 2
    assert result["raw"]["timings"]["tool_rounds"] == 1
    assert result["raw"]["timings"]["search_calls"] == 1
    assert result["raw"]["timings"]["enrichment_seconds"] == 0
    assert result["raw"]["timings"]["ocr_seconds"] == 0
    search_evidence = next(item for item in result["evidence"] if item["type"] == "external_search_result")
    assert search_evidence["content"]["evidence_id"] == evidence[0]["evidence_id"]
    assert search_evidence["url"] == evidence[0]["url"]
    assert result["raw"]["mfc_tool_evidence"]["search"]["queries"][0]["strategy"] == "model_selected"

from enum import Enum
from typing import Any, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class DetectionMode(str, Enum):
    AUTO = "auto"
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    MULTIMODAL = "multimodal"


class DetectionTaskType(str, Enum):
    MULTIMODAL_FACT_CHECKING = "multimodal_fact_checking"
    DEEPFAKE_DETECTION = "deepfake_detection"
    AIGC_DETECTION = "aigc_detection"


class DetectionLabel(str, Enum):
    MFC_TRUE = "True"
    MFC_MOSTLY_TRUE = "Mostly True"
    MFC_MIXTURE = "Mixture"
    MFC_MOSTLY_FALSE = "Mostly False"
    MFC_FALSE = "False"
    COMPLETELY_FALSE = "Completely_False"
    MOSTLY_FALSE = "Mostly_False"
    MIXED = "Mixed"
    MOSTLY_TRUE = "Mostly_True"
    COMPLETELY_TRUE = "Completely_True"
    LIKELY_FALSE = "Likely_False"
    LIKELY_TRUE = "Likely_True"
    REAL = "Real"
    LIKELY_REAL = "Likely_Real"
    AI_GENERATED = "AI_Generated"
    MANIPULATED = "Manipulated"
    DEEPFAKE_SUSPECTED = "Deepfake_Suspected"
    REAL_IMAGE_ZH = "真实图像"
    FORGED_IMAGE_ZH = "伪造图像"
    SYNTHETIC_IMAGE_ZH = "合成图像"
    REAL_IMAGE = "Real_Image"
    FORGED_IMAGE = "Forged_Image"
    SYNTHETIC_IMAGE = "Synthetic_Image"
    COMPLETELY_REAL_ZH = "完全真实"
    LOCAL_TAMPERED_ZH = "局部篡改"
    AI_MANIPULATED_ZH = "AI操纵"
    FORGERY_DETECTED_ZH = "存在伪造"
    AI_GENERATED_ZH = "AI生成"
    COMPLETELY_AI_GENERATED_ZH = "完全AI生成"
    FORGERY_DETECTED = "Forgery_Detected"
    COMPLETELY_AI_GENERATED = "Completely_AI_Generated"
    OUT_OF_CONTEXT = "Out_Of_Context"
    CROSS_MODAL_MISMATCH = "Cross_Modal_Mismatch"
    CONSISTENT_TRUE = "Consistent_True"
    TEXT_FALSE = "Text_False"
    IMAGE_FALSE = "Image_False"
    TEXT_ANOMALY = "文本异常"
    IMAGE_ANOMALY = "图像异常"
    NO_ANOMALY = "均无异常"
    ALL_ANOMALY = "全部异常"
    CANNOT_DETERMINE = "Cannot_Determine"


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str | None = Field(default=None, description="Evidence type, such as search_result or visual_clue.")
    title: str | None = Field(default=None, description="Evidence title when available.")
    source: str | None = Field(default=None, description="Evidence source or publisher.")
    url: str | None = Field(default=None, description="Evidence URL when available.")
    content: Any | None = Field(default=None, description="Evidence text, visual clue, or structured payload.")
    snippet: str | None = Field(default=None, description="Short evidence excerpt.")
    published_at: str | None = Field(default=None, description="Evidence publication or crawl date when available.")


class ClaimItem(BaseModel):
    model_config = ConfigDict(extra="allow")

    claim: str = Field(default="", description="Atomic claim extracted from the input text.")
    type: str = Field(default="other", description="Claim category, for example event, number, person, or other.")
    entities: list[str] = Field(default_factory=list, description="Main entities mentioned by the claim.")
    search_query: str = Field(default="", description="Search query generated for evidence retrieval.")
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Optional claim-level evidence.")
    evidence_count: int | None = Field(default=None, ge=0, description="Number of normalized evidence items for this claim.")


class TextVerdict(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str = Field(default="", description="Text detection type: fact checking or rumor detection.")
    label: str = Field(default="Cannot_Determine", description="Text-only detection label.")
    appraisalResults: str = Field(default="", description="Natural-language text fact-checking summary.")
    explainInDetail: dict[str, Any] = Field(default_factory=dict, description="Detailed legacy fact-checking payload.")
    claims: list[ClaimItem] = Field(default_factory=list, description="Extracted claims used by text detection.")
    user_answer: dict[str, Any] = Field(default_factory=dict, description="User-facing analysis and conclusion.")


class ImageVerdict(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = Field(default="Real_Image", description="English image authenticity label; legacy labels remain readable.")
    analysis: str = Field(default="", description="Raw model analysis for the image.")
    question: str = Field(default="", description="Optional user question used to focus the analysis.")
    deepfake_risk: str = Field(default="unknown", description="Deepfake risk level or explanation.")
    manipulation_risk: str = Field(default="unknown", description="Image manipulation risk level or explanation.")
    probability: float | None = Field(default=None, ge=0, le=1, description="Overall manipulated probability.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Confidence of the image verdict.")
    tamper_score: float | None = Field(default=None, ge=0, le=1, description="Fused local tampering score from forensic tools.")
    ai_generated_score: float | None = Field(default=None, ge=0, le=1, description="Fused AI-generated image score from forensic tools.")
    visual_description: str = Field(default="", description="Visible content described by the model.")
    evidence: list[dict[str, Any] | str] = Field(
        default_factory=list,
        validation_alias=AliasChoices("evidence", "tool_evidence"),
        description="Tool evidence entries; strings are accepted for older visual clues.",
    )
    reasoning: str = Field(default="", description="Reasoning behind the image verdict.")
    type: str | None = Field(
        default=None, validation_alias=AliasChoices("type", "forgery_type"),
        description="Most likely single image forgery or synthesis method.",
    )
    forgery_types: list[str] = Field(default_factory=list, description="Forgery type hints from the forensic pipeline.")
    suspicious_regions: list[dict[str, Any]] = Field(default_factory=list, description="Global-coordinate suspicious regions.")
    limitations: list[Any] = Field(default_factory=list, description="Known evidence limitations.")
    artifacts: dict[str, Any] = Field(default_factory=dict, description="Generated forensic artifact paths.")
    user_answer: dict[str, Any] = Field(default_factory=dict, description="User-facing analysis and conclusion.")


class CrossmodalVerdict(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = Field(default="Cannot_Determine", description="Cross-modal image-text consistency label.")
    analysis: str = Field(default="", description="Raw model analysis for cross-modal consistency.")
    text_elements: dict[str, Any] = Field(default_factory=dict, description="Structured elements extracted from text.")
    visual_elements: dict[str, Any] = Field(default_factory=dict, description="Structured elements observed in the image.")
    matched_elements: list[str] = Field(default_factory=list, description="Text-image elements that agree.")
    mismatched_elements: list[str] = Field(default_factory=list, description="Text-image elements that conflict.")
    missing_elements: list[str] = Field(default_factory=list, description="Text elements not verifiable from the image.")
    reasoning: str = Field(default="", description="Reasoning behind the cross-modal consistency verdict.")
    limitations: list[Any] = Field(default_factory=list, description="Known limitations of the cross-modal judgment.")
    consistency_score: float | None = Field(default=None, ge=0, le=1, description="Optional image-text consistency score.")


class MFCFineGrainedCheck(BaseModel):
    model_config = ConfigDict(extra="allow")

    assessment: Literal["confirmed", "suspected", "not_verifiable", "absent"] = "not_verifiable"
    severity: int = Field(default=0, ge=0, le=3)
    evidence_basis: Literal[
        "visual_input",
        "ocr_output",
        "external_search",
        "text_internal",
        "model_knowledge",
        "none",
    ] = "none"
    analysis: str = ""


class DetectionUserAnswer(BaseModel):
    judgment: str = ""
    analysis: str = ""
    notes: list[str] = Field(default_factory=list)


class MFCAtomicPoint(BaseModel):
    claim: str = ""
    visual_observation: str = ""
    search_query: str = ""
    point_type: Literal["factual_claim", "visual_claim", "crossmodal_claim"] = "factual_claim"


class MFCSearchEvidence(BaseModel):
    evidence_id: str
    query_id: str
    title: str = ""
    url: str
    snippet: str = ""
    domain: str = ""
    site_name: str = ""
    source_tier: str = "other"
    date_last_crawled: str = Field(default="", description="Crawl time, not publication time.")


class MFCDetectionResult(BaseModel):
    model_config = ConfigDict(extra="allow")

    status: Literal["success", "error"] = "success"
    label: Literal["True", "Mostly True", "Mixture", "Mostly False", "False"] | None = None
    type: str | None = Field(default=None, description="Primary fine-grained check key, none, or unknown.")
    confidence: float | None = Field(default=None, ge=0, le=1)
    atomic_points: list[MFCAtomicPoint] = Field(default_factory=list)
    fine_grained_checks: dict[str, MFCFineGrainedCheck] = Field(default_factory=dict)
    user_answer: DetectionUserAnswer = Field(default_factory=DetectionUserAnswer)
    evidence: list[MFCSearchEvidence] = Field(default_factory=list, description="Actual retrieved candidates, not automatically verified facts.")


class DetectionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "mode": "multimodal",
                    "label": "Cross_Modal_Mismatch",
                    "type": "图文不一致",
                    "summary": "文本核查结果为大部分真实，图文关系为图文不一致。",
                    "confidence": 0.72,
                    "text_verdict": {
                        "label": "Mostly_True",
                        "appraisalResults": "主要事实有公开报道支持。",
                        "claims": [
                            {
                                "claim": "某地发生了新闻文本描述的事件。",
                                "type": "event",
                                "entities": ["某地"],
                                "search_query": "某地 新闻事件",
                            }
                        ],
                    },
                    "image_verdict": None,
                    "crossmodal_verdict": {
                        "label": "Inconsistent",
                        "matched_elements": ["人物类别相近"],
                        "mismatched_elements": ["地点与文本不一致"],
                        "reasoning": "图片场景不能支持文本中的地点描述。",
                    },
                    "evidence": [
                        {
                            "type": "visual_clue",
                            "content": "地点标识与新闻文本不一致。",
                        }
                    ],
                    "claims": [
                        {
                            "claim": "某地发生了新闻文本描述的事件。",
                            "type": "event",
                            "entities": ["某地"],
                            "search_query": "某地 新闻事件",
                        }
                    ],
                    "raw": {},
                }
            ]
        }
    )

    mode: DetectionMode = Field(description="Resolved detection mode actually used by the service.")
    type: str = Field(default="", description="Primary MFC misinformation dimension or IFD forgery/synthesis method.")
    label: str | None = Field(description="Final label, or null when detection fails technically.")
    summary: str = Field(default="", description="Human-readable summary of the detection result.")
    question: str = Field(default="", description="Resolved user question used to focus the analysis.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional final confidence score.")
    decision_status: str = Field(default="", description="Whether the decision is confident or evidence-limited.")
    runtime_seconds: float | None = Field(default=None, ge=0, description="End-to-end API runtime in seconds.")
    text_verdict: TextVerdict | None = Field(default=None, description="Text-only detection result, when applicable.")
    image_verdict: ImageVerdict | None = Field(default=None, description="Image-only detection result, when applicable.")
    crossmodal_verdict: CrossmodalVerdict | None = Field(
        default=None,
        description="Image-text consistency result, only present for multimodal detection.",
    )
    mfc_detection: MFCDetectionResult | None = Field(
        default=None,
        validation_alias=AliasChoices("mfc_detection", "mmd_detection"),
        description="Direct five-way multimodal misinformation result with evidence and user explanation.",
    )
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Evidence collected for the final result.")
    claims: list[ClaimItem] = Field(default_factory=list, description="Claims extracted from the text input.")
    raw: dict[str, Any] = Field(default_factory=dict, description="Raw service payloads for debugging and demos.")
    report: dict[str, Any] = Field(default_factory=dict, description="Comprehensive multimodal fusion report.")
    user_answer: dict[str, Any] = Field(default_factory=dict, description="User-facing analysis and conclusion.")
    conversation_id: str = Field(default="", description="Backend conversation id for this detection session.")
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Saved conversation history for this conversation id.",
    )

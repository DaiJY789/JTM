from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class DetectionMode(str, Enum):
    AUTO = "auto"
    TEXT_ONLY = "text_only"
    IMAGE_ONLY = "image_only"
    MULTIMODAL = "multimodal"


class DetectionLabel(str, Enum):
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

    label: str = Field(default="真实图像", description="Image authenticity label.")
    analysis: str = Field(default="", description="Raw model analysis for the image.")
    question: str = Field(default="", description="Optional user question used to focus the analysis.")
    deepfake_risk: str = Field(default="unknown", description="Deepfake risk level or explanation.")
    manipulation_risk: str = Field(default="unknown", description="Image manipulation risk level or explanation.")
    probability: float | None = Field(default=None, ge=0, le=1, description="Overall manipulated probability.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Confidence of the image verdict.")
    tamper_score: float | None = Field(default=None, ge=0, le=1, description="Fused local tampering score from forensic tools.")
    ai_generated_score: float | None = Field(default=None, ge=0, le=1, description="Fused AI-generated image score from forensic tools.")
    provenance: dict[str, Any] = Field(default_factory=dict, description="C2PA/Content Credentials provenance findings when available.")
    visual_description: str = Field(default="", description="Visible content described by the model.")
    evidence: list[str] = Field(default_factory=list, description="Visual clues used for the image verdict.")
    reasoning: str = Field(default="", description="Reasoning behind the image verdict.")
    forgery_types: list[str] = Field(default_factory=list, description="Forgery type hints from the forensic pipeline.")
    tool_evidence: list[dict[str, Any]] = Field(default_factory=list, description="Tool evidence.")
    suspicious_regions: list[dict[str, Any]] = Field(default_factory=list, description="Global-coordinate suspicious regions.")
    limitations: list[Any] = Field(default_factory=list, description="Known evidence limitations.")
    artifacts: dict[str, Any] = Field(default_factory=dict, description="Generated forensic artifact paths.")
    user_answer: dict[str, Any] = Field(default_factory=dict, description="User-facing analysis and conclusion.")


class AlignmentVerdict(BaseModel):
    model_config = ConfigDict(extra="allow")

    label: str = Field(default="Cannot_Determine", description="Image-text alignment label.")
    analysis: str = Field(default="", description="Raw model analysis for cross-modal consistency.")
    text_elements: dict[str, Any] = Field(default_factory=dict, description="Structured elements extracted from text.")
    visual_elements: dict[str, Any] = Field(default_factory=dict, description="Structured elements observed in the image.")
    matched_elements: list[str] = Field(default_factory=list, description="Text-image elements that agree.")
    mismatched_elements: list[str] = Field(default_factory=list, description="Text-image elements that conflict.")
    missing_elements: list[str] = Field(default_factory=list, description="Text elements not verifiable from the image.")
    reasoning: str = Field(default="", description="Reasoning behind the alignment verdict.")
    limitations: list[Any] = Field(default_factory=list, description="Known limitations of the alignment judgment.")
    consistency_score: float | None = Field(default=None, ge=0, le=1, description="Optional image-text consistency score.")


class DetectionResponse(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {
                    "mode": "multimodal",
                    "label": "均无异常",
                    "type": "图文不一致",
                    "summary": "综合判断为均无异常，图文一致性为图文不一致。",
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
                    "image_verdict": {
                        "label": "Real",
                        "deepfake_risk": "low",
                        "manipulation_risk": "low",
                        "visual_description": "图片展示了相关人物在室内场景中发言。",
                        "evidence": ["面部边缘和光照未见明显生成痕迹。"],
                        "reasoning": "未发现明显伪造线索。",
                    },
                    "alignment_verdict": {
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
    type: str = Field(default="", description="Detection type, such as 事实核查/fact_checking or 谣言检测/rumor_detection for text.")
    label: str = Field(description="Final fused label.")
    summary: str = Field(default="", description="Human-readable summary of the detection result.")
    question: str = Field(default="", description="Resolved user question used to focus the analysis.")
    confidence: float | None = Field(default=None, ge=0, le=1, description="Optional final confidence score.")
    runtime_seconds: float | None = Field(default=None, ge=0, description="End-to-end API runtime in seconds.")
    text_verdict: TextVerdict | None = Field(default=None, description="Text-only detection result, when applicable.")
    image_verdict: ImageVerdict | None = Field(default=None, description="Image-only detection result, when applicable.")
    alignment_verdict: AlignmentVerdict | None = Field(
        default=None,
        description="Image-text consistency result, only present for multimodal detection.",
    )
    evidence: list[EvidenceItem] = Field(default_factory=list, description="Evidence collected for the final result.")
    claims: list[ClaimItem] = Field(default_factory=list, description="Claims extracted from the text input.")
    raw: dict[str, Any] = Field(default_factory=dict, description="Raw service payloads for debugging and demos.")
    user_answer: dict[str, Any] = Field(default_factory=dict, description="User-facing analysis and conclusion.")
    conversation_id: str = Field(default="", description="Backend conversation id for this detection session.")
    conversation_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Backend-maintained recent conversation history for this conversation id.",
    )

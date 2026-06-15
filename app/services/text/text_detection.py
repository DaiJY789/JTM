from typing import Any

from app.core.language import normalize_language
from app.services.text.claim_extractor import ClaimExtractor
from app.services.text.evidence_search import EvidenceSearchService
from app.services.text.text_verdict import TextVerdictService


class TextDetectionService:
    def __init__(
        self,
        text_verdict_service: TextVerdictService,
        claim_extractor: ClaimExtractor | None = None,
        evidence_search_service: EvidenceSearchService | None = None,
    ):
        self.text_verdict_service = text_verdict_service
        self.claim_extractor = claim_extractor or ClaimExtractor(
            config=text_verdict_service.config,
            model_factory=text_verdict_service.model_factory,
        )
        self.evidence_search_service = evidence_search_service or EvidenceSearchService()

    def detect(
        self,
        text: str,
        history: list | None = None,
        language: str = "zh",
    ) -> dict[str, Any]:
        language = normalize_language(language)
        claims = self.claim_extractor.extract(text, language)
        evidence = self.evidence_search_service.search(claims, language)
        claims = self.evidence_search_service.attach_evidence(claims, evidence)
        result = self.text_verdict_service.evaluate(
            text=text,
            claims=claims,
            evidence=evidence,
            history=history or [],
            language=language,
        )
        explain = result.get("explainInDetail", {})
        evidence_items = self.evidence_search_service.flatten_evidence(evidence)
        return {
            "mode": "text_only",
            "type": result.get("type", "谣言检测" if language == "zh" else "rumor_detection"),
            "label": result.get("label") or "Cannot_Determine",
            "summary": result.get("appraisalResults", ""),
            "confidence": result.get("confidence"),
            "text_verdict": {
                "type": result.get("type", "谣言检测" if language == "zh" else "rumor_detection"),
                "label": result.get("label") or "Cannot_Determine",
                "appraisalResults": result.get("appraisalResults", ""),
                "explainInDetail": explain,
                "claims": claims,
                "confidence": result.get("confidence"),
                "evidence_assessment": result.get("evidence_assessment", ""),
                "claim_assessments": result.get("claim_assessments", []),
                "user_answer": result.get("user_answer", {}),
            },
            "evidence": evidence_items,
            "claims": claims,
            "raw": {
                "text_detection": result,
            },
            "user_answer": result.get("user_answer", {}),
        }

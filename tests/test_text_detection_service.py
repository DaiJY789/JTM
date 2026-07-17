from app.services.textual.T_pipeline import TextualDetectionPipeline


class StubTextualVerdictService:
    config = object()
    model_factory = object()

    def evaluate(self, text, claims, evidence, history, language):
        assert evidence[0]["items"][0]["title"] == "source"
        return {
            "type": "事实核查",
            "label": "Mostly_True",
            "confidence": 0.6,
            "appraisalResults": "整体上有证据支持。",
            "explainInDetail": {"news": [{"type": "search_result", "title": "source", "snippet": "supporting evidence"}]},
            "evidence_assessment": "证据基本支持。",
            "claim_assessments": [{"claim": text, "evidence_status": "supported"}],
            "label_policy": {
                "model_judged_evidence_sufficiency": True,
            },
        }


class StubClaimExtractor:
    def extract(self, text, language):
        return [{"claim": text, "type": "event", "entities": [], "search_query": text}]


class StubEvidenceSearchService:
    def search(self, claims, language):
        return [{
            "claim_index": 0,
            "claim": claims[0]["claim"],
            "search_query": claims[0]["search_query"],
            "items": [{"type": "search_result", "title": "source", "snippet": "supporting evidence"}],
        }]

    def attach_evidence(self, claims, evidence):
        enriched = []
        for claim in claims:
            item = dict(claim)
            item.update(
                {
                    "evidence": evidence[0]["items"],
                    "evidence_count": 1,
                }
            )
            enriched.append(item)
        return enriched

    def flatten_evidence(self, evidence):
        return evidence[0]["items"]


def test_text_detection_includes_claim_level_evidence():
    service = TextualDetectionPipeline(
        textual_verdict_service=StubTextualVerdictService(),
        claim_extractor=StubClaimExtractor(),
        evidence_search_service=StubEvidenceSearchService(),
    )

    result = service.detect("新闻事实", language="zh")

    assert result["label"] == "Mostly_True"
    assert result["type"] == "事实核查"
    assert result["text_verdict"]["type"] == "事实核查"
    assert result["confidence"] == 0.6
    assert result["claims"][0]["evidence_count"] == 1
    assert result["evidence"][0]["title"] == "source"
    assert len(result["evidence"]) == 1
    assert result["text_verdict"]["claim_assessments"][0]["evidence_status"] == "supported"

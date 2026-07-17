from app.services.textual.claim_extractor import ClaimExtractor


class StubModelFactory:
    pass


def make_extractor():
    return ClaimExtractor(model_factory=StubModelFactory())


def test_parse_fenced_json_claims():
    extractor = make_extractor()
    raw = """```json
[
  {"claim": "A happened in B", "type": "event", "entities": ["A", "B"], "search_query": "A B"}
]
```"""
    claims = extractor.parse(raw, "fallback")
    assert claims == [{
        "claim": "A happened in B",
        "type": "event",
        "entities": ["A", "B"],
        "search_query": "A B",
    }]


def test_parse_invalid_json_falls_back():
    extractor = make_extractor()
    claims = extractor.parse("not json", "原始文本")
    assert claims[0]["claim"] == "原始文本"
    assert claims[0]["type"] == "other"


def test_parse_json_with_trailing_text():
    extractor = make_extractor()
    raw = """Here is the JSON:
[
  {"claim": "A happened", "type": "event", "entities": ["A"], "search_query": "A happened"}
]
Extra text that should be ignored."""
    claims = extractor.parse(raw, "fallback")
    assert claims[0]["claim"] == "A happened"


def test_parse_limits_claim_count_to_three():
    extractor = make_extractor()
    raw = str([
        {"claim": "claim 1"},
        {"claim": "claim 2"},
        {"claim": "claim 3"},
        {"claim": "claim 4"},
    ]).replace("'", '"')
    claims = extractor.parse(raw, "fallback")
    assert [item["claim"] for item in claims] == ["claim 1", "claim 2", "claim 3"]

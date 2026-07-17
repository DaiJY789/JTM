from threading import Event, Lock

from app.services.textual.evidence_search import EvidenceSearchService


def test_enrich_claims_adds_normalized_evidence():
    def search_runner(query, language):
        assert query == "A event"
        assert language == "en"
        return {
            "data": {
                "webPages": {
                    "value": [
                        {
                            "name": "Reliable report",
                            "siteName": "Reuters",
                            "url": "https://example.com/report",
                            "snippet": "The event was confirmed by officials.",
                        }
                    ]
                }
            }
        }

    service = EvidenceSearchService(search_runner=search_runner)
    original_claims = [{"claim": "A event happened", "type": "event", "entities": ["A"], "search_query": "A event"}]
    evidence = service.search(original_claims, language="en")
    claims = service.attach_evidence(original_claims, evidence)

    assert evidence[0]["items"][0]["title"] == "Reliable report"
    assert claims[0]["evidence_count"] == 1
    assert claims[0]["evidence"][0]["title"] == "Reliable report"
    assert "verdict" not in claims[0]
    assert "confidence" not in claims[0]


def test_retrieve_evidence_fails_closed_to_empty_list():
    def broken_search_runner(query, language):
        raise RuntimeError("network error")

    service = EvidenceSearchService(search_runner=broken_search_runner)
    assert service.retrieve_evidence("query") == []


def test_search_runs_claim_retrieval_in_parallel_and_keeps_order():
    started = []
    lock = Lock()
    both_started = Event()

    def search_runner(query, language):
        with lock:
            started.append(query)
            if len(started) == 2:
                both_started.set()
        assert both_started.wait(timeout=2)
        return {"results": [{"title": f"title {query}", "url": f"https://example.com/{query}"}]}

    service = EvidenceSearchService(search_runner=search_runner, max_workers=2, queries_per_claim=1)
    claims = [
        {"claim": "claim b", "search_query": "b"},
        {"claim": "claim a", "search_query": "a"},
    ]

    evidence = service.search(claims, language="en")

    assert {item["search_query"] for item in evidence} == {"b", "a"}
    assert [item["claim_index"] for item in evidence] == [0, 1]
    assert [item["items"][0]["title"] for item in evidence] == ["title b", "title a"]


def test_each_claim_retrieves_multiple_evidence_queries_in_parallel():
    started = []
    lock = Lock()
    enough_started = Event()

    def search_runner(query, language):
        with lock:
            started.append(query)
            if len(started) >= 3:
                enough_started.set()
        assert enough_started.wait(timeout=2)
        return {"results": [{"title": f"title {query}", "url": f"https://example.com/{query.replace(' ', '-')}"}]}

    service = EvidenceSearchService(search_runner=search_runner, max_workers=3, queries_per_claim=3, max_results=5)
    evidence = service.search(
        [{"claim": "A happened in B", "entities": ["A", "B"], "search_query": "A B event"}],
        language="en",
    )

    assert len(started) == 3
    assert len(evidence[0]["items"]) == 3
    assert evidence[0]["claim_index"] == 0

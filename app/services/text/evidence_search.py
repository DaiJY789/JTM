import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from app.core.language import normalize_language
from app.tools.text.search import BoChaSearchTool

SearchRunner = Callable[[str, str], Any]


class EvidenceSearchService:
    """Retrieve and normalize evidence for extracted claims."""

    def __init__(
        self,
        search_runner: SearchRunner | None = None,
        max_results: int = 5,
        max_workers: int = 4,
        queries_per_claim: int = 3,
    ):
        self.search_runner = search_runner
        self.max_results = max(3, min(5, max_results))
        self.max_workers = max(1, max_workers)
        self.queries_per_claim = max(1, min(5, queries_per_claim))

    def enrich_claims(
        self,
        claims: list[dict[str, Any]],
        language: str = "zh",
    ) -> list[dict[str, Any]]:
        evidence = self.search(claims, language)
        return self.attach_evidence(claims, evidence)

    def search(
        self,
        claims: list[dict[str, Any]],
        language: str = "zh",
    ) -> list[dict[str, Any]]:
        language = normalize_language(language)
        jobs = []
        for index, claim in enumerate(claims):
            query = self._query_for_claim(claim)
            jobs.append((index, claim, query))

        if not jobs:
            return []
        if len(jobs) == 1 or self.max_workers == 1:
            return [self._search_one(index, claim, query, language) for index, claim, query in jobs]

        worker_count = min(self.max_workers, len(jobs))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            return list(executor.map(lambda job: self._search_one(*job, language), jobs))

    def _search_one(
        self,
        index: int,
        claim: dict[str, Any],
        query: str,
        language: str,
    ) -> dict[str, Any]:
        items = self.retrieve_claim_evidence(claim, query, language)
        return {
            "claim_index": index,
            "claim": claim.get("claim", ""),
            "search_query": query,
            "items": items,
        }

    def retrieve_claim_evidence(
        self,
        claim: dict[str, Any],
        query: str,
        language: str = "zh",
    ) -> list[dict[str, Any]]:
        queries = self._queries_for_claim(claim, query, language)
        if not queries:
            return []
        if len(queries) == 1:
            return self.retrieve_evidence(queries[0], language)

        worker_count = min(self.max_workers, len(queries))
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            results = list(executor.map(lambda item: self.retrieve_evidence(item, language), queries))

        merged = []
        seen = set()
        for query_text, items in zip(queries, results):
            for item in items:
                key = self._evidence_key(item)
                if key in seen:
                    continue
                seen.add(key)
                enriched = dict(item)
                enriched.setdefault("search_query", query_text)
                merged.append(enriched)
                if len(merged) >= self.max_results:
                    return merged
        return merged

    def attach_evidence(self, claims: list[dict[str, Any]], evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        evidence_by_index = {
            item.get("claim_index"): item.get("items", [])
            for item in evidence
            if isinstance(item, dict)
        }
        enriched_claims = []
        for index, claim in enumerate(claims):
            enriched = dict(claim)
            items = evidence_by_index.get(index, [])
            enriched.update({
                "evidence": items,
                "evidence_count": len(items),
            })
            enriched_claims.append(enriched)
        return enriched_claims

    def flatten_evidence(self, evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flattened = []
        for bundle in evidence:
            if not isinstance(bundle, dict):
                continue
            for item in bundle.get("items", []):
                if not isinstance(item, dict):
                    continue
                enriched = dict(item)
                enriched["claim_index"] = bundle.get("claim_index")
                enriched["claim"] = bundle.get("claim", "")
                enriched["search_query"] = bundle.get("search_query", "")
                flattened.append(enriched)
        return flattened

    def retrieve_evidence(self, query: str, language: str = "zh") -> list[dict[str, Any]]:
        if not query.strip():
            return []
        try:
            raw_result = self._run_search(query, language)
        except Exception:
            return []
        return self._normalize_search_result(raw_result)[: self.max_results]

    def _run_search(self, query: str, language: str) -> Any:
        if self.search_runner:
            return self.search_runner(query, language)
        return BoChaSearchTool(selected_language=language)._run(query)

    def _query_for_claim(self, claim: dict[str, Any]) -> str:
        query = str(claim.get("search_query") or claim.get("claim") or "").strip()
        return re.sub(r"\s+", " ", query)[:160]

    def _queries_for_claim(self, claim: dict[str, Any], query: str, language: str) -> list[str]:
        claim_text = re.sub(r"\s+", " ", str(claim.get("claim") or "").strip())
        entities = claim.get("entities") if isinstance(claim.get("entities"), list) else []
        entity_text = " ".join(str(item).strip() for item in entities[:4] if str(item or "").strip())
        suffixes = ["事实核查", "官方 通报"] if normalize_language(language) == "zh" else ["fact check", "official source"]
        candidates = [
            query,
            claim_text,
            f"{entity_text} {claim_text}".strip(),
            f"{query} {suffixes[0]}".strip(),
            f"{query} {suffixes[1]}".strip(),
        ]
        normalized = []
        seen = set()
        for candidate in candidates:
            value = re.sub(r"\s+", " ", str(candidate or "").strip())[:160]
            key = value.lower()
            if not value or key in seen:
                continue
            seen.add(key)
            normalized.append(value)
            if len(normalized) >= self.queries_per_claim:
                break
        return normalized

    def _evidence_key(self, item: dict[str, Any]) -> tuple[str, str]:
        url = str(item.get("url") or "").strip().lower()
        title = str(item.get("title") or item.get("content") or "").strip().lower()
        return (url, title)

    def _normalize_search_result(self, raw_result: Any) -> list[dict[str, Any]]:
        if isinstance(raw_result, dict):
            web_pages = raw_result.get("data", {}).get("webPages", {}).get("value")
            if isinstance(web_pages, list):
                return [self._normalize_page(item) for item in web_pages if isinstance(item, dict)]
            results = raw_result.get("results")
            if isinstance(results, list):
                return [self._normalize_page(item) for item in results if isinstance(item, dict)]
            return [self._normalize_page(raw_result)]

        if isinstance(raw_result, list):
            return [self._normalize_page(item) for item in raw_result if isinstance(item, dict)]

        if isinstance(raw_result, str) and raw_result.strip() and raw_result != "No results found.":
            return [{"type": "search_result", "content": raw_result.strip()}]
        return []

    def _normalize_page(self, page: dict[str, Any]) -> dict[str, Any]:
        return {
            "type": "search_result",
            "title": page.get("name") or page.get("title") or "",
            "source": page.get("siteName") or page.get("source") or "",
            "url": page.get("url") or page.get("link") or "",
            "snippet": page.get("snippet") or page.get("content") or page.get("summary") or "",
            "published_at": page.get("dateLastCrawled") or page.get("published_date") or "",
        }

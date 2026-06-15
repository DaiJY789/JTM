import json
import re
from typing import Any

from app.agents.models import ModelFactory
from app.core.config import Settings, settings
from app.core.language import is_zh, normalize_language
from app.prompts.text import CLAIM_EXTRACTION_PROMPT_EN, CLAIM_EXTRACTION_PROMPT_ZH


class ClaimExtractor:
    def __init__(self, config: Settings = settings, model_factory: ModelFactory | None = None):
        self.config = config
        self.model_factory = model_factory or ModelFactory(config)

    def extract(self, text: str, language: str = "zh") -> list[dict[str, Any]]:
        language = normalize_language(language)
        if not text.strip():
            return []

        try:
            client = self.model_factory.deepseek_client(timeout=self.config.claim_extraction_timeout_seconds)
            completion = client.chat.completions.create(
                model=self.config.deepseek_model,
                messages=[
                    {"role": "system", "content": self._prompt(language)},
                    {"role": "user", "content": text},
                ],
                max_tokens=800,
            )
            raw_output = completion.choices[0].message.content
            return self.parse(raw_output, text)
        except Exception:
            return self.fallback_claims(text)

    def parse(self, raw_output: str, source_text: str) -> list[dict[str, Any]]:
        payload = self._extract_json_payload(raw_output)
        try:
            parsed = json.loads(payload)
        except (json.JSONDecodeError, TypeError):
            return self.fallback_claims(source_text)

        if isinstance(parsed, dict):
            parsed = parsed.get("claims", [parsed])
        if not isinstance(parsed, list):
            return self.fallback_claims(source_text)

        claims = [self._normalize_claim(item) for item in parsed if isinstance(item, dict)]
        claims = [item for item in claims if item["claim"]]
        return claims[: self.config.claim_max_count] or self.fallback_claims(source_text)

    def fallback_claims(self, text: str) -> list[dict[str, Any]]:
        cleaned = re.sub(r"\s+", " ", text).strip()
        return [{
            "claim": cleaned,
            "type": "other",
            "entities": [],
            "search_query": cleaned[:120],
        }] if cleaned else []

    def _prompt(self, language: str) -> str:
        return CLAIM_EXTRACTION_PROMPT_ZH if is_zh(language) else CLAIM_EXTRACTION_PROMPT_EN

    def _extract_json_payload(self, raw_output: str) -> str:
        text = raw_output.strip()
        candidates = [text]
        candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S))
        for candidate in candidates:
            parsed = self._first_valid_json(candidate)
            if parsed:
                return parsed
        return text

    def _first_valid_json(self, text: str) -> str:
        decoder = json.JSONDecoder()
        for idx, char in enumerate(text):
            if char not in "[{":
                continue
            try:
                parsed, _ = decoder.raw_decode(text[idx:])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, (list, dict)):
                return json.dumps(parsed, ensure_ascii=False)
        return ""

    def _normalize_claim(self, item: dict[str, Any]) -> dict[str, Any]:
        entities = item.get("entities", [])
        if isinstance(entities, str):
            entities = [entities]
        if not isinstance(entities, list):
            entities = []

        claim = str(item.get("claim", "")).strip()
        return {
            "claim": claim,
            "type": str(item.get("type", "other")).strip() or "other",
            "entities": [str(entity).strip() for entity in entities if str(entity).strip()],
            "search_query": str(item.get("search_query") or claim).strip(),
        }

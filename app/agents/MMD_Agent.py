from __future__ import annotations

import json
import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from difflib import SequenceMatcher
from time import perf_counter
from typing import Any
from urllib.parse import urlparse

from app.agents.models import ModelFactory
from app.agents.prompts.mmd import MMD_PROMPT_EN, MMD_PROMPT_ZH
from app.agents.tools.mmd.search_query_builder import MMDSearchQueryBuilder
from app.core.config import Settings, settings
from app.core.language import is_zh, normalize_language
from app.services.visual.image_io import ImageService


MMDEvaluator = Callable[[str, bytes | None, str | None, str], str]


class MMDDetectionAgent:
    """Enrich a text item with optional images, then make one final decision."""

    VERDICTS = ("True", "Mostly True", "Mixture", "Mostly False", "False")
    ASSESSMENTS = {"confirmed", "suspected", "not_verifiable", "absent"}
    CLAIM_STATUSES = {"supported", "contradicted", "not_verifiable"}
    EVIDENCE_BASES = {
        "visual_input",
        "ocr_output",
        "external_search",
        "text_internal",
        "model_knowledge",
        "none",
    }
    CROSSMODAL_LABELS = {
        "Consistent",
        "Partially_Consistent",
        "Inconsistent",
        "Cannot_Determine",
    }
    SIGNAL_KEYS = (
        "T1_fabrication",
        "T2_distortion",
        "T3_misleading_context",
        "V1_synthetic_visual_content",
        "V2_visual_editing",
        "C1_semantic_inconsistency",
        "C2_contextual_inconsistency",
        "C3_pragmatic_inconsistency",
    )
    VERDICT_ALIASES = {
        "true": "True",
        "completely_true": "True",
        "mostly true": "Mostly True",
        "mostly_true": "Mostly True",
        "likely_true": "Mostly True",
        "mixture": "Mixture",
        "mixed": "Mixture",
        "mostly false": "Mostly False",
        "mostly_false": "Mostly False",
        "likely_false": "Mostly False",
        "false": "False",
        "completely_false": "False",
    }
    LABELS_ZH = {
        "True": "真实",
        "Mostly True": "基本真实",
        "Mixture": "真伪混合",
        "Mostly False": "基本虚假",
        "False": "虚假",
    }

    def __init__(
        self,
        image_service: ImageService,
        config: Settings = settings,
        model_factory: ModelFactory | None = None,
        ocr_tool: Any | None = None,
        search_tool: Any | None = None,
        query_builder: MMDSearchQueryBuilder | None = None,
        evaluator: MMDEvaluator | None = None,
    ):
        self.image_service = image_service
        self.config = config
        self.model_factory = model_factory or image_service.model_factory
        self.ocr_tool = ocr_tool
        self.search_tool = search_tool
        self.query_builder = query_builder or MMDSearchQueryBuilder()
        self.evaluator = evaluator

    def detect(
        self,
        *,
        text: str,
        image_content: bytes | None = None,
        image_url: str | None = None,
        image_contents: list[bytes] | tuple[bytes, ...] | None = None,
        image_urls: list[str] | tuple[str, ...] | None = None,
        image_labels: list[str] | tuple[str, ...] | None = None,
        question: str = "",
        language: str = "zh",
    ) -> dict[str, Any]:
        started_at = perf_counter()
        language = normalize_language(language)
        normalized_text = str(text or "").strip()
        normalized_contents = list(
            image_contents or (() if image_content is None else (image_content,))
        )
        normalized_urls = list(
            dict.fromkeys(
                value.strip()
                for value in ([image_url] if image_url else []) + list(image_urls or [])
                if value and value.strip()
            )
        )
        image_parts = self._build_image_parts(normalized_contents, normalized_urls)
        normalized_image_labels = self._normalize_image_labels(
            image_labels, len(image_parts)
        )
        image_ready_at = perf_counter()
        if not normalized_text:
            return self._error_result(
                "缺少待检测文字。" if is_zh(language) else "Text is required.",
                started_at,
                language=language,
            )
        enrichment_started_at = perf_counter()
        tool_context = self._collect_tool_context(
            text=normalized_text,
            image_content=normalized_contents[0] if normalized_contents else None,
            image_url=normalized_urls[0] if normalized_urls else None,
            question=question,
            language=language,
        )
        enrichment_finished_at = perf_counter()
        prompt = self._build_prompt(
            normalized_text,
            question,
            language,
            tool_context,
            image_count=len(image_parts),
            image_labels=normalized_image_labels,
        )
        raw_output = ""
        model_call_count = 0
        model_attempts: list[dict[str, Any]] = []
        model_started_at = perf_counter()
        try:
            parsed = None
            max_invalid_retries = max(0, min(2, int(self.config.mmd_invalid_output_max_retries)))
            for attempt_index in range(max_invalid_retries + 1):
                model_call_count += 1
                attempt_started_at = perf_counter()
                attempt_prompt = self._retry_prompt(prompt, attempt_index, language)
                try:
                    raw_output, attempt_metadata = self._call_model(
                        attempt_prompt,
                        image_parts,
                        normalized_image_labels,
                        normalized_contents[0] if normalized_contents else None,
                        normalized_urls[0] if normalized_urls else None,
                        language,
                    )
                except Exception as exc:
                    model_attempts.append(
                        {
                            "attempt": model_call_count,
                            "seconds": round(perf_counter() - attempt_started_at, 4),
                            "output_characters": 0,
                            "error": str(exc),
                        }
                    )
                    if attempt_index >= max_invalid_retries:
                        raise
                    continue
                attempt_record = {
                    "attempt": model_call_count,
                    "seconds": round(perf_counter() - attempt_started_at, 4),
                    "output_characters": len(raw_output),
                    **attempt_metadata,
                }
                try:
                    candidate = self._parse_json(raw_output)
                    if not candidate:
                        raise ValueError("MMD model output must be a non-empty JSON object")
                    if self._normalize_verdict(candidate.get("verdict") or candidate.get("label")) is None:
                        raise ValueError("MMD model output must contain a valid five-way verdict")
                    parsed = candidate
                except ValueError as exc:
                    attempt_record["error"] = str(exc)
                    model_attempts.append(attempt_record)
                    if attempt_index >= max_invalid_retries:
                        raise
                    continue
                model_attempts.append(attempt_record)
                break
            model_finished_at = perf_counter()
            if parsed is None:
                raise ValueError("MMD model output must be a JSON object")
            result = self._normalize_result(
                parsed,
                language,
                external_search_available=bool(tool_context.get("search", {}).get("results")),
                has_image=bool(image_parts),
            )
            result["status"] = "success"
        except Exception as exc:
            result = self._error_result(
                "多模态检测模型调用失败。" if is_zh(language) else "The multimodal detection model call failed.",
                started_at,
                model_calls=model_call_count,
                language=language,
            )
            result["error"] = str(exc)
            result["tool_context"] = tool_context
            model_finished_at = perf_counter()
            result["timings"] = {
                "model_calls": model_call_count
                + int(tool_context.get("timings", {}).get("model_calls", 0)),
                "image_preprocess_seconds": round(image_ready_at - started_at, 4),
                "enrichment_seconds": round(enrichment_finished_at - enrichment_started_at, 4),
                "ocr_seconds": tool_context.get("timings", {}).get("ocr_seconds", 0.0),
                "search_seconds": tool_context.get("timings", {}).get("search_seconds", 0.0),
                "model_seconds": round(model_finished_at - model_started_at, 4),
                "total_seconds": round(model_finished_at - started_at, 4),
                "model_output_characters": len(raw_output),
                "input_image_count": len(image_parts),
                **self._aggregate_model_attempts(model_attempts),
            }
            return result

        result["tool_context"] = tool_context
        result["timings"] = {
            "model_calls": model_call_count
            + int(tool_context.get("timings", {}).get("model_calls", 0)),
            "image_preprocess_seconds": round(image_ready_at - started_at, 4),
            "enrichment_seconds": round(enrichment_finished_at - enrichment_started_at, 4),
            "ocr_seconds": tool_context.get("timings", {}).get("ocr_seconds", 0.0),
            "search_seconds": tool_context.get("timings", {}).get("search_seconds", 0.0),
            "model_seconds": round(model_finished_at - model_started_at, 4),
            "total_seconds": round(perf_counter() - started_at, 4),
            "model_output_characters": len(raw_output),
            "input_image_count": len(image_parts),
            **self._aggregate_model_attempts(model_attempts),
        }
        return result

    def _retry_prompt(self, prompt: str, attempt_index: int, language: str) -> str:
        if attempt_index == 0:
            return prompt
        correction = (
            "上一次调用失败、响应为空对象或 JSON/标签无效。本次必须返回非空 JSON 对象，并完整包含 verdict、confidence、atomic_points、claim_assessments、fine_grained_checks 和 explanation；不要返回 {}。"
            if is_zh(language)
            else "The previous call failed or returned an empty object, invalid JSON, or invalid verdict. Return a non-empty JSON object containing verdict, confidence, atomic_points, claim_assessments, fine_grained_checks, and explanation. Never return {}."
        )
        return f"{prompt}\n\nRETRY_REQUIREMENT_{attempt_index}: {correction}"

    def _aggregate_model_attempts(
        self,
        attempts: list[dict[str, Any]],
    ) -> dict[str, Any]:
        final = attempts[-1] if attempts else {}
        return {
            "thinking_enabled": self.config.mmd_enable_thinking,
            "thinking_characters": sum(
                int(item.get("thinking_characters") or 0) for item in attempts
            ),
            "finish_reason": str(final.get("finish_reason") or ""),
            "prompt_tokens": sum(int(item.get("prompt_tokens") or 0) for item in attempts),
            "completion_tokens": sum(
                int(item.get("completion_tokens") or 0) for item in attempts
            ),
            "reasoning_tokens": sum(
                int(item.get("reasoning_tokens") or 0) for item in attempts
            ),
            "invalid_output_retries": max(0, len(attempts) - 1),
            "total_model_output_characters": sum(
                int(item.get("output_characters") or 0) for item in attempts
            ),
            "model_attempts": attempts,
        }

    def _build_image_parts(
        self,
        image_contents: list[bytes],
        image_urls: list[str],
    ) -> list[dict[str, Any]]:
        builder = getattr(self.image_service, "build_mmd_image_content_part", None)
        if not callable(builder):
            builder = self.image_service.build_image_content_part
        parts = []
        for content in image_contents:
            part = builder(content, None)
            if isinstance(part, dict):
                parts.append(part)
        for url in image_urls:
            part = builder(None, url)
            if isinstance(part, dict):
                parts.append(part)
        return parts[: max(1, int(self.config.mmd_max_images))]

    def _build_prompt(
        self,
        text: str,
        question: str,
        language: str,
        tool_context: dict[str, Any] | None = None,
        image_count: int = 1,
        image_labels: list[str] | None = None,
    ) -> str:
        base = MMD_PROMPT_ZH if is_zh(language) else MMD_PROMPT_EN
        resolved_labels = image_labels or [
            f"Image {index}" for index in range(1, image_count + 1)
        ]
        text_only = image_count == 0
        payload = {
            "current_date": date.today().isoformat(),
            "input_scope": {
                "content_mode": "text_only" if text_only else "multimodal",
                "selected_image_count": image_count,
                "other_news_images_provided": image_count > 1,
                "image_labels": resolved_labels,
                "instruction": self._input_scope_instruction(text_only, language),
            },
            "news_text": text,
            "user_question": str(question or "").strip(),
            "tool_context": tool_context or {"enabled": False},
        }
        return f"{base}\n\nINPUT_JSON:\n{json.dumps(payload, ensure_ascii=False)}"

    def _input_scope_instruction(self, text_only: bool, language: str) -> str:
        if text_only:
            return (
                "本次未提供图片，仅执行文本事实核查。visual_observation 留空；"
                "V1–V2 与 C1–C3 标记 not_verifiable、severity=0、evidence_basis=none，且不得影响 verdict。"
                if is_zh(language)
                else "No image was supplied. Perform text-only fact-checking. Leave visual_observation empty; mark V1-V2 and C1-C3 not_verifiable with severity 0 and evidence_basis none, and do not use them to determine the verdict."
            )
        return (
            "按输入顺序分别检查所有已提供图片，并严格使用 image_labels 中的原始图片编号；"
            "未提供的其他图片不得当作反驳证据。"
            if is_zh(language)
            else "Inspect every supplied image in input order, use the original identifiers in image_labels, and never treat unavailable images as counter-evidence."
        )

    def _collect_tool_context(
        self,
        *,
        text: str,
        image_content: bytes | None,
        image_url: str | None,
        question: str,
        language: str,
    ) -> dict[str, Any]:
        if not self.config.mmd_external_evidence_enabled or (
            self.ocr_tool is None and self.search_tool is None
        ):
            return {
                "enabled": False,
                "ocr": {"status": "not_run"},
                "search": {"status": "not_run", "queries": [], "results": [], "images": []},
                "timings": {"ocr_seconds": 0.0, "search_seconds": 0.0, "model_calls": 0},
            }

        started_at = perf_counter()
        has_image = bool(image_content or image_url)
        ocr_enabled = self.ocr_tool is not None and has_image
        query_limit = max(1, min(3, int(self.config.mmd_search_query_count)))
        first_queries = self.query_builder.build_text_queries(text, question)[: min(2, query_limit)]
        if query_limit >= 3 and not ocr_enabled:
            first_queries.append(
                self.query_builder.build_crossmodal_query(text, None, question)
            )

        ocr_result: dict[str, Any] = {"status": "not_run"}
        search_batches: list[tuple[dict[str, str], dict[str, Any]]] = []
        phase_workers = int(ocr_enabled) + (
            len(first_queries) if self.search_tool is not None else 0
        )
        if phase_workers:
            with ThreadPoolExecutor(max_workers=phase_workers) as executor:
                ocr_future = (
                    executor.submit(
                        self.ocr_tool.run,
                        image_content=image_content,
                        image_url=image_url,
                        language=language,
                        focus=text[:300],
                    )
                    if ocr_enabled
                    else None
                )
                search_futures = [
                    (
                        query,
                        executor.submit(
                            self.search_tool.run,
                            query["query"],
                            count=max(1, min(10, int(self.config.mmd_search_results_per_query))),
                        ),
                    )
                    for query in first_queries
                    if self.search_tool is not None and query.get("query")
                ]
                if ocr_future is not None:
                    try:
                        value = ocr_future.result()
                        ocr_result = value if isinstance(value, dict) else {"status": "error"}
                    except Exception as exc:
                        ocr_result = {"status": "error", "error": str(exc)}
                for query, future in search_futures:
                    try:
                        value = future.result()
                        result = value if isinstance(value, dict) else {"status": "error", "results": []}
                    except Exception as exc:
                        result = {"status": "error", "results": [], "images": [], "error": str(exc)}
                    search_batches.append((query, result))

        if query_limit >= 3 and self.search_tool is not None and ocr_enabled:
            third_query = self.query_builder.build_crossmodal_query(text, ocr_result, question)
            try:
                third_result = self.search_tool.run(
                    third_query["query"],
                    count=max(1, min(10, int(self.config.mmd_search_results_per_query))),
                )
                if not isinstance(third_result, dict):
                    third_result = {"status": "error", "results": [], "images": []}
            except Exception as exc:
                third_result = {"status": "error", "results": [], "images": [], "error": str(exc)}
            search_batches.append((third_query, third_result))

        compact_ocr = self._compact_ocr_result(ocr_result)
        compact_search = self._compact_search_results(search_batches)
        ocr_seconds = self._bounded_nonnegative_float(
            ocr_result.get("timings", {}).get("total_seconds") if isinstance(ocr_result, dict) else 0
        )
        search_seconds = round(
            sum(
                self._bounded_nonnegative_float(result.get("timings", {}).get("total_seconds"))
                for _, result in search_batches
                if isinstance(result, dict)
            ),
            4,
        )
        model_calls = int(
            self._bounded_nonnegative_float(
                ocr_result.get("timings", {}).get("model_calls")
                if isinstance(ocr_result, dict)
                else 0
            )
        )
        return {
            "enabled": True,
            "ocr": compact_ocr,
            "search": compact_search,
            "timings": {
                "ocr_seconds": ocr_seconds,
                "search_seconds": search_seconds,
                "enrichment_wall_seconds": round(perf_counter() - started_at, 4),
                "model_calls": model_calls,
            },
        }

    def _compact_ocr_result(self, result: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(result, dict):
            return {"status": "error", "ocr_text": "", "entities": [], "visual_objects": []}
        entities = []
        for item in result.get("entities", []) if isinstance(result.get("entities"), list) else []:
            if not isinstance(item, dict):
                continue
            entities.append(
                {
                    "name": str(item.get("name") or "")[:80],
                    "type": str(item.get("type") or "other")[:30],
                    "evidence": str(item.get("evidence") or "")[:160],
                }
            )
        return {
            "status": str(result.get("status") or "error"),
            "ocr_text": str(result.get("ocr_text") or "")[:1000],
            "entities": entities[:20],
            "visual_objects": [str(item)[:100] for item in result.get("visual_objects", [])[:15]]
            if isinstance(result.get("visual_objects"), list)
            else [],
            "limitations": [str(item)[:160] for item in result.get("limitations", [])[:5]]
            if isinstance(result.get("limitations"), list)
            else [],
            **({"error": str(result.get("error"))[:300]} if result.get("error") else {}),
        }

    def _compact_search_results(
        self,
        batches: list[tuple[dict[str, str], dict[str, Any]]],
    ) -> dict[str, Any]:
        queries = []
        results = []
        images = []
        limitations = []
        seen_urls = set()
        seen_fingerprints: list[str] = []
        seen_image_urls = set()
        domain_counts: dict[str, int] = {}
        total_limit = max(1, int(self.config.mmd_search_total_results))
        snippet_limit = max(80, int(self.config.mmd_search_snippet_characters))

        for query, batch in batches:
            query_id = str(query.get("query_id") or f"Q{len(queries) + 1}")
            queries.append(
                {
                    "query_id": query_id,
                    "strategy": str(query.get("strategy") or "whole_post"),
                    "query": str(query.get("query") or "")[:160],
                    "status": str(batch.get("status") or "error"),
                }
            )
            for note in batch.get("limitations", []) if isinstance(batch.get("limitations"), list) else []:
                text = str(note or "").strip()
                if text and text not in limitations:
                    limitations.append(text[:200])
            for item in batch.get("results", []) if isinstance(batch.get("results"), list) else []:
                if len(results) >= total_limit:
                    break
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url or url in seen_urls:
                    continue
                title = str(item.get("title") or "")[:180]
                snippet = str(item.get("snippet") or "")[:snippet_limit]
                fingerprint = self._search_fingerprint(title, snippet)
                if fingerprint and any(
                    SequenceMatcher(None, fingerprint, previous).ratio() >= 0.82
                    for previous in seen_fingerprints
                ):
                    continue
                domain = urlparse(url).netloc.lower().removeprefix("www.")
                if domain and domain_counts.get(domain, 0) >= 2:
                    continue
                seen_urls.add(url)
                if fingerprint:
                    seen_fingerprints.append(fingerprint)
                if domain:
                    domain_counts[domain] = domain_counts.get(domain, 0) + 1
                results.append(
                    {
                        "evidence_id": f"S{len(results) + 1}",
                        "query_id": query_id,
                        "title": title,
                        "url": url[:1000],
                        "domain": domain,
                        "site_name": str(item.get("site_name") or "")[:100],
                        "snippet": snippet,
                        "source_tier": self._source_tier(domain),
                        "date_last_crawled": str(item.get("date_last_crawled") or "")[:60],
                    }
                )
                if len(results) >= total_limit:
                    break
            for item in batch.get("images", []) if isinstance(batch.get("images"), list) else []:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if not url or url in seen_image_urls:
                    continue
                seen_image_urls.add(url)
                images.append(
                    {
                        "query_id": query_id,
                        "name": str(item.get("name") or "")[:120],
                        "url": url[:1000],
                        "thumbnail_url": str(item.get("thumbnail_url") or "")[:1000],
                    }
                )
                if len(images) >= 3:
                    break
        successful = sum(1 for item in queries if item["status"] == "success")
        return {
            "status": "success" if successful else "error" if queries else "not_run",
            "queries": queries,
            "results": results[:total_limit],
            "images": images[:3],
            "limitations": limitations[:5],
        }

    def _search_fingerprint(self, title: str, snippet: str) -> str:
        normalized = re.sub(
            r"[^0-9a-z一-龥]+",
            "",
            f"{title} {snippet[:180]}".lower(),
        )
        return normalized if len(normalized) >= 20 else ""

    def _source_tier(self, domain: str) -> str:
        official_suffixes = (".gov", ".gov.cn", ".go.jp", ".gouv.fr", ".bund.de")
        established_media = {
            "apnews.com",
            "bbc.com",
            "cnn.com",
            "dw.com",
            "france24.com",
            "news.cn",
            "nhk.or.jp",
            "reuters.com",
            "theguardian.com",
        }
        if any(domain == suffix.lstrip(".") or domain.endswith(suffix) for suffix in official_suffixes):
            return "official_candidate"
        if domain in established_media or any(domain.endswith(f".{item}") for item in established_media):
            return "established_media"
        return "other"

    def _bounded_nonnegative_float(self, value: Any) -> float:
        try:
            return round(max(0.0, float(value or 0)), 4)
        except (TypeError, ValueError):
            return 0.0

    def _call_model(
        self,
        prompt: str,
        image_parts: list[dict[str, Any]],
        image_labels: list[str],
        image_content: bytes | None,
        image_url: str | None,
        language: str,
    ) -> tuple[str, dict[str, Any]]:
        if self.evaluator:
            return self.evaluator(prompt, image_content, image_url, language), {
                "thinking_enabled": self.config.mmd_enable_thinking,
                "thinking_characters": 0,
                "finish_reason": "evaluator",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "reasoning_tokens": 0,
            }
        client = self.model_factory.qwen_vision_client(
            timeout=self.config.mmd_detection_timeout_seconds,
            max_retries=0,
        )
        user_content = []
        for index, image_part in enumerate(image_parts):
            label = image_labels[index] if index < len(image_labels) else f"Image {index + 1}"
            user_content.extend(
                [
                    {"type": "text", "text": f"[{label}]"},
                    image_part,
                ]
            )
        user_content.append({"type": "text", "text": prompt})
        extra_body: dict[str, Any] = {"enable_thinking": self.config.mmd_enable_thinking}
        if self.config.mmd_enable_thinking:
            extra_body["thinking_budget"] = max(1, int(self.config.mmd_thinking_budget))
        completion = client.chat.completions.create(
            model=self.config.mmd_vision_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是惊堂木多模态虚假信息检测器，只输出符合要求的 JSON。"
                        if is_zh(language)
                        else "You are Jingtangmu's multimodal misinformation detector. Output valid JSON only."
                    ),
                },
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=self.config.mmd_max_tokens,
            extra_body=extra_body,
        )
        choice = completion.choices[0]
        message = choice.message
        usage = getattr(completion, "usage", None)
        completion_details = getattr(usage, "completion_tokens_details", None)
        reasoning_content = getattr(message, "reasoning_content", "") or ""
        return message.content or "", {
            "thinking_enabled": self.config.mmd_enable_thinking,
            "thinking_characters": len(reasoning_content),
            "finish_reason": str(getattr(choice, "finish_reason", "") or ""),
            "prompt_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
            "completion_tokens": int(getattr(usage, "completion_tokens", 0) or 0),
            "reasoning_tokens": int(getattr(completion_details, "reasoning_tokens", 0) or 0),
        }

    def _normalize_image_labels(
        self,
        labels: list[str] | tuple[str, ...] | None,
        image_count: int,
    ) -> list[str]:
        normalized = []
        for index in range(image_count):
            value = str(labels[index]).strip() if labels and index < len(labels) else ""
            if not re.fullmatch(r"Image\s+[1-9]\d*", value, flags=re.IGNORECASE):
                value = f"Image {index + 1}"
            else:
                value = f"Image {int(value.split()[-1])}"
            normalized.append(value)
        return normalized

    def _parse_json(self, raw: str) -> dict[str, Any]:
        cleaned = str(raw or "").strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        candidates = [cleaned]
        repaired = re.sub(r",\s*([}\]])", r"\1", cleaned)
        repaired = re.sub(r"([}\]])\s*(?=\"[^\"]+\"\s*:)", r"\1,", repaired)
        if repaired != cleaned:
            candidates.append(repaired)
        value = None
        last_error: Exception | None = None
        for candidate in candidates:
            try:
                value = json.loads(candidate)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
                start = candidate.find("{")
                if start >= 0:
                    try:
                        value, _ = json.JSONDecoder().raw_decode(candidate[start:])
                        break
                    except json.JSONDecodeError as nested_exc:
                        last_error = nested_exc
        if value is None:
            raise ValueError(f"MMD model did not return valid JSON: {last_error}")
        if not isinstance(value, dict):
            raise ValueError("MMD model output must be a JSON object")
        return value

    def _normalize_result(
        self,
        parsed: dict[str, Any],
        language: str,
        *,
        external_search_available: bool = False,
        has_image: bool = True,
    ) -> dict[str, Any]:
        model_verdict = self._normalize_verdict(parsed.get("verdict"))
        if model_verdict is None:
            raise ValueError("MMD verdict is missing or invalid")
        raw_confidence = self._bounded_float(parsed.get("confidence"), 0.5)
        decision_status = str(parsed.get("decision_status") or "").strip()
        if decision_status not in {"confident", "low_confidence", "insufficient_evidence"}:
            decision_status = "confident" if raw_confidence >= 0.7 else "low_confidence"

        source_signals = (
            parsed.get("fine_grained_checks")
            if isinstance(parsed.get("fine_grained_checks"), dict)
            else parsed.get("signals")
            if isinstance(parsed.get("signals"), dict)
            else {}
        )
        normalized_signals = [
            (key, self._normalize_signal(source_signals.get(key)))
            for key in self.SIGNAL_KEYS
        ]
        signals = dict(normalized_signals)
        claim_assessments = self._normalize_claim_assessments(parsed.get("claim_assessments"))
        atomic_points = self._normalize_atomic_points(parsed.get("atomic_points"), claim_assessments)
        if not has_image:
            unavailable = "未提供图片。" if is_zh(language) else "No image was supplied."
            for key in self.SIGNAL_KEYS:
                if not key.startswith(("V", "C")):
                    continue
                signals[key] = {
                    "assessment": "not_verifiable",
                    "present": False,
                    "severity": 0,
                    "evidence_basis": "none",
                    "analysis": unavailable,
                    "evidence": unavailable,
                }
            for item in atomic_points:
                item["visual_observation"] = ""
                item["point_type"] = "factual_claim"
            for item in claim_assessments:
                if item["evidence_basis"] in {"visual_input", "ocr_output"}:
                    item["status"] = "not_verifiable"
                    item["evidence_basis"] = "none"
                    item["evidence"] = unavailable
            crossmodal_assessment = {
                "label": "Cannot_Determine",
                "evidence_basis": "none",
                "reason": unavailable,
            }
        else:
            crossmodal_assessment = self._normalize_crossmodal_assessment(
                parsed.get("crossmodal_assessment"),
                signals,
                language,
            )
            self._align_crossmodal_signal(signals, crossmodal_assessment)
        verdict, verdict_adjustments = self._calibrate_verdict(model_verdict, claim_assessments, signals)
        evidence_coverage, model_knowledge_only = self._evidence_coverage(claim_assessments, signals)
        decision_status = self._calibrate_decision_status(
            decision_status,
            evidence_coverage,
            bool(verdict_adjustments),
        )
        confidence = self._calibrate_confidence(
            raw_confidence,
            decision_status,
            evidence_coverage,
            model_knowledge_only,
        )
        explanation = self._normalize_explanation(
            parsed.get("explanation")
            if isinstance(parsed.get("explanation"), dict)
            else parsed.get("user_explanation"),
            verdict,
            decision_status,
            signals,
            claim_assessments,
            language,
            verdict_adjustments=verdict_adjustments,
            external_search_available=external_search_available,
        )
        return {
            "verdict": verdict,
            "confidence": confidence,
            "decision_status": decision_status,
            "atomic_points": atomic_points,
            "claim_assessments": claim_assessments,
            "crossmodal_assessment": crossmodal_assessment,
            "fine_grained_checks": signals,
            "explanation": explanation,
            "diagnostics": {
                "model_verdict": model_verdict,
                "verdict_adjustments": verdict_adjustments,
                "evidence_coverage": evidence_coverage,
                "core_supported": sum(
                    item["importance"] == "core" and item["status"] == "supported"
                    for item in claim_assessments
                ),
                "core_contradicted": sum(
                    item["importance"] == "core" and item["status"] == "contradicted"
                    for item in claim_assessments
                ),
                "supporting_contradicted": sum(
                    item["importance"] == "supporting" and item["status"] == "contradicted"
                    for item in claim_assessments
                ),
            },
        }

    def _normalize_verdict(self, value: Any) -> str | None:
        text = str(value or "").strip()
        if text in self.VERDICTS:
            return text
        return self.VERDICT_ALIASES.get(text.lower())

    def _normalize_signal(self, value: Any) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        assessment = str(source.get("assessment") or "").strip().lower()
        if assessment not in self.ASSESSMENTS:
            present_value = source.get("present")
            if present_value is True:
                assessment = "confirmed"
            elif present_value is False:
                assessment = "absent"
            else:
                assessment = "not_verifiable"
        severity = self._bounded_int(source.get("severity"), 0, 3)
        if assessment in {"absent", "not_verifiable"}:
            severity = 0
        elif severity == 0:
            severity = 2 if assessment == "confirmed" else 1
        evidence_basis = str(source.get("evidence_basis") or "none").strip().lower()
        if evidence_basis not in self.EVIDENCE_BASES:
            evidence_basis = "none"
        result: dict[str, Any] = {
            "assessment": assessment,
            "present": assessment in {"confirmed", "suspected"},
            "severity": severity,
            "evidence_basis": evidence_basis,
            "analysis": str(source.get("analysis") or source.get("evidence") or "").strip(),
            "evidence": str(source.get("analysis") or source.get("evidence") or "").strip(),
        }
        regions = source.get("regions")
        if isinstance(regions, list):
            result["regions"] = [item for item in regions if isinstance(item, (dict, str))][:10]
        return result

    def _normalize_claim_assessments(self, value: Any) -> list[dict[str, Any]]:
        assessments = []
        for item in value if isinstance(value, list) else []:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue
            status = str(item.get("status") or "not_verifiable").strip().lower()
            if status not in self.CLAIM_STATUSES:
                status = "not_verifiable"
            basis = str(item.get("evidence_basis") or "none").strip().lower()
            if basis not in self.EVIDENCE_BASES:
                basis = "none"
            importance = str(item.get("importance") or "core").strip().lower()
            if importance not in {"core", "supporting"}:
                importance = "core"
            assessments.append(
                {
                    "claim": claim,
                    "importance": importance,
                    "status": status,
                    "evidence_basis": basis,
                    "evidence": str(item.get("evidence") or "").strip(),
                }
            )
        return assessments[:3]

    def _normalize_atomic_points(
        self,
        value: Any,
        legacy_claims: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        points: list[dict[str, str]] = []
        allowed_types = {"factual_claim", "visual_claim", "crossmodal_claim"}
        for item in value if isinstance(value, list) else []:
            if not isinstance(item, dict):
                continue
            claim = str(item.get("claim") or "").strip()
            if not claim:
                continue
            point_type = str(item.get("point_type") or "factual_claim").strip().lower()
            if point_type not in allowed_types:
                point_type = "factual_claim"
            points.append(
                {
                    "claim": claim,
                    "visual_observation": str(item.get("visual_observation") or "").strip(),
                    "search_query": str(item.get("search_query") or "").strip(),
                    "point_type": point_type,
                }
            )
        if points:
            return points[:3]
        for item in legacy_claims:
            points.append(
                {
                    "claim": str(item.get("claim") or "").strip(),
                    "visual_observation": (
                        str(item.get("evidence") or "").strip()
                        if item.get("evidence_basis") == "visual_input"
                        else ""
                    ),
                    "search_query": "",
                    "point_type": "factual_claim",
                }
            )
        return points[:3]

    def _normalize_crossmodal_assessment(
        self,
        value: Any,
        signals: dict[str, dict[str, Any]],
        language: str,
    ) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        label = str(source.get("label") or "").strip().replace(" ", "_")
        aliases = {
            "PartiallyConsistent": "Partially_Consistent",
            "CannotDetermine": "Cannot_Determine",
        }
        label = aliases.get(label, label)
        basis = str(source.get("evidence_basis") or "none").strip().lower()
        if basis not in {"visual_input", "ocr_output", "external_search", "none"}:
            basis = "none"
        reason = str(source.get("reason") or source.get("reasoning") or "").strip()

        crossmodal_signals = [
            item for key, item in signals.items()
            if key.startswith("C") and isinstance(item, dict)
        ]
        confirmed = [
            item for item in crossmodal_signals
            if item.get("assessment") == "confirmed" and int(item.get("severity") or 0) >= 2
        ]
        suspected = [
            item for item in crossmodal_signals
            if item.get("assessment") == "suspected"
            or (item.get("assessment") == "confirmed" and int(item.get("severity") or 0) == 1)
        ]
        if confirmed:
            label = "Inconsistent"
            for candidate_basis in ("visual_input", "ocr_output", "external_search"):
                if any(item.get("evidence_basis") == candidate_basis for item in confirmed):
                    basis = candidate_basis
                    break
            reason = reason or str(confirmed[0].get("evidence") or "").strip()
        elif suspected:
            label = "Partially_Consistent"
            reason = reason or str(suspected[0].get("evidence") or "").strip()
        elif label not in self.CROSSMODAL_LABELS:
            label = "Cannot_Determine"

        # An explicit, visible match is required for Consistent. Missing or generic
        # image evidence must not silently become a perfect consistency judgment.
        if label == "Consistent" and basis not in {"visual_input", "ocr_output"}:
            label = "Cannot_Determine"
        if not reason:
            reason = (
                "当前单张配图没有足够可见信息确认核心图文关系。"
                if is_zh(language)
                else "The selected image lacks enough visible information to confirm the core image-text relation."
            )
        return {"label": label, "evidence_basis": basis, "reason": reason}

    def _align_crossmodal_signal(
        self,
        signals: dict[str, dict[str, Any]],
        crossmodal: dict[str, Any],
    ) -> None:
        if any(key.startswith("C") for key in signals):
            return
        label = crossmodal.get("label")
        if label not in {"Inconsistent", "Partially_Consistent"}:
            return
        assessment = "confirmed" if label == "Inconsistent" else "suspected"
        signals["C1_semantic_inconsistency"] = {
            "assessment": assessment,
            "present": True,
            "severity": 2 if assessment == "confirmed" else 1,
            "evidence_basis": crossmodal.get("evidence_basis") or "none",
            "evidence": crossmodal.get("reason") or "",
        }

    def _calibrate_verdict(
        self,
        model_verdict: str,
        claims: list[dict[str, Any]],
        signals: dict[str, dict[str, Any]],
    ) -> tuple[str, list[str]]:
        core_contradictions = [
            item for item in claims
            if item["importance"] == "core" and item["status"] == "contradicted"
        ]
        input_core_contradictions = [
            item for item in core_contradictions
            if item["evidence_basis"]
            in {"visual_input", "ocr_output", "external_search", "text_internal"}
        ]
        model_core_contradictions = [
            item for item in core_contradictions if item["evidence_basis"] == "model_knowledge"
        ]
        confirmed_input_risks = [
            value
            for key, value in signals.items()
            if key.startswith(("T", "V", "C"))
            and value["assessment"] == "confirmed"
            and value["severity"] >= 2
            and value["evidence_basis"]
            in {"visual_input", "ocr_output", "external_search", "text_internal"}
        ]
        severe_input_risks = [item for item in confirmed_input_risks if item["severity"] >= 3]
        supporting_contradictions = [
            item
            for item in claims
            if item["importance"] == "supporting"
            and item["status"] == "contradicted"
            and item["evidence_basis"]
            in {"visual_input", "ocr_output", "external_search", "text_internal"}
        ]
        minor_label_risks = [
            value
            for key, value in signals.items()
            if key in {
                "T2_distortion",
                "V2_visual_editing",
                "C1_semantic_inconsistency",
                "C2_contextual_inconsistency",
            }
            and value["assessment"] == "confirmed"
            and 1 <= value["severity"] <= 2
            and value["evidence_basis"]
            in {"visual_input", "ocr_output", "external_search", "text_internal"}
        ]
        adjustments: list[str] = []
        verdict = model_verdict

        if verdict == "False" and not (
            len(input_core_contradictions) >= 2
            or (input_core_contradictions and severe_input_risks)
            or len(severe_input_risks) >= 2
        ):
            verdict = "Mostly False" if model_core_contradictions or confirmed_input_risks else "Mixture"
            adjustments.append("false_requires_input_grounded_core_conflict")
        if verdict == "Mostly False" and not (
            core_contradictions or confirmed_input_risks
        ):
            verdict = "Mixture"
            adjustments.append("mostly_false_requires_confirmed_conflict")
        if verdict == "True" and (input_core_contradictions or confirmed_input_risks):
            verdict = "Mixture" if severe_input_risks else "Mostly True"
            adjustments.append("true_conflicts_with_confirmed_risk")
        elif verdict == "True" and (supporting_contradictions or minor_label_risks):
            verdict = "Mostly True"
            adjustments.append("true_has_grounded_minor_issue")
        elif verdict == "Mostly True" and (input_core_contradictions or severe_input_risks):
            verdict = "Mixture"
            adjustments.append("mostly_true_conflicts_with_confirmed_risk")
        return verdict, adjustments

    def _evidence_coverage(
        self,
        claims: list[dict[str, Any]],
        signals: dict[str, dict[str, Any]],
    ) -> tuple[float, bool]:
        basis_weights = {
            "visual_input": 1.0,
            "ocr_output": 0.9,
            "external_search": 1.0,
            "model_knowledge": 0.55,
            "text_internal": 0.35,
            "none": 0.0,
        }
        evidence_bases: list[str] = []
        if claims:
            total_weight = 0.0
            covered_weight = 0.0
            for item in claims:
                importance_weight = 2.0 if item["importance"] == "core" else 1.0
                total_weight += importance_weight
                if item["status"] != "not_verifiable":
                    basis = item["evidence_basis"]
                    evidence_bases.append(basis)
                    covered_weight += importance_weight * basis_weights[basis]
            coverage = covered_weight / total_weight if total_weight else 0.0
        else:
            usable = [
                item for item in signals.values()
                if item["assessment"] in {"confirmed", "suspected"} and item["evidence"]
            ]
            for item in usable:
                evidence_bases.append(item["evidence_basis"])
            coverage = min(1.0, sum(basis_weights[item["evidence_basis"]] for item in usable) / 3.0)
        model_knowledge_only = bool(evidence_bases) and all(
            basis in {"model_knowledge", "text_internal", "none"} for basis in evidence_bases
        )
        return round(min(1.0, max(0.0, coverage)), 4), model_knowledge_only

    def _calibrate_decision_status(
        self,
        model_status: str,
        evidence_coverage: float,
        verdict_adjusted: bool,
    ) -> str:
        if evidence_coverage < 0.2:
            return "insufficient_evidence"
        if evidence_coverage < 0.6 or verdict_adjusted:
            return "low_confidence"
        return model_status

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        decision_status: str,
        evidence_coverage: float,
        model_knowledge_only: bool,
    ) -> float:
        status_cap = {
            "confident": 0.82,
            "low_confidence": 0.58,
            "insufficient_evidence": 0.38,
        }[decision_status]
        calibrated = raw_confidence * (0.35 + 0.45 * evidence_coverage)
        if model_knowledge_only:
            calibrated = min(calibrated, 0.65)
        return round(min(status_cap, max(0.05, calibrated)), 4)

    def _normalize_explanation(
        self,
        value: Any,
        verdict: str,
        decision_status: str,
        signals: dict[str, dict[str, Any]],
        claim_assessments: list[dict[str, Any]],
        language: str,
        *,
        verdict_adjustments: list[str],
        external_search_available: bool = False,
    ) -> dict[str, Any]:
        source = value if isinstance(value, dict) else {}
        findings = self._structured_findings(claim_assessments, signals, language)

        label = self.LABELS_ZH[verdict] if is_zh(language) else verdict
        verdict_statement = (
            f"该内容整体判断为{label}。"
            if is_zh(language)
            else f"The content is assessed as {label}."
        )
        analysis = str(source.get("analysis") or source.get("summary") or "").strip()
        if not analysis:
            analysis = self._fallback_summary(verdict, findings, language)
        if verdict_adjustments:
            boundary = (
                "部分线索仅来自模型已有知识、可疑迹象或无法验证的信息，未按明确冲突处理。"
                if is_zh(language)
                else "Some clues rely only on model knowledge, suspicion, or unverifiable information and were not treated as confirmed conflicts."
            )
            analysis = f"{analysis} {boundary}".strip()
        limitations = self._string_list(source.get("limitations"))
        bases = {
            item.get("evidence_basis")
            for item in claim_assessments
            if item.get("status") != "not_verifiable"
        }
        bases.update(
            item.get("evidence_basis")
            for item in signals.values()
            if item.get("assessment") in {"confirmed", "suspected"}
        )
        if "model_knowledge" in bases:
            limitations.append(
                "部分判断依据来自模型已有知识，未经外部来源核验。"
                if is_zh(language)
                else "Part of the assessment relies on model knowledge that was not externally verified."
            )
        if any(item.get("status") == "not_verifiable" for item in claim_assessments):
            limitations.append(
                "部分核心或次要声明无法仅凭当前文字和图片验证。"
                if is_zh(language)
                else "Some core or secondary claims cannot be verified from the supplied text and image alone."
            )
        if decision_status == "insufficient_evidence":
            limitations.append(
                "已检索到的外部候选材料仍不足以核验关键背景。"
                if is_zh(language) and external_search_available
                else "The retrieved external candidates remain insufficient to verify key context."
                if external_search_available
                else "当前结论仅基于输入文字和图片，关键背景缺少外部来源核验。"
                if is_zh(language)
                else "The decision uses only the supplied text and image; key context was not externally verified."
            )
        return {
            "verdict_statement": verdict_statement,
            "analysis": analysis,
            "limitations": list(dict.fromkeys(limitations))[:2],
        }

    def _structured_findings(
        self,
        claims: list[dict[str, Any]],
        signals: dict[str, dict[str, Any]],
        language: str,
    ) -> list[dict[str, str]]:
        findings: list[dict[str, str]] = []
        result_names = {
            "supported": ("有支持", "Supported"),
            "contradicted": ("有冲突", "Contradicted"),
            "not_verifiable": ("无法验证", "Not verifiable"),
        }
        for claim in claims:
            evidence = str(claim.get("evidence") or "").strip()
            if not evidence:
                continue
            status = str(claim.get("status") or "not_verifiable")
            findings.append(
                {
                    "title": str(claim.get("claim") or ("核心声明" if is_zh(language) else "Core claim")),
                    "result": result_names[status][0 if is_zh(language) else 1],
                    "detail": evidence,
                }
            )
            if len(findings) >= 2:
                break
        for key, signal in signals.items():
            if len(findings) >= 3:
                break
            if signal.get("assessment") not in {"confirmed", "suspected"}:
                continue
            evidence = str(signal.get("evidence") or "").strip()
            if evidence and all(item["detail"] != evidence for item in findings):
                findings.append(
                    {
                        "title": key,
                        "result": "发现风险" if is_zh(language) else "Risk detected",
                        "detail": evidence,
                    }
                )
        return findings or self._fallback_findings(signals, language)

    def _fallback_findings(
        self,
        signals: dict[str, dict[str, Any]],
        language: str,
    ) -> list[dict[str, str]]:
        findings = []
        for key, signal in signals.items():
            if not signal["present"] or not signal["evidence"]:
                continue
            findings.append(
                {
                    "title": key,
                    "result": "发现风险" if is_zh(language) else "Risk detected",
                    "detail": signal["evidence"],
                }
            )
        if findings:
            return findings[:3]
        return [
            {
                "title": "输入分析" if is_zh(language) else "Input analysis",
                "result": "未发现明确冲突" if is_zh(language) else "No explicit conflict found",
                "detail": (
                    "当前文字和图片中没有足够明确的异常线索。"
                    if is_zh(language)
                    else "The supplied text and image do not contain a sufficiently explicit anomaly."
                ),
            }
        ]

    def _fallback_summary(
        self,
        verdict: str,
        findings: list[dict[str, str]],
        language: str,
    ) -> str:
        detail = findings[0]["detail"] if findings else ""
        label = self.LABELS_ZH[verdict] if is_zh(language) else verdict
        return (
            f"综合文字、图片及图文关系，当前判断为{label}。{detail}"
            if is_zh(language)
            else f"Considering the text, image, and their relation, the current verdict is {label}. {detail}"
        ).strip()

    def _error_result(
        self,
        message: str,
        started_at: float,
        model_calls: int = 0,
        language: str = "zh",
    ) -> dict[str, Any]:
        elapsed = round(perf_counter() - started_at, 4)
        fallback_note = (
            "检测未完整完成，因此不输出五分类标签和置信度。"
            if is_zh(language)
            else "Detection did not complete, so no five-way label or confidence is returned."
        )
        return {
            "status": "error",
            "verdict": None,
            "confidence": None,
            "decision_status": "insufficient_evidence",
            "atomic_points": [],
            "claim_assessments": [],
            "crossmodal_assessment": {
                "label": "Cannot_Determine",
                "evidence_basis": "none",
                "reason": message,
            },
            "fine_grained_checks": {
                key: self._normalize_signal(None) for key in self.SIGNAL_KEYS
            },
            "explanation": {
                "verdict_statement": (
                    "本次检测未形成有效结论。"
                    if is_zh(language)
                    else "This detection did not produce a valid conclusion."
                ),
                "analysis": fallback_note,
                "limitations": [message],
            },
            "timings": {"model_calls": model_calls, "model_seconds": elapsed, "total_seconds": elapsed},
        }

    def _bounded_float(self, value: Any, default: float) -> float:
        try:
            return round(min(1.0, max(0.0, float(value))), 4)
        except (TypeError, ValueError):
            return default

    def _bounded_int(self, value: Any, minimum: int, maximum: int) -> int:
        try:
            return min(maximum, max(minimum, int(value)))
        except (TypeError, ValueError):
            return minimum

    def _string_list(self, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        result = []
        for item in value:
            text = str(item or "").strip()
            if text and text not in result:
                result.append(text)
        return result


__all__ = ["MMDDetectionAgent"]

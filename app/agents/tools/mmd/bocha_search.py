from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

import requests

from app.agents.tools.foundation import (
    BaseTool,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
)
from app.core.config import Settings, settings


RequestCallable = Callable[..., Any]


def _tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="bocha_web_search",
            description=(
                "Search the public web for candidate evidence about a complete post or one of its broad views. "
                "Returned snippets are candidate evidence and must not be treated as verified facts by themselves."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "query": OpenAIFunctionPropertySchema(
                        type="string",
                        description="A concise query containing discriminative entities, dates, places, and events.",
                    ),
                    "count": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Number of normalized web results to return, from 1 to 10.",
                    ),
                },
                required=["query"],
            ),
            strict=False,
        ),
    )


class BoChaSearchTool(BaseTool):
    """Small, failure-isolated wrapper around Bocha web search."""

    name = "bocha_web_search"

    def __init__(
        self,
        config: Settings = settings,
        requester: RequestCallable | None = None,
    ):
        super().__init__(tool_schema=_tool_schema())
        self.config = config
        self.requester = requester or requests.post

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    def run(self, query: str, count: int = 5) -> dict[str, Any]:
        started_at = perf_counter()
        normalized_query = " ".join(str(query or "").split())[:300]
        if not normalized_query:
            return self._error("搜索词不能为空。", normalized_query, started_at)
        if not self.config.bocha_api_key:
            return self._error("未配置 BOCHA_API_KEY。", normalized_query, started_at)
        try:
            result_count = max(1, min(10, int(count or 5)))
            response = self.requester(
                self.config.bocha_search_url,
                headers={
                    "Authorization": f"Bearer {self.config.bocha_api_key}",
                    "Content-Type": "application/json",
                },
                json={"query": normalized_query, "count": result_count, "summary": True},
                timeout=self.config.search_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("Bocha response must be a JSON object")
            response_code = payload.get("code")
            if response_code not in {None, 0, 200, "0", "200"}:
                message = str(payload.get("msg") or payload.get("message") or "unknown error")
                raise ValueError(f"Bocha API error {response_code}: {message}")
            data = payload.get("data") or {}
            if not isinstance(data, dict):
                raise ValueError("Bocha response data must be a JSON object")
            web_pages_container = data.get("webPages") or {}
            images_container = data.get("images") or {}
            if not isinstance(web_pages_container, dict):
                raise ValueError("Bocha webPages must be a JSON object")
            if not isinstance(images_container, dict):
                raise ValueError("Bocha images must be a JSON object or null")
            web_pages = web_pages_container.get("value") or []
            images = images_container.get("value") or []
            results = self._normalize_pages(web_pages, result_count)
            return {
                "status": "success",
                "query": normalized_query,
                "results": results,
                "images": self._normalize_images(images, min(3, result_count)),
                "result_count": len(results),
                "limitations": [
                    "搜索摘要是候选外部证据，仍需检查来源、发布时间和正文语境。"
                ],
                "timings": {"total_seconds": round(perf_counter() - started_at, 4)},
            }
        except Exception as exc:
            result = self._error("Bocha 搜索失败。", normalized_query, started_at)
            result["error"] = str(exc)
            return result

    def _run(self, query: str) -> dict[str, Any]:
        """Compatibility entry point used by the original search tool callers."""
        return self.run(query)

    def _normalize_pages(self, value: Any, limit: int) -> list[dict[str, str]]:
        results = []
        seen_urls = set()
        for item in value if isinstance(value, list) else []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("url") or "").strip()
            if not url or url in seen_urls:
                continue
            seen_urls.add(url)
            results.append(
                {
                    "title": str(item.get("name") or "").strip(),
                    "url": url,
                    "site_name": str(item.get("siteName") or "").strip(),
                    "snippet": str(item.get("snippet") or item.get("summary") or "").strip(),
                    "date_last_crawled": str(item.get("dateLastCrawled") or "").strip(),
                }
            )
            if len(results) >= limit:
                break
        return results

    def _normalize_images(self, value: Any, limit: int) -> list[dict[str, str]]:
        results = []
        for item in value if isinstance(value, list) else []:
            if not isinstance(item, dict):
                continue
            url = str(item.get("contentUrl") or item.get("url") or "").strip()
            if not url:
                continue
            results.append(
                {
                    "name": str(item.get("name") or "").strip(),
                    "url": url,
                    "thumbnail_url": str(item.get("thumbnailUrl") or "").strip(),
                }
            )
            if len(results) >= limit:
                break
        return results

    def _error(self, message: str, query: str, started_at: float) -> dict[str, Any]:
        return {
            "status": "error",
            "query": query,
            "results": [],
            "images": [],
            "result_count": 0,
            "limitations": [message],
            "timings": {"total_seconds": round(perf_counter() - started_at, 4)},
        }


__all__ = ["BoChaSearchTool"]

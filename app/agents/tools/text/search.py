import json
from typing import Any

import requests

from app.core.config import settings
from app.core.language import is_zh
from app.agents.tools.foundation import TextTool


class BoChaSearchTool(TextTool):
    name = "BoCha Webs Search"

    def __init__(self, selected_language: str):
        super().__init__(selected_language=selected_language)
        self.description = (
            "首选搜索引擎，当需要检索信息时调用，输入是检索query。"
            if is_zh(selected_language)
            else "The main search engine is called when information needs to be retrieved. Input is a search query."
        )

    def _run(self, query: str) -> dict[str, Any]:
        payload = json.dumps({"query": query, "count": 15, "summary": True})
        headers = {
            "Authorization": f"Bearer {settings.bocha_api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(
            "https://api.bochaai.com/v1/web-search",
            headers=headers,
            data=payload,
            timeout=settings.search_timeout_seconds,
        )
        response.raise_for_status()
        result = response.json()
        final_result = result.copy()

        web_pages = final_result.get("data", {}).get("webPages", {}).get("value", [])
        final_result.setdefault("data", {}).setdefault("webPages", {})["value"] = [
            {
                "name": item.get("name", ""),
                "url": item.get("url", ""),
                "siteName": item.get("siteName", ""),
                "snippet": item.get("snippet", ""),
                "dateLastCrawled": item.get("dateLastCrawled", ""),
            }
            for item in web_pages[:3]
        ]

        images = final_result.get("data", {}).get("images", {}).get("value", [])
        final_result.setdefault("data", {}).setdefault("images", {})["value"] = images[:3]
        return final_result

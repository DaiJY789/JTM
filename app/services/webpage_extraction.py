from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup


@dataclass
class WebpageExtractionResult:
    url: str
    final_url: str
    title: str = ""
    text: str = ""
    image_url: str = ""
    images: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class WebpageExtractionError(ValueError):
    pass


@dataclass
class LimitedHtml:
    content: str
    bytes_read: int
    truncated: bool = False


class WebpageExtractionService:
    """Extract text and representative images from a public webpage."""

    def __init__(
        self,
        timeout_seconds: float = 12.0,
        max_html_bytes: int = 5_000_000,
        max_text_chars: int = 6000,
        max_images: int = 8,
    ):
        self.timeout_seconds = timeout_seconds
        self.max_html_bytes = max_html_bytes
        self.max_text_chars = max_text_chars
        self.max_images = max_images

    def extract(self, url: str) -> WebpageExtractionResult:
        normalized_url = self._normalize_url(url)
        try:
            response = requests.get(
                normalized_url,
                timeout=self.timeout_seconds,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (compatible; JingtangmuDemo/1.0; "
                        "+https://localhost)"
                    )
                },
                stream=True,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WebpageExtractionError(f"无法读取网页：{exc}") from exc

        content_type = response.headers.get("content-type", "")
        if "html" not in content_type.lower() and "text/" not in content_type.lower():
            raise WebpageExtractionError("网页 URL 未返回 HTML 或文本内容。")

        html = self._read_limited_text(response)
        soup = BeautifulSoup(html.content, "html.parser")
        final_url = response.url or normalized_url
        title = self._extract_title(soup)
        text = self._extract_text(soup)
        images = self._extract_images(soup, final_url)
        return WebpageExtractionResult(
            url=normalized_url,
            final_url=final_url,
            title=title,
            text=text,
            image_url=images[0] if images else "",
            images=images,
            metadata={
                "content_type": content_type,
                "text_length": len(text),
                "image_count": len(images),
                "html_bytes_read": html.bytes_read,
                "html_truncated": html.truncated,
                "html_limit_bytes": self.max_html_bytes,
            },
        )

    def _normalize_url(self, url: str) -> str:
        candidate = str(url or "").strip()
        if not candidate:
            raise WebpageExtractionError("网页 URL 不能为空。")
        parsed = urlparse(candidate)
        if not parsed.scheme:
            candidate = f"https://{candidate}"
            parsed = urlparse(candidate)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise WebpageExtractionError("网页 URL 必须是 http 或 https 地址。")
        return candidate

    def _read_limited_text(self, response: requests.Response) -> LimitedHtml:
        chunks = []
        total = 0
        truncated = False
        for chunk in response.iter_content(chunk_size=64_000):
            if not chunk:
                continue
            remaining = self.max_html_bytes - total
            if remaining <= 0:
                truncated = True
                break
            if len(chunk) > remaining:
                chunks.append(chunk[:remaining])
                total += remaining
                truncated = True
                break
            chunks.append(chunk)
            total += len(chunk)
        content = b"".join(chunks)
        response.encoding = response.encoding or response.apparent_encoding or "utf-8"
        return LimitedHtml(
            content=content.decode(response.encoding, errors="replace"),
            bytes_read=total,
            truncated=truncated,
        )

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for selector in [
            ('meta[property="og:title"]', "content"),
            ('meta[name="twitter:title"]', "content"),
        ]:
            value = self._meta_value(soup, *selector)
            if value:
                return value
        return " ".join((soup.title.string or "").split()) if soup.title and soup.title.string else ""

    def _extract_text(self, soup: BeautifulSoup) -> str:
        for node in soup(["script", "style", "noscript", "template", "svg"]):
            node.decompose()
        parts: list[str] = []
        for selector in ["article", "main", "body"]:
            root = soup.select_one(selector)
            if not root:
                continue
            for element in root.find_all(["h1", "h2", "h3", "p", "li", "figcaption"], recursive=True):
                text = " ".join(element.get_text(" ", strip=True).split())
                if text and text not in parts:
                    parts.append(text)
            if parts:
                break
        if not parts:
            fallback = " ".join(soup.get_text(" ", strip=True).split())
            parts = [fallback] if fallback else []
        return "\n".join(parts)[: self.max_text_chars].strip()

    def _extract_images(self, soup: BeautifulSoup, base_url: str) -> list[str]:
        candidates: list[str] = []
        for selector in [
            ('meta[property="og:image"]', "content"),
            ('meta[name="twitter:image"]', "content"),
            ('meta[property="og:image:url"]', "content"),
            ('meta[property="og:image:secure_url"]', "content"),
        ]:
            value = self._meta_value(soup, *selector)
            if value:
                candidates.append(value)
        for image in soup.find_all("img"):
            src = image.get("src") or image.get("data-src") or image.get("data-original")
            if src:
                candidates.append(src)

        output: list[str] = []
        seen = set()
        for candidate in candidates:
            absolute = urljoin(base_url, str(candidate).strip())
            parsed = urlparse(absolute)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                continue
            if absolute in seen:
                continue
            seen.add(absolute)
            output.append(absolute)
            if len(output) >= self.max_images:
                break
        return output

    def _meta_value(self, soup: BeautifulSoup, selector: str, attr: str) -> str:
        node = soup.select_one(selector)
        return " ".join(str(node.get(attr) or "").split()) if node else ""

from dataclasses import dataclass, field
import re
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
    encoding: str = "utf-8"


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
        content_root = self._select_content_root(soup)
        title = self._extract_title(soup)
        text = self._extract_text(soup, content_root)
        images = self._extract_images(soup, final_url, content_root)
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
                "encoding": html.encoding,
                "content_selector": self._content_selector(content_root),
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
        encoding = self._detect_encoding(content, response)
        return LimitedHtml(
            content=content.decode(encoding, errors="replace"),
            bytes_read=total,
            truncated=truncated,
            encoding=encoding,
        )

    def _detect_encoding(self, content: bytes, response: requests.Response) -> str:
        if content.startswith(b"\xef\xbb\xbf"):
            return "utf-8-sig"
        head = content[:8192]
        meta_match = re.search(
            br"<meta[^>]+charset\s*=\s*[\"']?\s*([a-zA-Z0-9._-]+)",
            head,
            re.I,
        )
        if not meta_match:
            meta_match = re.search(
                br"<meta[^>]+content\s*=\s*[\"'][^\"']*charset\s*=\s*([a-zA-Z0-9._-]+)",
                head,
                re.I,
            )
        if meta_match:
            return meta_match.group(1).decode("ascii", errors="ignore") or "utf-8"

        content_type = str(response.headers.get("content-type", ""))
        header_match = re.search(r"charset\s*=\s*[\"']?\s*([a-zA-Z0-9._-]+)", content_type, re.I)
        if header_match:
            return header_match.group(1)
        return str(getattr(response, "apparent_encoding", "") or "utf-8")

    def _extract_title(self, soup: BeautifulSoup) -> str:
        for selector in [
            ('meta[property="og:title"]', "content"),
            ('meta[name="twitter:title"]', "content"),
        ]:
            value = self._meta_value(soup, *selector)
            if value:
                return value
        return " ".join((soup.title.string or "").split()) if soup.title and soup.title.string else ""

    def _select_content_root(self, soup: BeautifulSoup):
        selectors = [
            "article",
            '[itemprop="articleBody"]',
            ".content_maincontent_content",
            ".article-content",
            ".article_content",
            ".article-body",
            ".article_body",
            ".news-content",
            ".news_content",
            "#article-content",
            "#articleContent",
            "main",
        ]
        for selector in selectors:
            candidates = []
            for root in soup.select(selector):
                parts = self._text_parts(root)
                score = sum(len(item) for item in parts)
                if score:
                    candidates.append((score, root))
            if candidates:
                return max(candidates, key=lambda item: item[0])[1]
        return soup.body or soup

    def _content_selector(self, root: Any) -> str:
        name = str(getattr(root, "name", "") or "")
        element_id = str(root.get("id") or "") if hasattr(root, "get") else ""
        classes = root.get("class") if hasattr(root, "get") else []
        suffix = f"#{element_id}" if element_id else ""
        if isinstance(classes, list) and classes:
            suffix += "".join(f".{item}" for item in classes[:3])
        return f"{name}{suffix}" or "document"

    def _text_parts(self, root: Any) -> list[str]:
        parts = []
        for element in root.find_all(["h1", "h2", "h3", "p", "li", "figcaption"], recursive=True):
            text = " ".join(element.get_text(" ", strip=True).split())
            if text and text not in parts:
                parts.append(text)
        return parts

    def _extract_text(self, soup: BeautifulSoup, content_root: Any | None = None) -> str:
        for node in soup(["script", "style", "noscript", "template", "svg"]):
            node.decompose()
        root = content_root or self._select_content_root(soup)
        parts = self._text_parts(root)
        if not parts:
            fallback = " ".join(root.get_text(" ", strip=True).split())
            parts = [fallback] if fallback else []
        return "\n".join(parts)[: self.max_text_chars].strip()

    def _extract_images(self, soup: BeautifulSoup, base_url: str, content_root: Any | None = None) -> list[str]:
        candidates: list[tuple[str, Any | None]] = []
        for selector in [
            ('meta[property="og:image"]', "content"),
            ('meta[name="twitter:image"]', "content"),
            ('meta[property="og:image:url"]', "content"),
            ('meta[property="og:image:secure_url"]', "content"),
        ]:
            value = self._meta_value(soup, *selector)
            if value:
                candidates.append((value, None))

        root = content_root or self._select_content_root(soup)
        content_images = list(root.find_all("img"))
        for image in [*content_images, *soup.find_all("img")]:
            src = self._image_source(image)
            if src:
                candidates.append((src, image))

        output: list[str] = []
        seen = set()
        for candidate, image in candidates:
            if image is not None and self._is_decorative_image(image, candidate):
                continue
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

    def _image_source(self, image: Any) -> str:
        for attribute in ("data-original", "data-src", "data-lazy-src", "src"):
            value = str(image.get(attribute) or "").strip()
            if value and not value.startswith("data:image/gif;base64"):
                return value
        srcset = str(image.get("srcset") or "").strip()
        if srcset:
            return srcset.split(",")[-1].strip().split()[0]
        return ""

    def _is_decorative_image(self, image: Any, source: str) -> bool:
        marker = " ".join(
            [
                str(source or ""),
                str(image.get("alt") or ""),
                str(image.get("class") or ""),
                str(image.get("id") or ""),
            ]
        ).lower()
        decorative_tokens = ("logo", "icon", "arrow", "avatar", "qrcode", "qr-code", "spacer", "sprite")
        if any(token in marker for token in decorative_tokens):
            return True
        width = self._dimension(image.get("width"))
        height = self._dimension(image.get("height"))
        return bool(width and height and (width < 200 or height < 120))

    def _dimension(self, value: Any) -> int:
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else 0

    def _meta_value(self, soup: BeautifulSoup, selector: str, attr: str) -> str:
        node = soup.select_one(selector)
        return " ".join(str(node.get(attr) or "").split()) if node else ""

from __future__ import annotations

import ipaddress
import socket
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Callable
from urllib.parse import urljoin, urlparse

import requests
from PIL import Image, UnidentifiedImageError


MAX_REMOTE_IMAGE_BYTES = 15_000_000
MAX_REDIRECTS = 3
FORMAT_DETAILS = {
    "JPEG": (".jpg", "image/jpeg"),
    "PNG": (".png", "image/png"),
    "WEBP": (".webp", "image/webp"),
    "GIF": (".gif", "image/gif"),
    "BMP": (".bmp", "image/bmp"),
}


class RemoteImageDownloadError(ValueError):
    pass


@dataclass(frozen=True)
class DownloadedRemoteImage:
    content: bytes
    suffix: str
    media_type: str
    final_url: str


class RemoteImageDownloader:
    """Download a public HTTP(S) image with size, redirect, and format checks."""

    def __init__(
        self,
        requester: Callable[..., Any] | None = None,
        host_validator: Callable[[str], bool] | None = None,
    ):
        self.requester = requester or requests.get
        self.host_validator = host_validator or self._is_public_host

    def download(self, source_url: str) -> DownloadedRemoteImage:
        current_url = source_url
        for _ in range(MAX_REDIRECTS + 1):
            self._validate_remote_url(current_url)
            parsed = urlparse(current_url)
            try:
                response = self.requester(
                    current_url,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
                        ),
                        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
                        "Referer": f"{parsed.scheme}://{parsed.netloc}/",
                    },
                    timeout=10,
                    stream=True,
                    allow_redirects=False,
                )
            except requests.RequestException as exc:
                raise RemoteImageDownloadError("远程图片读取失败。") from exc
            if response.status_code in {301, 302, 303, 307, 308}:
                location = str(response.headers.get("Location") or "").strip()
                if not location:
                    raise RemoteImageDownloadError("远程图片重定向地址无效。")
                current_url = urljoin(current_url, location)
                continue
            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise RemoteImageDownloadError("远程图片读取失败。") from exc
            content = self._read_limited_content(response)
            suffix, media_type = self._validate_image_content(content)
            return DownloadedRemoteImage(content, suffix, media_type, current_url)
        raise RemoteImageDownloadError("远程图片重定向次数过多。")

    def _read_limited_content(self, response: Any) -> bytes:
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=64_000):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_REMOTE_IMAGE_BYTES:
                raise RemoteImageDownloadError("远程图片超过大小限制。")
            chunks.append(chunk)
        if not chunks:
            raise RemoteImageDownloadError("远程图片内容为空。")
        return b"".join(chunks)

    def _validate_image_content(self, content: bytes) -> tuple[str, str]:
        try:
            with Image.open(BytesIO(content)) as image:
                image_format = str(image.format or "").upper()
                image.verify()
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise RemoteImageDownloadError("远程资源不是有效图片。") from exc
        details = FORMAT_DETAILS.get(image_format)
        if not details:
            raise RemoteImageDownloadError("远程图片格式不受支持。")
        return details

    def _validate_remote_url(self, url: str) -> None:
        parsed = urlparse(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
            raise RemoteImageDownloadError("远程图片 URL 无效。")
        if parsed.username or parsed.password:
            raise RemoteImageDownloadError("远程图片 URL 不允许包含凭据。")
        if not self.host_validator(parsed.hostname):
            raise RemoteImageDownloadError("不允许访问本地或私有网络图片。")

    def _is_public_host(self, hostname: str) -> bool:
        normalized = hostname.strip().lower().rstrip(".")
        if normalized == "localhost":
            return False
        try:
            addresses = socket.getaddrinfo(normalized, None, type=socket.SOCK_STREAM)
        except socket.gaierror:
            return False
        if not addresses:
            return False
        for address in addresses:
            try:
                ip = ipaddress.ip_address(address[4][0])
            except ValueError:
                return False
            if not ip.is_global:
                return False
        return True


__all__ = ["DownloadedRemoteImage", "RemoteImageDownloadError", "RemoteImageDownloader"]

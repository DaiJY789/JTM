from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from app.services.artifact_store import ArtifactStore
from app.services.remote_image_downloader import (
    FORMAT_DETAILS,
    RemoteImageDownloader,
    RemoteImageDownloadError,
)
from app.services.session_service import SessionService


class ConversationImagePreviewError(ValueError):
    pass


@dataclass(frozen=True)
class ConversationImagePreview:
    path: Path
    media_type: str


class ConversationImagePreviewService:
    """Fetch and cache only remote images referenced by an existing conversation."""

    def __init__(
        self,
        session_service: SessionService,
        artifact_store: ArtifactStore,
        requester: Callable[..., Any] | None = None,
        host_validator: Callable[[str], bool] | None = None,
    ):
        self.session_service = session_service
        self.artifact_store = artifact_store
        self.downloader = RemoteImageDownloader(requester, host_validator)

    def get(self, conversation_id: str, image_index: int) -> ConversationImagePreview:
        urls = self._remote_image_urls(conversation_id)
        if image_index < 0 or image_index >= len(urls):
            raise ConversationImagePreviewError("未找到该会话图片。")
        source_url = urls[image_index]
        stored = self._stored_remote_image(conversation_id, source_url)
        if stored:
            return stored
        cached = self.artifact_store.cached_remote_preview(conversation_id, source_url)
        if cached:
            return ConversationImagePreview(cached, self._media_type_from_suffix(cached.suffix))

        try:
            downloaded = self.downloader.download(source_url)
        except RemoteImageDownloadError as exc:
            raise ConversationImagePreviewError(str(exc)) from exc
        path = self.artifact_store.save_remote_preview(
            conversation_id,
            source_url,
            downloaded.content,
            downloaded.suffix,
        )
        if not path:
            raise ConversationImagePreviewError("无法保存远程图片预览。")
        return ConversationImagePreview(path, downloaded.media_type)

    def _stored_remote_image(
        self,
        conversation_id: str,
        source_url: str,
    ) -> ConversationImagePreview | None:
        latest_input = self.session_service.snapshot(conversation_id).get("latest_input") or {}
        assets = latest_input.get("image_assets")
        if not isinstance(assets, list):
            return None
        for asset in assets:
            if not isinstance(asset, dict) or asset.get("source_url") != source_url:
                continue
            stored_filename = str(asset.get("stored_filename") or "").strip()
            if not stored_filename:
                continue
            path = self.session_service.resolve_asset(
                conversation_id,
                f"images/{stored_filename}",
            )
            if not path:
                continue
            media_type = str(asset.get("content_type") or "").strip()
            return ConversationImagePreview(
                path,
                media_type or self._media_type_from_suffix(path.suffix),
            )
        return None

    def _remote_image_urls(self, conversation_id: str) -> list[str]:
        latest_input = self.session_service.snapshot(conversation_id).get("latest_input") or {}
        candidates = []
        artifacts = latest_input.get("image_artifacts")
        if isinstance(artifacts, list):
            candidates.extend(
                str(item.get("source_url") or "").strip()
                for item in artifacts
                if isinstance(item, dict)
            )
        image_urls = latest_input.get("image_urls")
        if isinstance(image_urls, list):
            candidates.extend(str(item or "").strip() for item in image_urls)
        candidates.append(str(latest_input.get("image_url") or "").strip())
        return list(
            dict.fromkeys(
                url for url in candidates if urlparse(url).scheme.lower() in {"http", "https"}
            )
        )

    def _media_type_from_suffix(self, suffix: str) -> str:
        mapping = {details[0]: details[1] for details in FORMAT_DETAILS.values()}
        return mapping.get(suffix.lower(), "application/octet-stream")


__all__ = [
    "ConversationImagePreview",
    "ConversationImagePreviewError",
    "ConversationImagePreviewService",
]

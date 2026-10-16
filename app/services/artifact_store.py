from __future__ import annotations

import hashlib
import re
from pathlib import Path
from time import time
from typing import Any
from uuid import uuid4

from app.schemas.artifacts import ArtifactRole, ArtifactType, InputArtifact


class ArtifactStore:
    """Store and resolve local artifacts under conversation-scoped directories."""

    def __init__(self, root_dir: str | Path = "conversations"):
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def save_image(
        self,
        conversation_id: str,
        content: bytes | None,
        filename: str | None = None,
        content_type: str | None = None,
        source_url: str | None = None,
        role: ArtifactRole = ArtifactRole.UPLOADED_IMAGE,
        metadata: dict[str, Any] | None = None,
    ) -> InputArtifact | None:
        if not content:
            return None
        normalized = self._safe_id(conversation_id)
        if not normalized:
            return None
        suffix = Path(filename or "").suffix.lower() or self._suffix_from_content_type(content_type)
        artifact_id = uuid4().hex
        stored_filename = f"{role.value}_{int(time() * 1000)}_{artifact_id[:8]}{suffix}"
        image_dir = self.root_dir / normalized / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / stored_filename
        image_path.write_bytes(content)
        return InputArtifact(
            id=artifact_id,
            type=ArtifactType.IMAGE,
            role=role,
            source_url=(source_url or "").strip(),
            local_path=str(image_path),
            public_url=f"/conversations/{normalized}/assets/images/{stored_filename}",
            filename=filename or stored_filename,
            content_type=content_type or "",
            size=len(content),
            metadata={
                **(metadata or {}),
                "stored_filename": stored_filename,
                "created_at": time(),
                "storage_scope": "conversation",
            },
        )

    def resolve(self, conversation_id: str, relative_path: str) -> Path | None:
        normalized = self._safe_id(conversation_id)
        if not normalized:
            return None
        safe_parts = [part for part in Path(relative_path).parts if part not in {"", ".", ".."}]
        if not safe_parts:
            return None
        root = (self.root_dir / normalized).resolve()
        candidate = root.joinpath(*safe_parts).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def artifact_to_legacy_asset(self, artifact: InputArtifact | None) -> dict[str, Any]:
        if not artifact:
            return {}
        return {
            "filename": artifact.filename,
            "stored_filename": str(artifact.metadata.get("stored_filename") or Path(artifact.local_path).name),
            "content_type": artifact.content_type,
            "size": artifact.size,
            "path": artifact.local_path,
            "url": artifact.public_url,
            "artifact_id": artifact.id,
            "source_url": artifact.source_url,
        }

    def cached_remote_preview(self, conversation_id: str, source_url: str) -> Path | None:
        normalized = self._safe_id(conversation_id)
        if not normalized or not source_url:
            return None
        image_dir = self.root_dir / normalized / "images"
        prefix = self._remote_preview_prefix(source_url)
        for suffix in (".jpg", ".png", ".webp", ".gif", ".bmp"):
            candidate = image_dir / f"{prefix}{suffix}"
            if candidate.is_file():
                return candidate
        return None

    def save_remote_preview(
        self,
        conversation_id: str,
        source_url: str,
        content: bytes,
        suffix: str,
    ) -> Path | None:
        normalized = self._safe_id(conversation_id)
        normalized_suffix = suffix.lower()
        if (
            not normalized
            or not source_url
            or not content
            or normalized_suffix not in {".jpg", ".png", ".webp", ".gif", ".bmp"}
        ):
            return None
        image_dir = self.root_dir / normalized / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        path = image_dir / f"{self._remote_preview_prefix(source_url)}{normalized_suffix}"
        if not path.exists():
            path.write_bytes(content)
        return path

    def _remote_preview_prefix(self, source_url: str) -> str:
        digest = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:24]
        return f"remote_preview_{digest}"

    def _safe_id(self, conversation_id: str | None) -> str:
        normalized = str(conversation_id or "").strip()
        return normalized if re.fullmatch(r"[A-Za-z0-9_-]{1,80}", normalized) else ""

    def _suffix_from_content_type(self, content_type: str | None) -> str:
        mapping = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
        }
        return mapping.get(str(content_type or "").lower(), ".img")

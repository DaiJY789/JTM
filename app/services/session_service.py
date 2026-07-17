from __future__ import annotations

from pathlib import Path
from typing import Any

from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore


class SessionService:
    """Coordinate conversation state and conversation-scoped artifacts."""

    def __init__(self, conversation_store: ConversationHistoryStore, artifact_store: ArtifactStore):
        self.conversation_store = conversation_store
        self.artifact_store = artifact_store

    def prepare_detection(
        self,
        conversation_id: str | None,
        request_history: list[dict[str, Any]],
    ) -> tuple[str, list[dict[str, Any]]]:
        resolved_id = self.conversation_store.ensure_id(conversation_id)
        history = [*self.conversation_store.get(resolved_id), *request_history]
        return resolved_id, history

    def append_detection(
        self,
        conversation_id: str,
        user_input: dict[str, Any],
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return self.conversation_store.append_turn(conversation_id, user_input, result)

    def append_followup(
        self,
        conversation_id: str,
        question: str,
        result: dict[str, Any],
        language: str,
        *,
        assistant_content: str | None = None,
        update_latest_result: bool = True,
    ) -> list[dict[str, Any]]:
        return self.conversation_store.append_followup_turn(
            conversation_id,
            question,
            result,
            language,
            assistant_content=assistant_content,
            update_latest_result=update_latest_result,
        )

    def latest_context(self, conversation_id: str | None) -> tuple[str, dict[str, Any]]:
        resolved_id = self.conversation_store.ensure_id(conversation_id)
        return resolved_id, self.conversation_store.latest_context(resolved_id)

    def list_conversations(self) -> list[dict[str, Any]]:
        return self.conversation_store.list_conversations()

    def snapshot(self, conversation_id: str) -> dict[str, Any]:
        return self.conversation_store.snapshot(self.conversation_store.ensure_id(conversation_id))

    def delete(self, conversation_id: str) -> bool:
        return self.conversation_store.delete(conversation_id)

    def resolve_asset(self, conversation_id: str, asset_path: str) -> Path | None:
        return self.artifact_store.resolve(conversation_id, asset_path) or self.conversation_store.asset_path(
            conversation_id, asset_path
        )

    def load_latest_image(self, conversation_id: str, latest_input: dict[str, Any]) -> bytes | None:
        image_asset = latest_input.get("image_asset") if isinstance(latest_input, dict) else {}
        if not isinstance(image_asset, dict):
            return None
        stored_filename = image_asset.get("stored_filename")
        if stored_filename:
            path = self.resolve_asset(conversation_id, f"images/{stored_filename}")
            if path:
                return path.read_bytes()
        raw_path = image_asset.get("path")
        if raw_path:
            path = Path(raw_path)
            if path.is_file():
                return path.read_bytes()
        return None

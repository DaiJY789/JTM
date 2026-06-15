from __future__ import annotations

import json
import re
import shutil
from copy import deepcopy
from pathlib import Path
from threading import Lock
from time import time
from typing import Any
from uuid import uuid4


class ConversationHistoryStore:
    """Conversation history persisted under a local conversations directory."""

    def __init__(self, max_items: int = 20, log_dir: str | Path = "conversations"):
        self.max_items = max_items
        self.log_dir = Path(log_dir)
        self._items: dict[str, list[dict[str, Any]]] = {}
        self._meta: dict[str, dict[str, Any]] = {}
        self._lock = Lock()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._load_from_disk()

    def ensure_id(self, conversation_id: str | None = None) -> str:
        normalized = self._safe_id(conversation_id)
        if normalized:
            with self._lock:
                self._items.setdefault(normalized, [])
                self._meta.setdefault(normalized, self._new_meta(normalized))
            return normalized
        new_id = uuid4().hex
        with self._lock:
            self._items[new_id] = []
            self._meta[new_id] = self._new_meta(new_id)
        return new_id

    def get(self, conversation_id: str) -> list[dict[str, Any]]:
        with self._lock:
            return deepcopy(self._items.get(conversation_id, []))

    def snapshot(self, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            history = deepcopy(self._items.get(conversation_id, []))
            meta = deepcopy(self._meta.get(conversation_id, self._new_meta(conversation_id)))
            meta["history_count"] = len(history)
            return {
                "conversation_id": conversation_id,
                "conversation_history": history,
                "conversation": meta,
                "latest_input": deepcopy(meta.get("latest_input", {})),
                "latest_result": deepcopy(meta.get("latest_result", {})),
            }

    def latest_context(self, conversation_id: str) -> dict[str, Any]:
        with self._lock:
            meta = self._meta.get(conversation_id, {})
            return {
                "conversation_id": conversation_id,
                "conversation_history": deepcopy(self._items.get(conversation_id, [])),
                "latest_input": deepcopy(meta.get("latest_input", {})),
                "latest_result": deepcopy(meta.get("latest_result", {})),
            }

    def list_conversations(self) -> list[dict[str, Any]]:
        with self._lock:
            conversations = []
            for conversation_id, meta in self._meta.items():
                item = deepcopy(meta)
                item["history_count"] = len(self._items.get(conversation_id, []))
                if self._is_meaningful(item):
                    conversations.append(item)
            conversations.sort(key=lambda item: item.get("updated_at", 0), reverse=True)
            return conversations

    def delete(self, conversation_id: str) -> bool:
        normalized = self._safe_id(conversation_id)
        if not normalized:
            return False
        with self._lock:
            existed = normalized in self._items or normalized in self._meta or self._conversation_dir(normalized).exists()
            self._items.pop(normalized, None)
            self._meta.pop(normalized, None)
            shutil.rmtree(self._conversation_dir(normalized), ignore_errors=True)
            return existed

    def save_uploaded_image(
        self,
        conversation_id: str,
        content: bytes | None,
        filename: str | None = None,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        if not content:
            return {}
        normalized = self._safe_id(conversation_id)
        if not normalized:
            return {}
        suffix = Path(filename or "").suffix.lower()
        if not suffix:
            suffix = self._suffix_from_content_type(content_type)
        safe_name = f"upload_{int(time() * 1000)}{suffix}"
        image_dir = self._conversation_dir(normalized) / "images"
        image_dir.mkdir(parents=True, exist_ok=True)
        image_path = image_dir / safe_name
        image_path.write_bytes(content)
        return {
            "filename": filename or safe_name,
            "stored_filename": safe_name,
            "content_type": content_type or "",
            "size": len(content),
            "path": str(image_path),
            "url": f"/conversations/{normalized}/assets/images/{safe_name}",
        }

    def append_turn(
        self,
        conversation_id: str,
        user_input: dict[str, Any],
        result: dict[str, Any],
    ) -> list[dict[str, Any]]:
        label = str(result.get("label") or "")
        summary = self._assistant_content(result)
        turn_items = [
            {
                "role": "user",
                "content": self._format_user_content(user_input),
                "meta": user_input,
            },
            {
                "role": "assistant",
                "content": summary,
                "label": label,
                "mode": result.get("mode"),
            },
        ]
        with self._lock:
            history = self._items.setdefault(conversation_id, [])
            self._meta.setdefault(conversation_id, self._new_meta(conversation_id))
            history.extend(turn_items)
            if len(history) > self.max_items:
                del history[: len(history) - self.max_items]
            self._meta[conversation_id]["latest_input"] = deepcopy(user_input)
            self._meta[conversation_id]["latest_result"] = deepcopy(result)
            self._update_meta(conversation_id, user_input, result, history)
            self._persist(conversation_id)
            return deepcopy(history)

    def append_followup_turn(
        self,
        conversation_id: str,
        question: str,
        result: dict[str, Any],
        language: str,
        assistant_content: str | None = None,
        update_latest_result: bool = True,
    ) -> list[dict[str, Any]]:
        user_input = {
            "mode": "followup",
            "language": language,
            "question": question,
        }
        resolved_assistant_content = assistant_content or self._assistant_content(result)
        turn_items = [
            {
                "role": "user",
                "content": question,
                "meta": user_input,
            },
            {
                "role": "assistant",
                "content": resolved_assistant_content,
                "label": result.get("label"),
                "mode": result.get("mode"),
            },
        ]
        with self._lock:
            history = self._items.setdefault(conversation_id, [])
            self._meta.setdefault(conversation_id, self._new_meta(conversation_id))
            history.extend(turn_items)
            if len(history) > self.max_items:
                del history[: len(history) - self.max_items]
            if update_latest_result:
                self._meta[conversation_id]["latest_result"] = deepcopy(result)
            self._update_meta(conversation_id, user_input, result, history)
            self._persist(conversation_id)
            return deepcopy(history)

    def _new_meta(self, conversation_id: str) -> dict[str, Any]:
        now = time()
        return {
            "conversation_id": conversation_id,
            "title": "新对话",
            "last_label": "",
            "last_summary": "",
            "created_at": now,
            "updated_at": now,
            "history_count": 0,
            "latest_input": {},
            "latest_result": {},
        }

    def _update_meta(
        self,
        conversation_id: str,
        user_input: dict[str, Any],
        result: dict[str, Any],
        history: list[dict[str, Any]],
    ) -> None:
        meta = self._meta[conversation_id]
        title = self._title_from_input(user_input)
        if title:
            meta["title"] = title
        meta["last_label"] = str(result.get("label") or "")
        meta["last_summary"] = str(result.get("summary") or "")[:160]
        meta["updated_at"] = time()
        meta["history_count"] = len(history)

    def _title_from_input(self, user_input: dict[str, Any]) -> str:
        for key in ["text", "question", "image_url"]:
            value = " ".join(str(user_input.get(key) or "").split())
            if value:
                return f"{value[:18]}..." if len(value) > 18 else value
        if user_input.get("has_image"):
            return "图片检测"
        return ""

    def _format_user_content(self, user_input: dict[str, Any]) -> str:
        parts = []
        if user_input.get("text"):
            parts.append(f"text: {user_input['text']}")
        if user_input.get("question"):
            parts.append(f"question: {user_input['question']}")
        if user_input.get("image_url"):
            parts.append(f"image_url: {user_input['image_url']}")
        if user_input.get("has_image"):
            parts.append("image: uploaded")
        return "\n".join(parts) or "empty user input"

    def _assistant_content(self, result: dict[str, Any]) -> str:
        answer = result.get("user_answer") if isinstance(result.get("user_answer"), dict) else {}
        return str(
            answer.get("reason_analysis")
            or answer.get("analysis")
            or answer.get("judgment")
            or answer.get("conclusion")
            or result.get("summary")
            or ""
        )

    def asset_path(self, conversation_id: str, relative_path: str) -> Path | None:
        normalized = self._safe_id(conversation_id)
        if not normalized:
            return None
        safe_parts = [part for part in Path(relative_path).parts if part not in {"", ".", ".."}]
        if not safe_parts:
            return None
        candidate = self._conversation_dir(normalized).joinpath(*safe_parts).resolve()
        root = self._conversation_dir(normalized).resolve()
        try:
            candidate.relative_to(root)
        except ValueError:
            return None
        return candidate if candidate.is_file() else None

    def _load_from_disk(self) -> None:
        for directory in self.log_dir.iterdir():
            if not directory.is_dir():
                continue
            conversation_id = self._safe_id(directory.name)
            if not conversation_id:
                continue
            history = self._read_json(directory / "history.json", [])
            meta = self._read_json(directory / "meta.json", self._new_meta(conversation_id))
            if not isinstance(history, list):
                history = []
            if not isinstance(meta, dict):
                meta = self._new_meta(conversation_id)
            meta["conversation_id"] = conversation_id
            meta["history_count"] = len(history)
            meta["latest_input"] = self._read_json(directory / "latest_input.json", meta.get("latest_input", {}))
            meta["latest_result"] = self._read_json(directory / "latest_result.json", meta.get("latest_result", {}))
            self._items[conversation_id] = history[-self.max_items :]
            self._meta[conversation_id] = meta

    def _persist(self, conversation_id: str) -> None:
        directory = self._conversation_dir(conversation_id)
        directory.mkdir(parents=True, exist_ok=True)
        meta = deepcopy(self._meta.get(conversation_id, self._new_meta(conversation_id)))
        history = deepcopy(self._items.get(conversation_id, []))
        meta["history_count"] = len(history)
        self._write_json(directory / "meta.json", meta)
        self._write_json(directory / "history.json", history)
        self._write_json(directory / "latest_input.json", meta.get("latest_input", {}))
        self._write_json(directory / "latest_result.json", meta.get("latest_result", {}))

    def _read_json(self, path: Path, default: Any) -> Any:
        if not path.exists():
            return deepcopy(default)
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return deepcopy(default)

    def _write_json(self, path: Path, payload: Any) -> None:
        temp_path = path.with_suffix(path.suffix + ".tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        temp_path.replace(path)

    def _conversation_dir(self, conversation_id: str) -> Path:
        return self.log_dir / self._safe_id(conversation_id)

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

    def _is_meaningful(self, item: dict[str, Any]) -> bool:
        title = str(item.get("title") or "").strip()
        return bool(
            item.get("history_count")
            or item.get("last_label")
            or item.get("last_summary")
            or item.get("latest_result")
            or (title and title not in {"新对话", "New Chat"})
        )

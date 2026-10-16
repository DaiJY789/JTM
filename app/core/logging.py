import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from app.core.config import settings


_managed_handlers: list[logging.Handler] = []


def _local_day() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d")


class DailyFileHandler(logging.FileHandler):
    """Append to one local-date file, switching files at the first event after midnight."""

    def __init__(self, configured_path: Path) -> None:
        self.configured_path = configured_path
        self.current_day = _local_day()
        super().__init__(self._path_for_day(self.current_day), mode="a", encoding="utf-8")
        os.chmod(self.baseFilename, 0o600)

    def _path_for_day(self, day: str) -> Path:
        return self.configured_path.with_name(
            f"{self.configured_path.stem}_{day}{self.configured_path.suffix or '.log'}"
        )

    def emit(self, record: logging.LogRecord) -> None:
        day = _local_day()
        if day != self.current_day:
            if self.stream is not None:
                self.stream.close()
                self.stream = None
            self.baseFilename = str(self._path_for_day(day).resolve())
            self.stream = self._open()
            os.chmod(self.baseFilename, 0o600)
            self.current_day = day
        super().emit(record)


def setup_logging(log_file: str | None = None) -> Path:
    configured_path = Path(log_file or settings.log_file)
    configured_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    io_logger = logging.getLogger("app.io")

    for handler in _managed_handlers:
        io_logger.removeHandler(handler)
        root_logger.removeHandler(handler)
        handler.close()
    _managed_handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    file_handler = DailyFileHandler(configured_path)
    log_path = Path(file_handler.baseFilename)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)

    root_logger.addHandler(file_handler)
    root_logger.addHandler(stream_handler)
    io_logger.addHandler(file_handler)
    io_logger.setLevel(logging.INFO)
    io_logger.propagate = False
    _managed_handlers.extend((file_handler, stream_handler))
    root_logger.info("Service log started: %s", log_path)
    return log_path


def log_io_event(event: str, request_id: str, endpoint: str, payload: dict[str, Any]) -> None:
    """Write one JSON record without echoing request bodies to the console."""
    record = {
        "event": event,
        "request_id": request_id,
        "endpoint": endpoint,
        "payload": payload,
    }
    logging.getLogger("app.io").info(json.dumps(record, ensure_ascii=False, default=str))


def new_request_id() -> str:
    return uuid4().hex

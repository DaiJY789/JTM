from __future__ import annotations

import argparse
from datetime import datetime
import json
import mimetypes
from pathlib import Path
import re
from typing import Any

from app.api.adapters.detection_request import adapt_detection_request
from app.api.dependencies import get_detection_orchestrator
from app.schemas.detection import DetectionTaskType
from app.services.artifact_store import ArtifactStore
from app.services.conversation_history import ConversationHistoryStore
from app.services.detection_application import DetectionApplicationService
from app.services.input_builder import DetectionInputBuilder
from app.services.session_service import SessionService
from app.services.webpage_extraction import WebpageExtractionService


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONVERSATION_ROOT = ROOT_DIR / "conversations"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "demo_history"
DEFAULT_MFC_IDS = ("001", "010", "297")
DEFAULT_IFD_IDS = ("52", "18", "61")
MFC_METADATA = (ROOT_DIR / "data" / "mfc" if (ROOT_DIR / "data" / "mfc").exists() else ROOT_DIR / "data" / "mmd") / "dataset_metadata.json"
IFD_METADATA = ROOT_DIR / "data" / "ifd" / "openfake_sample_100" / "metadata.jsonl"
MFC_RESULT_DIR = ROOT_DIR / "outputs" / "test" / "qwen3_7_flash_validation_100_v2"
IFD_RESULT_DIR = ROOT_DIR / "outputs" / "ifd_validation"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run representative MFC/IFD samples and store them as frontend histories."
    )
    parser.add_argument("--conversation-root", type=Path, default=DEFAULT_CONVERSATION_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--mfc-ids", nargs="*", default=list(DEFAULT_MFC_IDS))
    parser.add_argument("--ifd-ids", nargs="*", default=list(DEFAULT_IFD_IDS))
    parser.add_argument(
        "--reuse-existing-results",
        action="store_true",
        help="Build histories from completed evaluation outputs without new external API calls.",
    )
    parser.add_argument(
        "--run-tag",
        default=datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Suffix used in conversation ids; defaults to the current local date and time.",
    )
    return parser.parse_args(argv)


def load_mfc_samples(sample_ids: list[str]) -> list[dict[str, Any]]:
    records = json.loads(MFC_METADATA.read_text(encoding="utf-8"))
    by_id = {str(item.get("sample_id") or ""): item for item in records}
    samples = []
    for sample_id in sample_ids:
        item = by_id.get(str(sample_id))
        if not item:
            raise ValueError(f"MFC sample not found: {sample_id}")
        paths = [(MFC_METADATA.parent / value).resolve() for value in item.get("image_paths", [])]
        paths = paths[:4]
        if not paths or any(not path.is_file() for path in paths):
            raise FileNotFoundError(f"MFC sample {sample_id} has missing images")
        samples.append({**item, "sample_id": str(sample_id), "paths": paths})
    return samples


def load_ifd_samples(sample_ids: list[str]) -> list[dict[str, Any]]:
    records = []
    for line in IFD_METADATA.read_text(encoding="utf-8").splitlines():
        if line.strip():
            records.append(json.loads(line))
    by_id = {str(item.get("index")): item for item in records}
    samples = []
    for sample_id in sample_ids:
        item = by_id.get(str(sample_id))
        if not item:
            raise ValueError(f"IFD sample not found: {sample_id}")
        path = (IFD_METADATA.parent / str(item.get("image_file") or "")).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        samples.append({**item, "sample_id": str(sample_id), "path": path})
    return samples


def normalized_run_tag(value: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_-]+", "-", str(value or "").strip()).strip("-")
    if not tag:
        raise ValueError("run tag must contain at least one letter or digit")
    return tag[:40]


def image_type(path: Path) -> str:
    return mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def make_application(conversation_root: Path) -> DetectionApplicationService:
    history_store = ConversationHistoryStore(log_dir=conversation_root)
    artifact_store = ArtifactStore(conversation_root)
    session_service = SessionService(history_store, artifact_store)
    input_builder = DetectionInputBuilder(WebpageExtractionService(), artifact_store)
    return DetectionApplicationService(
        get_detection_orchestrator(),
        input_builder,
        session_service,
    )


def existing_result(result_dir: Path, sample_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    records = [
        json.loads(line)
        for line in (result_dir / "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    record = next(
        (item for item in records if str(item.get("sample_id")) == sample_id),
        None,
    )
    raw_path = result_dir / "raw" / f"{sample_id}.json"
    if not record or not raw_path.is_file():
        raise FileNotFoundError(f"completed evaluation result not found for sample {sample_id}")
    return record, json.loads(raw_path.read_text(encoding="utf-8"))


def persist_existing_result(
    application: DetectionApplicationService,
    *,
    task_type: DetectionTaskType,
    conversation_id: str,
    text: str | None,
    paths: list[Path],
    question: str,
    result: dict[str, Any],
) -> dict[str, Any]:
    application.session_service.delete(conversation_id)
    resolved_id, history = application.session_service.prepare_detection(conversation_id, [])
    contents = [path.read_bytes() for path in paths]
    task = application.input_builder.build(
        task=task_type,
        text=text,
        image_content=contents[0],
        image_filename=paths[0].name,
        image_content_type=image_type(paths[0]),
        image_url=None,
        webpage_url=None,
        question=question,
        language="zh",
        history=history,
        conversation_id=resolved_id,
        image_contents=contents,
        image_filenames=[path.name for path in paths],
        image_content_types=[image_type(path) for path in paths],
    )
    restored = dict(result)
    restored["conversation_id"] = resolved_id
    user_input = {
        **task.user_input,
        "question": question,
    }
    restored["conversation_history"] = application.session_service.append_detection(
        resolved_id,
        user_input,
        restored,
    )
    return restored


def import_mfc_sample(
    application: DetectionApplicationService,
    sample: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    evaluation, raw_result = existing_result(MFC_RESULT_DIR, sample_id)
    conversation_id = f"demo-mfc-{sample_id}-{run_tag}"
    result = persist_existing_result(
        application,
        task_type=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
        conversation_id=conversation_id,
        text=str(sample.get("text") or ""),
        paths=list(sample["paths"]),
        question=f"MFC 数据集样本 {sample_id}：请核查这组图文信息。",
        result={**raw_result, "runtime_seconds": evaluation.get("runtime_seconds")},
    )
    expected = str(sample.get("verdict") or "")
    predicted = result.get("label")
    return {
        "dataset": "MFC",
        "sample_id": sample_id,
        "conversation_id": conversation_id,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "confidence": result.get("confidence"),
        "runtime_seconds": result.get("runtime_seconds"),
        "image_count": len(sample["paths"]),
        "summary": result.get("summary", ""),
        "source": "existing_evaluation",
    }


def import_ifd_sample(
    application: DetectionApplicationService,
    sample: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    sample_id = str(sample["sample_id"])
    evaluation, raw_result = existing_result(IFD_RESULT_DIR, sample_id)
    conversation_id = f"demo-ifd-{sample_id}-{run_tag}"
    result = persist_existing_result(
        application,
        task_type=DetectionTaskType.DEEPFAKE_DETECTION,
        conversation_id=conversation_id,
        text=None,
        paths=[Path(sample["path"])],
        question=f"OpenFake 数据集样本 {sample_id}：请判断图片是真实、伪造还是合成图像。",
        result={**raw_result, "runtime_seconds": evaluation.get("runtime_seconds")},
    )
    expected = str(sample.get("label") or "").lower()
    predicted = str(result.get("label") or "")
    predicted_binary = "real" if predicted in {"真实图像", "Real_Image"} else (
        "fake" if predicted in {"伪造图像", "合成图像", "Forged_Image", "Synthetic_Image"} else None
    )
    return {
        "dataset": "OpenFake",
        "sample_id": sample_id,
        "conversation_id": conversation_id,
        "expected": expected,
        "predicted": predicted or None,
        "correct": predicted_binary == expected,
        "confidence": result.get("confidence"),
        "runtime_seconds": result.get("runtime_seconds"),
        "image_count": 1,
        "summary": result.get("summary", ""),
        "source": "existing_evaluation",
    }


def run_mfc_sample(
    application: DetectionApplicationService,
    sample: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    paths = list(sample["paths"])
    contents = [path.read_bytes() for path in paths]
    sample_id = str(sample["sample_id"])
    conversation_id = f"demo-mfc-{sample_id}-{run_tag}"
    result = application.detect(
        adapt_detection_request(
            task=DetectionTaskType.MULTIMODAL_FACT_CHECKING,
            text=str(sample.get("text") or ""),
            image_content=contents[0],
            image_filename=paths[0].name,
            image_content_type=image_type(paths[0]),
            image_url=None,
            webpage_url=None,
            question=f"MFC 数据集样本 {sample_id}：请核查这组图文信息。",
            language="zh",
            history="[]",
            conversation_id=conversation_id,
            image_contents=contents,
            image_filenames=[path.name for path in paths],
            image_content_types=[image_type(path) for path in paths],
        )
    )
    expected = str(sample.get("verdict") or "")
    predicted = result.get("label")
    return {
        "dataset": "MFC",
        "sample_id": sample_id,
        "conversation_id": conversation_id,
        "expected": expected,
        "predicted": predicted,
        "correct": predicted == expected,
        "confidence": result.get("confidence"),
        "runtime_seconds": result.get("runtime_seconds"),
        "image_count": len(paths),
        "summary": result.get("summary", ""),
    }


def run_ifd_sample(
    application: DetectionApplicationService,
    sample: dict[str, Any],
    run_tag: str,
) -> dict[str, Any]:
    path = Path(sample["path"])
    content = path.read_bytes()
    sample_id = str(sample["sample_id"])
    conversation_id = f"demo-ifd-{sample_id}-{run_tag}"
    result = application.detect(
        adapt_detection_request(
            task=DetectionTaskType.DEEPFAKE_DETECTION,
            text=None,
            image_content=content,
            image_filename=path.name,
            image_content_type=image_type(path),
            image_url=None,
            webpage_url=None,
            question=f"OpenFake 数据集样本 {sample_id}：请判断图片是真实、伪造还是合成图像。",
            language="zh",
            history="[]",
            conversation_id=conversation_id,
        )
    )
    expected_binary = str(sample.get("label") or "").lower()
    predicted = str(result.get("label") or "")
    predicted_binary = "real" if predicted in {"真实图像", "Real_Image"} else (
        "fake" if predicted in {"伪造图像", "合成图像", "Forged_Image", "Synthetic_Image"} else None
    )
    return {
        "dataset": "OpenFake",
        "sample_id": sample_id,
        "conversation_id": conversation_id,
        "expected": expected_binary,
        "predicted": predicted or None,
        "correct": predicted_binary == expected_binary,
        "confidence": result.get("confidence"),
        "runtime_seconds": result.get("runtime_seconds"),
        "image_count": 1,
        "summary": result.get("summary", ""),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_tag = normalized_run_tag(args.run_tag)
    conversation_root = args.conversation_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    application = make_application(conversation_root)
    records = []

    jobs = [
        *(('mfc', item) for item in load_mfc_samples([str(value) for value in args.mfc_ids])),
        *(('ifd', item) for item in load_ifd_samples([str(value) for value in args.ifd_ids])),
    ]
    for index, (kind, sample) in enumerate(jobs, start=1):
        sample_id = str(sample["sample_id"])
        print(f"[{index}/{len(jobs)}] running {kind.upper()} sample={sample_id}", flush=True)
        try:
            if args.reuse_existing_results:
                record = (
                    import_mfc_sample(application, sample, run_tag)
                    if kind == "mfc"
                    else import_ifd_sample(application, sample, run_tag)
                )
            else:
                record = (
                    run_mfc_sample(application, sample, run_tag)
                    if kind == "mfc"
                    else run_ifd_sample(application, sample, run_tag)
                )
        except Exception as exc:
            record = {
                "dataset": "MFC" if kind == "mfc" else "OpenFake",
                "sample_id": sample_id,
                "conversation_id": f"demo-{kind}-{sample_id}-{run_tag}",
                "expected": sample.get("verdict") if kind == "mfc" else sample.get("label"),
                "predicted": None,
                "correct": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        records.append(record)
        if not record.get("predicted"):
            application.session_service.delete(record["conversation_id"])
        print(
            f"[{index}/{len(jobs)}] conversation={record['conversation_id']} "
            f"expected={record.get('expected')} predicted={record.get('predicted')} "
            f"correct={record.get('correct')} seconds={record.get('runtime_seconds')}",
            flush=True,
        )

    report = {
        "run_tag": run_tag,
        "conversation_root": str(conversation_root),
        "created_count": sum(1 for item in records if item.get("predicted")),
        "correct_count": sum(1 for item in records if item.get("correct")),
        "records": records,
    }
    report_path = output_dir / f"demo_history_{run_tag}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved report to {report_path}", flush=True)
    return 0 if report["created_count"] == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())

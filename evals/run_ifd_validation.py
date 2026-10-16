from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import threading
import time
from typing import Any

from app.api.dependencies import get_detection_orchestrator
from app.schemas.detection import DetectionTaskType


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT_DIR / "data" / "ifd"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "ifd_validation"

DATASET_LABELS = ("real", "fake")
IFD_LABELS = ("Real_Image", "Forged_Image", "Synthetic_Image")
NO_PREDICTION_COLUMN = "None"

_REAL_ALIASES = {
    "real",
    "likely_real",
    "real_image",
    "authentic",
    "completely_real",
    "完全真实",
    "真实图像",
}
_FORGED_ALIASES = {
    "manipulated",
    "forgery_detected",
    "forged_image",
    "local_tampering",
    "tampered",
    "存在伪造",
    "局部篡改",
    "伪造图像",
}
_SYNTHETIC_ALIASES = {
    "deepfake_suspected",
    "ai_manipulated",
    "ai_generated",
    "completely_ai_generated",
    "synthetic_image",
    "完全ai生成",
    "ai生成",
    "合成图像",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the IFD workflow on an OpenFake-style image dataset."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset directory or a parent directory containing one metadata.jsonl.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional explicit metadata JSONL path. Image paths are relative to this file.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Seeded random sample size. By default all records are evaluated.",
    )
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--language", choices=("zh", "en"), default="zh")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N selected records; useful for an online smoke test.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and summarize the dataset without calling the model.",
    )
    parser.add_argument(
        "--fail-under",
        type=float,
        default=None,
        help="Exit with status 1 when overall binary accuracy is below this threshold.",
    )
    return parser.parse_args(argv)


def discover_metadata(dataset_root: Path, explicit_path: Path | None = None) -> Path:
    if explicit_path is not None:
        metadata_path = explicit_path.resolve()
        if not metadata_path.is_file():
            raise FileNotFoundError(metadata_path)
        return metadata_path

    dataset_root = dataset_root.resolve()
    direct = dataset_root / "metadata.jsonl"
    if direct.is_file():
        return direct
    candidates = sorted(dataset_root.glob("**/metadata.jsonl"))
    if not candidates:
        raise FileNotFoundError(f"No metadata.jsonl found under {dataset_root}")
    if len(candidates) > 1:
        joined = ", ".join(str(path) for path in candidates)
        raise ValueError(f"Multiple metadata files found; select one with --metadata: {joined}")
    return candidates[0].resolve()


def load_records(metadata_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for line_number, line in enumerate(
        metadata_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Metadata line {line_number} is not a JSON object")

        raw_id = payload.get("index", payload.get("id"))
        if raw_id is None:
            raise ValueError(f"Metadata line {line_number} is missing index or id")
        sample_id = str(raw_id)
        if sample_id in seen_ids:
            raise ValueError(f"Duplicate sample id {sample_id!r} at line {line_number}")
        seen_ids.add(sample_id)

        expected_label = str(payload.get("label") or "").strip().lower()
        if expected_label not in DATASET_LABELS:
            raise ValueError(
                f"Invalid label {expected_label!r} for sample {sample_id}; "
                f"expected one of {DATASET_LABELS}"
            )
        image_file = str(payload.get("image_file") or "").strip()
        if not image_file:
            raise ValueError(f"Sample {sample_id} is missing image_file")
        image_path = (metadata_path.parent / image_file).resolve()
        if not image_path.is_file():
            raise FileNotFoundError(f"Image for sample {sample_id} does not exist: {image_path}")

        records.append(
            {
                "sample_id": sample_id,
                "source_index": raw_id,
                "image_file": image_file,
                "image_absolute_path": str(image_path),
                "expected_label": expected_label,
                "prompt": payload.get("prompt"),
                "source_model": str(payload.get("model") or ""),
                "source_type": str(payload.get("type") or ""),
                "release_date": str(payload.get("release_date") or ""),
            }
        )
    if not records:
        raise ValueError(f"No records found in {metadata_path}")
    return records


def select_records(
    records: list[dict[str, Any]], sample_size: int | None, seed: int
) -> list[dict[str, Any]]:
    if sample_size is None:
        return list(records)
    if sample_size <= 0:
        raise ValueError("sample_size must be greater than zero")
    if sample_size > len(records):
        raise ValueError(
            f"sample_size={sample_size} exceeds dataset size {len(records)}"
        )
    rng = random.Random(seed)
    selected_indices = sorted(rng.sample(range(len(records)), sample_size))
    return [records[index] for index in selected_indices]


def normalize_ifd_label(value: Any) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    normalized = raw.lower().replace("-", "_").replace(" ", "_")
    if normalized in _REAL_ALIASES:
        return "Real_Image"
    if normalized in _FORGED_ALIASES:
        return "Forged_Image"
    if normalized in _SYNTHETIC_ALIASES:
        return "Synthetic_Image"
    return None


def to_binary_label(ifd_label: str | None) -> str | None:
    if ifd_label == "Real_Image":
        return "real"
    if ifd_label in {"Forged_Image", "Synthetic_Image"}:
        return "fake"
    return None


def extract_prediction(result: dict[str, Any]) -> tuple[str | None, str | None]:
    image_verdict = (
        result.get("image_verdict")
        if isinstance(result.get("image_verdict"), dict)
        else {}
    )
    ifd_label = normalize_ifd_label(
        result.get("label") or image_verdict.get("label")
    )
    return ifd_label, to_binary_label(ifd_label)


def compact_result(result: dict[str, Any]) -> dict[str, Any]:
    image_verdict = (
        result.get("image_verdict")
        if isinstance(result.get("image_verdict"), dict)
        else {}
    )
    raw = result.get("raw") if isinstance(result.get("raw"), dict) else {}
    image_forensics = (
        raw.get("image_forensics")
        if isinstance(raw.get("image_forensics"), dict)
        else {}
    )
    tool_evidence = image_verdict.get("tool_evidence")
    if not isinstance(tool_evidence, list):
        tool_evidence = []
    return {
        "final_label": result.get("label"),
        "confidence": result.get("confidence"),
        "summary": result.get("summary", ""),
        "probability": image_verdict.get("probability"),
        "tamper_score": image_verdict.get("tamper_score"),
        "ai_generated_score": image_verdict.get("ai_generated_score"),
        "forgery_types": image_verdict.get("forgery_types", []),
        "suspicious_region_count": len(
            image_verdict.get("suspicious_regions", [])
            if isinstance(image_verdict.get("suspicious_regions"), list)
            else []
        ),
        "tool_count": len(tool_evidence),
        "tools": [str(item.get("tool") or "") for item in tool_evidence if isinstance(item, dict)],
        "limitations": image_verdict.get("limitations", []),
        "visual_analysis": image_verdict.get("visual_analysis", ""),
        "timings": raw.get("timings", {}),
        "agent_tool_call_count": len(
            image_forensics.get("tool_calls", [])
            if isinstance(image_forensics.get("tool_calls"), list)
            else []
        ),
    }


def evaluate_one(
    item: dict[str, Any], language: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_at = time.perf_counter()
    try:
        image_content = Path(item["image_absolute_path"]).read_bytes()
        question = (
            "请判断这张图片是真实图像、伪造图像还是合成图像。"
            if language == "zh"
            else "Determine whether this image is real, forged, or synthetic."
        )
        result = get_detection_orchestrator().detect(
            task_type=DetectionTaskType.DEEPFAKE_DETECTION,
            image_content=image_content,
            question=question,
            language=language,
        )
        ifd_label, predicted_label = extract_prediction(result)
        status = "completed" if predicted_label in DATASET_LABELS else "abstained"
        record = {
            **item,
            "status": status,
            "system_ifd_label": ifd_label,
            "predicted_label": predicted_label,
            "correct": predicted_label == item["expected_label"],
            "runtime_seconds": round(time.perf_counter() - started_at, 3),
            "prediction": compact_result(result),
            "error": "" if status == "completed" else str(result.get("summary") or "Unmapped IFD label"),
        }
        return record, result
    except Exception as exc:
        record = {
            **item,
            "status": "error",
            "system_ifd_label": None,
            "predicted_label": None,
            "correct": False,
            "runtime_seconds": round(time.perf_counter() - started_at, 3),
            "prediction": {},
            "error": f"{type(exc).__name__}: {exc}",
        }
        return record, {}


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    fraction = position - lower
    return values[lower] + (values[upper] - values[lower]) * fraction


def _group_accuracy(
    records: list[dict[str, Any]], field: str
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    values = sorted({str(item.get(field) or "unknown") for item in records})
    for value in values:
        group = [item for item in records if str(item.get(field) or "unknown") == value]
        correct = sum(bool(item.get("correct")) for item in group)
        output[value] = {
            "count": len(group),
            "correct": correct,
            "accuracy": round(correct / len(group), 4) if group else None,
        }
    return output


def summarize(
    records: list[dict[str, Any]], *, seed: int, metadata_path: Path
) -> dict[str, Any]:
    mapped = [item for item in records if item.get("predicted_label") in DATASET_LABELS]
    completed = [item for item in records if item.get("status") == "completed"]
    abstained = [item for item in records if item.get("status") == "abstained"]
    errors = [item for item in records if item.get("status") == "error"]
    correct_count = sum(bool(item.get("correct")) for item in records)

    confusion = {
        expected: {predicted: 0 for predicted in (*DATASET_LABELS, NO_PREDICTION_COLUMN)}
        for expected in DATASET_LABELS
    }
    for item in records:
        expected = item.get("expected_label")
        predicted = item.get("predicted_label")
        column = predicted if predicted in DATASET_LABELS else NO_PREDICTION_COLUMN
        if expected in confusion:
            confusion[expected][column] += 1

    per_class_metrics: dict[str, dict[str, float]] = {}
    for label in DATASET_LABELS:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[expected][label]
            for expected in DATASET_LABELS
            if expected != label
        )
        false_negative = sum(
            count for predicted, count in confusion[label].items() if predicted != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class_metrics[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }

    runtimes = sorted(float(item.get("runtime_seconds") or 0.0) for item in records)
    confidence_items: list[tuple[float, float]] = []
    for item in mapped:
        confidence = item.get("prediction", {}).get("confidence")
        if isinstance(confidence, (int, float)):
            confidence_items.append((float(confidence), float(bool(item.get("correct")))))
    brier_score = (
        sum((confidence - outcome) ** 2 for confidence, outcome in confidence_items)
        / len(confidence_items)
        if confidence_items
        else None
    )

    recalls = [per_class_metrics[label]["recall"] for label in DATASET_LABELS]
    macro_f1 = statistics.mean(
        per_class_metrics[label]["f1"] for label in DATASET_LABELS
    )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "OpenFake",
        "task": "image_forgery_detection",
        "metadata_path": str(metadata_path),
        "seed": seed,
        "evaluated_count": len(records),
        "completed_count": len(completed),
        "abstained_count": len(abstained),
        "error_count": len(errors),
        "technical_success_rate": round(
            (len(completed) + len(abstained)) / len(records), 4
        )
        if records
        else 0.0,
        "mapped_count": len(mapped),
        "coverage": round(len(mapped) / len(records), 4) if records else 0.0,
        "correct_count": correct_count,
        "binary_accuracy": round(correct_count / len(records), 4) if records else 0.0,
        "mapped_only_accuracy": round(correct_count / len(mapped), 4) if mapped else 0.0,
        "balanced_accuracy": round(statistics.mean(recalls), 4) if recalls else 0.0,
        "macro_f1": round(macro_f1, 4),
        "per_class_metrics": per_class_metrics,
        "fake_detection": per_class_metrics["fake"],
        "real_specificity": per_class_metrics["real"]["recall"],
        "expected_distribution": dict(
            Counter(str(item.get("expected_label")) for item in records)
        ),
        "predicted_binary_distribution": dict(
            Counter(str(item.get("predicted_label") or NO_PREDICTION_COLUMN) for item in records)
        ),
        "predicted_ifd_distribution": dict(
            Counter(str(item.get("system_ifd_label") or NO_PREDICTION_COLUMN) for item in records)
        ),
        "confusion_matrix": confusion,
        "accuracy_by_source_type": _group_accuracy(records, "source_type"),
        "accuracy_by_source_model": _group_accuracy(records, "source_model"),
        "confidence_calibration": {
            "sample_count": len(confidence_items),
            "brier_score": round(brier_score, 4) if brier_score is not None else None,
        },
        "runtime_seconds": {
            "mean_per_case": round(statistics.mean(runtimes), 3) if runtimes else None,
            "median_per_case": round(statistics.median(runtimes), 3) if runtimes else None,
            "min_per_case": round(min(runtimes), 3) if runtimes else None,
            "max_per_case": round(max(runtimes), 3) if runtimes else None,
            "p90_per_case": round(_percentile(runtimes, 0.90), 3) if runtimes else None,
            "p95_per_case": round(_percentile(runtimes, 0.95), 3) if runtimes else None,
            "sum_case_runtime": round(sum(runtimes), 3),
        },
        "label_mapping": {
            "Real_Image": "real",
            "Forged_Image": "fake",
            "Synthetic_Image": "fake",
        },
        "failure_policy": (
            "Exceptions count as errors. Cannot_Determine or unmapped labels count as "
            "abstentions; both remain in the overall accuracy denominator."
        ),
    }


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "expected_label",
        "system_ifd_label",
        "predicted_label",
        "correct",
        "confidence",
        "probability",
        "tamper_score",
        "ai_generated_score",
        "source_model",
        "source_type",
        "release_date",
        "runtime_seconds",
        "image_file",
        "status",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for item in records:
            prediction = item.get("prediction", {})
            writer.writerow(
                {
                    "sample_id": item.get("sample_id"),
                    "expected_label": item.get("expected_label"),
                    "system_ifd_label": item.get("system_ifd_label"),
                    "predicted_label": item.get("predicted_label"),
                    "correct": item.get("correct"),
                    "confidence": prediction.get("confidence"),
                    "probability": prediction.get("probability"),
                    "tamper_score": prediction.get("tamper_score"),
                    "ai_generated_score": prediction.get("ai_generated_score"),
                    "source_model": item.get("source_model"),
                    "source_type": item.get("source_type"),
                    "release_date": item.get("release_date"),
                    "runtime_seconds": item.get("runtime_seconds"),
                    "image_file": item.get("image_file"),
                    "status": item.get("status"),
                    "error": item.get("error"),
                }
            )


def write_report(path: Path, summary: dict[str, Any]) -> None:
    runtime = summary["runtime_seconds"]
    lines = [
        "# IFD 图像伪造检测验证报告",
        "",
        f"- 样本数：{summary['evaluated_count']}",
        f"- 完成/弃权/错误：{summary['completed_count']} / {summary['abstained_count']} / {summary['error_count']}",
        f"- 技术成功率：{summary['technical_success_rate']:.2%}",
        f"- 标签覆盖率：{summary['coverage']:.2%}",
        f"- 二分类准确率：{summary['binary_accuracy']:.2%}",
        f"- 平衡准确率：{summary['balanced_accuracy']:.2%}",
        f"- Macro-F1：{summary['macro_f1']:.2%}",
        f"- Fake Precision / Recall / F1：{summary['fake_detection']['precision']:.2%} / {summary['fake_detection']['recall']:.2%} / {summary['fake_detection']['f1']:.2%}",
        f"- Real Recall（Specificity）：{summary['real_specificity']:.2%}",
        f"- 单条平均/中位耗时：{runtime['mean_per_case']} / {runtime['median_per_case']} 秒",
        f"- P90/P95 耗时：{runtime['p90_per_case']} / {runtime['p95_per_case']} 秒",
        "",
        "## 标签映射",
        "",
        "`Real_Image → real`；`Forged_Image/Synthetic_Image → fake`。",
        "",
        "## 真实标签分布",
        "",
        f"```json\n{json.dumps(summary['expected_distribution'], ensure_ascii=False, indent=2)}\n```",
        "",
        "## IFD 三分类预测分布",
        "",
        f"```json\n{json.dumps(summary['predicted_ifd_distribution'], ensure_ascii=False, indent=2)}\n```",
        "",
        "## 二分类混淆矩阵",
        "",
        f"```json\n{json.dumps(summary['confusion_matrix'], ensure_ascii=False, indent=2)}\n```",
        "",
        "## 各类别指标",
        "",
        f"```json\n{json.dumps(summary['per_class_metrics'], ensure_ascii=False, indent=2)}\n```",
        "",
        "异常调用按错误计；`Cannot_Determine` 或无法映射的标签按弃权计。错误和弃权均保留在总体准确率分母中。",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_existing(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    output: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            output[str(item["sample_id"])] = item
    return output


def same_evaluation_input(
    existing: dict[str, Any], current: dict[str, Any]
) -> bool:
    return (
        existing.get("image_file") == current.get("image_file")
        and existing.get("expected_label") == current.get("expected_label")
        and existing.get("image_absolute_path") == current.get("image_absolute_path")
    )


def dataset_overview(records: list[dict[str, Any]], metadata_path: Path) -> dict[str, Any]:
    return {
        "metadata_path": str(metadata_path),
        "sample_count": len(records),
        "label_distribution": dict(Counter(item["expected_label"] for item in records)),
        "source_type_distribution": dict(Counter(item["source_type"] for item in records)),
        "source_model_distribution": dict(Counter(item["source_model"] for item in records)),
        "all_images_exist": all(Path(item["image_absolute_path"]).is_file() for item in records),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    metadata_path = discover_metadata(args.dataset_root, args.metadata)
    all_records = load_records(metadata_path)
    selected = select_records(all_records, args.sample_size, args.seed)
    run_manifest = selected[: args.limit] if args.limit is not None else selected
    if not run_manifest:
        raise ValueError("No samples selected for evaluation")

    if args.dry_run:
        print(
            json.dumps(dataset_overview(run_manifest, metadata_path), ensure_ascii=False, indent=2),
            flush=True,
        )
        return 0

    output_dir = args.output_dir.resolve()
    raw_dir = output_dir / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "sample_manifest.json"
    results_path = output_dir / "results.jsonl"
    summary_path = output_dir / "summary.json"
    csv_path = output_dir / "results.csv"
    report_path = output_dir / "report.md"

    manifest_path.write_text(
        json.dumps(
            {
                "dataset_root": str(args.dataset_root.resolve()),
                "metadata_path": str(metadata_path),
                "seed": args.seed,
                "sample_size": len(run_manifest),
                "language": args.language,
                "samples": run_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    if not args.resume:
        results_path.write_text("", encoding="utf-8")
    existing = read_existing(results_path) if args.resume else {}
    reusable = {
        item["sample_id"]: existing[item["sample_id"]]
        for item in run_manifest
        if item["sample_id"] in existing
        and existing[item["sample_id"]].get("status") in {"completed", "abstained"}
        and same_evaluation_input(existing[item["sample_id"]], item)
    }
    records = [
        reusable[item["sample_id"]]
        for item in run_manifest
        if item["sample_id"] in reusable
    ]
    pending = [item for item in run_manifest if item["sample_id"] not in reusable]
    workers = max(1, min(4, int(args.workers)))
    write_lock = threading.Lock()
    print(
        f"IFD validation: selected={len(run_manifest)}, pending={len(pending)}, "
        f"workers={workers}, seed={args.seed}",
        flush=True,
    )

    def save_record(record: dict[str, Any], raw_result: dict[str, Any]) -> None:
        with write_lock:
            with results_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
            if raw_result:
                (raw_dir / f"{record['sample_id']}.json").write_text(
                    json.dumps(raw_result, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

    def report_progress(index: int, record: dict[str, Any]) -> None:
        confidence = record.get("prediction", {}).get("confidence")
        confidence_text = (
            f"{float(confidence):.4f}"
            if isinstance(confidence, (int, float))
            else "None"
        )
        print(
            f"[{index}/{len(run_manifest)}] id={record['sample_id']} "
            f"expected={record['expected_label']} predicted={record['predicted_label']} "
            f"ifd={record['system_ifd_label']} confidence={confidence_text} "
            f"correct={record['correct']} seconds={record['runtime_seconds']}",
            flush=True,
        )

    completed_offset = len(records)
    if workers == 1:
        for index, item in enumerate(pending, start=completed_offset + 1):
            record, raw_result = evaluate_one(item, args.language)
            save_record(record, raw_result)
            records.append(record)
            report_progress(index, record)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(evaluate_one, item, args.language): item
                for item in pending
            }
            for index, future in enumerate(
                as_completed(futures), start=completed_offset + 1
            ):
                record, raw_result = future.result()
                save_record(record, raw_result)
                records.append(record)
                report_progress(index, record)

    by_id = {item["sample_id"]: item for item in records}
    records = [
        by_id[item["sample_id"]]
        for item in run_manifest
        if item["sample_id"] in by_id
    ]
    results_path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in records),
        encoding="utf-8",
    )
    summary = summarize(records, seed=args.seed, metadata_path=metadata_path)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_csv(csv_path, records)
    write_report(report_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved IFD validation results to {output_dir}", flush=True)

    if args.fail_under is not None and summary["binary_accuracy"] < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

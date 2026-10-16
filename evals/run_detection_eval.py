from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import requests


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SAMPLE_PATH = ROOT_DIR / "evals" / "samples" / "baseline.jsonl"
DEFAULT_OUTPUT_PATH = ROOT_DIR / "evals" / "results" / "baseline_last.json"

LABEL_FAMILIES = {
    "true": {
        "True",
        "Mostly True",
        "Completely_True",
        "Mostly_True",
        "Likely_True",
        "Consistent_True",
        "均无异常",
        "完全真实",
    },
    "false": {
        "False",
        "Mostly False",
        "Completely_False",
        "Mostly_False",
        "Likely_False",
        "Text_False",
        "文本异常",
    },
    "real_image": {
        "Real",
        "Likely_Real",
        "Real_Image",
        "真实图像",
    },
    "forged_image": {
        "Manipulated",
        "Forgery_Detected",
        "Forged_Image",
        "伪造图像",
        "存在伪造",
        "局部篡改",
    },
    "synthetic_image": {
        "AI_Generated",
        "Completely_AI_Generated",
        "Synthetic_Image",
        "合成图像",
        "AI生成",
        "完全AI生成",
    },
    "mismatch": {
        "Out_Of_Context",
        "Cross_Modal_Mismatch",
        "Image_False",
        "图像异常",
        "图文不一致",
    },
    "unknown": {
        "Mixture",
        "Cannot_Determine",
    },
}


@dataclass
class EvalSample:
    id: str
    task: str
    language: str
    input: dict[str, Any]
    expected: dict[str, Any]
    source: str = ""
    source_url: str = ""
    live_enabled: bool = True
    mock_response: dict[str, Any] | None = None


def load_samples(path: Path) -> list[EvalSample]:
    samples: list[EvalSample] = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            samples.append(
                EvalSample(
                    id=payload["id"],
                    task=payload["task"],
                    language=payload.get("language", "zh"),
                    input=payload.get("input", {}),
                    expected=payload.get("expected", {}),
                    source=payload.get("source", ""),
                    source_url=payload.get("source_url", ""),
                    live_enabled=payload.get("live_enabled", True),
                    mock_response=payload.get("mock_response"),
                )
            )
            if not samples[-1].expected.get("accepted_labels"):
                raise ValueError(f"Sample {samples[-1].id} at line {line_number} is missing expected.accepted_labels")
    return samples


def label_family(label: str | None) -> str:
    if not label:
        return "unknown"
    for family, labels in LABEL_FAMILIES.items():
        if label in labels:
            return family
    return "unknown"


def evaluate_result(sample: EvalSample, response: dict[str, Any], runtime_seconds: float | None = None) -> dict[str, Any]:
    expected = sample.expected
    actual_label = str(response.get("label") or "")
    actual_type = str(response.get("type") or "")
    expected_type = str(expected.get("type") or "")
    expected_family = str(expected.get("label_family") or label_family(expected.get("label")))
    actual_family = label_family(actual_label)
    accepted_labels = {str(label) for label in expected.get("accepted_labels", [])}
    confidence = response.get("confidence")
    min_confidence = expected.get("min_confidence")

    label_match = actual_label in accepted_labels
    family_match = actual_family == expected_family
    type_match = True if not expected_type else actual_type == expected_type
    confidence_ok = True
    if min_confidence is not None and isinstance(confidence, (int, float)):
        confidence_ok = confidence >= float(min_confidence)

    passed = label_match and type_match and confidence_ok
    return {
        "id": sample.id,
        "task": sample.task,
        "passed": passed,
        "label_match": label_match,
        "family_match": family_match,
        "type_match": type_match,
        "confidence_ok": confidence_ok,
        "expected_label": expected.get("label"),
        "accepted_labels": sorted(accepted_labels),
        "actual_label": actual_label,
        "expected_family": expected_family,
        "actual_family": actual_family,
        "expected_type": expected_type,
        "actual_type": actual_type,
        "confidence": confidence,
        "runtime_seconds": runtime_seconds,
        "summary": response.get("summary", ""),
    }


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    passed = sum(1 for item in results if item["passed"])
    label_matches = sum(1 for item in results if item["label_match"])
    family_matches = sum(1 for item in results if item["family_match"])
    by_task: dict[str, dict[str, Any]] = {}
    for item in results:
        task = item["task"]
        bucket = by_task.setdefault(task, {"total": 0, "passed": 0, "label_matches": 0, "family_matches": 0})
        bucket["total"] += 1
        bucket["passed"] += int(item["passed"])
        bucket["label_matches"] += int(item["label_match"])
        bucket["family_matches"] += int(item["family_match"])

    for bucket in by_task.values():
        bucket["pass_rate"] = round(bucket["passed"] / bucket["total"], 4) if bucket["total"] else 0
        bucket["label_accuracy"] = round(bucket["label_matches"] / bucket["total"], 4) if bucket["total"] else 0
        bucket["family_accuracy"] = round(bucket["family_matches"] / bucket["total"], 4) if bucket["total"] else 0

    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "pass_rate": round(passed / total, 4) if total else 0,
        "label_accuracy": round(label_matches / total, 4) if total else 0,
        "family_accuracy": round(family_matches / total, 4) if total else 0,
        "by_task": by_task,
    }


def mock_detector(sample: EvalSample) -> dict[str, Any]:
    if sample.mock_response is None:
        raise ValueError(f"Sample {sample.id} has no mock_response. Use --endpoint to call a live service.")
    return sample.mock_response


def endpoint_detector(endpoint: str, timeout: float) -> Callable[[EvalSample], dict[str, Any]]:
    def detect(sample: EvalSample) -> dict[str, Any]:
        payload = dict(sample.input)
        payload.pop("mode", None)
        payload.setdefault("task", sample.task)
        payload.setdefault("language", sample.language)

        files = None
        image_path = payload.pop("image_path", None)
        opened_file = None
        try:
            if image_path:
                image_file_path = Path(image_path)
                if not image_file_path.is_absolute():
                    image_file_path = ROOT_DIR / image_file_path
                opened_file = image_file_path.open("rb")
                files = {"image": (image_file_path.name, opened_file)}
            response = requests.post(endpoint, data=payload, files=files, timeout=timeout)
            response.raise_for_status()
            return response.json()
        finally:
            if opened_file:
                opened_file.close()

    return detect


def run_evaluation(samples: list[EvalSample], detector: Callable[[EvalSample], dict[str, Any]]) -> dict[str, Any]:
    results = []
    for sample in samples:
        started_at = time.perf_counter()
        response = detector(sample)
        runtime_seconds = round(time.perf_counter() - started_at, 3)
        results.append(evaluate_result(sample, response, runtime_seconds))
    return {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summarize_results(results),
        "results": results,
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline detection evaluation samples.")
    parser.add_argument("--samples", type=Path, default=DEFAULT_SAMPLE_PATH, help="JSONL sample file.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH, help="Path to write the JSON report.")
    parser.add_argument("--endpoint", default="", help="Optional live /detect endpoint. If omitted, mock_response is used.")
    parser.add_argument("--timeout", type=float, default=120.0, help="HTTP timeout per live sample.")
    parser.add_argument("--fail-under", type=float, default=None, help="Exit 1 if pass_rate is below this threshold.")
    parser.add_argument("--task", action="append", default=[], help="Only run samples for this task. Can be repeated.")
    parser.add_argument(
        "--include-non-live",
        action="store_true",
        help="Also run samples marked live_enabled=false against a live endpoint.",
    )
    parser.add_argument("--list", action="store_true", help="List sample ids and exit.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    samples = load_samples(args.samples)
    if args.task:
        selected_tasks = set(args.task)
        samples = [sample for sample in samples if sample.task in selected_tasks]
    if args.list:
        for sample in samples:
            live_flag = "live" if sample.live_enabled else "mock-only"
            print(f"{sample.id}\t{sample.task}\t{live_flag}\t{sample.expected.get('label')}")
        return 0

    skipped = []
    if args.endpoint and not args.include_non_live:
        runnable_samples = []
        for sample in samples:
            if sample.live_enabled:
                runnable_samples.append(sample)
            else:
                skipped.append({"id": sample.id, "task": sample.task, "reason": "live_enabled=false"})
        samples = runnable_samples

    detector = endpoint_detector(args.endpoint, args.timeout) if args.endpoint else mock_detector
    report = run_evaluation(samples, detector)
    report["skipped"] = skipped
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))
    print(f"Saved report: {args.output}")
    if args.fail_under is not None and report["summary"]["pass_rate"] < args.fail_under:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.run_mfc_eval import main as run_mfc_eval


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (ROOT_DIR / "data" / "mfc" if (ROOT_DIR / "data" / "mfc").exists() else ROOT_DIR / "data" / "mmd")
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "splits" / "validation.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "mfc_validation"
DEFAULT_BASELINE_SUMMARY = ROOT_DIR / "outputs" / "test" / "summary.json"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run online MFC evaluation on the fixed 100-record train validation split."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--baseline-summary",
        type=Path,
        default=DEFAULT_BASELINE_SUMMARY,
        help="Previous evaluation summary used to compare False/Inconsistent prediction ratios.",
    )
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--images-per-sample",
        type=int,
        default=4,
        help="Maximum images per sample (1-4); default uses all images up to 4.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args(argv)


def _distribution_ratio(summary: dict[str, Any], field: str, label: str) -> float:
    total = int(summary.get("evaluated_count") or 0)
    distribution = summary.get(field)
    count = int(distribution.get(label) or 0) if isinstance(distribution, dict) else 0
    return count / total if total else 0.0


def compare_with_baseline(
    current: dict[str, Any], baseline: dict[str, Any]
) -> dict[str, Any]:
    diagnostics = {}
    for name, field, label in (
        ("false_prediction_ratio", "predicted_distribution", "False"),
        ("inconsistent_crossmodal_ratio", "crossmodal_label_distribution", "Inconsistent"),
    ):
        baseline_ratio = _distribution_ratio(baseline, field, label)
        current_ratio = _distribution_ratio(current, field, label)
        diagnostics[name] = {
            "baseline": round(baseline_ratio, 4),
            "current": round(current_ratio, 4),
            "change": round(current_ratio - baseline_ratio, 4),
            "decreased": current_ratio < baseline_ratio,
        }
    runtime = current.get("runtime_seconds") or {}
    per_class = current.get("per_class_metrics") or {}
    predicted = current.get("predicted_distribution") or {}
    thresholds = {
        "technical_success_rate": {
            "value": float(current.get("technical_success_rate") or 0.0),
            "minimum": 0.98,
        },
        "exact_accuracy": {
            "value": float(current.get("exact_accuracy") or 0.0),
            "minimum": 0.40,
        },
        "macro_f1": {
            "value": float(current.get("macro_f1") or 0.0),
            "minimum": 0.35,
        },
        "minimum_class_recall": {
            "value": min(
                (float(value.get("recall") or 0.0) for value in per_class.values()),
                default=0.0,
            ),
            "minimum": 0.20,
        },
        "mostly_true_prediction_count": {
            "value": int(predicted.get("Mostly True") or 0),
            "minimum": 1,
        },
        "mean_runtime_seconds": {
            "value": runtime.get("mean_per_case"),
            "maximum": 20.0,
        },
        "p95_runtime_seconds": {
            "value": runtime.get("p95_per_case"),
            "maximum": 30.0,
        },
    }
    for item in thresholds.values():
        if not isinstance(item["value"], (int, float)):
            item["passed"] = False
            continue
        if "minimum" in item:
            item["passed"] = item["value"] >= item["minimum"]
        else:
            item["passed"] = item["value"] <= item["maximum"]
    return {
        "passed": all(item["passed"] for item in thresholds.values()),
        "criterion": "Accuracy, class coverage, technical success, and latency thresholds must all pass.",
        "note": "The baseline and validation use different source splits; this is a directional diagnostic, not a paired significance test.",
        "baseline_split": baseline.get("split"),
        "current_split": current.get("split"),
        "baseline_evaluated_count": baseline.get("evaluated_count"),
        "current_evaluated_count": current.get("evaluated_count"),
        "quality_checks": thresholds,
        "distribution_diagnostics": diagnostics,
        "current_exact_accuracy": current.get("exact_accuracy"),
        "current_macro_f1": current.get("macro_f1"),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    eval_args = [
        "--dataset-root",
        str(args.dataset_root),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--workers",
        str(args.workers),
    ]
    if args.images_per_sample != 4:
        eval_args.extend(["--images-per-sample", str(max(1, min(4, args.images_per_sample)))])
    if args.resume:
        eval_args.append("--resume")
    if args.limit is not None:
        eval_args.extend(["--limit", str(args.limit)])
    result = run_mfc_eval(eval_args)
    if result != 0:
        return result

    summary_path = args.output_dir.resolve() / "summary.json"
    baseline_path = args.baseline_summary.resolve()
    if summary_path.is_file() and baseline_path.is_file():
        current = json.loads(summary_path.read_text(encoding="utf-8"))
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        comparison = compare_with_baseline(current, baseline)
        comparison["baseline_summary"] = str(baseline_path)
        comparison_path = args.output_dir.resolve() / "validation_comparison.json"
        comparison_path.write_text(
            json.dumps(comparison, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(comparison, ensure_ascii=False, indent=2), flush=True)
        print(f"Saved validation comparison to {comparison_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

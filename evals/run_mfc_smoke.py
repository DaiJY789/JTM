from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from evals.run_mfc_eval import main as run_mfc_eval


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = (ROOT_DIR / "data" / "mfc" if (ROOT_DIR / "data" / "mfc").exists() else ROOT_DIR / "data" / "mmd")
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "splits" / "smoke10.json"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "outputs" / "mfc_smoke"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a balanced 10-record MFC smoke evaluation and check the latency target."
    )
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--images-per-sample",
        type=int,
        default=4,
        help="Maximum images per sample (1-4); default uses all images up to 4.",
    )
    parser.add_argument("--target-seconds", type=float, default=20.0)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def build_smoke_check(summary: dict[str, Any], target_seconds: float) -> dict[str, Any]:
    runtime = summary.get("runtime_seconds") if isinstance(summary.get("runtime_seconds"), dict) else {}
    mean_seconds = runtime.get("mean_per_case")
    median_seconds = runtime.get("median_per_case")
    mean_met = isinstance(mean_seconds, (int, float)) and mean_seconds <= target_seconds
    median_met = isinstance(median_seconds, (int, float)) and median_seconds <= target_seconds
    return {
        "sample_count": summary.get("evaluated_count"),
        "balanced_expected_distribution": summary.get("expected_distribution", {}),
        "latency_target_seconds": target_seconds,
        "mean_seconds": mean_seconds,
        "median_seconds": median_seconds,
        "max_seconds": runtime.get("max_per_case"),
        "mean_target_met": mean_met,
        "median_target_met": median_met,
        "latency_target_met": mean_met and median_met,
        "technical_success_rate": summary.get("technical_success_rate"),
        "exact_accuracy": summary.get("exact_accuracy"),
        "macro_f1": summary.get("macro_f1"),
        "note": "Smoke metrics are for fast regression checks only and do not replace the fixed 100-record validation run.",
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
    result = run_mfc_eval(eval_args)
    if result != 0:
        return result

    summary_path = args.output_dir.resolve() / "summary.json"
    if not summary_path.is_file():
        return 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    smoke_check = build_smoke_check(summary, args.target_seconds)
    check_path = args.output_dir.resolve() / "smoke_check.json"
    check_path.write_text(json.dumps(smoke_check, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(smoke_check, ensure_ascii=False, indent=2), flush=True)
    print(f"Saved smoke check to {check_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

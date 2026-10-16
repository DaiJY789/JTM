from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
from typing import Any

from PIL import Image


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT_DIR / "data" / "mmd"
DEFAULT_OUTPUT_DIR = DEFAULT_DATASET_ROOT / "splits"
LABELS = ("False", "Mostly False", "Mixture", "Mostly True", "True")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic prompt-dev and validation splits from MMD train.")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=20260902)
    parser.add_argument("--validation-per-label", type=int, default=20)
    return parser.parse_args(argv)


def split_train_records(
    records: list[dict[str, Any]],
    *,
    seed: int,
    validation_per_label: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_records = [item for item in records if str(item.get("split")) == "train"]
    grouped: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for item in train_records:
        verdict = str(item.get("verdict") or "")
        if verdict not in grouped:
            raise ValueError(f"unsupported train verdict: {verdict!r}")
        grouped[verdict].append(item)

    validation: list[dict[str, Any]] = []
    prompt_dev: list[dict[str, Any]] = []
    rng = random.Random(seed)
    for label in LABELS:
        group = sorted(grouped[label], key=lambda item: str(item.get("sample_id") or ""))
        if len(group) < validation_per_label:
            raise ValueError(
                f"label={label!r} has {len(group)} records, fewer than validation_per_label={validation_per_label}"
            )
        rng.shuffle(group)
        validation.extend(group[:validation_per_label])
        prompt_dev.extend(group[validation_per_label:])
    random.Random(seed + 1).shuffle(validation)
    random.Random(seed + 2).shuffle(prompt_dev)
    return prompt_dev, validation


def select_balanced_records(
    records: list[dict[str, Any]],
    *,
    per_label: int,
    seed: int,
) -> list[dict[str, Any]]:
    selected = []
    for label in LABELS:
        candidates = [item for item in records if str(item.get("verdict")) == label]
        if len(candidates) < per_label:
            raise ValueError(f"label={label!r} has fewer than {per_label} validation records")
        selected.extend(candidates[:per_label])
    random.Random(seed + 3).shuffle(selected)
    return selected


def to_fixed_sample(
    item: dict[str, Any],
    *,
    seed: int,
    dataset_root: Path,
) -> dict[str, Any]:
    image_paths = [str(path) for path in item.get("image_paths", []) if str(path).strip()]
    if not image_paths:
        raise ValueError(f"sample {item.get('sample_id')} has no image_paths")
    sample_id = str(item.get("sample_id") or "")
    compatible_images = []
    for relative_path in image_paths:
        try:
            with Image.open(dataset_root / relative_path) as image:
                if min(image.size) >= 11:
                    compatible_images.append(relative_path)
        except Exception:
            continue
    if not compatible_images:
        raise ValueError(f"sample {sample_id} has no model-compatible image")
    selected_image = random.Random(f"{seed}:{sample_id}").choice(compatible_images)
    return {
        "sample_id": sample_id,
        "source_split": "train",
        "text": str(item.get("text") or ""),
        "image_paths": image_paths,
        "selected_image_path": selected_image,
        "expected_verdict": str(item.get("verdict") or ""),
        "rationale": str(item.get("rationale") or ""),
        "distortion_taxonomy": list(item.get("distortion_taxonomy") or []),
        "language_code": str(item.get("language_code") or ""),
        "region_code": str(item.get("region_code") or ""),
        "theme_category": str(item.get("theme_category") or ""),
    }


def build_manifest(
    name: str,
    records: list[dict[str, Any]],
    *,
    seed: int,
    dataset_root: Path,
) -> dict[str, Any]:
    samples = [to_fixed_sample(item, seed=seed, dataset_root=dataset_root) for item in records]
    sample_ids = [item["sample_id"] for item in samples]
    return {
        "dataset": "MMD",
        "name": name,
        "source_split": "train",
        "seed": seed,
        "dataset_root": str(dataset_root.resolve()),
        "sample_count": len(samples),
        "sample_id_sha256": hashlib.sha256("\n".join(sample_ids).encode("utf-8")).hexdigest(),
        "verdict_distribution": dict(Counter(item["expected_verdict"] for item in samples)),
        "samples": samples,
    }


def validate_partition(
    prompt_dev: dict[str, Any],
    validation: dict[str, Any],
    *,
    expected_train_count: int,
    dataset_root: Path,
) -> None:
    dev_ids = {item["sample_id"] for item in prompt_dev["samples"]}
    validation_ids = {item["sample_id"] for item in validation["samples"]}
    if dev_ids & validation_ids:
        raise ValueError("prompt_dev and validation overlap")
    if len(dev_ids | validation_ids) != expected_train_count:
        raise ValueError("partition does not cover the complete train split")
    for manifest in (prompt_dev, validation):
        for item in manifest["samples"]:
            image_path = (dataset_root / item["selected_image_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset_root = args.dataset_root.resolve()
    records = json.loads((dataset_root / "dataset_metadata.json").read_text(encoding="utf-8"))
    prompt_dev_records, validation_records = split_train_records(
        records,
        seed=args.seed,
        validation_per_label=args.validation_per_label,
    )
    prompt_dev = build_manifest("prompt_dev", prompt_dev_records, seed=args.seed, dataset_root=dataset_root)
    validation = build_manifest("validation", validation_records, seed=args.seed, dataset_root=dataset_root)
    smoke_records = select_balanced_records(validation_records, per_label=2, seed=args.seed)
    smoke = build_manifest("validation_smoke10", smoke_records, seed=args.seed, dataset_root=dataset_root)
    train_count = sum(str(item.get("split")) == "train" for item in records)
    validate_partition(
        prompt_dev,
        validation,
        expected_train_count=train_count,
        dataset_root=dataset_root,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt_dev_path = output_dir / "prompt_dev.json"
    validation_path = output_dir / "validation.json"
    smoke_path = output_dir / "smoke10.json"
    summary_path = output_dir / "split_summary.json"
    prompt_dev_path.write_text(json.dumps(prompt_dev, ensure_ascii=False, indent=2), encoding="utf-8")
    validation_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    smoke_path.write_text(json.dumps(smoke, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "dataset": "MMD",
        "source_split": "train",
        "seed": args.seed,
        "strategy": f"stratified validation with {args.validation_per_label} records per verdict",
        "prompt_dev": {
            "path": str(prompt_dev_path),
            "sample_count": prompt_dev["sample_count"],
            "verdict_distribution": prompt_dev["verdict_distribution"],
            "sample_id_sha256": prompt_dev["sample_id_sha256"],
        },
        "validation": {
            "path": str(validation_path),
            "sample_count": validation["sample_count"],
            "verdict_distribution": validation["verdict_distribution"],
            "sample_id_sha256": validation["sample_id_sha256"],
        },
        "smoke10": {
            "path": str(smoke_path),
            "sample_count": smoke["sample_count"],
            "verdict_distribution": smoke["verdict_distribution"],
            "sample_id_sha256": smoke["sample_id_sha256"],
        },
        "overlap_count": 0,
        "covered_train_count": train_count,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

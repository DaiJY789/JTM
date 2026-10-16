import json
from pathlib import Path

import evals.run_mmd_validation as validation_runner
import evals.run_mmd_smoke as smoke_runner
from PIL import Image
from evals.prepare_mmd_splits import (
    LABELS,
    build_manifest,
    select_balanced_records,
    split_train_records,
)
from evals.run_mmd_eval import load_and_sample, load_fixed_manifest, same_evaluation_input


def make_records(per_label: int = 4) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for label in LABELS:
        for index in range(per_label):
            records.append(
                {
                    "sample_id": f"{label}-{index}",
                    "split": "train",
                    "text": f"sample {label} {index}",
                    "image_paths": [f"train/{label}/{index}.jpg"],
                    "verdict": label,
                }
            )
    records.append(
        {
            "sample_id": "test-only",
            "split": "test",
            "text": "not part of the train partition",
            "image_paths": ["test/test.jpg"],
            "verdict": "True",
        }
    )
    return records


def test_stratified_split_is_balanced_disjoint_and_deterministic():
    records = make_records()

    dev_a, validation_a = split_train_records(records, seed=17, validation_per_label=1)
    dev_b, validation_b = split_train_records(records, seed=17, validation_per_label=1)

    assert [item["sample_id"] for item in validation_a] == [
        item["sample_id"] for item in validation_b
    ]
    assert len(dev_a) == 15
    assert len(validation_a) == 5
    assert {item["verdict"] for item in validation_a} == set(LABELS)
    assert {item["sample_id"] for item in dev_a}.isdisjoint(
        {item["sample_id"] for item in validation_a}
    )
    assert {item["sample_id"] for item in dev_a + validation_a} == {
        item["sample_id"] for item in records if item["split"] == "train"
    }


def test_smoke_selection_uses_two_records_per_label_from_validation():
    _, validation = split_train_records(make_records(), seed=17, validation_per_label=2)

    smoke = select_balanced_records(validation, per_label=2, seed=17)

    assert len(smoke) == 10
    assert {item["sample_id"] for item in smoke}.issubset(
        {item["sample_id"] for item in validation}
    )
    assert {label: sum(item["verdict"] == label for item in smoke) for label in LABELS} == {
        label: 2 for label in LABELS
    }


def test_fixed_manifest_preserves_selected_image(tmp_path: Path):
    image_path = tmp_path / "train" / "True" / "sample.jpg"
    image_path.parent.mkdir(parents=True)
    Image.new("RGB", (32, 24), "white").save(image_path)
    source = {
        "sample_id": "001",
        "split": "train",
        "text": "example",
        "image_paths": ["train/True/sample.jpg"],
        "verdict": "True",
    }
    manifest = build_manifest("validation", [source], seed=23, dataset_root=tmp_path)
    manifest_path = tmp_path / "validation.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    samples, split_name, seed = load_fixed_manifest(manifest_path, tmp_path)

    assert split_name == "validation"
    assert seed == 23
    assert samples[0]["selected_image_path"] == "train/True/sample.jpg"
    assert samples[0]["selected_image_absolute_path"] == str(image_path.resolve())
    assert samples[0]["expected_verdict"] == "True"


def test_random_eval_uses_all_images_up_to_four_and_seeded_sample_above_four(
    tmp_path: Path,
):
    records = []
    expected_paths: dict[str, list[str]] = {}
    for sample_id, image_count in (("few", 3), ("many", 6)):
        paths = []
        for index in range(image_count):
            relative_path = f"test/{sample_id}-{index}.jpg"
            image_path = tmp_path / relative_path
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (32, 24), (index * 20, 30, 40)).save(image_path)
            paths.append(relative_path)
        expected_paths[sample_id] = paths
        records.append(
            {
                "sample_id": sample_id,
                "split": "test",
                "text": f"sample {sample_id}",
                "image_paths": paths,
                "verdict": "True",
            }
        )
    (tmp_path / "dataset_metadata.json").write_text(
        json.dumps(records), encoding="utf-8"
    )

    first = load_and_sample(tmp_path, "test", 2, seed=29)
    second = load_and_sample(tmp_path, "test", 2, seed=29)
    first_by_id = {item["sample_id"]: item for item in first}
    second_by_id = {item["sample_id"]: item for item in second}

    assert first_by_id["few"]["selected_image_paths"] == expected_paths["few"]
    assert first_by_id["few"]["selected_image_count"] == 3
    assert len(first_by_id["many"]["selected_image_paths"]) == 4
    assert set(first_by_id["many"]["selected_image_paths"]).issubset(expected_paths["many"])
    assert first_by_id["many"]["selected_image_paths"] == second_by_id["many"]["selected_image_paths"]
    selected_indices = [
        expected_paths["many"].index(path) for path in first_by_id["many"]["selected_image_paths"]
    ]
    assert selected_indices == sorted(selected_indices)
    assert first_by_id["many"]["selected_image_labels"] == [
        f"Image {index + 1}" for index in selected_indices
    ]


def test_fixed_manifest_randomly_selects_four_images_reproducibly(tmp_path: Path):
    image_paths = []
    for index in range(6):
        relative_path = f"train/True/sample-{index}.jpg"
        image_path = tmp_path / relative_path
        image_path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (32, 24), (index * 20, 40, 50)).save(image_path)
        image_paths.append(relative_path)
    source = {
        "sample_id": "multi",
        "split": "train",
        "text": "example",
        "image_paths": image_paths,
        "verdict": "True",
    }
    manifest = build_manifest("validation", [source], seed=23, dataset_root=tmp_path)
    manifest_path = tmp_path / "validation.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    first, _, _ = load_fixed_manifest(manifest_path, tmp_path)
    second, _, _ = load_fixed_manifest(manifest_path, tmp_path)

    assert len(first[0]["selected_image_paths"]) == 4
    assert set(first[0]["selected_image_paths"]).issubset(image_paths)
    assert first[0]["selected_image_paths"] == second[0]["selected_image_paths"]
    assert first[0]["available_image_count"] == 6


def test_resume_rejects_result_from_different_image_selection():
    current = {
        "text": "example",
        "selected_image_paths": ["one.jpg", "two.jpg"],
    }

    assert same_evaluation_input(
        {"text": "example", "selected_image_paths": ["one.jpg", "two.jpg"]},
        current,
    )
    assert not same_evaluation_input(
        {"text": "example", "selected_image_path": "one.jpg"},
        current,
    )


def test_validation_runner_forwards_fixed_manifest_arguments(monkeypatch, tmp_path: Path):
    captured: list[str] = []

    def fake_eval(argv: list[str]) -> int:
        captured.extend(argv)
        return 0

    monkeypatch.setattr(validation_runner, "run_mmd_eval", fake_eval)
    result = validation_runner.main(
        [
            "--dataset-root",
            str(tmp_path / "dataset"),
            "--manifest",
            str(tmp_path / "validation.json"),
            "--output-dir",
            str(tmp_path / "results"),
            "--workers",
            "2",
            "--resume",
            "--limit",
            "3",
        ]
    )

    assert result == 0
    assert captured == [
        "--dataset-root",
        str(tmp_path / "dataset"),
        "--manifest",
        str(tmp_path / "validation.json"),
        "--output-dir",
        str(tmp_path / "results"),
        "--workers",
        "2",
        "--resume",
        "--limit",
        "3",
    ]


def test_validation_comparison_requires_both_diagnostic_ratios_to_decrease():
    baseline = {
        "split": "test",
        "evaluated_count": 100,
        "predicted_distribution": {"False": 57},
        "crossmodal_label_distribution": {"Inconsistent": 91},
    }
    current = {
        "split": "validation",
        "evaluated_count": 100,
        "crossmodal_label_distribution": {"Inconsistent": 60},
        "exact_accuracy": 0.4,
        "macro_f1": 0.38,
        "technical_success_rate": 0.99,
        "per_class_metrics": {
            label: {"recall": 0.25} for label in LABELS
        },
        "runtime_seconds": {"mean_per_case": 18.0, "p95_per_case": 27.0},
        "predicted_distribution": {"False": 35, "Mostly True": 10},
    }

    comparison = validation_runner.compare_with_baseline(current, baseline)

    assert comparison["passed"] is True
    assert comparison["distribution_diagnostics"]["false_prediction_ratio"] == {
        "baseline": 0.57,
        "current": 0.35,
        "change": -0.22,
        "decreased": True,
    }
    assert comparison["distribution_diagnostics"]["inconsistent_crossmodal_ratio"]["decreased"] is True
    assert all(item["passed"] for item in comparison["quality_checks"].values())


def test_validation_comparison_rejects_label_collapse_even_when_diagnostics_improve():
    baseline = {
        "split": "test",
        "evaluated_count": 100,
        "predicted_distribution": {"False": 57},
        "crossmodal_label_distribution": {"Inconsistent": 91},
    }
    current = {
        "split": "validation",
        "evaluated_count": 100,
        "predicted_distribution": {"False": 20, "True": 80},
        "crossmodal_label_distribution": {"Inconsistent": 20},
        "technical_success_rate": 1.0,
        "exact_accuracy": 0.3,
        "macro_f1": 0.2,
        "per_class_metrics": {label: {"recall": 0.0} for label in LABELS},
        "runtime_seconds": {"mean_per_case": 15.0, "p95_per_case": 25.0},
    }

    comparison = validation_runner.compare_with_baseline(current, baseline)

    assert comparison["passed"] is False
    assert comparison["quality_checks"]["mostly_true_prediction_count"]["passed"] is False


def test_smoke_check_reports_twenty_second_target():
    summary = {
        "evaluated_count": 10,
        "expected_distribution": {label: 2 for label in LABELS},
        "technical_success_rate": 1.0,
        "exact_accuracy": 0.4,
        "macro_f1": 0.38,
        "runtime_seconds": {
            "mean_per_case": 18.5,
            "median_per_case": 17.0,
            "max_per_case": 27.0,
        },
    }

    check = smoke_runner.build_smoke_check(summary, 20.0)

    assert check["sample_count"] == 10
    assert check["mean_target_met"] is True
    assert check["median_target_met"] is True
    assert check["latency_target_met"] is True

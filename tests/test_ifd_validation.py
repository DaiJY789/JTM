import json
from pathlib import Path

import pytest

from evals.run_ifd_validation import (
    discover_metadata,
    extract_prediction,
    load_records,
    main,
    normalize_ifd_label,
    select_records,
    summarize,
    to_binary_label,
)


def _write_dataset(root: Path) -> Path:
    dataset = root / "openfake"
    images = dataset / "images"
    images.mkdir(parents=True)
    rows = [
        {"index": 0, "image_file": "images/0000.jpg", "label": "real", "type": "image"},
        {"index": 1, "image_file": "images/0001.jpg", "label": "fake", "type": "video"},
    ]
    for row in rows:
        (dataset / row["image_file"]).write_bytes(b"test")
    metadata = dataset / "metadata.jsonl"
    metadata.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return metadata


@pytest.mark.parametrize(
    ("raw", "ifd_label", "binary_label"),
    [
        ("真实图像", "Real_Image", "real"),
        ("Real_Image", "Real_Image", "real"),
        ("伪造图像", "Forged_Image", "fake"),
        ("Forgery_Detected", "Forged_Image", "fake"),
        ("合成图像", "Synthetic_Image", "fake"),
        ("Deepfake_Suspected", "Synthetic_Image", "fake"),
        ("Cannot_Determine", None, None),
    ],
)
def test_normalize_ifd_and_binary_labels(raw, ifd_label, binary_label):
    assert normalize_ifd_label(raw) == ifd_label
    assert to_binary_label(ifd_label) == binary_label


def test_extract_prediction_prefers_top_level_label():
    assert extract_prediction(
        {"label": "Synthetic_Image", "image_verdict": {"label": "Real_Image"}}
    ) == ("Synthetic_Image", "fake")


def test_discover_and_load_nested_openfake_metadata(tmp_path):
    metadata = _write_dataset(tmp_path)

    assert discover_metadata(tmp_path) == metadata.resolve()
    records = load_records(metadata)

    assert [item["sample_id"] for item in records] == ["0", "1"]
    assert [item["expected_label"] for item in records] == ["real", "fake"]
    assert all(Path(item["image_absolute_path"]).is_file() for item in records)


def test_load_records_rejects_missing_image(tmp_path):
    metadata = tmp_path / "metadata.jsonl"
    metadata.write_text(
        json.dumps({"index": 7, "image_file": "missing.jpg", "label": "fake"})
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(FileNotFoundError, match="sample 7"):
        load_records(metadata)


def test_seeded_record_selection_is_stable():
    records = [{"sample_id": str(index)} for index in range(20)]

    first = select_records(records, 5, 42)
    second = select_records(records, 5, 42)

    assert first == second
    assert len(first) == 5


def test_summarize_binary_metrics(tmp_path):
    records = [
        {
            "sample_id": "0",
            "expected_label": "real",
            "predicted_label": "real",
            "system_ifd_label": "Real_Image",
            "correct": True,
            "status": "completed",
            "runtime_seconds": 1.0,
            "source_type": "image",
            "source_model": "reddit",
            "prediction": {"confidence": 0.9},
        },
        {
            "sample_id": "1",
            "expected_label": "fake",
            "predicted_label": "fake",
            "system_ifd_label": "Synthetic_Image",
            "correct": True,
            "status": "completed",
            "runtime_seconds": 2.0,
            "source_type": "video",
            "source_model": "reddit",
            "prediction": {"confidence": 0.8},
        },
        {
            "sample_id": "2",
            "expected_label": "fake",
            "predicted_label": "real",
            "system_ifd_label": "Real_Image",
            "correct": False,
            "status": "completed",
            "runtime_seconds": 3.0,
            "source_type": "video",
            "source_model": "reddit",
            "prediction": {"confidence": 0.7},
        },
        {
            "sample_id": "3",
            "expected_label": "fake",
            "predicted_label": None,
            "system_ifd_label": None,
            "correct": False,
            "status": "abstained",
            "runtime_seconds": 4.0,
            "source_type": "video",
            "source_model": "reddit",
            "prediction": {},
        },
    ]

    summary = summarize(records, seed=1, metadata_path=tmp_path / "metadata.jsonl")

    assert summary["binary_accuracy"] == 0.5
    assert summary["coverage"] == 0.75
    assert summary["balanced_accuracy"] == 0.6666
    assert summary["fake_detection"]["recall"] == 0.3333
    assert summary["confusion_matrix"]["fake"] == {
        "real": 1,
        "fake": 1,
        "None": 1,
    }


def test_dry_run_validates_dataset_without_creating_outputs(tmp_path, capsys):
    metadata = _write_dataset(tmp_path / "data")
    output_dir = tmp_path / "outputs"

    result = main(
        [
            "--dataset-root",
            str(metadata.parent.parent),
            "--output-dir",
            str(output_dir),
            "--dry-run",
        ]
    )

    printed = json.loads(capsys.readouterr().out)
    assert result == 0
    assert printed["sample_count"] == 2
    assert printed["label_distribution"] == {"real": 1, "fake": 1}
    assert printed["all_images_exist"] is True
    assert not output_dir.exists()

from evals.run_mfc_eval import format_confidence, summarize


def record(expected, predicted, *, status="completed", confidence=0.6):
    return {
        "expected_verdict": expected,
        "predicted_verdict": predicted,
        "system_text_label": predicted,
        "status": status,
        "correct": expected == predicted,
        "ordinal_distance": 0 if expected == predicted else None,
        "runtime_seconds": 1.0,
        "prediction": {
            "final_confidence": confidence,
            "decision_status": "confident" if status == "completed" else "insufficient_evidence",
            "crossmodal_verdict": {},
            "evidence_quality": {},
        },
    }


def test_mfc_summary_counts_technical_failures_in_accuracy_denominator():
    records = [
        record("True", "True", confidence=0.8),
        record("False", "Mostly False", confidence=0.7),
        record("Mixture", None, status="technical_error", confidence=None),
    ]

    summary = summarize(records, seed=1, split="validation", requested_sample_size=3)

    assert summary["evaluated_count"] == 3
    assert summary["completed_count"] == 2
    assert summary["error_count"] == 1
    assert summary["technical_success_rate"] == 0.6667
    assert summary["exact_accuracy"] == 0.3333
    assert summary["mapped_only_accuracy"] == 0.5
    assert summary["coverage"] == 0.6667
    assert "macro_f1" in summary
    assert summary["confidence_calibration"]["brier_score"] is not None


def test_mfc_summary_records_input_image_count_distribution():
    records = [
        {**record("True", "True"), "selected_image_paths": ["one.jpg", "two.jpg"]},
        {
            **record("False", "False"),
            "selected_image_paths": ["a.jpg", "b.jpg", "c.jpg", "d.jpg"],
        },
    ]

    summary = summarize(records, seed=1, split="validation", requested_sample_size=2)

    assert summary["input_image_count_distribution"] == {2: 1, 4: 1}


def test_mfc_summary_reports_tail_latency_language_and_image_groups():
    records = []
    for index in range(20):
        item = record("True", "True")
        item.update(
            {
                "sample_id": str(index),
                "text": "中文样本" if index else "English-only sample",
                "runtime_seconds": float(index + 1),
                "available_image_count": 5 if index < 5 else 2,
                "selected_image_paths": ["one.jpg"],
            }
        )
        records.append(item)

    summary = summarize(records, seed=1, split="validation", requested_sample_size=20)

    assert summary["runtime_seconds"]["p95_per_case"] == 19.05
    assert summary["runtime_seconds"]["over_20_seconds_count"] == 0
    assert summary["text_language_profile"]["no_han_sample_ids"] == ["0"]
    assert summary["accuracy_by_available_image_count_group"]["over_4"]["count"] == 5


def test_mfc_console_confidence_format():
    assert format_confidence({"prediction": {"final_confidence": 0.9}}) == "0.9000"
    assert format_confidence({"prediction": {"final_confidence": 0.4185}}) == "0.4185"
    assert format_confidence({"prediction": {"final_confidence": None}}) == "None"

from pathlib import Path

from evals.run_detection_eval import (
    EvalSample,
    evaluate_result,
    label_family,
    load_samples,
    run_evaluation,
    summarize_results,
)


def test_load_baseline_samples():
    samples = load_samples(Path("evals/samples/baseline.jsonl"))

    assert len(samples) >= 5
    assert {sample.task for sample in samples} >= {"text_only", "image_only", "multimodal", "webpage"}
    assert all(sample.expected["accepted_labels"] for sample in samples)
    assert any(not sample.live_enabled for sample in samples)


def test_evaluate_result_accepts_label_family_but_requires_accepted_label_for_pass():
    sample = EvalSample(
        id="case",
        task="text_only",
        language="en",
        input={},
        expected={
            "label": "Likely_False",
            "accepted_labels": ["Likely_False"],
            "label_family": "false",
            "min_confidence": 0.5,
        },
    )

    result = evaluate_result(sample, {"label": "Mostly_False", "confidence": 0.9})

    assert not result["passed"]
    assert not result["label_match"]
    assert result["family_match"]
    assert result["confidence_ok"]


def test_run_evaluation_summarizes_by_task():
    samples = [
        EvalSample(
            id="case-a",
            task="text_only",
            language="en",
            input={},
            expected={"label": "Likely_True", "accepted_labels": ["Likely_True"], "label_family": "true"},
        ),
        EvalSample(
            id="case-b",
            task="image_only",
            language="zh",
            input={},
            expected={"label": "真实图像", "accepted_labels": ["真实图像"], "label_family": "real_image"},
        ),
    ]

    responses = {
        "case-a": {"label": "Likely_True", "confidence": 0.8},
        "case-b": {"label": "合成图像", "confidence": 0.8},
    }
    report = run_evaluation(samples, lambda sample: responses[sample.id])

    assert report["summary"]["total"] == 2
    assert report["summary"]["passed"] == 1
    assert report["summary"]["by_task"]["text_only"]["pass_rate"] == 1
    assert report["summary"]["by_task"]["image_only"]["pass_rate"] == 0


def test_label_family_and_summary_helpers():
    assert label_family("真实图像") == "real_image"
    assert label_family("Mostly_False") == "false"
    assert label_family("unknown-label") == "unknown"

    summary = summarize_results(
        [
            {"task": "text_only", "passed": True, "label_match": True, "family_match": True},
            {"task": "text_only", "passed": False, "label_match": False, "family_match": True},
        ]
    )

    assert summary["pass_rate"] == 0.5
    assert summary["label_accuracy"] == 0.5
    assert summary["family_accuracy"] == 1

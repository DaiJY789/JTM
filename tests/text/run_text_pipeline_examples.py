import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.api.dependencies import get_detection_orchestrator
from app.core.config import settings


EXAMPLES = [
    {
        "id": "covid_microchips",
        "source": "FactCheck.org / AFP Fact Check",
        "source_url": "https://www.factcheck.org/2021/07/scicheck-spoof-video-furthers-microchip-conspiracy-theory/",
        "text": "Social media posts claim that COVID-19 vaccines contain microchips used to track people.",
        "question": "Is the claim that COVID-19 vaccines contain tracking microchips true?",
        "language": "en",
    },
    {
        "id": "voting_machines_2020",
        "source": "PolitiFact / FactCheck.org",
        "source_url": "https://www.politifact.com/factchecks/2020/nov/13/facebook-posts/no-evidence-dominion-voting-systems-caused-widespr/",
        "text": "A viral claim says voting machines switched large numbers of votes in the 2020 U.S. presidential election.",
        "question": "Did voting machines switch large numbers of votes in the 2020 U.S. presidential election?",
        "language": "en",
    },
    {
        "id": "maui_energy_weapon",
        "source": "AP / FactCheck.org / AFP Fact Check",
        "source_url": "https://www.factcheck.org/2023/08/scicheck-high-winds-drought-conditions-led-to-maui-fires-no-evidence-intentionally-set/",
        "text": "Posts online claim that the Maui wildfires were caused by a directed energy weapon rather than natural and human factors.",
        "question": "Were the Maui wildfires caused by a directed energy weapon?",
        "language": "en",
    },
    {
        "id": "climate_no_warming",
        "source": "NASA / NOAA",
        "source_url": "https://science.nasa.gov/climate-change/faq/has-earth-continued-to-warm-since-1998/",
        "text": "A claim circulating online says global warming stopped after 1998 and recent climate change is therefore a hoax.",
        "question": "Did global warming stop after 1998?",
        "language": "en",
    },
    {
        "id": "who_pandemic_treaty",
        "source": "Reuters / FactCheck.org",
        "source_url": "https://www.factcheck.org/2023/03/scicheck-who-pandemic-treaty-draft-reaffirms-nations-sovereignty-to-dictate-health-policy/",
        "text": "Online posts claim that the World Health Organization pandemic treaty would give the WHO power to override national sovereignty.",
        "question": "Would the WHO pandemic treaty override national sovereignty?",
        "language": "en",
    },
]


def elapsed(start_time: float) -> float:
    return round(time.perf_counter() - start_time, 3)


def summarize_claims(claims: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "claim": claim.get("claim", ""),
            "search_query": claim.get("search_query", ""),
            "evidence_count": claim.get("evidence_count", 0),
        }
        for claim in claims
    ]


def run_example(text_service, example: dict[str, Any]) -> dict[str, Any]:
    total_start = time.perf_counter()

    step_start = time.perf_counter()
    claims = text_service.claim_extractor.extract(example["text"], example["language"])
    claim_extraction_seconds = elapsed(step_start)

    step_start = time.perf_counter()
    evidence = text_service.evidence_search_service.search(
        claims,
        example["language"],
    )
    evidence_search_seconds = elapsed(step_start)

    claims = text_service.evidence_search_service.attach_evidence(claims, evidence)

    step_start = time.perf_counter()
    result = text_service.text_verdict_service.evaluate(
        text=example["text"],
        claims=claims,
        evidence=evidence,
        history=[],
        language=example["language"],
    )
    text_verdict_seconds = elapsed(step_start)

    explain = result.get("explainInDetail", {})
    return {
        "id": example["id"],
        "source": example["source"],
        "source_url": example["source_url"],
        "text": example["text"],
        "question": example["question"],
        "timing_seconds": {
            "claim_extraction": claim_extraction_seconds,
            "evidence_search": evidence_search_seconds,
            "text_verdict": text_verdict_seconds,
            "total": elapsed(total_start),
        },
        "label": result.get("label", "Cannot_Determine"),
        "confidence": result.get("confidence"),
        "summary": result.get("appraisalResults", ""),
        "reasoning": explain.get("text", ""),
        "evidence_assessment": result.get("evidence_assessment", ""),
        "claims": summarize_claims(claims),
        "claim_assessments": result.get("claim_assessments", []),
        "raw_output": result.get("raw_output", ""),
    }


def main() -> None:
    orchestrator = get_detection_orchestrator()
    text_service = orchestrator.text_detection_service
    model_config = {
        "llm_api": "deepseek_openai_compatible",
        "llm_base_url": settings.deepseek_base_url,
        "llm_model": settings.deepseek_model,
        "claim_extraction_timeout_seconds": settings.claim_extraction_timeout_seconds,
        "text_verdict_timeout_seconds": settings.text_verdict_timeout_seconds,
        "search_timeout_seconds": settings.search_timeout_seconds,
    }
    print("===== Model Config =====", flush=True)
    print(json.dumps(model_config, ensure_ascii=False, indent=2), flush=True)

    output = []
    for index, example in enumerate(EXAMPLES, start=1):
        print(f"\n===== Example {index}: {example['id']} =====", flush=True)
        result = run_example(text_service, example)
        output.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    print("\n===== Summary =====", flush=True)
    summary = [
        {
            "id": item["id"],
            "label": item["label"],
            "confidence": item["confidence"],
            "timing_seconds": item["timing_seconds"],
            "evidence_counts": [claim["evidence_count"] for claim in item["claims"]],
        }
        for item in output
    ]
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)

    result_path = ROOT_DIR / "tests" / "text" / "result" / "text_pipeline_examples_last.json"
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(
        json.dumps({"model_config": model_config, "results": output, "summary": summary}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved result: {result_path}", flush=True)


if __name__ == "__main__":
    main()

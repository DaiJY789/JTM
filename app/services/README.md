# Services Flow

This directory contains the main detection logic. The orchestrator selects the input route, while textual, visual, and cross-modal services keep separate responsibilities.

## Entry Point

`DetectionOrchestrator` receives normalized API inputs from `/detect` and always resolves the pipeline automatically.

```text
DetectionOrchestrator.detect(...)
  -> text_only       -> TextualDetectionPipeline
  -> image_only      -> VisualVerdictService
  -> multimodal      -> textual + visual + CrossmodalConsistencyPipeline
                     -> ComprehensiveFusionService -> final report
```

Auto-routing rules:

- no image + text/question -> `text_only`
- image + generic image question -> `image_only`
- image + news text/title -> `multimodal`

## Text-Only Flow

Implemented in `app/services/textual/T_pipeline.py`.

```text
TextualDetectionPipeline.detect(text)
  -> ClaimExtractor.extract(text)
  -> evidence = EvidenceSearchService.search(claims)
  -> EvidenceSearchService.attach_evidence(claims, evidence)
  -> TextualVerdictService.evaluate(text, claims, evidence)
  -> format text_verdict, claims, evidence
```

Current behavior:

- Uses `ClaimExtractor` to produce structured claims.
- Uses `EvidenceSearchService` to retrieve claim-level evidence.
- Uses `TextualVerdictService` for the final text verdict.
- Uses factual labels only when retrieved evidence can support fact checking.
- Asks the verdict model to choose rumor-detection labels when evidence is insufficient:
  - `Likely_False`
  - `Likely_True`
  - `Cannot_Determine`
- Produces canonical English labels.
- Returns top-level `claims` and nested `text_verdict`.

Known gap:

- Evidence search does not yet score source quality or relevance.

## Image-Only Flow

Implemented in `app/services/visual/visual_verdict.py`.

```text
VisualVerdictService.detect(image, question)
  -> VisualForensicPipeline.detect(image, question)
  -> Qwen vision model + local forensic tools
  -> normalize label, evidence chain, suspicious regions
```

Current output:

- `label`
- `visual_analysis`
- `evidence_chain`
- `suspicious_regions`
- `forgery_types`
- `user_answer`

Image labels:

- `完全真实`
- `存在伪造`
- `完全AI生成`

## Cross-modal Consistency Flow

Implemented in `app/services/crossmodal/C_pipeline.py`.

```text
CrossmodalConsistencyPipeline.detect(text, image)
  -> extract textual elements
  -> inspect visible image elements
  -> compare people, place, time, event, objects, actions, and visible text
  -> return Consistent | Partially_Consistent | Inconsistent
```

This service does not fact-check the text, judge image authenticity, or produce the final credibility label. Its output contains:

- `text_elements`
- `visual_elements`
- `matched_elements`
- `mismatched_elements`
- `missing_elements`
- `consistency_score`
- `reasoning`

## Comprehensive Fusion Report

Implemented in `app/services/comprehensive_fusion.py`.

For text-and-image input, `DetectionOrchestrator` runs the textual, visual, and cross-modal branches in parallel. `ComprehensiveFusionService` then sends the following to the vision-capable multimodal model:

- original news text and user question
- original image
- textual authenticity result and evidence
- visual authenticity result and forensic evidence
- cross-modal consistency result

The model produces the final `report`, including `label`, `type`, `confidence`, `risk_level`, `summary`, `user_answer`, and `limitations`. A deterministic fallback is retained for model failures.

## Text Services

Textual components live under `app/services/textual/`:

- `claim_extractor.py`
- `evidence_search.py`
- `textual_verdict.py`
- `T_pipeline.py`

`TextualVerdictService`

- Owns the text-level label policy.
- Receives original text, extracted claims, and retrieved evidence.
- Calls a JSON-only verdict evaluator directly.
- Parses structured output directly.
- Leaves evidence sufficiency to the model, guided by the prompt label policy.

`ClaimExtractor`

- Uses a lightweight LLM prompt to extract up to three fact-checkable claims.
- Falls back to one whole-text claim if model output is not valid JSON.
- Recovers JSON from fenced output or explanatory text when possible.
- Applies a short LLM timeout to keep the demo responsive.

`EvidenceSearchService`

- Runs per-claim evidence retrieval with the existing web search tool.
- Normalizes search results into a shared evidence structure.
- Returns evidence as an explicit pipeline artifact before attaching it back to claims.
- Fails closed to an empty evidence list when search is unavailable.

## Shared Services

`ImageService`

- Loads images from bytes, local path, or URL.
- Resizes and base64-encodes images.
- Provides shared image loading helpers for visual, cross-modal, and comprehensive fusion services.

# Services Flow

This directory contains the main detection logic. The code is organized around a single orchestration entry point and three scenario-specific services.

## Entry Point

`DetectionOrchestrator` receives normalized API inputs from `/detect` and resolves the detection mode.

```text
DetectionOrchestrator.detect(...)
  -> text_only       -> TextDetectionService
  -> image_only      -> ImageDetectionService
  -> multimodal      -> MultimodalDetectionService
```

Auto-routing rules:

- no image + text/question -> `text_only`
- image + generic image question -> `image_only`
- image + news text/title -> `multimodal`
- explicit mode wins

## Text-Only Flow

Implemented in `app/services/text/TextDetectionService`.

```text
TextDetectionService.detect(text)
  -> ClaimExtractor.extract(text)
  -> evidence = EvidenceSearchService.search(claims)
  -> EvidenceSearchService.attach_evidence(claims, evidence)
  -> TextVerdictService.evaluate(text, claims, evidence)
  -> format text_verdict, claims, evidence
```

Current behavior:

- Uses `ClaimExtractor` to produce structured claims.
- Uses `EvidenceSearchService` to retrieve claim-level evidence.
- Uses `TextVerdictService` for the final text verdict.
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

Implemented in `ImageDetectionService`.

```text
ImageDetectionService.detect(image, question)
  -> ImageForensicPipeline.detect(image, question)
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

## Multimodal Flow

Implemented in `app/services/multimodal/`.

```text
MultimodalDetectionService.detect(text, image)
  -> TextDetectionService.detect(text)
  -> ImageDetectionService.detect(image)
  -> MultimodalAlignmentService.evaluate(text, image, text_result, image_result)
  -> MultimodalVerdictService.evaluate(text_result, image_result, alignment_result)
```

Current output:

- `text_verdict`
- `image_verdict`
- `alignment_verdict`
- `alignment_verdict.text_elements`
- `alignment_verdict.visual_elements`
- `matched_elements`
- `mismatched_elements`
- `missing_elements`
- final `user_answer`
- final fused `label`

Fusion priority:

- forged or completely AI-generated image -> `Image_False`
- image-text mismatch or out-of-context image -> `Cross_Modal_Mismatch`
- false text -> `Text_False`
- weak relation or mixed text -> `Mixed`
- true text + real image + consistent alignment -> `Consistent_True`
- otherwise -> `Cannot_Determine`

Known gap:

- Entity-level alignment is still VLM-prompt based, not a dedicated retrieval-backed entity linker.

## Text Services

Text-only components live under `app/services/text/`:

- `claim_extractor.py`
- `evidence_search.py`
- `text_verdict.py`
- `text_detection.py`

`TextVerdictService`

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
- Provides shared image loading helpers for the vision and multimodal services.

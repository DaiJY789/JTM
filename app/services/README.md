# Detection Services

`DetectionOrchestrator` only performs explicit two-way routing:

```text
multimodal_fact_checking + text + image -> MMDDetectionService -> MMD_Agent
deepfake_detection + image             -> IFDDetectionService -> IFD_Agent
```

There is no fallback to the removed P/E/T/C/F workflow. MMD input must contain both text and an image; deepfake input must contain an image.

## MMD path

`MMD_Agent` remains the only decision-making agent. Before its final vision-model call, the application runs image OCR/entity extraction and three whole-post Bocha searches: whole event, key detail, and OCR-assisted cross-modal context. OCR and the first two searches run in parallel; the third search uses OCR output. Atomic points remain part of final analysis but do not define the search scope.

`app/agents/tools/mmd/` contains `ImageOCREntityTool`, `BoChaSearchTool`, and the rule-based `MMDSearchQueryBuilder`. Tool failures are isolated and the final agent continues with available evidence. Search snippets are untrusted candidates, search absence is not contradiction, crawl time is not publication time, and returned image URLs are not treated as verified provenance.

## Deepfake path

`DetectionInputBuilder` persists uploaded, remote, and webpage-extracted images under the conversation directory before orchestration. Both detector paths consume those local copies; original remote URLs remain provenance metadata only. `IFDDetectionService` passes the persisted path directly to `IFDDetectionAgent`, avoiding a second input copy under the vision output directory. The pipeline lets the vision model select local forensic tools and returns the normalized image label, evidence chain, suspicious regions, forgery type, limitations, and user-facing explanation.

## Shared services

- `DetectionInputBuilder` validates and normalizes API inputs.
- `DetectionReportBuilder` maps detector output to the internal report schema.
- `DetectionResponseAdapter` preserves the public response shape.
- `FollowupAnalysisService` explains an existing result without rerunning detection unless the user explicitly asks to rerun it.

from __future__ import annotations

from typing import Any

from app.schemas.artifacts import ArtifactRole, ArtifactType, DetectionTask, EvidenceBundle, InputArtifact
from app.schemas.detection import DetectionMode
from app.services.artifact_store import ArtifactStore
from app.services.webpage_extraction import WebpageExtractionError, WebpageExtractionResult, WebpageExtractionService


class DetectionInputError(ValueError):
    pass


class DetectionInputBuilder:
    """Normalize raw API inputs into a DetectionTask."""

    def __init__(
        self,
        webpage_extraction_service: WebpageExtractionService,
        artifact_store: ArtifactStore,
    ):
        self.webpage_extraction_service = webpage_extraction_service
        self.artifact_store = artifact_store

    def build(
        self,
        *,
        text: str | None,
        image_content: bytes | None,
        image_filename: str | None,
        image_content_type: str | None,
        image_url: str | None,
        webpage_url: str | None,
        question: str | None,
        language: str,
        history: list[dict[str, Any]],
        conversation_id: str,
    ) -> DetectionTask:
        webpage_context = self._extract_webpage(webpage_url)
        bundle = EvidenceBundle()
        explicit_text = (text or "").strip()
        if explicit_text:
            bundle.text_artifacts.append(
                InputArtifact(
                    type=ArtifactType.TEXT,
                    role=ArtifactRole.USER_INPUT,
                    content=explicit_text,
                )
            )
        if webpage_context:
            bundle.webpage_artifact = self._webpage_artifact(webpage_context)
            extracted_text = "\n\n".join(
                part for part in [webpage_context.title, webpage_context.text] if part
            ).strip()
            if extracted_text and not explicit_text:
                bundle.text_artifacts.append(
                    InputArtifact(
                        type=ArtifactType.TEXT,
                        role=ArtifactRole.WEBPAGE_TEXT,
                        content=extracted_text,
                        source_url=webpage_context.final_url,
                        metadata={"title": webpage_context.title},
                    )
                )
            for index, url in enumerate(webpage_context.images):
                bundle.image_artifacts.append(
                    InputArtifact(
                        type=ArtifactType.IMAGE,
                        role=ArtifactRole.WEBPAGE_IMAGE,
                        source_url=url,
                        metadata={"index": index, "webpage_final_url": webpage_context.final_url},
                    )
                )
            bundle.evidence_items.append(self._webpage_evidence(webpage_context))

        stored_image = self.artifact_store.save_image(
            conversation_id,
            image_content,
            filename=image_filename,
            content_type=image_content_type,
        )
        if stored_image:
            bundle.image_artifacts.insert(0, stored_image)
        elif image_url:
            bundle.image_artifacts.insert(
                0,
                InputArtifact(
                    type=ArtifactType.IMAGE,
                    role=ArtifactRole.REMOTE_IMAGE,
                    source_url=image_url.strip(),
                ),
            )

        self._validate(bundle, image_content, question)
        user_input = self._user_input(
            language=language,
            text=bundle.primary_text,
            question=question,
            image_url=bundle.primary_image.source_url if bundle.primary_image else "",
            webpage_url=webpage_url,
            webpage_context=webpage_context,
            image_asset=self.artifact_store.artifact_to_legacy_asset(stored_image),
            image_artifacts=bundle.image_artifacts,
        )
        return DetectionTask(
            mode=DetectionMode.AUTO,
            language=language,
            question=(question or "").strip(),
            conversation_id=conversation_id,
            history=history,
            evidence_bundle=bundle,
            image_content=image_content,
            user_input=user_input,
        )

    def _extract_webpage(self, webpage_url: str | None) -> WebpageExtractionResult | None:
        if not webpage_url:
            return None
        try:
            return self.webpage_extraction_service.extract(webpage_url)
        except WebpageExtractionError as exc:
            raise DetectionInputError(str(exc)) from exc

    def _validate(
        self,
        bundle: EvidenceBundle,
        image_content: bytes | None,
        question: str | None,
    ) -> None:
        has_text = bool(bundle.primary_text)
        has_question = bool((question or "").strip())
        has_image = bool(image_content or bundle.primary_image)
        if not (has_text or has_question or has_image):
            raise DetectionInputError("text、question、image、image_url 或 webpage_url 至少需要提供一个。")

    def _webpage_artifact(self, webpage_context: WebpageExtractionResult) -> InputArtifact:
        return InputArtifact(
            type=ArtifactType.WEBPAGE,
            role=ArtifactRole.USER_INPUT,
            source_url=webpage_context.url,
            content=webpage_context.text,
            metadata={
                "final_url": webpage_context.final_url,
                "title": webpage_context.title,
                "image_url": webpage_context.image_url,
                "images": webpage_context.images,
                "metadata": webpage_context.metadata,
            },
        )

    def _webpage_evidence(self, webpage_context: WebpageExtractionResult) -> dict[str, Any]:
        return {
            "type": "webpage_source",
            "source": "WebpageExtractionService",
            "url": webpage_context.final_url,
            "title": webpage_context.title,
            "content": {
                "text_preview": webpage_context.text[:500],
                "image_url": webpage_context.image_url,
                "image_count": len(webpage_context.images),
            },
        }

    def _user_input(
        self,
        *,
        language: str,
        text: str,
        question: str | None,
        image_url: str,
        webpage_url: str | None,
        webpage_context: WebpageExtractionResult | None,
        image_asset: dict[str, Any],
        image_artifacts: list[InputArtifact],
    ) -> dict[str, Any]:
        return {
            "mode": DetectionMode.AUTO.value,
            "language": language,
            "text": text or "",
            "question": (question or "").strip(),
            "image_url": image_url or "",
            "webpage_url": webpage_context.url if webpage_context else webpage_url or "",
            "has_image": bool(image_artifacts),
            "image_asset": image_asset,
            "image_artifacts": [artifact.model_dump(mode="json", exclude={"content"}) for artifact in image_artifacts],
            "webpage": {
                "final_url": webpage_context.final_url,
                "title": webpage_context.title,
                "image_url": webpage_context.image_url,
                "image_count": len(webpage_context.images),
            }
            if webpage_context
            else {},
        }

from __future__ import annotations

import base64
import binascii
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import unquote, unquote_to_bytes, urlparse

from PIL import Image

from app.core.config import settings
from app.schemas.artifacts import ArtifactRole, ArtifactType, DetectionTask, EvidenceBundle, InputArtifact
from app.schemas.detection import DetectionMode, DetectionTaskType
from app.services.artifact_store import ArtifactStore
from app.services.remote_image_downloader import RemoteImageDownloader, RemoteImageDownloadError
from app.services.webpage_extraction import WebpageExtractionError, WebpageExtractionResult, WebpageExtractionService


class DetectionInputError(ValueError):
    pass


class DetectionInputBuilder:
    """Normalize raw API inputs into a DetectionTask."""

    def __init__(
        self,
        webpage_extraction_service: WebpageExtractionService,
        artifact_store: ArtifactStore,
        max_mmd_images: int = settings.mmd_max_images,
        remote_image_downloader: RemoteImageDownloader | None = None,
    ):
        self.webpage_extraction_service = webpage_extraction_service
        self.artifact_store = artifact_store
        self.max_mmd_images = max(1, int(max_mmd_images))
        self.remote_image_downloader = remote_image_downloader or RemoteImageDownloader()

    def build(
        self,
        *,
        task: DetectionTaskType,
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
        image_contents: list[bytes] | tuple[bytes, ...] | None = None,
        image_filenames: list[str | None] | tuple[str | None, ...] | None = None,
        image_content_types: list[str | None] | tuple[str | None, ...] | None = None,
        image_urls: list[str] | tuple[str, ...] | None = None,
    ) -> DetectionTask:
        if task == DetectionTaskType.AIGC_DETECTION:
            raise DetectionInputError("task=aigc_detection 暂未实现。")
        upload_contents = list(image_contents or (() if image_content is None else (image_content,)))
        upload_filenames = self._aligned_values(
            image_filenames or (() if image_filename is None else (image_filename,)),
            len(upload_contents),
        )
        upload_content_types = self._aligned_values(
            image_content_types or (() if image_content_type is None else (image_content_type,)),
            len(upload_contents),
        )
        remote_urls = list(
            dict.fromkeys(
                value.strip()
                for value in ([image_url] if image_url else []) + list(image_urls or [])
                if value and value.strip()
            )
        )
        submitted_image_count = len(upload_contents) + len(remote_urls)
        if task == DetectionTaskType.DEEPFAKE_DETECTION and len(upload_contents) + len(remote_urls) > 1:
            raise DetectionInputError("task=deepfake_detection 当前只支持一张图片。")
        for content in upload_contents:
            self._validate_uploaded_image(content)
        for url in remote_urls:
            self._validate_image_url(url)
        if task == DetectionTaskType.MULTIMODAL_FACT_CHECKING:
            upload_contents = upload_contents[: self.max_mmd_images]
            upload_filenames = upload_filenames[: len(upload_contents)]
            upload_content_types = upload_content_types[: len(upload_contents)]
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
            valid_webpage_images = [
                url for url in webpage_context.images if self._is_remote_image_url(url)
            ]
            for index, url in enumerate(valid_webpage_images):
                bundle.image_artifacts.append(
                    InputArtifact(
                        type=ArtifactType.IMAGE,
                        role=ArtifactRole.WEBPAGE_IMAGE,
                        source_url=url,
                        metadata={"index": index, "webpage_final_url": webpage_context.final_url},
                    )
                )
            bundle.evidence_items.append(self._webpage_evidence(webpage_context))

        explicit_images = []
        for index, content in enumerate(upload_contents):
            stored_image = self.artifact_store.save_image(
                conversation_id,
                content,
                filename=upload_filenames[index],
                content_type=upload_content_types[index],
                metadata={"input_index": index},
            )
            if stored_image:
                explicit_images.append(stored_image)
        for index, url in enumerate(remote_urls, start=len(explicit_images)):
            explicit_images.append(
                InputArtifact(
                    type=ArtifactType.IMAGE,
                    role=ArtifactRole.REMOTE_IMAGE,
                    source_url=url,
                    metadata={"input_index": index},
                )
            )
        bundle.image_artifacts = self._dedupe_images([*explicit_images, *bundle.image_artifacts])
        requested_image_count = max(len(bundle.image_artifacts), submitted_image_count)
        if task == DetectionTaskType.DEEPFAKE_DETECTION:
            bundle.image_artifacts = bundle.image_artifacts[:1]
        else:
            bundle.image_artifacts = bundle.image_artifacts[: self.max_mmd_images]

        persisted_images = []
        persisted_contents = []
        for index, image_artifact in enumerate(bundle.image_artifacts):
            if image_artifact.local_path:
                persisted_image = image_artifact
            else:
                persisted_image = self._persist_remote_image(
                    conversation_id,
                    image_artifact,
                    index,
                )
            try:
                content = Path(persisted_image.local_path).read_bytes()
            except OSError as exc:
                raise DetectionInputError(f"无法读取第 {index + 1} 张会话图片。") from exc
            persisted_images.append(persisted_image)
            persisted_contents.append(content)
        bundle.image_artifacts = persisted_images

        self._validate(task, bundle, persisted_contents, question)
        user_input = self._user_input(
            task=task,
            language=language,
            text=bundle.primary_text,
            question=question,
            image_url=bundle.primary_image.source_url if bundle.primary_image else "",
            webpage_url=webpage_url,
            webpage_context=webpage_context,
            image_asset=self.artifact_store.artifact_to_legacy_asset(
                persisted_images[0] if persisted_images else None
            ),
            image_assets=[
                self.artifact_store.artifact_to_legacy_asset(item) for item in persisted_images
            ],
            image_artifacts=bundle.image_artifacts,
            requested_image_count=requested_image_count,
        )
        resolved_mode = (
            DetectionMode.IMAGE_ONLY
            if task == DetectionTaskType.DEEPFAKE_DETECTION
            else DetectionMode.MULTIMODAL
            if persisted_images
            else DetectionMode.TEXT_ONLY
        )
        return DetectionTask(
            task=task,
            mode=resolved_mode,
            language=language,
            question=(question or "").strip(),
            conversation_id=conversation_id,
            history=history,
            evidence_bundle=bundle,
            image_content=persisted_contents[0] if persisted_contents else None,
            image_contents=persisted_contents,
            user_input=user_input,
        )

    def _persist_remote_image(
        self,
        conversation_id: str,
        artifact: InputArtifact,
        index: int,
    ) -> InputArtifact:
        source_url = artifact.source_url
        try:
            if source_url.startswith("data:image/"):
                content, content_type, suffix = self._decode_data_image(source_url)
                final_url = ""
            else:
                downloaded = self.remote_image_downloader.download(source_url)
                content = downloaded.content
                content_type = downloaded.media_type
                suffix = downloaded.suffix
                final_url = downloaded.final_url
        except (RemoteImageDownloadError, ValueError, binascii.Error) as exc:
            raise DetectionInputError(
                f"第 {index + 1} 张远程图片下载失败：{exc}"
            ) from exc

        source_name = Path(unquote(urlparse(final_url or source_url).path)).stem
        filename = f"{source_name[:80] or f'remote-{index + 1}'}{suffix}"
        stored = self.artifact_store.save_image(
            conversation_id,
            content,
            filename=filename,
            content_type=content_type,
            source_url=source_url if final_url else None,
            role=artifact.role,
            metadata={
                **artifact.metadata,
                "input_index": index,
                "remote_cached": True,
                "final_source_url": final_url,
            },
        )
        if not stored:
            raise DetectionInputError(f"无法保存第 {index + 1} 张远程图片。")
        return stored

    def _decode_data_image(self, value: str) -> tuple[bytes, str, str]:
        metadata, payload = value.split(",", 1)
        content = (
            base64.b64decode(payload, validate=True)
            if ";base64" in metadata.lower()
            else unquote_to_bytes(payload)
        )
        self._validate_uploaded_image(content)
        content_type = metadata.split(";", 1)[0].split(":", 1)[-1].lower()
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
            "image/gif": ".gif",
            "image/bmp": ".bmp",
        }.get(content_type, ".img")
        if suffix == ".img":
            raise DetectionInputError("data:image URL 的图片格式不受支持。")
        return content, content_type, suffix

    def _aligned_values(self, values: Any, size: int) -> list[Any]:
        output = list(values or [])[:size]
        return output + [None] * (size - len(output))

    def _dedupe_images(self, artifacts: list[InputArtifact]) -> list[InputArtifact]:
        output = []
        seen = set()
        for artifact in artifacts:
            key = artifact.source_url or artifact.local_path or artifact.id
            if not key or key in seen:
                continue
            seen.add(key)
            output.append(artifact)
        return output

    def _extract_webpage(self, webpage_url: str | None) -> WebpageExtractionResult | None:
        if not webpage_url:
            return None
        try:
            return self.webpage_extraction_service.extract(webpage_url)
        except WebpageExtractionError as exc:
            raise DetectionInputError(str(exc)) from exc

    def _validate_uploaded_image(self, content: bytes) -> None:
        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()
        except Exception as exc:
            raise DetectionInputError("上传文件不是可解析的有效图片。") from exc

    def _validate_image_url(self, value: str) -> None:
        cleaned = str(value or "").strip()
        if self._is_remote_image_url(cleaned):
            return
        if cleaned.startswith("data:image/"):
            try:
                metadata, payload = cleaned.split(",", 1)
                content = (
                    base64.b64decode(payload, validate=True)
                    if ";base64" in metadata.lower()
                    else unquote_to_bytes(payload)
                )
            except (ValueError, binascii.Error) as exc:
                raise DetectionInputError("image_url 中的 data URL 不是有效图片。") from exc
            self._validate_uploaded_image(content)
            return
        raise DetectionInputError("image_url 必须是有效的 http(s) 图片 URL 或 data:image URL。")

    def _is_remote_image_url(self, value: Any) -> bool:
        parsed = urlparse(str(value or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _validate(
        self,
        task: DetectionTaskType,
        bundle: EvidenceBundle,
        image_contents: list[bytes],
        question: str | None,
    ) -> None:
        has_text = bool(bundle.primary_text)
        has_question = bool((question or "").strip())
        has_image = bool(image_contents or bundle.primary_image)
        if not (has_text or has_question or has_image):
            raise DetectionInputError("text、question、image、image_url 或 webpage_url 至少需要提供一个。")
        if task == DetectionTaskType.MULTIMODAL_FACT_CHECKING and not has_text:
            raise DetectionInputError("task=multimodal_fact_checking 需要提供待核查的 text 或 webpage_url 文本。")
        if task == DetectionTaskType.DEEPFAKE_DETECTION and not has_image:
            raise DetectionInputError("task=deepfake_detection 需要提供 image、image_url 或包含图片的 webpage_url。")

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
        task: DetectionTaskType,
        language: str,
        text: str,
        question: str | None,
        image_url: str,
        webpage_url: str | None,
        webpage_context: WebpageExtractionResult | None,
        image_asset: dict[str, Any],
        image_assets: list[dict[str, Any]],
        image_artifacts: list[InputArtifact],
        requested_image_count: int,
    ) -> dict[str, Any]:
        return {
            "task": task.value,
            "mode": (
                DetectionMode.IMAGE_ONLY.value
                if task == DetectionTaskType.DEEPFAKE_DETECTION
                else DetectionMode.MULTIMODAL.value
                if image_artifacts
                else DetectionMode.TEXT_ONLY.value
            ),
            "language": language,
            "text": text or "",
            "question": (question or "").strip(),
            "image_url": image_url or "",
            "image_urls": [
                artifact.source_url
                for artifact in image_artifacts
                if artifact.source_url
            ],
            "webpage_url": webpage_context.url if webpage_context else webpage_url or "",
            "has_image": bool(image_artifacts),
            "image_asset": image_asset,
            "image_assets": image_assets,
            "image_count": len(image_artifacts),
            "images_truncated": requested_image_count > len(image_artifacts),
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

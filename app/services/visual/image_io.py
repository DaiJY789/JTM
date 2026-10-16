import base64
import io
import mimetypes
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from PIL import Image, ImageOps

from app.agents.models import ModelFactory
from app.core.config import Settings, settings
from app.services.prompt_service import PromptService


class ImageService:
    def __init__(
        self,
        config: Settings = settings,
        model_factory: ModelFactory | None = None,
        prompt_service: PromptService | None = None,
    ):
        self.config = config
        self.model_factory = model_factory or ModelFactory(config)
        self.prompt_service = prompt_service or PromptService()

    def resize_image(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        while width >= 500 or height >= 400:
            width *= 0.8
            height *= 0.8
        return image.resize((int(width), int(height)))

    def encode_image(self, image: Image.Image) -> str:
        buffered = io.BytesIO()
        image.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def build_image_content_part(
        self,
        image_content: bytes | None = None,
        image_url: str | None = None,
    ) -> dict[str, Any] | None:
        """Build an OpenAI-compatible image content part from bytes, a local path, or a URL."""
        if image_content:
            try:
                with Image.open(BytesIO(image_content)) as image:
                    image_format = image.format or "PNG"
            except Exception:
                return None
            mime_type = Image.MIME.get(image_format.upper(), "image/png")
            encoded = base64.b64encode(image_content).decode("utf-8")
            return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}

        cleaned_url = str(image_url or "").strip()
        if not cleaned_url:
            return None
        parsed = urlparse(cleaned_url)
        if not parsed.scheme:
            candidate = Path(cleaned_url)
            try:
                is_file = candidate.is_file()
            except OSError:
                is_file = False
            if is_file:
                try:
                    content = candidate.read_bytes()
                except OSError:
                    return None
                mime_type = mimetypes.guess_type(candidate.name)[0] or "image/png"
                encoded = base64.b64encode(content).decode("utf-8")
                return {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{encoded}"}}
        if parsed.scheme in {"http", "https"} and parsed.netloc:
            return {"type": "image_url", "image_url": {"url": cleaned_url}}
        if parsed.scheme == "data" and cleaned_url.startswith("data:image/"):
            return {"type": "image_url", "image_url": {"url": cleaned_url}}
        return None

    def build_mmd_image_content_part(
        self,
        image_content: bytes | None = None,
        image_url: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a size-bounded image input for the latency-sensitive MMD path."""
        if not image_content:
            return self.build_image_content_part(image_content, image_url)
        optimized = self._optimize_mmd_image(image_content)
        if optimized is None:
            return None
        encoded = base64.b64encode(optimized).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}

    def build_ocr_image_content_part(
        self,
        image_content: bytes | None = None,
        image_url: str | None = None,
    ) -> dict[str, Any] | None:
        """Build a higher-resolution image input so small text remains legible for OCR."""
        if not image_content:
            return self.build_image_content_part(image_content, image_url)
        optimized = self._optimize_ocr_image(image_content)
        if optimized is None:
            return None
        encoded = base64.b64encode(optimized).decode("utf-8")
        return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}

    def _optimize_mmd_image(self, content: bytes) -> bytes | None:
        try:
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source)
                if min(image.size) < 11:
                    return None
                max_side = max(64, int(self.config.mmd_image_max_side))
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                if "A" in image.getbands():
                    background = Image.new("RGB", image.size, "white")
                    background.paste(image, mask=image.getchannel("A"))
                    image = background
                else:
                    image = image.convert("RGB")
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=82, optimize=True)
                return buffered.getvalue()
        except Exception:
            return None

    def _optimize_ocr_image(self, content: bytes) -> bytes | None:
        try:
            with Image.open(BytesIO(content)) as source:
                image = ImageOps.exif_transpose(source)
                if min(image.size) < 11:
                    return None
                max_side = max(256, int(self.config.mmd_ocr_image_max_side))
                image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                if "A" in image.getbands():
                    background = Image.new("RGB", image.size, "white")
                    background.paste(image, mask=image.getchannel("A"))
                    image = background
                else:
                    image = image.convert("RGB")
                buffered = io.BytesIO()
                image.save(buffered, format="JPEG", quality=90, optimize=True)
                return buffered.getvalue()
        except Exception:
            return None

    def load_image(self, image_path_or_url: str) -> Image.Image | None:
        if image_path_or_url.startswith(("http://", "https://")):
            try:
                response = requests.get(image_path_or_url, timeout=10)
                response.raise_for_status()
                return Image.open(BytesIO(response.content))
            except Exception:
                return None

        try:
            return Image.open(image_path_or_url)
        except Exception:
            return None

    def load_image_from_bytes(self, content: bytes) -> Image.Image | None:
        try:
            return Image.open(BytesIO(content))
        except Exception:
            return None

    def analyze_image(self, image: Image.Image, prompt: str) -> str:
        return (
            "Label: Cannot_Determine\n"
            "Deepfake_Risk: unknown\n"
            "Manipulation_Risk: unknown\n"
            "Visual_Description: The direct fallback path does not invoke the configured Qwen vision model.\n"
            "Evidence:\n"
            "- No vision-capable model is configured.\n"
            "Reasoning: Use the Qwen-backed visual forensic pipeline for image analysis."
        )

    def process_path_or_url(self, image_path_or_url: str | None, selected_language: str) -> str:
        if not image_path_or_url:
            return ""
        image = self.load_image(image_path_or_url)
        return self.process_image(image, selected_language)

    def process_bytes(self, content: bytes | None, selected_language: str) -> str:
        if not content:
            return ""
        image = self.load_image_from_bytes(content)
        return self.process_image(image, selected_language)

    def analyze_path_or_url_with_prompt(self, image_path_or_url: str | None, prompt: str) -> str:
        if not image_path_or_url:
            return ""
        image = self.load_image(image_path_or_url)
        return self.analyze_loaded_image_with_prompt(image, prompt)

    def analyze_bytes_with_prompt(self, content: bytes | None, prompt: str) -> str:
        if not content:
            return ""
        image = self.load_image_from_bytes(content)
        return self.analyze_loaded_image_with_prompt(image, prompt)

    def process_image(self, image: Image.Image | None, selected_language: str) -> str:
        if not image:
            return ""
        resized = self.resize_image(image)
        prompt = self.prompt_service.get_image_prompt(selected_language)
        analysis = self.analyze_image(resized, prompt)
        return analysis.replace("*", "").replace("\n", " ").strip()

    def analyze_loaded_image_with_prompt(self, image: Image.Image | None, prompt: str) -> str:
        if not image:
            return ""
        resized = self.resize_image(image)
        analysis = self.analyze_image(resized, prompt)
        return analysis.replace("*", "").strip()

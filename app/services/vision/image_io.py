import base64
import io
from io import BytesIO

import requests
from PIL import Image

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
            "Visual_Description: DeepSeek chat is configured as the only LLM provider and does not process image pixels in this demo.\n"
            "Evidence:\n"
            "- No vision-capable model is configured.\n"
            "Reasoning: Image analysis is disabled until a DeepSeek-compatible vision model or non-LLM image model is added."
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

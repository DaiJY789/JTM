from app.core.language import is_zh
from app.agents.prompts.image import IMAGE_FORENSIC_PROMPT_EN, IMAGE_FORENSIC_PROMPT_ZH


class PromptService:
    def get_image_prompt(self, selected_language: str) -> str:
        return IMAGE_FORENSIC_PROMPT_ZH if is_zh(selected_language) else IMAGE_FORENSIC_PROMPT_EN

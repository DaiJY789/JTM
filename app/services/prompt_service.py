from app.core.language import is_zh
from app.agents.prompts.ifd import IFD_PROMPT_EN, IFD_PROMPT_ZH


class PromptService:
    def get_image_prompt(self, selected_language: str) -> str:
        return IFD_PROMPT_ZH if is_zh(selected_language) else IFD_PROMPT_EN

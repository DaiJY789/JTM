ZH_ALIASES = {"zh", "cn", "ch", "中文", "汉语", "Chinese", "chinese"}
EN_ALIASES = {"en", "英文", "英语", "English", "english"}


def normalize_language(language: str | None) -> str:
    if not language:
        return "zh"
    value = language.strip()
    if value in ZH_ALIASES:
        return "zh"
    if value in EN_ALIASES:
        return "en"
    return "zh"


def is_zh(language: str | None) -> bool:
    return normalize_language(language) == "zh"

from __future__ import annotations

import re
from typing import Any


class MMDSearchQueryBuilder:
    """Build three whole-post search views without splitting by atomic claim."""

    URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
    SPACE_PATTERN = re.compile(r"\s+")
    SENTENCE_PATTERN = re.compile(r"[。！？!?；;\n]+")
    DATE_NUMBER_PATTERN = re.compile(
        r"(?:\d{4}[年./-]\d{1,2}(?:[月./-]\d{1,2}日?)?|\d+(?:\.\d+)?%?|\d+(?:万|亿|千|百)?)"
    )
    QUOTED_PATTERN = re.compile(r"[“\"『「](.{2,30}?)[”\"』」]")
    HAN_PATTERN = re.compile(r"[一-龥]")
    ASSERTIVE_PATTERN = re.compile(
        r"宣布|证实|否认|取消|发生|导致|已经|正式|发布|上架|推出|启动|实施|逮捕|死亡|赢得|"
        r"announc|confirm|deny|cancel|official|launch|arrest|killed|won|"
        r"bestät|offiziell|annonc|confirme",
        re.IGNORECASE,
    )
    VISUAL_ONLY_PATTERN = re.compile(
        r"图片|配图|画面|截图|图注|照片|image|photo|screenshot|bild|foto",
        re.IGNORECASE,
    )
    GENERIC_HEADINGS = {
        "封面导语",
        "导语",
        "摘要",
        "快讯",
        "新闻",
        "breaking news",
        "summary",
    }

    def __init__(self, max_query_characters: int = 120):
        self.max_query_characters = max(40, int(max_query_characters))

    def build_text_queries(self, text: str, question: str = "") -> list[dict[str, str]]:
        cleaned = self._clean(text)
        sentences = self._sentences(text)
        if not cleaned:
            return []

        headline = self._headline(text, sentences)
        lead = next(
            (
                item
                for item in sentences
                if item != headline and item.lower() not in self.GENERIC_HEADINGS
            ),
            "",
        )
        overall = self._truncate(f"{headline} {lead}".strip() or cleaned)
        detail_sentence = self._best_detail_sentence(sentences, exclude=headline)
        detail = self._detail_query(cleaned, detail_sentence, question)
        return [
            {"query_id": "Q1", "strategy": "whole_post", "query": overall},
            {"query_id": "Q2", "strategy": "key_detail", "query": detail},
        ]

    def build_crossmodal_query(
        self,
        text: str,
        ocr_result: dict[str, Any] | None,
        question: str = "",
    ) -> dict[str, str]:
        cleaned = self._clean(text)
        ocr = ocr_result if isinstance(ocr_result, dict) else {}
        entity_names = []
        for item in ocr.get("entities", []) if isinstance(ocr.get("entities"), list) else []:
            if not isinstance(item, dict):
                continue
            name = self._clean(item.get("name"))
            if name and name not in entity_names:
                entity_names.append(name)
        ocr_text = self._clean(ocr.get("ocr_text"))
        visual_terms = " ".join(entity_names[:5])
        if ocr_text:
            visual_terms = f"{visual_terms} {self._truncate(ocr_text, 55)}".strip()

        sentences = self._sentences(text)
        headline = self._headline(text, sentences)
        anchor = self._truncate(headline or cleaned, 90)
        if visual_terms:
            query = f"{anchor} {visual_terms}".strip()
            strategy = "crossmodal_ocr"
        else:
            source_terms = (
                "原始来源 官方通报 事实核查"
                if self.HAN_PATTERN.search(cleaned)
                else "original source official statement fact check"
            )
            query = f"{anchor} {source_terms}".strip()
            strategy = "source_verification"
        if not query and question:
            query = self._clean(question)
        return {
            "query_id": "Q3",
            "strategy": strategy,
            "query": self._truncate(query),
        }

    def _detail_query(self, text: str, detail_sentence: str, question: str) -> str:
        anchors = []
        anchors.extend(
            match.group(0)
            for match in self.DATE_NUMBER_PATTERN.finditer(detail_sentence)
            if self._informative_numeric_anchor(match.group(0))
        )
        anchors.extend(
            match.group(1) for match in self.QUOTED_PATTERN.finditer(detail_sentence)
        )
        anchors = [
            item
            for item in dict.fromkeys(self._clean(item) for item in anchors if self._clean(item))
            if item not in detail_sentence
        ][:3]

        detail = detail_sentence or self._truncate(text, 90)
        if anchors:
            detail = f"{detail} {' '.join(anchors)}"
        elif question and len(detail) < 50:
            detail = f"{detail} {self._clean(question)}"
        if self._clean(detail) == self._clean(text):
            detail = f"{self._truncate(detail, 100)} 信息来源"
        return self._truncate(detail)

    def _best_detail_sentence(self, sentences: list[str], *, exclude: str) -> str:
        candidates = [item for item in sentences if item and item != exclude]
        if not candidates:
            candidates = sentences
        if not candidates:
            return ""

        def score(sentence: str) -> tuple[int, int]:
            dates = sum(
                self._informative_numeric_anchor(value)
                for value in self.DATE_NUMBER_PATTERN.findall(sentence)
            )
            quotations = len(self.QUOTED_PATTERN.findall(sentence))
            assertions = len(self.ASSERTIVE_PATTERN.findall(sentence))
            visual_penalty = 80 if self.VISUAL_ONLY_PATTERN.search(sentence) else 0
            length_score = min(len(sentence), 140) - abs(min(len(sentence), 140) - 70) // 2
            return (
                assertions * 60 + quotations * 35 + dates * 20 + length_score - visual_penalty,
                len(sentence),
            )

        return max(candidates, key=score)

    def _sentences(self, text: str) -> list[str]:
        result = []
        for item in self.SENTENCE_PATTERN.split(text):
            sentence = self._clean(item)
            if len(sentence) >= 4 and sentence not in result:
                result.append(sentence)
        return result

    def _headline(self, raw_text: str, sentences: list[str]) -> str:
        for raw_line in str(raw_text or "").splitlines():
            line = self._clean(raw_line)
            if (
                4 <= len(line) <= 90
                and line.lower() not in self.GENERIC_HEADINGS
            ):
                return line
        return sentences[0] if sentences else ""

    def _informative_numeric_anchor(self, value: str) -> bool:
        text = str(value or "")
        return bool(
            len(text) >= 4
            or any(marker in text for marker in ("年", "月", "日", "%", ".", "/", "-", "万", "亿"))
        )

    def _clean(self, value: Any) -> str:
        text = self.URL_PATTERN.sub(" ", str(value or ""))
        text = text.replace("#", " ").replace("@", " ")
        return self.SPACE_PATTERN.sub(" ", text).strip(" ，,。；;：:")

    def _truncate(self, value: str, limit: int | None = None) -> str:
        size = limit or self.max_query_characters
        text = self._clean(value)
        return text if len(text) <= size else text[:size].rstrip()


__all__ = ["MMDSearchQueryBuilder"]

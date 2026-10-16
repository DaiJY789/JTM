import re
import uuid
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

from app.agents.IFD_Agent import IFDDetectionAgent, IFDModelUnavailableError
from app.core.language import normalize_language
from app.services.visual.image_io import ImageService


DEFAULT_IMAGE_QUESTION_ZH = "请判断这张图片是真实图像，伪造图像还是合成图像。"
DEFAULT_IMAGE_QUESTION_EN = "Please determine whether this image is a real image, a forged image, or a synthetic image."
MAX_REMOTE_IMAGE_BYTES = 15_000_000
REMOTE_IMAGE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
    ),
    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
    "Referer": "https://mp.weixin.qq.com/",
}


class IFDDetectionService:
    LABELS = [
        "真实图像",
        "伪造图像",
        "合成图像",
        "Real_Image",
        "Forged_Image",
        "Synthetic_Image",
        "Cannot_Determine",
    ]
    LEGACY_LABEL_ALIASES = {
        "Real": "真实图像",
        "Likely_Real": "真实图像",
        "Deepfake_Suspected": "合成图像",
        "AI_Manipulated": "合成图像",
        "AI_Generated": "合成图像",
        "Local_Tampering": "伪造图像",
        "Manipulated": "伪造图像",
        "局部篡改": "伪造图像",
        "AI操纵": "合成图像",
        "AI生成": "合成图像",
        "完全真实": "真实图像",
        "存在伪造": "伪造图像",
        "完全AI生成": "合成图像",
        "Completely_Real": "Real_Image",
        "Forgery_Detected": "Forged_Image",
        "Completely_AI_Generated": "Synthetic_Image",
        "authentic": "Real_Image",
        "tampered": "Forged_Image",
        "ai_manipulated": "Synthetic_Image",
        "ai_generated": "Synthetic_Image",
    }

    def __init__(self, image_service: ImageService, ifd_agent: IFDDetectionAgent | None = None):
        self.image_service = image_service
        self.ifd_agent = ifd_agent

    def detect(
        self,
        image_content: bytes | None = None,
        image_url: str | None = None,
        question: str | None = None,
        language: str = "zh",
        visual_query: str | None = None,
        planning_context: dict[str, Any] | None = None,
        image_path: str | None = None,
    ) -> dict[str, Any]:
        language = normalize_language(language)
        resolved_question = (question or "").strip() or (
            DEFAULT_IMAGE_QUESTION_ZH if language == "zh" else DEFAULT_IMAGE_QUESTION_EN
        )
        resolved_image_path = self._materialize_image(image_content, image_url, image_path)
        if not resolved_image_path:
            message = "无法读取图像输入。" if language == "zh" else "Unable to read the image input."
            return self._error_result(resolved_question, message)

        agent = self.ifd_agent or IFDDetectionAgent(
            config=self.image_service.config,
            model_factory=self.image_service.model_factory,
        )
        try:
            result = agent.detect(
                image_path=resolved_image_path,
                question=resolved_question,
                language=language,
                visual_query=visual_query,
                planning_context=planning_context,
            )
        except IFDModelUnavailableError:
            message = (
                "服务器繁忙，检测暂时失败，请稍后重试。"
                if language == "zh"
                else "The server is busy and detection failed. Please try again later."
            )
            return {
                "mode": "image_only",
                "label": None,
                "summary": message,
                "confidence": None,
                "decision_status": "technical_error",
                "image_verdict": None,
                "evidence": [],
                "claims": [],
                "raw": {"workflow": {"name": "deepfake_detection", "steps": ["IFD_Agent"]}},
                "question": resolved_question,
                "user_answer": {"judgment": message, "reason_analysis": message},
            }

        label = self._extract_label(result.get("label"))
        tool_evidence = result.get("tool_evidence") or result.get("evidence_chain", [])
        suspicious_regions = result.get("suspicious_regions", [])
        final_summary = result.get("final_summary", "")
        evidence = self._evidence_from_chain(tool_evidence)
        return {
            "mode": "image_only",
            "label": label,
            "summary": final_summary,
            "confidence": result.get("confidence"),
            "image_verdict": {
                "label": label,
                "analysis": final_summary,
                "question": resolved_question,
                "deepfake_risk": "high" if label in {"合成图像", "Synthetic_Image", "完全AI生成", "Completely_AI_Generated"} else "low",
                "manipulation_risk": "high"
                if label in {"伪造图像", "合成图像", "Forged_Image", "Synthetic_Image", "存在伪造", "完全AI生成", "Forgery_Detected", "Completely_AI_Generated"}
                else "low",
                "probability": result.get("probability"),
                "confidence": result.get("confidence"),
                "tamper_score": result.get("tamper_score"),
                "ai_generated_score": result.get("ai_generated_score"),
                "visual_analysis": result.get("visual_analysis", ""),
                "visual_description": "基于本地图像取证工具生成的综合判定。",
                "evidence": evidence,
                "reasoning": final_summary,
                "forgery_types": result.get("forgery_types", []),
                "tool_evidence": tool_evidence,
                "suspicious_regions": suspicious_regions,
                "limitations": result.get("limitations", []),
                "artifacts": result.get("artifacts", {}),
                "user_answer": result.get("user_answer", {}),
            },
            "evidence": self._structured_evidence(tool_evidence, suspicious_regions),
            "claims": [],
            "raw": {"image_forensics": result},
            "question": resolved_question,
            "user_answer": result.get("user_answer", {}),
        }

    def _materialize_image(
        self,
        image_content: bytes | None,
        image_url: str | None,
        image_path: str | None = None,
    ) -> str:
        if image_path:
            candidate = Path(image_path)
            if candidate.is_file():
                return str(candidate)
        output_dir = Path(getattr(self.image_service.config, "vision_output_dir", "outputs/vision")) / "inputs"
        output_dir.mkdir(parents=True, exist_ok=True)
        if image_content:
            return self._save_original_image_bytes(image_content, output_dir)
        if image_url:
            candidate = Path(image_url)
            if candidate.exists():
                return str(candidate)
            try:
                response = requests.get(image_url, timeout=10, headers=REMOTE_IMAGE_HEADERS, stream=True)
                response.raise_for_status()
            except Exception:
                return ""
            content = self._read_limited_image_bytes(response)
            return self._save_original_image_bytes(content, output_dir) if content else ""
        return ""

    def _read_limited_image_bytes(self, response: requests.Response) -> bytes:
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=64_000):
            if not chunk:
                continue
            total += len(chunk)
            if total > MAX_REMOTE_IMAGE_BYTES:
                return b""
            chunks.append(chunk)
        return b"".join(chunks)

    def _save_original_image_bytes(self, content: bytes, output_dir: Path) -> str:
        try:
            with Image.open(BytesIO(content)) as image:
                image.verify()
                suffix = self._suffix_for_format(image.format)
        except Exception:
            return ""
        path = output_dir / f"{uuid.uuid4().hex}{suffix}"
        path.write_bytes(content)
        return str(path)

    def _suffix_for_format(self, image_format: str | None) -> str:
        return {
            "JPEG": ".jpg",
            "PNG": ".png",
            "WEBP": ".webp",
            "TIFF": ".tif",
            "BMP": ".bmp",
            "GIF": ".gif",
            "HEIF": ".heif",
            "HEIC": ".heic",
        }.get(str(image_format or "").upper(), ".img")

    def _evidence_from_chain(self, evidence_chain: list[dict[str, Any]]) -> list[str]:
        items = []
        for step in evidence_chain:
            findings = step.get("key_findings", [])
            if findings:
                items.append(f"{step.get('tool')}: {findings[0]}")
        return items

    def _structured_evidence(
        self,
        evidence_chain: list[dict[str, Any]],
        suspicious_regions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        evidence = [
            {
                "type": "visual_tool_evidence",
                "source": step.get("tool"),
                "content": step.get("key_findings", []),
            }
            for step in evidence_chain
        ]
        evidence.extend(
            {
                "type": "suspicious_region",
                "content": region,
            }
            for region in suspicious_regions
        )
        return evidence

    def _error_result(self, question: str | None, message: str) -> dict[str, Any]:
        return {
            "mode": "image_only",
            "label": "Cannot_Determine",
            "summary": message,
            "confidence": 0.0,
            "image_verdict": {
                "label": "Cannot_Determine",
                "analysis": message,
                "question": question or "",
                "deepfake_risk": "unknown",
                "manipulation_risk": "unknown",
                "visual_description": "",
                "evidence": [],
                "reasoning": message,
                "suspicious_regions": [],
            },
            "evidence": [],
            "claims": [],
            "raw": {},
            "question": question or "",
        }

    def _extract_label(self, analysis: str) -> str:
        analysis = str(analysis or "")
        if analysis in self.LABELS:
            return analysis
        if analysis in self.LEGACY_LABEL_ALIASES:
            return self.LEGACY_LABEL_ALIASES[analysis]
        for label in self.LABELS:
            if label in analysis:
                return label
        for label, normalized in self.LEGACY_LABEL_ALIASES.items():
            if label in analysis:
                return normalized
        match = re.search(r"Label\s*[:：]\s*([A-Za-z_]+)", analysis)
        return self.LEGACY_LABEL_ALIASES.get(match.group(1), "Cannot_Determine") if match else "Cannot_Determine"

    def _extract_field(self, analysis: str, field_name: str) -> str:
        match = re.search(rf"{field_name}\s*[:：]\s*([^\n\r]+)", analysis)
        return match.group(1).strip() if match else "unknown"

    def _extract_block(self, analysis: str, field_name: str) -> str:
        known_fields = [
            "Label",
            "Deepfake_Risk",
            "Manipulation_Risk",
            "Visual_Description",
            "Evidence",
            "Reasoning",
        ]
        next_fields = "|".join([re.escape(field) for field in known_fields if field != field_name])
        pattern = rf"{field_name}\s*[:：]\s*(.*?)(?=\n\s*(?:{next_fields})\s*[:：]|\Z)"
        match = re.search(pattern, analysis, re.S)
        return self._clean_block(match.group(1)) if match else ""

    def _extract_list_field(self, analysis: str, field_name: str) -> list[str]:
        block = self._extract_block(analysis, field_name)
        if not block:
            return []
        items = []
        for line in block.splitlines():
            cleaned = re.sub(r"^\s*[-*•\d.、)）]+\s*", "", line).strip()
            if cleaned:
                items.append(cleaned)
        if len(items) <= 1 and ("；" in block or ";" in block):
            items = [item.strip() for item in re.split(r"[；;]", block) if item.strip()]
        return items

    def _clean_block(self, value: str) -> str:
        return re.sub(r"\s+\n", "\n", value).strip()

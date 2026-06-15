import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

from PIL import Image


def _bounded_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return round(max(0.0, min(1.0, number)), 4)


def _load_json_text(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {}


def _response(
    tool_name: str,
    start_time: float,
    success: bool,
    status: str,
    description: str,
    result: dict[str, Any] | None = None,
    error_message: str | None = None,
) -> dict[str, Any]:
    return {
        "success": success,
        "status": status,
        "description": description,
        "result": result or {},
        "error_message": error_message,
        "metadata": {
            "tool_name": tool_name,
            "execution_time_ms": int((time.time() - start_time) * 1000),
        },
    }


def _normalize_region(value: Any) -> str:
    if isinstance(value, dict):
        if all(key in value for key in ("x0", "y0", "x1", "y1")):
            x0, y0, x1, y1 = [int(round(float(value[key]))) for key in ("x0", "y0", "x1", "y1")]
            return _format_region(x0, y0, x1, y1)
        if all(key in value for key in ("x", "y", "w", "h")):
            x = int(round(float(value["x"])))
            y = int(round(float(value["y"])))
            w = int(round(float(value["w"])))
            h = int(round(float(value["h"])))
            return _format_region(x, y, x + w, y + h)
        if "region" in value:
            return _normalize_region(value["region"])
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        x0, y0, x1, y1 = [int(round(float(v))) for v in value[:4]]
        return _format_region(x0, y0, x1, y1)
    return str(value or "")


def _format_region(x0: int, y0: int, x1: int, y1: int) -> str:
    x0, x1 = sorted((max(0, x0), max(0, x1)))
    y0, y1 = sorted((max(0, y0), max(0, y1)))
    return f"{x0},{y0},{x1},{y1}" if x1 > x0 and y1 > y0 else ""


def _regions_from_mask(mask_path: str, image_path: str, score: float) -> list[dict[str, Any]]:
    if not mask_path or not Path(mask_path).exists():
        return []
    try:
        import numpy as np

        mask = Image.open(mask_path).convert("L")
        image = Image.open(image_path)
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.BILINEAR)
        arr = np.asarray(mask, dtype=np.float32) / 255.0
        threshold = max(0.35, float(np.percentile(arr, 95)) * 0.75)
        ys, xs = np.where(arr >= threshold)
        if len(xs) < 16 or len(ys) < 16:
            return []
        return [
            {
                "region": _format_region(int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
                "score": score,
                "reason": "外部定位模型 mask 高响应区域。",
                "coordinate_space": "original_image",
                "source_coordinate_space": "mask_resized_to_original",
                "region_reliability": "mask_bbox",
            }
        ]
    except Exception:
        return []


class ExternalCommandDetector:
    """Adapter for model repos whose inference entrypoint is configured as a command template."""

    def __init__(
        self,
        tool_name: str,
        command_template: str,
        score_key: str,
        detector_family: str,
        description: str,
        timeout_seconds: float = 120.0,
    ):
        self.tool_name = tool_name
        self.command_template = command_template.strip()
        self.score_key = score_key
        self.detector_family = detector_family
        self.description = description
        self.timeout_seconds = timeout_seconds

    def __call__(self, args_str: str) -> dict[str, Any]:
        start_time = time.time()
        try:
            args = json.loads(args_str or "{}")
        except json.JSONDecodeError:
            return _response(self.tool_name, start_time, False, "error", self.description, error_message="工具参数必须是 JSON 字符串。")

        image_path = str(args.get("image_path") or args.get("image_input") or "")
        output_dir = Path(str(args.get("output_dir") or f"outputs/vision/{self.tool_name.lower()}"))
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self.tool_name.lower()}_result.json"

        if not image_path or not Path(image_path).exists():
            return _response(self.tool_name, start_time, False, "error", self.description, error_message=f"找不到图像文件: {image_path}")
        if not self.command_template:
            return _response(
                self.tool_name,
                start_time,
                False,
                "unavailable",
                self.description,
                result={"forensics_report": {"method": self.detector_family, "summary": "外部模型命令未配置。"}},
                error_message=f"未配置 {self.tool_name} 命令模板。",
            )

        command = self.command_template.format(
            image_path=shlex.quote(image_path),
            image_input=shlex.quote(image_path),
            output_dir=shlex.quote(str(output_dir)),
            json_path=shlex.quote(str(json_path)),
        )
        try:
            completed = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except FileNotFoundError as exc:
            return _response(self.tool_name, start_time, False, "unavailable", self.description, error_message=f"命令不可用: {exc}")
        except subprocess.TimeoutExpired:
            return _response(self.tool_name, start_time, False, "error", self.description, error_message="外部模型推理超时。")

        payload = _load_json_text(json_path.read_text(encoding="utf-8") if json_path.exists() else completed.stdout)
        if completed.returncode != 0 and not payload:
            return _response(
                self.tool_name,
                start_time,
                False,
                "error",
                self.description,
                error_message=(completed.stderr or completed.stdout or "外部模型命令执行失败。")[:1000],
            )
        result = self._normalize_payload(payload, image_path, output_dir)
        result["external_command"] = {
            "returncode": completed.returncode,
            "stderr": (completed.stderr or "")[:1000],
            "json_path": str(json_path) if json_path.exists() else "",
        }
        return _response(self.tool_name, start_time, True, "success", self.description, result=result)

    def _normalize_payload(self, payload: dict[str, Any], image_path: str, output_dir: Path) -> dict[str, Any]:
        score = _bounded_float(
            payload.get(self.score_key)
            or payload.get("score")
            or payload.get("probability")
            or payload.get("fake_probability")
            or payload.get("tamper_score")
            or payload.get("ai_generated_score"),
            0.0,
        )
        artifacts = {
            key: str(value)
            for key, value in (payload.get("artifacts") or {}).items()
            if isinstance(value, str) and value
        }
        for key in ["mask_path", "heatmap_path", "map_path", "reliability_path", "residual_path"]:
            value = payload.get(key)
            if isinstance(value, str) and value:
                artifacts[key] = value

        regions = []
        for region in payload.get("candidate_regions") or payload.get("suspicious_regions") or []:
            normalized = _normalize_region(region)
            if normalized:
                regions.append(
                    {
                        "region": normalized,
                        "score": _bounded_float(region.get("score") if isinstance(region, dict) else score, score),
                        "reason": str(region.get("reason") if isinstance(region, dict) else "外部模型输出的可疑区域。"),
                        "coordinate_space": str(region.get("coordinate_space") or "original_image") if isinstance(region, dict) else "original_image",
                        "region_reliability": str(region.get("region_reliability") or "external_model_region") if isinstance(region, dict) else "external_model_region",
                    }
                )
        if not regions:
            regions = _regions_from_mask(artifacts.get("mask_path", ""), image_path, score)

        return {
            "forensics_report": {
                "method": self.detector_family,
                "summary": str(payload.get("summary") or self.description),
                "top_suspicious_regions": regions[:5],
            },
            "heuristic_scores": {self.score_key: score},
            "evidence_scores": {self.score_key: score},
            "detector_family": self.detector_family,
            "raw_model_output": payload,
            "artifacts": artifacts,
        }


class TruForDetector(ExternalCommandDetector):
    def __init__(self, command_template: str = "", timeout_seconds: float = 120.0):
        super().__init__(
            tool_name="TruFor_Detector",
            command_template=command_template,
            score_key="trufor_tamper_score",
            detector_family="TruFor image forgery localization",
            description="TruFor 深度模型篡改检测与像素级定位。",
            timeout_seconds=timeout_seconds,
        )


class CATNetDetector(ExternalCommandDetector):
    def __init__(self, command_template: str = "", timeout_seconds: float = 120.0):
        super().__init__(
            tool_name="CATNet_Detector",
            command_template=command_template,
            score_key="catnet_compression_tamper_score",
            detector_family="CAT-Net compression artifact tracing",
            description="CAT-Net 压缩伪影篡改定位检测。",
            timeout_seconds=timeout_seconds,
        )


class DIREDetector(ExternalCommandDetector):
    def __init__(self, command_template: str = "", timeout_seconds: float = 120.0):
        super().__init__(
            tool_name="DIRE_Detector",
            command_template=command_template,
            score_key="dire_ai_generated_score",
            detector_family="DIRE diffusion-generated image detection",
            description="DIRE 扩散模型生成图像检测。",
            timeout_seconds=timeout_seconds,
        )


class C2PAProvenanceChecker:
    def __init__(self, command_template: str = "c2patool {image_path} --json", timeout_seconds: float = 30.0):
        self.tool_name = "C2PA_Provenance_Checker"
        self.command_template = command_template.strip()
        self.timeout_seconds = timeout_seconds

    def __call__(self, args_str: str) -> dict[str, Any]:
        start_time = time.time()
        try:
            args = json.loads(args_str or "{}")
        except json.JSONDecodeError:
            return _response(self.tool_name, start_time, False, "error", "C2PA 来源凭证检查。", error_message="工具参数必须是 JSON 字符串。")

        image_path = str(args.get("image_path") or args.get("image_input") or "")
        if not image_path or not Path(image_path).exists():
            return _response(self.tool_name, start_time, False, "error", "C2PA 来源凭证检查。", error_message=f"找不到图像文件: {image_path}")
        if not self.command_template:
            return _response(self.tool_name, start_time, False, "unavailable", "C2PA 来源凭证检查。", error_message="未配置 C2PA 命令模板。")

        executable = shlex.split(self.command_template)[0]
        if shutil.which(executable) is None and not Path(executable).exists():
            return _response(self.tool_name, start_time, False, "unavailable", "C2PA 来源凭证检查。", error_message=f"未找到 C2PA 工具: {executable}")

        command = self.command_template.format(image_path=shlex.quote(image_path), image_input=shlex.quote(image_path))
        try:
            completed = subprocess.run(
                shlex.split(command),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return _response(self.tool_name, start_time, False, "error", "C2PA 来源凭证检查。", error_message="C2PA 检查超时。")

        payload = _load_json_text(completed.stdout)
        result = self._normalize_payload(payload, completed.returncode, completed.stderr)
        return _response(self.tool_name, start_time, True, "success", "C2PA 来源凭证检查。", result=result)

    def _normalize_payload(self, payload: dict[str, Any], returncode: int, stderr: str) -> dict[str, Any]:
        manifest_store = payload.get("manifest_store") if isinstance(payload, dict) else {}
        manifests = manifest_store.get("manifests", {}) if isinstance(manifest_store, dict) else {}
        active_manifest = manifest_store.get("active_manifest") if isinstance(manifest_store, dict) else ""
        has_c2pa = bool(manifests or active_manifest)
        active = manifests.get(active_manifest, {}) if isinstance(manifests, dict) else {}
        assertions = active.get("assertions", []) if isinstance(active, dict) else []
        assertion_text = json.dumps(assertions, ensure_ascii=False).lower()
        declared_ai = any(token in assertion_text for token in ["ai_generated", "ai generated", "generative", "synthetic", "c2pa.created"])
        edited = any(token in assertion_text for token in ["edited", "actions", "ingredient", "photoshop"])
        provenance_score = 1.0 if declared_ai else (0.65 if edited else 0.0)
        summary = (
            "C2PA manifest declares generative or synthetic content."
            if declared_ai
            else ("C2PA manifest is present and indicates editing/provenance actions." if edited else ("C2PA manifest is present." if has_c2pa else "No C2PA manifest was found."))
        )
        return {
            "forensics_report": {
                "method": "C2PA provenance metadata",
                "summary": summary,
                "top_suspicious_regions": [],
            },
            "heuristic_scores": {
                "c2pa_ai_generated_score": provenance_score if declared_ai else 0.0,
                "c2pa_edit_score": provenance_score if edited and not declared_ai else 0.0,
            },
            "evidence_scores": {
                "c2pa_ai_generated_score": provenance_score if declared_ai else 0.0,
                "c2pa_edit_score": provenance_score if edited and not declared_ai else 0.0,
            },
            "provenance": {
                "has_c2pa": has_c2pa,
                "active_manifest": active_manifest,
                "declared_ai_generated": declared_ai,
                "declared_edited": edited,
                "returncode": returncode,
                "stderr": (stderr or "")[:1000],
            },
            "artifacts": {},
        }

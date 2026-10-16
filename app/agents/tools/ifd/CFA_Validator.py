import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

import numpy as np
from PIL import Image
from app.agents.tools.ifd.common import (
    build_call_output_dir,
    detector_contract,
    parse_bool,
    resolve_trusted_image_path,
    strip_model_controlled_io,
)
from app.agents.tools.foundation import (
    BaseTool,
    OpenAIFunctionParametersSchema,
    OpenAIFunctionPropertySchema,
    OpenAIFunctionSchema,
    OpenAIFunctionToolSchema,
    ToolResponse,
    rollout_trace_op,
)

try:
    from work.tool_llm.tool_use_logger import log_tool_use_record
except Exception:

    def log_tool_use_record(**kwargs):  # type: ignore
        return None


logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VISION_LOGGING_LEVEL", "WARN"))


def _default_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema(
        type="function",
        function=OpenAIFunctionSchema(
            name="cfa_validator",
            description=(
                "Original-camera-only Bayer CFA consistency check. Returns not_applicable/inconclusive when "
                "camera origin or a dominant CFA phase is unavailable."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "window_size": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Local consistency window size (odd number, default 7, range 3-15).",
                    ),
                    "prob_threshold": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Discontinuity probability threshold (default 0.65, range 0.30-0.95).",
                    ),
                    "max_side": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Maximum original image side accepted without resampling, range 256-4096.",
                    ),
                },
                required=[],
            ),
            strict=False,
        ),
    )


def _clamp_int(value: Any, low: int, high: int, default: int) -> int:
    try:
        ivalue = int(round(float(value)))
    except Exception:
        ivalue = default
    return max(low, min(high, ivalue))


def _clamp_float(value: Any, low: float, high: float, default: float) -> float:
    try:
        fvalue = float(value)
    except Exception:
        fvalue = default
    return max(low, min(high, fvalue))


def _normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = np.nan_to_num(arr.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    lo = float(np.min(arr))
    hi = float(np.percentile(arr, 99.0))
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def _conv2d_reflect(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(arr, ((ph, ph), (pw, pw)), mode="reflect")
    out = np.zeros_like(arr, dtype=np.float32)

    for i in range(kh):
        for j in range(kw):
            coeff = float(kernel[i, j])
            if abs(coeff) < 1e-12:
                continue
            out += coeff * padded[i : i + arr.shape[0], j : j + arr.shape[1]]
    return out


def _box_mean(arr: np.ndarray, win: int) -> np.ndarray:
    kernel = np.ones((win, win), dtype=np.float32) / float(win * win)
    return _conv2d_reflect(arr.astype(np.float32), kernel)


def _bilinear_fill(channel: np.ndarray, mask: np.ndarray) -> np.ndarray:
    k = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    weighted = _conv2d_reflect(channel * mask, k)
    weights = _conv2d_reflect(mask, k)
    filled = weighted / (weights + 1e-6)
    known = mask > 0.5
    filled[known] = channel[known]
    return filled


class CFAValidator:
    def __init__(
        self,
        default_window_size: int = 7,
        default_prob_threshold: float = 0.65,
        default_max_side: int = 2048,
        default_export_map: bool = False,
        default_output_dir: str = "./outputs/cfa_validator",
    ):
        self.tool_name = "CFA_Validator"
        self.default_window_size = default_window_size
        self.default_prob_threshold = default_prob_threshold
        self.default_max_side = default_max_side
        self.default_export_map = default_export_map
        self.default_output_dir = default_output_dir

    def _prepare_rgb(self, image: Image.Image, max_side: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        ow, oh = image.size
        if max(ow, oh) > max_side:
            raise ValueError(f"original image side {max(ow, oh)} exceeds CFA limit {max_side}; resampling is forbidden")
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        return rgb, (ow, oh), (ow, oh)

    def _phase_masks(self, h: int, w: int, sr: int, sc: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rr = np.arange(h, dtype=np.int32)[:, None]
        cc = np.arange(w, dtype=np.int32)[None, :]
        pr = (rr + sr) & 1
        pc = (cc + sc) & 1

        r_mask = ((pr == 0) & (pc == 0)).astype(np.float32)
        b_mask = ((pr == 1) & (pc == 1)).astype(np.float32)
        g_mask = 1.0 - r_mask - b_mask
        return r_mask, g_mask, b_mask

    def _phase_reconstruction_error(self, rgb: np.ndarray, sr: int, sc: int, smooth_win: int) -> np.ndarray:
        h, w, _ = rgb.shape
        r_mask, g_mask, b_mask = self._phase_masks(h, w, sr=sr, sc=sc)

        r = rgb[:, :, 0]
        g = rgb[:, :, 1]
        b = rgb[:, :, 2]

        r_hat = _bilinear_fill(r, r_mask)
        g_hat = _bilinear_fill(g, g_mask)
        b_hat = _bilinear_fill(b, b_mask)

        err = (np.abs(r - r_hat) + np.abs(g - g_hat) + np.abs(b - b_hat)) / 3.0
        err = _box_mean(err, smooth_win)
        return err.astype(np.float32)

    def _build_discontinuity_map(
        self,
        phase_errs: np.ndarray,
        window_size: int,
        prob_threshold: float,
    ) -> Tuple[np.ndarray, Dict[str, Any], List[Dict[str, Any]]]:
        # phase_errs: [4, H, W]
        best_idx = np.argmin(phase_errs, axis=0)
        best_err = np.take_along_axis(phase_errs, best_idx[None, :, :], axis=0)[0]

        # second best error for confidence margin
        part = np.partition(phase_errs, kth=1, axis=0)
        second_err = part[1]
        margin = np.maximum(second_err - best_err, 0.0)

        h, w = best_idx.shape
        global_phase_errors = np.mean(phase_errs, axis=(1, 2))
        global_phase = int(np.argmin(global_phase_errors))
        global_phase_mask = (best_idx == global_phase).astype(np.float32)
        global_phase_dominance = float(np.mean(global_phase_mask))
        phase_mismatch = 1.0 - global_phase_mask
        local_phase_mismatch = _box_mean(phase_mismatch, window_size)

        # A mismatch is meaningful only when the local phase choice has a
        # non-trivial error margin over the second-best phase.
        margin_confidence = np.clip(margin / (second_err + 1e-6), 0.0, 1.0).astype(np.float32)

        e_lo = float(np.percentile(best_err, 5.0))
        e_hi = float(np.percentile(best_err, 95.0))
        if e_hi - e_lo < 1e-8:
            err_norm = np.zeros_like(best_err, dtype=np.float32)
        else:
            err_norm = np.clip((best_err - e_lo) / (e_hi - e_lo), 0.0, 1.0).astype(np.float32)

        prob = phase_mismatch * (0.55 * local_phase_mismatch + 0.30 * margin_confidence + 0.15 * err_norm)
        prob = np.clip(prob, 0.0, 1.0).astype(np.float32)

        adaptive_thr = float(prob_threshold)
        hot_ratio = float(np.mean(prob >= adaptive_thr))
        extreme_ratio = float(np.mean(prob >= min(1.0, adaptive_thr + 0.12)))
        hot_support = min(1.0, hot_ratio / 0.08)
        extreme_support = min(1.0, extreme_ratio / 0.025)
        mismatch_support = min(1.0, float(np.mean(local_phase_mismatch)) / 0.45)
        phase_reliability = max(0.0, min(1.0, (global_phase_dominance - 0.40) / 0.30))
        tamper_probability = phase_reliability * (
            0.45 * hot_support + 0.30 * extreme_support + 0.25 * mismatch_support
        )
        tamper_probability = max(0.0, min(1.0, tamper_probability))

        gh, gw = 3, 3
        top_regions: List[Dict[str, Any]] = []
        for gy in range(gh):
            for gx in range(gw):
                y0 = int(round(gy * h / gh))
                y1 = int(round((gy + 1) * h / gh))
                x0 = int(round(gx * w / gw))
                x1 = int(round((gx + 1) * w / gw))
                block = prob[y0:y1, x0:x1]
                score = float(np.mean(block)) if block.size > 0 else 0.0
                top_regions.append(
                    {
                        "bbox": [x0, y0, x1, y1],
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "coordinate_space": "original",
                        "score": round(score, 6),
                    }
                )

        top_regions.sort(key=lambda x: x["score"], reverse=True)

        stats = {
            "hot_ratio": hot_ratio,
            "extreme_ratio": extreme_ratio,
            "adaptive_threshold": adaptive_thr,
            "global_phase": global_phase,
            "global_phase_dominance": global_phase_dominance,
            "phase_reliability": phase_reliability,
            "mean_phase_inconsistency": float(np.mean(local_phase_mismatch)),
            "mean_margin": float(np.mean(margin)),
            "mean_best_error": float(np.mean(best_err)),
            "tamper_probability": tamper_probability,
        }
        return prob, stats, top_regions[:3]

    def _save_probability_map(self, prob_map: np.ndarray, image_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        u8 = _normalize_to_uint8(prob_map)
        img = Image.fromarray(u8, mode="L").convert("RGB")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_cfa_discontinuity_prob.png")
        img.save(out_path)
        return out_path

    def _build_response(
        self,
        success: bool,
        start_time: float,
        description: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return {
            "success": success,
            "status": status or ("success" if success else "error"),
            "description": description,
            "result": result or {},
            "error_message": error_message,
            "metadata": {
                "tool_name": self.tool_name,
                "execution_time_ms": execution_time_ms,
            },
        }

    def _build_not_applicable_response(
        self,
        start_time: float,
        reason: str,
        source_format: Optional[str],
        original_size: Tuple[int, int],
    ) -> Dict[str, Any]:
        result = {
            "forensics_report": {
                "output_mode": "evidence_only",
                "method": "Bayer CFA phase consistency validation",
                "summary": "输入不满足原始 CFA 痕迹分析条件；该结果不能作为真实性证据。",
            },
            "evidence_contract": detector_contract(
                score=None,
                threshold=None,
                positive=None,
                coordinate_space="original",
                applicable=False,
                applicability_reason=reason,
                score_semantics="none",
            ),
            "regions": [],
            "applicability": {
                "applicable": False,
                "reason": reason,
                "source_format": source_format,
                "original_size": {"width": int(original_size[0]), "height": int(original_size[1])},
            },
            "evidence_scores": {"cfa_inconsistency_score": None},
            "artifacts": {"cfa_discontinuity_probability_map_path": ""},
        }
        return self._build_response(
            True,
            start_time,
            "CFA validation not applicable.",
            result=result,
            status="not_applicable",
        )

    def __call__(self, args_str: str) -> Dict[str, Any]:
        start_time = time.time()
        log_call_args: Dict[str, Any] = {"raw_args": args_str}

        def _finalize(response_obj: Dict[str, Any]) -> Dict[str, Any]:
            log_tool_use_record(
                tool_key="cfa_validator",
                tool_name=self.tool_name,
                call_args=log_call_args,
                response=response_obj,
            )
            return response_obj

        try:
            args = json.loads(args_str)
        except json.JSONDecodeError:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="CFA validation failed.",
                    error_message="工具参数解析失败，必须是有效的 JSON 格式字符串。",
                )
            )

        image_path = args.get("image_path") or args.get("image_input")
        window_size = _clamp_int(args.get("window_size", self.default_window_size), 3, 15, self.default_window_size)
        if window_size % 2 == 0:
            window_size += 1

        prob_threshold = _clamp_float(
            args.get("prob_threshold", self.default_prob_threshold),
            0.30,
            0.95,
            self.default_prob_threshold,
        )
        max_side = _clamp_int(args.get("max_side", self.default_max_side), 256, 4096, self.default_max_side)
        try:
            export_map = parse_bool(args.get("export_map", self.default_export_map), self.default_export_map)
        except ValueError as exc:
            return _finalize(self._build_response(False, start_time, "CFA validation failed.", error_message=str(exc)))
        output_dir = str(args.get("output_dir", self.default_output_dir))

        log_call_args = {
            "image_path": image_path,
            "window_size": window_size,
            "prob_threshold": prob_threshold,
            "max_side": max_side,
            "export_map": export_map,
            "output_dir": output_dir,
        }

        if not image_path:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="CFA validation failed.",
                    error_message="缺少必填参数 'image_path'。",
                )
            )

        if not os.path.exists(image_path):
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="CFA validation failed.",
                    error_message=f"找不到指定的图像文件: {image_path}。请检查文件路径是否正确。",
                )
            )

        try:
            image = Image.open(image_path)
            source_format = str(image.format or "").upper() or None
            orig_size = image.size
            exif = image.getexif()
            has_camera_metadata = bool(exif.get(271) or exif.get(272)) if exif else False

            if source_format not in {"JPEG", "JPG", "TIFF"} and not has_camera_metadata:
                return _finalize(
                    self._build_not_applicable_response(
                        start_time=start_time,
                        reason="no_camera_origin_signal",
                        source_format=source_format,
                        original_size=orig_size,
                    )
                )
            if max(orig_size) > max_side:
                return _finalize(
                    self._build_not_applicable_response(
                        start_time=start_time,
                        reason="original_resolution_exceeds_limit_resampling_forbidden",
                        source_format=source_format,
                        original_size=orig_size,
                    )
                )

            rgb, orig_size, proc_size = self._prepare_rgb(image, max_side=max_side)
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="CFA validation failed.",
                    error_message=f"读取图像失败: {exc}",
                )
            )

        try:
            # 4 Bayer phase hypotheses: shifts of RGGB pattern
            phase_errs = []
            for sr, sc in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                phase_errs.append(self._phase_reconstruction_error(rgb, sr=sr, sc=sc, smooth_win=3))
            phase_errs_arr = np.stack(phase_errs, axis=0)

            prob_map, stats, top_regions = self._build_discontinuity_map(
                phase_errs=phase_errs_arr,
                window_size=window_size,
                prob_threshold=prob_threshold,
            )

            tamper_probability = float(stats["tamper_probability"])
            evidence_score = max(0.0, min(1.0, tamper_probability))
            phase_reliable = bool(float(stats["global_phase_dominance"]) >= 0.55)
            positive = bool(
                phase_reliable
                and evidence_score >= 0.72
                and float(stats["hot_ratio"]) >= 0.02
                and float(stats["extreme_ratio"]) >= 0.003
            )
            decision_value: Optional[bool] = positive if phase_reliable else None
            if not positive:
                top_regions = []

            prob_map_path = ""
            if export_map:
                prob_map_path = self._save_probability_map(prob_map, image_path=image_path, output_dir=output_dir)

            result = {
                "forensics_report": {
                    "output_mode": "evidence_only",
                    "method": "Bayer CFA phase consistency validation",
                    "window_size": window_size,
                    "prob_threshold": round(float(prob_threshold), 4),
                    "top_suspicious_regions": top_regions,
                    "summary": "已输出 CFA 相位不一致统计与局部证据；不包含最终判定结论。",
                },
                "evidence_contract": detector_contract(
                    score=evidence_score,
                    threshold=0.72,
                    positive=decision_value,
                    coordinate_space="original",
                    score_semantics="uncalibrated_cfa_phase_inconsistency",
                ),
                "regions": top_regions,
                "evidence_scores": {
                    "cfa_inconsistency_score": round(float(evidence_score), 6),
                },
                "cfa_statistics": {
                    "hot_ratio": round(float(stats["hot_ratio"]), 6),
                    "extreme_ratio": round(float(stats["extreme_ratio"]), 6),
                    "adaptive_threshold": round(float(stats["adaptive_threshold"]), 6),
                    "global_phase": int(stats["global_phase"]),
                    "global_phase_dominance": round(float(stats["global_phase_dominance"]), 6),
                    "phase_reliability": round(float(stats["phase_reliability"]), 6),
                    "mean_phase_inconsistency": round(float(stats["mean_phase_inconsistency"]), 6),
                    "mean_margin": round(float(stats["mean_margin"]), 6),
                    "mean_best_error": round(float(stats["mean_best_error"]), 6),
                    "processed_size": {
                        "width": int(proc_size[0]),
                        "height": int(proc_size[1]),
                    },
                    "original_size": {
                        "width": int(orig_size[0]),
                        "height": int(orig_size[1]),
                    },
                },
                "artifacts": {
                    "cfa_discontinuity_probability_map_path": prob_map_path,
                },
            }

            return _finalize(
                self._build_response(
                    success=True,
                    start_time=start_time,
                    description="CFA validation completed.",
                    result=result,
                )
            )
        except Exception as exc:
            logger.exception("CFAValidator execution failed.")
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="CFA validation failed.",
                    error_message=f"CFA 相位分析阶段发生错误: {exc}",
                )
            )


class CFAValidatorTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_window_size = int(config.get("default_window_size", 7))
        self._default_prob_threshold = float(config.get("default_prob_threshold", 0.65))
        self._default_max_side = int(config.get("default_max_side", 2048))
        self._default_export_map = bool(config.get("default_export_map", False))
        self._default_output_dir = str(config.get("output_dir", "./outputs/cfa_validator"))
        self._tool_impl = CFAValidator(
            default_window_size=self._default_window_size,
            default_prob_threshold=self._default_prob_threshold,
            default_max_side=self._default_max_side,
            default_export_map=self._default_export_map,
            default_output_dir=self._default_output_dir,
        )

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(
        self,
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        **kwargs,
    ) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())

        default_image_path = kwargs.get("image_input") or kwargs.get("image_path")
        self._instance_dict[instance_id] = {
            "response": "",
            "ground_truth": ground_truth,
            "reward": [],
            "default_image_path": default_image_path,
        }
        return instance_id, ToolResponse()

    @rollout_trace_op
    async def execute(
        self,
        instance_id: str,
        parameters: dict[str, Any],
        env_states: Optional[dict] = None,
        **kwargs,
    ) -> tuple[ToolResponse, float, dict]:
        try:
            instance_state = self._instance_dict.get(instance_id, {})
            clean_parameters = strip_model_controlled_io(parameters)
            call_args = {
                "image_path": resolve_trusted_image_path(instance_state, kwargs),
                "window_size": clean_parameters.get("window_size", self._default_window_size),
                "prob_threshold": clean_parameters.get("prob_threshold", self._default_prob_threshold),
                "max_side": clean_parameters.get("max_side", self._default_max_side),
                "export_map": self._default_export_map,
                "output_dir": build_call_output_dir(self._default_output_dir, instance_id, kwargs),
            }
            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("CFAValidatorTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "CFA validation failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

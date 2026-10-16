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
    map_bbox_to_original,
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
            name="patchdense_scanner",
            description=(
                "Detect suspected copy-move forgery via PatchMatch dense offset field search and "
                "thresholded offset consistency analysis. Use selectively when duplicated regions are suspected because this tool is relatively expensive."
            ),
            parameters=OpenAIFunctionParametersSchema(
                type="object",
                properties={
                    "patch_size": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Patch size for PatchMatch (odd number, default 7, range 5-13).",
                    ),
                    "iterations": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="PatchMatch propagation iterations (default 4, range 2-8).",
                    ),
                    "consistency_threshold": OpenAIFunctionPropertySchema(
                        type="number",
                        description="Threshold for offset consistency map (default 0.60, range 0.30-0.95).",
                    ),
                    "max_side": OpenAIFunctionPropertySchema(
                        type="integer",
                        description="Max image side for internal processing (default 384, range 256-768).",
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


def _box_mean(arr: np.ndarray, win: int = 3) -> np.ndarray:
    pad = win // 2
    padded = np.pad(arr, ((pad, pad), (pad, pad)), mode="reflect")
    integ = np.pad(np.cumsum(np.cumsum(padded, axis=0), axis=1), ((1, 0), (1, 0)), mode="constant")
    h, w = arr.shape
    y0 = np.arange(0, h)
    y1 = y0 + win
    x0 = np.arange(0, w)
    x1 = x0 + win
    out = (
        integ[np.ix_(y1, x1)]
        - integ[np.ix_(y0, x1)]
        - integ[np.ix_(y1, x0)]
        + integ[np.ix_(y0, x0)]
    )
    return out / float(win * win)


class PatchDenseScanner:
    def __init__(
        self,
        default_patch_size: int = 7,
        default_iterations: int = 4,
        default_consistency_threshold: float = 0.60,
        default_max_side: int = 384,
        default_export_map: bool = False,
        default_output_dir: str = "./outputs/patchdense_scanner",
    ):
        self.tool_name = "PatchDense_Scanner"
        self.default_patch_size = default_patch_size
        self.default_iterations = default_iterations
        self.default_consistency_threshold = default_consistency_threshold
        self.default_max_side = default_max_side
        self.default_export_map = default_export_map
        self.default_output_dir = default_output_dir

    def _extract_luma(self, image: Image.Image) -> np.ndarray:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]

    def _prepare_gray(self, image: Image.Image, max_side: int) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
        ow, oh = image.size
        scale = 1.0
        if max(ow, oh) > max_side:
            scale = float(max_side) / float(max(ow, oh))
        nw = max(32, int(round(ow * scale)))
        nh = max(32, int(round(oh * scale)))
        resized = image.resize((nw, nh), Image.BILINEAR)
        gray = self._extract_luma(resized)
        return gray, (ow, oh), (nw, nh)

    def _patch_distance(self, gray: np.ndarray, y: int, x: int, dy: int, dx: int, r: int) -> float:
        y2 = y + dy
        x2 = x + dx
        p1 = gray[y - r : y + r + 1, x - r : x + r + 1]
        p2 = gray[y2 - r : y2 + r + 1, x2 - r : x2 + r + 1]
        diff = p1 - p2
        return float(np.mean(diff * diff))

    def _random_offset(self, y: int, x: int, h: int, w: int, r: int, min_shift: int, rng: np.random.Generator) -> Tuple[int, int]:
        max_try = 20
        for _ in range(max_try):
            y2 = int(rng.integers(r, h - r))
            x2 = int(rng.integers(r, w - r))
            dy = y2 - y
            dx = x2 - x
            if abs(dy) + abs(dx) >= min_shift:
                return dy, dx
        return min_shift, 0

    def _run_patchmatch(
        self,
        gray: np.ndarray,
        patch_size: int,
        iterations: int,
        scan_stride: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        h, w = gray.shape
        r = patch_size // 2
        min_shift = max(3, r + 1)
        rng = np.random.default_rng(20260409)

        dy_map = np.zeros((h, w), dtype=np.int32)
        dx_map = np.zeros((h, w), dtype=np.int32)
        dist_map = np.full((h, w), np.inf, dtype=np.float32)

        step = max(1, scan_stride)
        for y in range(r, h - r, step):
            for x in range(r, w - r, step):
                dy, dx = self._random_offset(y, x, h, w, r, min_shift=min_shift, rng=rng)
                dy_map[y, x] = dy
                dx_map[y, x] = dx
                dist_map[y, x] = self._patch_distance(gray, y, x, dy, dx, r)

        init_radius = max(h, w) // 2

        for it in range(iterations):
            if it % 2 == 0:
                yr = range(r, h - r, step)
                xr = range(r, w - r, step)
                neigh = [(-step, 0), (0, -step)]
            else:
                yr = range(h - r - 1, r - 1, -step)
                xr = range(w - r - 1, r - 1, -step)
                neigh = [(step, 0), (0, step)]

            for y in yr:
                for x in xr:
                    best_dy = int(dy_map[y, x])
                    best_dx = int(dx_map[y, x])
                    best_dist = float(dist_map[y, x])

                    for ny, nx in neigh:
                        yy = y + ny
                        xx = x + nx
                        if yy < r or yy >= h - r or xx < r or xx >= w - r:
                            continue
                        cdy = int(dy_map[yy, xx])
                        cdx = int(dx_map[yy, xx])
                        y2 = y + cdy
                        x2 = x + cdx
                        if y2 < r or y2 >= h - r or x2 < r or x2 >= w - r:
                            continue
                        if abs(cdy) + abs(cdx) < min_shift:
                            continue
                        cd = self._patch_distance(gray, y, x, cdy, cdx, r)
                        if cd < best_dist:
                            best_dist = cd
                            best_dy = cdy
                            best_dx = cdx

                    radius = init_radius
                    while radius >= 1:
                        cdy = best_dy + int(rng.integers(-radius, radius + 1))
                        cdx = best_dx + int(rng.integers(-radius, radius + 1))
                        y2 = y + cdy
                        x2 = x + cdx
                        if y2 >= r and y2 < h - r and x2 >= r and x2 < w - r:
                            if abs(cdy) + abs(cdx) >= min_shift:
                                cd = self._patch_distance(gray, y, x, cdy, cdx, r)
                                if cd < best_dist:
                                    best_dist = cd
                                    best_dy = cdy
                                    best_dx = cdx
                        radius //= 2

                    dy_map[y, x] = best_dy
                    dx_map[y, x] = best_dx
                    dist_map[y, x] = best_dist

        if step > 1:
            yi = (np.arange(h) // step) * step
            xi = (np.arange(w) // step) * step
            yi = np.clip(yi, 0, h - 1)
            xi = np.clip(xi, 0, w - 1)
            dy_map = dy_map[np.ix_(yi, xi)]
            dx_map = dx_map[np.ix_(yi, xi)]
            dist_map = dist_map[np.ix_(yi, xi)]

        return dy_map, dx_map, dist_map

    def _thresholded_offset_consistency(
        self,
        dy_map: np.ndarray,
        dx_map: np.ndarray,
        dist_map: np.ndarray,
        patch_size: int,
        consistency_threshold: float,
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        dy = dy_map.astype(np.float32)
        dx = dx_map.astype(np.float32)

        mean_dy = _box_mean(dy, win=3)
        mean_dx = _box_mean(dx, win=3)
        var_dy = np.maximum(_box_mean(dy * dy, win=3) - mean_dy * mean_dy, 0.0)
        var_dx = np.maximum(_box_mean(dx * dx, win=3) - mean_dx * mean_dx, 0.0)

        offset_var = var_dy + var_dx
        offset_consistency = np.exp(-offset_var / (patch_size * patch_size + 1e-6))

        finite_dist = dist_map[np.isfinite(dist_map)]
        if finite_dist.size == 0:
            norm_dist = np.ones_like(dist_map, dtype=np.float32)
        else:
            d_lo = float(np.percentile(finite_dist, 5.0))
            d_hi = float(np.percentile(finite_dist, 95.0))
            if d_hi - d_lo < 1e-8:
                norm_dist = np.ones_like(dist_map, dtype=np.float32)
            else:
                norm_dist = np.clip((dist_map - d_lo) / (d_hi - d_lo), 0.0, 1.0)

        magnitude = np.sqrt(dx * dx + dy * dy)
        shift_gate = np.clip((magnitude - float(patch_size)) / float(2 * patch_size + 1e-6), 0.0, 1.0)
        similarity = 1.0 - norm_dist

        # Copy-move matches should be approximately reciprocal: if A maps to B,
        # the best match around B should map back towards A. This suppresses
        # random offsets and many repeated-texture false positives.
        h, w = dy_map.shape
        yy, xx = np.indices((h, w))
        target_y = np.rint(yy + dy).astype(np.int32)
        target_x = np.rint(xx + dx).astype(np.int32)
        valid_target = (target_y >= 0) & (target_y < h) & (target_x >= 0) & (target_x < w)
        target_y_clip = np.clip(target_y, 0, h - 1)
        target_x_clip = np.clip(target_x, 0, w - 1)
        reverse_dy = dy[target_y_clip, target_x_clip]
        reverse_dx = dx[target_y_clip, target_x_clip]
        reverse_error = np.sqrt((dy + reverse_dy) ** 2 + (dx + reverse_dx) ** 2)
        reciprocal = np.exp(-reverse_error / (float(patch_size) + 1e-6)).astype(np.float32)
        reciprocal[~valid_target] = 0.0

        suspicion = offset_consistency * similarity * reciprocal * (0.5 + 0.5 * shift_gate)
        suspicion = np.clip(suspicion, 0.0, 1.0).astype(np.float32)

        adaptive_thr = float(consistency_threshold)
        binary = (suspicion >= adaptive_thr).astype(np.float32)
        cluster_ratio = float(np.mean(_box_mean(binary, win=5) >= 0.35))
        hot_ratio = float(np.mean(binary))
        very_hot_ratio = float(np.mean(suspicion >= min(1.0, adaptive_thr + 0.15)))
        reciprocal_ratio = float(np.mean((reciprocal >= 0.65) & (similarity >= adaptive_thr)))
        hot_support = min(1.0, hot_ratio / 0.04)
        very_hot_support = min(1.0, very_hot_ratio / 0.015)
        cluster_support = min(1.0, cluster_ratio / 0.02)
        reciprocal_support = min(1.0, reciprocal_ratio / 0.04)
        tamper_prob = 0.30 * hot_support + 0.25 * very_hot_support + 0.25 * cluster_support + 0.20 * reciprocal_support
        tamper_prob = max(0.0, min(1.0, tamper_prob))

        stats = {
            "hot_ratio": hot_ratio,
            "very_hot_ratio": very_hot_ratio,
            "cluster_ratio": cluster_ratio,
            "adaptive_threshold": adaptive_thr,
            "mean_consistency": float(np.mean(offset_consistency)),
            "mean_similarity": float(np.mean(similarity)),
            "mean_reciprocal_consistency": float(np.mean(reciprocal)),
            "reciprocal_ratio": reciprocal_ratio,
            "tamper_probability": tamper_prob,
        }
        return suspicion, stats

    def _top_regions(self, score_map: np.ndarray, topk: int = 3) -> List[Dict[str, Any]]:
        h, w = score_map.shape
        gh, gw = 3, 3
        regions: List[Dict[str, Any]] = []
        for gy in range(gh):
            for gx in range(gw):
                y0 = int(round(gy * h / gh))
                y1 = int(round((gy + 1) * h / gh))
                x0 = int(round(gx * w / gw))
                x1 = int(round((gx + 1) * w / gw))
                block = score_map[y0:y1, x0:x1]
                score = float(np.mean(block)) if block.size > 0 else 0.0
                regions.append(
                    {
                        "bbox": [x0, y0, x1, y1],
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "coordinate_space": "processed",
                        "score": round(score, 6),
                    }
                )
        regions.sort(key=lambda x: x["score"], reverse=True)
        return regions[:topk]

    def _save_heatmap(self, map_data: np.ndarray, image_path: str, output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        heat_u8 = _normalize_to_uint8(map_data)
        heat_img = Image.fromarray(heat_u8, mode="L").convert("RGB")
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(output_dir, f"{base_name}_patchdense_heatmap.png")
        heat_img.save(out_path)
        return out_path

    def _build_response(
        self,
        success: bool,
        start_time: float,
        description: str,
        result: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None,
    ) -> Dict[str, Any]:
        execution_time_ms = int((time.time() - start_time) * 1000)
        return {
            "success": success,
            "status": "success" if success else "error",
            "description": description,
            "result": result or {},
            "error_message": error_message,
            "metadata": {
                "tool_name": self.tool_name,
                "execution_time_ms": execution_time_ms,
            },
        }

    def _select_scan_profile(self, gray: np.ndarray, requested_iterations: int) -> Tuple[int, int]:
        max_dim = max(gray.shape[0], gray.shape[1])
        if max_dim >= 512:
            return 4, min(requested_iterations, 2)
        if max_dim >= 384:
            return 3, min(requested_iterations, 3)
        if max_dim >= 320:
            return 3, requested_iterations
        if max_dim >= 224:
            return 2, requested_iterations
        return 1, requested_iterations

    def __call__(self, args_str: str) -> Dict[str, Any]:
        start_time = time.time()
        log_call_args: Dict[str, Any] = {"raw_args": args_str}

        def _finalize(response_obj: Dict[str, Any]) -> Dict[str, Any]:
            log_tool_use_record(
                tool_key="patchdense_scanner",
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
                    description="PatchDense scanner failed.",
                    error_message="工具参数解析失败，必须是有效的 JSON 格式字符串。",
                )
            )

        image_path = args.get("image_path") or args.get("image_input")
        patch_size = _clamp_int(args.get("patch_size", self.default_patch_size), 5, 13, self.default_patch_size)
        if patch_size % 2 == 0:
            patch_size += 1

        iterations = _clamp_int(args.get("iterations", self.default_iterations), 2, 8, self.default_iterations)
        consistency_threshold = _clamp_float(
            args.get("consistency_threshold", self.default_consistency_threshold),
            0.30,
            0.95,
            self.default_consistency_threshold,
        )
        max_side = _clamp_int(args.get("max_side", self.default_max_side), 256, 768, self.default_max_side)
        try:
            export_map = parse_bool(args.get("export_map", self.default_export_map), self.default_export_map)
        except ValueError as exc:
            return _finalize(self._build_response(False, start_time, "PatchDense scanner failed.", error_message=str(exc)))
        output_dir = str(args.get("output_dir", self.default_output_dir))

        log_call_args = {
            "image_path": image_path,
            "patch_size": patch_size,
            "iterations": iterations,
            "consistency_threshold": consistency_threshold,
            "max_side": max_side,
            "export_map": export_map,
            "output_dir": output_dir,
        }

        if not image_path:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="PatchDense scanner failed.",
                    error_message="缺少必填参数 'image_path'。",
                )
            )

        if not os.path.exists(image_path):
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="PatchDense scanner failed.",
                    error_message=f"找不到指定的图像文件: {image_path}。请检查文件路径是否正确。",
                )
            )

        try:
            image = Image.open(image_path).convert("RGB")
            gray, orig_size, proc_size = self._prepare_gray(image, max_side=max_side)
        except Exception as exc:
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="PatchDense scanner failed.",
                    error_message=f"读取图像失败: {exc}",
                )
            )

        try:
            scan_stride, effective_iterations = self._select_scan_profile(gray, iterations)
            dy_map, dx_map, dist_map = self._run_patchmatch(
                gray=gray,
                patch_size=patch_size,
                iterations=effective_iterations,
                scan_stride=scan_stride,
            )
            suspicion_map, stats = self._thresholded_offset_consistency(
                dy_map=dy_map,
                dx_map=dx_map,
                dist_map=dist_map,
                patch_size=patch_size,
                consistency_threshold=consistency_threshold,
            )

            tamper_probability = float(stats["tamper_probability"])
            evidence_score = max(0.0, min(1.0, tamper_probability))
            positive = bool(
                evidence_score >= 0.68
                and float(stats["hot_ratio"]) >= 0.01
                and float(stats["cluster_ratio"]) >= 0.003
            )
            processed_regions = self._top_regions(suspicion_map, topk=3) if positive else []
            top_regions = [
                map_bbox_to_original(region, processed_size=proc_size, original_size=orig_size)
                for region in processed_regions
            ]

            heatmap_path = ""
            if export_map:
                heatmap_path = self._save_heatmap(suspicion_map, image_path=image_path, output_dir=output_dir)

            result = {
                "forensics_report": {
                    "output_mode": "evidence_only",
                    "method": "PatchMatch dense offset + Thresholded Offset Consistency",
                    "patch_size": patch_size,
                    "iterations": effective_iterations,
                    "requested_iterations": iterations,
                    "consistency_threshold": round(consistency_threshold, 4),
                    "top_suspicious_regions": top_regions,
                    "summary": "已输出偏移一致性统计与局部可疑区域证据；不包含最终判定结论。",
                },
                "evidence_contract": detector_contract(
                    score=evidence_score,
                    threshold=0.68,
                    positive=positive,
                    coordinate_space="original",
                    score_semantics="uncalibrated_reciprocal_copy_move_heuristic",
                ),
                "regions": top_regions,
                "evidence_scores": {
                    "copy_move_consistency_score": round(float(evidence_score), 6),
                },
                "offset_statistics": {
                    "hot_ratio": round(float(stats["hot_ratio"]), 6),
                    "very_hot_ratio": round(float(stats["very_hot_ratio"]), 6),
                    "cluster_ratio": round(float(stats["cluster_ratio"]), 6),
                    "adaptive_threshold": round(float(stats["adaptive_threshold"]), 6),
                    "mean_consistency": round(float(stats["mean_consistency"]), 6),
                    "mean_similarity": round(float(stats["mean_similarity"]), 6),
                    "mean_reciprocal_consistency": round(float(stats["mean_reciprocal_consistency"]), 6),
                    "reciprocal_ratio": round(float(stats["reciprocal_ratio"]), 6),
                    "scan_stride": int(scan_stride),
                    "requested_iterations": int(iterations),
                    "effective_iterations": int(effective_iterations),
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
                    "copymove_heatmap_path": heatmap_path,
                },
            }

            return _finalize(
                self._build_response(
                    success=True,
                    start_time=start_time,
                    description="PatchDense copy-move scan completed.",
                    result=result,
                )
            )
        except Exception as exc:
            logger.exception("PatchDenseScanner execution failed.")
            return _finalize(
                self._build_response(
                    success=False,
                    start_time=start_time,
                    description="PatchDense scanner failed.",
                    error_message=f"PatchMatch 计算阶段发生错误: {exc}",
                )
            )


class PatchDenseScannerTool(BaseTool):
    """verl-native wrapper aligned with BaseTool lifecycle."""

    def __init__(self, config: dict, tool_schema: Optional[OpenAIFunctionToolSchema]):
        super().__init__(config, tool_schema or _default_schema())
        self._instance_dict: dict[str, dict[str, Any]] = {}
        self._default_patch_size = int(config.get("default_patch_size", 7))
        self._default_iterations = int(config.get("default_iterations", 4))
        self._default_consistency_threshold = float(config.get("default_consistency_threshold", 0.60))
        self._default_max_side = int(config.get("default_max_side", 384))
        self._default_export_map = bool(config.get("default_export_map", False))
        self._default_output_dir = str(config.get("output_dir", "./outputs/patchdense_scanner"))
        self._tool_impl = PatchDenseScanner(
            default_patch_size=self._default_patch_size,
            default_iterations=self._default_iterations,
            default_consistency_threshold=self._default_consistency_threshold,
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
                "patch_size": clean_parameters.get("patch_size", self._default_patch_size),
                "iterations": clean_parameters.get("iterations", self._default_iterations),
                "consistency_threshold": clean_parameters.get(
                    "consistency_threshold",
                    self._default_consistency_threshold,
                ),
                "max_side": clean_parameters.get("max_side", self._default_max_side),
                "export_map": self._default_export_map,
                "output_dir": build_call_output_dir(self._default_output_dir, instance_id, kwargs),
            }
            result = self._tool_impl(json.dumps(call_args, ensure_ascii=False))
            if instance_id in self._instance_dict:
                self._instance_dict[instance_id]["response"] = result
            return ToolResponse(text=json.dumps(result, ensure_ascii=False)), 0.0, {}
        except Exception as exc:
            logger.exception("PatchDenseScannerTool execution failed.")
            error_result = {
                "success": False,
                "status": "error",
                "description": "PatchDense scanner failed.",
                "result": {},
                "error_message": f"Tool runtime error: {exc}",
            }
            return ToolResponse(text=json.dumps(error_result, ensure_ascii=False)), 0.0, {}

    async def calc_reward(self, instance_id: str, **kwargs) -> float:
        return 0.0

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]

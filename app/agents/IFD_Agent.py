import base64
import json
import mimetypes
import re
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from PIL import ExifTags, Image, ImageDraw

from app.agents.models import ModelFactory
from app.core.config import Settings, settings
from app.core.language import is_zh, normalize_language
from app.agents.prompts.ifd import IFD_PROMPT_EN, IFD_PROMPT_ZH
from app.agents.tools.ifd.CFA_Validator import CFAValidator
from app.agents.tools.ifd.Image_Resizer import ImageResizer
from app.agents.tools.ifd.Noise_Analyzer import NoiseAnalyzer
from app.agents.tools.ifd.PatchDense_Scanner import PatchDenseScanner
from app.agents.tools.ifd.ROI_Extractor import ROIExtractor
from app.agents.tools.ifd.Spectrum_Grid_Analyzer import SpectrumGridAnalyzer
from app.agents.tools.ifd.SRM_Filter import SRMFilter
from app.agents.tools.ifd.Visual_Enhancer import VisualEnhancer


class IFDModelUnavailableError(RuntimeError):
    """The vision model did not return a usable action after all attempts."""


class IFDDetectionAgent:
    """Run multi-round image forensics with local tools and a final LLM verdict."""

    LABELS_ZH = {"真实图像", "伪造图像", "合成图像"}
    LABELS_EN = {"Real_Image", "Forged_Image", "Synthetic_Image"}
    VERDICT_TO_LABEL_ZH = {
        "authentic": "真实图像",
        "tampered": "伪造图像",
        "ai_manipulated": "合成图像",
        "ai_generated": "合成图像",
        "manipulated": "伪造图像",
        "real": "真实图像",
    }
    VERDICT_TO_LABEL_EN = {
        "authentic": "Real_Image",
        "tampered": "Forged_Image",
        "ai_manipulated": "Synthetic_Image",
        "ai_generated": "Synthetic_Image",
        "manipulated": "Forged_Image",
        "real": "Real_Image",
    }

    def __init__(
        self,
        config: Settings = settings,
        model_factory: ModelFactory | None = None,
        tools: dict[str, Any] | None = None,
        evaluator=None,
    ):
        self.config = config
        self.model_factory = model_factory or ModelFactory(config)
        self.tools = tools or self._default_tools()
        self.evaluator = evaluator
        self.output_root = Path(config.vision_output_dir)
        self.max_rounds = max(1, min(3, int(config.vision_max_rounds)))
        self.tool_names = set(self.tools)

    def detect(
        self,
        image_path: str,
        question: str | None = None,
        language: str = "zh",
        visual_query: str | None = None,
        planning_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        language = normalize_language(language)
        session_dir = self.output_root / uuid.uuid4().hex
        session_dir.mkdir(parents=True, exist_ok=True)
        combined_question = "\n".join(item for item in [question or "", visual_query or ""] if item).strip()
        image_context = self._build_image_context(image_path, combined_question)
        image_context["p_agent"] = planning_context or {}

        tool_calls: list[dict[str, Any]] = []
        first_tools = self._initial_tool_batch(image_context)
        first_reasons = {tool_name: self._first_tool_reason(tool_name, language) for tool_name in first_tools}
        self._run_tool_batch(tool_calls, first_tools, image_path, session_dir, first_reasons)

        while len(tool_calls) < self.max_rounds:
            action = self._ask_model_for_action(
                tool_calls, combined_question, language, image_path, image_context, force_final=False
            )
            if action.get("action") == "final" or action.get("label") or action.get("verdict"):
                final_result = self._normalize_final_result(action.get("final_result") or action, tool_calls, image_path, session_dir, language, image_context)
                return final_result

            next_tools = self._select_next_tools(action, tool_calls, image_context)
            if not next_tools:
                break
            why_map = self._why_map(action, next_tools)
            parameter_map = self._parameter_map(action, next_tools)
            self._run_tool_batch(tool_calls, next_tools, image_path, session_dir, why_map, parameter_map)

        final_action = self._ask_model_for_action(
            tool_calls, combined_question, language, image_path, image_context, force_final=True
        )
        final_payload = final_action.get("final_result") if isinstance(final_action.get("final_result"), dict) else final_action
        return self._normalize_final_result(final_payload, tool_calls, image_path, session_dir, language, image_context)

    def _run_tool_round(
        self,
        tool_calls: list[dict[str, Any]],
        tool_name: str,
        image_path: str,
        session_dir: Path,
        why: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        output = self._run_tool(tool_name, image_path, session_dir, parameters)
        tool_calls.append(
            {
                "step": len(tool_calls) + 1,
                "tool": tool_name,
                "why": why,
                "output": output,
                "summary": self._summarize_tool_output(output),
            }
        )

    def _run_tool_batch(
        self,
        tool_calls: list[dict[str, Any]],
        tool_names: list[str],
        image_path: str,
        session_dir: Path,
        why_map: dict[str, str] | None = None,
        parameter_map: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        if not tool_names:
            return
        remaining = max(0, self.max_rounds - len(tool_calls))
        selected = tool_names[:remaining]
        if not selected:
            return
        why_map = why_map or {}
        parameter_map = parameter_map or {}

        def run_one(tool_name: str) -> tuple[str, dict[str, Any]]:
            return tool_name, self._run_tool(tool_name, image_path, session_dir, parameter_map.get(tool_name))

        if len(selected) == 1:
            results = [run_one(selected[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(selected)) as executor:
                results = list(executor.map(run_one, selected))

        for tool_name, output in results:
            tool_calls.append(
                {
                    "step": len(tool_calls) + 1,
                    "tool": tool_name,
                    "why": why_map.get(tool_name) or f"根据已有证据继续调用 {tool_name} 进行核验。",
                    "output": output,
                    "summary": self._summarize_tool_output(output),
                }
            )

    def _ask_model_for_action(
        self,
        tool_calls: list[dict[str, Any]],
        question: str | None,
        language: str,
        image_path: str,
        image_context: dict[str, Any],
        force_final: bool,
    ) -> dict[str, Any]:
        prompt = self._build_interaction_prompt(tool_calls, question, language, force_final, image_context)
        last_error: Exception | None = None
        max_retries = max(0, min(2, int(self.config.ifd_model_failure_max_retries)))
        for _ in range(max_retries + 1):
            try:
                raw = self.evaluator(prompt) if self.evaluator else self._call_model(prompt, language, image_path)
                parsed = self._parse_json(raw)
                if not parsed:
                    raise ValueError("IFD model output must be a non-empty JSON object")
                final = parsed.get("final_result") if isinstance(parsed.get("final_result"), dict) else parsed
                if (force_final or parsed.get("action") == "final" or parsed.get("label") or parsed.get("verdict")) and not (final.get("label") or final.get("verdict")):
                    raise ValueError("IFD final output must contain a label or verdict")
                if not force_final and not (
                    final.get("label") or final.get("verdict")
                    or parsed.get("action") in {"call_tool", "call_tools"}
                ):
                    raise ValueError("IFD model output must contain a final verdict or tool action")
                return parsed
            except Exception as exc:
                last_error = exc
        raise IFDModelUnavailableError("IFD model failed after three attempts") from last_error

    def _call_model(self, prompt: str, language: str, image_path: str) -> str:
        client = self.model_factory.qwen_vision_client(
            timeout=self.config.vision_verdict_timeout_seconds, max_retries=0
        )
        system_prompt = "你是惊堂木图像取证智能体，只输出 JSON。" if is_zh(language) else "You are Jingtangmu's image forensic agent. Output JSON only."
        user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        data_url = self._image_to_data_url(image_path)
        if data_url:
            user_content.insert(0, {"type": "image_url", "image_url": {"url": data_url}})
        completion = client.chat.completions.create(
            model=self.config.qwen_vision_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=0,
            max_tokens=1800,
        )
        return completion.choices[0].message.content or ""

    def _build_interaction_prompt(
        self,
        tool_calls: list[dict[str, Any]],
        question: str | None,
        language: str,
        force_final: bool,
        image_context: dict[str, Any],
    ) -> str:
        used_tools = [item["tool"] for item in tool_calls]
        base_prompt = IFD_PROMPT_ZH if is_zh(language) else IFD_PROMPT_EN
        remaining_tools = self._eligible_remaining_tools(tool_calls, image_context)
        evidence = [
            {
                "step": item["step"],
                "tool": item["tool"],
                "why": item["why"],
                "output_summary": item["summary"],
            }
            for item in tool_calls
        ]
        if is_zh(language):
            action_instruction = (
                "现在必须按照上方 JSON 格式输出最终 JSON，不要继续调用工具。"
                if force_final
                else (
                    "如果证据已经足够，按照上方 JSON 格式输出最终 JSON；否则输出工具调用 JSON。"
                    "单工具格式：{\"action\":\"call_tool\",\"tool\":\"Noise_Analyzer\",\"parameters\":{},\"why\":\"...\"}。"
                    "多工具并行格式：{\"action\":\"call_tools\",\"tools\":[{\"tool\":\"Noise_Analyzer\",\"parameters\":{},\"why\":\"...\"},{\"tool\":\"Spectrum_Grid_Analyzer\",\"parameters\":{},\"why\":\"...\"}]}。"
                )
            )
            label_instruction = "当前接口标签约束：最终 label 只能输出中文三分类：真实图像、伪造图像、合成图像。不要输出 uncertain、旧标签或英文标签。"
            tool_context = (
                f"language={language}\n用户问题={question or ''}\n已调用工具={used_tools}\n"
                f"当前可直接调用工具={remaining_tools}\n每轮可并行调用多个互不依赖工具，但总调用数不能超过上限。\n工具门控上下文={json.dumps(self._prompt_image_context(image_context), ensure_ascii=False)}\n"
                "视觉判断优先：即使已有工具结果，也必须重新直接观察图像，并把工具无法覆盖的语义、几何、物理和常识异常写入 visual_analysis。\n"
                "文字真实性规则：绶带、衣服、标牌、海报等载体上的文字属于图像语义事实；必须先转写关键文字并检查字体、透视、弯曲、遮挡和材质褶皱贴合，不能说“内容意图问题不影响真实性”。\n"
                "条件工具规则：PatchDense_Scanner 仅在疑似复制移动/克隆区域时调用；"
                "CFA_Validator 仅在图像来源可确认为真实相机链路时调用。\n"
            )
            round_context = f"工具调用上限={self.max_rounds}，当前已调用={len(tool_calls)}。\n"
        else:
            action_instruction = (
                "You must output the final JSON using the schema above now. Do not call more tools."
                if force_final
                else (
                    "If evidence is sufficient, output the final JSON using the schema above; otherwise output tool-call JSON. "
                    "Single-tool format: {\"action\":\"call_tool\",\"tool\":\"Noise_Analyzer\",\"parameters\":{},\"why\":\"...\"}. "
                    "Parallel multi-tool format: {\"action\":\"call_tools\",\"tools\":[{\"tool\":\"Noise_Analyzer\",\"parameters\":{},\"why\":\"...\"},{\"tool\":\"Spectrum_Grid_Analyzer\",\"parameters\":{},\"why\":\"...\"}]}."
                )
            )
            label_instruction = "Interface label constraint: final label must be one of Real_Image, Forged_Image, Synthetic_Image. Do not output Chinese labels, old labels, or uncertain."
            tool_context = (
                f"language={language}\nuser_question={question or ''}\ncalled_tools={used_tools}\n"
                f"currently_callable_tools={remaining_tools}\nMultiple independent tools may be called in parallel each round, but total calls must stay under the limit.\ntool_gate_context={json.dumps(self._prompt_image_context(image_context), ensure_ascii=False)}\n"
                "Visual judgment first: even when tool evidence exists, inspect the image directly again and include semantic, geometric, physical, and commonsense anomalies not covered by tools in visual_analysis.\n"
                "Text authenticity rule: text on sashes, clothing, signs, posters, and similar carriers is part of the image's semantic facts; transcribe key text and inspect font, perspective, curvature, occlusion, and material-fold alignment. Do not say content intent is unrelated to authenticity.\n"
                "Conditional tool rules: call PatchDense_Scanner only when copy-move/cloned regions are suspected; "
                "call CFA_Validator only when the image source is confirmed as a real camera pipeline.\n"
            )
            round_context = f"tool_call_limit={self.max_rounds}; already_called={len(tool_calls)}.\n"
        return (
            f"{base_prompt}\n\n"
            f"{label_instruction}\n"
            f"{tool_context}"
            f"{round_context}"
            f"{action_instruction}\n"
            f"EVIDENCE_JSON:\n{json.dumps(evidence, ensure_ascii=False)}"
        )

    def _select_next_tool(self, action: dict[str, Any], tool_calls: list[dict[str, Any]], image_context: dict[str, Any] | None = None) -> str | None:
        tools = self._select_next_tools(action, tool_calls, image_context)
        return tools[0] if tools else None

    def _select_next_tools(self, action: dict[str, Any], tool_calls: list[dict[str, Any]], image_context: dict[str, Any] | None = None) -> list[str]:
        image_context = image_context or {}
        used = {item["tool"] for item in tool_calls}
        remaining_slots = max(0, getattr(self, "max_rounds", 3) - len(tool_calls))
        selected = []
        for entry in self._action_tool_entries(action):
            raw_tool = str(entry.get("tool") or entry.get("tool_name") or "").strip()
            if (
                raw_tool in self.tool_names
                and raw_tool not in used
                and raw_tool not in selected
                and self._is_tool_allowed(raw_tool, tool_calls, image_context, entry)
            ):
                selected.append(raw_tool)
            if len(selected) >= remaining_slots:
                break
        if selected:
            return selected
        fallback = self._fallback_next_tools(tool_calls, image_context, remaining_slots)
        return fallback

    def _fallback_next_tool(self, tool_calls: list[dict[str, Any]], image_context: dict[str, Any] | None = None) -> str | None:
        tools = self._fallback_next_tools(tool_calls, image_context, 1)
        return tools[0] if tools else None

    def _fallback_next_tools(
        self,
        tool_calls: list[dict[str, Any]],
        image_context: dict[str, Any] | None = None,
        limit: int = 1,
    ) -> list[str]:
        image_context = image_context or {}
        used = {item["tool"] for item in tool_calls}
        selected = []
        limit = max(0, limit)
        if limit == 0:
            return selected
        for tool_name in [
            "SRM_Filter",
            "Noise_Analyzer",
            "Spectrum_Grid_Analyzer",
            "PatchDense_Scanner",
            "CFA_Validator",
        ]:
            if len(selected) >= limit:
                break
            if tool_name in self.tool_names and tool_name not in used and self._is_tool_allowed(tool_name, tool_calls, image_context):
                selected.append(tool_name)
        return selected

    def _action_tool_entries(self, action: dict[str, Any]) -> list[dict[str, Any]]:
        entries = []
        for key in ["tools", "tool_calls"]:
            value = action.get(key)
            if isinstance(value, list):
                entries.extend(item for item in value if isinstance(item, dict))
        if action.get("tool") or action.get("tool_name"):
            entries.insert(0, action)
        return entries

    def _why_map(self, action: dict[str, Any], tool_names: list[str]) -> dict[str, str]:
        output = {}
        for entry in self._action_tool_entries(action):
            tool_name = str(entry.get("tool") or entry.get("tool_name") or "").strip()
            if tool_name in tool_names:
                output[tool_name] = str(entry.get("why") or entry.get("reason") or "").strip()
        fallback = str(action.get("why") or action.get("reason") or "").strip()
        return {tool_name: output.get(tool_name) or fallback for tool_name in tool_names}

    def _parameter_map(self, action: dict[str, Any], tool_names: list[str]) -> dict[str, dict[str, Any]]:
        output = {}
        for entry in self._action_tool_entries(action):
            tool_name = str(entry.get("tool") or entry.get("tool_name") or "").strip()
            parameters = entry.get("parameters")
            if tool_name in tool_names and isinstance(parameters, dict):
                output[tool_name] = parameters
        if len(tool_names) == 1 and isinstance(action.get("parameters"), dict):
            output.setdefault(tool_names[0], action.get("parameters"))
        return output

    def _normalize_final_result(
        self,
        payload: dict[str, Any] | None,
        tool_calls: list[dict[str, Any]],
        image_path: str,
        session_dir: Path,
        language: str,
        image_context: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            payload = self._fallback_final_payload(tool_calls, language, image_context)

        label = self._normalize_label(payload.get("label") or payload.get("verdict"), payload.get("probability"), language)
        raw_probability = self._bounded_float(payload.get("probability"), self._fallback_probability(tool_calls, image_context))
        raw_confidence = self._bounded_float(payload.get("confidence"), 0.55 if self._is_real_label(label) else 0.65)
        tool_evidence = self._payload_tool_evidence(payload, tool_calls)
        payload_user_answer = payload.get("user_answer") if isinstance(payload.get("user_answer"), dict) else {}
        suspicious_regions = self._normalize_regions(payload.get("suspicious_regions"))[:1]
        if self._is_local_tampering_label(label) and not suspicious_regions:
            suspicious_regions = self._candidate_regions_from_tools(tool_calls)[:1]
        if self._is_local_tampering_label(label) and not suspicious_regions:
            suspicious_regions = self._fallback_visual_region(image_context, language)

        forgery_types = self._normalize_forgery_types(payload.get("forgery_types"), label, tool_calls, language)
        if self._is_real_label(label):
            suspicious_regions = []

        probability = self._calibrate_probability(label, raw_probability, tool_calls, suspicious_regions, image_context)
        confidence = self._calibrate_confidence(label, raw_confidence, tool_calls, suspicious_regions)
        tamper_score = self._tamper_score(tool_calls, image_context)
        ai_generated_score = self._ai_score(tool_calls)
        visual_analysis = str(payload.get("visual_analysis") or payload.get("visual_observation") or "").strip()
        missing_visual_analysis = not visual_analysis
        if not visual_analysis:
            visual_analysis = self._missing_visual_analysis(language)

        artifacts = self._collect_artifacts(tool_calls)
        annotated_path = self._annotate_image(image_path, suspicious_regions, session_dir) if suspicious_regions else ""
        if annotated_path:
            artifacts["annotated_image_path"] = annotated_path

        final_summary = str(payload.get("final_summary") or "").strip() or self._fallback_summary(label, probability, suspicious_regions, language)
        limitations = payload.get("limitations") if isinstance(payload.get("limitations"), list) else []
        limitations = self._merge_consistency_conflicts(limitations, payload.get("consistency_check"), language)
        if missing_visual_analysis:
            limitations.append(visual_analysis)
        user_answer = self._normalize_user_answer(
            payload_user_answer,
            final_summary,
            tool_evidence,
            limitations,
            language,
            visual_analysis,
        )
        return {
            "label": label,
            "probability": probability,
            "confidence": confidence,
            "tamper_score": tamper_score,
            "ai_generated_score": ai_generated_score,
            "visual_analysis": visual_analysis,
            "forgery_types": forgery_types,
            "tool_evidence": tool_evidence,
            "suspicious_regions": suspicious_regions,
            "limitations": limitations,
            "final_summary": final_summary,
            "artifacts": artifacts,
            "tool_calls": tool_calls,
            "user_answer": user_answer,
        }

    def _missing_visual_analysis(self, language: str) -> str:
        return (
            "模型未返回独立的画面视觉分析；请结合 limitations 谨慎解读本次图像判断。"
            if is_zh(language)
            else "The model did not return an independent visual analysis; interpret this image verdict with the limitations in mind."
        )

    def _fallback_final_payload(self, tool_calls: list[dict[str, Any]], language: str = "zh", image_context: dict[str, Any] | None = None) -> dict[str, Any]:
        ai_score = self._ai_score(tool_calls)
        tamper_score = self._tamper_score(tool_calls, image_context)
        regions = self._candidate_regions_from_tools(tool_calls)
        if tamper_score >= 0.55:
            label = "伪造图像" if is_zh(language) else "Forged_Image"
            probability = tamper_score
        elif ai_score >= 0.58:
            label = "合成图像" if is_zh(language) else "Synthetic_Image"
            probability = ai_score
        else:
            label = "真实图像" if is_zh(language) else "Real_Image"
            probability = max(ai_score, tamper_score)
        return {
            "action": "final",
            "label": label,
            "probability": probability,
            "confidence": 0.62 if self._is_real_label(label) else 0.68,
            "forgery_types": self._infer_forgery_types(label, tool_calls),
            "tool_evidence": self._build_evidence_chain(tool_calls),
            "suspicious_regions": (regions[:1] or self._fallback_visual_region(image_context, language)) if self._is_local_tampering_label(label) else regions[:1],
            "limitations": (
                ["当前结论基于本地取证工具输出与模型综合判断，工具证据不等同于人工鉴定。"]
                if is_zh(language)
                else ["The verdict is based on local forensic tool outputs and model synthesis; tool evidence is not a human forensic certification."]
            ),
            "final_summary": self._fallback_summary(label, probability, regions[:1] if self._is_local_tampering_label(label) else [], language),
        }

    def _choose_first_tool(self, image_context: dict[str, Any]) -> str:
        if image_context.get("copy_move_hint") and "PatchDense_Scanner" in self.tool_names:
            return "PatchDense_Scanner"
        if (image_context.get("face_hint") or image_context.get("ai_hint")) and "Spectrum_Grid_Analyzer" in self.tool_names:
            return "Spectrum_Grid_Analyzer"
        if "SRM_Filter" in self.tool_names:
            return "SRM_Filter"
        if "Spectrum_Grid_Analyzer" in self.tool_names:
            return "Spectrum_Grid_Analyzer"
        return next(iter(self.tool_names), "Spectrum_Grid_Analyzer")

    def _initial_tool_batch(self, image_context: dict[str, Any]) -> list[str]:
        first_tool = self._choose_first_tool(image_context)
        batch = [first_tool] if first_tool else []
        if self.max_rounds <= 1:
            return batch[:1]
        local_candidates = [
            "SRM_Filter",
            "Spectrum_Grid_Analyzer",
            "Noise_Analyzer",
        ]
        for tool_name in local_candidates:
            if (
                tool_name in self.tool_names
                and tool_name not in batch
                and self._is_tool_allowed(tool_name, [], image_context)
            ):
                batch.append(tool_name)
            if len(batch) >= min(2, self.max_rounds):
                break
        return batch[: self.max_rounds]

    def _first_tool_reason(self, tool_name: str, language: str) -> str:
        zh_reasons = {
            "PatchDense_Scanner": "阶段A定向粗检：用户问题提示复制移动/克隆风险，优先检查密集偏移一致性。",
            "Spectrum_Grid_Analyzer": "阶段A快速粗检：先检查 AI 频域网格、能量异常或人像生成线索。",
            "SRM_Filter": "阶段A通用粗检：先检查高阶残差噪声中的局部统计异常。",
        }
        en_reasons = {
            "PatchDense_Scanner": "Stage A targeted coarse check: the user question suggests copy-move/cloning risk, so inspect dense offset consistency first.",
            "Spectrum_Grid_Analyzer": "Stage A coarse check: inspect AI-related frequency grids, energy anomalies, or portrait generation clues first.",
            "SRM_Filter": "Stage A general coarse check: inspect local statistical anomalies in high-order residual noise first.",
        }
        reasons = zh_reasons if is_zh(language) else en_reasons
        return reasons.get(tool_name, f"Stage A coarse check: run {tool_name} first.")

    def _build_image_context(self, image_path: str, question: str | None = None) -> dict[str, Any]:
        path = Path(image_path)
        question_text = question or ""
        context: dict[str, Any] = {
            "path_suffix": path.suffix.lower(),
            "mime_type": mimetypes.guess_type(path.name)[0] or "image/png",
            "format": "",
            "width": 0,
            "height": 0,
            "has_exif": False,
            "exif_fields": [],
            "camera_make": "",
            "camera_model": "",
            "software": "",
            "likely_camera_source": False,
            "face_hint": self._text_has_face_signal(question_text),
            "copy_move_hint": self._text_has_copy_move_signal(question_text),
            "ai_hint": self._text_has_ai_signal(question_text),
        }
        try:
            with Image.open(path) as image:
                context["format"] = image.format or ""
                context["width"], context["height"] = image.size
                exif = image.getexif()
                if exif:
                    decoded = {
                        ExifTags.TAGS.get(tag, str(tag)): str(value)
                        for tag, value in exif.items()
                        if value is not None
                    }
                    context["has_exif"] = bool(decoded)
                    context["exif_fields"] = sorted(decoded.keys())[:20]
                    context["camera_make"] = decoded.get("Make", "")
                    context["camera_model"] = decoded.get("Model", "")
                    context["software"] = decoded.get("Software", "")
                    context["likely_camera_source"] = self._looks_like_camera_source(decoded, image.format or "")
        except Exception:
            return context
        return context

    def _looks_like_camera_source(self, exif: dict[str, str], image_format: str) -> bool:
        camera_fields = {"Make", "Model", "DateTimeOriginal", "DateTimeDigitized", "FocalLength", "ExposureTime", "ISOSpeedRatings"}
        has_camera_identity = bool(exif.get("Make") or exif.get("Model"))
        has_camera_capture = any(field in exif for field in camera_fields)
        is_camera_container = image_format.upper() in {"JPEG", "TIFF", "HEIF", "HEIC"}
        return is_camera_container and has_camera_identity and has_camera_capture

    def _prompt_image_context(self, image_context: dict[str, Any]) -> dict[str, Any]:
        return {
            "format": image_context.get("format"),
            "mime_type": image_context.get("mime_type"),
            "size": {"width": image_context.get("width"), "height": image_context.get("height")},
            "has_exif": image_context.get("has_exif"),
            "exif_fields": image_context.get("exif_fields", []),
            "camera_make": image_context.get("camera_make"),
            "camera_model": image_context.get("camera_model"),
            "likely_camera_source": image_context.get("likely_camera_source"),
            "face_hint_from_question": image_context.get("face_hint"),
            "copy_move_hint_from_question": image_context.get("copy_move_hint"),
            "ai_hint_from_question": image_context.get("ai_hint"),
            "p_agent": image_context.get("p_agent", {}),
        }

    def _eligible_remaining_tools(self, tool_calls: list[dict[str, Any]], image_context: dict[str, Any]) -> list[str]:
        used = {item["tool"] for item in tool_calls}
        return [
            tool_name
            for tool_name in sorted(self.tool_names)
            if tool_name not in used and self._is_tool_allowed(tool_name, tool_calls, image_context)
        ]

    def _is_tool_allowed(
        self,
        tool_name: str,
        tool_calls: list[dict[str, Any]],
        image_context: dict[str, Any],
        action: dict[str, Any] | None = None,
    ) -> bool:
        if tool_name in {
            "Spectrum_Grid_Analyzer",
            "SRM_Filter",
            "Noise_Analyzer",
        }:
            return True
        if tool_name == "PatchDense_Scanner":
            return self._has_copy_move_gate_signal(image_context, action)
        if tool_name == "CFA_Validator":
            return bool(image_context.get("likely_camera_source"))
        if tool_name in {"ROI_Extractor", "Image_Resizer", "Visual_Enhancer"}:
            return action is not None
        return False

    def _has_copy_move_gate_signal(self, image_context: dict[str, Any], action: dict[str, Any] | None = None) -> bool:
        if image_context.get("copy_move_hint"):
            return True
        if not action:
            return False
        for key in ["copy_move_suspected", "duplicate_region_suspected", "cloning_suspected"]:
            if action.get(key) is True:
                return True
        return self._text_has_copy_move_signal(self._action_text(action))

    def _action_text(self, action: dict[str, Any]) -> str:
        parts = [
            str(action.get("why") or ""),
            str(action.get("reason") or ""),
            str(action.get("visual_basis") or ""),
            json.dumps(action.get("parameters") or {}, ensure_ascii=False),
        ]
        return " ".join(parts)

    def _text_has_face_signal(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "face",
            "facial",
            "portrait",
            "person",
            "people",
            "human",
            "deepfake",
            "eyes",
            "mouth",
            "skin",
            "人脸",
            "脸",
            "面部",
            "人物",
            "人像",
            "肖像",
            "换脸",
            "眼睛",
            "嘴",
            "皮肤",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _text_has_ai_signal(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "ai",
            "generated",
            "generation",
            "synthetic",
            "deepfake",
            "gan",
            "diffusion",
            "人工智能",
            "生成",
            "合成",
            "深度伪造",
            "ai生成",
            "ai操纵",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _text_has_copy_move_signal(self, text: str) -> bool:
        lowered = text.lower()
        keywords = [
            "copy-move",
            "copy move",
            "copymove",
            "clone",
            "cloned",
            "duplicate",
            "duplicated",
            "repeated object",
            "复制移动",
            "复制",
            "克隆",
            "重复",
            "拷贝",
            "粘贴",
        ]
        return any(keyword in lowered for keyword in keywords)

    def _summarize_tool_output(self, output: dict[str, Any]) -> dict[str, Any]:
        result = output.get("result", {}) if isinstance(output, dict) else {}
        report = result.get("forensics_report", {}) if isinstance(result, dict) else {}
        summary = {
            "success": output.get("success"),
            "status": output.get("status"),
            "description": output.get("description"),
            "error_message": output.get("error_message"),
            "tool_name": output.get("metadata", {}).get("tool_name"),
            "execution_time_ms": output.get("metadata", {}).get("execution_time_ms"),
            "method": report.get("method"),
            "report_summary": report.get("summary"),
            "top_suspicious_regions": report.get("top_suspicious_regions", []),
            "heuristic_scores": result.get("heuristic_scores") or result.get("evidence_scores", {}),
            "evidence_scores": result.get("heuristic_scores") or result.get("evidence_scores", {}),
            "evidence_contract": result.get("evidence_contract", {}),
            "regions": result.get("regions", []),
            "artifacts": result.get("artifacts", {}),
        }
        for key in [
            "spectrum_statistics",
            "noise_statistics",
            "cfa_statistics",
            "offset_statistics",
        ]:
            if key in result:
                summary[key] = result[key]
        return summary

    def _build_evidence_chain(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        chain = []
        for item in tool_calls[:3]:
            summary = item.get("summary", {})
            findings = []
            contract = summary.get("evidence_contract") or {}
            if contract:
                findings.append(
                    "证据契约："
                    f"decision={contract.get('decision', 'inconclusive')}, "
                    f"score={contract.get('score')}, calibrated={contract.get('calibrated', False)}"
                )
            if summary.get("heuristic_scores") or summary.get("evidence_scores"):
                scores = summary.get("heuristic_scores") or summary.get("evidence_scores")
                findings.append(f"启发式线索分数：{json.dumps(scores, ensure_ascii=False)}")
            if summary.get("top_suspicious_regions"):
                findings.append(f"可疑区域：{json.dumps(summary['top_suspicious_regions'][:2], ensure_ascii=False)}")
            if not findings:
                findings.append(str(summary.get("description") or summary.get("report_summary") or "未得到显著异常。"))
            chain.append(
                {
                    "step": item["step"],
                    "tool": item["tool"],
                    "why": item["why"],
                    "key_findings": findings,
                    "artifacts": [value for value in self._extract_artifact_paths(summary).values() if value],
                }
            )
        return chain

    def _candidate_regions_from_tools(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        regions = []
        for item in tool_calls:
            summary = item.get("summary", {})
            candidates = summary.get("regions") or summary.get("top_suspicious_regions", []) or []
            for candidate in candidates:
                parsed = self._parse_region(candidate)
                if parsed:
                    regions.append({
                        "region": parsed,
                        "score": self._bounded_float(candidate.get("score"), 0.5),
                        "reason": f"{item['tool']} 输出的局部异常区域。",
                        "coordinate_space": candidate.get("coordinate_space", "original_image"),
                        "region_reliability": candidate.get("region_reliability", "tool_estimated"),
                    })
        regions.sort(key=lambda item: item.get("score", 0.0), reverse=True)
        return regions

    def _normalize_regions(self, value: Any) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            return []
        regions = []
        for item in value:
            if not isinstance(item, dict):
                continue
            region = self._parse_region(item.get("region") or item)
            if not region:
                continue
            regions.append({
                "region": region,
                "score": self._bounded_float(item.get("score"), 0.5),
                "reason": str(item.get("reason") or "模型综合工具证据标注的可疑区域。"),
                "coordinate_space": str(item.get("coordinate_space") or "original_image"),
                "region_reliability": str(item.get("region_reliability") or "model_estimated"),
            })
        return regions

    def _fallback_visual_region(self, image_context: dict[str, Any] | None, language: str) -> list[dict[str, Any]]:
        image_context = image_context or {}
        width = int(image_context.get("width") or 0)
        height = int(image_context.get("height") or 0)
        if width <= 1 or height <= 1:
            return []
        x0 = max(0, int(round(width * 0.25)))
        y0 = max(0, int(round(height * 0.25)))
        x1 = min(width - 1, int(round(width * 0.75)))
        y1 = min(height - 1, int(round(height * 0.75)))
        reason = (
            "工具未提供可靠坐标，按模型视觉判断要求生成的近似伪造区域。"
            if is_zh(language)
            else "The tools did not provide reliable coordinates; this is an approximate region required from visual judgment."
        )
        return [
            {
                "region": f"{x0},{y0},{x1},{y1}",
                "score": 0.5,
                "reason": reason,
                "coordinate_space": "original_image",
                "region_reliability": "model_estimated_fallback",
            }
        ]

    def _parse_region(self, value: Any) -> str:
        if isinstance(value, dict):
            nested = value.get("region")
            if nested is not None and nested is not value:
                parsed = self._parse_region(nested)
                if parsed:
                    return parsed

            if all(key in value for key in ("x0", "y0", "x1", "y1")):
                x0 = self._int_coord(value.get("x0"))
                y0 = self._int_coord(value.get("y0"))
                x1 = self._int_coord(value.get("x1"))
                y1 = self._int_coord(value.get("y1"))
                return self._format_region(x0, y0, x1, y1)

            if all(key in value for key in ("x", "y", "w", "h")):
                x0 = self._int_coord(value.get("x"))
                y0 = self._int_coord(value.get("y"))
                w = self._int_coord(value.get("w"))
                h = self._int_coord(value.get("h"))
                return self._format_region(x0, y0, x0 + w, y0 + h)

        text = str(value or "").strip()
        xy_match = re.search(
            r"x\s*[:：]\s*(\d+(?:\.\d+)?)\s*[-~至]\s*(\d+(?:\.\d+)?).*?"
            r"y\s*[:：]\s*(\d+(?:\.\d+)?)\s*[-~至]\s*(\d+(?:\.\d+)?)",
            text,
            re.I | re.S,
        )
        if xy_match:
            x0, x1, y0, y1 = [int(float(x)) for x in xy_match.groups()]
            x0, x1 = sorted((max(0, x0), max(0, x1)))
            y0, y1 = sorted((max(0, y0), max(0, y1)))
            if x1 > x0 and y1 > y0:
                return f"{x0},{y0},{x1},{y1}"
        nums = [int(float(x)) for x in re.findall(r"-?\d+(?:\.\d+)?", text)]
        if len(nums) >= 4:
            if "x:" in text and "y:" in text:
                x0, x1, y0, y1 = nums[:4]
            else:
                x0, y0, x1, y1 = nums[:4]
            x0, x1 = sorted((max(0, x0), max(0, x1)))
            y0, y1 = sorted((max(0, y0), max(0, y1)))
            if x1 > x0 and y1 > y0:
                return f"{x0},{y0},{x1},{y1}"
        return ""

    def _int_coord(self, value: Any) -> int:
        try:
            return int(round(float(value)))
        except (TypeError, ValueError):
            return 0

    def _format_region(self, x0: int, y0: int, x1: int, y1: int) -> str:
        x0, x1 = sorted((max(0, x0), max(0, x1)))
        y0, y1 = sorted((max(0, y0), max(0, y1)))
        return f"{x0},{y0},{x1},{y1}" if x1 > x0 and y1 > y0 else ""

    def _ai_score(self, tool_calls: list[dict[str, Any]]) -> float:
        score = 0.0
        for item in tool_calls:
            summary = item.get("summary", {})
            contract_score = self._positive_evidence_contract_score(summary)
            if contract_score is not None:
                if item["tool"] == "Spectrum_Grid_Analyzer":
                    score = max(score, contract_score)
                continue
            scores = summary.get("heuristic_scores") or summary.get("evidence_scores", {}) or {}
            if item["tool"] == "Spectrum_Grid_Analyzer":
                score = max(score, self._bounded_float(scores.get("peak_density_score"), 0.0))
                score = max(score, self._bounded_float(scores.get("energy_hole_score"), 0.0))
                score = max(score, self._bounded_float(scores.get("spatial_concentration_score"), 0.0))
            for key in ["ai_generated_score"]:
                score = max(score, self._bounded_float(scores.get(key), 0.0))
        return score

    def _tamper_score(self, tool_calls: list[dict[str, Any]], image_context: dict[str, Any] | None = None) -> float:
        score = 0.0
        cfa_allowed = True if image_context is None else bool(image_context.get("likely_camera_source"))
        for item in tool_calls:
            summary = item.get("summary", {})
            contract_score = self._positive_evidence_contract_score(summary)
            if contract_score is not None:
                if item["tool"] != "CFA_Validator" or cfa_allowed:
                    score = max(score, contract_score)
                continue
            scores = summary.get("heuristic_scores") or summary.get("evidence_scores", {}) or {}
            for key in [
                "noise_inconsistency_score",
                "copy_move_consistency_score",
                "tamper_score",
            ]:
                score = max(score, self._bounded_float(scores.get(key), 0.0))
            if cfa_allowed:
                for key in ["cfa_inconsistency_score", "cfa_discontinuity_score"]:
                    score = max(score, self._bounded_float(scores.get(key), 0.0))
            if item["tool"] == "SRM_Filter":
                score = max(score, min(1.0, self._bounded_float(scores.get("avg_abs_kurtosis"), 0.0) / 8.0))
        return score

    def _positive_evidence_contract_score(self, summary: dict[str, Any]) -> float | None:
        contract = summary.get("evidence_contract")
        if not isinstance(contract, dict) or not contract:
            return None
        if contract.get("decision") != "positive":
            return 0.0
        return self._bounded_float(contract.get("score"), 0.0)

    def _fallback_probability(self, tool_calls: list[dict[str, Any]], image_context: dict[str, Any] | None = None) -> float:
        return round(max(self._ai_score(tool_calls), self._tamper_score(tool_calls, image_context)), 4)

    def _calibrate_probability(
        self,
        label: str,
        model_probability: float,
        tool_calls: list[dict[str, Any]],
        suspicious_regions: list[dict[str, Any]],
        image_context: dict[str, Any] | None = None,
    ) -> float:
        evidence_probability = self._fallback_probability(tool_calls, image_context)
        model_probability = self._bounded_float(model_probability, 0.55 if self._is_real_label(label) else 0.65)
        if self._is_real_label(label):
            if evidence_probability >= 0.62:
                return self._bounded_float(min(model_probability, 0.58), 0.58)
            return model_probability

        calibrated = max(model_probability, min(1.0, evidence_probability * 0.9))
        if self._is_local_tampering_label(label) and not suspicious_regions:
            calibrated = min(calibrated, 0.6)
        return self._bounded_float(calibrated, evidence_probability)

    def _calibrate_confidence(
        self,
        label: str,
        model_confidence: float,
        tool_calls: list[dict[str, Any]],
        suspicious_regions: list[dict[str, Any]],
    ) -> float:
        successful_tools = sum(1 for item in tool_calls if item.get("summary", {}).get("success") is True)
        failed_tools = sum(1 for item in tool_calls if item.get("summary", {}).get("status") == "error")
        calibrated = model_confidence
        if successful_tools == 0 and tool_calls:
            calibrated = min(calibrated, 0.8)
        if failed_tools:
            calibrated = min(calibrated, 0.82)
        if self._is_local_tampering_label(label) and not suspicious_regions:
            calibrated = min(calibrated, 0.55)
        return self._bounded_float(calibrated, 0.55 if self._is_real_label(label) else 0.65)

    def _infer_forgery_types(self, label: str, tool_calls: list[dict[str, Any]]) -> list[str]:
        used = {item["tool"] for item in tool_calls}
        if self._is_ai_label(label):
            return ["none"]
        if self._is_local_tampering_label(label):
            if "Noise_Analyzer" in used:
                return ["splicing"]
            if "PatchDense_Scanner" in used:
                return ["copy_move"]
            if "CFA_Validator" in used:
                return ["inpainting"]
            return ["unknown"]
        return ["none"]

    def _normalize_forgery_types(
        self,
        value: Any,
        label: str,
        tool_calls: list[dict[str, Any]],
        language: str,
    ) -> list[str]:
        none_value = "无" if is_zh(language) else "none"
        if self._is_real_label(label) or self._is_ai_label(label):
            return [none_value]

        candidates = value if isinstance(value, list) else []
        candidates = [str(item).strip() for item in candidates if str(item or "").strip()]
        candidates = [item for item in candidates if item.lower() not in {"none", "无"}]
        if candidates:
            return [self._canonical_forgery_type(candidates[0], language)]

        inferred = self._infer_forgery_types(label, tool_calls)
        if not inferred:
            return ["未知" if is_zh(language) else "unknown"]
        return [self._canonical_forgery_type(inferred[0], language)]

    def _canonical_forgery_type(self, value: Any, language: str) -> str:
        first = str(value or "").strip()
        key = first.lower().replace("-", "_").replace(" ", "_")
        zh_aliases = {
            "无": "无",
            "拼接": "拼接",
            "复制移动": "复制移动",
            "内容移除": "内容移除",
            "内容删除": "内容移除",
            "删除": "内容移除",
            "局部修复": "内容移除",
            "修复": "内容移除",
            "内容替换": "内容替换",
            "替换": "内容替换",
            "人脸篡改": "人脸篡改",
            "换脸": "人脸篡改",
            "人脸替换": "人脸篡改",
            "局部重绘": "局部重绘",
            "重绘": "局部重绘",
            "ai辅助编辑": "局部重绘",
            "ai局部编辑": "局部重绘",
            "全局编辑": "全局编辑",
            "修饰改动": "全局编辑",
            "未知": "未知",
        }
        en_to_zh = {
            "none": "无",
            "splicing": "拼接",
            "copy_move": "复制移动",
            "copy_move_forgery": "复制移动",
            "inpainting": "内容移除",
            "content_removal": "内容移除",
            "content_deletion": "内容移除",
            "content_replacement": "内容替换",
            "face_swap": "人脸篡改",
            "deepfake": "人脸篡改",
            "face_tampering": "人脸篡改",
            "facial_tampering": "人脸篡改",
            "repainting": "局部重绘",
            "local_repainting": "局部重绘",
            "ai_edited": "局部重绘",
            "ai_generated": "局部重绘",
            "retouching": "全局编辑",
            "global_editing": "全局编辑",
            "global_edit": "全局编辑",
            "unknown": "未知",
            "unknown_forgery": "未知",
        }
        zh_value = zh_aliases.get(first) or zh_aliases.get(first.lower()) or en_to_zh.get(key) or "未知"
        if is_zh(language):
            return zh_value
        zh_to_en = {
            "无": "none",
            "拼接": "splicing",
            "复制移动": "copy_move",
            "内容移除": "content_removal",
            "内容替换": "content_replacement",
            "人脸篡改": "face_tampering",
            "局部重绘": "local_repainting",
            "全局编辑": "global_editing",
            "未知": "unknown",
        }
        return zh_to_en.get(zh_value, "unknown")

    def _payload_tool_evidence(self, payload: dict[str, Any], tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
        if isinstance(payload.get("tool_evidence"), list):
            return payload["tool_evidence"]
        if isinstance(payload.get("evidence_chain"), list):
            return payload["evidence_chain"]
        return self._build_evidence_chain(tool_calls)

    def _merge_consistency_conflicts(
        self,
        limitations: list[Any],
        consistency_check: Any,
        language: str,
    ) -> list[Any]:
        merged = list(limitations)
        if isinstance(consistency_check, dict):
            conflicts = consistency_check.get("conflicts")
            if isinstance(conflicts, list):
                prefix = "工具和视觉观察存在不一致：" if is_zh(language) else "Conflict between tools and visual observation: "
                merged.extend(f"{prefix}{item}" for item in conflicts if item)
        return merged

    def _normalize_label(self, label: Any, probability: Any = None, language: str = "zh") -> str:
        text = str(label or "").strip()
        labels = self.LABELS_ZH if is_zh(language) else self.LABELS_EN
        if text in labels:
            return text
        lowered = text.lower()
        label_aliases = {
            "real_image": "Real_Image",
            "real image": "Real_Image",
            "completely_real": "Real_Image",
            "completely real": "Real_Image",
            "authentic": "Real_Image",
            "real": "Real_Image",
            "forged_image": "Forged_Image",
            "forged image": "Forged_Image",
            "local_tampering": "Forged_Image",
            "local tampering": "Forged_Image",
            "forgery_detected": "Forged_Image",
            "forgery detected": "Forged_Image",
            "partially_tampered": "Forged_Image",
            "tampered": "Forged_Image",
            "manipulated": "Forged_Image",
            "synthetic_image": "Synthetic_Image",
            "synthetic image": "Synthetic_Image",
            "ai_manipulated": "Synthetic_Image",
            "ai manipulated": "Synthetic_Image",
            "ai_generated": "Synthetic_Image",
            "ai generated": "Synthetic_Image",
            "completely_ai_generated": "Synthetic_Image",
            "completely ai generated": "Synthetic_Image",
            "fully_ai_generated": "Synthetic_Image",
            "fully ai generated": "Synthetic_Image",
        }
        zh_aliases = {
            "真实图像": "真实图像",
            "伪造图像": "伪造图像",
            "合成图像": "合成图像",
            "完全真实": "真实图像",
            "局部篡改": "伪造图像",
            "存在伪造": "伪造图像",
            "AI操纵": "合成图像",
            "AI生成": "合成图像",
            "完全AI生成": "合成图像",
        }
        if is_zh(language):
            if text in zh_aliases:
                return zh_aliases[text]
            if lowered in self.VERDICT_TO_LABEL_ZH:
                return self.VERDICT_TO_LABEL_ZH[lowered]
            if lowered in label_aliases:
                return self._english_label_to_zh(label_aliases[lowered])
        else:
            if lowered in self.VERDICT_TO_LABEL_EN:
                return self.VERDICT_TO_LABEL_EN[lowered]
            if lowered in label_aliases:
                return label_aliases[lowered]
            if text in zh_aliases:
                return self._zh_label_to_english(text)
        if lowered == "uncertain":
            if self._bounded_float(probability, 0.0) >= 0.6:
                return "合成图像" if is_zh(language) else "Synthetic_Image"
            return "真实图像" if is_zh(language) else "Real_Image"
        return "真实图像" if is_zh(language) else "Real_Image"

    def _label_to_verdict(self, label: str) -> str:
        return {
            "真实图像": "authentic",
            "伪造图像": "tampered",
            "合成图像": "ai_manipulated",
            "完全真实": "authentic",
            "存在伪造": "tampered",
            "完全AI生成": "ai_manipulated",
            "Real_Image": "authentic",
            "Forged_Image": "tampered",
            "Synthetic_Image": "ai_manipulated",
            "Completely_Real": "authentic",
            "Forgery_Detected": "tampered",
            "Completely_AI_Generated": "ai_manipulated",
        }.get(label, "authentic")

    def _fallback_summary(self, label: str, probability: float, regions: list[dict[str, Any]], language: str = "zh") -> str:
        if not is_zh(language):
            region_text = f"; suspicious region: {regions[0]['region']}" if regions else ""
            return f"The image is classified as {label}, with an estimated manipulation probability of {probability:.2f}{region_text}."
        region_text = f"，可疑区域：{regions[0]['region']}" if regions else ""
        return f"图像判定为{label}，被操纵概率约为 {probability:.2f}{region_text}。"

    def _is_real_label(self, label: str) -> bool:
        return label in {"真实图像", "Real_Image", "完全真实", "Completely_Real"}

    def _is_local_tampering_label(self, label: str) -> bool:
        return label in {"伪造图像", "Forged_Image", "存在伪造", "Forgery_Detected"}

    def _is_ai_label(self, label: str) -> bool:
        return label in {"合成图像", "Synthetic_Image", "完全AI生成", "Completely_AI_Generated"}

    def _english_label_to_zh(self, label: str) -> str:
        return {
            "Real_Image": "真实图像",
            "Forged_Image": "伪造图像",
            "Synthetic_Image": "合成图像",
            "Completely_Real": "真实图像",
            "Local_Tampering": "伪造图像",
            "Forgery_Detected": "伪造图像",
            "AI_Manipulated": "合成图像",
            "AI_Generated": "合成图像",
            "Completely_AI_Generated": "合成图像",
        }.get(label, "真实图像")

    def _zh_label_to_english(self, label: str) -> str:
        return {
            "真实图像": "Real_Image",
            "伪造图像": "Forged_Image",
            "合成图像": "Synthetic_Image",
            "完全真实": "Real_Image",
            "局部篡改": "Forged_Image",
            "存在伪造": "Forged_Image",
            "AI操纵": "Synthetic_Image",
            "AI生成": "Synthetic_Image",
            "完全AI生成": "Synthetic_Image",
        }.get(label, "Real_Image")

    def _build_user_analysis(
        self,
        evidence_chain: list[dict[str, Any]],
        limitations: list[Any],
        language: str,
        visual_analysis: str = "",
    ) -> str:
        if not is_zh(language):
            parts = []
            if visual_analysis:
                parts.append(f"Visual observation: {visual_analysis}")
            elif evidence_chain:
                parts.append(
                    "Auxiliary image checks were reviewed for local tampering, AI-generation clues, and source consistency."
                )
            return " ".join(parts) or "The image was reviewed with visual observation and auxiliary forensic checks."

        parts = []
        if visual_analysis:
            parts.append(f"画面观察：{visual_analysis}")
        elif evidence_chain:
            parts.append("系统结合辅助图像检测结果，重点核验是否存在局部篡改、完全 AI 生成或来源异常。")
        return "".join(parts) or "系统结合画面观察和辅助图像检测结果完成判断。"

    def _normalize_user_answer(
        self,
        value: Any,
        final_summary: str,
        evidence_chain: list[dict[str, Any]],
        limitations: list[Any],
        language: str,
        visual_analysis: str = "",
    ) -> dict[str, Any]:
        if isinstance(value, dict):
            judgment = str(value.get("judgment") or value.get("conclusion") or "").strip()
            reason_analysis = str(value.get("reason_analysis") or value.get("analysis") or "").strip()
            notes = value.get("notes")
        else:
            judgment = ""
            reason_analysis = ""
            notes = []

        if not judgment:
            judgment = final_summary
        if not reason_analysis:
            reason_analysis = self._build_user_analysis(evidence_chain, limitations, language, visual_analysis)
        if not isinstance(notes, list):
            notes = [notes] if notes else []
        clean_notes = [str(item).strip() for item in notes if str(item or "").strip()]

        return {
            "judgment": judgment,
            "reason_analysis": reason_analysis,
            "notes": clean_notes,
            "conclusion": judgment,
            "analysis": reason_analysis,
        }

    def _collect_artifacts(self, tool_calls: list[dict[str, Any]]) -> dict[str, str]:
        artifacts: dict[str, str] = {}
        for item in tool_calls:
            summary = item.get("summary", {})
            for key, value in self._extract_artifact_paths(summary).items():
                if value:
                    artifacts[f"{item['tool']}.{key}"] = value
        return artifacts

    def _extract_artifact_paths(self, summary: dict[str, Any]) -> dict[str, str]:
        artifacts = dict(summary.get("artifacts", {}) or {})
        anomaly_mask = summary.get("anomaly_mask") or {}
        if anomaly_mask.get("mask_path"):
            artifacts["mask_path"] = anomaly_mask["mask_path"]
        return artifacts

    def _annotate_image(self, image_path: str, regions: list[dict[str, Any]], session_dir: Path) -> str:
        try:
            image = Image.open(image_path).convert("RGB")
            draw = ImageDraw.Draw(image)
            width, height = image.size
            outline_width = max(4, int(round(max(width, height) * 0.006)))
            for item in regions:
                x0, y0, x1, y1 = [int(x) for x in item["region"].split(",")]
                x0 = max(0, min(width - 1, x0))
                y0 = max(0, min(height - 1, y0))
                x1 = max(0, min(width - 1, x1))
                y1 = max(0, min(height - 1, y1))
                if x1 <= x0 or y1 <= y0:
                    continue
                draw.rectangle((x0, y0, x1, y1), outline=(255, 0, 0), width=outline_width)
            out_path = session_dir / "annotated_suspicious_regions.png"
            image.save(out_path)
            return str(out_path)
        except Exception:
            return ""

    def _bounded_float(self, value: Any, default: float) -> float:
        try:
            number = float(value)
        except (TypeError, ValueError):
            number = default
        return round(max(0.0, min(1.0, number)), 4)

    def _image_to_data_url(self, image_path: str) -> str:
        path = Path(image_path)
        if not path.exists():
            return ""
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        try:
            image_bytes = path.read_bytes()
        except OSError:
            return ""
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    def _run_tool(
        self,
        tool_name: str,
        image_path: str,
        session_dir: Path,
        parameters: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        tool = self.tools.get(tool_name)
        if not tool:
            return {
                "success": False,
                "status": "error",
                "description": f"Unknown vision tool: {tool_name}",
                "result": {},
                "error_message": f"Unsupported tool {tool_name}",
                "metadata": {"tool_name": tool_name, "execution_time_ms": 0},
            }
        tool_output_dir = session_dir / tool_name.lower()
        tool_output_dir.mkdir(parents=True, exist_ok=True)
        args = dict(parameters or {})
        args["image_path"] = image_path
        args.setdefault("image_input", image_path)
        args.setdefault("output_dir", str(tool_output_dir))
        return tool(json.dumps(args, ensure_ascii=False))

    def _default_tools(self) -> dict[str, Any]:
        tools = {
            "Spectrum_Grid_Analyzer": SpectrumGridAnalyzer(),
            "SRM_Filter": SRMFilter(),
            "Noise_Analyzer": NoiseAnalyzer(),
            "PatchDense_Scanner": PatchDenseScanner(),
            "CFA_Validator": CFAValidator(),
            "ROI_Extractor": ROIExtractor(),
            "Image_Resizer": ImageResizer(),
            "Visual_Enhancer": VisualEnhancer(),
        }
        return tools

    def _parse_json(self, raw_output: str) -> dict[str, Any]:
        text = (raw_output or "").strip()
        candidates = [text]
        candidates.extend(match.group(1).strip() for match in re.finditer(r"```(?:json)?\s*(.*?)```", text, re.S))
        decoder = json.JSONDecoder()
        for candidate in candidates:
            for idx, char in enumerate(candidate):
                if char != "{":
                    continue
                try:
                    parsed, _ = decoder.raw_decode(candidate[idx:])
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict):
                    return parsed
        return {}

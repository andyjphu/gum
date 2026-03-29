# Author: Andy Phu
# Retroactive observer with multi-pass analysis for egocentric footage.
# Odd passes extract activities, even passes infer intents.
# Each pass sees the previous pass_window_size passes as context.

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import traceback
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Any

from openai import AsyncOpenAI

from .observer import Observer
from ..schemas import (
    Update,
    PrimitiveStateSchema,
    HiddenIntentSchema,
    RefinedPrimitiveSchema,
    RefinedIntentSchema,
    get_schema,
)
from ..invoke import invoke
from ..config import (
    DEFAULT_NUM_PASSES,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    DEFAULT_TEMPORAL_WINDOW_SIZE,
    DEFAULT_PASS_WINDOW_SIZE,
    DEFAULT_MAX_STATES_IN_CONTEXT,
    PASS_OUTPUT_DIR,
    SHARED_DIR,
    TRAFFIC_LOG_DIR,
)
from gum.prompts.state_prompt import (
    PRIMITIVE_STATE_SYSTEM,
    PRIMITIVE_STATE_PROMPT,
    HIDDEN_INTENT_SYSTEM,
    HIDDEN_INTENT_PROMPT,
    REFINED_PRIMITIVE_SYSTEM,
    REFINED_PRIMITIVE_PROMPT,
    REFINED_INTENT_SYSTEM,
    REFINED_INTENT_PROMPT,
    PRIMITIVE_SUMMARY_PROMPT,
    INTENT_SUMMARY_PROMPT,
    get_pass_type,
    get_context_passes,
    build_temporal_context,
    build_multi_pass_context,
    _collapse_to_runs,
    _pass_type_to_file_key,
)
from gum.metrics import (
    compute_all_metrics,
    export_for_visualization,
    save_metrics_json,
)

###############################################################################
# Constants                                                                   #
###############################################################################

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_HISTORY_K = 10
SHORT_SLEEP_SEC = 0.05
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


###############################################################################
# Retro observer with multi-pass (alternating activity/intent)                #
###############################################################################


class Retro(Observer):
    """Observer that processes screenshots with multi-pass analysis.

    Implements an alternating activity/intent architecture:
    - Odd passes (1, 3, 5): Extract/refine activity (observable) states
    - Even passes (2, 4, 6): Infer/refine hidden intents

    Context window logic:
    - Pass 1: No context (raw observation)
    - Pass 2+: Sliding window of previous pass_window_size passes

    Args:
        images_dir: Directory containing screenshots to process.
        model_name: GPT model to use for vision analysis.
        num_passes: Number of analysis passes (default: 6).
        context_window_size: Temporal frame window (backward/forward split).
        temporal_window_size: Deprecated, kept for entrypoint compatibility.
        pass_window_size: Number of prior passes in the sliding window.
        max_states_in_context: Max unique state labels per pass in multi-pass context.
        history_k: Number of recent screenshots to keep in history.
        process_delay: Delay between processing each image (seconds).
        debug: Enable debug logging.
        api_key: OpenAI API key.
        api_base: OpenAI API base URL.
    """

    def __init__(
        self,
        images_dir: str,
        model_name: str = DEFAULT_MODEL,
        num_passes: int = DEFAULT_NUM_PASSES,
        context_window_size: int = DEFAULT_CONTEXT_WINDOW_SIZE,
        temporal_window_size: int = DEFAULT_TEMPORAL_WINDOW_SIZE,
        pass_window_size: int = DEFAULT_PASS_WINDOW_SIZE,
        max_states_in_context: int = DEFAULT_MAX_STATES_IN_CONTEXT,
        history_k: int = DEFAULT_HISTORY_K,
        process_delay: float = SHORT_SLEEP_SEC,
        debug: bool = False,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """Initialize the Retro observer with multi-pass support."""
        super().__init__()

        # Validate and setup image directory
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if not self.images_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.images_dir}")

        # Multi-pass configuration
        self.num_passes = max(1, num_passes)
        self.context_window_size = context_window_size
        self.pass_window_size = pass_window_size
        self.max_states_in_context = max_states_in_context
        self.temporal_window_size = temporal_window_size  # deprecated, kept for compat

        # Model config
        self.model_name = model_name
        self.debug = debug
        self.process_delay = process_delay
        self.history_k = history_k

        # History (per video/subfolder)
        self._history: deque[str] = deque(maxlen=max(0, history_k))
        self._current_video: Optional[str] = None

        # Pass state storage (persists across all passes for a video)
        self._pass_states: Dict[int, List[Dict[str, Any]]] = {}  # pass_num -> states
        self._pass_summaries: Dict[int, str] = {}  # pass_num -> summary text
        self._current_pass: int = 1

        # Output directory
        self.output_dir = PASS_OUTPUT_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = logging.getLogger("Retro")

        # OpenAI client
        resolved_key = api_key or os.getenv("GUM_LM_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError("API key not configured — set GUM_LM_API_KEY or OPENAI_API_KEY")
        self.client = AsyncOpenAI(
            base_url=api_base or os.getenv("VLLM_ENDPOINT"),
            api_key=resolved_key,
        )

        # To ensure gum stops when retro stops
        self.stopped = asyncio.Event()

    # ─────────────────────────────── Helper methods

    def _get_video_name(self, img_path: Path) -> Optional[str]:
        """Extract video/subfolder name from image path."""
        relative = img_path.relative_to(self.images_dir)
        if len(relative.parts) > 1:
            return relative.parts[0]
        return None

    def _get_sorted_images_recursive(self) -> list[tuple[Path, Optional[str]]]:
        """Get all images recursively, sorted by video then filename."""
        images = []
        for p in self.images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                video_name = self._get_video_name(p)
                images.append((p, video_name))
        return sorted(images, key=lambda x: (x[1] or "", x[0].name))

    @staticmethod
    def _encode_image(img_path: str) -> str:
        """Encode an image file as base64."""
        with open(img_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    def _get_pass_suffix(self) -> str:
        """Get the current pass suffix (e.g., '_P1', '_P2')."""
        return f"_P{self._current_pass}"

    def _get_pass_type_str(self) -> str:
        """Get the current pass type as string."""
        return get_pass_type(self._current_pass)


    def _save_single_state(self, video_name: Optional[str], state_data: Dict[str, Any]) -> None:
        """Save a single state entry to its timestamp folder immediately.

        Layout: data/passes/{video}/{timestamp}/states_{file_key}_P{N}.json
        """
        suffix = self._get_pass_suffix()
        pass_type = self._get_pass_type_str()
        file_key = _pass_type_to_file_key(pass_type)
        video_folder = video_name or ""
        output_folder = self.output_dir / video_folder

        img_path = state_data.get("image_path", "")
        m = re.search(r"(\d{8}_\d{6})", img_path)
        ts_key = m.group(1) if m else str(hash(img_path))

        ts_dir = output_folder / ts_key
        ts_dir.mkdir(parents=True, exist_ok=True)

        filename = f"states_{file_key}{suffix}.json"
        ts_file = ts_dir / filename
        with open(ts_file, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

    def _save_pass_output(self, video_name: Optional[str], data_type: str, data: Any) -> Path:
        """Save non-state pass output (summaries, etc).

        Layout: data/passes/{video}/{data_type}/{data_type}_{file_key}_P{N}.json
        """
        suffix = self._get_pass_suffix()
        pass_type = self._get_pass_type_str()
        file_key = _pass_type_to_file_key(pass_type)
        video_folder = video_name or ""
        output_folder = self.output_dir / video_folder

        sub_dir = output_folder / data_type
        sub_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{data_type}_{file_key}{suffix}.json"
        filepath = sub_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        if self.debug:
            self.logger.info(f"Saved {data_type} to {filepath}")

        return filepath

    def _save_provenance(self, video_name: Optional[str]) -> None:
        """Generate unified provenance.json for each timestamp after all passes complete.

        Creates: data/passes/{video}/{timestamp}/provenance.json

        Aggregates all passes for that frame with log paths for auditability.
        """
        video_folder = video_name or ""
        output_folder = self.output_dir / video_folder
        log_base = TRAFFIC_LOG_DIR / video_folder

        # Group states by timestamp
        timestamps: Dict[str, Dict[int, Dict[str, Any]]] = {}

        for pass_num, states in self._pass_states.items():
            pass_type = get_pass_type(pass_num)
            for state_data in states:
                img_path = state_data.get("image_path", "")
                m = re.search(r"(\d{8}_\d{6})", img_path)
                if not m:
                    continue
                ts = m.group(1)
                if ts not in timestamps:
                    timestamps[ts] = {}
                timestamps[ts][pass_num] = {
                    "type": pass_type,
                    "state": state_data.get("primitive_state") or state_data.get("hidden_intent", "unknown"),
                    "confidence": state_data.get("confidence") or state_data.get("intent_confidence", 0),
                    "body_position": state_data.get("body_position"),
                    "objects": state_data.get("objects_interacted", []),
                    "supporting_primitives": state_data.get("supporting_primitives", []),
                    "alternatives": state_data.get("alternative_intents", []),
                    "refined_from": state_data.get("refined_from"),
                    "refinement_reason": state_data.get("refinement_reason"),
                    "informed_by_intent": state_data.get("informed_by_intent"),
                    "validated_by_outcome": state_data.get("validated_by_outcome"),
                    "log_path": str(log_base / f"{ts}-Retro_P{pass_num}:{pass_type}"),
                    "image_path": img_path,
                }

        # Write provenance.json for each timestamp
        sorted_timestamps = sorted(timestamps.keys())
        for i, ts in enumerate(sorted_timestamps):
            ts_dir = output_folder / ts
            ts_dir.mkdir(parents=True, exist_ok=True)

            passes_data = timestamps[ts]
            # Determine final state (highest activity pass)
            activity_passes = [p for p in passes_data.keys() if passes_data[p]["type"] == "activity"]
            final_pass = max(activity_passes) if activity_passes else max(passes_data.keys())
            final_state = passes_data[final_pass]["state"]

            # Determine transition from previous frame
            transition = None
            if i > 0:
                prev_ts = sorted_timestamps[i - 1]
                prev_passes = timestamps.get(prev_ts, {})
                prev_activity_passes = [p for p in prev_passes.keys() if prev_passes[p]["type"] == "activity"]
                if prev_activity_passes:
                    prev_final_pass = max(prev_activity_passes)
                    prev_state = prev_passes[prev_final_pass]["state"]
                    if final_state != prev_state:
                        # Get intent from current frame's highest intent pass as trigger
                        intent_passes = [p for p in passes_data.keys() if passes_data[p]["type"] == "intent"]
                        trigger = passes_data[max(intent_passes)]["state"] if intent_passes else f"to_{final_state}"
                        transition = {
                            "source": prev_state,
                            "dest": final_state,
                            "trigger": trigger,
                            "timestamp": i * 5.0,
                        }

            provenance = {
                "timestamp": ts,
                "image_path": passes_data[final_pass].get("image_path", ""),
                "passes": {f"P{p}": data for p, data in sorted(passes_data.items())},
                "final_state": final_state,
                "final_confidence": passes_data[final_pass]["confidence"],
                "transition_from_prev": transition,
            }

            prov_file = ts_dir / "provenance.json"
            with open(prov_file, "w") as f:
                json.dump(provenance, f, indent=2, default=str)

        self.logger.info(f"Saved provenance for {len(timestamps)} frames")

    # ─────────────────────────────── Vision API calls

    async def _call_vision_api(
        self,
        prompt: str,
        img_paths: list[str],
        video_name: Optional[str] = None,
        response_format: Optional[dict] = None,
        max_tokens: Optional[int] = None,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Call GPT Vision API to analyze images.

        When system_prompt is provided, instructions go in the system role
        and prompt + images go in the user role. This helps the model
        distinguish task instructions from context data.
        """
        content = []

        # Only encode images if we have any
        if img_paths:
            encoded_images = await asyncio.gather(
                *[asyncio.to_thread(self._encode_image, p) for p in img_paths]
            )
            content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
                }
                for encoded in encoded_images
            ]

        content.append({"type": "text", "text": prompt})

        if self.debug:
            self.logger.info(f"[Pass {self._current_pass}] Sending {len(img_paths)} image(s)")

        # Build debug tag with pass suffix and type
        pass_suffix = self._get_pass_suffix()
        pass_type = self._get_pass_type_str()
        debug_tag = f"[Retro{pass_suffix}:{pass_type}]"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        extra_kwargs = {}
        if max_tokens is not None:
            extra_kwargs["max_tokens"] = max_tokens

        rsp = await invoke(
            model=self.model_name,
            messages=messages,
            response_format=response_format or {"type": "text"},
            debug_tag=debug_tag,
            debug_img_paths=img_paths,
            debug_path=video_name,
            client=self.client,
            **extra_kwargs,
        )

        choice = rsp.choices[0]
        if max_tokens is not None and choice.finish_reason == "length":
            self._log_hard_cap_hit(
                video_name=video_name,
                max_tokens=max_tokens,
                truncated_content=choice.message.content,
                debug_tag=debug_tag,
            )

        return choice.message.content

    def _log_hard_cap_hit(
        self,
        video_name: Optional[str],
        max_tokens: int,
        truncated_content: str,
        debug_tag: str,
    ) -> None:
        """Log when an LLM response was truncated by the max_tokens hard cap."""
        from datetime import datetime, timezone
        log_dir = SHARED_DIR / "log"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "hard_cap_truncations.log"

        timestamp = datetime.now(timezone.utc).isoformat()
        video = video_name or "unknown"
        entry = (
            f"[{timestamp}] {debug_tag} video={video} "
            f"pass={self._current_pass} max_tokens={max_tokens}\n"
            f"  TRUNCATED OUTPUT ({len(truncated_content)} chars):\n"
            f"  {truncated_content}\n"
            f"{'─' * 80}\n"
        )

        try:
            with open(log_file, "a") as f:
                f.write(entry)
            self.logger.warning(
                f"Hard cap hit: {debug_tag} video={video} pass={self._current_pass} "
                f"— output truncated at {max_tokens} tokens. See {log_file}"
            )
        except OSError as exc:
            self.logger.error(f"Failed to write hard cap log: {exc}")

    # ─────────────────────────────── Pass 1: Activity State Extraction

    async def _extract_activity_pass1(
        self,
        img_path: str,
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract activity (observable) state from a single image (Pass 1)."""
        max_retries = 3
        last_exc = None
        for attempt in range(max_retries):
            try:
                result = await self._call_vision_api(
                    prompt=PRIMITIVE_STATE_PROMPT,
                    img_paths=[img_path],
                    video_name=video_name,
                    response_format=get_schema(PrimitiveStateSchema.model_json_schema()),
                    system_prompt=PRIMITIVE_STATE_SYSTEM,
                )

                parsed = json.loads(result)

                return {
                    "primitive_state": parsed.get("primitive_state", "unknown"),
                    "body_position": parsed.get("body_position"),
                    "objects_interacted": parsed.get("objects_interacted", []),
                    "confidence": parsed.get("confidence", 5),
                    "image_path": img_path,
                    "video_name": video_name,
                    "pass": self._current_pass,
                    "pass_type": "activity"
                }

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    self.logger.warning(f"Activity extraction attempt {attempt + 1} failed, retrying: {exc}")
                    continue
                if self.debug:
                    self.logger.error(f"Activity extraction failed after {max_retries} attempts: {exc}")
                    self.logger.error(traceback.format_exc())
        return {
            "primitive_state": "extraction_failed",
            "confidence": 0,
            "image_path": img_path,
            "video_name": video_name,
            "pass": self._current_pass,
            "pass_type": "activity",
            "error": str(last_exc)
        }

    # ─────────────────────────────── Pass 2: Hidden Intent Inference

    async def _infer_intent_pass2(
        self,
        img_path: str,
        frame_idx: int,
        total_frames: int,
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Infer hidden intent from activity states (Pass 2)."""
        prior_summary = self._pass_summaries.get(1, "No prior summary available.")

        temporal_context = build_temporal_context(
            self._pass_states,
            frame_idx,
            total_frames,
            self._current_pass,
            self.context_window_size,
        )

        prompt = HIDDEN_INTENT_PROMPT.format(
            prior_summary=prior_summary,
            temporal_context=temporal_context,
        )

        max_retries = 3
        last_exc = None
        for attempt in range(max_retries):
            try:
                result = await self._call_vision_api(
                    prompt=prompt,
                    img_paths=[img_path],
                    video_name=video_name,
                    response_format=get_schema(HiddenIntentSchema.model_json_schema()),
                    system_prompt=HIDDEN_INTENT_SYSTEM,
                )

                parsed = json.loads(result)

                return {
                    "hidden_intent": parsed.get("hidden_intent", "unknown"),
                    "supporting_primitives": parsed.get("supporting_primitives", []),
                    "intent_confidence": parsed.get("intent_confidence", 5),
                    "alternative_intents": parsed.get("alternative_intents", []),
                    "image_path": img_path,
                    "video_name": video_name,
                    "pass": self._current_pass,
                    "pass_type": "intent"
                }

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    self.logger.warning(f"Intent inference attempt {attempt + 1} failed, retrying: {exc}")
                    continue
                if self.debug:
                    self.logger.error(f"Intent inference failed after {max_retries} attempts: {exc}")
                    self.logger.error(traceback.format_exc())
        return {
            "hidden_intent": "inference_failed",
            "intent_confidence": 0,
            "image_path": img_path,
            "video_name": video_name,
            "pass": self._current_pass,
            "pass_type": "intent",
            "error": str(last_exc)
        }

    # ─────────────────────────────── Pass 3+: Refined Activity State

    async def _refine_activity(
        self,
        img_path: str,
        frame_idx: int,
        total_frames: int,
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refine activity state with context from prior passes (Pass 3, 5, ...)."""
        context_passes = get_context_passes(self._current_pass, self.pass_window_size)

        prior_context = build_multi_pass_context(
            self._pass_states,
            self._pass_summaries,
            context_passes,
            self.max_states_in_context,
        )

        temporal_context = build_temporal_context(
            self._pass_states,
            frame_idx,
            total_frames,
            self._current_pass,
            self.context_window_size,
        )

        same_type_prior = self._current_pass - 2
        prior_label = "unknown"
        if same_type_prior >= 1 and same_type_prior in self._pass_states:
            if frame_idx < len(self._pass_states[same_type_prior]):
                prior_label = self._pass_states[same_type_prior][frame_idx].get(
                    "primitive_state", "unknown"
                )

        prompt = REFINED_PRIMITIVE_PROMPT.format(
            prior_label=prior_label,
            prior_context=prior_context,
            temporal_context=temporal_context,
        )

        max_retries = 3
        last_exc = None
        for attempt in range(max_retries):
            try:
                result = await self._call_vision_api(
                    prompt=prompt,
                    img_paths=[img_path],
                    video_name=video_name,
                    response_format=get_schema(RefinedPrimitiveSchema.model_json_schema()),
                    system_prompt=REFINED_PRIMITIVE_SYSTEM,
                )

                parsed = json.loads(result)

                return {
                    "primitive_state": parsed.get("primitive_state", "unknown"),
                    "body_position": parsed.get("body_position"),
                    "objects_interacted": parsed.get("objects_interacted", []),
                    "confidence": parsed.get("confidence", 5),
                    "refined_from": parsed.get("refined_from"),
                    "refinement_reason": parsed.get("refinement_reason"),
                    "informed_by_intent": parsed.get("informed_by_intent"),
                    "image_path": img_path,
                    "video_name": video_name,
                    "pass": self._current_pass,
                    "pass_type": "activity"
                }

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    self.logger.warning(f"Activity refinement attempt {attempt + 1} failed, retrying: {exc}")
                    continue
                if self.debug:
                    self.logger.error(f"Activity refinement failed after {max_retries} attempts: {exc}")
                    self.logger.error(traceback.format_exc())
        return {
            "primitive_state": "refinement_failed",
            "confidence": 0,
            "image_path": img_path,
            "video_name": video_name,
            "pass": self._current_pass,
            "pass_type": "activity",
            "error": str(last_exc)
        }

    # ─────────────────────────────── Pass 4+: Refined Hidden Intent

    async def _refine_intent(
        self,
        img_path: str,
        frame_idx: int,
        total_frames: int,
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refine hidden intent with broader context (Pass 4, 6, ...)."""
        context_passes = get_context_passes(self._current_pass, self.pass_window_size)

        prior_context = build_multi_pass_context(
            self._pass_states,
            self._pass_summaries,
            context_passes,
            self.max_states_in_context,
        )

        temporal_context = build_temporal_context(
            self._pass_states,
            frame_idx,
            total_frames,
            self._current_pass,
            self.context_window_size,
        )

        same_type_prior = self._current_pass - 2
        prior_label = "unknown"
        if same_type_prior >= 1 and same_type_prior in self._pass_states:
            if frame_idx < len(self._pass_states[same_type_prior]):
                prior_label = self._pass_states[same_type_prior][frame_idx].get(
                    "hidden_intent", "unknown"
                )

        prompt = REFINED_INTENT_PROMPT.format(
            context_passes=", ".join(str(p) for p in context_passes),
            prior_label=prior_label,
            prior_context=prior_context,
            temporal_context=temporal_context,
        )

        max_retries = 3
        last_exc = None
        for attempt in range(max_retries):
            try:
                result = await self._call_vision_api(
                    prompt=prompt,
                    img_paths=[img_path],
                    video_name=video_name,
                    response_format=get_schema(RefinedIntentSchema.model_json_schema()),
                    system_prompt=REFINED_INTENT_SYSTEM,
                )

                parsed = json.loads(result)

                return {
                    "hidden_intent": parsed.get("hidden_intent", "unknown"),
                    "supporting_primitives": parsed.get("supporting_primitives", []),
                    "intent_confidence": parsed.get("intent_confidence", 5),
                    "alternative_intents": parsed.get("alternative_intents", []),
                    "refined_from": parsed.get("refined_from"),
                    "refinement_reason": parsed.get("refinement_reason"),
                    "validated_by_outcome": parsed.get("validated_by_outcome"),
                    "image_path": img_path,
                    "video_name": video_name,
                    "pass": self._current_pass,
                    "pass_type": "intent"
                }

            except Exception as exc:
                last_exc = exc
                if attempt < max_retries - 1:
                    self.logger.warning(f"Intent refinement attempt {attempt + 1} failed, retrying: {exc}")
                    continue
                if self.debug:
                    self.logger.error(f"Intent refinement failed after {max_retries} attempts: {exc}")
                    self.logger.error(traceback.format_exc())
        return {
            "hidden_intent": "refinement_failed",
            "intent_confidence": 0,
            "image_path": img_path,
            "video_name": video_name,
            "pass": self._current_pass,
            "pass_type": "intent",
            "error": str(last_exc)
        }

    # ─────────────────────────────── Summary generation

    # Token budget for RLE content in summary prompts.
    # Leaves room for prompt template (~100 tokens), prior summary (~500 tokens),
    # and output (max_tokens=500) within the 32K model context.
    _SUMMARY_RLE_TOKEN_BUDGET = 6000
    _tokenizer = None

    @classmethod
    def _get_tokenizer(cls):
        """Lazy-load the model tokenizer (cached as class-level singleton)."""
        if cls._tokenizer is None:
            from transformers import AutoTokenizer
            cls._tokenizer = AutoTokenizer.from_pretrained(
                "Qwen/Qwen3-VL-8B-Instruct", trust_remote_code=True,
            )
        return cls._tokenizer

    @classmethod
    def _count_tokens(cls, text: str) -> int:
        """Count tokens using the model's tokenizer."""
        return len(cls._get_tokenizer().encode(text))

    def _chunk_rle_lines(
        self,
        rle_lines: List[str],
        runs: list,
        budget_tokens: int,
    ) -> List[tuple]:
        """Split RLE lines into chunks that fit within a token budget.

        Uses the model's tokenizer for accurate token counting.
        Returns list of (chunk_text, start_frame, end_frame) tuples.
        """
        chunks = []
        current_lines = []
        current_tokens = 0
        chunk_start = runs[0].start_idx if runs else 0
        last_run_end = chunk_start

        for line, run in zip(rle_lines, runs):
            line_tokens = self._count_tokens(line)
            if current_tokens + line_tokens > budget_tokens and current_lines:
                chunks.append((
                    "\n".join(current_lines),
                    chunk_start,
                    last_run_end,
                ))
                current_lines = []
                current_tokens = 0
                chunk_start = run.start_idx
            current_lines.append(line)
            current_tokens += line_tokens
            last_run_end = run.end_idx

        if current_lines:
            chunks.append((
                "\n".join(current_lines),
                chunk_start,
                last_run_end,
            ))

        return chunks

    async def _generate_summary(
        self,
        states: List[Dict[str, Any]],
        video_name: Optional[str],
        pass_type: str,
        pass_num: int,
    ) -> str:
        """Generate a pass summary using map-reduce when RLE exceeds token budget.

        For small passes: single LLM call with full RLE (same as before).
        For large passes: chunk the RLE, summarize each chunk independently
        (map), then merge chunk summaries into a final summary (reduce).
        """
        from gum.prompts.state_prompt import (
            CHUNK_PRIMITIVE_SUMMARY_PROMPT,
            CHUNK_INTENT_SUMMARY_PROMPT,
            MERGE_SUMMARIES_PROMPT,
        )

        runs = _collapse_to_runs(states, 0, len(states), pass_type)
        rle_lines = [run.format_line() for run in runs]
        type_label = "activities" if pass_type == "activity" else "intents"
        rle_text = "\n".join(rle_lines) if rle_lines else f"No {type_label} extracted"

        prior_pass = pass_num - 1
        prior_summary = self._pass_summaries.get(prior_pass, "")

        rle_token_count = self._count_tokens(rle_text)
        needs_map_reduce = rle_token_count > self._SUMMARY_RLE_TOKEN_BUDGET

        if not needs_map_reduce:
            return await self._generate_summary_single(
                rle_text, prior_summary, pass_type, pass_num, video_name,
            )

        # Map phase: chunk the RLE and summarize each chunk
        chunks = self._chunk_rle_lines(rle_lines, runs, self._SUMMARY_RLE_TOKEN_BUDGET)
        if self.debug:
            self.logger.info(
                f"[Pass {pass_num}] Map-reduce summary: "
                f"{len(rle_lines)} RLE runs ({rle_token_count} tokens) -> {len(chunks)} chunks"
            )

        chunk_template = (
            CHUNK_PRIMITIVE_SUMMARY_PROMPT if pass_type == "activity"
            else CHUNK_INTENT_SUMMARY_PROMPT
        )

        async def summarize_chunk(idx: int, chunk_text: str, start: int, end: int) -> str:
            prompt = chunk_template.format(
                chunk_num=idx + 1,
                total_chunks=len(chunks),
                start_frame=start,
                end_frame=end,
                states=chunk_text,
                intents=chunk_text,
            )
            return await self._call_vision_api(
                prompt=prompt, img_paths=[], video_name=video_name, max_tokens=300,
            )

        chunk_summaries = await asyncio.gather(*[
            summarize_chunk(i, text, start, end)
            for i, (text, start, end) in enumerate(chunks)
        ])

        # Reduce phase: merge chunk summaries with prior context
        numbered = "\n\n".join(
            f"### Chunk {i+1} (frames {chunks[i][1]}-{chunks[i][2]}):\n{s}"
            for i, s in enumerate(chunk_summaries)
        )
        merge_prompt = MERGE_SUMMARIES_PROMPT.format(
            prior_summary=prior_summary or "No prior summary available.",
            chunk_summaries=numbered,
        )

        try:
            summary = await self._call_vision_api(
                prompt=merge_prompt, img_paths=[], video_name=video_name, max_tokens=500,
            )
            return summary
        except Exception as exc:
            if self.debug:
                self.logger.error(f"Summary merge failed: {exc}")
                self.logger.error(traceback.format_exc())
            return f"Summary generation failed: {exc}"

    async def _generate_summary_single(
        self,
        rle_text: str,
        prior_summary: str,
        pass_type: str,
        pass_num: int,
        video_name: Optional[str],
    ) -> str:
        """Generate a summary from a single RLE sequence that fits in one call."""
        if pass_type == "activity":
            context = f"## Pass {pass_num} Activity States (RLE sequence):\n{rle_text}"
            if prior_summary:
                context = f"## Prior Pass Summary:\n{prior_summary}\n\n{context}"
            prompt = PRIMITIVE_SUMMARY_PROMPT.format(states=context)
        else:
            activity_passes = [p for p in self._pass_summaries.keys() if p % 2 == 1]
            latest_activity_pass = max(activity_passes) if activity_passes else None
            activity_summary = self._pass_summaries.get(
                latest_activity_pass, "No activity summary"
            ) if latest_activity_pass else "No activity summary"

            context = f"## Pass {pass_num} Hidden Intents (RLE sequence):\n{rle_text}"
            if prior_summary:
                context = f"## Prior Pass Summary:\n{prior_summary}\n\n{context}"
            prompt = INTENT_SUMMARY_PROMPT.format(
                primitive_summary=activity_summary,
                intents=context,
            )

        try:
            summary = await self._call_vision_api(
                prompt=prompt, img_paths=[], video_name=video_name, max_tokens=500,
            )
            return summary
        except Exception as exc:
            if self.debug:
                self.logger.error(f"Summary generation failed: {exc}")
                self.logger.error(traceback.format_exc())
            return f"Summary generation failed: {exc}"

    # ─────────────────────────────── Main processing

    async def _process_single_pass(
        self,
        images: List[tuple[Path, Optional[str]]],
        pass_num: int
    ) -> List[Dict[str, Any]]:
        """Process all images for a single pass."""
        self._current_pass = pass_num
        pass_type = get_pass_type(pass_num)
        states = []
        total = len(images)

        for idx, (img_path, video_name) in enumerate(images):
            if not self._running:
                self.logger.info(f"Pass {pass_num} stopped early.")
                break

            video_tag = f"[{video_name}] " if video_name else ""
            self.logger.info(
                f"Pass {pass_num} ({pass_type}) [{idx + 1}/{total}]: {video_tag}{img_path.name}"
            )

            # Reset history if video changed
            if video_name != self._current_video:
                self._history.clear()
                self._current_video = video_name

            self._history.append(str(img_path))

            # Route to appropriate extraction method based on pass
            if pass_num == 1:
                # Pass 1: Raw activity extraction
                state_data = await self._extract_activity_pass1(str(img_path), video_name)
            elif pass_num == 2:
                # Pass 2: Initial intent inference
                state_data = await self._infer_intent_pass2(
                    str(img_path), idx, total, video_name
                )
            elif pass_type == "activity":
                # Pass 3, 5, ...: Refined activity
                state_data = await self._refine_activity(
                    str(img_path), idx, total, video_name
                )
            else:
                # Pass 4, 6, ...: Refined intent
                state_data = await self._refine_intent(
                    str(img_path), idx, total, video_name
                )

            states.append(state_data)

            # Save immediately per-frame
            self._save_single_state(video_name, state_data)

            # Emit update to gum pipeline
            # Activity passes (odd passes) generate propositions
            if pass_type == "activity":
                activity = state_data.get("primitive_state", "unknown")
                conf = state_data.get("confidence", 5)
                content = f"[{video_name or 'video'}] Activity: {activity} (confidence: {conf})"

                # Include intent context if available from prior pass
                if pass_num > 2:
                    prior_intent_pass = pass_num - 1
                    if prior_intent_pass in self._pass_states and idx < len(self._pass_states[prior_intent_pass]):
                        prior_intent = self._pass_states[prior_intent_pass][idx].get("hidden_intent", "")
                        if prior_intent:
                            content += f" | Intent: {prior_intent}"

                await self.update_queue.put(
                    Update(content=content, content_type="input_text")
                )

            if self.process_delay > 0:
                await asyncio.sleep(self.process_delay)

        return states

    async def _run_multi_pass(self, images: List[tuple[Path, Optional[str]]]) -> None:
        """Run multi-pass analysis on all images."""
        # Group images by video
        videos: Dict[Optional[str], List[tuple[Path, Optional[str]]]] = {}
        for img_path, video_name in images:
            if video_name not in videos:
                videos[video_name] = []
            videos[video_name].append((img_path, video_name))

        # Process each video
        for video_name, video_images in videos.items():
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Processing video: {video_name or 'root'} ({len(video_images)} images)")
            self.logger.info(f"{'='*60}")

            # Clear pass state for this video
            self._pass_states.clear()
            self._pass_summaries.clear()

            # Run all passes for this video
            for pass_num in range(1, self.num_passes + 1):
                pass_type = get_pass_type(pass_num)
                context_passes = get_context_passes(pass_num, self.pass_window_size)

                self.logger.info(f"\n--- Pass {pass_num}/{self.num_passes} ({pass_type.upper()}) ---")
                if context_passes:
                    self.logger.info(f"    Context from passes: {context_passes}")

                states = await self._process_single_pass(video_images, pass_num)
                self._pass_states[pass_num] = states

                # States already saved per-frame in _process_single_pass

                # Generate and save summary
                summary = await self._generate_summary(states, video_name, pass_type, pass_num)
                self._pass_summaries[pass_num] = summary
                self._save_pass_output(video_name, "summary", {"summary": summary})

                # Log pass statistics
                if pass_type == "activity":
                    unique_states = set(s.get("primitive_state", "unknown") for s in states)
                    refined_count = sum(1 for s in states if s.get("refined_from"))
                    self.logger.info(
                        f"Pass {pass_num} complete: {len(states)} frames, "
                        f"{len(unique_states)} unique activities"
                    )
                    if pass_num > 1:
                        self.logger.info(f"  Refined: {refined_count} states")
                else:
                    unique_intents = set(s.get("hidden_intent", "unknown") for s in states)
                    high_conf = sum(1 for s in states if s.get("intent_confidence", 0) >= 7)
                    self.logger.info(
                        f"Pass {pass_num} complete: {len(states)} frames, "
                        f"{len(unique_intents)} unique intents, {high_conf} high-confidence"
                    )

            # Save provenance files for auditability
            self._save_provenance(video_name)

            # Compute and save metrics after all passes
            self._save_metrics_and_viz(video_name)

    # ─────────────────────────────── Metrics and visualization

    def _save_metrics_and_viz(self, video_name: Optional[str]) -> None:
        """Compute and save metrics and visualization data after all passes."""
        video_folder = video_name or ""
        output_folder = self.output_dir / video_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        # Compute comprehensive metrics
        self.logger.info("Computing metrics...")
        metrics = compute_all_metrics(self._pass_states, self._pass_summaries)
        metrics_path = output_folder / "metrics.json"
        save_metrics_json(metrics, str(metrics_path))
        self.logger.info(f"Saved metrics to {metrics_path}")

        # Export visualization data
        self.logger.info("Exporting visualization data...")
        viz_data = export_for_visualization(self._pass_states, self._pass_summaries)
        viz_path = output_folder / "visualization.json"
        save_metrics_json(viz_data, str(viz_path))
        self.logger.info(f"Saved visualization data to {viz_path}")

        # Log key metrics
        if "convergence" in metrics and "avg_activity_stability" in metrics["convergence"]:
            self.logger.info(
                f"  Activity stability: {metrics['convergence']['avg_activity_stability']}"
            )
        if "convergence" in metrics and "avg_intent_stability" in metrics["convergence"]:
            self.logger.info(
                f"  Intent stability: {metrics['convergence']['avg_intent_stability']}"
            )
        if "alignment" in metrics and "avg_activity_consistency" in metrics["alignment"]:
            self.logger.info(
                f"  Activity-intent consistency: {metrics['alignment']['avg_activity_consistency']}"
            )

    # ─────────────────────────────── Main worker

    async def _worker(self) -> None:
        """Main worker that processes all images with multi-pass analysis."""
        logger = logging.getLogger("Retro")

        if not logger.handlers:
            fmt = logging.Formatter("%(asctime)s - [RETRO] - %(message)s")
            h = logging.StreamHandler()
            h.setFormatter(fmt)
            logger.addHandler(h)

            # Persist logs to mounted volume so they survive container removal
            log_dir = os.environ.get("GUM_TRAFFIC_LOG_DIR", "")
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)
                fh = logging.FileHandler(os.path.join(log_dir, "retro.log"))
                fh.setFormatter(fmt)
                logger.addHandler(fh)

        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Get sorted list of images
        images = self._get_sorted_images_recursive()
        total = len(images)

        if total == 0:
            logger.warning(f"No images found in {self.images_dir}")
            self.stopped.set()
            return

        # Count videos
        video_names = set(v for _, v in images if v is not None)
        if video_names:
            logger.info(f"Found {total} images across {len(video_names)} videos")
        else:
            logger.info(f"Found {total} images to process")

        logger.info(f"Running {self.num_passes} pass(es) (alternating activity/intent)")
        logger.info(f"  - Odd passes (1,3,5): Activity state extraction")
        logger.info(f"  - Even passes (2,4,6): Hidden intent inference")
        logger.info(f"  - Context window: {self.context_window_size} frames (±{self.context_window_size // 2})")
        logger.info(f"  - Pass window: {self.pass_window_size} prior passes")

        # Run multi-pass analysis
        try:
            await self._run_multi_pass(images)
            logger.info(
                f"Retro observer completed. Processed {total} images with "
                f"{self.num_passes} passes (alternating activity/intent)."
            )
        except Exception as exc:
            logger.error(f"Multi-pass analysis failed: {exc}")
            logger.error(traceback.format_exc())
            raise
        finally:
            self.stopped.set()

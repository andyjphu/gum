# Author: Andy Phu
# Retroactive observer with multi-pass analysis for egocentric footage
#
# Pass structure (alternating primitive/intent):
#   Pass 1: Primitive states (observable actions)
#   Pass 2: Hidden intents (informed by Pass 1)
#   Pass 3: Refined primitives (informed by Pass 1, 2)
#   Pass 4: Refined intents (informed by Pass 1, 2, 3)
#   Pass 5: Refined primitives (informed by Pass 2, 3, 4)  <- sliding window
#   Pass 6: Refined intents (informed by Pass 3, 4, 5)

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
    PASS_OUTPUT_DIR,
    TRAFFIC_LOG_DIR,
)
from gum.prompts.state_prompt import (
    PRIMITIVE_STATE_PROMPT,
    HIDDEN_INTENT_PROMPT,
    REFINED_PRIMITIVE_PROMPT,
    REFINED_INTENT_PROMPT,
    PRIMITIVE_SUMMARY_PROMPT,
    INTENT_SUMMARY_PROMPT,
    COMBINED_SUMMARY_PROMPT,
    get_pass_type,
    get_context_passes,
    build_temporal_context,
    build_multi_pass_context,
    build_context_window,
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
# Retro observer with multi-pass (alternating primitive/intent)               #
###############################################################################


class Retro(Observer):
    """Observer that processes screenshots with multi-pass analysis.

    Implements an alternating primitive/intent architecture:
    - Odd passes (1, 3, 5): Extract/refine primitive (observable) states
    - Even passes (2, 4, 6): Infer/refine hidden intents

    Context window logic:
    - Pass 1: No context (raw observation)
    - Pass 2: Context from Pass 1
    - Pass 3: Context from Pass 1, 2
    - Pass 4: Context from Pass 1, 2, 3
    - Pass 5+: Sliding window of previous 3 passes

    Args:
        images_dir: Directory containing screenshots to process.
        model_name: GPT model to use for vision analysis.
        num_passes: Number of analysis passes (default: 6).
        context_window_size: Max unique states in context for re-analysis.
        temporal_window_size: Number of temporally nearest frames for context.
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
        self.temporal_window_size = temporal_window_size

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

        # OpenAI client
        self.client = AsyncOpenAI(
            base_url=api_base or os.getenv("VLLM_ENDPOINT"),
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "no-key",
        )
        self.logger = logging.getLogger("Retro")

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

        Layout: data/passes/{video}/{timestamp}/states_{pass_type}_P{N}.json
        """
        suffix = self._get_pass_suffix()
        pass_type = self._get_pass_type_str()
        video_folder = video_name or "root"
        output_folder = self.output_dir / video_folder

        img_path = state_data.get("image_path", "")
        m = re.search(r"(\d{8}_\d{6})", img_path)
        ts_key = m.group(1) if m else str(hash(img_path))

        ts_dir = output_folder / ts_key
        ts_dir.mkdir(parents=True, exist_ok=True)

        filename = f"states_{pass_type}{suffix}.json"
        ts_file = ts_dir / filename
        with open(ts_file, "w") as f:
            json.dump(state_data, f, indent=2, default=str)

    def _save_pass_output(self, video_name: Optional[str], data_type: str, data: Any) -> Path:
        """Save non-state pass output (summaries, etc).

        Layout: data/passes/{video}/{data_type}/{data_type}_{pass_type}_P{N}.json
        """
        suffix = self._get_pass_suffix()
        pass_type = self._get_pass_type_str()
        video_folder = video_name or "root"
        output_folder = self.output_dir / video_folder

        sub_dir = output_folder / data_type
        sub_dir.mkdir(parents=True, exist_ok=True)

        filename = f"{data_type}_{pass_type}{suffix}.json"
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
        video_folder = video_name or "root"
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
            # Determine final state (highest primitive pass)
            primitive_passes = [p for p in passes_data.keys() if passes_data[p]["type"] == "primitive"]
            final_pass = max(primitive_passes) if primitive_passes else max(passes_data.keys())
            final_state = passes_data[final_pass]["state"]

            # Determine transition from previous frame
            transition = None
            if i > 0:
                prev_ts = sorted_timestamps[i - 1]
                prev_passes = timestamps.get(prev_ts, {})
                prev_primitive_passes = [p for p in prev_passes.keys() if prev_passes[p]["type"] == "primitive"]
                if prev_primitive_passes:
                    prev_final_pass = max(prev_primitive_passes)
                    prev_state = prev_passes[prev_final_pass]["state"]
                    if prev_state != final_state:
                        # Get intent from current frame's highest intent pass as trigger
                        intent_passes = [p for p in passes_data.keys() if passes_data[p]["type"] == "intent"]
                        trigger = passes_data[max(intent_passes)]["state"] if intent_passes else f"to_{final_state}"
                        # Compute relative timestamp (5 seconds per frame)
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
    ) -> str:
        """Call GPT Vision API to analyze images."""
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

        rsp = await invoke(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            response_format=response_format or {"type": "text"},
            debug_tag=debug_tag,
            debug_img_paths=img_paths,
            debug_path=video_name,
            client=self.client,
        )

        return rsp.choices[0].message.content

    # ─────────────────────────────── Pass 1: Primitive State Extraction

    async def _extract_primitive_pass1(
        self,
        img_path: str,
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract primitive (observable) state from a single image (Pass 1)."""
        prompt = PRIMITIVE_STATE_PROMPT.format(additional_fields="")

        try:
            result = await self._call_vision_api(
                prompt=prompt,
                img_paths=[img_path],
                video_name=video_name,
                response_format=get_schema(PrimitiveStateSchema.model_json_schema()),
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
                "pass_type": "primitive"
            }

        except Exception as exc:
            if self.debug:
                self.logger.error(f"Primitive extraction failed: {exc}")
                self.logger.error(traceback.format_exc())
            return {
                "primitive_state": "extraction_failed",
                "confidence": 0,
                "image_path": img_path,
                "video_name": video_name,
                "pass": self._current_pass,
                "pass_type": "primitive",
                "error": str(exc)
            }

    # ─────────────────────────────── Pass 2: Hidden Intent Inference

    async def _infer_intent_pass2(
        self,
        img_path: str,
        frame_idx: int,
        total_frames: int,
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Infer hidden intent from primitive states (Pass 2)."""
        # Get context from Pass 1
        context_passes = get_context_passes(self._current_pass)
        prior_summary = self._pass_summaries.get(1, "No prior summary available.")

        temporal_context = build_temporal_context(
            self._pass_states,
            frame_idx,
            total_frames,
            context_passes,
            self.temporal_window_size
        )

        prompt = HIDDEN_INTENT_PROMPT.format(
            prior_summary=prior_summary,
            temporal_context=temporal_context,
            additional_fields=""
        )

        try:
            result = await self._call_vision_api(
                prompt=prompt,
                img_paths=[img_path],
                video_name=video_name,
                response_format=get_schema(HiddenIntentSchema.model_json_schema()),
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
            if self.debug:
                self.logger.error(f"Intent inference failed: {exc}")
                self.logger.error(traceback.format_exc())
            return {
                "hidden_intent": "inference_failed",
                "intent_confidence": 0,
                "image_path": img_path,
                "video_name": video_name,
                "pass": self._current_pass,
                "pass_type": "intent",
                "error": str(exc)
            }

    # ─────────────────────────────── Pass 3+: Refined Primitive State

    async def _refine_primitive(
        self,
        img_path: str,
        frame_idx: int,
        total_frames: int,
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Refine primitive state with context from prior passes (Pass 3, 5, ...)."""
        context_passes = get_context_passes(self._current_pass)

        prior_context = build_multi_pass_context(
            self._pass_states,
            self._pass_summaries,
            context_passes,
            self.context_window_size
        )

        prompt = REFINED_PRIMITIVE_PROMPT.format(
            prior_context=prior_context,
            additional_fields=""
        )

        try:
            result = await self._call_vision_api(
                prompt=prompt,
                img_paths=[img_path],
                video_name=video_name,
                response_format=get_schema(RefinedPrimitiveSchema.model_json_schema()),
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
                "pass_type": "primitive"
            }

        except Exception as exc:
            if self.debug:
                self.logger.error(f"Primitive refinement failed: {exc}")
                self.logger.error(traceback.format_exc())
            return {
                "primitive_state": "refinement_failed",
                "confidence": 0,
                "image_path": img_path,
                "video_name": video_name,
                "pass": self._current_pass,
                "pass_type": "primitive",
                "error": str(exc)
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
        context_passes = get_context_passes(self._current_pass)

        prior_context = build_multi_pass_context(
            self._pass_states,
            self._pass_summaries,
            context_passes,
            self.context_window_size
        )

        temporal_context = build_temporal_context(
            self._pass_states,
            frame_idx,
            total_frames,
            context_passes,
            self.temporal_window_size
        )

        prompt = REFINED_INTENT_PROMPT.format(
            context_passes=", ".join(str(p) for p in context_passes),
            prior_context=prior_context,
            temporal_context=temporal_context,
            additional_fields=""
        )

        try:
            result = await self._call_vision_api(
                prompt=prompt,
                img_paths=[img_path],
                video_name=video_name,
                response_format=get_schema(RefinedIntentSchema.model_json_schema()),
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
            if self.debug:
                self.logger.error(f"Intent refinement failed: {exc}")
                self.logger.error(traceback.format_exc())
            return {
                "hidden_intent": "refinement_failed",
                "intent_confidence": 0,
                "image_path": img_path,
                "video_name": video_name,
                "pass": self._current_pass,
                "pass_type": "intent",
                "error": str(exc)
            }

    # ─────────────────────────────── Summary generation

    async def _generate_summary(
        self,
        states: List[Dict[str, Any]],
        video_name: Optional[str],
        pass_type: str
    ) -> str:
        """Generate summary based on pass type."""
        if pass_type == "primitive":
            # Format primitive states
            state_lines = []
            for s in states:
                prim = s.get("primitive_state", "unknown")
                conf = s.get("confidence", "?")
                body = s.get("body_position", "")
                objs = s.get("objects_interacted", [])
                line = f"- {prim} (confidence: {conf})"
                if body:
                    line += f" [position: {body}]"
                if objs:
                    line += f" [objects: {', '.join(objs)}]"
                state_lines.append(line)

            states_text = "\n".join(state_lines) if state_lines else "No primitives extracted"
            prompt = PRIMITIVE_SUMMARY_PROMPT.format(states=states_text)

        else:  # intent
            # Get primitive summary from most recent primitive pass
            primitive_passes = [p for p in self._pass_summaries.keys() if p % 2 == 1]
            latest_primitive_pass = max(primitive_passes) if primitive_passes else None
            primitive_summary = self._pass_summaries.get(latest_primitive_pass, "No primitive summary") if latest_primitive_pass else "No primitive summary"

            # Format intents
            intent_lines = []
            for s in states:
                intent = s.get("hidden_intent", "unknown")
                conf = s.get("intent_confidence", "?")
                supporting = s.get("supporting_primitives", [])
                line = f"- {intent} (confidence: {conf})"
                if supporting:
                    line += f" [supporting: {', '.join(supporting[:3])}...]"
                intent_lines.append(line)

            intents_text = "\n".join(intent_lines) if intent_lines else "No intents inferred"
            prompt = INTENT_SUMMARY_PROMPT.format(
                primitive_summary=primitive_summary,
                intents=intents_text
            )

        try:
            summary = await self._call_vision_api(
                prompt=prompt,
                img_paths=[],  # Text-only call
                video_name=video_name,
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
                # Pass 1: Raw primitive extraction
                state_data = await self._extract_primitive_pass1(str(img_path), video_name)
            elif pass_num == 2:
                # Pass 2: Initial intent inference
                state_data = await self._infer_intent_pass2(
                    str(img_path), idx, total, video_name
                )
            elif pass_type == "primitive":
                # Pass 3, 5, ...: Refined primitive
                state_data = await self._refine_primitive(
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
            # Pass 1 primitives and refined primitives generate propositions
            if pass_type == "primitive":
                prim = state_data.get("primitive_state", "unknown")
                conf = state_data.get("confidence", 5)
                content = f"[{video_name or 'video'}] Primitive: {prim} (confidence: {conf})"

                # Include intent context if available from prior pass
                if pass_num > 2:
                    # Get intent from previous pass for this frame
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
                context_passes = get_context_passes(pass_num)

                self.logger.info(f"\n--- Pass {pass_num}/{self.num_passes} ({pass_type.upper()}) ---")
                if context_passes:
                    self.logger.info(f"    Context from passes: {context_passes}")

                states = await self._process_single_pass(video_images, pass_num)
                self._pass_states[pass_num] = states

                # States already saved per-frame in _process_single_pass

                # Generate and save summary
                summary = await self._generate_summary(states, video_name, pass_type)
                self._pass_summaries[pass_num] = summary
                self._save_pass_output(video_name, "summary", {"summary": summary})

                # Log pass statistics
                if pass_type == "primitive":
                    unique_states = set(s.get("primitive_state", "unknown") for s in states)
                    refined_count = sum(1 for s in states if s.get("refined_from"))
                    self.logger.info(
                        f"Pass {pass_num} complete: {len(states)} frames, "
                        f"{len(unique_states)} unique primitives"
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
        video_folder = video_name or "root"
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
        if "convergence" in metrics and "avg_primitive_stability" in metrics["convergence"]:
            self.logger.info(
                f"  Primitive stability: {metrics['convergence']['avg_primitive_stability']}"
            )
        if "convergence" in metrics and "avg_intent_stability" in metrics["convergence"]:
            self.logger.info(
                f"  Intent stability: {metrics['convergence']['avg_intent_stability']}"
            )
        if "alignment" in metrics and "avg_primitive_consistency" in metrics["alignment"]:
            self.logger.info(
                f"  Primitive-intent consistency: {metrics['alignment']['avg_primitive_consistency']}"
            )

    # ─────────────────────────────── Main worker

    async def _worker(self) -> None:
        """Main worker that processes all images with multi-pass analysis."""
        logger = logging.getLogger("Retro")

        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s - [RETRO] - %(message)s")
            )
            logger.addHandler(h)

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

        logger.info(f"Running {self.num_passes} pass(es) (alternating primitive/intent)")
        logger.info(f"  - Odd passes (1,3,5): Primitive state extraction")
        logger.info(f"  - Even passes (2,4,6): Hidden intent inference")
        logger.info(f"  - Temporal window: {self.temporal_window_size} frames")
        logger.info(f"  - Context window: {self.context_window_size} unique states")

        # Run multi-pass analysis
        await self._run_multi_pass(images)

        logger.info(
            f"Retro observer completed. Processed {total} images with "
            f"{self.num_passes} passes (alternating primitive/intent)."
        )
        self.stopped.set()

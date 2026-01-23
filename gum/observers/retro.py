# Author: Andy Phu
# Retroactive observer with multi-pass analysis for egocentric footage
#
# Pass 1: Extract states from images → Generate propositions → Summary
# Pass 2-K: Re-analyze with context → Collapse states → Refined summary
#

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import traceback
from collections import deque
from pathlib import Path
from typing import Optional, List, Dict, Any

from openai import AsyncOpenAI

from .observer import Observer
from ..schemas import Update, StateExtractionSchema, RefinedStateSchema, get_schema
from ..invoke import invoke
from ..config import (
    DEFAULT_NUM_PASSES,
    DEFAULT_CONTEXT_WINDOW_SIZE,
    PASS_OUTPUT_DIR,
)
from gum.prompts.state_prompt import (
    STATE_EXTRACTION_PROMPT,
    RETRO_SUMMARY_PROMPT,
    REANALYZE_PROMPT,
    build_context_window
)

###############################################################################
# Constants                                                                   #
###############################################################################

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_HISTORY_K = 10
SHORT_SLEEP_SEC = 0.05
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


###############################################################################
# Retro observer with multi-pass                                              #
###############################################################################


class Retro(Observer):
    """Observer that processes screenshots with multi-pass analysis.

    Implements a farsight-style multi-pass architecture:
    - Pass 1: Extract user states from egocentric images
    - Pass 2-K: Re-analyze with prior context, collapse similar states

    Args:
        images_dir: Directory containing screenshots to process.
        model_name: GPT model to use for vision analysis.
        num_passes: Number of analysis passes (default: 2).
        context_window_size: Max unique states in context for re-analysis.
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

        # Model config
        self.model_name = model_name
        self.debug = debug
        self.process_delay = process_delay
        self.history_k = history_k

        # History (per video/subfolder)
        self._history: deque[str] = deque(maxlen=max(0, history_k))
        self._current_video: Optional[str] = None

        # Pass state storage
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

    def _save_pass_output(self, video_name: Optional[str], data_type: str, data: Any) -> Path:
        """Save pass output to file with pass suffix."""
        suffix = self._get_pass_suffix()
        video_folder = video_name or "root"
        output_folder = self.output_dir / video_folder
        output_folder.mkdir(parents=True, exist_ok=True)

        filename = f"{data_type}{suffix}.json"
        filepath = output_folder / filename

        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        if self.debug:
            self.logger.info(f"Saved {data_type} to {filepath}")

        return filepath

    # ─────────────────────────────── Vision API calls

    async def _call_vision_api(
        self,
        prompt: str,
        img_paths: list[str],
        video_name: Optional[str] = None,
        response_format: Optional[dict] = None,
        debug_tag_suffix: str = ""
    ) -> str:
        """Call GPT Vision API to analyze images."""
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

        # Build debug tag with pass suffix
        pass_suffix = self._get_pass_suffix()
        debug_tag = f"[Retro{pass_suffix}]{debug_tag_suffix}"

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

    # ─────────────────────────────── Pass 1: State Extraction

    async def _extract_state_pass1(
        self,
        img_path: str,
        video_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Extract user state from a single image (Pass 1)."""
        prompt = STATE_EXTRACTION_PROMPT.format(additional_fields="")

        try:
            result = await self._call_vision_api(
                prompt=prompt,
                img_paths=[img_path],
                video_name=video_name,
                response_format=get_schema(StateExtractionSchema.model_json_schema()),
                debug_tag_suffix=" State"
            )

            # Parse JSON response - now expects {state, confidence} directly
            parsed = json.loads(result)

            return {
                "state": parsed.get("state", "unknown"),
                "confidence": parsed.get("confidence", 5),
                "image_path": img_path,
                "video_name": video_name,
                "pass": self._current_pass
            }

        except Exception as exc:
            if self.debug:
                self.logger.error(f"State extraction failed: {exc}")
                self.logger.error(traceback.format_exc())
            return {
                "state": "extraction_failed",
                "confidence": 0,
                "image_path": img_path,
                "video_name": video_name,
                "pass": self._current_pass,
                "error": str(exc)
            }

    # ─────────────────────────────── Pass 2+: Re-analysis with context

    async def _reanalyze_state(
        self,
        img_path: str,
        video_name: Optional[str],
        prior_context: str
    ) -> Dict[str, Any]:
        """Re-analyze image with context from prior pass (Pass 2+)."""
        prompt = REANALYZE_PROMPT.format(
            prior_summary=self._pass_summaries.get(self._current_pass - 1, "No prior summary"),
            prior_states_context=prior_context,
            additional_fields=""
        )

        try:
            result = await self._call_vision_api(
                prompt=prompt,
                img_paths=[img_path],
                video_name=video_name,
                response_format=get_schema(RefinedStateSchema.model_json_schema()),
                debug_tag_suffix=" Reanalyze"
            )

            parsed = json.loads(result)
            states = parsed.get("states", [])

            if states:
                state_data = states[0]
                return {
                    "state": state_data.get("state", "unknown"),
                    "confidence": state_data.get("confidence", 5),
                    "collapsed_from": state_data.get("collapsed_from"),
                    "collapse_reason": state_data.get("collapse_reason"),
                    "image_path": img_path,
                    "video_name": video_name,
                    "pass": self._current_pass
                }
            else:
                return {
                    "state": "unclear",
                    "confidence": 1,
                    "image_path": img_path,
                    "video_name": video_name,
                    "pass": self._current_pass
                }

        except Exception as exc:
            if self.debug:
                self.logger.error(f"Re-analysis failed: {exc}")
                self.logger.error(traceback.format_exc())
            return {
                "state": "reanalysis_failed",
                "confidence": 0,
                "image_path": img_path,
                "video_name": video_name,
                "pass": self._current_pass,
                "error": str(exc)
            }

    # ─────────────────────────────── Summary generation

    async def _generate_summary(
        self,
        states: List[Dict[str, Any]],
        video_name: Optional[str],
        propositions: str = ""
    ) -> str:
        """Generate summary of states and propositions."""
        # Format states for prompt
        state_lines = []
        for s in states:
            state_lines.append(f"- {s['state']} (confidence: {s.get('confidence', '?')})")
        states_text = "\n".join(state_lines) if state_lines else "No states extracted"

        prompt = RETRO_SUMMARY_PROMPT.format(
            states=states_text,
            propositions=propositions or "No propositions yet"
        )

        try:
            summary = await self._call_vision_api(
                prompt=prompt,
                img_paths=[],  # Text-only call, no images needed for summary
                video_name=video_name,
                debug_tag_suffix=" Summary"
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
        states = []
        total = len(images)

        # Build context from prior pass if not pass 1
        prior_context = ""
        if pass_num > 1 and (pass_num - 1) in self._pass_states:
            prior_states = self._pass_states[pass_num - 1]
            prior_summaries = [self._pass_summaries.get(i, "") for i in range(1, pass_num)]
            prior_context = build_context_window(
                prior_states,
                prior_summaries,
                self.context_window_size
            )

        for idx, (img_path, video_name) in enumerate(images):
            if not self._running:
                self.logger.info(f"Pass {pass_num} stopped early.")
                break

            video_tag = f"[{video_name}] " if video_name else ""
            self.logger.info(f"Pass {pass_num} [{idx + 1}/{total}]: {video_tag}{img_path.name}")

            # Reset history if video changed
            if video_name != self._current_video:
                self._history.clear()
                self._current_video = video_name

            self._history.append(str(img_path))

            # Extract or re-analyze state based on pass number
            if pass_num == 1:
                state_data = await self._extract_state_pass1(str(img_path), video_name)
            else:
                state_data = await self._reanalyze_state(str(img_path), video_name, prior_context)

            states.append(state_data)

            # Emit update to gum pipeline (Pass 1 only generates propositions)
            if pass_num == 1:
                # Format state as content for proposition generation
                content = f"[{video_name or 'video'}] User state: {state_data['state']} (confidence: {state_data['confidence']})"
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

            # Run all passes for this video
            for pass_num in range(1, self.num_passes + 1):
                self.logger.info(f"\n--- Pass {pass_num}/{self.num_passes} ---")

                states = await self._process_single_pass(video_images, pass_num)
                self._pass_states[pass_num] = states

                # Save states
                self._save_pass_output(video_name, "states", states)

                # Generate and save summary
                summary = await self._generate_summary(states, video_name)
                self._pass_summaries[pass_num] = summary
                self._save_pass_output(video_name, "summary", {"summary": summary})

                # Log pass statistics
                unique_states = set(s['state'] for s in states)
                collapsed_count = sum(1 for s in states if s.get('collapsed_from'))

                self.logger.info(f"Pass {pass_num} complete: {len(states)} frames, {len(unique_states)} unique states")
                if pass_num > 1:
                    self.logger.info(f"  Collapsed: {collapsed_count} states")

            # Clear pass state for next video
            self._pass_states.clear()
            self._pass_summaries.clear()

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

        logger.info(f"Running {self.num_passes} pass(es) with context window size {self.context_window_size}")

        # Run multi-pass analysis
        await self._run_multi_pass(images)

        logger.info(f"Retro observer completed. Processed {total} images with {self.num_passes} passes.")
        self.stopped.set()

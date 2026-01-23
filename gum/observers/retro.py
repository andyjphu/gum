# Author: Andy Phu   
# Retroactive observer, performs normal observation steps on pre-recorded image set 
# (rather than capturing live)
#

from __future__ import annotations

import asyncio
import base64
import logging
import os
from collections import deque
from pathlib import Path
from typing import Optional

from openai import AsyncOpenAI

from .observer import Observer
from ..schemas import Update
from ..invoke import invoke
from gum.prompts.screen import TRANSCRIPTION_PROMPT, SUMMARY_PROMPT

###############################################################################
# Constants                                                                   #
###############################################################################

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_HISTORY_K = 10
SHORT_SLEEP_SEC = 0.05  # Small delay between processing to avoid overwhelming
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


###############################################################################
# Retro observer                                                              #
###############################################################################


class Retro(Observer):
    """Observer that processes an existing directory of screenshots retroactively.

    Unlike Manual which captures live, this observer reads from a pre-existing
    image directory and processes them in chronological order. Supports both
    flat directories and nested subdirectories (organized by video name).

    Args:
        images_dir: Directory containing screenshots to process (supports subdirs).
        model_name: GPT model to use for vision analysis.
        transcription_prompt: Custom prompt for transcribing screenshots.
        summary_prompt: Custom prompt for summarizing screenshots.
        history_k: Number of recent screenshots to keep in history for summary.
        process_delay: Delay between processing each image (seconds).
        debug: Enable debug logging.
        api_key: OpenAI API key (or set OPENAI_API_KEY env var).
        api_base: OpenAI API base URL (or set VLLM_ENDPOINT env var).
    """

    def __init__(
        self,
        images_dir: str,
        model_name: str = DEFAULT_MODEL,
        transcription_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        history_k: int = DEFAULT_HISTORY_K,
        process_delay: float = SHORT_SLEEP_SEC,
        debug: bool = False,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """Initialize the Retro observer."""
        super().__init__()

        # Validate and setup image directory
        self.images_dir = Path(images_dir).expanduser().resolve()
        self.images_dir.mkdir(parents=True, exist_ok=True)
        if not self.images_dir.is_dir():
            raise ValueError(f"Path is not a directory: {self.images_dir}")

        # Prompts and model config
        self.transcription_prompt = transcription_prompt or TRANSCRIPTION_PROMPT
        self.summary_prompt = summary_prompt or SUMMARY_PROMPT
        self.model_name = model_name
        self.debug = debug
        self.process_delay = process_delay

        # History (per video/subfolder)
        self._history: deque[str] = deque(maxlen=max(0, history_k))
        self._current_video: Optional[str] = None

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
        """Extract video/subfolder name from image path.
        
        Returns:
            Subfolder name if image is in a subdirectory, None if in root.
        """
        relative = img_path.relative_to(self.images_dir)
        if len(relative.parts) > 1:
            return relative.parts[0]
        return None

    def _get_sorted_images_recursive(self) -> list[tuple[Path, Optional[str]]]:
        """Get all images recursively, sorted by video then filename.

        Returns:
            List of (image_path, video_name) tuples sorted by video then chronologically.
        """
        images = []
        for p in self.images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                video_name = self._get_video_name(p)
                images.append((p, video_name))
        
        # Sort by video name (None last), then by filename
        return sorted(images, key=lambda x: (x[1] or "", x[0].name))

    @staticmethod
    def _encode_image(img_path: str) -> str:
        """Encode an image file as base64."""
        with open(img_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    async def _call_gpt_vision(
        self, prompt: str, img_paths: list[str], video_name: Optional[str] = None
    ) -> str:
        """Call GPT Vision API to analyze images."""
        # Encode images concurrently
        encoded_images = await asyncio.gather(
            *[asyncio.to_thread(self._encode_image, p) for p in img_paths]
        )

        # Build content for API call
        content = [
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encoded}"},
            }
            for encoded in encoded_images
        ]
        content.append({"type": "text", "text": prompt})

        if self.debug:
            self.logger.info(f"Sending {len(img_paths)} image(s) to vision API")

        # Determine debug tag with video name
        video_tag = f"[{video_name}] " if video_name else ""
        if prompt == self.summary_prompt:
            debug_tag = f"[Retro Summary]"
        else:
            debug_tag = f"[Retro Transcription]"

        rsp = await invoke(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "text"},
            debug_tag=debug_tag,
            debug_img_paths=img_paths,
            debug_path=video_name,
            client=self.client,
        )

        result = rsp.choices[0].message.content

        if self.debug:
            self.logger.info(
                f"\n\n{'▇'*60}\nFor: \n{img_paths} \n\n {'▇ '*30} \n\n "
                f"Received:\n {result[:100]} \n{'▇'*60}\n"
            )

        return result

    async def _process_and_emit(self, path: str, video_name: Optional[str] = None) -> None:
        """Process a screenshot and emit an update."""
        # Reset history if we switched to a new video
        if video_name != self._current_video:
            self._history.clear()
            self._current_video = video_name
            if self.debug:
                self.logger.info(f"Switched to video: {video_name or 'root'}")

        self._history.append(path)
        prev_paths = list(self._history)

        if self.debug:
            self.logger.info(f"Processing screenshot: {path}")

        # Get transcription
        try:
            transcription = await self._call_gpt_vision(
                self.transcription_prompt,
                [path],
                video_name,
            )
        except Exception as exc:
            transcription = f"[transcription failed: {exc}]"
            if self.debug:
                self.logger.info(f"Transcription error: {exc}")

        # Get summary
        try:
            summary = await self._call_gpt_vision(
                self.summary_prompt,
                prev_paths,
                video_name,
            )
        except Exception as exc:
            summary = f"[summary failed: {exc}]"
            if self.debug:
                self.logger.info(f"Summary error: {exc}")

        # Emit combined result with video tag
        txt = (transcription + summary).strip()
        if video_name:
            txt = f"[{video_name}] {txt}"
        
        if self.debug:
            self.logger.info(f"\n{'='*60}\nResult: {txt[:100]}\n{'='*60}\n")

        await self.update_queue.put(
            Update(content=txt, content_type="input_text")
        )

    # ─────────────────────────────── Main worker

    async def _worker(self) -> None:
        """Main worker that processes all images in the directory recursively."""
        logger = logging.getLogger("Retro")

        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(
                logging.Formatter("%(asctime)s - [VIA-RETRO] - %(message)s")
            )
            logger.addHandler(h)

        logger.setLevel(logging.INFO)
        logger.propagate = False

        # Get sorted list of images (recursive)
        images = self._get_sorted_images_recursive()
        total = len(images)

        if total == 0:
            logger.warning(f"No images found in {self.images_dir}")
            return

        # Count videos
        video_names = set(v for _, v in images if v is not None)
        if video_names:
            logger.info(f"Found {total} images across {len(video_names)} videos in {self.images_dir}")
        else:
            logger.info(f"Found {total} images to process in {self.images_dir}")

        for idx, (img_path, video_name) in enumerate(images):
            if not self._running:
                logger.info("Retro observer stopped early.")
                break

            video_tag = f"[{video_name}] " if video_name else ""
            logger.info(f"Processing [{idx + 1}/{total}]: {video_tag}{img_path.name}")

            await self._process_and_emit(str(img_path), video_name)

            # Small delay to avoid overwhelming the API / queue
            if self.process_delay > 0:
                await asyncio.sleep(self.process_delay)

        logger.info(f"Retro observer completed. Processed {idx + 1} images.")
        self.stopped.set()
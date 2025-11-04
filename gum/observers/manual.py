#!/usr/bin/env python3
"""Manual video observer - captures and analyzes screen content on-demand.

This observer captures screenshots when recording is active and uses GPT-4 Vision
to analyze the content. It can skip captures when certain applications are visible.

Run manually by pressing p
"""

from __future__ import annotations

###############################################################################
# Imports                                                                     #
###############################################################################

# — Standard library —
import asyncio
import base64
import logging
import os
import time
from collections import deque
from typing import Iterable, Optional

# — Third-party —
import mss
from PIL import Image
from openai import AsyncOpenAI

try:
    import Quartz
except ImportError:
    Quartz = None

from shapely.geometry import box
from shapely.ops import unary_union

# — Local —
from .observer import Observer
from ..schemas import Update
from gum.prompts.screen import TRANSCRIPTION_PROMPT, SUMMARY_PROMPT #TODO: create new prompt file for manual capture observer

###############################################################################
# Constants                                                                   #
###############################################################################

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_SCREENSHOTS_DIR = "~/.cache/gum/screenshots"
DEFAULT_HISTORY_K = 10
CAPTURE_INTERVAL_SEC = 10
SHORT_SLEEP_SEC = 0.1
JPEG_QUALITY = 70

###############################################################################
# Window geometry helpers                                                     #
###############################################################################


def _get_global_bounds() -> tuple[float, float, float, float]:
    """Return a bounding box enclosing **all** physical displays.

    Returns
    -------
    (min_x, min_y, max_x, max_y) tuple in Quartz global coordinates.
    """
    if Quartz is None:
        return 0.0, 0.0, 0.0, 0.0
        
    err, ids, cnt = Quartz.CGGetActiveDisplayList(16, None, None)
    if err != Quartz.kCGErrorSuccess:
        raise OSError(f"CGGetActiveDisplayList failed: {err}")

    min_x = min_y = float("inf")
    max_x = max_y = -float("inf")
    
    for did in ids[:cnt]:
        r = Quartz.CGDisplayBounds(did)
        x0, y0 = r.origin.x, r.origin.y
        x1, y1 = x0 + r.size.width, y0 + r.size.height
        min_x, min_y = min(min_x, x0), min(min_y, y0)
        max_x, max_y = max(max_x, x1), max(max_y, y1)
        
    return min_x, min_y, max_x, max_y


def _get_visible_windows() -> list[tuple[dict, float]]:
    """List *onscreen* windows with their visible-area ratio.

    Each tuple is ``(window_info_dict, visible_ratio)`` where *visible_ratio*
    is in ``[0.0, 1.0]``. Internal system windows (Dock, WindowServer, …) are
    ignored.
    """
    if Quartz is None:
        return []
        
    _, _, _, gmax_y = _get_global_bounds()

    opts = (
        Quartz.kCGWindowListOptionOnScreenOnly
        | Quartz.kCGWindowListOptionIncludingWindow
    )
    wins = Quartz.CGWindowListCopyWindowInfo(opts, Quartz.kCGNullWindowID)

    occupied = None  # running union of opaque regions above the current window
    result: list[tuple[dict, float]] = []

    for info in wins:
        owner = info.get("kCGWindowOwnerName", "")
        if owner in ("Dock", "WindowServer", "Window Server"):
            continue

        bounds = info.get("kCGWindowBounds", {})
        x, y, w, h = (
            bounds.get("X", 0),
            bounds.get("Y", 0),
            bounds.get("Width", 0),
            bounds.get("Height", 0),
        )
        if w <= 0 or h <= 0:
            continue  # hidden or minimized

        inv_y = gmax_y - y - h  # Quartz→Shapely Y-flip
        poly = box(x, inv_y, x + w, inv_y + h)
        if poly.is_empty:
            continue

        visible = poly if occupied is None else poly.difference(occupied)
        if not visible.is_empty:
            ratio = visible.area / poly.area
            result.append((info, ratio))
            occupied = poly if occupied is None else unary_union([occupied, poly])

    return result


def _is_app_visible(names: Iterable[str]) -> bool:
    """Return *True* if **any** window from *names* is at least partially visible."""
    if Quartz is None:
        return False
        
    targets = set(names)
    return any(
        info.get("kCGWindowOwnerName", "") in targets and ratio > 0
        for info, ratio in _get_visible_windows()
    )


###############################################################################
# Manual video observer                                                       #
###############################################################################


class Manual(Observer):
    """Observer that manually captures and analyzes screen content when recording.

    This observer periodically captures screenshots when recording is active and uses
    GPT-4 Vision to analyze the content. It can skip captures when certain applications
    are visible.

    Args:
        model_name: GPT model to use for vision analysis.
        screenshots_dir: Directory to store screenshots.
        skip_when_visible: Application names to skip when visible.
        transcription_prompt: Custom prompt for transcribing screenshots.
        summary_prompt: Custom prompt for summarizing screenshots.
        history_k: Number of recent screenshots to keep in history.
        debug: Enable debug logging.
        api_key: OpenAI API key (or set OPENAI_API_KEY env var).
        api_base: OpenAI API base URL (or set VLLM_ENDPOINT env var).
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        screenshots_dir: str = DEFAULT_SCREENSHOTS_DIR,
        skip_when_visible: Optional[str | list[str]] = None,
        transcription_prompt: Optional[str] = None,
        summary_prompt: Optional[str] = None,
        history_k: int = DEFAULT_HISTORY_K,
        debug: bool = False,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        """Initialize the Manual video observer."""
        super().__init__()
        
        # Setup screenshot directory
        self.screens_dir = os.path.abspath(os.path.expanduser(screenshots_dir))
        os.makedirs(self.screens_dir, exist_ok=True)

        # Application visibility guard
        if isinstance(skip_when_visible, str):
            self._guard = {skip_when_visible}
        else:
            self._guard = set(skip_when_visible or [])

        # Prompts and model config
        self.transcription_prompt = transcription_prompt or TRANSCRIPTION_PROMPT
        self.summary_prompt = summary_prompt or SUMMARY_PROMPT
        self.model_name = model_name
        self.debug = debug

        # History and state
        self._history: deque[str] = deque(maxlen=max(0, history_k))
        self._is_recording = False

        # OpenAI client
        self.client = AsyncOpenAI(
            base_url=api_base or os.getenv("VLLM_ENDPOINT"),
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "no-key"
        )

    # ─────────────────────────────── Recording control

    def start_recording(self) -> None:
        """Start the recording."""
        self._is_recording = True
        if self.debug:
            print("[Manual] Recording started.")

    def stop_recording(self) -> None:
        """Stop the recording."""
        self._is_recording = False
        if self.debug:
            print("[Manual] Recording stopped.")

    def toggle_recording(self) -> None:
        """Toggle the recording state."""
        if self._is_recording:
            self.stop_recording()
        else:
            self.start_recording()

    # ─────────────────────────────── Helper methods

    @staticmethod
    def _encode_image(img_path: str) -> str:
        """Encode an image file as base64.
        
        Args:
            img_path: Path to the image file.
            
        Returns:
            Base64 encoded image data.
        """
        with open(img_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode()

    async def _call_gpt_vision(self, prompt: str, img_paths: list[str]) -> str:
        """Call GPT Vision API to analyze images.
        
        Args:
            prompt: Prompt to guide the analysis.
            img_paths: List of image paths to analyze.
            
        Returns:
            GPT's analysis of the images.
        """
        # TODO: Remove this limitation - currently only processes first image
        # img_paths = img_paths[:1]
        
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
            print(f"[Manual] Sending {len(img_paths)} image(s) to vision API")
            print(f"[Manual] Prompt: {prompt[:100]}...")

        # Call API
        rsp = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            response_format={"type": "text"},
        )
        
        result = rsp.choices[0].message.content
        if self.debug:
            print(f"[Manual] API response: {result[:100]}...")
            
        return result

    async def _save_frame(self, frame, tag: str) -> str:
        """Save a frame as a JPEG image.
        
        Args:
            frame: Frame data from mss.
            tag: Tag to include in the filename.
            
        Returns:
            Path to the saved image.
        """
        ts = f"{time.time():.5f}"
        path = os.path.join(self.screens_dir, f"{ts}_{tag}.jpg")
        
        await asyncio.to_thread(
            Image.frombytes("RGB", (frame.width, frame.height), frame.rgb).save,
            path,
            "JPEG",
            quality=JPEG_QUALITY,
        )
        
        return path

    async def _process_and_emit(self, path: str) -> None:
        """Process a screenshot and emit an update.
        
        Args:
            path: Path to the screenshot to process.
        """
        self._history.append(path)
        prev_paths = list(self._history)

        if self.debug:
            print(f"[Manual] Processing screenshot: {path}")

        # Get transcription
        try:
            transcription = await self._call_gpt_vision(
                self.transcription_prompt, 
                [path]
            )
        except Exception as exc:
            transcription = f"[transcription failed: {exc}]"
            if self.debug:
                print(f"[Manual] Transcription error: {exc}")

        # Get summary
        try:
            summary = await self._call_gpt_vision(
                self.summary_prompt, 
                prev_paths
            )
        except Exception as exc:
            summary = f"[summary failed: {exc}]"
            if self.debug:
                print(f"[Manual] Summary error: {exc}")

        # Emit combined result
        txt = (transcription + summary).strip()
        if self.debug:
            print(f"[Manual] Emitting update: {txt[:100]}...")
            
        await self.update_queue.put(
            Update(content=txt, content_type="input_text")
        )

    def _skip(self) -> bool:
        """Check if capture should be skipped based on visible applications.
        
        Returns:
            True if capture should be skipped, False otherwise.
        """
        if not self._guard:
            return False
        return _is_app_visible(self._guard)

    # ─────────────────────────────── Main worker

    async def _worker(self) -> None:
        """Main worker that captures and processes screenshots periodically."""
        log = logging.getLogger("Manual")
        
        if self.debug:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s [Manual] %(message)s",
                datefmt="%H:%M:%S"
            )
        else:
            log.addHandler(logging.NullHandler())
            log.propagate = False

        log.info(
            f"Manual observer started — guarding {self._guard or '∅'}"
        )

        with mss.mss() as sct:
            while self._running:
                if not self._is_recording:
                    # Not recording - sleep briefly to avoid busy-waiting
                    await asyncio.sleep(SHORT_SLEEP_SEC)
                    continue

                # Check if we should skip this capture
                if self._skip():
                    log.info("Skipping capture (guarded app visible)")
                    await asyncio.sleep(CAPTURE_INTERVAL_SEC)
                    continue

                # Capture screenshot
                log.info("Capturing screenshot...")
                monitor = sct.monitors[1]  # Main monitor
                sct_img = sct.grab(monitor)

                # Save and process
                path = await self._save_frame(sct_img, "periodic")
                await self._process_and_emit(path)

                log.info("Screenshot captured and processed.")

                # Wait before next capture
                await asyncio.sleep(CAPTURE_INTERVAL_SEC)

        log.info("Manual observer stopped.")
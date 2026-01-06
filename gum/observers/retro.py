# Author: Andy Phu   
# Retroactive observer, performs structured state extraction on pre-recorded image sets
#

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List, Dict

from openai import AsyncOpenAI

from .observer import Observer
from ..schemas import Update
from ..invoke import invoke

from gum.prompts.retro import COMMAND_VOCABULARY, WIDGET_VOCABULARY, LOCATION_VOCABULARY, ACTIVITY_PROMPT, TRANSCRIPTION_PROMPT

# Convert vocabularies to sets for O(1) lookup
_COMMAND_SET = set(COMMAND_VOCABULARY)
_WIDGET_SET = set(WIDGET_VOCABULARY)
_LOCATION_SET = set(LOCATION_VOCABULARY)

###############################################################################
# Constants                                                                   #
###############################################################################

DEFAULT_MODEL = "gpt-4o-mini"
DEFAULT_BATCH_SIZE = 5
SHORT_SLEEP_SEC = 0.05
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

###############################################################################
# Structured Output Schemas                                                   #
###############################################################################

@dataclass
class ActivityState:
    """Activity state for a single frame."""
    image_index: int
    command: str
    widget: Optional[str]
    location: Optional[str]
    application: str
    activity: str
    details: str
    # Added by post-processing
    timestamp: Optional[str] = None
    image_path: Optional[str] = None
    
    def state_key(self) -> str:
        """Key for detecting state changes.
        
        Uses a separator that's unlikely to appear in the data fields.
        Escapes any occurrences of the separator in the data.
        """
        # Use a separator that's very unlikely to appear in activity names
        separator = "|||"
        # Escape any occurrences of separator in the fields
        def escape(s: Optional[str]) -> str:
            if s is None:
                return ""
            return str(s).replace(separator, "\\" + separator)
        
        return separator.join([
            escape(self.command),
            escape(self.widget),
            escape(self.location),
            escape(self.application),
            escape(self.activity)
        ])


@dataclass 
class CollapsedState:
    """Activity state collapsed across consecutive identical frames."""
    command: str
    widget: Optional[str]
    location: Optional[str]
    application: str
    activity: str
    details: str
    start_timestamp: str
    end_timestamp: str
    start_image: str
    end_image: str
    frame_count: int


###############################################################################
# Retro observer                                                              #
###############################################################################


class Retro(Observer):
    """Observer that extracts structured behavioral states from pre-recorded screenshots."""

    def __init__(
        self,
        images_dir: str,
        model_name: str = DEFAULT_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        emit_collapsed: bool = True,
        include_transcription: bool = False,
        process_delay: float = SHORT_SLEEP_SEC,
        debug: bool = False,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> None:
        super().__init__()
        
        # Initialize logger early so worker task can access it even if initialization fails
        self.logger = logging.getLogger("Retro")
        self.stopped = asyncio.Event()

        self.images_dir = Path(images_dir).expanduser().resolve()
        if not self.images_dir.is_dir():
            raise ValueError(f"Not a directory: {self.images_dir}")

        self.model_name = model_name
        self.batch_size = batch_size
        self.emit_collapsed = emit_collapsed
        self.include_transcription = include_transcription
        self.process_delay = process_delay
        self.debug = debug

        self._all_states: List[ActivityState] = []
        
        self.client = AsyncOpenAI(
            base_url=api_base or os.getenv("VLLM_ENDPOINT"),
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "no-key",
        )

    # ─────────────────────────────── Image utilities

    def _get_video_name(self, img_path: Path) -> Optional[str]:
        """Extract video/subfolder name from image path."""
        relative = img_path.relative_to(self.images_dir)
        return relative.parts[0] if len(relative.parts) > 1 else None

    def _get_sorted_images(self) -> List[tuple[Path, Optional[str]]]:
        """Get all images recursively, sorted by video then filename."""
        images = []
        for p in self.images_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS:
                video_name = self._get_video_name(p)
                images.append((p, video_name))
        return sorted(images, key=lambda x: (x[1] or "", x[0].name))

    def _get_timestamp_from_filename(self, img_path: Path) -> str:
        """Extract timestamp from filename, or return filename as identifier.
        
        Returns:
            ISO format datetime string in UTC if filename is a Unix timestamp,
            otherwise returns the filename stem as-is.
        """
        stem = img_path.stem
        # If filename is a unix timestamp
        try:
            ts = float(stem)
            if ts > 1e9:  # Unix timestamp check (after 2001-09-09)
                # Use UTC timezone for consistency
                dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                return dt.isoformat()
        except (ValueError, OSError):
            # ValueError for invalid float, OSError for out-of-range timestamps
            pass
        # Otherwise just return the stem as the timestamp identifier
        return stem

    @staticmethod
    def _encode_image(img_path: str) -> str:
        """Encode image as base64.
        
        Args:
            img_path: Path to the image file
            
        Returns:
            Base64 encoded image string
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            IOError: If image file can't be read
        """
        path = Path(img_path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
        if not path.is_file():
            raise ValueError(f"Path is not a file: {img_path}")
        
        try:
            with open(img_path, "rb") as f:
                return base64.b64encode(f.read()).decode()
        except IOError as e:
            raise IOError(f"Failed to read image file {img_path}: {e}") from e

    # ─────────────────────────────── Vision API

    async def _call_vision(self, prompt: str, img_paths: List[str], video_name: Optional[str] = None, max_retries: int = 3) -> str:
        """Call vision API. No system message, no json_object mode for vLLM compat.
        
        Args:
            prompt: The prompt to send to the vision API
            img_paths: List of image paths to analyze
            video_name: Optional video/subfolder name for organizing debug logs
            max_retries: Maximum number of retry attempts for transient failures
            
        Returns:
            The API response content as a string
            
        Raises:
            ValueError: If API response is invalid or empty
            Exception: If all retry attempts fail
        """
        # Check if API key is available (will fail gracefully when making API calls if not provided)
        # Note: We allow initialization without API key, but API calls will fail with a clear error
        
        last_exception = None
        
        for attempt in range(max_retries):
            try:
                encoded = await asyncio.gather(
                    *[asyncio.to_thread(self._encode_image, p) for p in img_paths]
                )

                content = [
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{enc}"}}
                    for enc in encoded
                ]
                content.append({"type": "text", "text": prompt})

                # Single user message, no system message, text response format
                messages = [{"role": "user", "content": content}]

                if self.debug:
                    self.logger.info(f"Calling vision API with {len(img_paths)} images (attempt {attempt + 1}/{max_retries})")

                rsp = await invoke(
                    model=self.model_name,
                    messages=messages,
                    response_format={"type": "text"},  # Always text for vLLM
                    debug_tag="[Retro]",
                    debug_img_paths=img_paths,
                    debug_path=Path(video_name) if video_name else None,
                    client=self.client,
                )

                if not rsp or not rsp.choices or not rsp.choices[0].message.content:
                    raise ValueError("Empty or invalid API response")
                
                result = rsp.choices[0].message.content
                
                if self.debug:
                    self.logger.info(f"Got response: {result[:200]}...")

                return result
                
            except Exception as e:
                last_exception = e
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff
                    self.logger.warning(f"Vision API call failed (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                    await asyncio.sleep(wait_time)
                else:
                    self.logger.error(f"Vision API call failed after {max_retries} attempts: {e}")
                    raise
        
        # Should never reach here, but type checker needs it
        raise last_exception or Exception("Unknown error in vision API call")

    def _validate_vocabulary(self, command: str, widget: Optional[str], location: Optional[str]) -> tuple[str, Optional[str], Optional[str]]:
        """Validate and normalize vocabulary terms.
        
        Args:
            command: Command term to validate
            widget: Widget term to validate (can be None)
            location: Location term to validate (can be None)
            
        Returns:
            Tuple of (validated_command, validated_widget, validated_location)
            
        Raises:
            ValueError: If command is not in vocabulary
        """
        # Validate command (required)
        if command not in _COMMAND_SET:
            self.logger.warning(f"Invalid command '{command}', defaulting to 'idle'. Valid commands: {COMMAND_VOCABULARY}")
            command = "idle"
        
        # Validate widget (optional)
        if widget is not None and widget not in _WIDGET_SET:
            self.logger.warning(f"Invalid widget '{widget}', setting to None. Valid widgets: {WIDGET_VOCABULARY}")
            widget = None
        
        # Validate location (optional)
        if location is not None and location not in _LOCATION_SET:
            self.logger.warning(f"Invalid location '{location}', setting to None. Valid locations: {LOCATION_VOCABULARY}")
            location = None
        
        return command, widget, location

    def _parse_json_from_response(self, response: str) -> Optional[List[Dict]]:
        """Extract JSON array from LLM response (handles markdown code blocks)."""
        # Try to find JSON array in response
        # Handle ```json ... ``` blocks
        json_match = re.search(r'```(?:json)?\s*([\[\{][\s\S]*?[\]\}])\s*```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find raw JSON array
        array_match = re.search(r'\[[\s\S]*\]', response)
        if array_match:
            try:
                return json.loads(array_match.group())
            except json.JSONDecodeError:
                pass
        
        # Try to find single JSON object
        obj_match = re.search(r'\{[\s\S]*\}', response)
        if obj_match:
            try:
                obj = json.loads(obj_match.group())
                return [obj] if isinstance(obj, dict) else obj
            except json.JSONDecodeError:
                pass
        
        self.logger.warning(f"Could not parse JSON from response: {response[:500]}")  # Increased from 300 to 500 chars
        return None

    # ─────────────────────────────── State extraction

    async def _extract_batch(
        self, 
        img_paths: List[str],
        timestamps: List[str],
        video_name: Optional[str] = None,
    ) -> List[ActivityState]:
        """Extract activity states from a batch of images.
        
        Args:
            img_paths: List of image file paths
            timestamps: List of timestamps corresponding to each image
            video_name: Optional video/subfolder name for organizing debug logs
        """
        if self.debug:
            self.logger.info(f"Extracting batch of {len(img_paths)} images")

        try:
            prompt = ACTIVITY_PROMPT.format(n_images=len(img_paths), interval=5.0)
            response = await self._call_vision(prompt, img_paths, video_name=video_name)
            data = self._parse_json_from_response(response)
            
            if not data:
                self.logger.warning("No valid JSON in response, returning idle states")
                return [
                    ActivityState(
                        image_index=i + 1,
                        command="idle",
                        widget=None,
                        location=None,
                        application="unknown",
                        activity="unknown",
                        details="parse failed",
                        timestamp=timestamps[i],
                        image_path=img_paths[i],
                    )
                    for i in range(len(img_paths))
                ]

            states = []
            for i, item in enumerate(data):
                if i >= len(img_paths):
                    break
                
                # Validate image_index is within bounds
                image_index = item.get("image", i + 1)
                if not isinstance(image_index, int) or image_index < 1 or image_index > len(img_paths):
                    self.logger.warning(f"Invalid image_index {image_index} for item {i}, using {i + 1}")
                    image_index = i + 1
                
                # Get and validate vocabulary terms
                command = item.get("command", "idle")
                widget = item.get("widget")
                location = item.get("location")
                command, widget, location = self._validate_vocabulary(command, widget, location)
                    
                state = ActivityState(
                    image_index=image_index,
                    command=command,
                    widget=widget,
                    location=location,
                    application=item.get("application", "unknown"),
                    activity=item.get("activity", "unknown"),
                    details=item.get("details", ""),
                    timestamp=timestamps[i],
                    image_path=img_paths[i],
                )
                states.append(state)
            
            # Fill in any missing indices
            while len(states) < len(img_paths):
                i = len(states)
                states.append(ActivityState(
                    image_index=i + 1,
                    command="idle",
                    widget=None,
                    location=None,
                    application="unknown",
                    activity="unknown",
                    details="no llm output for this frame",
                    timestamp=timestamps[i],
                    image_path=img_paths[i],
                ))
            
            return states

        except (ValueError, KeyError, TypeError) as e:
            # Programming errors or data validation errors - should not happen
            self.logger.error(f"Batch extraction failed due to data error: {e}", exc_info=True)
            raise  # Re-raise to fail fast rather than silently continue
            
        except Exception as e:
            # API or network errors that couldn't be recovered
            self.logger.error(f"Batch extraction failed after retries: {e}", exc_info=True)
            # Return error states so processing can continue, but mark them clearly
            return [
                ActivityState(
                    image_index=i + 1,
                    command="idle",
                    widget=None,
                    location=None,
                    application="unknown",
                    activity="error",
                    details=f"API error: {str(e)}",
                    timestamp=timestamps[i],
                    image_path=img_paths[i],
                )
                for i in range(len(img_paths))
            ]

    async def _get_transcription(self, img_path: str, video_name: Optional[str] = None) -> Optional[str]:
        """Get transcription for single image.
        
        Args:
            img_path: Path to the image file
            video_name: Optional video/subfolder name for organizing debug logs
        """
        if not self.include_transcription:
            return None
        try:
            return await self._call_vision(TRANSCRIPTION_PROMPT, [img_path], video_name=video_name)
        except Exception as e:
            self.logger.warning(f"Transcription failed: {e}")
            return None

    # ─────────────────────────────── Post-processing

    def _collapse_states(self, states: List[ActivityState]) -> List[CollapsedState]:
        """Collapse consecutive identical states into spans."""
        if not states:
            return []
        
        collapsed = []
        group = [states[0]]
        
        for state in states[1:]:
            if state.state_key() == group[0].state_key():
                group.append(state)
            else:
                collapsed.append(self._make_collapsed(group))
                group = [state]
        
        collapsed.append(self._make_collapsed(group))
        return collapsed

    def _make_collapsed(self, group: List[ActivityState]) -> CollapsedState:
        """Create CollapsedState from group of identical states."""
        first, last = group[0], group[-1]
        return CollapsedState(
            command=first.command,
            widget=first.widget,
            location=first.location,
            application=first.application,
            activity=first.activity,
            details=first.details,
            start_timestamp=first.timestamp,
            end_timestamp=last.timestamp,
            start_image=first.image_path,
            end_image=last.image_path,
            frame_count=len(group),
        )

    # ─────────────────────────────── Main worker

    async def _worker(self) -> None:
        """Process all images with structured state extraction."""
        logger = self.logger
        if not logger.handlers:
            h = logging.StreamHandler()
            h.setFormatter(logging.Formatter("%(asctime)s - [RETRO] - %(message)s"))
            logger.addHandler(h)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        images = self._get_sorted_images()
        total = len(images)

        if total == 0:
            logger.warning(f"No images found in {self.images_dir}")
            self.stopped.set()
            return

        logger.info(f"Processing {total} images in batches of {self.batch_size}")

        # Group by video
        videos: Dict[Optional[str], List[tuple[Path, int]]] = {}
        for idx, (path, video_name) in enumerate(images):
            videos.setdefault(video_name, []).append((path, idx))

        # Process each video
        for video_name, video_images in videos.items():
            if not self._running:
                break

            video_label = video_name or "root"
            logger.info(f"Video '{video_label}': {len(video_images)} images")

            # Process in batches
            for batch_start in range(0, len(video_images), self.batch_size):
                if not self._running:
                    break

                batch = video_images[batch_start:batch_start + self.batch_size]
                batch_paths = [str(p) for p, _ in batch]
                batch_timestamps = [
                    self._get_timestamp_from_filename(p) for p, _ in batch
                ]

                batch_num = batch_start // self.batch_size + 1
                logger.info(f"  Batch {batch_num}: images {batch_start + 1}-{batch_start + len(batch)}")

                # Extract states
                states = await self._extract_batch(batch_paths, batch_timestamps, video_name=video_name)
                self._all_states.extend(states)

                # Emit per-frame if not collapsing
                if not self.emit_collapsed:
                    for state in states:
                        await self.update_queue.put(
                            Update(
                                content=json.dumps(asdict(state), default=str),
                                content_type="state_json",
                            )
                        )

                if self.process_delay > 0:
                    await asyncio.sleep(self.process_delay)

        # Final emission: collapsed or summary
        if self.emit_collapsed and self._all_states:
            logger.info("Collapsing states...")
            collapsed = self._collapse_states(self._all_states)
            logger.info(f"Collapsed {len(self._all_states)} frames → {len(collapsed)} state spans")

            for cs in collapsed:
                await self.update_queue.put(
                    Update(
                        content=json.dumps(asdict(cs), default=str),
                        content_type="collapsed_state_json",
                    )
                )

        logger.info(f"Retro completed. Processed {len(self._all_states)} frames.")
        self.stopped.set()
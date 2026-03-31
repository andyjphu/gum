"""Ablation observer: all passes use activity extraction (no intent passes).

Subclasses Retro and overrides pass routing so every pass — odd or even —
runs activity extraction/refinement. Pass 1 uses _extract_activity_pass1(),
all subsequent passes use _refine_activity(). Intent methods are never called.

Output uses the same file format but all files are states_primitive_P{N}.json
(no states_intent_P{N}.json files are produced).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from .retro import Retro

logger = logging.getLogger(__name__)


class RetroActivityOnly(Retro):
    """Ablation variant: all passes extract activities, no intent passes."""

    def _get_video_name(self, img_path: Path) -> Optional[str]:
        """Always return the images_dir name as the video name.

        In ablation, images_dir points to a single video's screenshots
        (e.g., screenshots/behindP12/). The parent class returns None
        for flat directories, causing output to dump without a video
        subfolder. This override ensures output nests under the video ID.
        """
        return self.images_dir.name

    async def _process_single_pass(
        self,
        images: List[tuple[Path, Optional[str]]],
        pass_num: int,
    ) -> List[Dict[str, Any]]:
        """Process all images for a single pass — always activity."""
        self._current_pass = pass_num
        states = []
        total = len(images)

        for idx, (img_path, video_name) in enumerate(images):
            if not self._running:
                logger.info(f"Pass {pass_num} stopped early.")
                break

            self.logger.info(
                f"Pass {pass_num} (activity-only) [{idx + 1}/{total}]: {img_path.name}"
            )

            if video_name != self._current_video:
                self._history.clear()
                self._current_video = video_name

            self._history.append(str(img_path))

            if pass_num == 1:
                state_data = await self._extract_activity_pass1(str(img_path), video_name)
            else:
                state_data = await self._refine_activity(
                    str(img_path), idx, total, video_name
                )

            states.append(state_data)
            self._save_single_state(video_name, state_data)

            activity = state_data.get("primitive_state", "unknown")
            conf = state_data.get("confidence", 5)
            content = f"[{video_name or 'video'}] Activity: {activity} (confidence: {conf})"

            from gum.schemas import Update
            await self.update_queue.put(
                Update(content=content, content_type="input_text")
            )

            if self.process_delay > 0:
                import asyncio
                await asyncio.sleep(self.process_delay)

        return states

    def _get_pass_type_str(self) -> str:
        """All passes are activity in this ablation."""
        return "activity"

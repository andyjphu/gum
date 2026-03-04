# Auth: Andy Phu
# Centralized VLLM server communications

from typing import List, Dict, Any
from openai import AsyncOpenAI, BadRequestError
from datetime import datetime, timezone
from pathlib import Path
import logging
import os
import urllib.request

from .data import save_to_file, copy_imgs

logger = logging.getLogger("Invoke")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


_NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")
_GUM_MACHINE = os.environ.get("GUM_MACHINE", "")


def _notify_error(msg: str) -> None:
    """Send error notification via ntfy.sh (fire-and-forget)."""
    if not _NTFY_TOPIC:
        return
    prefix = f"[{_GUM_MACHINE}] " if _GUM_MACHINE else ""
    try:
        req = urllib.request.Request(
            f"https://ntfy.sh/{_NTFY_TOPIC}",
            data=f"{prefix}{msg}".encode(),
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception:
        pass


def newest_img_timestamp(img_paths: List[str] | str | None) -> str:
    """Get timestamp from newest image filename, or current time if none."""
    if not img_paths:
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    # Handle both single path and list of paths
    paths = img_paths if isinstance(img_paths, list) else [img_paths]
    
    try:
        # Extract timestamp from filename (format: YYYYMMDD_HHMMSS.jpg)
        timestamps = []
        for p in paths:
            if p:
                filename = Path(p).stem  # e.g., "20251203_000000"
                timestamps.append(filename)
        
        return max(timestamps) if timestamps else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    except (ValueError, OSError):
        return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


async def invoke(
    client: AsyncOpenAI,
    model: str,
    messages: List[Dict[str, Any]],
    response_format: dict,
    debug_tag: str = "",
    debug_img_paths: List[str] | str | None = None,
    debug_path: Path | str | None = None,
    **kwargs,
):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    # Folder named by newest image
    folder_ts = newest_img_timestamp(debug_img_paths)
    debug_path_value = Path(debug_path) if debug_path else Path("")
    subfolder_path = debug_path_value / f"{folder_ts}-{debug_tag}"
    
    # Save debug for Retro tags (including pass suffixes like [Retro_P1], [Retro_P2])
    save_debug = (
        debug_tag.startswith("[Retro") or
        debug_tag in ("[Retro]", "[Retro Transcription]", "[Retro Summary]")
    )
    
    if save_debug:
        save_to_file(text=f"{messages}", subfolder=subfolder_path, filename=f"{folder_ts}-{debug_tag}-SND.txt")
        if debug_img_paths:
            # Convert to list if it's a single string
            img_list = debug_img_paths if isinstance(debug_img_paths, list) else [debug_img_paths]
            copy_imgs(img_paths=img_list, subfolder=subfolder_path)
    logger.info(f"{ts} [INVOKE] {debug_tag} sent, img_paths: {debug_img_paths}")
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            response_format=response_format,
            frequency_penalty=0.01,
            temperature=0.1,
            **kwargs,
        )
    except BadRequestError as e:
        video_label = f" {debug_path}" if debug_path else ""
        _notify_error(f"400{video_label} {debug_tag}: {e.message}")
        raise
    
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    if save_debug:
        save_to_file(text=str(response.choices[0].message.content), subfolder=subfolder_path, filename=f"{folder_ts}-{debug_tag}-RCV.txt")
    
    logger.info(f"{ts} [INVOKE] {debug_tag} received")
    
    return response
# Auth: Andy Phu
# Centralized VLLM server communications

from typing import List, Dict, Any
from openai import AsyncOpenAI
from datetime import datetime, timezone
from pathlib import Path
import logging
import os

from .data import save_to_file, copy_imgs

logger = logging.getLogger("Invoke")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    logger.addHandler(logging.StreamHandler())


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
    debug_path: Path | None = None,
    **kwargs,
):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    # Folder named by newest image
    folder_ts = newest_img_timestamp(debug_img_paths)
    debug_path_value = debug_path or Path("")
    subfolder_path = debug_path_value / f"{folder_ts}-{debug_tag}"
    
    save_debug = debug_tag in ("[Retro]", "[Retro Transcription]", "[Retro Summary]")
    
    if save_debug:
        save_to_file(text=f"{messages}", subfolder=subfolder_path, filename=f"{folder_ts}-{debug_tag}-SND.txt")
        if debug_img_paths:
            # Convert to list if it's a single string
            img_list = debug_img_paths if isinstance(debug_img_paths, list) else [debug_img_paths]
            copy_imgs(img_paths=img_list, subfolder=subfolder_path)
    logger.info(f"{ts} [INVOKE] {debug_tag} sent, img_paths: {debug_img_paths}")
    
    response = await client.chat.completions.create(
        model=model,
        messages=messages,
        response_format=response_format,
        frequency_penalty=0.01,
        temperature=0.1,
        **kwargs,
    )
    
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    
    if save_debug:
        save_to_file(text=str(response.choices[0].message.content), subfolder=subfolder_path, filename=f"{folder_ts}-{debug_tag}-RCV.txt")
    
    logger.info(f"{ts} [INVOKE] {debug_tag} received")
    
    return response
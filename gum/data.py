# Auth: Andy Phu
# Introduced for client-side logging, central file handler
import os
from pathlib import Path
from typing import List

from gum.config import TRAFFIC_LOG_DIR

import shutil

BASE = TRAFFIC_LOG_DIR

def save_to_file(
    text: str,
    filename: str, 
    subfolder: Path):
    
    clean_name = subfolder.name.replace(" ", "").replace("[", "").replace("]", "")
    subfolder = subfolder.parent / clean_name
    filename = filename.replace(" ", "").replace("[", "").replace("]", "")
    
    folder = BASE / subfolder
    folder.mkdir(parents=True, exist_ok=True)  # create if not exists

    filepath = folder / filename

    
    with open(filepath, "w") as f:
        f.write(text)
        
    return filepath

def copy_imgs(img_paths: List[str], subfolder: Path):
    
    # subfolder = subfolder.replace(" ", "").replace("[", "").replace("]", "")
    clean_name = subfolder.name.replace(" ", "").replace("[", "").replace("]", "")
    subfolder = subfolder.parent / clean_name
    
    dest = BASE / subfolder
    dest.mkdir(parents=True, exist_ok=True)

    out_paths = []
    for p in img_paths:
        p = Path(p)
        target = dest / p.name
        shutil.copy(p, target)
        out_paths.append(target)

    return out_paths

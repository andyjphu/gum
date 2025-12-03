# Author: Andy Phu
# Central place to coordinate data movement on local system for GUM
import os
from pathlib import Path


BASE_DIR = Path(os.getenv("GUM_BASE_DIR", Path(__file__).parent))
SHARED_DIR = Path(os.getenv("SHARED_BASE_DIR", Path(__file__).parent))

print(BASE_DIR)

DATA_DIR = BASE_DIR / "data"

CACHE_DIR = BASE_DIR / ".cache"

# to store logs from gum client to server and responses to correspond network traffic
TRAFFIC_LOG_DIR = SHARED_DIR / "log"

# to store images taken ahead of run to be analyzed retroactively
RETRO_IMAGES_DIR = BASE_DIR / "retro"  / "src" / "images"


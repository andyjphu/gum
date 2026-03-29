# Author: Andy Phu
# Central place to coordinate data movement on local system for GUM
import os
from pathlib import Path

# Base dirs
BASE_DIR = Path(os.getenv("GUM_BASE_DIR", Path(__file__).parent))
SHARED_DIR = Path(os.getenv("SHARED_BASE_DIR", Path(__file__).parent))

DATA_DIR = Path(os.getenv("GUM_DATA_DIR", BASE_DIR / "data"))

#where old logic used for the internal running of the GUM engine occurs
CACHE_DIR = Path(os.getenv("GUM_CACHE_DIR", BASE_DIR / ".cache"))

# to store logs from gum client to server and responses to correspond network traffic
TRAFFIC_LOG_DIR = Path(os.getenv("GUM_TRAFFIC_LOG_DIR", SHARED_DIR / "log"))

# to store images taken ahead of run to be analyzed retroactively
RETRO_IMAGES_DIR = Path(os.getenv("GUM_RETRO_IMAGES_DIR", SHARED_DIR / "screenshots"))

# Multi-pass configuration (alternating activity/intent architecture)

# Odd passes extract activities, even passes infer intents.
# Each pass sees the previous pass_window_size passes as context.
DEFAULT_NUM_PASSES = int(os.getenv("GUM_NUM_PASSES", "6"))

# Maximum unique states to include in context window for re-analysis
DEFAULT_CONTEXT_WINDOW_SIZE = int(os.getenv("GUM_CONTEXT_WINDOW_SIZE", "20"))

# Number of temporally nearest frames to include in context (for intent inference)
DEFAULT_TEMPORAL_WINDOW_SIZE = int(os.getenv("GUM_TEMPORAL_WINDOW_SIZE", "50"))

# Number of prior passes to include summaries from in the pass window
DEFAULT_PASS_WINDOW_SIZE = int(os.getenv("GUM_PASS_WINDOW_SIZE", "2"))

# Max unique state labels shown per pass in multi-pass context
DEFAULT_MAX_STATES_IN_CONTEXT = int(os.getenv("GUM_MAX_STATES_IN_CONTEXT", "15"))

# Directory for storing pass outputs (states, summaries)
PASS_OUTPUT_DIR = Path(os.getenv("GUM_PASS_OUTPUT_DIR", DATA_DIR / "passes"))

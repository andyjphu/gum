# Auth: Andy Phu
# Introduced for client-side logging, central file handler
import os
from pathlib import Path

BASE = Path(__file__).parent 

def save_to_file(
    text: str,
    file_name: str, 
    file_path: str = "./data"):
    with open(os.path.join(BASE, file_path, file_name), "w") as f:
        f.write(text)
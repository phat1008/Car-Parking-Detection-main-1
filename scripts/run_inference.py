"""Compatibility entry point; use scripts/run.py as the inference runner."""

import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from scripts.run import main

if __name__ == "__main__":
    main()

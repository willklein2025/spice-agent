"""Compatibility entry point for the RC sweep CLI."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from spice_agent.sweep import main


if __name__ == "__main__":
    main()

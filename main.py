"""Backwards-compatible entry point: `python main.py --team ...` == `nbaopt lineup --team ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nbaopt.cli import lineup_main  # noqa: E402

if __name__ == "__main__":
    lineup_main()

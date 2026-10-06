"""Backwards-compatible entry point: `python train.py --team ...` == `nbaopt train --team ...`."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nbaopt.train import main  # noqa: E402

if __name__ == "__main__":
    main()

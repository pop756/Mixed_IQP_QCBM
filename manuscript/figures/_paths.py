"""Locate the experiments tree from either layout:
- public repo:   <repo>/manuscript/figures -> <repo>/legacy/experiments/paper_ours
- or the $PAPER_OURS env var."""
import os
from pathlib import Path

_CANDIDATES = [
    Path(os.environ["PAPER_OURS"]) if "PAPER_OURS" in os.environ else None,
    Path(__file__).resolve().parents[2] / "legacy" / "experiments" / "paper_ours",
]
for _c in _CANDIDATES:
    if _c is not None and _c.exists():
        PAPER_OURS = _c
        break
else:
    raise FileNotFoundError(
        "paper_ours experiments dir not found; set PAPER_OURS env var")

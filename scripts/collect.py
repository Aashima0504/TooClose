"""Opt-in data collection. Implemented in Phase 4.

Requires an explicit consent confirmation. Does not save raw frames unless
``--save-crops`` is passed. Crops, if saved, stay under ``data/`` which is gitignored.
"""

from __future__ import annotations


def main() -> int:
    raise NotImplementedError("Phase 4: scripts/collect.py")


if __name__ == "__main__":
    raise SystemExit(main())

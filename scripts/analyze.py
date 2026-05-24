"""Plot sweep curves from JSONL logs. Stub."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="runs/default.jsonl")
    args = parser.parse_args()
    print(f"analyze({args.input}) not yet implemented", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

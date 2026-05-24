"""Drive the Phase 1 L-sweep across providers, schemes, and benchmarks.

This is a stub. Real implementation pending Phase 1.
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/sweep/phase1.yaml")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print(f"would run sweep with config={args.config}")
        return 0

    print("sweep_phase1 not yet implemented; see DESIGN.md section 3", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

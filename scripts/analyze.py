from __future__ import annotations

import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a pickled checkpoint buffer.")
    parser.add_argument("checkpoint", type=Path)
    args = parser.parse_args()

    with (args.checkpoint / "engine.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    buffer = payload["buffer"]
    print(buffer.stats())
    for solution in buffer.best(k=5):
        print(f"reward={solution.reward:.4f} step={solution.step} id={solution.id}")


if __name__ == "__main__":
    main()

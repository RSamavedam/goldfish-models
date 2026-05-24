from __future__ import annotations

import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Sequence

from ttt_discover.reward.base import RewardFunction, RewardResult


class SandboxedReward(RewardFunction):
    """Execute generated Python code in a subprocess with a timeout."""

    def __init__(self, command: Sequence[str] | None = None, timeout_s: float = 5.0) -> None:
        self.command = list(command) if command is not None else [sys.executable]
        self.timeout_s = timeout_s

    def __call__(self, solution: str) -> RewardResult:
        started = time.perf_counter()
        with tempfile.TemporaryDirectory() as tmpdir:
            script = Path(tmpdir) / "candidate.py"
            script.write_text(solution, encoding="utf-8")
            try:
                proc = subprocess.run(
                    [*self.command, str(script)],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return RewardResult(0.0, False, {"error": "timeout", "timeout_s": self.timeout_s})
        elapsed = time.perf_counter() - started
        valid = proc.returncode == 0
        return RewardResult(
            reward=1.0 if valid else 0.0,
            valid=valid,
            metadata={
                "returncode": proc.returncode,
                "stdout": proc.stdout,
                "stderr": proc.stderr,
                "runtime_s": elapsed,
            },
        )

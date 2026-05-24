from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ttt_discover.engine import TTTEngine


@dataclass
class CheckpointInfo:
    path: str
    tag: str
    step: int


class CheckpointManager:
    def __init__(self, directory: str = "./checkpoints") -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def save(self, engine: "TTTEngine", tag: str) -> str:
        path = self.directory / tag
        path.mkdir(parents=True, exist_ok=True)
        engine.policy.save_checkpoint(str(path / "policy"))
        with (path / "engine.pkl").open("wb") as handle:
            pickle.dump({"buffer": engine.buffer, "step_idx": engine.step_idx, "config": engine.config}, handle)
        return str(path)

    def load(self, path: str) -> "TTTEngine":
        raise NotImplementedError("Checkpoint load requires application-specific dependency wiring.")

    def branch(self, path: str, new_tag: str) -> str:
        source = Path(path)
        target = self.directory / new_tag
        if target.exists():
            raise FileExistsError(target)
        shutil.copytree(source, target)
        return str(target)

    def list_checkpoints(self) -> list[CheckpointInfo]:
        infos: list[CheckpointInfo] = []
        for child in sorted(self.directory.iterdir()):
            if child.is_dir():
                infos.append(CheckpointInfo(path=str(child), tag=child.name, step=-1))
        return infos

from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    from omegaconf import OmegaConf
except Exception:  # pragma: no cover
    OmegaConf = None  # type: ignore[assignment]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if OmegaConf is not None:
        cfg = OmegaConf.load(path)
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]

    import yaml

    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)

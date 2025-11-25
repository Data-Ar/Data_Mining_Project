from __future__ import annotations

from pathlib import Path

import yaml

from .training.engine import TrainingConfig


def load_config(path: str | Path) -> TrainingConfig:
    """Load a YAML config file into a TrainingConfig dataclass."""

    data = yaml.safe_load(Path(path).read_text())
    return TrainingConfig(**data)


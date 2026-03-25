import os
from pathlib import Path

root = Path(__file__).resolve().parents[1]


def _env_path(name: str, default: Path) -> Path:
    value = os.environ.get(name)
    return Path(value).expanduser().resolve() if value else default.resolve()


DATA_PATH = _env_path("SGAT_DATA_PATH", root / "data")
WEIGHTS_PATH = _env_path("SGAT_WEIGHTS_PATH", root / "weights")
EXP_ROOT = _env_path("SGAT_EXP_ROOT", root / "outputs")

TRAINING_PATH = EXP_ROOT / "training"
EVAL_PATH = EXP_ROOT / "results"

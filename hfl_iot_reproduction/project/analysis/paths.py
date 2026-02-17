from pathlib import Path


def _find_repo_root() -> Path:
    # Docker services use /workspace; local runs can use the checkout path.
    workspace = Path("/workspace")
    if workspace.exists():
        return workspace
    return Path(__file__).resolve().parents[2]


ROOT = _find_repo_root()
OUT = ROOT / "outputs"
LOGS = ROOT / "logs"
DATA = ROOT / "data"
CONFIG = ROOT / "config"


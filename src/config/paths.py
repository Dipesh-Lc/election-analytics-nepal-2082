from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @property
    def data(self) -> Path:
        return self.root / "data"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def outputs(self) -> Path:
        return self.data / "outputs"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def artifacts(self) -> Path:
        return self.root / "models" / "artifacts"


def get_project_root() -> Path:
    # repo_root/src/config/paths.py -> repo_root
    return Path(__file__).resolve().parents[2]


PATHS = ProjectPaths(root=get_project_root())

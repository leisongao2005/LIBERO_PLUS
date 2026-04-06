from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def episode_metadata_path(episode_dir: Path) -> Path:
    return episode_dir / "episode.json"


def session_metadata_path(session_dir: Path) -> Path:
    return session_dir / "session.json"


def manifest_path(directory: Path, name: str = "manifest.json") -> Path:
    return directory / name

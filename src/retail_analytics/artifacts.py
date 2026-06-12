"""Artifact persistence and run manifests."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def new_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]


def save_artifact(obj: Any, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    return path


def load_artifact(path: Path) -> Any:
    return joblib.load(path)


def write_manifest(
    path: Path,
    *,
    run_id: str,
    stage: str,
    metadata: dict[str, Any],
) -> Path:
    manifest = {
        "run_id": run_id,
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat(),
        **metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return path


def write_json_report(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path

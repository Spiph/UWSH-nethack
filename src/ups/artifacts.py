"""Content-addressed manifests and atomic structured artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ups.config import Phase0Config


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path) -> str:
    if path.is_file():
        return sha256_file(path)
    if not path.is_dir():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    for child in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(str(child.relative_to(path)).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(child)))
    return digest.hexdigest()


def source_tree_hash() -> str:
    """Bind provenance to the implementation, including uncommitted source."""
    digest = hashlib.sha256()
    roots = (Path("src"), Path("configs"), Path("tests"), Path(".github"))
    files = [path for root in roots if root.exists() for path in root.rglob("*") if path.is_file()]
    files.extend(
        path
        for path in (
            Path("pyproject.toml"),
            Path("uv.lock"),
            Path("Dockerfile"),
            Path("compose.yaml"),
        )
        if path.is_file()
    )
    for path in sorted(files):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def provenance() -> dict[str, Any]:
    def command(*args: str) -> str | None:
        try:
            return subprocess.run(
                args, check=True, capture_output=True, text=True, timeout=5
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "git_commit": command("git", "rev-parse", "HEAD"),
        "git_dirty": bool(command("git", "status", "--porcelain")),
        "source_tree_sha256": source_tree_hash(),
        "dependency_lock_sha256": (
            sha256_file(Path("uv.lock")) if Path("uv.lock").is_file() else None
        ),
        "container_image": os.environ.get("UPS_CONTAINER_IMAGE"),
        "sol_commit": os.environ.get("SOL_COMMIT"),
    }


def write_manifest(
    config: Phase0Config,
    command_name: str,
    outputs: list[Path],
    extra: dict[str, Any] | None = None,
) -> Path:
    snapshots: list[dict[str, str]] = []
    for output in outputs:
        if not output.is_file():
            continue
        digest = sha256_file(output)
        snapshot = config.artifact_root / "objects" / digest / output.name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        if not snapshot.exists():
            shutil.copy2(output, snapshot)
        snapshots.append({"path": str(snapshot), "sha256": digest, "source_path": str(output)})
    manifest = {
        "schema_version": 1,
        "command": command_name,
        "config_hash": config.digest,
        "config": json.loads(config.canonical_json()),
        "outputs": snapshots,
        "provenance": provenance(),
        "extra": extra or {},
    }
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    path = config.artifact_root / "manifests" / f"{command_name}-{digest}.json"
    atomic_json(path, manifest)
    return path

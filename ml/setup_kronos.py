from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

# The optional Xet transport can stall before creating the partial file on
# ordinary public checkpoints. The standard HTTP client is resumable and is
# sufficient for these five pinned artifacts.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

from kronos_model_zoo import (
    KRONOS_SOURCE_COMMIT,
    KRONOS_SOURCE_URL,
    selected_specs,
    snapshot_dir,
    snapshots_root,
    source_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install the pinned Kronos source and public checkpoints."
    )
    parser.add_argument("--models", default="all", help="all or mini,small,base")
    return parser.parse_args()


def run(*args: str, cwd: Path | None = None) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout.strip()


def file_sha256(file: Path) -> str:
    digest = hashlib.sha256()
    with file.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_source(repo_root: Path) -> Path:
    target = source_root(repo_root)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--no-checkout", KRONOS_SOURCE_URL, str(target))
    if not (target / ".git").is_dir():
        raise RuntimeError(f"Kronos source path is not a Git checkout: {target}")
    current = run("git", "rev-parse", "HEAD", cwd=target)
    if current != KRONOS_SOURCE_COMMIT:
        dirty = run("git", "status", "--porcelain", cwd=target)
        if dirty:
            raise RuntimeError(
                f"Kronos source has local edits at {target}; expected pinned "
                f"commit {KRONOS_SOURCE_COMMIT}"
            )
        run("git", "fetch", "--depth", "1", "origin", KRONOS_SOURCE_COMMIT, cwd=target)
        run("git", "checkout", "--detach", KRONOS_SOURCE_COMMIT, cwd=target)
    actual = run("git", "rev-parse", "HEAD", cwd=target)
    if actual != KRONOS_SOURCE_COMMIT:
        raise RuntimeError(f"Kronos source pin failed: {actual}")
    return target


def download_snapshot(repo_root: Path, repo_id: str, revision: str) -> dict:
    target = snapshot_dir(repo_root, repo_id)
    target.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=target,
        allow_patterns=("config.json", "model.safetensors", "README.md"),
    )
    required = (target / "config.json", target / "model.safetensors")
    if any(not file.is_file() for file in required):
        raise RuntimeError(f"incomplete Kronos snapshot: {target}")
    files = {
        file.name: {
            "bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in required
    }
    return {
        "repoId": repo_id,
        "revision": revision,
        "path": str(target.relative_to(repo_root)).replace("\\", "/"),
        "bytes": sum(int(identity["bytes"]) for identity in files.values()),
        "files": files,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    specs = selected_specs(args.models)
    source = ensure_source(repo_root)
    requested: dict[tuple[str, str], None] = {}
    for spec in specs:
        requested[(spec.model_repo, spec.model_revision)] = None
        requested[(spec.tokenizer_repo, spec.tokenizer_revision)] = None
    snapshots = [
        download_snapshot(repo_root, repo_id, revision)
        for repo_id, revision in requested
    ]
    manifest = {
        "version": 2,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "source": {
            "url": KRONOS_SOURCE_URL,
            "commit": KRONOS_SOURCE_COMMIT,
            "path": str(source.relative_to(repo_root)).replace("\\", "/"),
        },
        "models": [spec.id for spec in specs],
        "snapshots": snapshots,
    }
    root = snapshots_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "manifest.json"
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)
    total = sum(int(snapshot["bytes"]) for snapshot in snapshots)
    print(
        f"Kronos setup complete: {len(specs)} public models, "
        f"{len(snapshots)} pinned snapshots, {total / 2**20:.1f} MiB."
    )


if __name__ == "__main__":
    main()

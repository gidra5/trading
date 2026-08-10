from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
from urllib.parse import quote

# Public multi-gigabyte weights are more reliable through resumable HTTP on the
# machines used for this project than through the optional Xet transport.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from forecast_model_zoo import (
    ForecastModelSpec,
    selected_specs,
    snapshot_dir,
    snapshots_root,
    source_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Install pinned source and weights for the forecast model zoo."
    )
    parser.add_argument("--models", default="all", help="all or fincast,tirex2,chronos2")
    parser.add_argument(
        "--sources-only",
        action="store_true",
        help="clone and pin source without downloading model weights",
    )
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


def remote_file_size(url: str) -> int | None:
    result = subprocess.run(
        ("curl", "--silent", "--show-error", "--location", "--range", "0-0", "--dump-header", "-", "--output", os.devnull, url),
        check=True,
        text=True,
        encoding="utf-8",
        stdout=subprocess.PIPE,
    )
    matches = re.findall(r"content-range:\s*bytes\s+\d+-\d+/(\d+)", result.stdout, flags=re.IGNORECASE)
    return int(matches[-1]) if matches else None


def download_file(url: str, destination: Path) -> None:
    partial = destination.with_name(f"{destination.name}.partial")
    total = remote_file_size(url)
    existing = partial.stat().st_size if partial.is_file() else 0
    if total is None or total < 64 * 2**20:
        subprocess.run(
            (
                "curl", "--fail", "--location", "--retry", "5", "--retry-delay", "2",
                "--continue-at", "-", "--output", str(partial), url,
            ),
            check=True,
        )
        os.replace(partial, destination)
        return
    if existing > total:
        raise RuntimeError(f"partial download is larger than remote file: {partial}")
    if existing == total:
        os.replace(partial, destination)
        return

    requested_parts = max(1, int(os.environ.get("TRADING_DOWNLOAD_PARTS", "8")))
    remaining = total - existing
    part_count = min(requested_parts, max(1, remaining // (64 * 2**20)))
    chunk = (remaining + part_count - 1) // part_count
    ranges = []
    for index in range(part_count):
        start = existing + index * chunk
        end = min(total - 1, start + chunk - 1)
        if start <= end:
            ranges.append((index, start, end))
    print(
        f"Downloading {destination.name}: {existing / 2**20:.1f} MiB already present, "
        f"{(total - existing) / 2**30:.2f} GiB in {len(ranges)} parallel ranges.",
        flush=True,
    )

    def fetch(item: tuple[int, int, int]) -> Path:
        index, start, end = item
        part = partial.with_name(f"{partial.name}.part-{index:03d}")
        expected = end - start + 1
        if part.is_file() and part.stat().st_size == expected:
            return part
        subprocess.run(
            (
                "curl", "--silent", "--show-error", "--fail", "--location", "--retry", "5",
                "--retry-all-errors", "--retry-delay", "2", "--range", f"{start}-{end}",
                "--output", str(part), url,
            ),
            check=True,
        )
        if part.stat().st_size != expected:
            raise RuntimeError(
                f"range {start}-{end} returned {part.stat().st_size} bytes, expected {expected}"
            )
        return part

    with ThreadPoolExecutor(max_workers=len(ranges)) as executor:
        parts = list(executor.map(fetch, ranges))
    assembled = partial.with_name(f"{partial.name}.assembled")
    with assembled.open("wb") as output:
        if partial.is_file():
            with partial.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 2**20)
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, output, length=4 * 2**20)
    if assembled.stat().st_size != total:
        raise RuntimeError(f"assembled download has {assembled.stat().st_size} bytes, expected {total}")
    os.replace(assembled, destination)
    partial.unlink(missing_ok=True)
    for part in parts:
        part.unlink(missing_ok=True)


def ensure_source(repo_root: Path, spec: ForecastModelSpec) -> Path:
    target = source_root(repo_root, spec)
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--no-checkout", spec.source_url, str(target))
    if not (target / ".git").is_dir():
        raise RuntimeError(f"{spec.display_name} source is not a Git checkout: {target}")
    current = run("git", "rev-parse", "HEAD", cwd=target)
    if current != spec.source_commit:
        dirty = run("git", "status", "--porcelain", cwd=target)
        if dirty:
            raise RuntimeError(
                f"{spec.display_name} source has local edits at {target}; "
                f"expected pinned commit {spec.source_commit}"
            )
        run("git", "fetch", "--depth", "1", "origin", spec.source_commit, cwd=target)
        run("git", "checkout", "--detach", spec.source_commit, cwd=target)
    actual = run("git", "rev-parse", "HEAD", cwd=target)
    if actual != spec.source_commit:
        raise RuntimeError(f"{spec.display_name} source pin failed: {actual}")
    return target


def download_snapshot(repo_root: Path, spec: ForecastModelSpec) -> dict:
    target = snapshot_dir(repo_root, spec)
    target.mkdir(parents=True, exist_ok=True)
    for relative in spec.required_files:
        destination = target / relative
        if destination.is_file():
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        url = (
            f"https://huggingface.co/{spec.model_repo}/resolve/"
            f"{spec.model_revision}/{quote(relative)}?download=true"
        )
        download_file(url, destination)
    required = tuple(target / relative for relative in spec.required_files)
    if any(not file.is_file() for file in required):
        raise RuntimeError(f"incomplete {spec.display_name} snapshot: {target}")
    files = {
        str(file.relative_to(target)).replace("\\", "/"): {
            "bytes": file.stat().st_size,
            "sha256": file_sha256(file),
        }
        for file in required
    }
    for relative, expected in spec.expected_sha256:
        actual = str(files[relative]["sha256"])
        if actual != expected:
            raise RuntimeError(
                f"{spec.display_name} checksum mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
    return {
        "id": spec.id,
        "displayName": spec.display_name,
        "repoId": spec.model_repo,
        "revision": spec.model_revision,
        "path": str(target.relative_to(repo_root)).replace("\\", "/"),
        "bytes": sum(int(identity["bytes"]) for identity in files.values()),
        "files": files,
        "license": spec.license,
        "nativeDistribution": spec.native_distribution,
        "adaptation": spec.adaptation,
    }


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    specs = selected_specs(args.models)
    sources = []
    for spec in specs:
        source = ensure_source(repo_root, spec)
        sources.append(
            {
                "id": spec.id,
                "url": spec.source_url,
                "commit": spec.source_commit,
                "path": str(source.relative_to(repo_root)).replace("\\", "/"),
            }
        )
    if args.sources_only:
        print(f"Pinned {len(sources)} forecast-model source repositories.")
        return

    snapshots = [download_snapshot(repo_root, spec) for spec in specs]
    manifest = {
        "version": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "sources": sources,
        "snapshots": snapshots,
    }
    root = snapshots_root(repo_root)
    root.mkdir(parents=True, exist_ok=True)
    output = root / "manifest.json"
    temporary = output.with_name(f"{output.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    total = sum(int(snapshot["bytes"]) for snapshot in snapshots)
    print(
        f"Forecast model setup complete: {len(snapshots)} pinned checkpoints, "
        f"{total / 2**30:.2f} GiB verified."
    )


if __name__ == "__main__":
    main()

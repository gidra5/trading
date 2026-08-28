from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil

import numpy as np

from trading_storage import require_under, training_storage_layout
from train_normalized_glu_next_return import atomic_json


DATASET_CONTRACT = "sparse-sequential-timeline-episodes-v1"
SPLITS = ("train", "validation", "test")


def episode_lengths(example_count: int, target_episode_seconds: int) -> list[int]:
    if min(example_count, target_episode_seconds) <= 0:
        raise ValueError("example count and episode length must be positive")
    episode_count = math.ceil(example_count / target_episode_seconds)
    base, remainder = divmod(example_count, episode_count)
    return [base + (index < remainder) for index in range(episode_count)]


def sparse_episode_origins(
    *,
    timeline_rows: int,
    example_count: int,
    target_episode_seconds: int,
    seed: int,
    boundary_guard_seconds: int = 60,
) -> tuple[np.ndarray, list[dict[str, int]]]:
    """Choose one contiguous episode at a random offset in each time stratum."""
    usable_origins = int(timeline_rows) - 1
    lengths = episode_lengths(example_count, target_episode_seconds)
    if usable_origins < example_count:
        raise ValueError("timeline cannot supply the requested sparse episodes")
    boundaries = np.linspace(
        0, usable_origins, num=len(lengths) + 1, dtype=np.int64
    )
    rng = np.random.default_rng(seed)
    rng.shuffle(lengths)
    episodes: list[dict[str, int]] = []
    selected: list[np.ndarray] = []
    for index, length in enumerate(lengths):
        left = int(boundaries[index])
        right = int(boundaries[index + 1])
        slack = right - left - length
        if slack < 0:
            raise ValueError("an episode does not fit in its time stratum")
        guard = min(int(boundary_guard_seconds), slack // 2)
        minimum = left + guard
        maximum = right - length - guard
        start = int(rng.integers(minimum, maximum + 1))
        stop = start + length
        selected.append(np.arange(start, stop, dtype=np.int32))
        episodes.append({
            "index": index,
            "startOrigin": start,
            "stopOriginExclusive": stop,
            "examples": length,
            "stratumStartOrigin": left,
            "stratumStopOriginExclusive": right,
        })
    origins = np.concatenate(selected)
    if origins.size != example_count or not np.all(np.diff(origins) > 0):
        raise RuntimeError("sparse episode origins are not chronological and unique")
    return origins, episodes


def parse_examples(value: str) -> dict[str, int]:
    parts = value.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected train,validation,test counts")
    counts = {split: int(count) for split, count in zip(SPLITS, parts)}
    if any(count <= 0 for count in counts.values()):
        raise argparse.ArgumentTypeError("all split counts must be positive")
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Materialize randomized sparse episodes over a complete 1s feature "
            "timeline without copying the large timeline arrays."
        )
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--examples", type=parse_examples, default=parse_examples("256000,65536,65536")
    )
    parser.add_argument("--episode-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=1337)
    return parser.parse_args()


def linked_timeline(source: Path, output: Path) -> str:
    if output.exists():
        raise FileExistsError(output)
    try:
        os.link(source, output)
        return "hard-link"
    except OSError:
        shutil.copy2(source, output)
        return "copy"


def timeline_time_offset(
    source_root: Path, manifest: dict, split: str
) -> float:
    count = int(manifest["examplesBySplit"][split])
    files = manifest["files"][split]
    origins = np.memmap(
        source_root / files["origins"], dtype="<i4", mode="r", shape=(count,)
    )
    times = np.memmap(
        source_root / files["times"], dtype="<f8", mode="r", shape=(count,)
    )
    indices = np.linspace(0, count - 1, min(1024, count), dtype=np.int64)
    offsets = np.asarray(times[indices], dtype=np.float64) - (
        np.asarray(origins[indices], dtype=np.float64) * 1000.0
    )
    if not np.all(offsets == offsets[0]):
        raise ValueError(f"{split} timeline timestamps are not a regular 1s grid")
    return float(offsets[0])


def main() -> None:
    args = parse_args()
    if args.episode_seconds <= 0:
        raise ValueError("episode seconds must be positive")
    repo = Path(__file__).resolve().parents[1]
    datasets_root = training_storage_layout(repo).datasets
    source_root = require_under((
        args.source if args.source.is_absolute() else repo / args.source
    ), datasets_root, "source dataset")
    output_root = require_under((
        args.output if args.output.is_absolute() else repo / args.output
    ), datasets_root, "output dataset")
    if output_root.exists():
        raise FileExistsError(f"refusing to replace existing dataset: {output_root}")
    source_manifest = json.loads(
        (source_root / "manifest.json").read_text(encoding="utf-8")
    )
    if source_manifest.get("storageLayout") != "temporal-channel-timeline-v1":
        raise ValueError("source dataset does not contain a temporal timeline")
    output_root.mkdir(parents=True)
    files: dict[str, dict[str, str]] = {}
    episode_metadata: dict[str, list[dict[str, int]]] = {}
    link_modes: dict[str, str] = {}
    try:
        for split_index, split in enumerate(SPLITS):
            rows = int(source_manifest["timelineRowsBySplit"][split])
            count = int(args.examples[split])
            origins, episodes = sparse_episode_origins(
                timeline_rows=rows,
                example_count=count,
                target_episode_seconds=args.episode_seconds,
                seed=args.seed + split_index * 1009,
            )
            timeline_name = f"{split}.timeline-features.f32"
            origins_name = f"{split}.origins.i32"
            targets_name = f"{split}.targets.f32"
            times_name = f"{split}.times.f64"
            source_timeline = (
                source_root
                / source_manifest["files"][split]["timelineFeatures"]
            )
            link_modes[split] = linked_timeline(
                source_timeline, output_root / timeline_name
            )
            origins.astype("<i4", copy=False).tofile(output_root / origins_name)
            timeline = np.memmap(
                output_root / timeline_name,
                dtype="<f4",
                mode="r",
                shape=(rows, int(source_manifest["temporalChannelCount"])),
            )
            np.asarray(timeline[origins.astype(np.int64) + 1, 0]).astype(
                "<f4", copy=False
            ).tofile(output_root / targets_name)
            offset = timeline_time_offset(source_root, source_manifest, split)
            (offset + origins.astype(np.float64) * 1000.0).astype(
                "<f8", copy=False
            ).tofile(output_root / times_name)
            mapping = getattr(timeline, "_mmap", None)
            if mapping is not None:
                mapping.close()
            files[split] = {
                "timelineFeatures": timeline_name,
                "origins": origins_name,
                "targets": targets_name,
                "times": times_name,
            }
            episode_metadata[split] = episodes

        manifest = {
            **source_manifest,
            "version": 1,
            "generatedAt": datetime.now(timezone.utc).isoformat().replace(
                "+00:00", "Z"
            ),
            "contract": DATASET_CONTRACT,
            "sourceDataset": str(source_root.relative_to(repo)).replace("\\", "/"),
            "targetFilter": (
                "none: every consecutive second inside each selected episode is "
                "included, including exact-zero next returns"
            ),
            "examplesBySplit": args.examples,
            "files": files,
            "episodeSampling": {
                "type": "random-offset-within-uniform-time-strata-v1",
                "cadenceSeconds": 1,
                "targetEpisodeSeconds": args.episode_seconds,
                "seed": args.seed,
                "chronologyWithinEpisode": "strict consecutive timeline rows",
                "episodeOrder": "chronological after randomized placement",
                "timelineStorage": link_modes,
                "splits": {
                    split: {
                        "episodes": len(episode_metadata[split]),
                        "minimumExamples": min(
                            item["examples"] for item in episode_metadata[split]
                        ),
                        "maximumExamples": max(
                            item["examples"] for item in episode_metadata[split]
                        ),
                    }
                    for split in SPLITS
                },
            },
        }
        atomic_json(manifest, output_root / "manifest.json")
        atomic_json({
            "contract": DATASET_CONTRACT,
            "seed": args.seed,
            "targetEpisodeSeconds": args.episode_seconds,
            "splits": episode_metadata,
        }, output_root / "episodes.json")
    except BaseException:
        shutil.rmtree(output_root, ignore_errors=True)
        raise
    print(json.dumps({
        "dataset": str(output_root.relative_to(repo)).replace("\\", "/"),
        "examplesBySplit": args.examples,
        "episodeSampling": manifest["episodeSampling"],
    }, separators=(",", ":")))


if __name__ == "__main__":
    main()

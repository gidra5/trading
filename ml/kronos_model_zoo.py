from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


KRONOS_SOURCE_URL = "https://github.com/shiyu-coder/Kronos.git"
KRONOS_SOURCE_COMMIT = "67b630e67f6a18c9e9be918d9b4337c960db1e9a"


@dataclass(frozen=True)
class KronosModelSpec:
    id: str
    model_repo: str
    model_revision: str
    tokenizer_repo: str
    tokenizer_revision: str
    max_context: int
    parameters: int
    default_batch_size: int


MODEL_SPECS = (
    KronosModelSpec(
        id="mini",
        model_repo="NeoQuasar/Kronos-mini",
        model_revision="f4e68697d9d5aed55cef5c96aabc3376bcad9f81",
        tokenizer_repo="NeoQuasar/Kronos-Tokenizer-2k",
        tokenizer_revision="26966d0035065a0cae0ebad7af8ece35bc1fb51c",
        max_context=2_048,
        parameters=4_100_000,
        default_batch_size=64,
    ),
    KronosModelSpec(
        id="small",
        model_repo="NeoQuasar/Kronos-small",
        model_revision="901c26c1332695a2a8f243eb2f37243a37bea320",
        tokenizer_repo="NeoQuasar/Kronos-Tokenizer-base",
        tokenizer_revision="0e0117387f39004a9016484a186a908917e22426",
        max_context=512,
        parameters=24_700_000,
        # Flattened stochastic-path batch tested on an RTX 3070 (8 GB).
        default_batch_size=80,
    ),
    KronosModelSpec(
        id="base",
        model_repo="NeoQuasar/Kronos-base",
        model_revision="2b554741eca47781b64468546e77fef3e85130e6",
        tokenizer_repo="NeoQuasar/Kronos-Tokenizer-base",
        tokenizer_revision="0e0117387f39004a9016484a186a908917e22426",
        max_context=512,
        parameters=102_300_000,
        default_batch_size=20,
    ),
)


def source_root(repo_root: Path) -> Path:
    return repo_root / ".tools" / "Kronos"


def snapshots_root(repo_root: Path) -> Path:
    return repo_root / ".tools" / "Kronos-models"


def snapshot_dir(repo_root: Path, repo_id: str) -> Path:
    return snapshots_root(repo_root) / repo_id.replace("/", "--")


def selected_specs(value: str) -> tuple[KronosModelSpec, ...]:
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not requested or requested == {"all"}:
        return MODEL_SPECS
    available = {spec.id: spec for spec in MODEL_SPECS}
    unknown = requested - available.keys()
    if unknown:
        raise ValueError(
            f"unknown Kronos model(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(available)}"
        )
    return tuple(spec for spec in MODEL_SPECS if spec.id in requested)

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ForecastModelSpec:
    id: str
    display_name: str
    source_url: str
    source_commit: str
    source_directory: str
    model_repo: str
    model_revision: str
    snapshot_files: tuple[str, ...]
    required_files: tuple[str, ...]
    expected_sha256: tuple[tuple[str, str], ...]
    license: str
    native_distribution: str
    adaptation: str


MODEL_SPECS = (
    ForecastModelSpec(
        id="fincast",
        display_name="FinCast",
        source_url="https://github.com/vincent05r/FinCast-fts.git",
        source_commit="488b19d1d85fa2b3d4b93469530cefdcf1cc97a4",
        source_directory="FinCast",
        model_repo="Vincent05R/FinCast",
        model_revision="2d7d90b159db8961d27c2cf165d51195902ef92b",
        snapshot_files=("README.md", "checksum_f.txt", "v1.pth"),
        required_files=("v1.pth",),
        expected_sha256=(("v1.pth", "d5ca999b02c944effa60d2b94174dc4d5a0cd2c0543ae289b2e36f37431492a8"),),
        license="Apache-2.0",
        native_distribution="mean plus 0.1-0.9 quantiles",
        adaptation="official experimental PEFT code; no complete training entrypoint",
    ),
    ForecastModelSpec(
        id="tirex2",
        display_name="TiRex-2",
        source_url="https://github.com/NX-AI/tirex-2.git",
        source_commit="ad7ce6a2a0cb639ea58eedc3afc472e7e5b2bae0",
        source_directory="TiRex-2",
        model_repo="NX-AI/TiRex-2",
        model_revision="05e5b26db52bfb256f1ae1bdf785589850482de3",
        snapshot_files=("LICENSE", "NOTICE", "README.md", "model-config.yaml", "model.ckpt"),
        required_files=("model-config.yaml", "model.ckpt"),
        expected_sha256=(),
        license="Apache-2.0",
        native_distribution="0.1-0.9 quantiles",
        adaptation="public release is inference-only",
    ),
    ForecastModelSpec(
        id="chronos2",
        display_name="Chronos-2",
        source_url="https://github.com/amazon-science/chronos-forecasting.git",
        source_commit="7dc4435706a4454feb79df44ca9f33631f3027bf",
        source_directory="chronos-forecasting",
        model_repo="amazon/chronos-2",
        model_revision="29ec3766d36d6f73f0696f85560a422f50e8498c",
        snapshot_files=("README.md", "config.json", "model.safetensors"),
        required_files=("config.json", "model.safetensors"),
        expected_sha256=(),
        license="Apache-2.0",
        native_distribution="0.1-0.9 quantiles",
        adaptation="official full and PEFT/LoRA fit API",
    ),
)


def source_root(repo_root: Path, spec: ForecastModelSpec) -> Path:
    return repo_root / ".tools" / spec.source_directory


def snapshots_root(repo_root: Path) -> Path:
    return repo_root / ".tools" / "forecast-models"


def snapshot_dir(repo_root: Path, spec: ForecastModelSpec) -> Path:
    return snapshots_root(repo_root) / spec.model_repo.replace("/", "--")


def selected_specs(value: str) -> tuple[ForecastModelSpec, ...]:
    requested = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not requested or requested == {"all"}:
        return MODEL_SPECS
    available = {spec.id: spec for spec in MODEL_SPECS}
    unknown = requested - available.keys()
    if unknown:
        raise ValueError(
            f"unknown forecast model(s): {', '.join(sorted(unknown))}; "
            f"available: {', '.join(available)}"
        )
    return tuple(spec for spec in MODEL_SPECS if spec.id in requested)

"""Leakage-safe USD-M-vs-Spot basis/flow residual audit beyond frozen v18.

Official USD-M one-minute klines are joined to canonical Spot candles only
after both candles have closed.  Existing Spot aggTrade shards contribute the
matching completed previous minute's quote imbalance; the target's current
second is never included.  Every candidate table's bins, shrinkage, and
joint/control backoff are selected on training only.

Frozen v18 is evaluated once on development validation.  The first validation
half jointly selects one train-fitted regime and its log-ratio scalar.  The
following 60 rows are embargoed, and only that fixed regime feeds the raw
T=.01 KL >=.002 screen on the remaining holdout.  Other holdouts are descriptive
only.  The paired absolute-value controls are binned and therefore do not imply
exact magnitude conditioning or sign-only attribution.  Sealed-test references
and payloads are never opened, and no training is performed.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

from audit_causal_oracle_ohlcv_predictability import (
    ACTION_COUNT,
    DAY_ROWS,
    FEATURE_NAMES as OHLCV_FEATURE_NAMES,
    causal_ohlcv_features,
    completed_candle_windows,
    completed_close_windows,
    mean_kl_indexed,
    mean_kl_stacked_indexed,
    normalized_mean,
    smoothed_table,
    split_and_purge,
    sufficient_table,
    utc_day_start_ms,
)
from audit_v18_calendar_fusion import (
    convex_probability_fusion,
    select_scalar,
    softmax,
    validate_preserved_checkpoint_without_test_access,
)
from audit_v18_futures_metrics_fusion import (
    CALIBRATION_FRACTION,
    DEVELOPMENT_SCREEN_GATE,
    EMBARGO_ROWS,
    EXPECTED_CHECKPOINT_EPOCH,
    MinuteCandleDayCache,
    PairedRegime,
    embargoed_split_metrics,
    mean_kl_dense,
    primary_development_screen,
    select_first_half_complementarity,
    validate_segment_clock_alignment,
)
from audit_v18_ohlcv_fusion import (
    AuditedMarketCloseCache,
    AuditedOracleTargetCache,
    fusion_metrics,
    log_ratio_feature_fusion,
    validation_timestamps,
)
from audit_v18_on_v28_validation_rows import collect_sequence_core_rows
from audit_v18_trade_flow_fusion import (
    CORPUS_SPLIT_CONTRACT,
    TRADE_FLOW_REFERENCE_DIR,
    read_trade_flow_day,
    validate_corpus_split_contract,
    validate_runtime_split_assignment,
)
from evaluate_joint_price_oracle_actions import resolve_device
from oracle_futures_basis_features import (
    FUTURES_BASIS_FEATURE_NAMES,
    FUTURES_KLINE_COLUMNS,
    causal_futures_basis_features,
)
from oracle_trade_flow_features import SECOND_ROWS
from trading_storage import (
    load_torch_checkpoint,
    read_derivatives_kline_columns,
    read_shard_array,
    require_under,
    training_storage_layout,
)
from train_joint_price_oracle import (
    CausalOracleDataset,
    CausalSegment,
    build_model,
    resolve,
    resolve_training_config,
    validate_plan,
)


DEFAULT_PLAN = Path(
    "ml/training-plans/"
    "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
)
FUTURES_KLINE_REFERENCE_DIR = Path(
    "data/market/immutable/refs/derivatives-klines/"
    "usdm-futures/btcusdt/1m"
)
FUTURES_KLINE_SCHEMA = "binance-usdm-futures-klines-v1"
FUTURES_KLINE_ENCODING = "derivatives-klines-columnar-v1"
FUTURES_KLINE_SOURCE_ROOT = (
    "https://data.binance.vision/data/futures/um/daily/klines/"
    "BTCUSDT/1m"
)
EXPECTED_REFERENCE_DAYS = 420
MINUTE_MS = 60_000
FUTURES_CLOSE_OFFSET_MS = 59_999
SPOT_QUOTE_COLUMNS = (
    "aggressiveBuyQuoteVolume",
    "aggressiveSellQuoteVolume",
)
SPOT_CROSS_FEATURE_NAMES = (
    "spotQuoteImbalance1m",
    "absSpotQuoteImbalance1m",
    "spotLogQuoteRate1mVs1h",
    "absSpotLogQuoteRate1mVs1h",
    "futuresMinusSpotQuoteImbalance1m",
    "absFuturesMinusSpotQuoteImbalance1m",
    "futuresSpotQuoteImbalanceAgreement1m",
    "absFuturesSpotQuoteImbalanceAgreement1m",
    "futuresSpotLogQuoteVolumeRatio1m",
    "absFuturesSpotLogQuoteVolumeRatio1m",
    "absFuturesSpotLogApproxQuoteVolumeRatio15m",
    "futuresSourceState",
)
FEATURE_NAMES = (
    OHLCV_FEATURE_NAMES
    + FUTURES_BASIS_FEATURE_NAMES
    + SPOT_CROSS_FEATURE_NAMES
)
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
PRIOR_STRENGTHS = (64.0, 256.0, 1_024.0)
CATEGORICAL_FEATURE_EDGES = {
    # 0 = missing source row, 1 = official no-trade row, 2 = live row.
    "futuresSourceState": np.asarray((0.5, 1.5), dtype=np.float64),
}


# Signed variables always carry binned absolute-value, market-state, activity,
# staleness, and causal OHLCV controls.  Separate quantile bins can still refine
# magnitude, so these regimes test complementarity rather than pure direction.
REGIMES = (
    PairedRegime(
        "basis-level-reversion",
        (
            "return60m", "rmsReturn60m", "meanLogRange60m",
            "absBasisLogLevel", "absBasisLogDeviation1h",
            "futuresLogQuoteRate15mVs1h", "basisObservationAge24h",
            "futuresSourceState",
        ),
        ("basisLogLevel", "basisLogDeviation1h"),
        (5, 4, 3, 3, 3, 3, 2, 3),
        (3, 3),
    ),
    PairedRegime(
        "basis-change",
        (
            "return15m", "return60m", "rmsReturn60m",
            "absBasisLogChange15m", "absBasisLogChange1h",
            "futuresLogQuoteRate15mVs1h", "basisObservationAge24h",
            "futuresSourceState",
        ),
        ("basisLogChange15m", "basisLogChange1h"),
        (4, 5, 4, 3, 3, 3, 2, 3),
        (3, 3),
    ),
    PairedRegime(
        "futures-taker-pressure",
        (
            "return15m", "rmsReturn15m", "futuresMeanLogRange15m",
            "absFuturesTakerQuoteImbalance1m",
            "absFuturesTakerQuoteImbalance15m",
            "futuresLogQuoteRate1mVs1h",
            "futuresPriceObservationAge24h",
            "futuresSourceState",
        ),
        (
            "futuresTakerQuoteImbalance1m",
            "futuresTakerQuoteImbalance15m",
        ),
        (4, 4, 3, 3, 3, 3, 2, 3),
        (3, 3),
    ),
    PairedRegime(
        "futures-spot-imbalance-divergence-agreement",
        (
            "return15m", "rmsReturn15m",
            "absFuturesTakerQuoteImbalance1m",
            "absSpotQuoteImbalance1m",
            "absFuturesMinusSpotQuoteImbalance1m",
            "absFuturesSpotQuoteImbalanceAgreement1m",
            "spotLogQuoteRate1mVs1h",
            "futuresPriceObservationAge24h",
            "futuresSourceState",
        ),
        (
            "futuresMinusSpotQuoteImbalance1m",
            "futuresSpotQuoteImbalanceAgreement1m",
        ),
        (4, 4, 3, 3, 3, 3, 3, 2, 3),
        (3, 3),
    ),
    PairedRegime(
        "relative-activity",
        (
            "return60m", "rmsReturn60m", "logVolume1mVs60m",
            "absFuturesSpotLogQuoteVolumeRatio1m",
            "absFuturesSpotLogApproxQuoteVolumeRatio15m",
            "spotLogQuoteRate1mVs1h",
            "futuresPriceObservationAge24h",
            "futuresSourceState",
        ),
        (
            "futuresSpotLogQuoteVolumeRatio1m",
            "futuresSpotLogApproxQuoteVolumeRatio15m",
        ),
        (5, 4, 3, 3, 3, 3, 2, 3),
        (3, 3),
    ),
)


@dataclass(frozen=True)
class TrainSelectedRegime:
    regime: PairedRegime
    control_strength: float
    joint_strength: float
    joint_weight: float


@dataclass(frozen=True)
class FittedRegimeTables:
    selection: TrainSelectedRegime
    control_edges: tuple[np.ndarray, ...]
    control_table: np.ndarray
    control_counts: np.ndarray
    joint_edges: tuple[np.ndarray, ...]
    joint_table: np.ndarray
    joint_counts: np.ndarray


@dataclass
class AccessLog:
    target_files: set[Path]
    candle_files: set[Path]
    futures_kline_files: set[Path]
    spot_flow_files: set[Path]


class FuturesKlineDayCache:
    def __init__(
        self,
        root: Path,
        opened: set[Path],
        *,
        target_contract: str,
        target_days: set[str],
        sealed_test_start: str,
        sealed_test_end: str,
        maximum_days: int = 3,
    ) -> None:
        self.root = root
        self.opened = opened
        self.target_contract = target_contract
        self.target_days = target_days
        self.sealed_test_start = sealed_test_start
        self.sealed_test_end = sealed_test_end
        self.maximum_days = max(2, int(maximum_days))
        self.days: OrderedDict[
            str, tuple[dict[str, np.ndarray], np.ndarray]
        ] = OrderedDict()
        self.recorded_days: set[str] = set()
        self.source_csv_rows = 0
        self.observed_grid_rows = 0
        self.live_grid_rows = 0
        self.no_trade_grid_rows = 0
        self.missing_grid_rows = 0
        self.off_grid_rows = 0
        self.source_archive_bytes = 0

    def load(
        self,
        day_value: str,
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        cached = self.days.pop(day_value, None)
        if cached is not None:
            self.days[day_value] = cached
            return cached
        values = read_futures_kline_day(
            self.root,
            day_value,
            self.opened,
            target_contract=self.target_contract,
            oracle_scope=(
                "train-or-validation-target"
                if day_value in self.target_days
                else "predecessor-context"
            ),
            sealed_test_start=self.sealed_test_start,
            sealed_test_end=self.sealed_test_end,
            metadata_callback=self.record_metadata,
        )
        self.days[day_value] = values
        while len(self.days) > self.maximum_days:
            self.days.popitem(last=False)
        return values

    def record_metadata(self, day_value: str, metadata: dict) -> None:
        if day_value in self.recorded_days:
            return
        self.recorded_days.add(day_value)
        self.source_csv_rows += int(metadata["sourceCsvRows"])
        self.observed_grid_rows += int(metadata["observedGridRows"])
        self.live_grid_rows += int(metadata["liveGridRows"])
        self.no_trade_grid_rows += int(metadata["noTradeGridRows"])
        self.missing_grid_rows += int(metadata["missingGridRows"])
        self.off_grid_rows += int(metadata["offGridRows"])
        self.source_archive_bytes += int(metadata["sourceArchiveBytes"])


class SpotFlowDayCache:
    def __init__(
        self,
        root: Path,
        opened: set[Path],
        *,
        target_contract: str,
        target_days: set[str],
        sealed_test_start: str,
        sealed_test_end: str,
        maximum_days: int = 3,
    ) -> None:
        self.root = root
        self.opened = opened
        self.target_contract = target_contract
        self.target_days = target_days
        self.sealed_test_start = sealed_test_start
        self.sealed_test_end = sealed_test_end
        self.maximum_days = max(2, int(maximum_days))
        self.days: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.recorded_days: set[str] = set()
        self.source_csv_rows = 0
        self.source_archive_bytes = 0
        self.invalid_sentinel_rows = 0
        self.false_best_match_rows = 0

    def load(self, day_value: str) -> dict[str, np.ndarray]:
        cached = self.days.pop(day_value, None)
        if cached is not None:
            self.days[day_value] = cached
            return cached
        values = read_spot_flow_day(
            self.root,
            day_value,
            self.opened,
            target_contract=self.target_contract,
            oracle_scope=(
                "train-or-validation-target"
                if day_value in self.target_days
                else "predecessor-context"
            ),
            sealed_test_start=self.sealed_test_start,
            sealed_test_end=self.sealed_test_end,
            metadata_callback=self.record_metadata,
        )
        self.days[day_value] = values
        while len(self.days) > self.maximum_days:
            self.days.popitem(last=False)
        return values

    def record_metadata(self, day_value: str, metadata: dict) -> None:
        if day_value in self.recorded_days:
            return
        self.recorded_days.add(day_value)
        self.source_csv_rows += int(metadata["sourceCsvRows"])
        self.source_archive_bytes += int(metadata["sourceArchiveBytes"])
        self.invalid_sentinel_rows += int(metadata["invalidSentinelRows"])
        self.false_best_match_rows += int(metadata["falseBestPriceMatchRows"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional JSON artifact path under data/runtime/logs.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = audit_v18_futures_basis_fusion(
        arguments.plan,
        requested_device=arguments.device,
    )
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if arguments.output is None:
        print(payload, end="")
        return
    repo_root = Path(__file__).resolve().parents[1]
    output = arguments.output
    if not output.is_absolute():
        output = repo_root / output
    output = output.resolve()
    runtime_logs = (repo_root / "data/runtime/logs").resolve()
    if output.parent != runtime_logs:
        raise ValueError("audit output must be directly under data/runtime/logs")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite audit artifact: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    if temporary.exists():
        raise FileExistsError(f"refusing stale audit temporary file: {temporary}")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    temporary.replace(output)
    print(f"Wrote audit artifact: {output}", file=sys.stderr, flush=True)


def audit_v18_futures_basis_fusion(
    plan_file: Path,
    *,
    requested_device: str = "auto",
) -> dict[str, object]:
    repo_root = Path(__file__).resolve().parents[1]
    storage = training_storage_layout(repo_root)
    resolved_plan = resolve(repo_root, plan_file).resolve()
    plan = json.loads(resolved_plan.read_text(encoding="utf-8"))
    validate_plan(plan)
    if plan.get("testPolicy", "sealed-never-load") != "sealed-never-load":
        raise ValueError("futures-basis audit requires sealed-never-load")
    model_config = plan["model"]
    training = plan["training"]
    resolved_training = resolve_training_config(training)
    if not bool(resolved_training.get("policyOnly", False)):
        raise ValueError("futures-basis audit requires policy-only v18")
    target_root = require_under(
        resolve(repo_root, Path(plan["targetReferenceDir"])),
        storage.immutable / "refs" / "oracle",
        "targetReferenceDir",
    )
    history_root = require_under(
        resolve(repo_root, Path(plan["historyDir"])),
        repo_root / "data" / "market" / "immutable" / "refs" / "candles",
        "historyDir",
    )
    minute_history_root = history_root.parent / "1m"
    futures_root = require_under(
        (repo_root / FUTURES_KLINE_REFERENCE_DIR).resolve(),
        repo_root / "data" / "market" / "immutable" / "refs"
        / "derivatives-klines",
        "futuresKlineReferenceDir",
    )
    spot_flow_root = require_under(
        (repo_root / TRADE_FLOW_REFERENCE_DIR).resolve(),
        repo_root / "data" / "market" / "immutable" / "refs" / "trade-flow",
        "spotTradeFlowReferenceDir",
    )
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])),
        storage.runs,
        "runDir",
    )
    target_files = sorted(target_root.glob("*.json"))
    corpus_contract = validate_corpus_split_contract(
        repo_root,
        target_root,
        target_files,
    )
    segments = split_and_purge(target_files)
    validate_runtime_split_assignment(target_files, segments, corpus_contract)
    validate_segment_clock_alignment(segments)
    counts = {
        split: sum(segment.count for segment in segments[split])
        for split in ("train", "validation", "test")
    }
    test_target_files = {
        segment.target_file.resolve() for segment in segments["test"]
    }
    test_days = {segment.target_file.stem for segment in segments["test"]}
    sealed_test_start = str(corpus_contract["test"]["first"])
    sealed_test_end = str(corpus_contract["test"]["last"])
    target_days = {
        segment.target_file.stem
        for split in ("train", "validation")
        for segment in segments[split]
    }

    # Both external namespaces must be complete before any target, source
    # payload, or checkpoint is opened.  This prevents partial-ingestion bias.
    required_days = validate_reference_coverage(
        futures_root,
        segments,
        sealed_test_start=sealed_test_start,
        label="USD-M kline",
    )
    spot_required_days = validate_reference_coverage(
        spot_flow_root,
        segments,
        sealed_test_start=sealed_test_start,
        label="Spot aggTrade",
    )
    if spot_required_days != required_days:
        raise RuntimeError("futures and Spot source scopes differ")

    access = AccessLog(set(), set(), set(), set())
    candle_cache = MinuteCandleDayCache(
        minute_history_root,
        access.candle_files,
        sealed_test_start=sealed_test_start,
    )
    futures_cache = FuturesKlineDayCache(
        futures_root,
        access.futures_kline_files,
        target_contract=target_root.name,
        target_days=target_days,
        sealed_test_start=sealed_test_start,
        sealed_test_end=sealed_test_end,
    )
    spot_flow_cache = SpotFlowDayCache(
        spot_flow_root,
        access.spot_flow_files,
        target_contract=target_root.name,
        target_days=target_days,
        sealed_test_start=sealed_test_start,
        sealed_test_end=sealed_test_end,
    )
    print(
        "Loading causal train/validation USD-M basis and completed Spot flow; "
        "sealed test remains untouched.",
        file=sys.stderr,
        flush=True,
    )
    train_features, train_targets = load_split(
        "train",
        segments["train"],
        candle_cache,
        futures_cache,
        spot_flow_cache,
        access,
    )
    validation_features, validation_targets = load_split(
        "validation",
        segments["validation"],
        candle_cache,
        futures_cache,
        spot_flow_cache,
        access,
    )
    required_futures_files = {
        (futures_root / f"{day_value}.json").resolve()
        for day_value in required_days
    }
    required_spot_files = {
        (spot_flow_root / f"{day_value}.json").resolve()
        for day_value in required_days
    }
    required_candle_files = {
        (minute_history_root / f"{day_value}.json").resolve()
        for day_value in required_days
    }
    if access.futures_kline_files != required_futures_files \
            or access.spot_flow_files != required_spot_files \
            or access.candle_files != required_candle_files:
        raise RuntimeError("source access differs from preflight scope")
    if access.target_files & test_target_files \
            or test_days & {path.stem for path in access.candle_files} \
            or test_days & {path.stem for path in access.futures_kline_files} \
            or test_days & {path.stem for path in access.spot_flow_files}:
        raise RuntimeError("sealed-test reference access detected")

    train_selected, train_candidates, table_selection_report = (
        select_paired_regime(train_features, train_targets)
    )
    fitted_regimes = fit_all_regime_tables(
        train_features,
        train_targets,
        train_candidates,
    )

    checkpoint_file = run_dir / "checkpoints" / "best.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file,
        map_location="cpu",
        weights_only=False,
    )
    validate_preserved_checkpoint_without_test_access(
        checkpoint,
        plan,
        model_config,
        resolved_training,
        counts,
        run_dir,
    )
    if int(checkpoint["epoch"]) != EXPECTED_CHECKPOINT_EPOCH:
        raise ValueError("preserved best checkpoint is no longer v18 epoch 7")
    device = resolve_device(requested_device, str(training["device"]))
    dataset = CausalOracleDataset(
        history_root,
        segments,
        int(model_config["contextLength"]),
        int(model_config["forecastHorizon"]),
        target_rows_per_file=DAY_ROWS,
        action_count=int(model_config["actionCount"]),
        close_cache_days=int(training.get("closeCacheDays", 10)),
        target_cache_days=int(training.get("targetCacheDays", 3)),
        pin_memory=device.type == "cuda",
        include_future_closes=False,
    )
    inference_history: set[Path] = set()
    inference_targets: set[Path] = set()
    dataset.close_cache = AuditedMarketCloseCache(
        history_root,
        int(training.get("closeCacheDays", 10)),
        inference_history,
    )
    dataset.target_cache = AuditedOracleTargetCache(
        int(training.get("targetCacheDays", 3)),
        rows_per_file=DAY_ROWS,
        action_count=int(model_config["actionCount"]),
        pin_memory=device.type == "cuda",
        opened=inference_targets,
    )
    model = build_model(model_config).to(device)
    model.load_state_dict(checkpoint["model"])
    print(
        "Running one frozen v18 development-validation inference pass.",
        file=sys.stderr,
        flush=True,
    )
    logits, model_targets = collect_sequence_core_rows(
        model,
        dataset,
        "validation",
        int(training["sequenceCoreTraining"]["coreRows"]),
        device,
        resolved_training,
    )
    del model, dataset
    if device.type == "cuda":
        torch.cuda.empty_cache()

    expected_shape = (counts["validation"], ACTION_COUNT)
    if any(value.shape != expected_shape for value in (
        logits,
        model_targets,
        validation_targets,
    )):
        raise RuntimeError("futures-basis fusion rows are misaligned")
    if not np.array_equal(model_targets, validation_targets):
        raise RuntimeError("v18 and futures-basis target order differs")
    expected_validation_targets = {
        segment.target_file.resolve() for segment in segments["validation"]
    }
    if inference_targets != expected_validation_targets \
            or inference_targets & test_target_files \
            or test_days & {path.stem for path in inference_history}:
        raise RuntimeError("unexpected v18 inference reference access")

    probabilities = softmax(logits)
    split_at = probabilities.shape[0] // 2
    holdout_start = split_at + EMBARGO_ROWS
    if holdout_start >= probabilities.shape[0]:
        raise RuntimeError("validation embargo leaves no untouched holdout")
    first_half_selected, first_half_candidates = (
        select_first_half_complementarity(
            validation_targets[:split_at],
            probabilities[:split_at],
            (
                complementarity_probability_pair(
                    fitted,
                    validation_features[:split_at],
                )
                for fitted in fitted_regimes
            ),
        )
    )
    first_half_by_name = {
        candidate.regime_name: candidate
        for candidate in first_half_candidates
    }
    expected_regimes = {
        fitted.selection.regime.name for fitted in fitted_regimes
    }
    if set(first_half_by_name) != expected_regimes:
        raise RuntimeError("first-half complementarity candidates are incomplete")
    v18_metrics = embargoed_split_metrics(
        validation_targets,
        probabilities,
        split_at,
        holdout_start,
    )
    candidate_reports: list[dict[str, object]] = []
    candidate_ratio_metrics: dict[str, dict[str, float | bool]] = {}
    primary_estimator: dict[str, object] | None = None
    train_selected_diagnostic: dict[str, object] | None = None
    for fitted in fitted_regimes:
        regime = fitted.selection.regime
        name = regime.name
        first_half = first_half_by_name[name]
        (
            control_probabilities,
            joint_probabilities,
            backed_joint_probabilities,
            validation_control_ids,
            validation_joint_ids,
        ) = validation_regime_probabilities(fitted, validation_features)
        ratio_fused = log_ratio_feature_fusion(
            probabilities,
            backed_joint_probabilities,
            control_probabilities,
            first_half.weight,
        )
        ratio_metrics = fusion_metrics(
            first_half.weight,
            first_half.raw_kl,
            embargoed_split_metrics(
                validation_targets,
                ratio_fused,
                split_at,
                holdout_start,
            ),
            v18_metrics,
        )
        candidate_ratio_metrics[name] = ratio_metrics
        candidate_report: dict[str, object] = {
            "regime": name,
            "selectedOnValidationFirstHalf": (
                name == first_half_selected.regime_name
            ),
            "usedForPrimaryGate": name == first_half_selected.regime_name,
            "trainOnlyConfiguration": {
                "controlPriorStrength": fitted.selection.control_strength,
                "jointPriorStrength": fitted.selection.joint_strength,
                "jointWeight": fitted.selection.joint_weight,
            },
            "controlFeatures": list(regime.control_features),
            "signedFeatures": list(regime.directional_features),
            "validationRowsWithEmptyControlCell": int(np.count_nonzero(
                fitted.control_counts[validation_control_ids] == 0
            )),
            "validationRowsWithEmptyJointCell": int(np.count_nonzero(
                fitted.joint_counts[validation_joint_ids] == 0
            )),
            "pairedAbsoluteActivityStalenessOhlcvControl": (
                embargoed_split_metrics(
                    validation_targets,
                    control_probabilities,
                    split_at,
                    holdout_start,
                )
            ),
            "jointSignedFeatureTable": embargoed_split_metrics(
                validation_targets,
                joint_probabilities,
                split_at,
                holdout_start,
            ),
            "trainSelectedBackoff": embargoed_split_metrics(
                validation_targets,
                backed_joint_probabilities,
                split_at,
                holdout_start,
            ),
            "logRatioResidualFusion": ratio_metrics,
        }
        candidate_reports.append(candidate_report)
        if name == first_half_selected.regime_name:
            primary_estimator = {
                "selectedRegime": name,
                "controlFeatures": list(regime.control_features),
                "signedFeatures": list(regime.directional_features),
                "controlPriorStrength": fitted.selection.control_strength,
                "jointPriorStrength": fitted.selection.joint_strength,
                "trainSelectedJointWeight": fitted.selection.joint_weight,
                "validationFirstHalfLogRatioWeight": first_half.weight,
                "validationFirstHalfRawKl": first_half.raw_kl,
                "validationRowsWithEmptyControlCell": candidate_report[
                    "validationRowsWithEmptyControlCell"
                ],
                "validationRowsWithEmptyJointCell": candidate_report[
                    "validationRowsWithEmptyJointCell"
                ],
            }
        if name == train_selected.regime.name:
            convex_weight, convex_fit_kl = select_scalar(
                lambda value: mean_kl_dense(
                    validation_targets[:split_at],
                    convex_probability_fusion(
                        probabilities[:split_at],
                        backed_joint_probabilities[:split_at],
                        value,
                    ),
                ),
                0,
                1,
            )
            convex_metrics = fusion_metrics(
                convex_weight,
                convex_fit_kl,
                embargoed_split_metrics(
                    validation_targets,
                    convex_probability_fusion(
                        probabilities,
                        backed_joint_probabilities,
                        convex_weight,
                    ),
                    split_at,
                    holdout_start,
                ),
                v18_metrics,
            )
            train_selected_diagnostic = {
                **candidate_report,
                "sameAsPrimaryComplementarityRegime": (
                    name == first_half_selected.regime_name
                ),
                "convexFusionDiagnostic": convex_metrics,
            }
    if primary_estimator is None or train_selected_diagnostic is None:
        raise RuntimeError("regime diagnostics are incomplete")
    primary_ratio_metrics = primary_development_screen(
        first_half_selected,
        candidate_ratio_metrics,
    )

    timestamps = validation_timestamps(segments["validation"])
    expected_feature_targets = {
        segment.target_file.resolve()
        for split in ("train", "validation")
        for segment in segments[split]
    }
    if access.target_files != expected_feature_targets:
        raise RuntimeError("unexpected feature-audit target access set")
    return {
        "schemaVersion": 1,
        "audit": "v18-plus-causal-usdm-spot-basis-flow-residual",
        "corpusSplitContract": {
            "file": CORPUS_SPLIT_CONTRACT.as_posix(),
            "targetContract": corpus_contract["targetContract"],
            "referenceCount": corpus_contract["referenceCount"],
            "referenceFilenameSha256": corpus_contract[
                "referenceFilenameSha256"
            ],
            "sealedTestStart": sealed_test_start,
            "sealedTestEnd": sealed_test_end,
        },
        "accessContract": {
            "trainTargetReferencesOpened": len({
                segment.target_file.resolve() for segment in segments["train"]
            }),
            "validationTargetReferencesOpened": len(
                expected_validation_targets
            ),
            "futuresKlineReferencesOpened": len(access.futures_kline_files),
            "spotAggTradeReferencesOpened": len(access.spot_flow_files),
            "causalOneMinuteSpotReferencesOpened": len(access.candle_files),
            "causalOneSecondV18ReferencesOpened": len(inference_history),
            "testTargetReferencesOpened": 0,
            "testTargetPayloadsOpened": 0,
            "testSpotCandleReferencesOpened": 0,
            "testSpotCandlePayloadsOpened": 0,
            "testFuturesKlineReferencesOpened": 0,
            "testFuturesKlinePayloadsOpened": 0,
            "testSpotAggTradeReferencesOpened": 0,
            "testSpotAggTradePayloadsOpened": 0,
            "modelWeightsChanged": False,
            "trainingStarted": False,
            "inferenceDevice": str(device),
        },
        "sourceQuality": {
            "requiredReferenceDaysPerSource": len(required_days),
            "futuresKlines": {
                "featureSchema": FUTURES_KLINE_SCHEMA,
                "closeAvailabilityOffsetMs": FUTURES_CLOSE_OFFSET_MS,
                "sourceDays": len(futures_cache.recorded_days),
                "sourceCsvRows": futures_cache.source_csv_rows,
                "observedGridRows": futures_cache.observed_grid_rows,
                "liveGridRows": futures_cache.live_grid_rows,
                "noTradeGridRows": futures_cache.no_trade_grid_rows,
                "missingGridRows": futures_cache.missing_grid_rows,
                "offGridRows": futures_cache.off_grid_rows,
                "sourceArchiveBytes": futures_cache.source_archive_bytes,
            },
            "spotAggTrades": {
                "sourceDays": len(spot_flow_cache.recorded_days),
                "sourceCsvRows": spot_flow_cache.source_csv_rows,
                "sourceArchiveBytes": spot_flow_cache.source_archive_bytes,
                "invalidSentinelRows": (
                    spot_flow_cache.invalid_sentinel_rows
                ),
                "falseBestPriceMatchRows": (
                    spot_flow_cache.false_best_match_rows
                ),
            },
        },
        "featureContract": {
            "featureCount": len(FEATURE_NAMES),
            "ohlcvControlFeatureCount": len(OHLCV_FEATURE_NAMES),
            "futuresBasisFeatureCount": len(FUTURES_BASIS_FEATURE_NAMES),
            "spotCrossMarketFeatureCount": len(SPOT_CROSS_FEATURE_NAMES),
            "futuresCandleUsedAtTargetMinuteK": "k-1-completed-minute",
            "spotFlowWindowUsedAtTargetMinuteK": "k-1-completed-minute",
            "currentTargetSecondExcludedFromSpotFlow": True,
            "perpMinusSpotReturnFeaturePresent": False,
            "deltaBasisIsSoleRelativeReturnRepresentation": True,
        },
        "estimator": {
            "primaryComplementarity": primary_estimator,
            "validationFirstHalfCandidates": [
                {
                    "regime": candidate.regime_name,
                    "logRatioWeight": candidate.weight,
                    "rawKl": candidate.raw_kl,
                    "selected": (
                        candidate.regime_name
                        == first_half_selected.regime_name
                    ),
                }
                for candidate in first_half_candidates
            ],
            "trainOnlyTableSelectionDiagnostic": table_selection_report,
        },
        "validationSplit": {
            "rows": counts["validation"],
            "jointSelectionRows": split_at,
            "forecastHorizonEmbargoRows": EMBARGO_ROWS,
            "embargoTimestampStart": int(timestamps[split_at]),
            "embargoTimestampEnd": int(timestamps[holdout_start - 1]),
            "withinAuditHoldoutRows": counts["validation"] - holdout_start,
            "globallyFresh": False,
            "role": "reused-development-validation",
            "jointSelectionTimestampStart": int(timestamps[0]),
            "jointSelectionTimestampEnd": int(timestamps[split_at - 1]),
            "withinAuditHoldoutTimestampStart": int(timestamps[holdout_start]),
            "withinAuditHoldoutTimestampEnd": int(timestamps[-1]),
        },
        "raw01Kl": {
            "v18": v18_metrics,
            "allRegimeComplementarityCandidates": candidate_reports,
            "logRatioResidualFusionPrimary": primary_ratio_metrics,
            "trainOnlySelectedRegimeDiagnostic": train_selected_diagnostic,
        },
        "selectionContract": {
            "both420ReferenceScopesCheckedBeforePayloadAccess": True,
            "allRegimeBinsShrinkageAndBackoffSelectedOnTrainOnly": True,
            "regimeAndFusionScalarSelectedOnFirstValidationHalfOnly": True,
            "validationFirstHalfSelectionCriterion": "lowest-raw-kl",
            "forecastHorizonEmbargoApplied": True,
            "allCandidateHoldoutsReportedOnlyAsDiagnostics": True,
            "holdoutDiagnosticsUsedToSelectRegime": False,
            "primaryGateUsesFirstHalfSelectedRegimeOnly": True,
            "withinAuditHoldoutUsedForParameterSelection": False,
            "withinAuditHoldoutUsedForDevelopmentScreen": True,
            "developmentScreenGate": DEVELOPMENT_SCREEN_GATE,
            "validationReusedAcrossEarlierExperiments": True,
            "sealedTestRemainsGloballyUntouched": True,
            "testUsed": False,
        },
    }


def required_feature_days(
    segments: dict[str, list[CausalSegment]],
) -> set[str]:
    result: set[str] = set()
    for split in ("train", "validation"):
        for segment in segments[split]:
            current = date.fromisoformat(segment.target_file.stem)
            result.add(current.isoformat())
            result.add((current - timedelta(days=1)).isoformat())
    return result


def validate_reference_coverage(
    root: Path,
    segments: dict[str, list[CausalSegment]],
    *,
    sealed_test_start: str,
    label: str,
    expected_count: int = EXPECTED_REFERENCE_DAYS,
) -> tuple[str, ...]:
    """Validate an exact external namespace without parsing any reference."""
    days = required_feature_days(segments)
    if len(days) != expected_count:
        raise ValueError(
            f"{label} scope has {len(days)} days, expected {expected_count}"
        )
    if any(day_value >= sealed_test_start for day_value in days):
        raise ValueError(f"{label} scope crosses sealed-test boundary")
    missing = sorted(
        day_value
        for day_value in days
        if not (root / f"{day_value}.json").is_file()
    )
    if missing:
        raise FileNotFoundError(
            f"{label} ingestion incomplete: {len(missing)} required refs missing"
        )
    actual = {reference.stem for reference in root.glob("*.json")}
    unexpected = sorted(actual - days)
    if unexpected:
        raise ValueError(
            f"{label} namespace contains references outside the fixed "
            f"train/validation/context scope: {len(unexpected)}"
        )
    return tuple(sorted(days))


def read_futures_kline_day(
    root: Path,
    day_value: str,
    opened: set[Path],
    *,
    target_contract: str,
    oracle_scope: str,
    sealed_test_start: str,
    sealed_test_end: str,
    metadata_callback=None,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    if day_value >= sealed_test_start:
        raise ValueError(f"refusing sealed-test futures-kline date {day_value}")
    reference = (root / f"{day_value}.json").resolve()
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    opened.add(reference)
    if manifest.get("sequence") != {
        "start": utc_day_start_ms(day_value),
        "step": MINUTE_MS,
        "count": DAY_ROWS,
        "unit": "unix-ms",
    }:
        raise ValueError(f"invalid futures-kline sequence: {reference}")
    layout = manifest.get("layout", {})
    if layout.get("encoding") != FUTURES_KLINE_ENCODING \
            or layout.get("closeTimeOffsetMs") != FUTURES_CLOSE_OFFSET_MS \
            or layout.get("closed") is not True:
        raise ValueError(f"invalid futures-kline layout: {reference}")
    metadata = manifest.get("metadata", {})
    archive = f"BTCUSDT-1m-{day_value}.zip"
    csv_entry = f"BTCUSDT-1m-{day_value}.csv"
    expected_url = f"{FUTURES_KLINE_SOURCE_ROOT}/{archive}"
    try:
        counts = {
            name: int(metadata[name])
            for name in (
                "sourceArchiveBytes",
                "sourceCsvBytes",
                "sourceCsvRows",
                "sourceCsvHeaderRows",
                "observedGridRows",
                "liveGridRows",
                "noTradeGridRows",
                "missingGridRows",
                "outsideUtcDayRows",
                "offGridRows",
                "timestampAdjustedRows",
                "filledGridRows",
            )
        }
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid futures-kline quality counters: {reference}"
        ) from error
    if metadata.get("featureSchema") != FUTURES_KLINE_SCHEMA \
            or metadata.get("source") != "data.binance.vision" \
            or metadata.get("sourceDataset") != "futures/um/daily/klines" \
            or metadata.get("sourceArchiveUrl") != expected_url \
            or metadata.get("sourceArchiveChecksumUrl") \
            != f"{expected_url}.CHECKSUM" \
            or metadata.get("sourceArchiveChecksumAlgorithm") != "sha256" \
            or metadata.get("sourceArchiveChecksumFilename") != archive \
            or not re.fullmatch(
                r"[a-f0-9]{64}",
                str(metadata.get("sourceArchiveSha256", "")),
            ) \
            or metadata.get("sourceCsvEntry") != csv_entry \
            or metadata.get("sourceTimestampUnit") \
            not in {"millisecond", "microsecond"} \
            or metadata.get("market") != "usdm-futures" \
            or metadata.get("symbol") != "BTCUSDT" \
            or metadata.get("interval") != "1m" \
            or metadata.get("denseUtcDayAxis") is not True \
            or metadata.get("rowValidity") \
            != "official-source-row-present" \
            or metadata.get("liveObservationRule") \
            != "validMask && tradeCount > 0" \
            or metadata.get("closeAvailability") \
            != "openTime+59999ms" \
            or int(metadata.get("closeTimeOffsetMs", -1)) \
            != FUTURES_CLOSE_OFFSET_MS \
            or metadata.get("oracleTargetContract") != target_contract \
            or metadata.get("oracleScope") != oracle_scope \
            or metadata.get("sealedTestStart") != sealed_test_start \
            or metadata.get("sealedTestEnd") != sealed_test_end:
        raise ValueError(f"invalid futures-kline source contract: {reference}")
    if counts["sourceArchiveBytes"] < 1 \
            or counts["sourceCsvBytes"] < 1 \
            or counts["sourceCsvRows"] < 1 \
            or counts["sourceCsvHeaderRows"] not in {0, 1} \
            or min(
                counts["observedGridRows"],
                counts["liveGridRows"],
                counts["noTradeGridRows"],
                counts["missingGridRows"],
                counts["outsideUtcDayRows"],
                counts["offGridRows"],
            ) < 0 \
            or counts["observedGridRows"] + counts["missingGridRows"] \
            != DAY_ROWS \
            or counts["liveGridRows"] + counts["noTradeGridRows"] \
            != counts["observedGridRows"] \
            or counts["observedGridRows"] \
            + counts["outsideUtcDayRows"] + counts["offGridRows"] \
            != counts["sourceCsvRows"] \
            or counts["timestampAdjustedRows"] != 0 \
            or counts["filledGridRows"] != 0:
        raise ValueError(f"invalid futures-kline quality counters: {reference}")
    values, validity = read_derivatives_kline_columns(
        reference,
        FUTURES_KLINE_COLUMNS,
    )
    trade_count = values["tradeCount"]
    live = validity & (trade_count > 0)
    no_trade = validity & ~live
    if validity.shape != (DAY_ROWS,) \
            or int(np.count_nonzero(validity)) \
            != counts["observedGridRows"] \
            or int(np.count_nonzero(live)) != counts["liveGridRows"] \
            or int(np.count_nonzero(no_trade)) != counts["noTradeGridRows"]:
        raise ValueError(f"futures-kline payload counters disagree: {reference}")
    if metadata_callback is not None:
        metadata_callback(day_value, metadata)
    return values, validity


def read_spot_flow_day(
    root: Path,
    day_value: str,
    opened: set[Path],
    *,
    target_contract: str,
    oracle_scope: str,
    sealed_test_start: str,
    sealed_test_end: str,
    metadata_callback=None,
) -> dict[str, np.ndarray]:
    if day_value >= sealed_test_start:
        raise ValueError(f"refusing sealed-test Spot-flow date {day_value}")
    reference = (root / f"{day_value}.json").resolve()
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    if manifest.get("metadata", {}).get("oracleScope") != oracle_scope:
        raise ValueError(f"invalid Spot-flow oracle scope: {reference}")
    return read_trade_flow_day(
        root,
        day_value,
        opened,
        target_contract=target_contract,
        sealed_test_start=sealed_test_start,
        sealed_test_end=sealed_test_end,
        metadata_callback=metadata_callback,
    )


def safe_log_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    numerator = np.asarray(numerator, dtype=np.float64)
    denominator = np.asarray(denominator, dtype=np.float64)
    scale = np.maximum(np.maximum(numerator, denominator), 0.0)
    floor = np.maximum(scale * 1e-9, np.finfo(np.float64).tiny)
    return np.log((numerator + floor) / (denominator + floor))


def completed_spot_quote_features(
    previous_flow: dict[str, np.ndarray],
    current_flow: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return previous-completed-minute imbalance, activity surprise, volume."""
    for source in (previous_flow, current_flow):
        if any(
            name not in source
            or np.asarray(source[name]).shape != (SECOND_ROWS,)
            or not np.isfinite(source[name]).all()
            or bool((source[name] < 0).any())
            for name in SPOT_QUOTE_COLUMNS
        ):
            raise ValueError("invalid Spot quote-flow day")
    minute: dict[str, np.ndarray] = {}
    for name in SPOT_QUOTE_COLUMNS:
        minute[name] = np.concatenate((
            np.asarray(previous_flow[name], dtype=np.float64).reshape(
                DAY_ROWS, 60,
            ).sum(axis=1, dtype=np.float64),
            np.asarray(current_flow[name], dtype=np.float64).reshape(
                DAY_ROWS, 60,
            ).sum(axis=1, dtype=np.float64),
        ))
    sample_indexes = DAY_ROWS - 1 + np.arange(DAY_ROWS, dtype=np.int64)
    buy = minute["aggressiveBuyQuoteVolume"]
    sell = minute["aggressiveSellQuoteVolume"]
    total = buy + sell
    sampled_total = total[sample_indexes]
    sampled_imbalance = np.divide(
        buy[sample_indexes] - sell[sample_indexes],
        sampled_total,
        out=np.zeros(DAY_ROWS, dtype=np.float64),
        where=sampled_total > 0,
    )
    cumulative = np.concatenate(([0.0], np.cumsum(total, dtype=np.float64)))
    starts = sample_indexes - 59
    baseline = (
        cumulative[sample_indexes + 1] - cumulative[starts]
    ) / 60
    activity = safe_log_ratio(sampled_total, baseline)
    return sampled_imbalance, activity, sampled_total


def completed_cross_market_flow_features(
    previous_flow: dict[str, np.ndarray],
    current_flow: dict[str, np.ndarray],
    previous_futures: dict[str, np.ndarray],
    previous_futures_validity: np.ndarray,
    current_futures: dict[str, np.ndarray],
    current_futures_validity: np.ndarray,
    basis_features: np.ndarray,
) -> np.ndarray:
    spot_imbalance, spot_activity, spot_quote = (
        completed_spot_quote_features(previous_flow, current_flow)
    )
    if basis_features.shape != (DAY_ROWS, len(FUTURES_BASIS_FEATURE_NAMES)):
        raise ValueError("invalid futures-basis rows for cross-market flow")
    futures_quote = np.concatenate((
        np.asarray(previous_futures["quoteVolume"], dtype=np.float64),
        np.asarray(current_futures["quoteVolume"], dtype=np.float64),
    ))
    futures_valid = np.concatenate((
        np.asarray(previous_futures_validity, dtype=bool),
        np.asarray(current_futures_validity, dtype=bool),
    ))
    if futures_quote.shape != (2 * DAY_ROWS,) \
            or futures_valid.shape != (2 * DAY_ROWS,):
        raise ValueError("invalid futures rows for cross-market flow")
    sample_indexes = DAY_ROWS - 1 + np.arange(DAY_ROWS, dtype=np.int64)
    sampled_futures_quote = futures_quote[sample_indexes]
    sampled_futures_valid = futures_valid[sample_indexes]
    exact_activity_ratio = safe_log_ratio(sampled_futures_quote, spot_quote)
    exact_activity_ratio[~sampled_futures_valid] = 0.0
    basis_index = {
        name: index for index, name in enumerate(FUTURES_BASIS_FEATURE_NAMES)
    }
    futures_imbalance = basis_features[
        :, basis_index["futuresTakerQuoteImbalance1m"]
    ].astype(np.float64, copy=False)
    difference = futures_imbalance - spot_imbalance
    agreement = futures_imbalance * spot_imbalance
    approximate_activity = basis_features[
        :, basis_index["futuresSpotLogApproxQuoteVolumeRatio15m"]
    ].astype(np.float64, copy=False)
    row_observed = basis_features[
        :, basis_index["futuresRowCurrentObserved"]
    ] > 0.5
    no_trade = basis_features[
        :, basis_index["futuresNoTradeCurrent"]
    ] > 0.5
    source_state = np.where(row_observed, np.where(no_trade, 1.0, 2.0), 0.0)
    result = np.column_stack((
        spot_imbalance,
        np.abs(spot_imbalance),
        spot_activity,
        np.abs(spot_activity),
        difference,
        np.abs(difference),
        agreement,
        np.abs(agreement),
        exact_activity_ratio,
        np.abs(exact_activity_ratio),
        np.abs(approximate_activity),
        source_state,
    )).astype(np.float32, copy=False)
    if result.shape != (DAY_ROWS, len(SPOT_CROSS_FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError("invalid completed cross-market flow features")
    return result


def daily_features(
    day_value: str,
    candle_cache: MinuteCandleDayCache,
    futures_cache: FuturesKlineDayCache,
    spot_flow_cache: SpotFlowDayCache,
) -> np.ndarray:
    current = date.fromisoformat(day_value)
    previous_value = (current - timedelta(days=1)).isoformat()
    previous_spot = candle_cache.load(previous_value)
    current_spot = candle_cache.load(day_value)
    ohlcv = causal_ohlcv_features(
        completed_candle_windows(previous_spot, current_spot),
        completed_close_windows(previous_spot, current_spot),
    )
    previous_futures, previous_validity = futures_cache.load(previous_value)
    current_futures, current_validity = futures_cache.load(day_value)
    basis = causal_futures_basis_features(
        previous_futures,
        previous_validity,
        current_futures,
        current_validity,
        previous_spot,
        current_spot,
    )
    previous_flow = spot_flow_cache.load(previous_value)
    current_flow = spot_flow_cache.load(day_value)
    reconcile_spot_flow_with_minutes(
        previous_value,
        previous_spot,
        previous_flow,
    )
    reconcile_spot_flow_with_minutes(
        day_value,
        current_spot,
        current_flow,
    )
    cross = completed_cross_market_flow_features(
        previous_flow,
        current_flow,
        previous_futures,
        previous_validity,
        current_futures,
        current_validity,
        basis,
    )
    result = np.column_stack((ohlcv, basis, cross)).astype(
        np.float32,
        copy=False,
    )
    if result.shape != (DAY_ROWS, len(FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError(f"invalid combined futures-basis features for {day_value}")
    return result


def reconcile_spot_flow_with_minutes(
    day_value: str,
    candles: np.ndarray,
    flow: dict[str, np.ndarray],
) -> None:
    """Prove that second-binned aggTrades reconstruct Spot minute volume."""
    if candles.shape != (DAY_ROWS, 5):
        raise ValueError(f"invalid Spot minute candles for {day_value}")
    names = ("aggressiveBuyBaseVolume", "aggressiveSellBaseVolume")
    if any(
        name not in flow or np.asarray(flow[name]).shape != (SECOND_ROWS,)
        for name in names
    ):
        raise ValueError(f"invalid Spot base flow for {day_value}")
    flow_volume = (
        np.asarray(flow[names[0]], dtype=np.float64)
        + np.asarray(flow[names[1]], dtype=np.float64)
    ).reshape(DAY_ROWS, 60).sum(axis=1, dtype=np.float64)
    if not np.allclose(flow_volume, candles[:, 4], rtol=1e-10, atol=1e-10):
        difference = float(np.max(np.abs(flow_volume - candles[:, 4])))
        raise ValueError(
            f"Spot aggTrade/minute volume mismatch for {day_value}: {difference}"
        )


def load_split(
    split: str,
    segments: list[CausalSegment],
    candle_cache: MinuteCandleDayCache,
    futures_cache: FuturesKlineDayCache,
    spot_flow_cache: SpotFlowDayCache,
    access: AccessLog,
) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "validation"} \
            or any(segment.split != split for segment in segments):
        raise ValueError("futures-basis audit may load only train or validation")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, segment in enumerate(segments, start=1):
        values = daily_features(
            segment.target_file.stem,
            candle_cache,
            futures_cache,
            spot_flow_cache,
        )
        target_file = segment.target_file.resolve()
        _shard, day_targets = read_shard_array(
            target_file,
            "<f4",
            (DAY_ROWS, ACTION_COUNT),
        )
        access.target_files.add(target_file)
        start = segment.target_row_offset
        end = start + segment.count
        features.append(values[start:end])
        targets.append(np.asarray(day_targets[start:end], dtype=np.float32))
        if index % 25 == 0 or index == len(segments):
            print(
                f"{split}: {index}/{len(segments)} days loaded",
                file=sys.stderr,
                flush=True,
            )
    feature_rows = np.concatenate(features)
    target_rows = np.concatenate(targets)
    if feature_rows.shape != (target_rows.shape[0], len(FEATURE_NAMES)) \
            or not np.isfinite(feature_rows).all() \
            or not np.isfinite(target_rows).all() \
            or bool((target_rows < 0).any()) \
            or not np.allclose(
                target_rows.sum(axis=1),
                1,
                atol=2e-4,
                rtol=2e-4,
            ):
        raise ValueError(f"invalid {split} futures-basis rows")
    return feature_rows, target_rows


def fit_edges(
    features: np.ndarray,
    names: tuple[str, ...],
    bins: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    result: list[np.ndarray] = []
    for name, count in zip(names, bins, strict=True):
        if name in CATEGORICAL_FEATURE_EDGES:
            edges = CATEGORICAL_FEATURE_EDGES[name]
            if count != edges.size + 1:
                raise ValueError(f"invalid categorical bin count for {name}")
        else:
            edges = np.unique(np.quantile(
                features[:, FEATURE_INDEX[name]].astype(
                    np.float64,
                    copy=False,
                ),
                np.linspace(0, 1, count + 1)[1:-1],
            ))
        result.append(edges)
    return tuple(result)


def encode_cells(
    features: np.ndarray,
    names: tuple[str, ...],
    edges: tuple[np.ndarray, ...],
) -> tuple[np.ndarray, int]:
    if len(names) != len(edges):
        raise ValueError("feature names and quantile edges differ")
    ids = np.zeros(features.shape[0], dtype=np.int64)
    multiplier = 1
    for name, current_edges in zip(names, edges, strict=True):
        ids += np.searchsorted(
            current_edges,
            features[:, FEATURE_INDEX[name]],
            side="right",
        ) * multiplier
        multiplier *= current_edges.size + 1
    return ids, multiplier


def fit_probability_table(
    features: np.ndarray,
    targets: np.ndarray,
    names: tuple[str, ...],
    bins: tuple[int, ...],
    strength: float,
    prior: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], np.ndarray, np.ndarray]:
    edges = fit_edges(features, names, bins)
    ids, count = encode_cells(features, names, edges)
    sums, counts = sufficient_table(targets, ids, count)
    return edges, smoothed_table(sums, counts, prior, strength), counts


def candidate_scores(
    fit_features: np.ndarray,
    fit_targets: np.ndarray,
    calibration_features: np.ndarray,
    calibration_targets: np.ndarray,
    names: tuple[str, ...],
    bins: tuple[int, ...],
    prior: np.ndarray,
) -> tuple[
    float,
    float,
    dict[str, float],
    int,
    int,
    tuple[np.ndarray, ...],
    np.ndarray,
]:
    edges = fit_edges(fit_features, names, bins)
    fit_ids, cell_count = encode_cells(fit_features, names, edges)
    calibration_ids, _ = encode_cells(calibration_features, names, edges)
    sums, counts = sufficient_table(fit_targets, fit_ids, cell_count)
    tables = {
        strength: smoothed_table(sums, counts, prior, strength)
        for strength in PRIOR_STRENGTHS
    }
    scores = {
        str(int(strength)): mean_kl_indexed(
            calibration_targets,
            calibration_ids,
            table,
        )
        for strength, table in tables.items()
    }
    strength, score = min(
        ((float(name), value) for name, value in scores.items()),
        key=lambda item: (item[1], item[0]),
    )
    return (
        strength,
        score,
        scores,
        cell_count,
        int(np.count_nonzero(counts)),
        edges,
        tables[strength],
    )


def select_paired_regime(
    features: np.ndarray,
    targets: np.ndarray,
) -> tuple[
    TrainSelectedRegime,
    tuple[TrainSelectedRegime, ...],
    dict[str, object],
]:
    fit_end = int(features.shape[0] * (1 - CALIBRATION_FRACTION))
    calibration_start = fit_end + EMBARGO_ROWS
    if calibration_start >= features.shape[0]:
        raise ValueError("training embargo leaves no calibration rows")
    fit_features = features[:fit_end]
    fit_targets = targets[:fit_end]
    calibration_features = features[calibration_start:]
    calibration_targets = targets[calibration_start:]
    prior = normalized_mean(fit_targets)
    rows: list[dict[str, object]] = []
    selections: list[TrainSelectedRegime] = []
    choices: list[tuple[float, float, str, TrainSelectedRegime]] = []
    for regime in REGIMES:
        control = candidate_scores(
            fit_features,
            fit_targets,
            calibration_features,
            calibration_targets,
            regime.control_features,
            regime.control_bins,
            prior,
        )
        joint = candidate_scores(
            fit_features,
            fit_targets,
            calibration_features,
            calibration_targets,
            regime.joint_features,
            regime.joint_bins,
            prior,
        )
        control_ids, _ = encode_cells(
            calibration_features,
            regime.control_features,
            control[5],
        )
        joint_ids, _ = encode_cells(
            calibration_features,
            regime.joint_features,
            joint[5],
        )
        weight, backoff_kl = select_scalar(
            lambda value: mean_kl_stacked_indexed(
                calibration_targets,
                control_ids,
                control[6],
                joint_ids,
                joint[6],
                value,
            ),
            0,
            1,
        )
        improvement = control[1] - backoff_kl
        positive = improvement > 1e-9
        selection = TrainSelectedRegime(
            regime,
            control[0],
            joint[0],
            weight,
        )
        selections.append(selection)
        rows.append({
            "name": regime.name,
            "controlFeatures": list(regime.control_features),
            "signedFeatures": list(regime.directional_features),
            "controlKlByStrength": control[2],
            "jointKlByStrength": joint[2],
            "selectedControlKl": control[1],
            "selectedJointKl": joint[1],
            "trainSelectedJointWeight": weight,
            "trainBackoffKl": backoff_kl,
            "jointKlReductionFromPairedControl": improvement,
            "positiveJointImprovement": positive,
            "controlCells": control[3],
            "jointCells": joint[3],
            "populatedControlCells": control[4],
            "populatedJointCells": joint[4],
        })
        choices.append((
            0 if positive else 1,
            backoff_kl,
            regime.name,
            selection,
        ))
    winner = min(choices, key=lambda item: (item[0], item[1], item[2]))
    return winner[3], tuple(selections), {
        "fitRows": fit_end,
        "forecastHorizonEmbargoRows": EMBARGO_ROWS,
        "calibrationStartRow": calibration_start,
        "calibrationRows": features.shape[0] - calibration_start,
        "validationUsedForSelection": False,
        "candidates": rows,
        "selected": {
            "name": winner[3].regime.name,
            "controlPriorStrength": winner[3].control_strength,
            "jointPriorStrength": winner[3].joint_strength,
            "jointWeight": winner[3].joint_weight,
            "trainBackoffKl": winner[1],
            "positiveJointImprovement": winner[0] == 0,
        },
    }


def fit_all_regime_tables(
    features: np.ndarray,
    targets: np.ndarray,
    selections: tuple[TrainSelectedRegime, ...],
) -> tuple[FittedRegimeTables, ...]:
    expected = tuple(regime.name for regime in REGIMES)
    actual = tuple(selection.regime.name for selection in selections)
    if actual != expected:
        raise ValueError("train-selected regimes differ from the fixed candidates")
    prior = normalized_mean(targets)
    result: list[FittedRegimeTables] = []
    for selection in selections:
        regime = selection.regime
        control_edges, control_table, control_counts = fit_probability_table(
            features,
            targets,
            regime.control_features,
            regime.control_bins,
            selection.control_strength,
            prior,
        )
        joint_edges, joint_table, joint_counts = fit_probability_table(
            features,
            targets,
            regime.joint_features,
            regime.joint_bins,
            selection.joint_strength,
            prior,
        )
        result.append(FittedRegimeTables(
            selection,
            control_edges,
            control_table,
            control_counts,
            joint_edges,
            joint_table,
            joint_counts,
        ))
    return tuple(result)


def validation_regime_probabilities(
    fitted: FittedRegimeTables,
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    regime = fitted.selection.regime
    control_ids, _ = encode_cells(
        features,
        regime.control_features,
        fitted.control_edges,
    )
    joint_ids, _ = encode_cells(
        features,
        regime.joint_features,
        fitted.joint_edges,
    )
    control = fitted.control_table[control_ids]
    joint = fitted.joint_table[joint_ids]
    backed = convex_probability_fusion(
        control,
        joint,
        fitted.selection.joint_weight,
    )
    return control, joint, backed, control_ids, joint_ids


def complementarity_probability_pair(
    fitted: FittedRegimeTables,
    features: np.ndarray,
) -> tuple[str, np.ndarray, np.ndarray]:
    control, _joint, backed, _control_ids, _joint_ids = (
        validation_regime_probabilities(fitted, features)
    )
    return fitted.selection.regime.name, control, backed


if __name__ == "__main__":
    main()

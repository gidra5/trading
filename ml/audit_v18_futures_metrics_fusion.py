"""Leakage-safe USD-M positioning-metrics residual audit beyond frozen v18.

Binance USD-M five-minute metrics are consumed through the existing causal
feature layer, which exposes each observation only after one complete 5m lag
and carries missing values forward together with age/observed indicators.
Paired tables compare signed positioning information with binned absolute-value
and causal OHLCV controls.  Regime-specific bins, shrinkage, and joint/control
backoff are fitted and selected on training only with a 60-row forecast-horizon
embargo.  The separate signed and absolute-value bins do not constitute exact
magnitude conditioning, so results are interpreted only as complementarity.

Frozen v18 is evaluated once on development validation.  The first validation
half jointly selects one train-fitted regime and its scalar; the following 60
rows are embargoed; the remainder is the untouched within-audit holdout.  The
predeclared primary metric is raw T=.01 KL of the selected log-ratio residual
over v18.  Sealed-test reference JSON and payload objects are never opened.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from collections.abc import Iterable
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
    read_minute_day,
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
    embargoed_split_metrics,
    mean_kl_dense,
    validate_corpus_split_contract,
    validate_runtime_split_assignment,
)
from evaluate_joint_price_oracle_actions import resolve_device
from oracle_futures_metrics_features import (
    FUTURES_FEATURE_NAMES,
    METRIC_COLUMNS,
    METRIC_ROWS,
    causal_futures_metrics_features,
)
from trading_storage import (
    load_torch_checkpoint,
    read_derivatives_metrics_columns,
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
FUTURES_METRICS_REFERENCE_DIR = Path(
    "data/market/immutable/refs/derivatives-metrics/"
    "usdm-futures/btcusdt/5m"
)
FUTURES_METRICS_SCHEMA = "binance-usdm-futures-metrics-v1"
FUTURES_SOURCE_ROOT = (
    "https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT"
)
FEATURE_NAMES = OHLCV_FEATURE_NAMES + FUTURES_FEATURE_NAMES
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
PRIOR_STRENGTHS = (64.0, 256.0, 1_024.0)
CALIBRATION_FRACTION = 0.2
EXPECTED_CHECKPOINT_EPOCH = 7
EXPECTED_FUTURES_REFERENCE_DAYS = 420
DEVELOPMENT_SCREEN_GATE = 0.002
FORECAST_HORIZON_SECONDS = 3_600
MINUTE_SECONDS = 60
MINUTE_MS = MINUTE_SECONDS * 1_000
PREDICTION_CLOSE_OFFSET_MS = 999
EMBARGO_ROWS = FORECAST_HORIZON_SECONDS // MINUTE_SECONDS
if FORECAST_HORIZON_SECONDS % MINUTE_SECONDS:
    raise RuntimeError("oracle horizon does not align to minute rows")


@dataclass(frozen=True)
class PairedRegime:
    name: str
    control_features: tuple[str, ...]
    directional_features: tuple[str, ...]
    control_bins: tuple[int, ...]
    directional_bins: tuple[int, ...]

    @property
    def joint_features(self) -> tuple[str, ...]:
        return self.control_features + self.directional_features

    @property
    def joint_bins(self) -> tuple[int, ...]:
        return self.control_bins + self.directional_bins


# Each signed metric has its absolute-value counterpart in the paired control.
# Because both are quantile-binned separately, the joint table can still refine
# magnitude within a control cell; it is not an exact sign-only attribution.
REGIMES = (
    PairedRegime(
        "open-interest-pressure",
        (
            "return15m", "return60m", "rmsReturn60m",
            "absOpenInterestLogChange15m",
            "absOpenInterestLogChange1h",
            "sumOpenInterestObservationAge24h",
        ),
        ("openInterestLogChange15m", "openInterestLogChange1h"),
        (4, 5, 4, 3, 3, 2),
        (3, 3),
    ),
    PairedRegime(
        "open-interest-value-pressure",
        (
            "return15m", "return60m", "rmsReturn60m",
            "absOpenInterestValueLogChange15m",
            "absOpenInterestValueLogChange1h",
            "sumOpenInterestValueObservationAge24h",
        ),
        ("openInterestValueLogChange15m", "openInterestValueLogChange1h"),
        (4, 5, 4, 3, 3, 2),
        (3, 3),
    ),
    PairedRegime(
        "implied-mark-dislocation",
        (
            "return15m", "return60m", "rmsReturn60m",
            "meanLogRange15m",
            "absImpliedMarkPriceLogChange15m",
            "absImpliedMarkPriceLogChange1h",
        ),
        ("impliedMarkPriceLogChange15m", "impliedMarkPriceLogChange1h"),
        (4, 5, 4, 3, 3, 3),
        (3, 3),
    ),
    PairedRegime(
        "futures-taker-pressure",
        (
            "return15m", "return60m", "rmsReturn60m",
            "logVolume5mVs60m",
            "absTakerBuySellVolumeRatioLogLevel",
            "absTakerBuySellVolumeRatioLogChange1h",
        ),
        (
            "takerBuySellVolumeRatioLogLevel",
            "takerBuySellVolumeRatioLogChange1h",
        ),
        (4, 5, 4, 3, 3, 3),
        (3, 3),
    ),
    PairedRegime(
        "global-positioning",
        (
            "return60m", "rmsReturn60m", "meanLogRange60m",
            "absGlobalLongShortRatioLogLevel",
            "absGlobalLongShortRatioLogChange1h",
            "globalLongShortRatioObservationAge24h",
        ),
        ("globalLongShortRatioLogLevel", "globalLongShortRatioLogChange1h"),
        (5, 4, 3, 3, 3, 2),
        (3, 3),
    ),
    PairedRegime(
        "top-trader-divergence",
        (
            "return60m", "rmsReturn60m", "meanLogRange60m",
            "absTopTraderPositionMinusAccountLog",
            "absTopTraderAccountMinusGlobalLog",
        ),
        (
            "topTraderPositionMinusAccountLog",
            "topTraderAccountMinusGlobalLog",
        ),
        (5, 4, 3, 3, 3),
        (3, 3),
    ),
)


DIRECTIONAL_MAGNITUDE_CONTROLS = {
    "openInterestLogChange15m": "absOpenInterestLogChange15m",
    "openInterestLogChange1h": "absOpenInterestLogChange1h",
    "openInterestValueLogChange15m": "absOpenInterestValueLogChange15m",
    "openInterestValueLogChange1h": "absOpenInterestValueLogChange1h",
    "impliedMarkPriceLogChange15m": "absImpliedMarkPriceLogChange15m",
    "impliedMarkPriceLogChange1h": "absImpliedMarkPriceLogChange1h",
    "takerBuySellVolumeRatioLogLevel": "absTakerBuySellVolumeRatioLogLevel",
    "takerBuySellVolumeRatioLogChange1h": (
        "absTakerBuySellVolumeRatioLogChange1h"
    ),
    "globalLongShortRatioLogLevel": "absGlobalLongShortRatioLogLevel",
    "globalLongShortRatioLogChange1h": (
        "absGlobalLongShortRatioLogChange1h"
    ),
    "topTraderPositionMinusAccountLog": (
        "absTopTraderPositionMinusAccountLog"
    ),
    "topTraderAccountMinusGlobalLog": "absTopTraderAccountMinusGlobalLog",
}


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


@dataclass(frozen=True)
class FirstHalfComplementaritySelection:
    regime_name: str
    weight: float
    raw_kl: float


@dataclass
class AccessLog:
    target_files: set[Path]
    candle_files: set[Path]
    futures_metric_files: set[Path]


class MinuteCandleDayCache:
    def __init__(
        self,
        root: Path,
        opened: set[Path],
        *,
        sealed_test_start: str,
        maximum_days: int = 3,
    ) -> None:
        self.root = root
        self.opened = opened
        self.sealed_test_start = sealed_test_start
        self.maximum_days = max(2, int(maximum_days))
        self.days: OrderedDict[str, np.ndarray] = OrderedDict()

    def load(self, day_value: str) -> np.ndarray:
        if day_value >= self.sealed_test_start:
            raise ValueError(f"refusing sealed-test candle date {day_value}")
        cached = self.days.pop(day_value, None)
        if cached is not None:
            self.days[day_value] = cached
            return cached
        transient: dict[str, np.ndarray] = {}
        values = read_minute_day(self.root, day_value, transient, self.opened)
        self.days[day_value] = values
        while len(self.days) > self.maximum_days:
            self.days.popitem(last=False)
        return values


class FuturesMetricsDayCache:
    def __init__(
        self,
        root: Path,
        opened: set[Path],
        *,
        target_contract: str,
        sealed_test_start: str,
        sealed_test_end: str,
        maximum_days: int = 3,
    ) -> None:
        self.root = root
        self.opened = opened
        self.target_contract = target_contract
        self.sealed_test_start = sealed_test_start
        self.sealed_test_end = sealed_test_end
        self.maximum_days = max(2, int(maximum_days))
        self.days: OrderedDict[
            str,
            tuple[dict[str, np.ndarray], dict[str, np.ndarray]],
        ] = OrderedDict()
        self.recorded_days: set[str] = set()
        self.source_csv_rows = 0
        self.observed_grid_rows = 0
        self.missing_grid_rows = 0
        self.off_grid_rows = 0
        self.timestamp_adjusted_rows = 0
        self.source_archive_bytes = 0
        self.missing_value_counts = {name: 0 for name in METRIC_COLUMNS}

    def load(
        self,
        day_value: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        cached = self.days.pop(day_value, None)
        if cached is not None:
            self.days[day_value] = cached
            return cached
        values = read_futures_metrics_day(
            self.root,
            day_value,
            self.opened,
            target_contract=self.target_contract,
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
        source_rows = int(metadata["sourceCsvRows"])
        observed = int(metadata["observedGridRows"])
        outside = int(metadata["outsideUtcDayRows"])
        off_grid = int(metadata.get(
            "offGridRows", source_rows - observed - outside,
        ))
        self.source_csv_rows += source_rows
        self.observed_grid_rows += observed
        self.missing_grid_rows += int(metadata["missingGridRows"])
        self.off_grid_rows += off_grid
        self.timestamp_adjusted_rows += int(metadata["timestampAdjustedRows"])
        self.source_archive_bytes += int(metadata["sourceArchiveBytes"])
        for name in METRIC_COLUMNS:
            self.missing_value_counts[name] += int(
                metadata["missingValueCounts"][name]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    result = audit_v18_futures_metrics_fusion(
        arguments.plan,
        requested_device=arguments.device,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


def audit_v18_futures_metrics_fusion(
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
        raise ValueError("futures-metrics audit requires sealed-never-load")
    model_config = plan["model"]
    training = plan["training"]
    resolved_training = resolve_training_config(training)
    if not bool(resolved_training.get("policyOnly", False)):
        raise ValueError("futures-metrics audit requires policy-only v18")
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
    metrics_root = require_under(
        (repo_root / FUTURES_METRICS_REFERENCE_DIR).resolve(),
        repo_root / "data" / "market" / "immutable" / "refs" / "derivatives-metrics",
        "futuresMetricsReferenceDir",
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

    # This is deliberately before any feature/target payload read or model
    # load.  A partial ingestion cannot accidentally produce a biased audit.
    required_metric_days = validate_futures_reference_coverage(
        metrics_root,
        segments,
        sealed_test_start=sealed_test_start,
    )

    access = AccessLog(set(), set(), set())
    candle_cache = MinuteCandleDayCache(
        minute_history_root,
        access.candle_files,
        sealed_test_start=sealed_test_start,
    )
    metric_cache = FuturesMetricsDayCache(
        metrics_root,
        access.futures_metric_files,
        target_contract=target_root.name,
        sealed_test_start=sealed_test_start,
        sealed_test_end=sealed_test_end,
    )
    print(
        "Loading causal train/validation USD-M metrics and OHLCV controls; "
        "sealed test remains untouched.",
        file=sys.stderr,
        flush=True,
    )
    train_features, train_targets = load_split(
        "train",
        segments["train"],
        candle_cache,
        metric_cache,
        access,
    )
    validation_features, validation_targets = load_split(
        "validation",
        segments["validation"],
        candle_cache,
        metric_cache,
        access,
    )
    required_metric_files = {
        (metrics_root / f"{day_value}.json").resolve()
        for day_value in required_metric_days
    }
    if access.futures_metric_files != required_metric_files:
        raise RuntimeError("futures-metrics access differs from preflight scope")
    if access.target_files & test_target_files \
            or test_days & {path.stem for path in access.candle_files} \
            or test_days & {path.stem for path in access.futures_metric_files}:
        raise RuntimeError("sealed-test reference access detected")

    train_selected, train_candidates, selection_report = select_paired_regime(
        train_features,
        train_targets,
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
        raise RuntimeError("futures-metrics fusion rows are misaligned")
    if not np.array_equal(model_targets, validation_targets):
        raise RuntimeError("v18 and futures-metrics target order differs")
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
            "pairedAbsoluteMagnitudeAndOhlcvControl": (
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
    ratio_metrics = primary_development_screen(
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
        "schemaVersion": 2,
        "audit": "v18-plus-causal-usdm-futures-metrics-residual",
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
            "futuresMetricReferencesOpened": len(
                access.futures_metric_files
            ),
            "causalOneMinuteControlReferencesOpened": len(access.candle_files),
            "causalOneSecondV18ReferencesOpened": len(inference_history),
            "testTargetReferencesOpened": 0,
            "testTargetPayloadsOpened": 0,
            "testCandleReferencesOpened": 0,
            "testCandlePayloadsOpened": 0,
            "testFuturesMetricReferencesOpened": 0,
            "testFuturesMetricPayloadsOpened": 0,
            "modelWeightsChanged": False,
            "trainingStarted": False,
            "inferenceDevice": str(device),
        },
        "sourceQuality": {
            "featureSchema": FUTURES_METRICS_SCHEMA,
            "availabilityLagMinutes": 5,
            "requiredReferenceDays": len(required_metric_days),
            "sourceDays": len(metric_cache.recorded_days),
            "sourceCsvRows": metric_cache.source_csv_rows,
            "observedGridRows": metric_cache.observed_grid_rows,
            "missingGridRows": metric_cache.missing_grid_rows,
            "offGridRows": metric_cache.off_grid_rows,
            "timestampAdjustedRows": metric_cache.timestamp_adjusted_rows,
            "sourceArchiveBytes": metric_cache.source_archive_bytes,
            "missingValueCounts": metric_cache.missing_value_counts,
        },
        "estimator": {
            "featureCount": len(FEATURE_NAMES),
            "ohlcvControlFeatureCount": len(OHLCV_FEATURE_NAMES),
            "futuresMetricFeatureCount": len(FUTURES_FEATURE_NAMES),
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
            "trainOnlyTableSelectionDiagnostic": selection_report,
        },
        "validationSplit": {
            "rows": counts["validation"],
            "scalarFitRows": split_at,
            "forecastHorizonEmbargoRows": EMBARGO_ROWS,
            "embargoTimestampStart": int(timestamps[split_at]),
            "embargoTimestampEnd": int(timestamps[holdout_start - 1]),
            "withinAuditHoldoutRows": counts["validation"] - holdout_start,
            "globallyFresh": False,
            "role": "reused-development-validation",
            "scalarFitTimestampStart": int(timestamps[0]),
            "scalarFitTimestampEnd": int(timestamps[split_at - 1]),
            "withinAuditHoldoutTimestampStart": int(timestamps[holdout_start]),
            "withinAuditHoldoutTimestampEnd": int(timestamps[-1]),
        },
        "raw01Kl": {
            "v18": v18_metrics,
            "allRegimeComplementarityCandidates": candidate_reports,
            "logRatioResidualFusionPrimary": ratio_metrics,
            "trainOnlySelectedRegimeDiagnostic": train_selected_diagnostic,
        },
        "selectionContract": {
            "referenceCoverageCheckedBeforePayloadAccess": True,
            "allRegimeBinsShrinkageAndBackoffSelectedOnTrainOnly": True,
            "trainOnlyDiagnosticRequiresPositiveJointImprovementWhenAvailable": (
                True
            ),
            "primaryFusionPredeclared": "logRatioResidualFusionPrimary",
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


def validate_segment_clock_alignment(
    segments: dict[str, list[CausalSegment]],
) -> None:
    """Bind each target slice to the exact minute prediction-time axis."""
    if set(segments) != {"train", "validation", "test"}:
        raise ValueError("runtime segment splits differ from the fixed contract")
    for split, current_segments in segments.items():
        for segment in current_segments:
            offset = int(segment.target_row_offset)
            count = int(segment.count)
            expected_time = (
                utc_day_start_ms(segment.target_file.stem)
                + PREDICTION_CLOSE_OFFSET_MS
                + offset * MINUTE_MS
            )
            if segment.split != split \
                    or int(segment.step_ms) != MINUTE_MS \
                    or offset < 0 \
                    or count < 1 \
                    or offset + count > DAY_ROWS \
                    or int(segment.prediction_time_start) != expected_time:
                raise ValueError(
                    "runtime segment clock/target slice is misaligned: "
                    f"{segment.target_file}"
                )


def validate_futures_reference_coverage(
    root: Path,
    segments: dict[str, list[CausalSegment]],
    *,
    sealed_test_start: str,
    expected_count: int = EXPECTED_FUTURES_REFERENCE_DAYS,
) -> tuple[str, ...]:
    days = required_feature_days(segments)
    if len(days) != expected_count:
        raise ValueError(
            f"futures-metrics scope has {len(days)} days, expected {expected_count}"
        )
    if any(day_value >= sealed_test_start for day_value in days):
        raise ValueError("futures-metrics scope crosses sealed-test boundary")
    missing = sorted(
        day_value
        for day_value in days
        if not (root / f"{day_value}.json").is_file()
    )
    if missing:
        raise FileNotFoundError(
            f"futures-metrics ingestion incomplete: {len(missing)} required refs missing"
        )
    actual = {reference.stem for reference in root.glob("*.json")}
    unexpected = sorted(actual - days)
    if unexpected:
        raise ValueError(
            "futures-metrics namespace contains references outside the fixed "
            f"train/validation/context scope: {len(unexpected)}"
        )
    return tuple(sorted(days))


def read_futures_metrics_day(
    root: Path,
    day_value: str,
    opened: set[Path],
    *,
    target_contract: str,
    sealed_test_start: str,
    sealed_test_end: str,
    metadata_callback=None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    if day_value >= sealed_test_start:
        raise ValueError(f"refusing sealed-test futures-metrics date {day_value}")
    reference = (root / f"{day_value}.json").resolve()
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    opened.add(reference)
    if manifest.get("sequence") != {
        "start": utc_day_start_ms(day_value),
        "step": 300_000,
        "count": METRIC_ROWS,
        "unit": "unix-ms",
    }:
        raise ValueError(f"invalid futures-metrics sequence: {reference}")
    if manifest.get("layout", {}).get("encoding") \
            != "derivatives-metrics-columnar-v1":
        raise ValueError(f"invalid futures-metrics layout: {reference}")
    metadata = manifest.get("metadata", {})
    archive = f"BTCUSDT-metrics-{day_value}.zip"
    expected_url = f"{FUTURES_SOURCE_ROOT}/{archive}"
    expected_scope = {
        "train-or-validation-target",
        "predecessor-context",
    }
    if metadata.get("featureSchema") != FUTURES_METRICS_SCHEMA \
            or metadata.get("source") != "data.binance.vision" \
            or metadata.get("sourceDataset") != "futures/um/daily/metrics" \
            or metadata.get("sourceArchiveUrl") != expected_url \
            or not re.fullmatch(
                r"[a-f0-9]{64}",
                str(metadata.get("sourceArchiveSha256", "")),
            ) \
            or int(metadata.get("sourceArchiveBytes", -1)) <= 0 \
            or metadata.get("sourceCsvEntry") \
            != f"BTCUSDT-metrics-{day_value}.csv" \
            or int(metadata.get("sourceCsvBytes", -1)) <= 0 \
            or metadata.get("market") != "usdm-futures" \
            or metadata.get("symbol") != "BTCUSDT" \
            or metadata.get("interval") != "5m" \
            or metadata.get("denseUtcDayAxis") is not True \
            or int(metadata.get("availabilityLagMs", -1)) != 300_000 \
            or metadata.get("oracleTargetContract") != target_contract \
            or metadata.get("oracleScope") not in expected_scope \
            or metadata.get("sealedTestStart") != sealed_test_start \
            or metadata.get("sealedTestEnd") != sealed_test_end:
        raise ValueError(f"invalid futures-metrics source contract: {reference}")
    source_rows = int(metadata.get("sourceCsvRows", -1))
    observed = int(metadata.get("observedGridRows", -1))
    missing = int(metadata.get("missingGridRows", -1))
    outside = int(metadata.get("outsideUtcDayRows", -1))
    adjusted = int(metadata.get("timestampAdjustedRows", -1))
    off_grid = int(metadata.get(
        "offGridRows", source_rows - observed - outside,
    ))
    missing_counts = metadata.get("missingValueCounts")
    if not isinstance(missing_counts, dict) \
            or set(missing_counts) != set(METRIC_COLUMNS) \
            or observed + missing != METRIC_ROWS \
            or min(
                source_rows, observed, missing, outside, off_grid, adjusted,
            ) < 0 \
            or source_rows != observed + outside + off_grid \
            or adjusted != 0 \
            or any(
                not missing <= int(missing_counts[name]) <= METRIC_ROWS
                for name in METRIC_COLUMNS
            ):
        raise ValueError(f"invalid futures-metrics quality counters: {reference}")
    values, validity = read_derivatives_metrics_columns(reference, METRIC_COLUMNS)
    for name in METRIC_COLUMNS:
        if values[name].shape != (METRIC_ROWS,) \
                or validity[name].shape != (METRIC_ROWS,) \
                or int(np.count_nonzero(~validity[name])) \
                != int(missing_counts[name]):
            raise ValueError(
                f"futures-metrics payload counters disagree: {reference}"
            )
    if metadata_callback is not None:
        metadata_callback(day_value, metadata)
    return values, validity


def daily_features(
    day_value: str,
    candle_cache: MinuteCandleDayCache,
    metric_cache: FuturesMetricsDayCache,
) -> np.ndarray:
    current = date.fromisoformat(day_value)
    previous_value = (current - timedelta(days=1)).isoformat()
    previous_candles = candle_cache.load(previous_value)
    current_candles = candle_cache.load(day_value)
    ohlcv = causal_ohlcv_features(
        completed_candle_windows(previous_candles, current_candles),
        completed_close_windows(previous_candles, current_candles),
    )
    previous_values, previous_validity = metric_cache.load(previous_value)
    current_values, current_validity = metric_cache.load(day_value)
    futures = causal_futures_metrics_features(
        previous_values,
        previous_validity,
        current_values,
        current_validity,
    )
    result = np.column_stack((ohlcv, futures)).astype(np.float32, copy=False)
    if result.shape != (DAY_ROWS, len(FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError(f"invalid combined futures features for {day_value}")
    return result


def load_split(
    split: str,
    segments: list[CausalSegment],
    candle_cache: MinuteCandleDayCache,
    metric_cache: FuturesMetricsDayCache,
    access: AccessLog,
) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "validation"} \
            or any(segment.split != split for segment in segments):
        raise ValueError("futures audit may load only train or validation")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, segment in enumerate(segments, start=1):
        values = daily_features(
            segment.target_file.stem,
            candle_cache,
            metric_cache,
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
        raise ValueError(f"invalid {split} futures-metrics rows")
    return feature_rows, target_rows


def fit_edges(
    features: np.ndarray,
    names: tuple[str, ...],
    bins: tuple[int, ...],
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.unique(np.quantile(
            features[:, FEATURE_INDEX[name]].astype(np.float64, copy=False),
            np.linspace(0, 1, count + 1)[1:-1],
        ))
        for name, count in zip(names, bins, strict=True)
    )


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


def select_first_half_complementarity(
    targets: np.ndarray,
    base_probabilities: np.ndarray,
    candidates: Iterable[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[
    FirstHalfComplementaritySelection,
    tuple[FirstHalfComplementaritySelection, ...],
]:
    """Select regime and residual scalar using only the supplied fit rows."""
    if targets.shape != base_probabilities.shape or targets.ndim != 2:
        raise ValueError("first-half complementarity rows are incompatible")
    rows: list[FirstHalfComplementaritySelection] = []
    seen: set[str] = set()
    for name, control, backed in candidates:
        if not name or name in seen \
                or control.shape != targets.shape \
                or backed.shape != targets.shape:
            raise ValueError("invalid first-half complementarity candidate")
        seen.add(name)
        weight, raw_kl = select_scalar(
            lambda value: mean_kl_dense(
                targets,
                log_ratio_feature_fusion(
                    base_probabilities,
                    backed,
                    control,
                    value,
                ),
            ),
            0,
            4,
        )
        rows.append(FirstHalfComplementaritySelection(name, weight, raw_kl))
    if not rows:
        raise ValueError("first-half complementarity candidates are empty")
    selected = min(rows, key=lambda row: (row.raw_kl, row.regime_name))
    return selected, tuple(rows)


def primary_development_screen(
    selected: FirstHalfComplementaritySelection,
    candidate_metrics: dict[str, dict[str, float | bool]],
) -> dict[str, object]:
    """Gate only the regime fixed by the validation first-half selection."""
    if selected.regime_name not in candidate_metrics:
        raise ValueError("selected complementarity regime has no diagnostics")
    result: dict[str, object] = {
        "selectedRegime": selected.regime_name,
        **candidate_metrics[selected.regime_name],
    }
    reduction = float(result["secondHalfKlReductionFromV18"])
    result["passesWithinAudit0.002Screen"] = bool(
        reduction >= DEVELOPMENT_SCREEN_GATE
    )
    return result


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


if __name__ == "__main__":
    main()

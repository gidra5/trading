"""Leakage-safe residual audit of true Spot aggressor flow beyond frozen v18.

The feature table is built from Binance Spot BTCUSDT aggTrades, aggregated into
immutable one-second shards. Windows end at the currently closed second for
each minute-spaced oracle target. Paired controls expose the same activity or
magnitude without true direction; train-only selection chooses a regime,
shrinkage, and joint/control backoff. The predeclared primary validation test
is a log-ratio residual on frozen v18. Its scalar is fitted on validation half
one and scored on within-audit holdout half two. This validation region is a
reused development set across the broader project, not a globally fresh test.
Sealed test references and payloads are never opened.
"""

from __future__ import annotations

import argparse
from collections import OrderedDict
from dataclasses import dataclass
from datetime import date, timedelta
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch

from audit_causal_oracle_ohlcv_predictability import (
    ACTION_COUNT,
    FORECAST_HORIZON,
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
from audit_v18_ohlcv_fusion import (
    AuditedMarketCloseCache,
    AuditedOracleTargetCache,
    fusion_metrics,
    log_ratio_feature_fusion,
    validation_timestamps,
)
from audit_v18_on_v28_validation_rows import collect_sequence_core_rows
from audit_v18_second_microstructure_fusion import (
    SecondDayCache,
    aggregate_minutes,
)
from evaluate_joint_price_oracle_actions import resolve_device
from oracle_trade_flow_features import (
    FLOW_FEATURE_NAMES,
    SECOND_ROWS,
    TRADE_FLOW_COLUMNS,
    causal_trade_flow_features,
    safe_divide,
)
from trading_storage import (
    load_torch_checkpoint,
    read_shard_array,
    read_trade_flow_columns,
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
TRADE_FLOW_REFERENCE_DIR = Path(
    "data/market/immutable/refs/trade-flow/spot-btcusdt/btcusdt/1s"
)
TRADE_FLOW_SCHEMA = "binance-spot-agg-trade-flow-v1"
TRADE_FLOW_SOURCE_ROOT = (
    "https://data.binance.vision/data/spot/daily/aggTrades/BTCUSDT"
)
CORPUS_SPLIT_CONTRACT = Path(
    "ml/corpus-contracts/oracle-hindsight-bot-71391c44-split-v1.json"
)
SECOND_MS = 1_000
MINUTE_SECONDS = 60
FEATURE_NAMES = OHLCV_FEATURE_NAMES + (
    "tickRuleSignedVolume60s",
    "absTickRuleSignedVolume60s",
) + FLOW_FEATURE_NAMES
FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
PRIOR_STRENGTHS = (64.0, 256.0, 1_024.0)
CALIBRATION_FRACTION = 0.2
EXPECTED_CHECKPOINT_EPOCH = 7
DEVELOPMENT_SCREEN_GATE = 0.002
EMBARGO_ROWS = FORECAST_HORIZON // MINUTE_SECONDS
if FORECAST_HORIZON % MINUTE_SECONDS != 0:
    raise RuntimeError("oracle horizon does not align to minute target rows")


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


REGIMES = (
    PairedRegime(
        "true-vs-price-sign-proxy",
        (
            "return5m", "return60m", "rmsReturn60m",
            "tickRuleSignedVolume60s", "absQuoteImbalance60s",
        ),
        ("quoteImbalance60s",),
        (4, 5, 4, 5, 3),
        (3,),
    ),
    PairedRegime(
        "current-taker-pressure",
        (
            "return1m", "return15m", "rmsReturn15m",
            "logQuoteRate1sVs60m", "absQuoteImbalance1s",
            "absLastAggressorSide1s",
        ),
        ("quoteImbalance1s", "lastAggressorSide1s"),
        (4, 5, 4, 4, 3, 2),
        (3, 3),
    ),
    PairedRegime(
        "burst-persistence",
        (
            "return5m", "return60m", "rmsReturn60m",
            "quoteShare5sOf60s", "absQuoteImbalance5s",
            "absQuoteImbalance60s",
        ),
        ("quoteImbalance5s", "quoteImbalance60s"),
        (3, 4, 3, 3, 3, 3),
        (3, 3),
    ),
    PairedRegime(
        "slow-taker-pressure",
        (
            "return15m", "return60m", "rmsReturn60m",
            "logQuoteRate5mVs60m", "absQuoteImbalance5m",
            "absTradeCountImbalance5m",
        ),
        ("quoteImbalance5m", "tradeCountImbalance5m"),
        (3, 4, 3, 4, 3, 3),
        (3, 3),
    ),
    PairedRegime(
        "order-count-persistence",
        (
            "return5m", "return60m", "rmsReturn60m",
            "rawPerAggregate60s", "absTradeCountImbalance5s",
            "absTradeCountImbalance60s",
        ),
        ("tradeCountImbalance5s", "tradeCountImbalance60s"),
        (3, 4, 3, 3, 3, 3),
        (3, 3),
    ),
    PairedRegime(
        "large-taker-side",
        (
            "return15m", "return60m", "rmsReturn60m",
            "absQuantitySquaredSkew60s", "absMaxAggregateSkew60s",
        ),
        ("quantitySquaredSkew60s", "maxAggregateSkew60s"),
        (3, 4, 3, 3, 3),
        (3, 3),
    ),
    PairedRegime(
        "timing-vwap-pressure",
        (
            "return1m", "return15m", "rmsReturn15m",
            "absVwapGap60s", "absArrivalCentroidGap60s",
        ),
        ("signedVwapGap60s", "signedArrivalCentroidGap60s"),
        (4, 5, 4, 3, 3),
        (3, 3),
    ),
)


@dataclass
class AccessLog:
    target_files: set[Path]
    candle_files: set[Path]
    trade_flow_files: set[Path]


class TradeFlowDayCache:
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
        self.days: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        self.sentinel_rows = 0
        self.false_best_match_rows = 0
        self.days_with_sentinels: set[str] = set()
        self.days_with_false_best_match: set[str] = set()
        self.recorded_days: set[str] = set()
        self.source_csv_rows = 0
        self.valid_aggregate_rows = 0
        self.raw_constituent_rows = 0
        self.source_archive_bytes = 0

    def load(self, day_value: str) -> dict[str, np.ndarray]:
        cached = self.days.pop(day_value, None)
        if cached is not None:
            self.days[day_value] = cached
            return cached
        values = read_trade_flow_day(
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
        sentinels = int(metadata.get("invalidSentinelRows", -1))
        false_best = int(metadata.get("falseBestPriceMatchRows", -1))
        if sentinels < 0 or false_best < 0:
            raise ValueError(f"missing source-quality counters for {day_value}")
        self.sentinel_rows += sentinels
        self.false_best_match_rows += false_best
        if sentinels:
            self.days_with_sentinels.add(day_value)
        if false_best:
            self.days_with_false_best_match.add(day_value)
        source_rows = int(metadata["sourceCsvRows"])
        self.source_csv_rows += source_rows
        self.valid_aggregate_rows += source_rows - sentinels
        self.raw_constituent_rows += (
            int(metadata["lastTradeId"])
            - int(metadata["firstTradeId"])
            + 1
        )
        self.source_archive_bytes += int(metadata["sourceArchiveBytes"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    report = audit_v18_trade_flow_fusion(
        arguments.plan,
        requested_device=arguments.device,
    )
    print(json.dumps(report, indent=2, allow_nan=False))


def audit_v18_trade_flow_fusion(
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
        raise ValueError("trade-flow audit requires sealed-never-load")
    model_config = plan["model"]
    training = plan["training"]
    resolved_training = resolve_training_config(training)
    if not bool(resolved_training.get("policyOnly", False)):
        raise ValueError("trade-flow audit requires policy-only v18")
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
    trade_flow_root = require_under(
        (repo_root / TRADE_FLOW_REFERENCE_DIR).resolve(),
        repo_root / "data" / "market" / "immutable" / "refs" / "trade-flow",
        "tradeFlowReferenceDir",
    )
    run_dir = require_under(
        resolve(repo_root, Path(plan["runDir"])), storage.runs, "runDir",
    )
    target_files = sorted(target_root.glob("*.json"))
    corpus_contract = validate_corpus_split_contract(
        repo_root, target_root, target_files,
    )
    segments = split_and_purge(target_files)
    validate_runtime_split_assignment(
        target_files, segments, corpus_contract,
    )
    counts = {
        split: sum(segment.count for segment in segments[split])
        for split in ("train", "validation", "test")
    }
    test_target_files = {
        segment.target_file.resolve() for segment in segments["test"]
    }
    test_days = {segment.target_file.stem for segment in segments["test"]}
    sealed_test_start = str(corpus_contract["test"]["first"])
    if test_days != set(day_range(
        sealed_test_start, str(corpus_contract["test"]["last"]),
    )):
        raise RuntimeError("runtime split differs from immutable corpus contract")

    access = AccessLog(set(), set(), set())
    candle_cache = SecondDayCache(history_root, access.candle_files)
    flow_cache = TradeFlowDayCache(
        trade_flow_root,
        access.trade_flow_files,
        target_contract=target_root.name,
        sealed_test_start=sealed_test_start,
        sealed_test_end=str(corpus_contract["test"]["last"]),
    )
    print(
        "Loading causal train/validation aggressor flow; test remains sealed.",
        file=sys.stderr,
        flush=True,
    )
    train_features, train_targets = load_split(
        "train", segments["train"], candle_cache, flow_cache, access,
    )
    validation_features, validation_targets = load_split(
        "validation", segments["validation"], candle_cache, flow_cache, access,
    )
    if access.target_files & test_target_files:
        raise RuntimeError("sealed test target was opened")
    if test_days & {path.stem for path in access.candle_files}:
        raise RuntimeError("sealed test candle was opened")
    if test_days & {path.stem for path in access.trade_flow_files}:
        raise RuntimeError("sealed test trade flow was opened")

    selected, selection_report = select_paired_regime(
        train_features, train_targets,
    )
    regime, control_strength, joint_strength, joint_weight = selected
    prior = normalized_mean(train_targets)
    control_edges, control_table, control_counts = fit_probability_table(
        train_features,
        train_targets,
        regime.control_features,
        regime.control_bins,
        control_strength,
        prior,
    )
    joint_edges, joint_table, joint_counts = fit_probability_table(
        train_features,
        train_targets,
        regime.joint_features,
        regime.joint_bins,
        joint_strength,
        prior,
    )
    validation_control_ids, _ = encode_cells(
        validation_features, regime.control_features, control_edges,
    )
    validation_joint_ids, _ = encode_cells(
        validation_features, regime.joint_features, joint_edges,
    )
    control_probabilities = control_table[validation_control_ids]
    joint_probabilities = joint_table[validation_joint_ids]
    residual_probabilities = convex_probability_fusion(
        control_probabilities, joint_probabilities, joint_weight,
    )

    checkpoint_file = run_dir / "checkpoints" / "best.json"
    checkpoint = load_torch_checkpoint(
        checkpoint_file, map_location="cpu", weights_only=False,
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
        "Running one frozen v18 validation inference pass.",
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
        control_probabilities,
        joint_probabilities,
        residual_probabilities,
    )):
        raise RuntimeError("trade-flow fusion rows are misaligned")
    if not np.array_equal(model_targets, validation_targets):
        raise RuntimeError("v18 and trade-flow target order differs")
    expected_validation_files = {
        segment.target_file.resolve() for segment in segments["validation"]
    }
    if inference_targets != expected_validation_files \
            or inference_targets & test_target_files \
            or test_days & {path.stem for path in inference_history}:
        raise RuntimeError("unexpected v18 inference reference access")

    probabilities = softmax(logits)
    split_at = probabilities.shape[0] // 2
    holdout_start = split_at + EMBARGO_ROWS
    if holdout_start >= probabilities.shape[0]:
        raise RuntimeError("validation embargo leaves no holdout rows")
    ratio_weight, ratio_fit_kl = select_scalar(
        lambda value: mean_kl_dense(
            validation_targets[:split_at],
            log_ratio_feature_fusion(
                probabilities[:split_at],
                residual_probabilities[:split_at],
                control_probabilities[:split_at],
                value,
            ),
        ),
        0,
        4,
    )
    convex_weight, convex_fit_kl = select_scalar(
        lambda value: mean_kl_dense(
            validation_targets[:split_at],
            convex_probability_fusion(
                probabilities[:split_at],
                residual_probabilities[:split_at],
                value,
            ),
        ),
        0,
        1,
    )
    ratio_fused = log_ratio_feature_fusion(
        probabilities,
        residual_probabilities,
        control_probabilities,
        ratio_weight,
    )
    convex_fused = convex_probability_fusion(
        probabilities, residual_probabilities, convex_weight,
    )
    v18_metrics = embargoed_split_metrics(
        validation_targets, probabilities, split_at, holdout_start,
    )
    ratio_metrics = fusion_metrics(
        ratio_weight,
        ratio_fit_kl,
        embargoed_split_metrics(
            validation_targets, ratio_fused, split_at, holdout_start,
        ),
        v18_metrics,
    )
    convex_metrics = fusion_metrics(
        convex_weight,
        convex_fit_kl,
        embargoed_split_metrics(
            validation_targets, convex_fused, split_at, holdout_start,
        ),
        v18_metrics,
    )
    ratio_metrics["passesWithinAudit0.002Screen"] = bool(
        ratio_metrics["secondHalfKlReductionFromV18"]
        >= DEVELOPMENT_SCREEN_GATE
    )
    convex_metrics["passesWithinAudit0.002Screen"] = bool(
        convex_metrics["secondHalfKlReductionFromV18"]
        >= DEVELOPMENT_SCREEN_GATE
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
        "audit": "v18-plus-causal-spot-agg-trade-flow-residual",
        "corpusSplitContract": {
            "file": CORPUS_SPLIT_CONTRACT.as_posix(),
            "targetContract": corpus_contract["targetContract"],
            "referenceCount": corpus_contract["referenceCount"],
            "referenceFilenameSha256": corpus_contract[
                "referenceFilenameSha256"
            ],
            "sealedTestStart": corpus_contract["test"]["first"],
            "sealedTestEnd": corpus_contract["test"]["last"],
        },
        "accessContract": {
            "trainTargetReferenceFilesOpened": len({
                segment.target_file.resolve() for segment in segments["train"]
            }),
            "validationTargetReferenceFilesOpened": len(expected_validation_files),
            "tradeFlowFeatureReferencesOpened": len(access.trade_flow_files),
            "causalOneSecondFeatureReferencesOpened": len(access.candle_files),
            "causalOneSecondV18ReferencesOpened": len(inference_history),
            "testTargetReferencesOpened": 0,
            "testTargetPayloadsOpened": 0,
            "testCandleReferencesOpened": 0,
            "testCandlePayloadsOpened": 0,
            "testTradeFlowReferencesOpened": 0,
            "testTradeFlowPayloadsOpened": 0,
            "modelWeightsChanged": False,
            "trainingStarted": False,
            "inferenceDevice": str(device),
        },
        "sourceQuality": {
            "aggregateIdGapRows": 0,
            "sourceDays": len(flow_cache.recorded_days),
            "sourceCsvRows": flow_cache.source_csv_rows,
            "validAggregateRows": flow_cache.valid_aggregate_rows,
            "rawConstituentTradeRows": flow_cache.raw_constituent_rows,
            "sourceArchiveBytes": flow_cache.source_archive_bytes,
            "invalidSentinelRowsFiltered": flow_cache.sentinel_rows,
            "daysWithInvalidSentinels": sorted(flow_cache.days_with_sentinels),
            "falseBestPriceMatchRows": flow_cache.false_best_match_rows,
            "daysWithFalseBestPriceMatch": sorted(
                flow_cache.days_with_false_best_match
            ),
            "completedSecondLabelSemantics": (
                "open+999ms labels the complete half-open one-second interval"
            ),
        },
        "estimator": {
            "featureCount": len(FEATURE_NAMES),
            "tradeFlowFeatureCount": len(FLOW_FEATURE_NAMES),
            "selectedRegime": regime.name,
            "controlFeatures": list(regime.control_features),
            "directionalFeatures": list(regime.directional_features),
            "controlPriorStrength": control_strength,
            "jointPriorStrength": joint_strength,
            "trainSelectedJointWeight": joint_weight,
            "validationRowsWithEmptyControlCell": int(np.count_nonzero(
                control_counts[validation_control_ids] == 0
            )),
            "validationRowsWithEmptyJointCell": int(np.count_nonzero(
                joint_counts[validation_joint_ids] == 0
            )),
            "selection": selection_report,
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
            "matchedMagnitudeControl": embargoed_split_metrics(
                validation_targets,
                control_probabilities,
                split_at,
                holdout_start,
            ),
            "jointDirectionalTable": embargoed_split_metrics(
                validation_targets,
                joint_probabilities,
                split_at,
                holdout_start,
            ),
            "trainSelectedBackoff": embargoed_split_metrics(
                validation_targets,
                residual_probabilities,
                split_at,
                holdout_start,
            ),
            "logRatioResidualFusionPrimary": ratio_metrics,
            "convexFusionDiagnostic": convex_metrics,
        },
        "selectionContract": {
            "regimeShrinkageAndBackoffSelectedOnTrainOnly": True,
            "regimeRequiresPositiveDirectionalImprovementWhenAvailable": True,
            "primaryFusionPredeclared": "logRatioResidualFusionPrimary",
            "fusionScalarSelectedOnFirstValidationHalfOnly": True,
            "forecastHorizonEmbargoApplied": True,
            "withinAuditHoldoutUsedForSelection": False,
            "developmentScreenGate": DEVELOPMENT_SCREEN_GATE,
            "validationReusedAcrossEarlierExperiments": True,
            "sealedTestRemainsGloballyUntouched": True,
            "testUsed": False,
        },
    }


def validate_corpus_split_contract(
    repo_root: Path,
    target_root: Path,
    target_files: list[Path],
) -> dict:
    """Refuse any corpus mutation before a target reference can be opened."""
    contract_file = (repo_root / CORPUS_SPLIT_CONTRACT).resolve()
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    required = {"train", "validation", "test"}
    if contract.get("schemaVersion") != 1 \
            or contract.get("targetContract") != target_root.name \
            or contract.get("filenameHashEncoding") \
            != "utf8-lf-with-trailing-lf" \
            or not required.issubset(contract) \
            or contract.get("test", {}).get("policy") != "sealed-never-load":
        raise ValueError("invalid immutable oracle corpus split contract")
    filenames = [file.name for file in target_files]
    digest = hashlib.sha256(
        ("\n".join(filenames) + "\n").encode("utf-8")
    ).hexdigest()
    reference_count = int(contract.get("referenceCount", -1))
    split_count = sum(int(contract[name].get("count", -1)) for name in required)
    if len(filenames) != reference_count \
            or split_count != reference_count \
            or digest != contract.get("referenceFilenameSha256"):
        raise ValueError("oracle filenames differ from immutable split contract")
    train_end = int(contract["train"]["count"])
    validation_end = train_end + int(contract["validation"]["count"])
    boundaries = (
        (0, "train", "first"),
        (train_end - 1, "train", "last"),
        (train_end, "validation", "first"),
        (validation_end - 1, "validation", "last"),
        (validation_end, "test", "first"),
        (reference_count - 1, "test", "last"),
    )
    if any(
        filenames[index] != f"{contract[split][edge]}.json"
        for index, split, edge in boundaries
    ):
        raise ValueError("oracle boundaries differ from immutable split contract")
    return contract


def day_range(first: str, last: str) -> list[str]:
    start = date.fromisoformat(first)
    end = date.fromisoformat(last)
    if end < start:
        raise ValueError("invalid corpus day range")
    return [
        (start + timedelta(days=offset)).isoformat()
        for offset in range((end - start).days + 1)
    ]


def validate_runtime_split_assignment(
    target_files: list[Path],
    segments: dict[str, list[CausalSegment]],
    contract: dict,
) -> None:
    """Bind dynamic segment construction to every immutable split slice."""
    split_names = ("train", "validation", "test")
    if set(segments) != set(split_names):
        raise ValueError("runtime corpus split names differ from contract")
    offset = 0
    all_actual: set[Path] = set()
    for split in split_names:
        count = int(contract[split]["count"])
        expected = {
            file.resolve() for file in target_files[offset:offset + count]
        }
        actual = {
            segment.target_file.resolve() for segment in segments[split]
        }
        if any(segment.split != split for segment in segments[split]) \
                or actual != expected \
                or all_actual & actual:
            raise ValueError(
                f"runtime {split} assignment differs from immutable contract"
            )
        all_actual.update(actual)
        offset += count
    if offset != len(target_files) or len(all_actual) != len(target_files):
        raise ValueError("runtime corpus split coverage differs from contract")


def read_trade_flow_day(
    root: Path,
    day_value: str,
    opened: set[Path],
    *,
    target_contract: str,
    sealed_test_start: str,
    sealed_test_end: str,
    metadata_callback=None,
) -> dict[str, np.ndarray]:
    if day_value >= sealed_test_start:
        raise ValueError(f"refusing sealed-test trade-flow date {day_value}")
    reference = (root / f"{day_value}.json").resolve()
    manifest = json.loads(reference.read_text(encoding="utf-8"))
    opened.add(reference)
    if manifest.get("sequence") != {
        "start": utc_day_start_ms(day_value),
        "step": SECOND_MS,
        "count": SECOND_ROWS,
        "unit": "unix-ms",
    }:
        raise ValueError(f"invalid trade-flow sequence: {reference}")
    if manifest.get("layout", {}).get("encoding") != "trade-flow-columnar-v1":
        raise ValueError(f"invalid trade-flow layout: {reference}")
    metadata = manifest.get("metadata", {})
    expected_archive = (
        f"{TRADE_FLOW_SOURCE_ROOT}/BTCUSDT-aggTrades-{day_value}.zip"
    )
    if metadata.get("featureSchema") != TRADE_FLOW_SCHEMA \
            or metadata.get("oracleTargetContract") != target_contract \
            or metadata.get("sealedTestStart") != sealed_test_start \
            or metadata.get("sealedTestEnd") != sealed_test_end \
            or metadata.get("completeUtcDay") is not True \
            or str(metadata.get("aggregateIdGapCount")) != "0" \
            or metadata.get("source") != "data.binance.vision" \
            or metadata.get("sourceDataset") != "spot/daily/aggTrades" \
            or metadata.get("sourceArchiveUrl") != expected_archive \
            or not re.fullmatch(
                r"[a-f0-9]{64}", str(metadata.get("sourceArchiveSha256", "")),
            ) \
            or int(metadata.get("sourceArchiveBytes", -1)) <= 0 \
            or metadata.get("sourceCsvEntry") \
            != f"BTCUSDT-aggTrades-{day_value}.csv" \
            or int(metadata.get("sourceCsvBytes", -1)) <= 0 \
            or metadata.get("market") != "spot" \
            or metadata.get("symbol") != "BTCUSDT" \
            or metadata.get("interval") != "1s":
        raise ValueError(f"invalid trade-flow source contract: {reference}")
    expected_unit = "microsecond" if day_value >= "2025-01-01" \
        else "millisecond"
    if metadata.get("sourceTimestampUnit") != expected_unit:
        raise ValueError(f"invalid trade-flow timestamp unit: {reference}")
    if metadata_callback is not None:
        metadata_callback(day_value, metadata)
    values = read_trade_flow_columns(reference, TRADE_FLOW_COLUMNS)
    if any(value.shape != (SECOND_ROWS,) or not np.isfinite(value).all()
           for value in values.values()):
        raise ValueError(f"invalid trade-flow payload: {reference}")
    source_rows = int(metadata.get("sourceCsvRows", -1))
    sentinel_rows = int(metadata.get("invalidSentinelRows", -1))
    expected_aggregate_rows = source_rows - sentinel_rows
    aggregate_rows = int(np.sum(
        values["aggressiveBuyAggregateTradeCount"], dtype=np.uint64,
    ) + np.sum(
        values["aggressiveSellAggregateTradeCount"], dtype=np.uint64,
    ))
    raw_rows = int(np.sum(
        values["aggressiveBuyTradeCount"], dtype=np.uint64,
    ) + np.sum(
        values["aggressiveSellTradeCount"], dtype=np.uint64,
    ))
    try:
        aggregate_id_span = (
            int(metadata["lastAggregateTradeId"])
            - int(metadata["firstAggregateTradeId"])
            + 1
        )
        raw_id_span = (
            int(metadata["lastTradeId"])
            - int(metadata["firstTradeId"])
            + 1
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid trade-flow source ID metadata: {reference}"
        ) from error
    if expected_aggregate_rows <= 0 \
            or aggregate_rows != expected_aggregate_rows \
            or aggregate_id_span != expected_aggregate_rows \
            or raw_rows != raw_id_span:
        raise ValueError(f"trade-flow source counters disagree: {reference}")
    return values


def daily_features(
    day_value: str,
    candle_cache: SecondDayCache,
    flow_cache: TradeFlowDayCache,
) -> np.ndarray:
    current_date = date.fromisoformat(day_value)
    previous_value = (current_date - timedelta(days=1)).isoformat()
    previous_seconds = candle_cache.load(previous_value)
    current_seconds = candle_cache.load(day_value)
    previous_minutes = aggregate_minutes(previous_seconds)
    current_minutes = aggregate_minutes(current_seconds)
    aggregate = causal_ohlcv_features(
        completed_candle_windows(previous_minutes, current_minutes),
        completed_close_windows(previous_minutes, current_minutes),
    )
    proxy = rolling_tick_rule_proxy(previous_seconds, current_seconds)
    previous_flow = flow_cache.load(previous_value)
    current_flow = flow_cache.load(day_value)
    reconcile_trade_flow_with_candles(
        previous_value, previous_seconds, previous_flow,
    )
    reconcile_trade_flow_with_candles(
        day_value, current_seconds, current_flow,
    )
    flow = causal_trade_flow_features(previous_flow, current_flow)
    result = np.column_stack((aggregate, proxy, np.abs(proxy), flow)).astype(
        np.float32, copy=False,
    )
    if result.shape != (DAY_ROWS, len(FEATURE_NAMES)) \
            or not np.isfinite(result).all():
        raise ValueError(f"invalid combined trade-flow features for {day_value}")
    return result


def reconcile_trade_flow_with_candles(
    day_value: str,
    candles: np.ndarray,
    flow: dict[str, np.ndarray],
) -> None:
    """Prove that trade parsing/binning reconstructs canonical base volume."""
    if candles.shape != (SECOND_ROWS, 5):
        raise ValueError(f"invalid reconciliation candles for {day_value}")
    total_volume = (
        flow["aggressiveBuyBaseVolume"]
        + flow["aggressiveSellBaseVolume"]
    )
    if not np.allclose(total_volume, candles[:, 4], rtol=1e-10, atol=1e-10):
        difference = float(np.max(np.abs(total_volume - candles[:, 4])))
        raise ValueError(
            f"trade-flow/candle volume mismatch for {day_value}: {difference}"
        )
    tolerance = np.maximum(candles[:, 1] * 1e-12, 1e-10)
    for side in ("Buy", "Sell"):
        base = flow[f"aggressive{side}BaseVolume"]
        quote = flow[f"aggressive{side}QuoteVolume"]
        active = base > 0
        vwap = safe_divide(quote, base)
        if bool((vwap[active] < candles[active, 2] - tolerance[active]).any()) \
                or bool((vwap[active] > candles[active, 1] + tolerance[active]).any()):
            raise ValueError(
                f"trade-flow {side.lower()} VWAP escapes candle range for "
                f"{day_value}"
            )


def rolling_tick_rule_proxy(
    previous_seconds: np.ndarray,
    current_seconds: np.ndarray,
) -> np.ndarray:
    if previous_seconds.shape != (SECOND_ROWS, 5) \
            or current_seconds.shape != (SECOND_ROWS, 5):
        raise ValueError("tick-rule proxy requires two canonical one-second days")
    closes = np.concatenate((previous_seconds[-60:, 3], current_seconds[:, 3]))
    returns = np.diff(np.log(closes))
    volumes = np.concatenate((previous_seconds[-59:, 4], current_seconds[:, 4]))
    signed = np.sign(returns) * volumes
    windows = np.lib.stride_tricks.sliding_window_view(signed, 60)[
        np.arange(DAY_ROWS) * MINUTE_SECONDS
    ]
    volume_windows = np.lib.stride_tricks.sliding_window_view(volumes, 60)[
        np.arange(DAY_ROWS) * MINUTE_SECONDS
    ]
    result = safe_divide(windows.sum(axis=1), volume_windows.sum(axis=1))
    if result.shape != (DAY_ROWS,) or not np.isfinite(result).all():
        raise ValueError("invalid tick-rule signed-volume proxy")
    return result


def load_split(
    split: str,
    segments: list[CausalSegment],
    candle_cache: SecondDayCache,
    flow_cache: TradeFlowDayCache,
    access: AccessLog,
) -> tuple[np.ndarray, np.ndarray]:
    if split not in {"train", "validation"}:
        raise ValueError("trade-flow audit only loads train or validation")
    if any(segment.split != split for segment in segments):
        raise ValueError("segment split differs from requested split")
    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    for index, segment in enumerate(segments, start=1):
        values = daily_features(segment.target_file.stem, candle_cache, flow_cache)
        target_file = segment.target_file.resolve()
        _shard, day_targets = read_shard_array(
            target_file, "<f4", (DAY_ROWS, ACTION_COUNT),
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
            or not np.allclose(target_rows.sum(axis=1), 1, atol=2e-4, rtol=2e-4):
        raise ValueError(f"invalid {split} trade-flow rows")
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
) -> tuple[float, float, dict[str, float], int, int, tuple[np.ndarray, ...], np.ndarray]:
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
            calibration_targets, calibration_ids, table,
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
) -> tuple[tuple[PairedRegime, float, float, float], dict[str, object]]:
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
    choices: list[tuple[float, float, str, PairedRegime, float, float, float]] = []
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
            calibration_features, regime.control_features, control[5],
        )
        joint_ids, _ = encode_cells(
            calibration_features, regime.joint_features, joint[5],
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
        rows.append({
            "name": regime.name,
            "controlFeatures": list(regime.control_features),
            "directionalFeatures": list(regime.directional_features),
            "controlKlByStrength": control[2],
            "jointKlByStrength": joint[2],
            "selectedControlKl": control[1],
            "selectedJointKl": joint[1],
            "trainSelectedJointWeight": weight,
            "trainBackoffKl": backoff_kl,
            "directionalKlReductionFromControl": improvement,
            "positiveDirectionalImprovement": positive,
            "controlCells": control[3],
            "jointCells": joint[3],
            "populatedControlCells": control[4],
            "populatedJointCells": joint[4],
        })
        choices.append((
            0 if positive else 1,
            backoff_kl,
            regime.name,
            regime,
            control[0],
            joint[0],
            weight,
        ))
    winner = min(choices, key=lambda item: (item[0], item[1], item[2]))
    return (winner[3], winner[4], winner[5], winner[6]), {
        "fitRows": fit_end,
        "forecastHorizonEmbargoRows": EMBARGO_ROWS,
        "calibrationStartRow": calibration_start,
        "calibrationRows": features.shape[0] - calibration_start,
        "validationUsedForSelection": False,
        "candidates": rows,
        "selected": {
            "name": winner[3].name,
            "controlPriorStrength": winner[4],
            "jointPriorStrength": winner[5],
            "jointWeight": winner[6],
            "trainBackoffKl": winner[1],
            "positiveDirectionalImprovement": winner[0] == 0,
        },
    }


def embargoed_split_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    split_at: int,
    holdout_start: int,
) -> dict[str, float]:
    """Score selection rows, purged holdout rows, and the actual full set."""
    if targets.shape != probabilities.shape \
            or targets.ndim != 2 \
            or not 0 < split_at < holdout_start < targets.shape[0]:
        raise ValueError("embargoed fusion metric rows are incompatible")
    return {
        "firstHalfKl": mean_kl_dense(
            targets[:split_at], probabilities[:split_at],
        ),
        "secondHalfKl": mean_kl_dense(
            targets[holdout_start:], probabilities[holdout_start:],
        ),
        "fullValidationKl": mean_kl_dense(targets, probabilities),
    }


def mean_kl_dense(targets: np.ndarray, predictions: np.ndarray) -> float:
    if targets.shape != predictions.shape:
        raise ValueError("dense KL inputs differ")
    return mean_kl_indexed(
        targets,
        np.arange(targets.shape[0], dtype=np.int64),
        predictions,
    )


if __name__ == "__main__":
    main()

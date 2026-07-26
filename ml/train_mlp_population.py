from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

# Windows-mounted temporary directories do not support the Unix sockets and
# shared-file handles used by PyTorch DataLoader workers under WSL.
if os.name == "posix" and os.environ.get("TMPDIR", "").startswith("/mnt/"):
    os.environ["TMPDIR"] = "/tmp"
    tempfile.tempdir = "/tmp"

import torch
import onnx
from torch import Tensor
from onnx import numpy_helper

from mlp_model import (
    DEPLOYMENT_PROBABILITY_FLOOR,
    ParameterExposureMlp as ExposureMlp,
    LossWeights,
    PolicySupport,
    conditional_policy_logits,
    fitted_teacher_loss,
)
from population_mlp import PopulationExposureMlp, population_clip_grad_norm_
from train_mlp import (
    KL_MOMENT_METRIC_NAMES,
    METRIC_NAMES,
    TRAIN_METRIC_NAMES,
    TIME_BLOCK_METRIC_NAMES,
    FittedPolicyDataset,
    atomic_bytes,
    atomic_json,
    atomic_torch_save,
    cached_training_normalization,
    cached_training_parameter_scale,
    capture_rng,
    deterministic_current_states,
    emit,
    learning_rate_multiplier,
    loader,
    merge_weighted_moments,
    parse_time_weighting,
    report_stored_example_weights,
    resolve_device,
    restore_rng,
    set_determinism,
    is_centered_power_of_two_grid,
    time_weighting_metadata,
    validate_dataset_manifest,
    weighted_standard_deviation,
)


LOSS_WEIGHT_KEYS = (
    "crossEntropy",
    "probabilityMse",
    "parameterMse",
    "excessEntropy",
    "oracleMutualInformation",
)


def parse_population_loss_weights(value: dict) -> LossWeights:
    return LossWeights(
        cross_entropy=float(value.get("crossEntropy", 1)),
        probability_mse=float(value.get("probabilityMse", 1)),
        parameter_mse=float(value.get("parameterMse", 1)),
        excess_entropy=float(value.get("excessEntropy", 1)),
        oracle_mutual_information=float(value.get("oracleMutualInformation", 1)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a vectorized population of loss-weight MLP variants."
    )
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--population-size", type=int, required=True)
    parser.add_argument(
        "--retrain-completed",
        action="store_true",
        help="Retrain and atomically replace every job, including completed jobs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.population_size < 1:
        raise ValueError("population size must be positive")
    specification = json.loads(args.jobs.read_text())
    common, jobs = validate_specification(specification)
    set_determinism(common["seed"])

    dataset_root = Path(common["dataset"])
    manifest = json.loads((dataset_root / "dataset.json").read_text())
    validate_dataset_manifest(manifest, dataset_root)
    plan = json.loads(Path(common["plan"]).read_text())
    if (
        manifest.get("planId") != plan.get("id")
        or int(manifest.get("predictionDelayMs", -1))
        != int(plan.get("predictionDelayMs", -2))
    ):
        raise ValueError(
            "population dataset does not match its training plan: "
            f"dataset={manifest.get('planId')} "
            f"delayMs={manifest.get('predictionDelayMs')}, "
            f"plan={plan.get('id')} delayMs={plan.get('predictionDelayMs')}"
        )
    train = FittedPolicyDataset(
        manifest, dataset_root, "train", target="teacherParameters",
    )
    validation = FittedPolicyDataset(
        manifest, dataset_root, "validation", target="teacherParameters",
    )
    test = FittedPolicyDataset(
        manifest, dataset_root, "test", target="teacherParameters",
    )
    if min(len(train), len(validation), len(test)) == 0:
        raise RuntimeError("train, validation, and test datasets must all be non-empty")
    if manifest["exampleWeighting"]["timeWeighting"] != time_weighting_metadata(
        parse_time_weighting(json.dumps(common["timeWeighting"]))
    ):
        raise ValueError("training time-weighting configuration does not match stored weights")

    device = resolve_device(common["device"])
    feature_mean, feature_std = cached_training_normalization(
        train,
        Path(common["featureStatisticsCache"]),
    )
    parameter_scale = cached_training_parameter_scale(
        train,
        Path(common["targetStatisticsCache"]),
    ).to(device)
    for dataset in (train, validation, test):
        report_stored_example_weights(dataset)

    support = PolicySupport(**manifest["policySupport"])
    actions = torch.linspace(
        support.visible_lower,
        support.visible_upper,
        int(manifest["actionCount"]),
        dtype=torch.float32,
        device=device,
    )
    current = deterministic_current_states(
        common["statesPerExample"],
        support,
        device,
        visible=True,
    )
    # Keep the exact validation blocks used by the scalar trainer. Oracle MI is
    # a block statistic, so shrinking these batches for a wider population
    # would change both the sampled rows and the reported selection metric.
    evaluation_batch_size = common["evaluationBatchSize"]
    loader_args = argparse.Namespace(workers=common["workers"])
    train_loader = loader(
        train,
        loader_args,
        shuffle=True,
        batch_size=common["batchSize"],
    )
    validation_loader = loader(
        validation,
        loader_args,
        shuffle=False,
        batch_size=evaluation_batch_size,
        sample_fraction=common["validationFraction"],
    )
    train_batches_per_epoch = min(
        len(train_loader),
        common.get("maxBatchesPerEpoch", len(train_loader)),
    )
    steps_per_epoch = math.ceil(train_batches_per_epoch / common["accumulate"])
    total_steps = max(1, steps_per_epoch * common["epochs"])

    pending = jobs if args.retrain_completed else [
        job for job in jobs if not completed_job(job, manifest, common)
    ]
    emit({
        "event": "population-training-start",
        "device": str(device),
        "configuredPopulationSize": args.population_size,
        "jobs": len(jobs),
        "pendingJobs": len(pending),
        "trainExamples": len(train),
        "validationExamples": len(validation),
        "screeningValidationExamples": validation_loader.batch_sampler.example_count,
        "batchSize": common["batchSize"],
        "trainBatchesPerEpoch": train_batches_per_epoch,
        "fullTrainBatchesPerEpoch": len(train_loader),
        "evaluationBatchSize": evaluation_batch_size,
        "compiledObjective": common["compile"],
    })
    runtimes: dict[int, PopulationRuntime] = {}
    winner = None if args.retrain_completed else completed_winner(
        jobs,
        common["selectionMetric"],
        manifest,
        common,
    )
    population_groups = math.ceil(len(pending) / args.population_size)
    for offset in range(0, len(pending), args.population_size):
        group = pending[offset:offset + args.population_size]
        group_index = offset // args.population_size
        population_size = len(group)
        runtime = runtimes.get(population_size)
        if runtime is None:
            runtime = PopulationRuntime(
                population_size,
                feature_mean,
                feature_std,
                common,
                actions,
                current,
                support,
                parameter_scale,
                int(manifest["samplingIntervalMs"]),
                device,
            )
            runtimes[population_size] = runtime
        winner = run_group(
            runtime,
            group,
            args.jobs,
            common,
            manifest,
            train,
            validation,
            train_loader,
            validation_loader,
            total_steps,
            device,
            group_index,
            population_groups,
            len(pending),
            offset,
            winner,
        )
    cleanup_population_checkpoints(args.jobs)
    emit({
        "event": "population-training-complete",
        "jobs": len(jobs),
        "trainedJobs": len(pending),
        "populationSize": args.population_size,
    })


class PopulationRuntime:
    def __init__(
        self,
        population_size: int,
        feature_mean: Tensor,
        feature_std: Tensor,
        common: dict,
        actions: Tensor,
        current: Tensor,
        support: PolicySupport,
        parameter_scale: Tensor,
        sampling_interval_ms: int,
        device: torch.device,
    ) -> None:
        set_determinism(common["seed"])
        self.prototype = ExposureMlp(
            feature_mean,
            feature_std,
            common["dropout"],
        ).to(device)
        self.model = PopulationExposureMlp(
            self.prototype,
            population_size,
        ).to(device)
        self.initial_rng = capture_rng()
        self.population_size = population_size
        self.actions = actions
        self.current = current
        self.support = support
        self.parameter_scale = parameter_scale
        self.sampling_interval_ms = sampling_interval_ms
        self.device = device

        def objective(include_deployment_metrics):
            def compute(
                features,
                targets,
                time_weights,
                times,
                weight_matrix,
            ):
                predicted = self.model(features)
                expanded_current = self.current.expand(features.shape[0], -1)

                def member_loss(member_prediction, member_weights):
                    weights = LossWeights(*member_weights.unbind())
                    return fitted_teacher_loss(
                        member_prediction,
                        targets,
                        self.actions,
                        expanded_current,
                        self.support,
                        self.parameter_scale,
                        weights,
                        time_weights,
                        times,
                        self.sampling_interval_ms,
                        include_deployment_metrics=
                            include_deployment_metrics,
                    )

                return torch.vmap(member_loss)(predicted, weight_matrix)
            return compute

        self.training_objective = objective(False)
        self.evaluation_objective = objective(True)
        if common["compile"]:
            self.training_objective = torch.compile(
                self.training_objective,
                mode="reduce-overhead",
                fullgraph=False,
            )
            self.evaluation_objective = torch.compile(
                self.evaluation_objective,
                mode="reduce-overhead",
                fullgraph=False,
            )

    def reset(self, jobs: list[dict]) -> None:
        parent_states = [
            load_parent_state(job, self.prototype)
            for job in jobs
        ]
        self.model.reset_from_state_dicts(parent_states)
        # Match a fresh scalar process immediately after constructing its one
        # model. This reproduces its batch order and dropout stream for every
        # loss-weight candidate while avoiding redundant model initialization.
        restore_rng(self.initial_rng)


def run_group(
    runtime: PopulationRuntime,
    group: list[dict],
    jobs_file: Path,
    common: dict,
    manifest: dict,
    train: FittedPolicyDataset,
    validation: FittedPolicyDataset,
    train_loader,
    validation_loader,
    total_steps: int,
    device: torch.device,
    group_index: int,
    population_groups: int,
    pending_jobs: int,
    completed_before: int,
    previous_winner: tuple[float, dict] | None,
) -> tuple[float, dict] | None:
    keys = [job["key"] for job in group]
    checkpoint_file = population_checkpoint_file(jobs_file, keys)
    train_batches_per_epoch = min(
        len(train_loader),
        common.get("maxBatchesPerEpoch", len(train_loader)),
    )
    steps_per_epoch = math.ceil(
        train_batches_per_epoch / common["accumulate"]
    )
    runtime.reset(group)
    optimizer = torch.optim.AdamW(
        runtime.model.parameters(),
        lr=common["learningRate"] * learning_rate_multiplier(0, total_steps),
        weight_decay=common["weightDecay"],
        betas=(0.9, 0.95),
        foreach=device.type == "cuda",
    )
    scaler = torch.amp.GradScaler(
        "cuda",
        init_scale=256.0,
        enabled=device.type == "cuda",
    )
    training_contract = {
        "version": 1,
        "jobs": keys,
        "datasetPlanId": manifest["planId"],
        "componentStoreId": manifest["componentLayout"].get(
            "storeId",
            manifest["planId"],
        ),
        "predictionDelayMs": int(manifest["predictionDelayMs"]),
        "common": common,
        "parents": [parent_checkpoint_identity(job) for job in group],
    }
    start_epoch = 0
    global_step = 0
    best_epochs = [-1] * len(group)
    best_validation = [math.inf] * len(group)
    best_metrics: list[dict[str, float]] = [{} for _ in group]
    best_states: list[dict[str, Tensor] | None] = [None] * len(group)
    stale_epochs = [0] * len(group)
    epochs_trained = [0] * len(group)
    active = [True] * len(group)
    if checkpoint_file.exists():
        checkpoint = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if checkpoint.get("trainingContract") != training_contract:
            raise RuntimeError(f"population checkpoint contract mismatch: {checkpoint_file}")
        runtime.model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        global_step = checkpoint["globalStep"]
        best_epochs = checkpoint["bestEpochs"]
        best_validation = checkpoint["bestValidation"]
        best_metrics = checkpoint["bestMetrics"]
        best_states = checkpoint["bestStates"]
        stale_epochs = checkpoint["staleEpochs"]
        epochs_trained = checkpoint.get(
            "epochsTrained",
            [start_epoch] * len(group),
        )
        active = checkpoint["active"]
        restore_rng(checkpoint["rng"])

    weight_matrix = torch.tensor(
        [[float(job["lossWeights"][key]) for key in LOSS_WEIGHT_KEYS] for job in group],
        dtype=torch.float32,
        device=device,
    )
    emit({
        "event": "population-group-start",
        "jobs": keys,
        "populationSize": len(group),
        "populationGroup": group_index + 1,
        "populationGroups": population_groups,
        "completedPopulationJobs": min(
            pending_jobs,
            completed_before,
        ),
        "pendingPopulationJobs": pending_jobs,
        "startEpoch": start_epoch,
        "epochs": common["epochs"],
        "checkpoint": str(checkpoint_file),
        "parents": [
            job.get("initializeFromCheckpoint")
            for job in group
        ],
    })
    for epoch in range(start_epoch, common["epochs"]):
        if not any(active):
            break
        started = time.monotonic()
        active_at_epoch_start = list(active)
        train_metrics, global_step = train_epoch(
            runtime,
            train_loader,
            optimizer,
            scaler,
            weight_matrix,
            active,
            common,
            epoch,
            global_step,
            total_steps,
            keys,
            group_index,
            population_groups,
        )
        validation_metrics = evaluate_population(
            runtime,
            validation_loader,
            weight_matrix,
        )
        for member, was_active in enumerate(active_at_epoch_start):
            if was_active:
                epochs_trained[member] += 1
        for member, job in enumerate(group):
            if not active[member]:
                continue
            score = validation_metrics[member][common["selectionMetric"]]
            if (
                score
                < best_validation[member]
                - float(common.get("minimumImprovement", 1e-6))
            ):
                best_validation[member] = score
                best_metrics[member] = validation_metrics[member]
                best_epochs[member] = epoch
                stale_epochs[member] = 0
                best_states[member] = runtime.model.member_state_dict(member)
            else:
                stale_epochs[member] += 1
                if stale_epochs[member] >= common["patience"]:
                    active[member] = False
        checkpoint_every = common.get("checkpointEveryEpochs", 0)
        if checkpoint_every > 0 \
                and (epoch + 1) % checkpoint_every == 0 \
                and epoch + 1 < common["epochs"]:
            atomic_torch_save({
                "epoch": epoch,
                "globalStep": global_step,
                "bestEpochs": best_epochs,
                "bestValidation": best_validation,
                "bestMetrics": best_metrics,
                "bestStates": best_states,
                "staleEpochs": stale_epochs,
                "epochsTrained": epochs_trained,
                "active": active,
                "model": runtime.model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "rng": capture_rng(),
                "trainingContract": training_contract,
            }, checkpoint_file)
        emit({
            "event": "population-epoch",
            "epoch": epoch,
            "epochs": common["epochs"],
            "globalStep": global_step,
            "seconds": round(time.monotonic() - started, 2),
            "populationGroup": group_index + 1,
            "populationGroups": population_groups,
            "jobs": [{
                "key": job["key"],
                "active": active[index],
                "train": train_metrics[index],
                "validation": validation_metrics[index],
                "bestEpoch": best_epochs[index],
                "epochsTrained": epochs_trained[index],
                "bestValidation": best_validation[index],
                "staleEpochs": stale_epochs[index],
            } for index, job in enumerate(group)],
        })

    for member, job in enumerate(group):
        if best_epochs[member] < 0 or best_states[member] is None:
            raise RuntimeError(f"population member {job['key']} has no validated checkpoint")
    runtime.model.reset_from_state_dicts([
        state for state in best_states if state is not None
    ])
    best_metrics = evaluate_population(
        runtime,
        validation_loader,
        weight_matrix,
    )
    best_validation = [
        metrics[common["selectionMetric"]]
        for metrics in best_metrics
    ]
    group_winner_member = min(
        range(len(group)),
        key=lambda member: best_validation[member],
    )
    group_winner_score = best_validation[group_winner_member]
    group_winner_job = group[group_winner_member]
    replaces_winner = previous_winner is None \
        or group_winner_score < previous_winner[0]
    equivalence_signatures = materialize_equivalence_signatures(
        runtime,
        group,
        best_states,
        validation,
        common.get("equivalenceSignature"),
    )
    if common.get("retainAllBestModels", False):
        for member, job in enumerate(group):
            atomic_torch_save(
                best_states[member],
                Path(job["output"]) / "best-model.pt",
            )
    elif replaces_winner:
        # Materialize the recoverable model before publishing its study result.
        # A crash can then leave an extra model, but never a winning metric row
        # whose weights are unavailable for artifact export.
        atomic_torch_save(
            best_states[group_winner_member],
            Path(group_winner_job["output"]) / "best-model.pt",
        )

    for member, job in enumerate(group):
        loss_weights = parse_population_loss_weights(job["lossWeights"])
        training_batches = epochs_trained[member] * train_batches_per_epoch
        training_examples_per_epoch = min(
            len(train),
            train_batches_per_epoch * common["batchSize"],
        )
        study = {
            "modelId": job["modelId"],
            "datasetPlanId": manifest["planId"],
            "componentStoreId": manifest["componentLayout"].get(
                "storeId",
                manifest["planId"],
            ),
            "predictionDelayMs": int(manifest["predictionDelayMs"]),
            "selectionMetric": common["selectionMetric"],
            "policyMetricDefinitions": {
                "klDivergence": "teacher-hard-cutoff shape KL",
                "deploymentKlDivergence":
                    "predicted-hard-cutoff KL(target || prediction)",
                "deploymentProbabilityFloor":
                    DEPLOYMENT_PROBABILITY_FLOOR,
            },
            "bestEpoch": best_epochs[member],
            "bestValidationScore": best_validation[member],
            "bestValidationMetrics": best_metrics[member],
            "screeningBestValidationScore": best_validation[member],
            "screeningBestValidationMetrics": best_metrics[member],
            "screeningValidationExamples":
                validation_loader.batch_sampler.example_count,
            "validationFraction": common["validationFraction"],
            "lossWeights": asdict(loss_weights),
            "trainExamples": len(train),
            "screeningTrainExamples": min(
                len(train),
                common.get("maxBatchesPerEpoch", len(train_loader))
                * common["batchSize"],
            ),
            "validationExamples": len(validation),
            "epochs": common["epochs"],
            "epochsTrained": epochs_trained[member],
            "patience": common["patience"],
            "minimumImprovement": float(
                common.get("minimumImprovement", 1e-6)
            ),
            "trainingBatches": training_batches,
            "trainingExamples": (
                epochs_trained[member] * training_examples_per_epoch
            ),
            "optimizerUpdates": (
                epochs_trained[member] * steps_per_epoch
            ),
            "seed": common["seed"],
            "device": str(device),
            "stoppedByPatience": not active[member],
            "finalizedEarly": epochs_trained[member] < common["epochs"],
            "populationTraining": {
                "vectorized": True,
                "populationSize": len(group),
                "sharedDataLoader": True,
                "independentAdamWMoments": True,
                "independentEarlyStopping": True,
                "parentCheckpoint": job.get("initializeFromCheckpoint"),
                "parentKey": job.get("parentKey"),
                "maxBatchesPerEpoch": common.get("maxBatchesPerEpoch"),
                "epochsTrained": epochs_trained[member],
                "trainingBatches": training_batches,
                "optimizerUpdates": (
                    epochs_trained[member] * steps_per_epoch
                ),
            },
            **(
                {"equivalenceSignature": equivalence_signatures[member]}
                if equivalence_signatures is not None else {}
            ),
        }
        atomic_json(study, Path(job["resultFile"]))
    winner = previous_winner
    if replaces_winner:
        if winner is not None and not common.get("retainAllBestModels", False):
            (Path(winner[1]["output"]) / "best-model.pt").unlink(missing_ok=True)
        winner = (group_winner_score, group_winner_job)
    if not common.get("retainAllBestModels", False):
        for job in group:
            if winner is not None and job["key"] == winner[1]["key"]:
                continue
            (Path(job["output"]) / "best-model.pt").unlink(missing_ok=True)
            (Path(job["output"]) / "checkpoint.pt").unlink(missing_ok=True)
    checkpoint_file.unlink(missing_ok=True)
    emit({
        "event": "population-group-complete",
        "jobs": keys,
        "populationGroup": group_index + 1,
        "populationGroups": population_groups,
        "completedPopulationJobs": min(
            pending_jobs,
            completed_before + len(group),
        ),
        "pendingPopulationJobs": pending_jobs,
        "bestValidation": best_validation,
        "bestEpochs": best_epochs,
        "epochsTrained": epochs_trained,
    })
    return winner


@torch.inference_mode()
def materialize_equivalence_signatures(
    runtime: PopulationRuntime,
    group: list[dict],
    best_states: list[dict[str, Tensor] | None],
    validation: FittedPolicyDataset,
    configuration: dict | None,
) -> list[dict] | None:
    """Persist compact policy-surface probes used to collapse equivalent branches."""
    if configuration is None:
        return None
    example_count = min(int(configuration["examples"]), len(validation))
    indices = torch.linspace(
        0,
        len(validation) - 1,
        example_count,
        dtype=torch.int64,
    ).unique().tolist()
    features = torch.stack([
        validation[int(index)][0] for index in indices
    ]).to(runtime.device)
    states = [state for state in best_states if state is not None]
    if len(states) != len(best_states):
        raise RuntimeError("cannot sign an unvalidated population state")
    runtime.model.reset_from_state_dicts(states)
    runtime.model.eval()
    predicted = runtime.model(features).float()
    actions = torch.linspace(
        runtime.support.visible_lower,
        runtime.support.visible_upper,
        int(configuration["actionStates"]),
        dtype=torch.float32,
        device=runtime.device,
    )
    current = torch.linspace(
        runtime.support.visible_lower,
        runtime.support.visible_upper,
        int(configuration["currentStates"]),
        dtype=torch.float32,
        device=runtime.device,
    ).view(1, -1).expand(len(indices), -1)
    signatures = []
    for member, job in enumerate(group):
        rows = predicted[member, :, None, :].expand(-1, current.shape[1], -1)
        probabilities = torch.softmax(
            conditional_policy_logits(
                rows,
                actions,
                current,
                runtime.support,
            ),
            dim=-1,
        ).to(device="cpu", dtype=torch.float16).contiguous()
        value = probabilities.numpy().astype("<f2", copy=False).tobytes()
        target = Path(job["output"]) / "equivalence-signature.f16"
        atomic_bytes(value, target)
        signatures.append({
            "version": 1,
            "file": target.name,
            "dtype": "float16",
            "layout": "[validationExample,currentExposure,targetExposure]",
            "shape": list(probabilities.shape),
            "validationIndices": [int(index) for index in indices],
            "currentRange": [
                runtime.support.visible_lower,
                runtime.support.visible_upper,
            ],
            "actionRange": [
                runtime.support.visible_lower,
                runtime.support.visible_upper,
            ],
            "sha256": hashlib.sha256(value).hexdigest(),
        })
    return signatures


def train_epoch(
    runtime: PopulationRuntime,
    data,
    optimizer,
    scaler,
    weight_matrix: Tensor,
    active: list[bool],
    common: dict,
    epoch: int,
    global_step: int,
    total_steps: int,
    group_keys: list[str],
    group_index: int,
    population_groups: int,
) -> tuple[list[dict[str, float]], int]:
    runtime.model.train()
    optimizer.zero_grad(set_to_none=True)
    totals = {
        name: torch.zeros(runtime.population_size, device=runtime.device)
        for name in TRAIN_METRIC_NAMES
    }
    total_examples = 0
    time_weight_sum = torch.zeros(runtime.population_size, device=runtime.device)
    time_weight_square_sum = torch.zeros(runtime.population_size, device=runtime.device)
    kl_weight_sum = torch.zeros(runtime.population_size, device=runtime.device)
    kl_mean = torch.zeros(runtime.population_size, device=runtime.device)
    kl_centered_square_sum = torch.zeros(
        runtime.population_size,
        device=runtime.device,
    )
    information_example_count = torch.zeros(runtime.population_size, device=runtime.device)
    active_tensor = torch.tensor(active, dtype=torch.float32, device=runtime.device)
    started = time.monotonic()
    epoch_batches = min(
        len(data),
        common.get("maxBatchesPerEpoch", len(data)),
    )
    for batch_step, (features, targets, time_weights, times, _) in enumerate(data):
        if batch_step >= epoch_batches:
            break
        features = features.to(runtime.device, non_blocking=True)
        targets = targets.to(runtime.device, non_blocking=True)
        time_weights = time_weights.to(runtime.device, non_blocking=True)
        times = times.to(runtime.device, non_blocking=True)
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.float16,
            enabled=runtime.device.type == "cuda",
        ):
            metrics = runtime.training_objective(
                features,
                targets,
                time_weights,
                times,
                weight_matrix,
            )
            member_losses = metrics["loss"]
            loss = (member_losses * active_tensor).sum() / common["accumulate"]
        if runtime.device.type == "cuda":
            torch._assert_async(torch.isfinite(loss), "non-finite population loss")
        elif not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite population loss")
        scaler.scale(loss).backward()
        should_step = (
            (batch_step + 1) % common["accumulate"] == 0
            or batch_step + 1 == min(
                len(data),
                common.get("maxBatchesPerEpoch", len(data)),
            )
        )
        if should_step:
            scaler.unscale_(optimizer)
            gradient_norms = population_clip_grad_norm_(
                runtime.model.parameters(),
                runtime.population_size,
                1.0,
            )
            scale_before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            optimizer_stepped = scaler.get_scale() >= scale_before
            if optimizer_stepped:
                global_step += 1
                multiplier = learning_rate_multiplier(global_step, total_steps)
                optimizer.param_groups[0]["lr"] = common["learningRate"] * multiplier
            if global_step == 1 or global_step % common["logEverySteps"] == 0:
                emit({
                    "event": "population-train-step",
                    "epoch": epoch,
                    "epochs": common["epochs"],
                    "batch": batch_step + 1,
                    "batches": epoch_batches,
                    "globalStep": global_step,
                    "learningRate": optimizer.param_groups[0]["lr"],
                    "gradientNorms": gradient_norms.detach().cpu().tolist(),
                    "populationExamplesPerSecond": round(
                        runtime.population_size * total_examples
                        / max(time.monotonic() - started, 1e-6),
                        1,
                    ),
                    "gpuMemoryMiB": round(
                        torch.cuda.max_memory_allocated() / 1_048_576,
                        1,
                    ) if runtime.device.type == "cuda" else 0,
                    "losses": member_losses.detach().cpu().tolist(),
                    "jobs": group_keys,
                    "populationGroup": group_index + 1,
                    "populationGroups": population_groups,
                })
            if not optimizer_stepped:
                emit({
                    "event": "population-gradient-overflow",
                    "epoch": epoch,
                    "globalStep": global_step,
                    "scale": scaler.get_scale(),
                })
        count = features.shape[0]
        total_examples += count
        time_weight_sum += metrics["timeWeightSum"].detach()
        time_weight_square_sum += metrics["timeWeightSquareSum"].detach()
        kl_weight_sum, kl_mean, kl_centered_square_sum = merge_weighted_moments(
            kl_weight_sum,
            kl_mean,
            kl_centered_square_sum,
            metrics["klWeightSum"].detach(),
            metrics["klDivergence"].detach(),
            metrics["klCenteredSquareSum"].detach(),
        )
        information_count = metrics["informationExampleCount"].detach()
        information_example_count += information_count
        for name in TRAIN_METRIC_NAMES:
            if name in KL_MOMENT_METRIC_NAMES:
                continue
            metric_count = information_count if name in TIME_BLOCK_METRIC_NAMES else count
            totals[name] += metrics[name].detach() * metric_count
    return finalize_metrics(
        totals,
        total_examples,
        information_example_count,
        time_weight_sum,
        time_weight_square_sum,
        kl_weight_sum,
        kl_mean,
        kl_centered_square_sum,
        None,
        None,
        None,
    ), global_step


@torch.inference_mode()
def evaluate_population(
    runtime: PopulationRuntime,
    data,
    weight_matrix: Tensor,
) -> list[dict[str, float]]:
    runtime.model.eval()
    totals = {
        name: torch.zeros(runtime.population_size, device=runtime.device)
        for name in METRIC_NAMES
    }
    total_examples = 0
    time_weight_sum = torch.zeros(runtime.population_size, device=runtime.device)
    time_weight_square_sum = torch.zeros(runtime.population_size, device=runtime.device)
    kl_weight_sum = torch.zeros(runtime.population_size, device=runtime.device)
    kl_mean = torch.zeros(runtime.population_size, device=runtime.device)
    kl_centered_square_sum = torch.zeros(
        runtime.population_size,
        device=runtime.device,
    )
    deployment_kl_weight_sum = torch.zeros(
        runtime.population_size,
        device=runtime.device,
    )
    deployment_kl_mean = torch.zeros(
        runtime.population_size,
        device=runtime.device,
    )
    deployment_kl_centered_square_sum = torch.zeros(
        runtime.population_size,
        device=runtime.device,
    )
    information_example_count = torch.zeros(runtime.population_size, device=runtime.device)
    for features, targets, time_weights, times, _ in data:
        features = features.to(runtime.device, non_blocking=True)
        targets = targets.to(runtime.device, non_blocking=True)
        time_weights = time_weights.to(runtime.device, non_blocking=True)
        times = times.to(runtime.device, non_blocking=True)
        with torch.autocast(
            device_type=runtime.device.type,
            dtype=torch.float16,
            enabled=runtime.device.type == "cuda",
        ):
            metrics = runtime.evaluation_objective(
                features,
                targets,
                time_weights,
                times,
                weight_matrix,
            )
        count = features.shape[0]
        total_examples += count
        time_weight_sum += metrics["timeWeightSum"]
        time_weight_square_sum += metrics["timeWeightSquareSum"]
        kl_weight_sum, kl_mean, kl_centered_square_sum = merge_weighted_moments(
            kl_weight_sum,
            kl_mean,
            kl_centered_square_sum,
            metrics["klWeightSum"],
            metrics["klDivergence"],
            metrics["klCenteredSquareSum"],
        )
        deployment_kl_weight_sum, deployment_kl_mean, \
            deployment_kl_centered_square_sum = merge_weighted_moments(
                deployment_kl_weight_sum,
                deployment_kl_mean,
                deployment_kl_centered_square_sum,
                metrics["deploymentKlWeightSum"],
                metrics["deploymentKlDivergence"],
                metrics["deploymentKlCenteredSquareSum"],
            )
        information_count = metrics["informationExampleCount"]
        information_example_count += information_count
        for name in METRIC_NAMES:
            if name in KL_MOMENT_METRIC_NAMES:
                continue
            metric_count = information_count if name in TIME_BLOCK_METRIC_NAMES else count
            totals[name] += metrics[name] * metric_count
    return finalize_metrics(
        totals,
        total_examples,
        information_example_count,
        time_weight_sum,
        time_weight_square_sum,
        kl_weight_sum,
        kl_mean,
        kl_centered_square_sum,
        deployment_kl_weight_sum,
        deployment_kl_mean,
        deployment_kl_centered_square_sum,
    )


def finalize_metrics(
    totals: dict[str, Tensor],
    total_examples: int,
    information_example_count: Tensor,
    time_weight_sum: Tensor,
    time_weight_square_sum: Tensor,
    kl_weight_sum: Tensor,
    kl_mean: Tensor,
    kl_centered_square_sum: Tensor,
    deployment_kl_weight_sum: Tensor | None,
    deployment_kl_mean: Tensor | None,
    deployment_kl_centered_square_sum: Tensor | None,
) -> list[dict[str, float]]:
    population_size = time_weight_sum.numel()
    results = [{} for _ in range(population_size)]
    for member in range(population_size):
        information_count = max(1.0, float(information_example_count[member]))
        for name, values in totals.items():
            denominator = information_count if name in TIME_BLOCK_METRIC_NAMES \
                else max(1, total_examples)
            results[member][name] = float(values[member]) / denominator
        results[member]["timeWeightEffectiveSampleRatio"] = float(
            time_weight_sum[member].square()
            / (
                total_examples * time_weight_square_sum[member]
            ).clamp_min(1e-12)
        )
        results[member]["klDivergence"] = float(kl_mean[member])
        results[member]["klDivergenceStdDev"] = float(
            weighted_standard_deviation(
                kl_centered_square_sum[member],
                kl_weight_sum[member],
            )
        )
        if deployment_kl_mean is not None:
            if deployment_kl_weight_sum is None \
                    or deployment_kl_centered_square_sum is None:
                raise ValueError("deployment KL moments are incomplete")
            results[member]["deploymentKlDivergence"] = float(
                deployment_kl_mean[member]
            )
            results[member]["deploymentKlDivergenceStdDev"] = float(
                weighted_standard_deviation(
                    deployment_kl_centered_square_sum[member],
                    deployment_kl_weight_sum[member],
                )
            )
    return results


def validate_specification(specification: dict) -> tuple[dict, list[dict]]:
    if specification.get("version") != 1:
        raise ValueError("unsupported population jobs format")
    common = specification.get("common")
    jobs = specification.get("jobs")
    required_common = {
        "dataset", "epochs", "batchSize", "evaluationBatchSize",
        "validationFraction", "accumulate", "learningRate", "weightDecay",
        "dropout", "statesPerExample", "patience", "workers", "seed", "device",
        "logEverySteps", "timeWeighting", "selectionMetric",
        "featureStatisticsCache", "targetStatisticsCache", "compile",
    }
    if not isinstance(common, dict) or not required_common.issubset(common):
        raise ValueError("population jobs common configuration is incomplete")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("population jobs must be a non-empty list")
    if common["selectionMetric"] not in ("loss", "klDivergence"):
        raise ValueError("invalid population selection metric")
    positive_counts = (
        "epochs", "batchSize", "evaluationBatchSize", "accumulate",
        "statesPerExample", "patience", "logEverySteps",
    )
    if any(not isinstance(common[key], int) or common[key] < 1 for key in positive_counts) \
            or not isinstance(common["workers"], int) or common["workers"] < 0 \
            or not isinstance(common["seed"], int) \
            or not 0 < common["validationFraction"] <= 1 \
            or common["learningRate"] <= 0 or common["weightDecay"] < 0 \
            or not 0 <= common["dropout"] < 1:
        raise ValueError("invalid population training configuration")
    if not is_centered_power_of_two_grid(common["statesPerExample"]):
        raise ValueError("population current-state grid must contain 2^n-1 cells")
    if not isinstance(common["compile"], bool) \
            or common["device"] not in ("auto", "cuda", "cpu") \
            or not isinstance(common["timeWeighting"], dict):
        raise ValueError("invalid population execution configuration")
    checkpoint_every = common.get("checkpointEveryEpochs", 0)
    if not isinstance(checkpoint_every, int) or checkpoint_every < 0:
        raise ValueError("checkpointEveryEpochs must be a non-negative integer")
    max_batches = common.get("maxBatchesPerEpoch")
    if max_batches is not None and (
        not isinstance(max_batches, int) or max_batches < 1
    ):
        raise ValueError("maxBatchesPerEpoch must be a positive integer")
    minimum_improvement = common.get("minimumImprovement", 1e-6)
    if (
        not isinstance(minimum_improvement, (int, float))
        or not math.isfinite(float(minimum_improvement))
        or float(minimum_improvement) < 0
    ):
        raise ValueError("minimumImprovement must be finite and non-negative")
    if not isinstance(common.get("retainAllBestModels", False), bool):
        raise ValueError("retainAllBestModels must be a boolean")
    signature = common.get("equivalenceSignature")
    if signature is not None and (
        not isinstance(signature, dict)
        or any(
            not isinstance(signature.get(key), int)
            or signature[key] < 1
            for key in ("examples", "currentStates", "actionStates")
        )
        or not is_centered_power_of_two_grid(signature["currentStates"])
        or not is_centered_power_of_two_grid(signature["actionStates"])
    ):
        raise ValueError(
            "equivalenceSignature requires positive examples and 2^n-1 state grids"
        )
    seen = set()
    for job in jobs:
        required_job = {
            "key", "output", "resultFile", "modelId", "label", "lossWeights",
        }
        if not isinstance(job, dict) or not required_job.issubset(job):
            raise ValueError("population job is incomplete")
        if job["key"] in seen:
            raise ValueError(f"duplicate population job key: {job['key']}")
        seen.add(job["key"])
        if set(job["lossWeights"]) != set(LOSS_WEIGHT_KEYS) \
                or any(
                    not isinstance(job["lossWeights"][key], (int, float))
                    or not math.isfinite(job["lossWeights"][key])
                    or job["lossWeights"][key] < 0
                    for key in LOSS_WEIGHT_KEYS
                ):
            raise ValueError(f"invalid loss weights for {job['key']}")
        parent = job.get("initializeFromCheckpoint")
        if parent is not None and (
            not isinstance(parent, str) or not Path(parent).is_file()
        ):
            raise ValueError(f"invalid parent checkpoint for {job['key']}")
        if job.get("parentKey") is not None and not isinstance(job["parentKey"], str):
            raise ValueError(f"invalid parent key for {job['key']}")
        Path(job["output"]).mkdir(parents=True, exist_ok=True)
        Path(job["resultFile"]).parent.mkdir(parents=True, exist_ok=True)
    return common, jobs


def completed_job(job: dict, manifest: dict, common: dict) -> bool:
    result_file = Path(job["resultFile"])
    if not result_file.exists():
        return False
    try:
        result = json.loads(result_file.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    expected_weights = asdict(parse_population_loss_weights(job["lossWeights"]))
    epochs_trained = result.get("epochsTrained")
    stopped_by_patience = result.get(
        "stoppedByPatience",
        result.get("finalizedEarly") is True,
    )
    training_finished = (
        stopped_by_patience is True
        or (
            isinstance(epochs_trained, int)
            and epochs_trained == common["epochs"]
        )
    )
    population_training = result.get("populationTraining")
    result_max_batches = (
        population_training.get("maxBatchesPerEpoch")
        if isinstance(population_training, dict)
        else None
    )
    result_minimum_improvement = result.get("minimumImprovement", 1e-6)
    complete = (
        result.get("datasetPlanId") == manifest["planId"]
        and result.get("predictionDelayMs") == int(manifest["predictionDelayMs"])
        and result.get("selectionMetric") == common["selectionMetric"]
        and result.get("epochs") == common["epochs"]
        and result.get("validationFraction") == common["validationFraction"]
        and result.get("patience") == common["patience"]
        and isinstance(result_minimum_improvement, (int, float))
        and float(result_minimum_improvement)
            == float(common.get("minimumImprovement", 1e-6))
        and result_max_batches == common.get("maxBatchesPerEpoch")
        and result.get("seed") == common["seed"]
        and training_finished
        and result.get("lossWeights") == expected_weights
        and isinstance(result.get("bestValidationMetrics"), dict)
    )
    if common.get("retainAllBestModels", False):
        complete = complete and (Path(job["output"]) / "best-model.pt").is_file()
    parent = job.get("initializeFromCheckpoint")
    if parent is not None:
        complete = complete and isinstance(population_training, dict) \
            and population_training.get("parentCheckpoint") == parent \
            and population_training.get("parentKey") == job.get("parentKey")
    return complete


def population_checkpoint_file(jobs_file: Path, keys: list[str]) -> Path:
    digest = hashlib.sha256("\0".join(keys).encode()).hexdigest()[:12]
    return jobs_file.with_name(f"{jobs_file.stem}-{digest}.population-checkpoint.pt")


def cleanup_population_checkpoints(jobs_file: Path) -> None:
    for checkpoint in jobs_file.parent.glob(
        f"{jobs_file.stem}-*.population-checkpoint.pt"
    ):
        checkpoint.unlink(missing_ok=True)


def completed_winner(
    jobs: list[dict],
    metric: str,
    manifest: dict,
    common: dict,
) -> tuple[float, dict] | None:
    completed = []
    for job in jobs:
        if not completed_job(job, manifest, common):
            continue
        result_file = Path(job["resultFile"])
        try:
            result = json.loads(result_file.read_text())
            score = float(result["bestValidationMetrics"][metric])
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
        if math.isfinite(score):
            completed.append((score, job))
    if not completed:
        return None
    return min(completed, key=lambda item: item[0])


def load_parent_state(
    job: dict,
    prototype: ExposureMlp,
) -> dict[str, Tensor]:
    checkpoint = job.get("initializeFromCheckpoint")
    if checkpoint is None:
        return {
            name: value.detach().cpu().clone()
            for name, value in prototype.state_dict().items()
        }
    if Path(checkpoint).suffix == ".onnx":
        model = onnx.load(checkpoint, load_external_data=True)
        initializers = {
            value.name: torch.from_numpy(
                numpy_helper.to_array(value).copy()
            )
            for value in model.graph.initializer
        }
        expected = prototype.state_dict()
        missing = set(expected) - set(initializers)
        if missing:
            raise ValueError(
                f"ONNX parent is missing model state {sorted(missing)}: {checkpoint}"
            )
        return {
            name: initializers[name].to(dtype=value.dtype)
            for name, value in expected.items()
        }
    value = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if isinstance(value, dict) and isinstance(value.get("model"), dict):
        value = value["model"]
    if not isinstance(value, dict) or not all(
        isinstance(name, str) and isinstance(tensor, Tensor)
        for name, tensor in value.items()
    ):
        raise ValueError(f"parent checkpoint is not a model state: {checkpoint}")
    return value


def parent_checkpoint_identity(job: dict) -> dict | None:
    checkpoint = job.get("initializeFromCheckpoint")
    if checkpoint is None:
        return None
    file = Path(checkpoint).resolve()
    stat = file.stat()
    return {
        "path": str(file),
        "size": stat.st_size,
        "modifiedNs": stat.st_mtime_ns,
        "parentKey": job.get("parentKey"),
    }


if __name__ == "__main__":
    main()

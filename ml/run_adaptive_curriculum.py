from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from adaptive_curriculum import (
    LOSS_TERMS,
    DelayContinuation,
    ProjectedQuadratic,
    WeightCandidate,
    candidate_training_batches,
    candidate_matrix,
    enumerate_weight_candidates,
    initial_delay_continuation,
    interpolate_delay_reference,
    one_coordinate_neighbor_keys,
    propose_delay_seconds,
    recommended_schedule,
    schedule_pareto_front,
    schedule_point,
    select_gp_acquisition_indices,
    update_delay_continuation,
)
from run_dynamic_curriculum import (
    atomic_json,
    candidate_score,
    collapse_equivalent_candidates,
    pareto_beam,
    read_json,
    source_parent_rows,
    utc_now,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a resumable absolute-loss-weight and arbitrary-delay MLP "
            "curriculum with projected screening and measured promotion."
        ),
    )
    parser.add_argument("--plan", type=Path, default=Path("ml/training-plan.json"))
    parser.add_argument(
        "--study-key",
        default="adaptiveCurriculumStudy",
        help="Training-plan section containing the adaptive study configuration.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--maximum-rounds",
        type=int,
        help="Temporarily cap this invocation without changing the persisted plan.",
    )
    return parser.parse_args()


def stable_digest(*values: object, length: int = 12) -> str:
    encoded = "\0".join(str(value) for value in values).encode()
    return hashlib.sha256(encoded).hexdigest()[:length]


def public_job(job: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in job.items()
        if not key.startswith("_")
    }


def candidates_in_persisted_order(
    candidates: Sequence[dict[str, Any]],
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    """Recover a ranked candidate list without changing population grouping."""
    candidates_by_key = {
        candidate["key"]: candidate
        for candidate in candidates
    }
    return [
        candidates_by_key[key]
        for key in keys
        if key in candidates_by_key
    ]


def constrained_parent_weight_candidate(
    weights: dict[str, Any],
    fixed_loss_weights: dict[str, Any] | None,
) -> WeightCandidate:
    resolved = {
        term: float(weights.get(term, 0))
        for term in LOSS_TERMS
    }
    resolved.update({
        str(term): float(value)
        for term, value in (fixed_loss_weights or {}).items()
    })
    if set(resolved) != set(LOSS_TERMS) or any(
        not math.isfinite(value) or value < 0
        for value in resolved.values()
    ):
        raise ValueError("initial-parent loss weights are invalid")
    if not any(resolved[term] > 0 for term in (
        "crossEntropy",
        "probabilityMse",
        "parameterMse",
    )):
        raise ValueError(
            "initial-parent loss weights lack a distribution-matching term"
        )
    values = tuple(resolved[term] for term in LOSS_TERMS)
    return WeightCandidate(
        key=(
            "initial-"
            + stable_digest(
                "constrained-initial-parent-v1",
                json.dumps(resolved, sort_keys=True),
            )
        ),
        values=values,
    )


def merge_configuration(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            isinstance(value, dict)
            and isinstance(merged.get(key), dict)
        ):
            merged[key] = merge_configuration(merged[key], value)
        else:
            merged[key] = value
    return merged


def finite_validation(candidate: dict[str, Any]) -> float:
    return float(candidate["validation"]["klDivergence"])


def unlink_branch_model(branches_dir: Path, model_file: Path) -> int:
    """Delete only a materialized adaptive branch checkpoint."""
    root = branches_dir.resolve()
    model = model_file.resolve()
    if (
        model.name != "best-model.pt"
        or not model.is_relative_to(root)
        or not model.is_file()
    ):
        return 0
    size = model.stat().st_size
    model.unlink()
    return size


def sweep_branch_models(
    branches_dir: Path,
    retained_models: Sequence[Path],
    active_round_prefix: str | None,
) -> dict[str, Any]:
    """Remove obsolete weights while preserving metrics and active resume state."""
    root = branches_dir.resolve()
    retained = {
        model.resolve()
        for model in retained_models
        if model.name == "best-model.pt"
    }
    removed_models = 0
    removed_bytes = 0
    retained_count = 0
    for model in root.glob("*/best-model.pt"):
        resolved = model.resolve()
        active = (
            active_round_prefix is not None
            and model.parent.name.startswith(active_round_prefix)
        )
        if resolved in retained or active:
            retained_count += 1
            continue
        removed_bytes += unlink_branch_model(root, model)
        removed_models += 1
    return {
        "activeRoundPrefix": active_round_prefix,
        "removedModels": removed_models,
        "removedBytes": removed_bytes,
        "retainedModels": retained_count,
    }


def delay_state_dict(state: DelayContinuation) -> dict[str, Any]:
    return {
        "anchor_delay_seconds": state.anchor_delay_seconds,
        "trial_delay_seconds": state.trial_delay_seconds,
        "step_seconds": state.step_seconds,
        "dwell_epochs": state.dwell_epochs,
        "stale_epochs": state.stale_epochs,
        "best_validation": (
            state.best_validation
            if math.isfinite(state.best_validation)
            else None
        ),
    }


def delay_state_from_dict(value: dict[str, Any]) -> DelayContinuation:
    return DelayContinuation(
        anchor_delay_seconds=int(value["anchor_delay_seconds"]),
        trial_delay_seconds=int(value["trial_delay_seconds"]),
        step_seconds=int(value["step_seconds"]),
        dwell_epochs=int(value["dwell_epochs"]),
        stale_epochs=int(value["stale_epochs"]),
        best_validation=(
            float(value["best_validation"])
            if value.get("best_validation") is not None
            else math.inf
        ),
    )


def select_promotions(
    candidates: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    """Keep the best child of every search parent, then fill by validation KL."""
    ordered = sorted(candidates, key=candidate_score)
    selected: list[dict[str, Any]] = []
    parent_keys: set[str] = set()
    for candidate in ordered:
        parent_key = str(candidate["searchParentKey"])
        if parent_key in parent_keys:
            continue
        selected.append(candidate)
        parent_keys.add(parent_key)
    if len(selected) > count:
        selected = sorted(selected, key=candidate_score)[:count]
    selected_keys = {candidate["key"] for candidate in selected}
    selected.extend(
        candidate
        for candidate in ordered
        if candidate["key"] not in selected_keys
    )
    return selected[: min(count, len(selected))]


class AdaptiveCurriculumRunner:
    def __init__(
        self,
        plan_file: Path,
        study_key: str = "adaptiveCurriculumStudy",
    ) -> None:
        self.plan_file = plan_file.resolve()
        self.repo = self.plan_file.parent.parent
        self.plan = read_json(self.plan_file)
        self.study_key = study_key
        configured = self.plan[study_key]
        inherited_key = configured.get("extends")
        if inherited_key is None:
            self.config = configured
        else:
            if inherited_key == study_key:
                raise ValueError("an adaptive study cannot extend itself")
            self.config = merge_configuration(
                self.plan[inherited_key],
                configured,
            )
            self.config.pop("extends", None)
        self.output = (self.repo / self.config["outputDir"]).resolve()
        self.run_dir = (self.repo / self.config["runDir"]).resolve()
        self.dataset = (self.repo / self.config["datasetDir"]).resolve()
        self.source_summary_file = (
            self.repo / self.config["sourceSummary"]
        ).resolve()
        self.source_priority_plan_file = (
            self.repo / self.config["sourcePriorityPlan"]
        ).resolve()
        bootstrap_summary = self.config.get("bootstrapSummary")
        self.bootstrap_summary_file = (
            (self.repo / bootstrap_summary).resolve()
            if bootstrap_summary
            else None
        )
        self.source = read_json(self.source_summary_file)
        self.summary_file = self.output / "summary.json"
        self.status_file = self.run_dir / "status.json"
        self.log_file = self.run_dir / "study.log"
        self.plans_dir = self.output / "plans"
        self.jobs_dir = self.run_dir / "population-jobs"
        self.projections_dir = self.output / "projections"
        self.statistics_dir = self.output / "statistics"
        self.branches_dir = self.output / "branches"
        self.interrupted = False
        self.status: dict[str, Any] = {
            "pid": os.getpid(),
            "planId": self.plan["id"],
            "studyKey": self.study_key,
            "stage": "initializing",
            "startedAt": utc_now(),
            "updatedAt": utc_now(),
            "sourceSummary": str(self.source_summary_file),
        }
        self.candidates = enumerate_weight_candidates(
            self.config["absoluteWeightLevels"],
            canonicalize_global_scale=bool(
                self.config.get("canonicalizeGlobalScale", True)
            ),
            fixed_loss_weights=self.config.get("fixedLossWeights"),
            term_weight_levels=self.config.get("termWeightLevels"),
        )
        self.raw_candidates = enumerate_weight_candidates(
            self.config["absoluteWeightLevels"],
            canonicalize_global_scale=False,
            fixed_loss_weights=self.config.get("fixedLossWeights"),
            term_weight_levels=self.config.get("termWeightLevels"),
        )
        self.candidate_by_key = {
            candidate.key: candidate
            for candidate in self.candidates
        }
        self.candidate_index = {
            candidate.key: index
            for index, candidate in enumerate(self.candidates)
        }
        self.candidate_values = candidate_matrix(self.candidates)
        self.candidate_space_digest = stable_digest(
            "adaptive-weight-space-v1",
            *(candidate.key for candidate in self.candidates),
            length=16,
        )
        self.search_design_digest = stable_digest(
            "adaptive-search-design-v1",
            self.candidate_space_digest,
            json.dumps({
                "projection": self.config["projection"],
                "fidelities": self.config["fidelities"],
                "beamWidth": self.config["beamWidth"],
                "populationSize": self.config["populationSize"],
                "initialDelaySeconds": self.config["initialDelaySeconds"],
                "initialStepSeconds": self.config["initialStepSeconds"],
                "bootstrapSummary": self.config.get("bootstrapSummary"),
                "initialWeightVariants": self.config.get(
                    "initialWeightVariants"
                ),
                "trainInitialParentsUntilPlateau": bool(
                    self.config.get(
                        "trainInitialParentsUntilPlateau",
                        False,
                    )
                ),
                "initialParentApplyFixedLossWeights": bool(
                    self.config.get(
                        "initialParentApplyFixedLossWeights",
                        True,
                    )
                ),
            }, sort_keys=True),
            length=16,
        )
        self.round_design_digest = stable_digest(
            "adaptive-round-design-v1",
            self.search_design_digest,
            json.dumps(
                self.config.get("neighborhoodExpansion", {}),
                sort_keys=True,
            ),
            length=16,
        )
        self.validate()

    def validate(self) -> None:
        if int(self.config["minimumDelaySeconds"]) < 1:
            raise ValueError("minimumDelaySeconds must be at least one second")
        if (
            int(self.config["initialDelaySeconds"])
            < int(self.config["minimumDelaySeconds"])
        ):
            raise ValueError("initialDelaySeconds is below the minimum delay")
        for key in (
            "initialStepSeconds",
            "maximumRounds",
            "initialParentCount",
            "beamWidth",
            "populationSize",
            "workers",
        ):
            if int(self.config[key]) < 1:
                raise ValueError(f"{key} must be positive")
        projection = self.config["projection"]
        for key in ("batchSize", "initialProbeCount", "adaptiveProbeCount"):
            if int(projection[key]) < 1:
                raise ValueError(f"projection.{key} must be positive")
        fidelities = self.config["fidelities"]
        if not fidelities or fidelities[-1]["key"] != "full":
            raise ValueError("the last adaptive fidelity must be full")
        for index, item in enumerate(fidelities):
            if (
                int(item["promote"]) < 1
                or not 0 < float(item["validationFraction"]) <= 1
            ):
                raise ValueError("adaptive fidelity configuration is invalid")
            if bool(item.get("trainUntilPlateau", False)):
                if index != len(fidelities) - 1:
                    raise ValueError(
                        "only the full adaptive fidelity may train until plateau"
                    )
                maximum_epochs = int(item.get("maximumEpochs", 0))
                patience = int(item.get("patience", 0))
                minimum_improvement = float(
                    item.get("minimumImprovement", math.nan)
                )
                if (
                    maximum_epochs < 1
                    or patience < 1
                    or patience > maximum_epochs
                    or not math.isfinite(minimum_improvement)
                    or minimum_improvement < 0
                ):
                    raise ValueError(
                        "adaptive plateau fidelity configuration is invalid"
                    )
            elif int(item.get("additionalBatches", 0)) < 1:
                raise ValueError("adaptive fidelity configuration is invalid")
        if any(
            int(fidelities[index]["promote"])
            > int(fidelities[index - 1]["promote"])
            for index in range(1, len(fidelities))
        ):
            raise ValueError("fidelity promotion widths must be non-increasing")
        if (
            int(self.source.get("completedRuns", 0))
            != int(self.source.get("plannedRuns", -1))
        ):
            raise RuntimeError("the source response study is not complete")
        if not self.candidates:
            raise ValueError("the canonical absolute-weight space is empty")
        if (
            self.bootstrap_summary_file is not None
            and not self.bootstrap_summary_file.is_file()
        ):
            raise FileNotFoundError(
                f"bootstrap curriculum summary is missing: "
                f"{self.bootstrap_summary_file}"
            )
        if float(self.config.get("stepGrowthFactor", 1.5)) <= 1:
            raise ValueError("stepGrowthFactor must be greater than one")
        if int(self.config.get("bootstrapStepSeconds", 1)) < 1:
            raise ValueError("bootstrapStepSeconds must be positive")
        if not isinstance(
            self.config.get("trainInitialParentsUntilPlateau", False),
            bool,
        ):
            raise ValueError(
                "trainInitialParentsUntilPlateau must be a boolean"
            )
        if not isinstance(
            self.config.get("initialParentApplyFixedLossWeights", True),
            bool,
        ):
            raise ValueError(
                "initialParentApplyFixedLossWeights must be a boolean"
            )
        initial_variants = self.config.get("initialWeightVariants")
        if initial_variants is not None and (
            not isinstance(initial_variants, list)
            or len(initial_variants) != int(self.config["initialParentCount"])
            or len(set(initial_variants)) != len(initial_variants)
            or any(
                not isinstance(variant, str) or not variant
                for variant in initial_variants
            )
        ):
            raise ValueError(
                "initialWeightVariants must uniquely name every initial parent"
            )
        if (
            self.config.get("trainInitialParentsUntilPlateau", False)
            and not bool(fidelities[-1].get("trainUntilPlateau", False))
        ):
            raise ValueError(
                "initial-parent plateau training requires a plateau full fidelity"
            )
        neighborhood = self.config.get("neighborhoodExpansion")
        if neighborhood is not None:
            if not isinstance(neighborhood, dict):
                raise ValueError("neighborhoodExpansion must be an object")
            if not isinstance(neighborhood.get("enabled", False), bool):
                raise ValueError("neighborhoodExpansion.enabled must be a boolean")
            if neighborhood.get("enabled", False) and (
                int(neighborhood.get("seedCountPerParent", 0)) < 1
                or int(neighborhood.get("maximumRounds", 0)) < 1
                or not math.isfinite(float(
                    neighborhood.get(
                        "minimumValidationKlImprovement",
                        math.nan,
                    )
                ))
                or float(
                    neighborhood["minimumValidationKlImprovement"]
                ) < 0
            ):
                raise ValueError(
                    "enabled neighborhoodExpansion configuration is invalid"
                )

    def prepare_directories(self) -> None:
        for directory in (
            self.output,
            self.run_dir,
            self.plans_dir,
            self.jobs_dir,
            self.projections_dir,
            self.statistics_dir,
            self.branches_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    def write_status(self, **updates: Any) -> None:
        self.status = {
            **self.status,
            **updates,
            "updatedAt": utc_now(),
        }
        atomic_json(self.status, self.status_file)

    def append_log(self, value: str) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a") as stream:
            stream.write(value.rstrip() + "\n")

    def run_child(self, stage: str, command: list[str]) -> None:
        if self.interrupted:
            raise KeyboardInterrupt
        self.write_status(
            stage=stage,
            command=command,
            message=f"Running {stage}.",
        )
        self.append_log(json.dumps({
            "event": "stage-start",
            "stage": stage,
            "command": command,
        }))
        process = subprocess.Popen(
            command,
            cwd=self.repo,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            start_new_session=True,
            env=os.environ.copy(),
        )
        self.write_status(childPid=process.pid)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                self.append_log(line)
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self.write_status(latest=event)
        except KeyboardInterrupt:
            os.killpg(process.pid, signal.SIGTERM)
            raise
        exit_code = process.wait()
        self.write_status(childPid=None, childExitCode=exit_code)
        if exit_code != 0:
            raise RuntimeError(f"{stage} exited with code {exit_code}")

    def direct_references(self) -> dict[int, float]:
        references: dict[int, float] = {}
        for row in self.source["results"]:
            delay_seconds = int(row["delayMs"]) // 1_000
            validation = float(
                row.get("validation", {}).get("klDivergence", math.inf)
            )
            if math.isfinite(validation):
                references[delay_seconds] = min(
                    references.get(delay_seconds, math.inf),
                    validation,
                )
        return references

    def transition_candidate(
        self,
        candidates: Sequence[dict[str, Any]],
        reference_kl: float,
    ) -> dict[str, Any]:
        tolerance = max(
            float(self.config["dwell"]["absoluteReferenceTolerance"]),
            float(self.config["dwell"]["relativeReferenceTolerance"])
            * reference_kl,
        )
        feasible = [
            candidate
            for candidate in candidates
            if finite_validation(candidate) <= reference_kl + tolerance
        ]
        if feasible:
            return min(
                feasible,
                key=lambda candidate: (
                    candidate_training_batches(candidate),
                    *candidate_score(candidate),
                ),
            )
        return min(candidates, key=candidate_score)

    def update_schedule_optimization(
        self,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        candidates_by_key: dict[str, dict[str, Any]] = {}
        for candidate in (
            *summary.get("anchorParents", []),
            *summary.get("trialParents", []),
        ):
            candidates_by_key[candidate["key"]] = candidate
        for accepted in summary.get("acceptedDelays", []):
            for candidate in accepted.get("survivors", []):
                candidates_by_key[candidate["key"]] = candidate
        for round_summary in summary.get("rounds", []):
            for candidate in round_summary.get("survivors", []):
                candidates_by_key[candidate["key"]] = candidate
        references = self.direct_references()
        training = self.plan["training"]
        points = []
        for candidate in candidates_by_key.values():
            delay_seconds = int(candidate["delayMs"]) // 1_000
            points.append(schedule_point(
                candidate,
                initial_delay_seconds=int(self.config["initialDelaySeconds"]),
                reference_kl=interpolate_delay_reference(
                    delay_seconds,
                    references,
                ),
                absolute_quality_tolerance=float(
                    self.config["dwell"]["absoluteReferenceTolerance"]
                ),
                relative_quality_tolerance=float(
                    self.config["dwell"]["relativeReferenceTolerance"]
                ),
                batch_size=int(training["batchSize"]),
                gradient_accumulation=int(
                    training["gradientAccumulation"]
                ),
            ))
        frontier = schedule_pareto_front(points)
        recommended = recommended_schedule(points)
        recommended_candidate = (
            candidates_by_key[recommended["key"]]
            if recommended is not None
            else None
        )
        summary["scheduleOptimization"] = {
            "version": 1,
            "objective": (
                "furthest delay reduction within the direct-reference KL "
                "tolerance; then minimum realized training batches; then "
                "minimum worst positive KL gap and KL standard deviation"
            ),
            "effortScope": (
                "training of the realized final-model lineage only; "
                "hyperparameter-search probes are excluded"
            ),
            "evaluatedSchedules": len(points),
            "paretoFront": frontier,
            "recommended": (
                {
                    **recommended,
                    "weights": recommended_candidate["weights"],
                    "lineage": recommended_candidate.get("lineage", []),
                }
                if recommended is not None
                and recommended_candidate is not None
                else None
            ),
        }
        return summary

    def initial_parents(self) -> list[dict[str, Any]]:
        delay_seconds = int(self.config["initialDelaySeconds"])
        variants = self.config.get("initialWeightVariants")
        if variants is None:
            rows = source_parent_rows(
                self.source,
                delay_seconds * 1_000,
                int(self.config["initialParentCount"]),
            )
        else:
            rows_by_variant = {
                str(row["weightVariant"]): row
                for row in self.source["results"]
                if int(row["delayMs"]) == delay_seconds * 1_000
            }
            missing = [
                variant
                for variant in variants
                if variant not in rows_by_variant
            ]
            if missing:
                raise RuntimeError(
                    f"the source study lacks initial variants: {missing}"
                )
            rows = [rows_by_variant[variant] for variant in variants]
        if len(rows) != int(self.config["initialParentCount"]):
            raise RuntimeError(
                "the source study lacks enough initial-delay parent models"
            )
        parents = []
        for row in rows:
            directory = Path(row["resultFile"]).parent
            model = directory / "model.onnx"
            if not model.is_file():
                raise FileNotFoundError(f"source parent model is missing: {model}")
            parents.append({
                "key": (
                    f"source-{delay_seconds}s-"
                    f"{row['weightVariant']}"
                ),
                "delayMs": delay_seconds * 1_000,
                "model": str(model),
                "directory": str(directory),
                "resultFile": row["resultFile"],
                "weights": row["weights"],
                "validation": row["validation"],
                "sourceResultFile": row["resultFile"],
                "lineage": [],
            })
        return parents

    def plan_for_delay(self, delay_seconds: int) -> tuple[dict[str, Any], Path]:
        source = read_json(self.source_priority_plan_file)
        selection = dict(source["exampleSelection"])
        coverage = {
            int(value)
            for value in selection.get(
                "componentCoveragePredictionDelaysMs",
                [],
            )
        }
        coverage.add(delay_seconds * 1_000)
        selection["componentCoveragePredictionDelaysMs"] = sorted(coverage)
        key = f"delay-{delay_seconds}s"
        plan = {
            **source,
            "id": f"{self.plan['id']}-adaptive-{key}",
            "label": (
                self.plan["label"].replace(" · 60-second policy delay", "")
                + f" · adaptive curriculum {delay_seconds / 60:g}m"
            ),
            "predictionDelayMs": delay_seconds * 1_000,
            "exampleSelection": selection,
            "artifactDir": str(
                (self.output / key / "artifact-unused").relative_to(self.repo)
            ),
            "runDir": str(
                (self.output / key / "run-unused").relative_to(self.repo)
            ),
        }
        file = self.plans_dir / f"{key}.json"
        atomic_json(plan, file)
        return plan, file

    def pair_delay(self, plan_file: Path, delay_seconds: int) -> None:
        manifest_file = self.dataset / "dataset.json"
        plan = read_json(plan_file)
        if manifest_file.is_file():
            manifest = read_json(manifest_file)
            if (
                manifest.get("planId") == plan["id"]
                and int(manifest.get("predictionDelayMs", -1))
                == delay_seconds * 1_000
            ):
                return
        self.run_child(
            f"pair-delay-{delay_seconds}s",
            [
                "node",
                "scripts/run-node-with-ml-libs.mjs",
                "node_modules/tsx/dist/cli.mjs",
                "scripts/build-mlp-dataset.ts",
                "--plan",
                str(plan_file),
            ],
        )

    def summary(self) -> dict[str, Any]:
        if self.summary_file.is_file():
            return read_json(self.summary_file)
        bootstrap = (
            read_json(self.bootstrap_summary_file)
            if self.bootstrap_summary_file is not None
            else None
        )
        if bootstrap is None:
            state = initial_delay_continuation(
                int(self.config["initialDelaySeconds"]),
                int(self.config["minimumDelaySeconds"]),
                int(self.config["initialStepSeconds"]),
            )
            anchor_parents = self.initial_parents()
            trial_parents = anchor_parents
            accepted_delays = [{
                "delaySeconds": int(self.config["initialDelaySeconds"]),
                "reason": "source parents",
                "survivors": anchor_parents,
            }]
            bootstrap_best = None
        else:
            if (
                bootstrap.get("sourceExperimentFingerprint")
                != self.source["experimentFingerprint"]
            ):
                raise RuntimeError(
                    "bootstrap curriculum and source response study differ"
                )
            state = delay_state_from_dict(bootstrap["state"])
            if "bootstrapStepSeconds" in self.config:
                bootstrap_step = int(self.config["bootstrapStepSeconds"])
                state = DelayContinuation(
                    anchor_delay_seconds=state.anchor_delay_seconds,
                    trial_delay_seconds=propose_delay_seconds(
                        state.anchor_delay_seconds,
                        bootstrap_step,
                        int(self.config["minimumDelaySeconds"]),
                        1,
                    )[0],
                    step_seconds=bootstrap_step,
                    dwell_epochs=0,
                    stale_epochs=0,
                    best_validation=math.inf,
                )
            anchor_parents = bootstrap["anchorParents"]
            trial_parents = bootstrap["anchorParents"]
            accepted_delays = bootstrap["acceptedDelays"]
            bootstrap_best = bootstrap.get("bestObserved")
        summary = {
            "version": 1,
            "planId": self.plan["id"],
            "studyKey": self.study_key,
            "sourceSummary": str(self.source_summary_file),
            "sourceExperimentFingerprint": self.source["experimentFingerprint"],
            "absoluteWeightLevels": self.config["absoluteWeightLevels"],
            "fixedLossWeights": self.config.get("fixedLossWeights", {}),
            "termWeightLevels": self.config.get("termWeightLevels", {}),
            "canonicalizeGlobalScale": bool(
                self.config.get("canonicalizeGlobalScale", True)
            ),
            "rawAnchorValidWeightTuples": len(self.raw_candidates),
            "canonicalWeightCandidates": len(self.candidates),
            "candidateSpaceDigest": self.candidate_space_digest,
            "searchDesignDigest": self.search_design_digest,
            "roundDesignDigest": self.round_design_digest,
            "directDelayReferences": {
                str(key): value
                for key, value in self.direct_references().items()
            },
            "state": delay_state_dict(state),
            "anchorParents": anchor_parents,
            "trialParents": trial_parents,
            "rounds": [],
            "acceptedDelays": accepted_delays,
            "completedRounds": 0,
            "generatedAt": utc_now(),
        }
        if bootstrap is not None:
            summary.update({
                "bootstrapSummary": str(self.bootstrap_summary_file),
                "bootstrapCompletedRounds": int(
                    bootstrap.get("completedRounds", 0)
                ),
                "bootstrapBestObserved": bootstrap_best,
                "bestObserved": bootstrap_best,
            })
        return self.update_schedule_optimization(summary)

    def train_initial_parents_until_plateau(
        self,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.config.get("trainInitialParentsUntilPlateau", False):
            return summary
        completed = summary.get("initialParentPlateau")
        if isinstance(completed, dict) and completed.get("complete") is True:
            return summary
        if summary.get("rounds"):
            raise RuntimeError(
                "initial parents cannot be retrained after adaptive rounds"
            )
        parents = list(summary["anchorParents"])
        if not parents:
            raise RuntimeError("there are no initial parents to train")
        full_fidelity = self.config["fidelities"][-1]
        trained_by_parent: dict[str, dict[str, Any]] = {}
        delays = sorted({
            int(parent["delayMs"]) // 1_000
            for parent in parents
        })
        for delay_seconds in delays:
            delay_parents = [
                parent
                for parent in parents
                if int(parent["delayMs"]) // 1_000 == delay_seconds
            ]
            plan, plan_file = self.plan_for_delay(delay_seconds)
            self.pair_delay(plan_file, delay_seconds)
            trials = []
            for parent in delay_parents:
                trials.append({
                    "_candidate": constrained_parent_weight_candidate(
                        parent["weights"],
                        (
                            self.config.get("fixedLossWeights")
                            if self.config.get(
                                "initialParentApplyFixedLossWeights",
                                True,
                            )
                            else None
                        ),
                    ),
                    "_initialize": parent,
                    "_searchParent": parent,
                })
            trained = self.train_fidelity(
                -1,
                delay_seconds,
                full_fidelity,
                trials,
                plan,
                plan_file,
                "initial-parents",
            )
            for candidate in trained:
                parent = candidate["_searchParent"]
                candidate["lineage"] = [
                    *parent.get("lineage", []),
                    {
                        "round": 0,
                        "stage": "initial-parent-plateau",
                        "delaySeconds": delay_seconds,
                        "weightCandidate": candidate["weightCandidate"],
                        "absoluteLossWeights": candidate["weights"],
                        "trainingBatches": int(
                            candidate["transitionTrainingBatches"]
                        ),
                        "trainingExamples": int(
                            candidate["transitionTrainingExamples"]
                        ),
                        "optimizerUpdates": int(
                            candidate["transitionOptimizerUpdates"]
                        ),
                        "trainingStages": list(
                            candidate["transitionStages"]
                        ),
                        "plateauEpochsTrained": int(
                            candidate["stageEpochsTrained"]
                        ),
                        "plateauReached": bool(
                            candidate["stagePlateauReached"]
                        ),
                    },
                ]
                candidate.pop("_candidate", None)
                candidate.pop("_searchParent", None)
                trained_by_parent[parent["key"]] = candidate

        converged = [
            trained_by_parent[parent["key"]]
            for parent in parents
        ]
        old_parent_keys = {parent["key"] for parent in parents}
        replaced_accepted_delay = False
        for accepted in reversed(summary["acceptedDelays"]):
            accepted_keys = {
                candidate["key"]
                for candidate in accepted.get("survivors", [])
            }
            if accepted_keys == old_parent_keys:
                accepted["survivors"] = converged
                accepted["reason"] = (
                    f"{accepted.get('reason', 'initial parents')}; "
                    "parents trained until plateau"
                )
                replaced_accepted_delay = True
                break
        if not replaced_accepted_delay:
            summary["acceptedDelays"].append({
                "delaySeconds": int(converged[0]["delayMs"]) // 1_000,
                "reason": "initial parents trained until plateau",
                "survivors": converged,
            })
        best = min(converged, key=candidate_score)
        summary["anchorParents"] = converged
        summary["trialParents"] = converged
        summary["bootstrapBestObserved"] = self.serialize_candidates([best])[0]
        summary["bestObserved"] = self.serialize_candidates([best])[0]
        summary["initialParentPlateau"] = {
            "complete": True,
            "completedAt": utc_now(),
            "inputParents": [parent["key"] for parent in parents],
            "outputParents": [candidate["key"] for candidate in converged],
            "maximumEpochs": int(full_fidelity["maximumEpochs"]),
            "patience": int(full_fidelity["patience"]),
            "minimumImprovement": float(
                full_fidelity["minimumImprovement"]
            ),
            "parents": [
                {
                    "key": candidate["key"],
                    "validation": candidate["validation"],
                    "weights": candidate["weights"],
                    "epochsTrained": int(
                        candidate["stageEpochsTrained"]
                    ),
                    "plateauReached": bool(
                        candidate["stagePlateauReached"]
                    ),
                }
                for candidate in converged
            ],
        }
        summary["generatedAt"] = utc_now()
        summary = self.update_schedule_optimization(summary)
        atomic_json(summary, self.summary_file)
        self.write_status(
            stage="initial-parent-plateau-complete",
            initialParentPlateau=summary["initialParentPlateau"],
            message=(
                f"Trained {len(converged)} initial parents to independent "
                "validation plateaus."
            ),
        )
        return summary

    def projected_screen(
        self,
        round_index: int,
        delay_seconds: int,
        parent: dict[str, Any],
        plan_file: Path,
    ) -> tuple[ProjectedQuadratic, list[str], dict[str, Any]]:
        digest = stable_digest(
            round_index,
            delay_seconds,
            parent["key"],
            self.search_design_digest,
        )
        specification_file = self.projections_dir / (
            f"r{round_index + 1:03d}-{delay_seconds}s-{digest}.spec.json"
        )
        output_file = self.projections_dir / (
            f"r{round_index + 1:03d}-{delay_seconds}s-{digest}.json"
        )
        training = self.plan["training"]
        projection = self.config["projection"]
        specification = {
            "version": 1,
            "dataset": str(self.dataset),
            "plan": str(plan_file),
            "parent": parent["model"],
            "absoluteWeightLevels": self.config["absoluteWeightLevels"],
            "fixedLossWeights": self.config.get("fixedLossWeights", {}),
            "termWeightLevels": self.config.get("termWeightLevels", {}),
            "candidateSpaceDigest": self.candidate_space_digest,
            "searchDesignDigest": self.search_design_digest,
            "roundDesignDigest": self.round_design_digest,
            "canonicalizeGlobalScale": bool(
                self.config.get("canonicalizeGlobalScale", True)
            ),
            "batchSize": int(projection["batchSize"]),
            "statesPerExample": int(training["statesPerExample"]),
            "dropout": float(training["dropout"]),
            "seed": int(training["seed"]),
            "device": training["device"],
            "workers": int(self.config["workers"]),
            "timeWeighting": training["timeWeighting"],
            "featureStatisticsCache": str(
                self.statistics_dir
                / f"features-{self.source['experimentFingerprint']}.npz"
            ),
            "targetStatisticsCache": str(
                self.statistics_dir
                / (
                    f"delay-{delay_seconds}s-targets-"
                    f"{self.source['experimentFingerprint']}.npz"
                )
            ),
            "learningRate": float(training["learningRate"]),
            "maximumGradientNorm": float(projection["maximumGradientNorm"]),
            "initialProbeCount": int(projection["initialProbeCount"]),
            "exploitationFraction": float(
                projection["exploitationFraction"]
            ),
            "randomFraction": float(projection["randomFraction"]),
        }
        atomic_json(specification, specification_file)
        valid = False
        if output_file.is_file():
            result = read_json(output_file)
            valid = (
                result.get("version") == 2
                and result.get("parent") == parent["model"]
                and int(result.get("delaySeconds", -1)) == delay_seconds
                and int(result.get("candidateCount", -1))
                == len(self.candidates)
                and result.get("candidateSpaceDigest")
                == self.candidate_space_digest
                and result.get("searchDesignDigest")
                == self.search_design_digest
                and len(result.get("selectedProbeKeys", []))
                == int(projection["initialProbeCount"])
            )
        if not valid:
            self.run_child(
                f"project-r{round_index + 1}-{delay_seconds}s-{parent['key']}",
                [
                    sys.executable,
                    str(self.repo / "ml/project_mlp_loss_weights.py"),
                    "--spec",
                    str(specification_file),
                    "--output",
                    str(output_file),
                ],
            )
        result = read_json(output_file)
        model = ProjectedQuadratic(
            base_validation=float(result["baseValidationKl"]),
            linear=np.asarray(result["linear"], dtype=np.float64),
            quadratic=np.asarray(result["quadratic"], dtype=np.float64),
            gradient_gram=np.asarray(result["gradientGram"], dtype=np.float64),
            learning_rate=float(result["learningRate"]),
            maximum_gradient_norm=float(result["maximumGradientNorm"]),
        )
        selected = [
            str(key)
            for key in result["selectedProbeKeys"]
            if key in self.candidate_by_key
        ]
        if not selected:
            raise RuntimeError("projected screen selected no known weight candidates")
        return model, selected, result

    def build_jobs(
        self,
        round_index: int,
        delay_seconds: int,
        fidelity: dict[str, Any],
        trials: Sequence[dict[str, Any]],
        plan: dict[str, Any],
        plan_file: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        tier = str(fidelity["key"])
        train_until_plateau = bool(
            fidelity.get("trainUntilPlateau", False)
        )
        jobs: list[dict[str, Any]] = []
        for trial in trials:
            candidate: WeightCandidate = trial["_candidate"]
            parent = trial["_initialize"]
            search_parent = trial["_searchParent"]
            digest = stable_digest(
                round_index,
                delay_seconds,
                tier,
                search_parent["key"],
                candidate.key,
            )
            key = f"r{round_index + 1:03d}-{delay_seconds}s-{tier}-{digest}"
            directory = self.branches_dir / key
            jobs.append({
                "key": key,
                "output": str(directory),
                "resultFile": str(directory / "study.json"),
                "modelId": f"{self.plan['id']}-adaptive-{key}",
                "label": (
                    f"{plan['label']} · round {round_index + 1} "
                    f"· {tier} · {candidate.key}"
                ),
                "lossWeights": candidate.as_dict(),
                "initializeFromCheckpoint": parent["model"],
                "parentKey": parent["key"],
                "_candidate": candidate,
                "_initialize": parent,
                "_searchParent": search_parent,
                "_directory": directory,
                "_configuredTrainingBatches": (
                    None
                    if train_until_plateau
                    else int(fidelity["additionalBatches"])
                ),
            })
        training = self.plan["training"]
        population_groups = math.ceil(
            len(jobs) / int(self.config["populationSize"])
        )
        group_batches = (
            math.inf
            if train_until_plateau
            else population_groups * int(fidelity["additionalBatches"])
        )
        compile_objective = bool(
            self.config.get("compile", training.get("compile", False))
        ) and group_batches >= int(
            self.config.get("compileMinimumGroupBatches", 1)
        )
        specification = {
            "version": 1,
            "common": {
                "dataset": str(self.dataset),
                "plan": str(plan_file),
                "epochs": (
                    int(fidelity["maximumEpochs"])
                    if train_until_plateau
                    else 1
                ),
                "batchSize": int(training["batchSize"]),
                "evaluationBatchSize": int(training["evaluationBatchSize"]),
                "validationFraction": float(fidelity["validationFraction"]),
                "accumulate": int(training["gradientAccumulation"]),
                "learningRate": float(training["learningRate"]),
                "weightDecay": float(training["weightDecay"]),
                "dropout": float(training["dropout"]),
                "statesPerExample": int(training["statesPerExample"]),
                "patience": (
                    int(fidelity["patience"])
                    if train_until_plateau
                    else 1
                ),
                "minimumImprovement": (
                    float(fidelity["minimumImprovement"])
                    if train_until_plateau
                    else 1e-6
                ),
                "workers": int(self.config["workers"]),
                "seed": int(training["seed"]),
                "device": training["device"],
                "logEverySteps": int(training["logEverySteps"]),
                "timeWeighting": training["timeWeighting"],
                "selectionMetric": "klDivergence",
                "featureStatisticsCache": str(
                    self.statistics_dir
                    / f"features-{self.source['experimentFingerprint']}.npz"
                ),
                "targetStatisticsCache": str(
                    self.statistics_dir
                    / (
                        f"delay-{delay_seconds}s-targets-"
                        f"{self.source['experimentFingerprint']}.npz"
                    )
                ),
                "compile": compile_objective,
                "checkpointEveryEpochs": 1 if train_until_plateau else 0,
                "retainAllBestModels": True,
                **(
                    {"equivalenceSignature": self.config["equivalenceSignature"]}
                    if tier == "full"
                    else {}
                ),
                **(
                    {}
                    if train_until_plateau
                    else {
                        "maxBatchesPerEpoch":
                            int(fidelity["additionalBatches"])
                    }
                ),
            },
            "jobs": [public_job(job) for job in jobs],
        }
        return specification, jobs

    def train_fidelity(
        self,
        round_index: int,
        delay_seconds: int,
        fidelity: dict[str, Any],
        trials: Sequence[dict[str, Any]],
        plan: dict[str, Any],
        plan_file: Path,
        phase: str = "",
    ) -> list[dict[str, Any]]:
        if not trials:
            return []
        suffix = f"-{phase}" if phase else ""
        jobs_file = self.jobs_dir / (
            f"round-{round_index + 1:03d}-{delay_seconds}s-"
            f"{fidelity['key']}{suffix}-"
            f"{self.search_design_digest[:8]}.json"
        )
        specification, jobs = self.build_jobs(
            round_index,
            delay_seconds,
            fidelity,
            trials,
            plan,
            plan_file,
        )
        atomic_json(specification, jobs_file)
        self.write_status(
            stage="adaptive-population-training",
            adaptiveRound=round_index + 1,
            delaySeconds=delay_seconds,
            fidelity=fidelity["key"],
            fidelityPhase=phase or None,
            branches=len(jobs),
            message=(
                f"Training {len(jobs)} {fidelity['key']} probes "
                f"at {delay_seconds}s delay."
            ),
        )
        self.run_child(
            (
                f"adaptive-r{round_index + 1}-{delay_seconds}s-"
                f"{fidelity['key']}{suffix}"
            ),
            [
                sys.executable,
                str(self.repo / "ml/run_with_observability.py"),
                str(self.repo / "ml/train_mlp_population.py"),
                "--jobs",
                str(jobs_file),
                "--population-size",
                str(self.config["populationSize"]),
            ],
        )
        return [
            self.collect_candidate(
                round_index,
                delay_seconds,
                str(fidelity["key"]),
                job,
            )
            for job in jobs
        ]

    def collect_candidate(
        self,
        round_index: int,
        delay_seconds: int,
        fidelity: str,
        job: dict[str, Any],
    ) -> dict[str, Any]:
        result_file = Path(job["resultFile"])
        result = read_json(result_file)
        candidate: WeightCandidate = job["_candidate"]
        search_parent = job["_searchParent"]
        initialize = job["_initialize"]
        metadata = {
            "version": 1,
            "round": round_index + 1,
            "delaySeconds": delay_seconds,
            "fidelity": fidelity,
            "searchParentKey": search_parent["key"],
            "initializeParentKey": initialize["key"],
            "weightCandidate": candidate.key,
            "absoluteLossWeights": candidate.as_dict(),
        }
        result["adaptiveCurriculumTraining"] = metadata
        atomic_json(result, result_file)
        configured_batches = job.get("_configuredTrainingBatches")
        stage_training_batches = int(
            result.get(
                "trainingBatches",
                configured_batches if configured_batches is not None else 0,
            )
        )
        stage_training_examples = int(
            result.get(
                "trainingExamples",
                stage_training_batches
                * int(self.plan["training"]["batchSize"]),
            )
        )
        stage_optimizer_updates = int(
            result.get(
                "optimizerUpdates",
                math.ceil(
                    stage_training_batches
                    / int(self.plan["training"]["gradientAccumulation"])
                ),
            )
        )
        continuing_same_transition = (
            initialize["key"] != search_parent["key"]
        )
        previous_training_batches = (
            int(initialize.get("transitionTrainingBatches", 0))
            if continuing_same_transition
            else 0
        )
        previous_training_examples = (
            int(initialize.get("transitionTrainingExamples", 0))
            if continuing_same_transition
            else 0
        )
        previous_optimizer_updates = (
            int(initialize.get("transitionOptimizerUpdates", 0))
            if continuing_same_transition
            else 0
        )
        previous_stages = (
            list(initialize.get("transitionStages", []))
            if continuing_same_transition
            else []
        )
        transition_stages = [
            *previous_stages,
            {
                "fidelity": fidelity,
                "weightCandidate": candidate.key,
                "absoluteLossWeights": candidate.as_dict(),
                "trainingBatches": stage_training_batches,
                "trainingExamples": stage_training_examples,
                "optimizerUpdates": stage_optimizer_updates,
                "epochsTrained": int(result.get("epochsTrained", 1)),
                "plateauReached": bool(
                    result.get(
                        "stoppedByPatience",
                        result.get("finalizedEarly", False),
                    )
                ),
            },
        ]
        return {
            "key": job["key"],
            "delayMs": delay_seconds * 1_000,
            "model": str(job["_directory"] / "best-model.pt"),
            "directory": str(job["_directory"]),
            "resultFile": str(result_file),
            "parentKey": initialize["key"],
            "searchParentKey": search_parent["key"],
            "weightCandidate": candidate.key,
            "weights": candidate.as_dict(),
            "validation": result["bestValidationMetrics"],
            "equivalenceSignature": result.get("equivalenceSignature"),
            "lineage": search_parent.get("lineage", []),
            "stageTrainingBatches": stage_training_batches,
            "stageTrainingExamples": stage_training_examples,
            "stageOptimizerUpdates": stage_optimizer_updates,
            "stageEpochsTrained": int(result.get("epochsTrained", 1)),
            "stagePlateauReached": bool(
                result.get(
                    "stoppedByPatience",
                    result.get("finalizedEarly", False),
                )
            ),
            "transitionTrainingBatches": (
                previous_training_batches + stage_training_batches
            ),
            "transitionTrainingExamples": (
                previous_training_examples + stage_training_examples
            ),
            "transitionOptimizerUpdates": (
                previous_optimizer_updates + stage_optimizer_updates
            ),
            "transitionStages": transition_stages,
            "_candidate": candidate,
            "_searchParent": search_parent,
        }

    def initial_trials(
        self,
        parents: Sequence[dict[str, Any]],
        selections: dict[str, list[str]],
    ) -> list[dict[str, Any]]:
        trials = []
        for parent in parents:
            for key in selections[parent["key"]]:
                trials.append({
                    "_candidate": self.candidate_by_key[key],
                    "_initialize": parent,
                    "_searchParent": parent,
                })
        return trials

    def acquired_trials(
        self,
        parents: Sequence[dict[str, Any]],
        projections: dict[str, ProjectedQuadratic],
        screen_candidates: Sequence[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        count = int(self.config["projection"]["adaptiveProbeCount"])
        exploration = float(
            self.config["projection"]["gpExploration"]
        )
        trials: list[dict[str, Any]] = []
        diagnostics: dict[str, Any] = {}
        for parent in parents:
            observed = [
                candidate
                for candidate in screen_candidates
                if candidate["searchParentKey"] == parent["key"]
            ]
            observed_indices = [
                self.candidate_index[candidate["weightCandidate"]]
                for candidate in observed
            ]
            observed_values = [
                finite_validation(candidate)
                for candidate in observed
            ]
            prior = projections[parent["key"]].scores(self.candidate_values)
            selected, prediction, uncertainty = select_gp_acquisition_indices(
                self.candidate_values,
                prior,
                observed_indices,
                observed_values,
                count,
                exploration=exploration,
            )
            diagnostics[parent["key"]] = {
                "observed": len(observed),
                "acquired": len(selected),
                "predictedBestValidationKl": float(np.min(prediction)),
                "maximumPosteriorStdDev": float(np.max(uncertainty)),
            }
            trials.extend({
                "_candidate": self.candidates[index],
                "_initialize": parent,
                "_searchParent": parent,
            } for index in selected)
        return trials, diagnostics

    def continuation_trials(
        self,
        selected: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [{
            "_candidate": candidate["_candidate"],
            "_initialize": candidate,
            "_searchParent": candidate["_searchParent"],
        } for candidate in selected]

    def prune_models(
        self,
        candidates: Sequence[dict[str, Any]],
        retained: Sequence[dict[str, Any]],
    ) -> None:
        retained_models = {
            str(candidate["model"])
            for candidate in retained
        }
        for candidate in candidates:
            if candidate["model"] not in retained_models:
                unlink_branch_model(
                    self.branches_dir,
                    Path(candidate["model"]),
                )

    def retained_model_paths(
        self,
        summary: dict[str, Any],
    ) -> list[Path]:
        candidates = [
            *summary.get("anchorParents", []),
            *summary.get("trialParents", []),
        ]
        for accepted in summary.get("acceptedDelays", []):
            candidates.extend(
                sorted(
                    accepted.get("survivors", []),
                    key=candidate_score,
                )[: int(self.config["finalModelsPerAcceptedDelay"])]
            )
        retained = {
            Path(candidate["model"]).resolve()
            for candidate in candidates
            if str(candidate.get("model", "")).endswith("best-model.pt")
        }
        for artifact in summary.get("artifacts", []):
            retained.add(
                (Path(artifact["directory"]) / "best-model.pt").resolve()
            )
        return sorted(retained)

    def sweep_obsolete_models(
        self,
        summary: dict[str, Any],
    ) -> dict[str, Any]:
        active_round = int(summary.get("completedRounds", 0)) + 1
        active_prefix = None if summary.get("complete") else f"r{active_round:03d}-"
        result = sweep_branch_models(
            self.branches_dir,
            self.retained_model_paths(summary),
            active_prefix,
        )
        result["completedRounds"] = int(summary.get("completedRounds", 0))
        return result

    def serialize_candidates(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [{
            key: value
            for key, value in candidate.items()
            if not key.startswith("_")
        } for candidate in candidates]

    def hydrate_candidates(
        self,
        candidates: Sequence[dict[str, Any]],
        search_parents: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        parent_by_key = {
            parent["key"]: parent
            for parent in search_parents
        }
        hydrated = []
        for stored in candidates:
            candidate = dict(stored)
            candidate["_candidate"] = self.candidate_by_key[
                candidate["weightCandidate"]
            ]
            candidate["_searchParent"] = parent_by_key[
                candidate["searchParentKey"]
            ]
            hydrated.append(candidate)
        return hydrated

    def neighborhood_trials(
        self,
        candidates: Sequence[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        configuration = self.config["neighborhoodExpansion"]
        seed_count = int(configuration["seedCountPerParent"])
        by_parent: dict[str, list[dict[str, Any]]] = {}
        for candidate in candidates:
            by_parent.setdefault(
                str(candidate["searchParentKey"]),
                [],
            ).append(candidate)
        evaluated = {
            (
                str(candidate["searchParentKey"]),
                str(candidate["weightCandidate"]),
            )
            for candidate in candidates
        }
        proposals: dict[
            tuple[str, str],
            tuple[WeightCandidate, dict[str, Any]],
        ] = {}
        for parent_key, parent_candidates in sorted(by_parent.items()):
            seeds = sorted(
                parent_candidates,
                key=candidate_score,
            )[:seed_count]
            for seed in seeds:
                origin = self.candidate_by_key[
                    seed["weightCandidate"]
                ]
                for neighbor_key in one_coordinate_neighbor_keys(
                    origin,
                    self.candidates,
                    self.config["absoluteWeightLevels"],
                    canonicalize_global_scale=bool(
                        self.config.get(
                            "canonicalizeGlobalScale",
                            True,
                        )
                    ),
                    fixed_loss_weights=self.config.get(
                        "fixedLossWeights"
                    ),
                    term_weight_levels=self.config.get(
                        "termWeightLevels"
                    ),
                ):
                    pair = (parent_key, neighbor_key)
                    if pair in evaluated:
                        continue
                    current = proposals.get(pair)
                    if (
                        current is None
                        or candidate_score(seed)
                        < candidate_score(current[1])
                    ):
                        proposals[pair] = (
                            self.candidate_by_key[neighbor_key],
                            seed,
                        )
        return [
            {
                "_candidate": candidate,
                "_initialize": seed,
                "_searchParent": seed["_searchParent"],
            }
            for _, (candidate, seed) in sorted(proposals.items())
        ]

    @staticmethod
    def best_validation_by_parent(
        candidates: Sequence[dict[str, Any]],
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for candidate in candidates:
            parent_key = str(candidate["searchParentKey"])
            result[parent_key] = min(
                result.get(parent_key, math.inf),
                finite_validation(candidate),
            )
        return result

    def expand_neighborhood(
        self,
        round_index: int,
        delay_seconds: int,
        candidates: Sequence[dict[str, Any]],
        parents: Sequence[dict[str, Any]],
        plan: dict[str, Any],
        plan_file: Path,
        progress: dict[str, Any],
        progress_file: Path,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        configuration = self.config.get("neighborhoodExpansion", {})
        if not configuration.get("enabled", False):
            return list(candidates), {
                "enabled": False,
                "rounds": [],
                "stopReason": "disabled",
            }
        stored_rounds = progress.get("neighborhoodExpansionRounds", [])
        if not isinstance(stored_rounds, list):
            raise RuntimeError("invalid neighborhood expansion progress")
        evaluated = list(candidates)
        for stored in stored_rounds:
            evaluated.extend(
                self.hydrate_candidates(
                    stored.get("candidates", []),
                    parents,
                )
            )
        minimum_improvement = float(
            configuration["minimumValidationKlImprovement"]
        )
        maximum_rounds = int(configuration["maximumRounds"])
        stop_reason = progress.get("neighborhoodExpansionStopReason")
        if isinstance(stop_reason, str):
            return evaluated, {
                "enabled": True,
                "rounds": stored_rounds,
                "stopReason": stop_reason,
                "evaluatedConfigurations": len(evaluated),
            }

        while len(stored_rounds) < maximum_rounds:
            expansion_index = len(stored_rounds)
            before = self.best_validation_by_parent(evaluated)
            trials = self.neighborhood_trials(evaluated)
            if not trials:
                stop_reason = "no-unseen-neighbors"
                progress["neighborhoodExpansionStopReason"] = stop_reason
                atomic_json(progress, progress_file)
                break
            expanded = self.train_fidelity(
                round_index,
                delay_seconds,
                self.config["fidelities"][-1],
                trials,
                plan,
                plan_file,
                f"neighborhood-{expansion_index + 1:02d}",
            )
            combined = [*evaluated, *expanded]
            after = self.best_validation_by_parent(combined)
            improvements = {
                parent_key: before[parent_key] - after[parent_key]
                for parent_key in before
            }
            improved_parents = sorted(
                parent_key
                for parent_key, improvement in improvements.items()
                if improvement >= minimum_improvement
            )
            stored_rounds.append({
                "round": expansion_index + 1,
                "seedCountPerParent": int(
                    configuration["seedCountPerParent"]
                ),
                "proposedConfigurations": len(trials),
                "trainedConfigurations": len(expanded),
                "minimumValidationKlImprovement":
                    minimum_improvement,
                "bestValidationKlBefore": before,
                "bestValidationKlAfter": after,
                "validationKlImprovements": improvements,
                "improvedParents": improved_parents,
                "candidates": self.serialize_candidates(expanded),
            })
            progress["neighborhoodExpansionRounds"] = stored_rounds
            evaluated = combined
            if not improved_parents:
                stop_reason = "no-meaningful-improvement"
                progress["neighborhoodExpansionStopReason"] = stop_reason
            atomic_json(progress, progress_file)
            if stop_reason is not None:
                break

        if stop_reason is None:
            stop_reason = "maximum-rounds"
            progress["neighborhoodExpansionStopReason"] = stop_reason
            atomic_json(progress, progress_file)
        return evaluated, {
            "enabled": True,
            "rounds": stored_rounds,
            "stopReason": stop_reason,
            "evaluatedConfigurations": len(evaluated),
            "addedConfigurations": len(evaluated) - len(candidates),
        }

    def run_round(
        self,
        round_index: int,
        delay_seconds: int,
        parents: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        progress_file = self.run_dir / (
            f"round-{round_index + 1:03d}-{delay_seconds}s-"
            f"{self.round_design_digest[:8]}-progress.json"
        )
        progress = read_json(progress_file) if progress_file.is_file() else {
            "version": 1,
            "round": round_index + 1,
            "delaySeconds": delay_seconds,
            "parents": [parent["key"] for parent in parents],
        }
        if progress.get("parents") != [parent["key"] for parent in parents]:
            raise RuntimeError("adaptive round progress belongs to different parents")
        if isinstance(progress.get("roundSummary"), dict):
            return (
                [dict(candidate) for candidate in progress["survivors"]],
                progress["roundSummary"],
            )
        plan, plan_file = self.plan_for_delay(delay_seconds)
        self.pair_delay(plan_file, delay_seconds)
        projections: dict[str, ProjectedQuadratic] = {}
        projection_results: dict[str, dict[str, Any]] = {}
        selections: dict[str, list[str]] = {}
        for parent in parents:
            model, selected, result = self.projected_screen(
                round_index,
                delay_seconds,
                parent,
                plan_file,
            )
            projections[parent["key"]] = model
            projection_results[parent["key"]] = {
                "backend": result["projectionBackend"],
                "baseValidationKl": result["baseValidationKl"],
                "selectedProbes": len(selected),
                "gpuMemoryMiB": result["gpuMemoryMiB"],
            }
            selections[parent["key"]] = selected

        fidelities = self.config["fidelities"]
        if isinstance(progress.get("initialScreen"), list):
            screen = self.hydrate_candidates(progress["initialScreen"], parents)
        else:
            screen = self.train_fidelity(
                round_index,
                delay_seconds,
                fidelities[0],
                self.initial_trials(parents, selections),
                plan,
                plan_file,
                "initial",
            )
            progress["initialScreen"] = self.serialize_candidates(screen)
            atomic_json(progress, progress_file)
        if (
            isinstance(progress.get("adaptiveScreen"), list)
            and isinstance(progress.get("acquisitionDiagnostics"), dict)
        ):
            adaptive_screen = self.hydrate_candidates(
                progress["adaptiveScreen"],
                parents,
            )
            acquisition_diagnostics = progress["acquisitionDiagnostics"]
        else:
            acquired, acquisition_diagnostics = self.acquired_trials(
                parents,
                projections,
                screen,
            )
            adaptive_screen = self.train_fidelity(
                round_index,
                delay_seconds,
                fidelities[0],
                acquired,
                plan,
                plan_file,
                "acquired",
            )
            progress["adaptiveScreen"] = self.serialize_candidates(
                adaptive_screen
            )
            progress["acquisitionDiagnostics"] = acquisition_diagnostics
            atomic_json(progress, progress_file)
        all_screen = [*screen, *adaptive_screen]
        if isinstance(progress.get("screenPromotedKeys"), list):
            promoted = candidates_in_persisted_order(
                all_screen,
                progress["screenPromotedKeys"],
            )
        else:
            promoted = select_promotions(
                all_screen,
                int(fidelities[0]["promote"]),
            )
            progress["screenPromotedKeys"] = [
                candidate["key"]
                for candidate in promoted
            ]
            atomic_json(progress, progress_file)
        self.prune_models(all_screen, promoted)
        tier_summaries = [{
            "key": fidelities[0]["key"],
            "branches": len(all_screen),
            "promoted": len(promoted),
            "bestValidationKl": finite_validation(promoted[0]),
        }]
        previous = promoted
        all_tier_candidates: list[list[dict[str, Any]]] = [all_screen]
        for fidelity in fidelities[1:]:
            tier_parents = list(previous)
            progress_key = f"{fidelity['key']}Candidates"
            promoted_key = f"{fidelity['key']}PromotedKeys"
            if isinstance(progress.get(progress_key), list):
                candidates = self.hydrate_candidates(
                    progress[progress_key],
                    parents,
                )
            else:
                candidates = self.train_fidelity(
                    round_index,
                    delay_seconds,
                    fidelity,
                    self.continuation_trials(previous),
                    plan,
                    plan_file,
                )
                progress[progress_key] = self.serialize_candidates(candidates)
                atomic_json(progress, progress_file)
            # Every child checkpoint and its result are durable now. The
            # preceding fidelity's weights are no longer needed to resume.
            self.prune_models(tier_parents, [])
            all_tier_candidates.append(candidates)
            if fidelity["key"] == "full":
                promoted = candidates
            elif isinstance(progress.get(promoted_key), list):
                promoted = candidates_in_persisted_order(
                    candidates,
                    progress[promoted_key],
                )
            else:
                promoted = select_promotions(
                    candidates,
                    int(fidelity["promote"]),
                )
                progress[promoted_key] = [
                    candidate["key"]
                    for candidate in promoted
                ]
                atomic_json(progress, progress_file)
                self.prune_models(candidates, promoted)
            tier_summaries.append({
                "key": fidelity["key"],
                "branches": len(candidates),
                "promoted": len(promoted),
                "bestValidationKl": finite_validation(
                    min(promoted, key=candidate_score)
                ),
            })
            previous = promoted

        previous, neighborhood_summary = self.expand_neighborhood(
            round_index,
            delay_seconds,
            previous,
            parents,
            plan,
            plan_file,
            progress,
            progress_file,
        )
        if any(
            not isinstance(candidate.get("equivalenceSignature"), dict)
            for candidate in previous
        ):
            raise RuntimeError("full-fidelity candidates lack equivalence signatures")
        representatives, collapsed = collapse_equivalent_candidates(
            list(previous),
            self.config["equivalence"],
        )
        survivors, pareto_fronts = pareto_beam(
            representatives,
            int(self.config["beamWidth"]),
        )
        for survivor in survivors:
            training_batches = int(
                survivor["transitionTrainingBatches"]
            )
            survivor["lineage"] = [
                *survivor["_searchParent"].get("lineage", []),
                {
                    "round": round_index + 1,
                    "delaySeconds": delay_seconds,
                    "weightCandidate": survivor["weightCandidate"],
                    "absoluteLossWeights": survivor["weights"],
                    "trainingBatches": training_batches,
                    "trainingExamples": int(
                        survivor["transitionTrainingExamples"]
                    ),
                    "optimizerUpdates": int(
                        survivor["transitionOptimizerUpdates"]
                    ),
                    "trainingStages": list(
                        survivor["transitionStages"]
                    ),
                    "plateauEpochsTrained": int(
                        survivor["stageEpochsTrained"]
                    ),
                    "plateauReached": bool(
                        survivor["stagePlateauReached"]
                    ),
                },
            ]
            survivor.pop("_candidate", None)
            survivor.pop("_searchParent", None)
        self.prune_models(previous, survivors)
        round_summary = {
            "round": round_index + 1,
            "delaySeconds": delay_seconds,
            "delayMinutes": delay_seconds / 60,
            "parents": [parent["key"] for parent in parents],
            "canonicalWeightCandidates": len(self.candidates),
            "candidateSpaceDigest": self.candidate_space_digest,
            "searchDesignDigest": self.search_design_digest,
            "roundDesignDigest": self.round_design_digest,
            "projection": projection_results,
            "acquisition": acquisition_diagnostics,
            "fidelities": tier_summaries,
            "neighborhoodExpansion": neighborhood_summary,
            "equivalenceRepresentatives": len(representatives),
            "collapsed": collapsed,
            "paretoFronts": pareto_fronts,
            "survivors": survivors,
            "bestObserved": self.serialize_candidates([
                min(previous, key=candidate_score)
            ])[0],
        }
        progress["survivors"] = self.serialize_candidates(survivors)
        progress["roundSummary"] = round_summary
        atomic_json(progress, progress_file)
        return survivors, round_summary

    def dry_run(self) -> dict[str, Any]:
        projection = self.config["projection"]
        initial_parents = (
            len(read_json(self.bootstrap_summary_file)["anchorParents"])
            if self.bootstrap_summary_file is not None
            else int(self.config["initialParentCount"])
        )
        initial_probes = int(projection["initialProbeCount"])
        adaptive_probes = int(projection["adaptiveProbeCount"])
        first_branches = initial_parents * (initial_probes + adaptive_probes)
        fixed_batches_before_plateau = sum(
            int(fidelity.get("additionalBatches", 0))
            for fidelity in self.config["fidelities"]
            if not fidelity.get("trainUntilPlateau", False)
        )
        final_fidelity = self.config["fidelities"][-1]
        return {
            "event": "adaptive-curriculum-design",
            "weightSemantics": (
                "absolute canonical directions; no multiplication by base weights"
            ),
            "absoluteWeightLevels": self.config["absoluteWeightLevels"],
            "fixedLossWeights": self.config.get("fixedLossWeights", {}),
            "termWeightLevels": self.config.get("termWeightLevels", {}),
            "activeLossTerms": [
                term
                for term in LOSS_TERMS
                if term not in self.config.get("fixedLossWeights", {})
            ],
            "rawWeightTuples": math.prod(
                len(
                    self.config.get("termWeightLevels", {}).get(
                        term,
                        self.config["absoluteWeightLevels"],
                    )
                )
                for term in LOSS_TERMS
                if term not in self.config.get("fixedLossWeights", {})
            ),
            "anchorValidRawWeightTuples": len(self.raw_candidates),
            "canonicalWeightCandidates": len(self.candidates),
            "distributionAnchorTerms": [
                "crossEntropy",
                "probabilityMse",
                "parameterMse",
            ],
            "initialDelaySeconds": self.config["initialDelaySeconds"],
            "minimumDelaySeconds": self.config["minimumDelaySeconds"],
            "initialStepSeconds": self.config["initialStepSeconds"],
            "effectiveStartingStepSeconds": self.config.get(
                "bootstrapStepSeconds",
                self.config["initialStepSeconds"],
            ),
            "stepGrowthFactor": self.config.get("stepGrowthFactor", 1.5),
            "delayResolutionSeconds": 1,
            "maximumRounds": self.config["maximumRounds"],
            "initialParents": initial_parents,
            "trainInitialParentsUntilPlateau": bool(
                self.config.get(
                    "trainInitialParentsUntilPlateau",
                    False,
                )
            ),
            "initialWeightVariants": self.config.get(
                "initialWeightVariants"
            ),
            "initialParentApplyFixedLossWeights": bool(
                self.config.get(
                    "initialParentApplyFixedLossWeights",
                    True,
                )
            ),
            "neighborhoodExpansion": self.config.get(
                "neighborhoodExpansion",
                {"enabled": False},
            ),
            "initialProjectedProbesPerParent": initial_probes,
            "adaptiveGpProbesPerParent": adaptive_probes,
            "firstRoundScreenBranches": first_branches,
            "populationSize": self.config["populationSize"],
            "compile": self.config.get("compile", False),
            "compileMinimumGroupBatches":
                self.config.get("compileMinimumGroupBatches"),
            "firstRoundScreenPopulationGroups": math.ceil(
                first_branches / int(self.config["populationSize"])
            ),
            "fidelities": self.config["fidelities"],
            "fixedBatchesBeforePlateauFinalist":
                fixed_batches_before_plateau,
            "finalistTraining": (
                {
                    "mode": "until-plateau",
                    "maximumEpochs": int(final_fidelity["maximumEpochs"]),
                    "patience": int(final_fidelity["patience"]),
                    "minimumImprovement": float(
                        final_fidelity["minimumImprovement"]
                    ),
                    "checkpointEveryEpochs": 1,
                }
                if final_fidelity.get("trainUntilPlateau", False)
                else {
                    "mode": "fixed-batches",
                    "additionalBatches": int(
                        final_fidelity["additionalBatches"]
                    ),
                }
            ),
            "selection": (
                "projected Hessian prior, measured GP residual acquisition, "
                + (
                    "successive halving, "
                    if len(self.config["fidelities"]) > 2
                    else "plateau training of every measured configuration, "
                )
                + (
                    "iterative one-coordinate plateau audit, "
                    if self.config.get(
                        "neighborhoodExpansion",
                        {},
                    ).get("enabled", False)
                    else ""
                )
                + "policy-JSD collapse, KL mean/std Pareto beam"
            ),
            "delayControl": (
                "delay-specific reference recovery with dwell, backtrack, "
                "and integer-second trust-region steps"
            ),
        }

    def run(self, maximum_rounds_override: int | None = None) -> None:
        self.prepare_directories()
        summary = self.summary()
        summary = self.train_initial_parents_until_plateau(summary)
        rounds = summary["rounds"]
        state = delay_state_from_dict(summary["state"])
        maximum_rounds = (
            int(maximum_rounds_override)
            if maximum_rounds_override is not None
            else int(self.config["maximumRounds"])
        )
        if maximum_rounds < 1:
            raise ValueError("maximum-rounds must be positive")
        retention = self.sweep_obsolete_models(summary)
        if self.accepted_export_pending(summary):
            self.export_accepted_models(summary)
        self.write_status(
            stage="starting",
            completedRounds=len(rounds),
            retention=retention,
            message="Starting or resuming adaptive curriculum search.",
        )
        references = self.direct_references()
        while len(rounds) < maximum_rounds:
            if summary.get("complete"):
                break
            delay_seconds = state.trial_delay_seconds
            parents = summary["trialParents"]
            round_index = len(rounds)
            survivors, round_summary = self.run_round(
                round_index,
                delay_seconds,
                parents,
            )
            reference = interpolate_delay_reference(delay_seconds, references)
            for survivor in survivors:
                transition = survivor["lineage"][-1]
                transition.update({
                    "validationKl": finite_validation(survivor),
                    "validationKlStdDev": float(
                        survivor["validation"].get(
                            "klDivergenceStdDev",
                            math.inf,
                        )
                    ),
                    "referenceKl": reference,
                    "qualityGap": (
                        finite_validation(survivor) - reference
                    ),
                })
            best = self.transition_candidate(survivors, reference)
            decision = update_delay_continuation(
                state,
                finite_validation(best),
                reference,
                minimum_delay_seconds=int(self.config["minimumDelaySeconds"]),
                absolute_tolerance=float(
                    self.config["dwell"]["absoluteReferenceTolerance"]
                ),
                relative_tolerance=float(
                    self.config["dwell"]["relativeReferenceTolerance"]
                ),
                minimum_improvement=float(
                    self.config["dwell"]["minimumImprovement"]
                ),
                patience=int(self.config["dwell"]["patience"]),
                maximum_dwell_epochs=int(
                    self.config["dwell"]["maximumEpochs"]
                ),
                step_growth_factor=float(
                    self.config.get("stepGrowthFactor", 1.5)
                ),
            )
            round_summary["delayReferenceKl"] = reference
            round_summary["decisionCandidateKey"] = best["key"]
            round_summary["survivors"] = self.serialize_candidates(
                survivors
            )
            round_summary["decision"] = {
                "action": decision.action,
                "reason": decision.reason,
                "nextState": delay_state_dict(decision.state),
            }
            rounds.append(round_summary)
            accepted = summary["acceptedDelays"]
            if decision.action in ("advance", "complete", "accept-plateau"):
                accepted.append({
                    "delaySeconds": delay_seconds,
                    "reason": decision.reason,
                    "survivors": survivors,
                })
                summary["anchorParents"] = survivors
                summary["trialParents"] = survivors
            elif decision.action == "dwell":
                summary["trialParents"] = survivors
            elif decision.action == "backtrack":
                summary["trialParents"] = summary["anchorParents"]
                self.prune_models(survivors, [])
            else:
                raise RuntimeError(f"unknown delay decision: {decision.action}")

            if decision.action == "accept-plateau":
                if delay_seconds <= int(self.config["minimumDelaySeconds"]):
                    summary["complete"] = True
                else:
                    next_delay = max(
                        int(self.config["minimumDelaySeconds"]),
                        delay_seconds - 1,
                    )
                    state = DelayContinuation(
                        anchor_delay_seconds=delay_seconds,
                        trial_delay_seconds=next_delay,
                        step_seconds=1,
                        dwell_epochs=0,
                        stale_epochs=0,
                        best_validation=math.inf,
                    )
            else:
                state = decision.state
            if decision.action == "complete":
                summary["complete"] = True
            summary = {
                **summary,
                "state": delay_state_dict(state),
                "rounds": rounds,
                "completedRounds": len(rounds),
                "bestObserved": min(
                    [
                        round_["bestObserved"]
                        for round_ in rounds
                    ] + (
                        [summary["bootstrapBestObserved"]]
                        if summary.get("bootstrapBestObserved") is not None
                        else []
                    ),
                    key=candidate_score,
                ),
                "generatedAt": utc_now(),
            }
            summary = self.update_schedule_optimization(summary)
            atomic_json(summary, self.summary_file)
            retention = self.sweep_obsolete_models(summary)
            if decision.action in ("advance", "complete", "accept-plateau"):
                self.export_accepted_models(summary)
            self.write_status(
                stage=(
                    "complete"
                    if summary.get("complete")
                    else "round-complete"
                ),
                completedRounds=len(rounds),
                delaySeconds=delay_seconds,
                decision=decision.action,
                nextDelaySeconds=state.trial_delay_seconds,
                bestValidationKl=finite_validation(best),
                referenceValidationKl=reference,
                retention=retention,
                message=(
                    f"Round {len(rounds)} {decision.action}: {decision.reason}."
                ),
            )
        if summary.get("complete"):
            self.finalize_models(summary)
            self.write_status(
                stage="complete",
                completedAt=utc_now(),
                message="Adaptive curriculum and finalist export complete.",
            )
        else:
            self.write_status(
                stage="paused-at-round-limit",
                completedRounds=len(rounds),
                nextDelaySeconds=state.trial_delay_seconds,
                message=(
                    "Stopped at the configured round limit; rerun the same "
                    "command to resume after increasing the limit."
                ),
            )

    def finalize_models(self, summary: dict[str, Any]) -> None:
        self.export_accepted_models(summary)
        summary["completedAt"] = utc_now()
        atomic_json(summary, self.summary_file)

    def accepted_export_pending(self, summary: dict[str, Any]) -> bool:
        for accepted in summary["acceptedDelays"]:
            for candidate in sorted(
                accepted["survivors"],
                key=candidate_score,
            )[: int(self.config["finalModelsPerAcceptedDelay"])]:
                if (
                    str(candidate["model"]).endswith(".pt")
                    and not (
                        Path(candidate["directory"]) / "manifest.json"
                    ).is_file()
                ):
                    return True
        return False

    def export_accepted_models(self, summary: dict[str, Any]) -> None:
        finalists = []
        for accepted in summary["acceptedDelays"]:
            finalists.extend(
                sorted(
                    accepted["survivors"],
                    key=candidate_score,
                )[: int(self.config["finalModelsPerAcceptedDelay"])]
            )
        artifacts = []
        training = self.plan["training"]
        for candidate in finalists:
            if not str(candidate["model"]).endswith(".pt"):
                continue
            delay_seconds = int(candidate["delayMs"]) // 1_000
            directory = Path(candidate["directory"])
            if not (directory / "manifest.json").is_file():
                plan, plan_file = self.plan_for_delay(delay_seconds)
                self.pair_delay(plan_file, delay_seconds)
                self.run_child(
                    f"export-{candidate['key']}",
                    [
                        sys.executable,
                        str(self.repo / "ml/export_mlp_study_artifact.py"),
                        "--dataset",
                        str(self.dataset),
                        "--output",
                        str(directory),
                        "--study-file",
                        candidate["resultFile"],
                        "--target-statistics-cache",
                        str(
                            self.statistics_dir
                            / (
                                f"delay-{delay_seconds}s-targets-"
                                f"{self.source['experimentFingerprint']}.npz"
                            )
                        ),
                        "--model-id",
                        f"{self.plan['id']}-adaptive-{candidate['key']}",
                        "--label",
                        f"{plan['label']} · adaptive selected {candidate['key']}",
                        "--plan",
                        str(plan_file),
                        "--evaluation-batch-size",
                        str(training["evaluationBatchSize"]),
                        "--states-per-example",
                        str(training["statesPerExample"]),
                        "--workers",
                        str(self.config["workers"]),
                        "--dropout",
                        str(training["dropout"]),
                        "--seed",
                        str(training["seed"]),
                        "--device",
                        training["device"],
                        "--selection-metric",
                        "klDivergence",
                        "--loss-weights-json",
                        json.dumps(candidate["weights"]),
                        "--time-weighting-json",
                        json.dumps(training["timeWeighting"]),
                    ],
                )
                self.run_child(
                    f"verify-{candidate['key']}",
                    [
                        "node",
                        "scripts/run-node-with-ml-libs.mjs",
                        "scripts/verify-mlp-model.mjs",
                        str(directory),
                    ],
                )
            artifacts.append({
                "key": candidate["key"],
                "delayMs": candidate["delayMs"],
                "directory": str(directory),
                "validation": candidate["validation"],
                "weights": candidate["weights"],
            })
        summary["artifacts"] = artifacts
        atomic_json(summary, self.summary_file)


def main() -> None:
    args = parse_args()
    runner = AdaptiveCurriculumRunner(args.plan, args.study_key)
    if args.dry_run:
        print(json.dumps(runner.dry_run(), indent=2))
        return
    previous = (
        read_json(runner.status_file)
        if runner.status_file.is_file()
        else {}
    )
    previous_pid = previous.get("pid")
    if isinstance(previous_pid, int) and previous_pid != os.getpid():
        try:
            os.kill(previous_pid, 0)
        except OSError:
            pass
        else:
            raise RuntimeError(
                f"adaptive curriculum is already running as PID {previous_pid}"
            )
    try:
        runner.run(args.maximum_rounds)
    except KeyboardInterrupt:
        runner.write_status(
            stage="paused",
            pausedAt=utc_now(),
            message="Adaptive curriculum paused and remains resumable.",
        )
        raise
    except Exception as error:
        runner.write_status(
            stage="failed",
            failedAt=utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        raise


if __name__ == "__main__":
    main()

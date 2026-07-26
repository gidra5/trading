from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


LOSS_TERMS = (
    "crossEntropy",
    "probabilityMse",
    "parameterMse",
    "excessEntropy",
    "oracleMutualInformation",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a resumable beam search over delay/loss-weight curricula."
    )
    parser.add_argument("--plan", type=Path, default=Path("ml/training-plan.json"))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-for-source", action="store_true")
    parser.add_argument(
        "--refresh-last-completed-stage",
        action="store_true",
        help=(
            "Recompute the last completed stage's equivalence collapse and Pareto "
            "beam after its branch checkpoints have been rebuilt."
        ),
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def read_json(file: Path) -> dict[str, Any]:
    return json.loads(file.read_text())


def atomic_json(value: dict[str, Any], file: Path) -> None:
    file.parent.mkdir(parents=True, exist_ok=True)
    temporary = file.with_suffix(file.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(file)


def normalized_weights(value: dict[str, Any]) -> dict[str, float]:
    weights = {term: float(value[term]) for term in LOSS_TERMS}
    if any(not math.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("curriculum weights must be finite and non-negative")
    return weights


def weight_profiles(source: dict[str, Any]) -> list[dict[str, Any]]:
    profiles: dict[str, dict[str, Any]] = {}
    for row in source.get("results", []):
        key = str(row.get("weightVariant", ""))
        if not key or key in profiles:
            continue
        profiles[key] = {
            "key": key,
            "weights": normalized_weights(row["weights"]),
        }
    return sorted(profiles.values(), key=lambda profile: profile["key"])


def source_parent_rows(
    source: dict[str, Any],
    delay_ms: int,
    count: int,
) -> list[dict[str, Any]]:
    rows = [
        row for row in source.get("results", [])
        if int(row.get("delayMs", -1)) == delay_ms
        and math.isfinite(float(row.get("validation", {}).get("klDivergence", math.nan)))
    ]
    return sorted(
        rows,
        key=lambda row: (
            float(row["validation"]["klDivergence"]),
            float(row["validation"].get("klDivergenceStdDev", math.inf)),
            str(row["weightVariant"]),
        ),
    )[:count]


def candidate_score(candidate: dict[str, Any]) -> tuple[float, float, str]:
    metrics = candidate["validation"]
    return (
        float(metrics["klDivergence"]),
        float(metrics.get("klDivergenceStdDev", math.inf)),
        str(candidate["key"]),
    )


def load_signature(candidate: dict[str, Any]) -> np.ndarray:
    metadata = candidate["equivalenceSignature"]
    file = Path(candidate["directory"]) / metadata["file"]
    shape = tuple(int(value) for value in metadata["shape"])
    values = np.fromfile(file, dtype="<f2")
    if values.size != math.prod(shape):
        raise ValueError(f"invalid equivalence signature size: {file}")
    digest = hashlib.sha256(values.tobytes()).hexdigest()
    if digest != metadata["sha256"]:
        raise ValueError(f"equivalence signature checksum mismatch: {file}")
    return values.astype(np.float32).reshape(shape)


def mean_policy_js_divergence(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("policy signatures must have the same shape")
    epsilon = np.finfo(np.float32).tiny
    left_probability = np.clip(left.astype(np.float64), epsilon, 1)
    right_probability = np.clip(right.astype(np.float64), epsilon, 1)
    midpoint = 0.5 * (left_probability + right_probability)
    divergence = 0.5 * (
        np.sum(left_probability * np.log(left_probability / midpoint), axis=-1)
        + np.sum(right_probability * np.log(right_probability / midpoint), axis=-1)
    )
    return float(np.mean(divergence))


def equivalent_candidates(
    left: dict[str, Any],
    right: dict[str, Any],
    configuration: dict[str, Any],
    signatures: dict[str, np.ndarray],
) -> bool:
    left_kl = float(left["validation"]["klDivergence"])
    right_kl = float(right["validation"]["klDivergence"])
    kl_threshold = max(
        float(configuration["absoluteKl"]),
        float(configuration["relativeKl"]) * min(left_kl, right_kl),
    )
    if abs(left_kl - right_kl) > kl_threshold:
        return False
    return mean_policy_js_divergence(
        signatures[left["key"]],
        signatures[right["key"]],
    ) <= float(configuration["maximumPolicyJsd"])


def collapse_equivalent_candidates(
    candidates: list[dict[str, Any]],
    configuration: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    signatures = {
        candidate["key"]: load_signature(candidate)
        for candidate in candidates
    }
    representatives: list[dict[str, Any]] = []
    collapsed: list[dict[str, Any]] = []
    for candidate in sorted(candidates, key=candidate_score):
        representative = next((
            existing for existing in representatives
            if equivalent_candidates(candidate, existing, configuration, signatures)
        ), None)
        if representative is None:
            representatives.append(candidate)
            continue
        collapsed.append({
            "key": candidate["key"],
            "representative": representative["key"],
            "validationKl": candidate["validation"]["klDivergence"],
            "representativeValidationKl":
                representative["validation"]["klDivergence"],
            "policyJsd": mean_policy_js_divergence(
                signatures[candidate["key"]],
                signatures[representative["key"]],
            ),
        })
    return representatives, collapsed


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    left_metrics = left["validation"]
    right_metrics = right["validation"]
    left_mean = float(left_metrics["klDivergence"])
    right_mean = float(right_metrics["klDivergence"])
    left_std = float(left_metrics.get("klDivergenceStdDev", math.inf))
    right_std = float(right_metrics.get("klDivergenceStdDev", math.inf))
    return left_mean <= right_mean and left_std <= right_std \
        and (left_mean < right_mean or left_std < right_std)


def pareto_beam(
    candidates: list[dict[str, Any]],
    width: int,
) -> tuple[list[dict[str, Any]], list[list[str]]]:
    remaining = list(candidates)
    selected: list[dict[str, Any]] = []
    fronts: list[list[str]] = []
    while remaining and len(selected) < width:
        front = [
            candidate for candidate in remaining
            if not any(
                dominates(other, candidate)
                for other in remaining
                if other is not candidate
            )
        ]
        front.sort(key=candidate_score)
        fronts.append([candidate["key"] for candidate in front])
        selected.extend(front[:width - len(selected)])
        front_keys = {candidate["key"] for candidate in front}
        remaining = [
            candidate for candidate in remaining
            if candidate["key"] not in front_keys
        ]
    return selected, fronts


class CurriculumRunner:
    def __init__(self, plan_file: Path) -> None:
        self.repo = plan_file.resolve().parent.parent
        self.plan_file = plan_file.resolve()
        self.plan = read_json(self.plan_file)
        self.config = self.plan["dynamicCurriculumStudy"]
        self.output = (self.repo / self.config["outputDir"]).resolve()
        self.run_dir = (self.repo / self.config["runDir"]).resolve()
        self.dataset = (self.repo / self.config["datasetDir"]).resolve()
        self.source_summary_file = (
            self.repo / self.config["sourceSummary"]
        ).resolve()
        self.source = read_json(self.source_summary_file)
        self.status_file = self.run_dir / "status.json"
        self.log_file = self.run_dir / "study.log"
        self.summary_file = self.output / "summary.json"
        self.plans_dir = self.output / "plans"
        self.jobs_dir = self.run_dir / "population-jobs"
        self.statistics_dir = self.output / "statistics"
        self.interrupted = False
        self.status: dict[str, Any] = {
            "pid": os.getpid(),
            "planId": self.plan["id"],
            "studyKey": "dynamicCurriculumStudy",
            "stage": "initializing",
            "startedAt": utc_now(),
            "updatedAt": utc_now(),
            "sourceSummary": str(self.source_summary_file),
        }
        self.validate()

    def validate(self) -> None:
        schedules = self.config["delayScheduleMinutes"]
        if not schedules or any(float(value) < 0 for value in schedules):
            raise ValueError("dynamic curriculum delay schedule is invalid")
        initial = float(self.config["initialDelayMinutes"])
        if any(float(value) > initial for value in schedules):
            raise ValueError("curriculum delays cannot exceed the initial delay")
        if any(
            float(schedules[index]) > float(schedules[index - 1])
            for index in range(1, len(schedules))
        ):
            raise ValueError("curriculum delay schedule must be non-increasing")
        for key in (
            "initialParentCount", "beamWidth", "populationSize",
            "epochsPerTransition", "patience",
        ):
            if int(self.config[key]) < 1:
                raise ValueError(f"{key} must be positive")
        if self.config["epochsPerTransition"] != 1:
            raise ValueError(
                "one epoch per transition is required to identify epoch-wise schedules"
            )
        source_complete = (
            int(self.source.get("completedRuns", 0))
            == int(self.source.get("plannedRuns", -1))
        )
        if not source_complete:
            self.status["sourceComplete"] = False

    def dry_run(self) -> dict[str, Any]:
        profiles = weight_profiles(self.source)
        parent_count = int(self.config["initialParentCount"])
        beam = int(self.config["beamWidth"])
        stages = []
        for index, delay in enumerate(self.config["delayScheduleMinutes"]):
            branches = parent_count * len(profiles) if index == 0 \
                else beam * len(profiles)
            stages.append({
                "index": index + 1,
                "delayMinutes": delay,
                "parents": parent_count if index == 0 else beam,
                "weightProfiles": len(profiles),
                "maximumBranchesBeforeEquivalenceCollapse": branches,
                "populationGroups": math.ceil(
                    branches / int(self.config["populationSize"])
                ),
            })
        return {
            "event": "dynamic-curriculum-design",
            "sourceComplete": self.status.get("sourceComplete", True),
            "sourceCompletedRuns": self.source.get("completedRuns"),
            "sourcePlannedRuns": self.source.get("plannedRuns"),
            "initialDelayMinutes": self.config["initialDelayMinutes"],
            "delayScheduleMinutes": self.config["delayScheduleMinutes"],
            "weightProfiles": len(profiles),
            "beamWidth": beam,
            "stages": stages,
            "maximumTransitionRuns": sum(
                stage["maximumBranchesBeforeEquivalenceCollapse"] for stage in stages
            ),
            "trainingExamplesPerBranch": (
                int(self.config["maxBatchesPerEpoch"])
                * int(self.plan["training"]["batchSize"])
            ),
            "selection": "policy-JSD equivalence collapse, then KL mean/std Pareto beam",
        }

    def write_status(self, **updates: Any) -> None:
        self.status = {
            **self.status,
            **updates,
            "updatedAt": utc_now(),
        }
        atomic_json(self.status, self.status_file)

    def append_log(self, line: str) -> None:
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_file.open("a") as stream:
            stream.write(line.rstrip() + "\n")

    def run_child(self, stage: str, command: list[str]) -> None:
        if self.interrupted:
            raise KeyboardInterrupt
        self.write_status(
            stage=stage,
            command=command,
            message=f"Running {stage}.",
        )
        self.append_log(
            json.dumps({"event": "stage-start", "stage": stage, "command": command})
        )
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

    def plan_for_delay(self, delay_ms: int) -> tuple[dict[str, Any], Path]:
        priority_plan = read_json(
            self.repo
            / self.config["sourcePriorityPlan"]
        )
        key = f"delay-{delay_ms // 1000}s"
        plan = {
            **priority_plan,
            "id": f"{self.plan['id']}-dynamic-{key}",
            "label": (
                self.plan["label"].replace(" · 60-second policy delay", "")
                + f" · dynamic curriculum {delay_ms / 60_000:g}m"
            ),
            "predictionDelayMs": delay_ms,
            "artifactDir": str(
                (self.output / key / "artifact-unused").relative_to(self.repo)
            ),
            "runDir": str((self.output / key / "run").relative_to(self.repo)),
        }
        file = self.plans_dir / f"{key}.json"
        atomic_json(plan, file)
        return plan, file

    def pair_delay(self, plan_file: Path, delay_ms: int) -> None:
        manifest_file = self.dataset / "dataset.json"
        if manifest_file.is_file():
            manifest = read_json(manifest_file)
            if int(manifest.get("predictionDelayMs", -1)) == delay_ms:
                return
        self.run_child(
            f"pair-delay-{delay_ms // 1000}s",
            [
                "node",
                "scripts/run-node-with-ml-libs.mjs",
                "node_modules/tsx/dist/cli.mjs",
                "scripts/build-mlp-dataset.ts",
                "--plan",
                str(plan_file),
            ],
        )

    def initial_parents(self) -> list[dict[str, Any]]:
        delay_ms = round(float(self.config["initialDelayMinutes"]) * 60_000)
        rows = source_parent_rows(
            self.source,
            delay_ms,
            int(self.config["initialParentCount"]),
        )
        if len(rows) != int(self.config["initialParentCount"]):
            raise RuntimeError(
                "the source study has not completed enough initial-delay parents"
            )
        parents = []
        for row in rows:
            directory = Path(row["resultFile"]).parent
            model = directory / "model.onnx"
            if not model.is_file():
                raise FileNotFoundError(
                    f"source parent has no verified ONNX model: {model}"
                )
            parents.append({
                "key": f"source-{row['weightVariant']}",
                "delayMs": delay_ms,
                "model": str(model),
                "weights": row["weights"],
                "validation": row["validation"],
                "sourceResultFile": row["resultFile"],
                "lineage": [],
            })
        return parents

    def build_jobs(
        self,
        stage_index: int,
        delay_ms: int,
        parents: list[dict[str, Any]],
        profiles: list[dict[str, Any]],
        plan: dict[str, Any],
        plan_file: Path,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        jobs = []
        for parent_index, parent in enumerate(parents):
            for profile in profiles:
                digest = hashlib.sha256(
                    f"{stage_index}|{parent['key']}|{profile['key']}".encode()
                ).hexdigest()[:12]
                key = f"s{stage_index + 1:02d}-p{parent_index:02d}-{profile['key']}-{digest}"
                directory = self.output / "branches" / key
                jobs.append({
                    "key": key,
                    "output": str(directory),
                    "resultFile": str(directory / "study.json"),
                    "modelId": f"{self.plan['id']}-dynamic-{key}",
                    "label": f"{plan['label']} · epoch {stage_index + 1} · {profile['key']}",
                    "lossWeights": profile["weights"],
                    "initializeFromCheckpoint": parent["model"],
                    "parentKey": parent["key"],
                    "_parent": parent,
                    "_profile": profile,
                    "_directory": directory,
                })
        public_jobs = [
            {key: value for key, value in job.items() if not key.startswith("_")}
            for job in jobs
        ]
        training = self.plan["training"]
        fingerprint = self.source["experimentFingerprint"]
        specification = {
            "version": 1,
            "common": {
                "dataset": str(self.dataset),
                "plan": str(plan_file),
                "epochs": int(self.config["epochsPerTransition"]),
                "batchSize": int(training["batchSize"]),
                "evaluationBatchSize": int(
                    training.get("evaluationBatchSize", training["batchSize"])
                ),
                "validationFraction": float(self.config["validationFraction"]),
                "accumulate": int(training["gradientAccumulation"]),
                "learningRate": float(training["learningRate"]),
                "weightDecay": float(training["weightDecay"]),
                "dropout": float(training["dropout"]),
                "statesPerExample": int(training["statesPerExample"]),
                "patience": int(self.config["patience"]),
                "workers": int(self.config["workers"]),
                "seed": int(training["seed"]),
                "device": training["device"],
                "logEverySteps": int(training["logEverySteps"]),
                "timeWeighting": training["timeWeighting"],
                "selectionMetric": "klDivergence",
                "featureStatisticsCache": str(
                    self.statistics_dir / f"features-{fingerprint}.npz"
                ),
                "targetStatisticsCache": str(
                    self.statistics_dir
                    / f"delay-{delay_ms // 1000}s-targets-{fingerprint}.npz"
                ),
                "compile": bool(training.get("compile", False)),
                "checkpointEveryEpochs": 0,
                "maxBatchesPerEpoch": int(self.config["maxBatchesPerEpoch"]),
                "retainAllBestModels": True,
                "equivalenceSignature": self.config["equivalenceSignature"],
            },
            "jobs": public_jobs,
        }
        return specification, jobs

    def collect_candidates(
        self,
        stage_index: int,
        delay_ms: int,
        jobs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        candidates = []
        for job in jobs:
            result = read_json(Path(job["resultFile"]))
            signature = result.get("equivalenceSignature")
            if not isinstance(signature, dict):
                raise RuntimeError(f"branch has no equivalence signature: {job['key']}")
            curriculum = {
                "version": 1,
                "stage": stage_index + 1,
                "delayMs": delay_ms,
                "parentKey": job["parentKey"],
                "weightProfile": job["_profile"]["key"],
                "lineage": [
                    *job["_parent"].get("lineage", []),
                    {
                        "stage": stage_index + 1,
                        "delayMs": delay_ms,
                        "weights": job["lossWeights"],
                    },
                ],
            }
            result["curriculumTraining"] = curriculum
            atomic_json(result, Path(job["resultFile"]))
            candidates.append({
                "key": job["key"],
                "delayMs": delay_ms,
                "model": str(job["_directory"] / "best-model.pt"),
                "directory": str(job["_directory"]),
                "resultFile": job["resultFile"],
                "parentKey": job["parentKey"],
                "weightProfile": job["_profile"]["key"],
                "weights": job["lossWeights"],
                "validation": result["bestValidationMetrics"],
                "equivalenceSignature": signature,
                "lineage": curriculum["lineage"],
            })
        return candidates

    def summary(self) -> dict[str, Any]:
        if self.summary_file.is_file():
            return read_json(self.summary_file)
        return {
            "version": 1,
            "planId": self.plan["id"],
            "sourceExperimentFingerprint": self.source["experimentFingerprint"],
            "sourceSummary": str(self.source_summary_file),
            "delayScheduleMinutes": self.config["delayScheduleMinutes"],
            "beamWidth": self.config["beamWidth"],
            "weightProfiles": len(weight_profiles(self.source)),
            "stages": [],
            "generatedAt": utc_now(),
        }

    def run(self) -> None:
        if self.status.get("sourceComplete") is False:
            raise RuntimeError("source response study must finish before curriculum search")
        self.output.mkdir(parents=True, exist_ok=True)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.jobs_dir.mkdir(parents=True, exist_ok=True)
        self.statistics_dir.mkdir(parents=True, exist_ok=True)
        self.write_status(stage="starting", message="Starting dynamic curriculum search.")
        summary = self.summary()
        profiles = weight_profiles(self.source)
        completed_stages = summary["stages"]
        parents = self.initial_parents() if not completed_stages else [
            candidate
            for candidate in completed_stages[-1]["survivors"]
        ]
        for stage_index, delay_minutes in enumerate(
            self.config["delayScheduleMinutes"]
        ):
            if stage_index < len(completed_stages):
                continue
            delay_ms = round(float(delay_minutes) * 60_000)
            plan, plan_file = self.plan_for_delay(delay_ms)
            self.pair_delay(plan_file, delay_ms)
            specification, jobs = self.build_jobs(
                stage_index,
                delay_ms,
                parents,
                profiles,
                plan,
                plan_file,
            )
            jobs_file = self.jobs_dir / f"stage-{stage_index + 1:02d}.json"
            atomic_json(specification, jobs_file)
            self.write_status(
                stage="population-training",
                curriculumStage=stage_index + 1,
                curriculumStages=len(self.config["delayScheduleMinutes"]),
                delayMinutes=delay_minutes,
                parents=len(parents),
                branches=len(jobs),
                message=(
                    f"Training {len(jobs)} one-epoch branches at {delay_minutes:g}m."
                ),
            )
            self.run_child(
                f"dynamic-stage-{stage_index + 1:02d}",
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
            candidates = self.collect_candidates(stage_index, delay_ms, jobs)
            representatives, collapsed = collapse_equivalent_candidates(
                candidates,
                self.config["equivalence"],
            )
            survivors, pareto_fronts = pareto_beam(
                representatives,
                int(self.config["beamWidth"]),
            )
            stage_summary = {
                "stage": stage_index + 1,
                "delayMs": delay_ms,
                "delayMinutes": delay_minutes,
                "parents": [parent["key"] for parent in parents],
                "branches": len(candidates),
                "equivalenceRepresentatives": len(representatives),
                "collapsed": collapsed,
                "paretoFronts": pareto_fronts,
                "survivors": survivors,
            }
            completed_stages.append(stage_summary)
            summary = {
                **summary,
                "stages": completed_stages,
                "completedStages": len(completed_stages),
                "bestObserved": min(candidates, key=candidate_score),
                "generatedAt": utc_now(),
            }
            atomic_json(summary, self.summary_file)
            survivor_keys = {candidate["key"] for candidate in survivors}
            for candidate in candidates:
                if candidate["key"] not in survivor_keys:
                    Path(candidate["model"]).unlink(missing_ok=True)
            parents = survivors
        self.finalize_models(summary)
        self.write_status(
            stage="complete",
            completedAt=utc_now(),
            message="Dynamic curriculum search and finalist export complete.",
        )

    def refresh_last_completed_stage(self) -> None:
        summary = self.summary()
        completed_stages = summary.get("stages", [])
        if not completed_stages:
            raise RuntimeError("there is no completed curriculum stage to refresh")
        stage_index = len(completed_stages) - 1
        previous = completed_stages[stage_index]
        delay_ms = int(previous["delayMs"])
        plan, plan_file = self.plan_for_delay(delay_ms)
        self.pair_delay(plan_file, delay_ms)
        parents = self.initial_parents() if stage_index == 0 else [
            candidate
            for candidate in completed_stages[stage_index - 1]["survivors"]
        ]
        _, jobs = self.build_jobs(
            stage_index,
            delay_ms,
            parents,
            weight_profiles(self.source),
            plan,
            plan_file,
        )
        missing = [
            job["key"]
            for job in jobs
            if not Path(job["resultFile"]).is_file()
            or not (job["_directory"] / "best-model.pt").is_file()
        ]
        if missing:
            raise RuntimeError(
                "cannot refresh a stage with missing rebuilt branches: "
                + ", ".join(missing[:8])
            )
        candidates = self.collect_candidates(stage_index, delay_ms, jobs)
        representatives, collapsed = collapse_equivalent_candidates(
            candidates,
            self.config["equivalence"],
        )
        survivors, pareto_fronts = pareto_beam(
            representatives,
            int(self.config["beamWidth"]),
        )
        completed_stages[stage_index] = {
            "stage": stage_index + 1,
            "delayMs": delay_ms,
            "delayMinutes": previous["delayMinutes"],
            "parents": [parent["key"] for parent in parents],
            "branches": len(candidates),
            "equivalenceRepresentatives": len(representatives),
            "collapsed": collapsed,
            "paretoFronts": pareto_fronts,
            "survivors": survivors,
        }
        summary = {
            **summary,
            "stages": completed_stages,
            "completedStages": len(completed_stages),
            "bestObserved": min(candidates, key=candidate_score),
            "generatedAt": utc_now(),
        }
        atomic_json(summary, self.summary_file)
        survivor_keys = {candidate["key"] for candidate in survivors}
        for candidate in candidates:
            if candidate["key"] not in survivor_keys:
                Path(candidate["model"]).unlink(missing_ok=True)
        self.write_status(
            stage="stage-refresh-complete",
            curriculumStage=stage_index + 1,
            delayMinutes=previous["delayMinutes"],
            parents=len(parents),
            branches=len(candidates),
            message=(
                f"Refreshed dynamic stage {stage_index + 1}; "
                f"retained {len(survivors)} parents."
            ),
        )

    def wait_for_source(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        while True:
            self.source = read_json(self.source_summary_file)
            completed = int(self.source.get("completedRuns", 0))
            planned = int(self.source.get("plannedRuns", -1))
            if completed == planned and planned > 0:
                self.status.pop("sourceComplete", None)
                self.write_status(
                    stage="source-complete",
                    sourceCompletedRuns=completed,
                    sourcePlannedRuns=planned,
                    message="Source response study complete; starting curriculum search.",
                )
                return
            self.write_status(
                stage="waiting-for-source",
                sourceCompletedRuns=completed,
                sourcePlannedRuns=planned,
                message=(
                    f"Waiting for the source response study ({completed}/{planned})."
                ),
            )
            time.sleep(30)

    def finalize_models(self, summary: dict[str, Any]) -> None:
        finalists = []
        seen_delays = []
        for stage in reversed(summary["stages"]):
            delay_ms = int(stage["delayMs"])
            if delay_ms in seen_delays:
                continue
            seen_delays.append(delay_ms)
            finalists.extend(
                sorted(stage["survivors"], key=candidate_score)[
                    :int(self.config["finalModelsPerDelay"])
                ]
            )
        artifacts = []
        for candidate in reversed(finalists):
            delay_ms = int(candidate["delayMs"])
            plan, plan_file = self.plan_for_delay(delay_ms)
            self.pair_delay(plan_file, delay_ms)
            directory = Path(candidate["directory"])
            manifest_file = directory / "manifest.json"
            if not manifest_file.is_file():
                training = self.plan["training"]
                self.run_child(
                    f"export-{candidate['key']}",
                    [
                        sys.executable,
                        str(self.repo / "ml/export_mlp_study_artifact.py"),
                        "--dataset", str(self.dataset),
                        "--output", str(directory),
                        "--study-file", candidate["resultFile"],
                        "--target-statistics-cache", str(
                            self.statistics_dir
                            / (
                                f"delay-{delay_ms // 1000}s-targets-"
                                f"{self.source['experimentFingerprint']}.npz"
                            )
                        ),
                        "--model-id", f"{self.plan['id']}-dynamic-{candidate['key']}",
                        "--label", (
                            f"{plan['label']} · selected curriculum "
                            f"{candidate['key']}"
                        ),
                        "--plan", str(plan_file),
                        "--evaluation-batch-size", str(
                            training.get(
                                "evaluationBatchSize",
                                training["batchSize"],
                            )
                        ),
                        "--states-per-example", str(training["statesPerExample"]),
                        "--workers", str(self.config["workers"]),
                        "--dropout", str(training["dropout"]),
                        "--seed", str(training["seed"]),
                        "--device", training["device"],
                        "--selection-metric", "klDivergence",
                        "--loss-weights-json", json.dumps(candidate["weights"]),
                        "--time-weighting-json", json.dumps(
                            training["timeWeighting"]
                        ),
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
                "delayMs": delay_ms,
                "directory": str(directory),
                "validation": candidate["validation"],
            })
        summary["artifacts"] = artifacts
        summary["completedAt"] = utc_now()
        atomic_json(summary, self.summary_file)
        retained = {str(Path(item["directory"]) / "best-model.pt") for item in artifacts}
        for stage in summary["stages"]:
            for survivor in stage["survivors"]:
                checkpoint = str(Path(survivor["model"]))
                if checkpoint not in retained:
                    Path(checkpoint).unlink(missing_ok=True)
        for checkpoint in retained:
            Path(checkpoint).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    runner = CurriculumRunner(args.plan)
    if args.dry_run:
        print(json.dumps(runner.dry_run(), indent=2))
        return
    previous = read_json(runner.status_file) if runner.status_file.is_file() else {}
    previous_pid = previous.get("pid")
    if isinstance(previous_pid, int) and previous_pid != os.getpid():
        try:
            os.kill(previous_pid, 0)
        except OSError:
            pass
        else:
            raise RuntimeError(
                f"dynamic curriculum study is already running as PID {previous_pid}"
            )
    try:
        if args.wait_for_source:
            runner.wait_for_source()
        if args.refresh_last_completed_stage:
            runner.refresh_last_completed_stage()
            return
        runner.run()
    except KeyboardInterrupt:
        runner.write_status(stage="paused", pausedAt=utc_now(), message="Paused.")
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

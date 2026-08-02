from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from joint_price_oracle import parameter_count
from joint_price_oracle_sequence import (
    BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
    BOUNDARY_SEQUENCE_POLICY_DROPOUT_ARCHITECTURE_CONTRACT,
)
from train_joint_price_oracle import (
    DATA_CONTRACT,
    SEQUENCE_REUSE_TRAINING_CONTRACT,
    architecture_contract_for_model_config,
    atomic_torch_save,
    build_model,
    build_training_checkpoint,
    configuration_fingerprint,
    load_resume_checkpoint,
    resolve_training_config,
    validate_plan,
)


PLAN_FILES = {
    "lr1e4": (
        "joint-price-oracle-kl-minute-sequence-boundary-tcn-"
        "v23-reuse-lr1e4.json"
    ),
    "lr5e5": (
        "joint-price-oracle-kl-minute-sequence-boundary-tcn-"
        "v24-reuse-lr5e5.json"
    ),
}

EXPECTED_SETTINGS = {
    "lr1e4": {
        "version": 23,
        "learningRate": 1e-4,
        "schedulePatience": 12,
        "trainingFingerprint": (
            "0dd1abec6d57734215acc5a3718c7173fa9bc963ee54e73267e335a248ffdc46"
        ),
    },
    "lr5e5": {
        "version": 24,
        "learningRate": 5e-5,
        "schedulePatience": 16,
        "trainingFingerprint": (
            "75e72c53e5faff1d817eecee7ebfb7ab1db743684f6e9a0bbf6cede441fa1880"
        ),
    },
}

POLICY_DROPOUT_PLAN_FILES = {
    "policy_dropout": (
        "joint-price-oracle-kl-minute-sequence-boundary-tcn-"
        "v25-reuse-policy-dropout5e2.json"
    ),
    "policy_dropout_wd5e2": (
        "joint-price-oracle-kl-minute-sequence-boundary-tcn-"
        "v26-reuse-policy-dropout5e2-wd5e2.json"
    ),
}

POLICY_DROPOUT_EXPECTED_SETTINGS = {
    "policy_dropout": {
        "version": 25,
        "weightDecay": 0.01,
        "trainingFingerprint": (
            "06c958c709b59e3e4cad45befd334e898537abb9ee80cb042a7cca370dd8ffff"
        ),
    },
    "policy_dropout_wd5e2": {
        "version": 26,
        "weightDecay": 0.05,
        "trainingFingerprint": (
            "6f05cf5a503bbb1867f2de52dd5bd32661c27591e56cefee7355b28b30c27e85"
        ),
    },
}


def _optimizer_and_scheduler(
    model: torch.nn.Module,
    training: dict,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.ReduceLROnPlateau]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learningRate"]),
        betas=tuple(training["betas"]),
        eps=float(training["epsilon"]),
        weight_decay=float(training["weightDecay"]),
    )
    schedule = training["learningRateSchedule"]
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(schedule["factor"]),
        patience=int(schedule["patience"]),
        threshold=float(schedule["threshold"]),
        min_lr=float(schedule["minimumLearningRate"]),
    )
    return optimizer, scheduler


class RegularizedSequenceReusePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        plan_root = cls.repo_root / "ml" / "training-plans"
        cls.base = json.loads((
            plan_root
            / "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
        ).read_text(encoding="utf-8"))
        cls.plans = {
            name: json.loads((plan_root / file_name).read_text(encoding="utf-8"))
            for name, file_name in PLAN_FILES.items()
        }

    def test_plans_preserve_exact_reuse_model_data_and_update_semantics(
        self,
    ) -> None:
        immutable_top_level = (
            "targetReferenceDir",
            "historyDir",
            "samplingIntervalMs",
            "predictionDelayMs",
            "oracleTarget",
            "dataSplit",
            "model",
        )
        immutable_training = (
            "targetRepresentation",
            "policyOnly",
            "sequenceCoreTraining",
            "epochs",
            "batchSize",
            "evaluationBatchSize",
            "gradientAccumulationSteps",
            "betas",
            "epsilon",
            "weightDecay",
            "gradientClip",
            "seed",
            "device",
            "mixedPrecision",
            "logEverySteps",
            "prefetchBatches",
            "selectionMetric",
            "forecastHuberDelta",
            "volatilityFloor",
            "lossWeights",
        )
        identifiers: set[str] = set()
        run_dirs: set[str] = set()
        artifact_dirs: set[str] = set()
        fingerprints: set[str] = set()
        for name, plan in self.plans.items():
            with self.subTest(plan=name):
                validate_plan(plan)
                self.assertEqual(
                    plan["version"],
                    EXPECTED_SETTINGS[name]["version"],
                )
                self.assertEqual(plan["testPolicy"], "sealed-never-load")
                for key in immutable_top_level:
                    self.assertEqual(plan[key], self.base[key], key)
                for key in immutable_training:
                    self.assertEqual(
                        plan["training"][key],
                        self.base["training"][key],
                        key,
                    )
                training = plan["training"]
                self.assertEqual(training["sequenceCoreTraining"], {
                    "contract": SEQUENCE_REUSE_TRAINING_CONTRACT,
                    "coreRows": 1_440,
                })
                self.assertEqual(training["batchSize"], 1_440)
                self.assertEqual(training["evaluationBatchSize"], 1_440)
                self.assertEqual(training["gradientAccumulationSteps"], 1)
                self.assertEqual(plan["model"]["dropout"], 0)
                self.assertEqual(training["selectionMetric"], "klDivergence")
                self.assertEqual(
                    plan["oracleTarget"]["options"]["temperature"],
                    0.01,
                )
                self.assertEqual(training["patience"], 64)
                self.assertGreaterEqual(training["patience"], 40)
                self.assertEqual(training["closeCacheDays"], 512)
                self.assertEqual(training["targetCacheDays"], 512)
                self.assertNotIn("initialCheckpoint", plan)
                self.assertNotIn("initialCheckpoint", training)
                identifiers.add(plan["id"])
                run_dirs.add(plan["runDir"])
                artifact_dirs.add(plan["artifactDir"])
                fingerprint = configuration_fingerprint(
                    resolve_training_config(training)
                )
                self.assertEqual(
                    fingerprint,
                    EXPECTED_SETTINGS[name]["trainingFingerprint"],
                )
                fingerprints.add(fingerprint)

        self.assertEqual(len(identifiers), len(self.plans))
        self.assertEqual(len(run_dirs), len(self.plans))
        self.assertEqual(len(artifact_dirs), len(self.plans))
        self.assertEqual(len(fingerprints), len(self.plans))
        self.assertNotIn(self.base["id"], identifiers)
        self.assertNotIn(self.base["runDir"], run_dirs)
        self.assertNotIn(self.base["artifactDir"], artifact_dirs)

    def test_lr_and_plateau_settings_are_the_only_semantic_training_changes(
        self,
    ) -> None:
        allowed_changes = {
            "learningRate",
            "learningRateSchedule",
            "patience",
            "closeCacheDays",
            "targetCacheDays",
        }
        base_training = self.base["training"]
        for name, plan in self.plans.items():
            with self.subTest(plan=name):
                training = plan["training"]
                changed = {
                    key
                    for key in set(base_training) | set(training)
                    if base_training.get(key) != training.get(key)
                }
                self.assertEqual(changed, allowed_changes)
                learning_rate = EXPECTED_SETTINGS[name]["learningRate"]
                schedule_patience = EXPECTED_SETTINGS[name][
                    "schedulePatience"
                ]
                self.assertEqual(training["learningRate"], learning_rate)
                schedule = training["learningRateSchedule"]
                self.assertEqual(
                    schedule["type"],
                    "reduce-on-validation-plateau",
                )
                self.assertEqual(schedule["factor"], 0.5)
                self.assertEqual(schedule["patience"], schedule_patience)
                self.assertGreaterEqual(schedule_patience, 12)
                self.assertLessEqual(schedule_patience, 16)
                self.assertEqual(schedule["threshold"], 1e-5)
                self.assertEqual(schedule["minimumLearningRate"], 1e-6)
                self.assertEqual(training["weightDecay"], 0.01)

    def test_plan_validation_rejects_broken_reuse_or_sealed_test_contracts(
        self,
    ) -> None:
        mutations = (
            ("testPolicy", None, "load-during-training", "sealed"),
            ("model", "dropout", 0.05, "dropout=0"),
            ("training", "batchSize", 720, "batch sizes"),
            ("training", "selectionMetric", "loss", "raw klDivergence"),
        )
        for section, key, value, message in mutations:
            with self.subTest(section=section, key=key):
                changed = copy.deepcopy(self.plans["lr1e4"])
                if key is None:
                    changed[section] = value
                else:
                    changed[section][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_plan(changed)

    def test_each_checkpoint_resumes_only_matching_plan_and_fingerprint(
        self,
    ) -> None:
        dataset_fingerprint = "regularized-reuse-dataset-v1"
        names = tuple(self.plans)
        for index, name in enumerate(names):
            plan = self.plans[name]
            other = self.plans[names[1 - index]]
            config = copy.deepcopy(plan["model"])
            config["tokenWidth"] = 8
            config["policyHiddenWidth"] = 12
            model = build_model(config)
            optimizer, scheduler = _optimizer_and_scheduler(
                model,
                plan["training"],
            )
            count = parameter_count(model)
            fingerprint = configuration_fingerprint(
                resolve_training_config(plan["training"])
            )
            checkpoint = build_training_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=9,
                global_step=3_720,
                best_validation=0.91,
                best_epoch=7,
                stale_epochs=2,
                validation={"klDivergence": 0.93},
                model_parameters=count,
                dataset_fingerprint=dataset_fingerprint,
                model_config=config,
                plan_id=plan["id"],
                device=torch.device("cpu"),
                interrupted=False,
                training_config_fingerprint=fingerprint,
            )
            self.assertEqual(checkpoint["dataContract"], DATA_CONTRACT)
            with self.subTest(plan=name), tempfile.TemporaryDirectory() as directory:
                file = (
                    Path(directory)
                    / "data"
                    / "training"
                    / "runs"
                    / "regularized-reuse"
                    / "checkpoints"
                    / "last.json"
                )
                atomic_torch_save(checkpoint, file)
                resumed = load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": plan["id"]},
                    config,
                    dataset_fingerprint,
                    count,
                    torch.device("cpu"),
                    training_config_fingerprint=fingerprint,
                )
                self.assertEqual(resumed, (10, 3_720, 0.91, 7, 2))
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_resume_checkpoint(
                        file,
                        model,
                        optimizer,
                        scheduler,
                        {"id": other["id"]},
                        config,
                        dataset_fingerprint,
                        count,
                        torch.device("cpu"),
                        training_config_fingerprint=fingerprint,
                    )
                other_fingerprint = configuration_fingerprint(
                    resolve_training_config(other["training"])
                )
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_resume_checkpoint(
                        file,
                        model,
                        optimizer,
                        scheduler,
                        {"id": plan["id"]},
                        config,
                        dataset_fingerprint,
                        count,
                        torch.device("cpu"),
                        training_config_fingerprint=other_fingerprint,
                    )


class PolicyDropoutSequenceReusePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        plan_root = cls.repo_root / "ml" / "training-plans"
        cls.base = json.loads((
            plan_root
            / "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
        ).read_text(encoding="utf-8"))
        cls.lr_controls = {
            name: json.loads((plan_root / file_name).read_text(encoding="utf-8"))
            for name, file_name in PLAN_FILES.items()
        }
        cls.plans = {
            name: json.loads((plan_root / file_name).read_text(encoding="utf-8"))
            for name, file_name in POLICY_DROPOUT_PLAN_FILES.items()
        }

    def test_plans_preserve_reuse_topology_data_and_raw_kl_contract(
        self,
    ) -> None:
        immutable_top_level = (
            "targetReferenceDir",
            "historyDir",
            "samplingIntervalMs",
            "predictionDelayMs",
            "oracleTarget",
            "dataSplit",
        )
        immutable_training = (
            "targetRepresentation",
            "policyOnly",
            "sequenceCoreTraining",
            "epochs",
            "batchSize",
            "evaluationBatchSize",
            "gradientAccumulationSteps",
            "learningRate",
            "betas",
            "epsilon",
            "gradientClip",
            "seed",
            "device",
            "mixedPrecision",
            "logEverySteps",
            "prefetchBatches",
            "selectionMetric",
            "forecastHuberDelta",
            "volatilityFloor",
            "lossWeights",
        )
        identifiers: set[str] = set()
        run_dirs: set[str] = set()
        artifact_dirs: set[str] = set()
        fingerprints: set[str] = set()
        for name, plan in self.plans.items():
            with self.subTest(plan=name):
                validate_plan(plan)
                expected = POLICY_DROPOUT_EXPECTED_SETTINGS[name]
                self.assertEqual(plan["version"], expected["version"])
                self.assertEqual(plan["testPolicy"], "sealed-never-load")
                for key in immutable_top_level:
                    self.assertEqual(plan[key], self.base[key], key)

                self.assertEqual(
                    set(plan["model"]),
                    set(self.base["model"]) | {"policyDropout"},
                )
                for key, value in self.base["model"].items():
                    self.assertEqual(plan["model"][key], value, key)
                self.assertEqual(plan["model"]["dropout"], 0)
                self.assertEqual(plan["model"]["policyDropout"], 0.05)

                training = plan["training"]
                for key in immutable_training:
                    self.assertEqual(
                        training[key],
                        self.base["training"][key],
                        key,
                    )
                self.assertEqual(training["sequenceCoreTraining"], {
                    "contract": SEQUENCE_REUSE_TRAINING_CONTRACT,
                    "coreRows": 1_440,
                })
                self.assertEqual(training["batchSize"], 1_440)
                self.assertEqual(training["evaluationBatchSize"], 1_440)
                self.assertEqual(training["gradientAccumulationSteps"], 1)
                self.assertEqual(training["selectionMetric"], "klDivergence")
                self.assertEqual(
                    plan["oracleTarget"]["options"]["temperature"],
                    0.01,
                )
                self.assertEqual(training["patience"], 64)
                self.assertEqual(training["closeCacheDays"], 512)
                self.assertEqual(training["targetCacheDays"], 512)
                self.assertNotIn("initialCheckpoint", plan)
                self.assertNotIn("initialCheckpoint", training)

                identifiers.add(plan["id"])
                run_dirs.add(plan["runDir"])
                artifact_dirs.add(plan["artifactDir"])
                fingerprint = configuration_fingerprint(
                    resolve_training_config(training)
                )
                self.assertEqual(
                    fingerprint,
                    expected["trainingFingerprint"],
                )
                fingerprints.add(fingerprint)

        self.assertEqual(len(identifiers), len(self.plans))
        self.assertEqual(len(run_dirs), len(self.plans))
        self.assertEqual(len(artifact_dirs), len(self.plans))
        self.assertEqual(len(fingerprints), len(self.plans))
        for reserved in (self.base, *self.lr_controls.values()):
            self.assertNotIn(reserved["id"], identifiers)
            self.assertNotIn(reserved["runDir"], run_dirs)
            self.assertNotIn(reserved["artifactDir"], artifact_dirs)

    def test_policy_dropout_and_weight_decay_are_the_only_new_semantics(
        self,
    ) -> None:
        base_training = self.base["training"]
        for name, plan in self.plans.items():
            with self.subTest(plan=name):
                model_changes = {
                    key
                    for key in set(self.base["model"]) | set(plan["model"])
                    if self.base["model"].get(key) != plan["model"].get(key)
                }
                self.assertEqual(model_changes, {"policyDropout"})

                training = plan["training"]
                expected_changes = {
                    "learningRateSchedule",
                    "patience",
                    "closeCacheDays",
                    "targetCacheDays",
                }
                if name == "policy_dropout_wd5e2":
                    expected_changes.add("weightDecay")
                training_changes = {
                    key
                    for key in set(base_training) | set(training)
                    if base_training.get(key) != training.get(key)
                }
                self.assertEqual(training_changes, expected_changes)
                self.assertEqual(training["learningRate"], 2e-4)
                self.assertEqual(
                    training["weightDecay"],
                    POLICY_DROPOUT_EXPECTED_SETTINGS[name]["weightDecay"],
                )
                schedule = training["learningRateSchedule"]
                self.assertEqual(
                    schedule,
                    {
                        "type": "reduce-on-validation-plateau",
                        "factor": 0.5,
                        "patience": 12,
                        "threshold": 1e-5,
                        "minimumLearningRate": 1e-6,
                    },
                )

        ordinary = self.plans["policy_dropout"]
        stronger = self.plans["policy_dropout_wd5e2"]
        self.assertEqual(ordinary["model"], stronger["model"])
        training_differences = {
            key
            for key in set(ordinary["training"]) | set(stronger["training"])
            if ordinary["training"].get(key)
            != stronger["training"].get(key)
        }
        self.assertEqual(training_differences, {"weightDecay"})

    def test_policy_dropout_preserves_parameterized_model_topology(self) -> None:
        base_config = copy.deepcopy(self.base["model"])
        base_config["tokenWidth"] = 8
        base_config["policyHiddenWidth"] = 12
        base_model = build_model(base_config)
        self.assertEqual(
            architecture_contract_for_model_config(base_config),
            BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
        )
        self.assertEqual(
            base_model.architecture_contract,
            BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
        )
        self.assertEqual(base_model.core_dropout, 0)
        self.assertEqual(base_model.policy_dropout, 0)
        self.assertEqual(base_model.policy_head[2].p, 0)
        base_signature = {
            name: tuple(parameter.shape)
            for name, parameter in base_model.named_parameters()
        }
        for name, plan in self.plans.items():
            with self.subTest(plan=name):
                config = copy.deepcopy(plan["model"])
                config["tokenWidth"] = 8
                config["policyHiddenWidth"] = 12
                model = build_model(config)
                self.assertEqual(
                    architecture_contract_for_model_config(config),
                    BOUNDARY_SEQUENCE_POLICY_DROPOUT_ARCHITECTURE_CONTRACT,
                )
                self.assertEqual(
                    model.architecture_contract,
                    BOUNDARY_SEQUENCE_POLICY_DROPOUT_ARCHITECTURE_CONTRACT,
                )
                self.assertEqual(model.core_dropout, 0)
                self.assertEqual(model.policy_dropout, 0.05)
                self.assertEqual(model.policy_head[2].p, 0.05)
                for block in model.blocks:
                    self.assertEqual(block.dropout.p, 0)
                signature = {
                    parameter_name: tuple(parameter.shape)
                    for parameter_name, parameter in model.named_parameters()
                }
                self.assertEqual(signature, base_signature)
                self.assertEqual(
                    parameter_count(model),
                    parameter_count(base_model),
                )

    def test_legacy_reuse_plans_retain_original_architecture_contract(
        self,
    ) -> None:
        legacy_plans = {"v18_reuse": self.base, **self.lr_controls}
        for name, plan in legacy_plans.items():
            with self.subTest(plan=name):
                config = copy.deepcopy(plan["model"])
                config["tokenWidth"] = 8
                config["policyHiddenWidth"] = 12
                self.assertNotIn("policyDropout", config)
                model = build_model(config)
                self.assertEqual(
                    architecture_contract_for_model_config(config),
                    BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
                )
                self.assertEqual(
                    model.architecture_contract,
                    BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
                )
                self.assertEqual(model.core_dropout, 0)
                self.assertEqual(model.policy_dropout, 0)
                self.assertEqual(model.policy_head[2].p, 0)

    def test_policy_dropout_model_runs_train_and_eval_sequence_forwards(
        self,
    ) -> None:
        config = copy.deepcopy(self.plans["policy_dropout"]["model"])
        config["tokenWidth"] = 8
        config["policyHiddenWidth"] = 12
        model = build_model(config)
        close_count = 61 * 60 + 1
        log_closes = torch.linspace(10.0, 10.01, close_count)
        closes = torch.exp(log_closes).reshape(1, close_count, 1)

        model.train()
        train_logits = model.forward_sequence_core(closes)
        self.assertEqual(tuple(train_logits.shape), (1, 2, 101))
        self.assertTrue(torch.isfinite(train_logits).all().item())

        model.eval()
        with torch.no_grad():
            eval_logits = model.forward_sequence_core(closes)
            fixed_window_logits = model.forward_policy_logits(
                closes[:, :3_601, :]
            )
        self.assertEqual(tuple(eval_logits.shape), (1, 2, 101))
        self.assertEqual(tuple(fixed_window_logits.shape), (1, 101))
        self.assertTrue(torch.isfinite(eval_logits).all().item())
        self.assertTrue(torch.isfinite(fixed_window_logits).all().item())
        torch.testing.assert_close(eval_logits[:, 0, :], fixed_window_logits)

    def test_plan_validation_keeps_core_dropout_forbidden(self) -> None:
        changed = copy.deepcopy(self.plans["policy_dropout"])
        changed["model"]["dropout"] = 0.05
        with self.assertRaisesRegex(ValueError, "model.dropout=0"):
            validate_plan(changed)

    def test_each_checkpoint_resumes_only_matching_plan_and_fingerprint(
        self,
    ) -> None:
        dataset_fingerprint = "policy-dropout-reuse-dataset-v1"
        names = tuple(self.plans)
        for index, name in enumerate(names):
            plan = self.plans[name]
            other = self.plans[names[1 - index]]
            config = copy.deepcopy(plan["model"])
            config["tokenWidth"] = 8
            config["policyHiddenWidth"] = 12
            model = build_model(config)
            optimizer, scheduler = _optimizer_and_scheduler(
                model,
                plan["training"],
            )
            count = parameter_count(model)
            fingerprint = configuration_fingerprint(
                resolve_training_config(plan["training"])
            )
            checkpoint = build_training_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=9,
                global_step=3_720,
                best_validation=0.91,
                best_epoch=7,
                stale_epochs=2,
                validation={"klDivergence": 0.93},
                model_parameters=count,
                dataset_fingerprint=dataset_fingerprint,
                model_config=config,
                plan_id=plan["id"],
                device=torch.device("cpu"),
                interrupted=False,
                training_config_fingerprint=fingerprint,
            )
            self.assertEqual(checkpoint["dataContract"], DATA_CONTRACT)
            self.assertEqual(checkpoint["modelConfig"], config)
            self.assertEqual(checkpoint["modelConfig"]["policyDropout"], 0.05)
            self.assertEqual(
                checkpoint["architectureContract"],
                BOUNDARY_SEQUENCE_POLICY_DROPOUT_ARCHITECTURE_CONTRACT,
            )
            with self.subTest(plan=name), tempfile.TemporaryDirectory() as directory:
                file = (
                    Path(directory)
                    / "data"
                    / "training"
                    / "runs"
                    / "policy-dropout-reuse"
                    / "checkpoints"
                    / "last.json"
                )
                atomic_torch_save(checkpoint, file)
                resumed = load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": plan["id"]},
                    config,
                    dataset_fingerprint,
                    count,
                    torch.device("cpu"),
                    training_config_fingerprint=fingerprint,
                )
                self.assertEqual(resumed, (10, 3_720, 0.91, 7, 2))
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_resume_checkpoint(
                        file,
                        model,
                        optimizer,
                        scheduler,
                        {"id": other["id"]},
                        config,
                        dataset_fingerprint,
                        count,
                        torch.device("cpu"),
                        training_config_fingerprint=fingerprint,
                    )
                other_fingerprint = configuration_fingerprint(
                    resolve_training_config(other["training"])
                )
                with self.assertRaisesRegex(ValueError, "incompatible"):
                    load_resume_checkpoint(
                        file,
                        model,
                        optimizer,
                        scheduler,
                        {"id": plan["id"]},
                        config,
                        dataset_fingerprint,
                        count,
                        torch.device("cpu"),
                        training_config_fingerprint=other_fingerprint,
                    )


if __name__ == "__main__":
    unittest.main()

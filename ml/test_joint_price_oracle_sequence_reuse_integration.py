from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from joint_price_oracle import JointLossWeights, parameter_count
from train_joint_price_oracle import (
    DATA_CONTRACT,
    SEQUENCE_REUSE_TRAINING_CONTRACT,
    CausalOracleDataset,
    CausalSegment,
    atomic_torch_save,
    build_model,
    build_training_checkpoint,
    configuration_fingerprint,
    evaluate,
    load_resume_checkpoint,
    resolve_training_config,
    validate_plan,
)


class _CloseCache:
    def __init__(self) -> None:
        self.requests: list[tuple[int, int]] = []

    def range(self, first_close_time: int, last_close_time: int) -> np.ndarray:
        self.requests.append((first_close_time, last_close_time))
        count = (last_close_time - first_close_time) // 1_000 + 1
        time = np.arange(count, dtype=np.float32)
        return np.exp(10.5 + time * 1e-6).astype(np.float32)


class _TargetCache:
    rows_per_file = 1_440
    action_count = 101

    def __init__(self, target_file: Path) -> None:
        self.target_file = target_file
        self.rows = torch.zeros(self.rows_per_file, self.action_count)
        indices = torch.arange(self.rows_per_file) % self.action_count
        self.rows[torch.arange(self.rows_per_file), indices] = 1

    def load(self, target_file: Path) -> torch.Tensor:
        if target_file != self.target_file:
            raise AssertionError("unexpected or sealed target file requested")
        return self.rows


def make_dataset(count: int = 7) -> tuple[
    CausalOracleDataset,
    _CloseCache,
    _TargetCache,
]:
    target_file = Path("validation-targets.json")
    prediction_time = 120 * 60_000 + 999
    segment = CausalSegment(
        split="validation",
        prediction_time_start=prediction_time,
        count=count,
        target_file=target_file,
        target_row_offset=0,
        step_ms=60_000,
    )
    dataset = CausalOracleDataset(
        Path("unused-history"),
        {"train": [], "validation": [segment], "test": []},
        context_length=3_601,
        forecast_horizon=3_600,
        target_rows_per_file=1_440,
        action_count=101,
        include_future_closes=False,
    )
    closes = _CloseCache()
    targets = _TargetCache(target_file)
    dataset.close_cache = closes
    dataset.target_cache = targets
    return dataset, closes, targets


class SequenceCoreDatasetIntegrationTest(unittest.TestCase):
    def test_core_chunks_preserve_timestamps_targets_and_weighting(self) -> None:
        dataset, close_cache, _targets = make_dataset(count=5)
        self.assertEqual(dataset.sequence_core_batch_count("validation", 3), 2)
        batches = list(dataset.iter_sequence_core_batches(
            "validation",
            3,
            receptive_field_minutes=60,
            shuffle=False,
            seed=79,
        ))
        self.assertEqual(len(batches), 2)
        first_closes, first_targets = batches[0]
        second_closes, second_targets = batches[1]
        self.assertEqual(tuple(first_closes.shape), (1, 62 * 60 + 1, 1))
        self.assertEqual(tuple(second_closes.shape), (1, 61 * 60 + 1, 1))
        self.assertEqual(tuple(first_targets.shape), (1, 3, 101))
        self.assertEqual(tuple(second_targets.shape), (1, 2, 101))
        self.assertEqual(
            first_targets[0].argmax(dim=-1).tolist(),
            [0, 1, 2],
        )
        self.assertEqual(
            second_targets[0].argmax(dim=-1).tolist(),
            [3, 4],
        )
        prediction_time = dataset.segments["validation"][0].prediction_time_start
        self.assertEqual(close_cache.requests, [
            (
                prediction_time - 3_600_000,
                prediction_time + 2 * 60_000,
            ),
            (
                prediction_time + 3 * 60_000 - 3_600_000,
                prediction_time + 4 * 60_000,
            ),
        ])

    def test_sequence_validation_raw_kl_equals_fixed_window_validation(
        self,
    ) -> None:
        fixed_dataset, _fixed_closes, _fixed_targets = make_dataset(count=7)
        reused_dataset, _reused_closes, _reused_targets = make_dataset(count=7)
        torch.manual_seed(83)
        model = build_model({
            "variant": "minute_sequence_boundary_tcn",
            "contextLength": 3_601,
            "forecastHorizon": 3_600,
            "variableCount": 1,
            "actionCount": 101,
            "receptiveFieldMinutes": 60,
            "tokenWidth": 8,
            "policyHiddenWidth": 12,
            "dropout": 0,
            "featureEpsilon": 1e-8,
        })
        common_training = {
            "policyOnly": True,
            "seed": 83,
            "prefetchBatches": 1,
            "mixedPrecision": "float32",
        }
        weights = JointLossWeights(forecast=0, soft_layer_norm=0)
        action_grid = torch.arange(101, dtype=torch.float32)
        fixed = evaluate(
            model,
            fixed_dataset,
            "validation",
            7,
            torch.device("cpu"),
            common_training,
            weights,
            action_grid,
            0.00175,
            0.01,
            maximum_batches=None,
        )
        reused_training = dict(common_training)
        reused_training["sequenceCoreTraining"] = {
            "contract": SEQUENCE_REUSE_TRAINING_CONTRACT,
            "coreRows": 7,
        }
        reused = evaluate(
            model,
            reused_dataset,
            "validation",
            7,
            torch.device("cpu"),
            reused_training,
            weights,
            action_grid,
            0.00175,
            0.01,
            maximum_batches=None,
        )
        self.assertEqual(set(reused), set(fixed))
        for name in fixed:
            with self.subTest(metric=name):
                self.assertAlmostEqual(
                    float(reused[name]),
                    float(fixed[name]),
                    places=7,
                )


class SequenceCorePlanAndResumeIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        plan_root = cls.repo_root / "ml" / "training-plans"
        cls.plan = json.loads((
            plan_root
            / "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
        ).read_text(encoding="utf-8"))
        cls.v18 = json.loads((
            plan_root
            / "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18.json"
        ).read_text(encoding="utf-8"))

    def test_reuse_plan_is_immutable_raw_kl_and_isolated_from_v18(self) -> None:
        validate_plan(self.plan)
        self.assertEqual(self.plan["version"], 20)
        self.assertIn("v18-reuse", self.plan["id"])
        self.assertIn("v18-reuse", self.plan["runDir"])
        self.assertNotEqual(self.plan["id"], self.v18["id"])
        self.assertNotEqual(self.plan["runDir"], self.v18["runDir"])
        self.assertEqual(self.plan["model"]["dropout"], 0)
        self.assertEqual(self.v18["model"]["dropout"], 0.05)
        training = self.plan["training"]
        self.assertEqual(training["selectionMetric"], "klDivergence")
        self.assertEqual(
            self.plan["oracleTarget"]["options"]["temperature"],
            0.01,
        )
        self.assertEqual(training["sequenceCoreTraining"], {
            "contract": SEQUENCE_REUSE_TRAINING_CONTRACT,
            "coreRows": 1_440,
        })
        self.assertNotIn("evaluateTest", training)
        self.assertNotIn("calibration", training)

    def test_plan_rejects_non_equivalent_reuse_configurations(self) -> None:
        cases = (
            (
                ("model", "dropout", 0.05),
                "dropout=0",
            ),
            (
                ("model", "variant", "minute_sequence_boundary_ma_tcn"),
                "non-MA",
            ),
            (
                ("training", "batchSize", 720),
                "batch sizes",
            ),
            (
                (
                    "training",
                    "sequenceCoreTraining",
                    {"contract": "wrong", "coreRows": 1_440},
                ),
                "contract",
            ),
        )
        for (section, key, value), message in cases:
            with self.subTest(key=key, value=value):
                changed = copy.deepcopy(self.plan)
                changed[section][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_plan(changed)

    def test_reuse_checkpoint_resumes_only_its_immutable_plan(self) -> None:
        config = copy.deepcopy(self.plan["model"])
        config["tokenWidth"] = 8
        config["policyHiddenWidth"] = 12
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=48,
        )
        training_fingerprint = configuration_fingerprint(
            resolve_training_config(self.plan["training"])
        )
        dataset_fingerprint = (
            "a9e75c9edd2660190d77b0933b1f7ccc93d107531fdd6d04fd90357b6d8a3783"
        )
        count = parameter_count(model)
        checkpoint = build_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=3,
            global_step=1_116,
            best_validation=0.9,
            best_epoch=3,
            stale_epochs=0,
            validation={"klDivergence": 0.9},
            model_parameters=count,
            dataset_fingerprint=dataset_fingerprint,
            model_config=config,
            plan_id=self.plan["id"],
            device=torch.device("cpu"),
            interrupted=False,
            training_config_fingerprint=training_fingerprint,
        )
        self.assertEqual(checkpoint["dataContract"], DATA_CONTRACT)
        with tempfile.TemporaryDirectory() as directory:
            file = (
                Path(directory)
                / "data"
                / "training"
                / "runs"
                / "reuse"
                / "checkpoints"
                / "last.json"
            )
            atomic_torch_save(checkpoint, file)
            resumed = load_resume_checkpoint(
                file,
                model,
                optimizer,
                scheduler,
                {"id": self.plan["id"]},
                config,
                dataset_fingerprint,
                count,
                torch.device("cpu"),
                training_config_fingerprint=training_fingerprint,
            )
            self.assertEqual(resumed, (4, 1_116, 0.9, 3, 0))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": self.v18["id"]},
                    config,
                    dataset_fingerprint,
                    count,
                    torch.device("cpu"),
                    training_config_fingerprint=training_fingerprint,
                )


if __name__ == "__main__":
    unittest.main()

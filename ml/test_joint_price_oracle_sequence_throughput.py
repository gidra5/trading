from __future__ import annotations

import argparse
import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from joint_price_oracle import JointLossWeights
from train_joint_price_oracle import (
    PACKED_SEQUENCE_REUSE_TRAINING_CONTRACT,
    CausalOracleDataset,
    CausalSegment,
    build_model,
    configuration_fingerprint,
    resolve_training_config,
    sequence_core_policy_forward_objective,
    sequence_core_target_count,
    validate_experiment_arguments,
    validate_plan,
)


class _CloseCache:
    def __init__(self) -> None:
        self.requests: list[tuple[int, int]] = []

    def range(self, first_close_time: int, last_close_time: int) -> np.ndarray:
        self.requests.append((first_close_time, last_close_time))
        count = (last_close_time - first_close_time) // 1_000 + 1
        return np.linspace(100, 101, count, dtype=np.float32)


class _TargetCache:
    rows_per_file = 100
    action_count = 101

    def __init__(self, rows: dict[Path, torch.Tensor]) -> None:
        self.rows = rows
        self.requests: list[Path] = []

    def load(self, target_file: Path) -> torch.Tensor:
        self.requests.append(target_file)
        if target_file not in self.rows:
            raise AssertionError("unexpected or sealed target file requested")
        return self.rows[target_file]


def _target_rows(base: int) -> torch.Tensor:
    rows = torch.zeros(100, 101)
    indices = (torch.arange(100) + base) % 101
    rows[torch.arange(100), indices] = 1
    return rows


def _packed_dataset() -> tuple[
    CausalOracleDataset,
    _CloseCache,
    _TargetCache,
    tuple[CausalSegment, CausalSegment, CausalSegment],
]:
    first_file = Path("train-a.json")
    second_file = Path("train-b.json")
    third_file = Path("train-after-gap.json")
    sealed_test_file = Path("sealed-test.json")
    start = 120 * 60_000 + 999
    first = CausalSegment(
        split="train",
        prediction_time_start=start,
        count=4,
        target_file=first_file,
        target_row_offset=10,
        step_ms=60_000,
    )
    second = CausalSegment(
        split="train",
        prediction_time_start=first.prediction_time_end + 60_000,
        count=4,
        target_file=second_file,
        target_row_offset=20,
        step_ms=60_000,
    )
    third = CausalSegment(
        split="train",
        prediction_time_start=second.prediction_time_end + 120_000,
        count=4,
        target_file=third_file,
        target_row_offset=30,
        step_ms=60_000,
    )
    sealed = CausalSegment(
        split="test",
        prediction_time_start=third.prediction_time_end + 60_000,
        count=1,
        target_file=sealed_test_file,
        target_row_offset=0,
        step_ms=60_000,
    )
    dataset = CausalOracleDataset(
        Path("unused-history"),
        {
            "train": [first, second, third],
            "validation": [],
            "test": [sealed],
        },
        context_length=3_601,
        forecast_horizon=3_600,
        target_rows_per_file=100,
        action_count=101,
        include_future_closes=False,
    )
    closes = _CloseCache()
    targets = _TargetCache({
        first_file: _target_rows(0),
        second_file: _target_rows(0),
        third_file: _target_rows(0),
    })
    dataset.close_cache = closes
    dataset.target_cache = targets
    return dataset, closes, targets, (first, second, third)


def _model_config() -> dict:
    return {
        "variant": "minute_sequence_boundary_tcn",
        "contextLength": 3_601,
        "forecastHorizon": 3_600,
        "actionCount": 101,
        "variableCount": 1,
        "receptiveFieldMinutes": 60,
        "tokenWidth": 8,
        "policyHiddenWidth": 12,
        "dropout": 0,
        "featureEpsilon": 1e-8,
    }


def _synthetic_closes(core_rows: int, offset: float) -> torch.Tensor:
    token_count = core_rows + 59
    returns = torch.linspace(
        -2e-4 + offset,
        2e-4 + offset,
        token_count * 60,
    )
    return torch.cat([torch.zeros(1), returns]).cumsum(0).exp().view(
        1,
        token_count * 60 + 1,
        1,
    )


class PackedSequenceCoreDatasetTest(unittest.TestCase):
    def test_packs_across_files_but_not_across_history_gaps(self) -> None:
        dataset, closes, targets, (first, second, third) = _packed_dataset()
        self.assertEqual(dataset.sequence_core_batch_count(
            "train",
            6,
            pack_contiguous_runs=True,
        ), 2)
        batches = list(dataset.iter_sequence_core_batches(
            "train",
            6,
            receptive_field_minutes=60,
            shuffle=False,
            seed=101,
            pack_contiguous_runs=True,
        ))
        self.assertEqual(len(batches), 2)

        first_closes, first_targets, first_mask = batches[0]
        self.assertEqual(tuple(first_closes.shape), (1, 65 * 60 + 1, 1))
        self.assertEqual(tuple(first_targets.shape), (1, 6, 101))
        self.assertTrue(bool(first_mask.all()))
        self.assertEqual(
            first_targets[first_mask].argmax(dim=-1).tolist(),
            [10, 11, 12, 13, 20, 21],
        )

        second_closes, second_targets, second_mask = batches[1]
        self.assertEqual(tuple(second_closes.shape), (2, 63 * 60 + 1, 1))
        self.assertEqual(tuple(second_targets.shape), (2, 4, 101))
        self.assertEqual(second_mask.tolist(), [
            [True, True, False, False],
            [True, True, True, True],
        ])
        self.assertEqual(
            second_targets[second_mask].argmax(dim=-1).tolist(),
            [22, 23, 30, 31, 32, 33],
        )
        self.assertEqual(sequence_core_target_count(
            second_targets,
            second_mask,
        ), 6)

        self.assertEqual(closes.requests, [
            (
                first.prediction_time_start - 3_600_000,
                second.prediction_time_start + 60_000,
            ),
            (
                second.prediction_time_start + 2 * 60_000 - 3_600_000,
                second.prediction_time_end,
            ),
            (
                third.prediction_time_start - 3_600_000,
                third.prediction_time_end,
            ),
        ])
        self.assertNotIn(Path("sealed-test.json"), targets.requests)

    def test_shuffle_is_deterministic_and_covers_every_target_once(self) -> None:
        def rows_for(seed: int) -> list[list[int]]:
            dataset, _closes, _targets, _segments = _packed_dataset()
            return [
                target[mask].argmax(dim=-1).tolist()
                for _input, target, mask in dataset.iter_sequence_core_batches(
                    "train",
                    6,
                    receptive_field_minutes=60,
                    shuffle=True,
                    seed=seed,
                    pack_contiguous_runs=True,
                )
            ]

        first = rows_for(103)
        self.assertEqual(first, rows_for(103))
        self.assertEqual(
            sorted(value for batch in first for value in batch),
            [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33],
        )


class PackedSequenceCoreObjectiveTest(unittest.TestCase):
    def test_masked_batch_matches_row_weighted_microcore_accumulation(self) -> None:
        torch.manual_seed(107)
        packed_model = build_model(_model_config())
        reference_model = copy.deepcopy(packed_model)
        short_closes = _synthetic_closes(2, 0)
        long_closes = _synthetic_closes(4, 1e-6)
        padded_short = torch.empty_like(long_closes)
        padded_short[:, :short_closes.shape[1]] = short_closes
        padded_short[:, short_closes.shape[1]:] = short_closes[:, -1:]
        packed_closes = torch.cat([padded_short, long_closes], dim=0)
        short_target = torch.softmax(torch.randn(1, 2, 101), dim=-1)
        long_target = torch.softmax(torch.randn(1, 4, 101), dim=-1)
        packed_target = torch.zeros(2, 4, 101)
        packed_target[0, :2] = short_target[0]
        packed_target[1] = long_target[0]
        mask = torch.tensor([
            [True, True, False, False],
            [True, True, True, True],
        ])
        weights = JointLossWeights(forecast=0, soft_layer_norm=0)

        packed_logits, packed_metrics = (
            sequence_core_policy_forward_objective(
                packed_model,
                packed_closes,
                packed_target,
                weights,
                target_mask=mask,
            )
        )
        short_logits, short_metrics = (
            sequence_core_policy_forward_objective(
                reference_model,
                short_closes,
                short_target,
                weights,
            )
        )
        long_logits, long_metrics = (
            sequence_core_policy_forward_objective(
                reference_model,
                long_closes,
                long_target,
                weights,
            )
        )
        torch.testing.assert_close(packed_logits[0, :2], short_logits[0])
        torch.testing.assert_close(packed_logits[1], long_logits[0])
        for name, packed_value in packed_metrics.items():
            expected = (
                short_metrics[name] * (2 / 6)
                + long_metrics[name] * (4 / 6)
            )
            torch.testing.assert_close(packed_value, expected)

        packed_metrics["loss"].backward()
        (
            short_metrics["loss"] * (2 / 6)
            + long_metrics["loss"] * (4 / 6)
        ).backward()
        for packed_parameter, reference_parameter in zip(
            packed_model.parameters(),
            reference_model.parameters(),
        ):
            torch.testing.assert_close(
                packed_parameter.grad,
                reference_parameter.grad,
                atol=2e-6,
                rtol=2e-5,
            )

        packed_optimizer = torch.optim.AdamW(
            packed_model.parameters(),
            lr=2e-4,
            weight_decay=0.01,
        )
        reference_optimizer = torch.optim.AdamW(
            reference_model.parameters(),
            lr=2e-4,
            weight_decay=0.01,
        )
        packed_optimizer.step()
        reference_optimizer.step()
        for packed_parameter, reference_parameter in zip(
            packed_model.parameters(),
            reference_model.parameters(),
        ):
            torch.testing.assert_close(
                packed_parameter,
                reference_parameter,
                atol=2e-6,
                rtol=3e-4,
            )


class PackedSequenceCorePlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parent / "training-plans"
        cls.reuse = json.loads((root /
            "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
        ).read_text(encoding="utf-8"))
        cls.plans = [
            json.loads((root / name).read_text(encoding="utf-8"))
            for name in (
                "joint-price-oracle-kl-minute-sequence-boundary-tcn-v21-packed-4x.json",
                "joint-price-oracle-kl-minute-sequence-boundary-tcn-v22-packed-8x.json",
            )
        ]

    def test_plans_are_valid_isolated_large_batches(self) -> None:
        fingerprints = set()
        for plan, core_rows in zip(self.plans, (5_760, 11_520)):
            validate_plan(plan)
            self.assertNotEqual(plan["id"], self.reuse["id"])
            self.assertNotEqual(plan["runDir"], self.reuse["runDir"])
            training = plan["training"]
            self.assertEqual(
                training["sequenceCoreTraining"]["contract"],
                PACKED_SEQUENCE_REUSE_TRAINING_CONTRACT,
            )
            self.assertEqual(training["batchSize"], core_rows)
            self.assertEqual(training["evaluationBatchSize"], core_rows)
            self.assertEqual(training["gradientAccumulationSteps"], 1)
            self.assertEqual(training["closeCacheDays"], 512)
            self.assertEqual(training["targetCacheDays"], 512)
            fingerprints.add(configuration_fingerprint(
                resolve_training_config(training)
            ))
        self.assertEqual(len(fingerprints), 2)

    def test_plan_rejects_ambiguous_packed_optimizer_batches(self) -> None:
        changed = copy.deepcopy(self.plans[0])
        changed["training"]["gradientAccumulationSteps"] = 2
        with self.assertRaisesRegex(ValueError, "nested accumulation"):
            validate_plan(changed)
        changed = copy.deepcopy(self.plans[0])
        changed["training"]["sequenceCoreTraining"]["coreRows"] = 5_000
        changed["training"]["batchSize"] = 5_000
        changed["training"]["evaluationBatchSize"] = 5_000
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_plan(changed)

    def test_runtime_cache_override_is_nonsemantic_and_validated(self) -> None:
        before = configuration_fingerprint(resolve_training_config(
            self.reuse["training"]
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arguments = argparse.Namespace(
                maximum_batches=None,
                stop_after_epoch=None,
                evaluate_test=False,
                check_data=False,
                disposable_smoke=False,
                runtime_cache_days=512,
            )
            validate_experiment_arguments(
                arguments,
                root / "runs" / "reuse",
                root / "runs",
            )
            arguments.runtime_cache_days = 0
            with self.assertRaisesRegex(ValueError, "positive"):
                validate_experiment_arguments(
                    arguments,
                    root / "runs" / "reuse",
                    root / "runs",
                )
        after = configuration_fingerprint(resolve_training_config(
            self.reuse["training"]
        ))
        self.assertEqual(after, before)


if __name__ == "__main__":
    unittest.main()

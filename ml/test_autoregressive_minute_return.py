from __future__ import annotations

import unittest
from datetime import date
from pathlib import Path
import tempfile

import torch
from torch import nn

from autoregressive_minute_return import (
    AutoregressiveMinuteReturn,
    autoregressive_return_path,
)
from next_return_dataset import ExampleShard, HISTORY_RETURN_COUNT
from train_autoregressive_minute_return import (
    aligned_minute_shards,
    direct_calendar_shards,
    minute_target,
    scalar_target,
    validate_plan,
)
from train_next_return_memorization import fixed_subset_shards


class LastTwoPlusOne(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values[:, -2:] + 1


class AutoregressiveMinuteReturnTest(unittest.TestCase):
    def test_direct_calendar_split_is_chronological_and_embargoed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day in range(1, 6):
                (root / f"2024-01-{day:02d}.json").touch()
            shards = direct_calendar_shards({
                "type": "direct-calendar-months-v1",
                "trainStart": "2024-01-02",
                "trainEnd": "2024-01-02",
                "validationStart": "2024-01-03",
                "validationEnd": "2024-01-03",
                "testStart": "2024-01-04",
                "testEnd": "2024-01-04",
            }, root)
            self.assertEqual(shards["train"][0].count, 86_280)
            self.assertEqual(shards["validation"][0].count, 86_280)
            self.assertEqual(shards["test"][0].count, 86_400)
            train = shards["train"][0]
            last_minute_decision = train.decision_time_start + (
                (train.count - 1) // 60 * 60_000
            )
            self.assertEqual(
                shards["validation"][0].decision_time_start
                - last_minute_decision,
                180_000,
            )

    def test_next_second_calendar_split_has_a_121_second_embargo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day in range(1, 6):
                (root / f"2024-01-{day:02d}.json").touch()
            shards = direct_calendar_shards({
                "type": "direct-calendar-months-v1",
                "trainStart": "2024-01-02",
                "trainEnd": "2024-01-02",
                "validationStart": "2024-01-03",
                "validationEnd": "2024-01-03",
                "testStart": "2024-01-04",
                "testEnd": "2024-01-04",
            }, root, horizon_seconds=1, decision_stride_seconds=1)
            self.assertEqual(shards["train"][0].count, 86_280)
            self.assertEqual(shards["validation"][0].count, 86_280)
            self.assertEqual(shards["test"][0].count, 86_400)
            train = shards["train"][0]
            last_train_decision = train.decision_time_start + (
                train.count - 1
            ) * 1_000
            self.assertEqual(
                shards["validation"][0].decision_time_start
                - last_train_decision,
                121_000,
            )

    def test_shards_are_advanced_to_a_minute_boundary(self) -> None:
        shard = ExampleShard("train", 30_000, 100, "2024-01-01", 7)
        aligned = aligned_minute_shards({"train": [shard]})["train"][0]
        self.assertEqual(aligned.decision_time_start, 60_000)
        self.assertEqual(aligned.row_offset, 37)
        self.assertEqual(aligned.count, 70)

    def test_minute_target_is_the_single_path_sum(self) -> None:
        paths = torch.arange(120, dtype=torch.float32).reshape(2, 60)
        torch.testing.assert_close(minute_target(paths), paths.sum(dim=1))

    def test_scalar_target_keeps_a_next_second_target(self) -> None:
        targets = torch.tensor([0.1, -0.2, 0.3])
        torch.testing.assert_close(scalar_target(targets, 1), targets)

    def test_direct_contract_accepts_two_glu_layers(self) -> None:
        validate_plan({
            "id": "two-layer",
            "datasetDir": "data/training/datasets/two-layer",
            "runDir": "data/training/runs/two-layer",
            "historyDir": "data/history",
            "sourceDatasetDir": "data/source",
            "initialCheckpoint": "data/checkpoint",
            "testExamples": 1,
            "architecture": {
                "contract": "direct-glu-to-single-return-v3",
                "horizonSeconds": 1,
                "widths": [512, 512],
            },
            "training": {
                "epochs": 1,
                "batchSize": 1,
                "evaluationBatchSize": 1,
                "earlyStoppingPatience": 1,
                "seed": 1,
                "device": "cpu",
                "mixedPrecision": "bfloat16",
                "learningRate": 1e-4,
                "gradientClip": 1,
            },
        })

    def test_memorization_subset_is_fixed_and_contiguous(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for day in range(1, 5):
                (root / f"2024-01-{day:02d}.json").touch()
            shards = fixed_subset_shards(
                root,
                date(2024, 1, 2),
                131_072,
            )
            self.assertEqual(tuple(shards), ("train", "validation", "test"))
            self.assertEqual(
                [value.count for value in shards["train"]],
                [86_400, 44_672],
            )
            self.assertEqual(
                [value.row_offset for value in shards["train"]], [0, 0]
            )
            self.assertEqual(
                shards["train"][0].decision_time_end + 1_000,
                shards["train"][1].decision_time_start,
            )

    def test_rollout_appends_each_prediction_to_the_next_input(self) -> None:
        features = torch.arange(
            HISTORY_RETURN_COUNT, dtype=torch.float32
        ).unsqueeze(0)
        path = autoregressive_return_path(
            LastTwoPlusOne(), features, horizon_seconds=6, chunk_seconds=2
        )
        torch.testing.assert_close(
            path, torch.tensor([[119.0, 120.0, 120.0, 121.0, 121.0, 122.0]])
        )

    def test_model_returns_one_minute_scalar_and_backpropagates(self) -> None:
        model = AutoregressiveMinuteReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.zeros(2),
            torch.ones(2),
            widths=(8,),
            dropout=0,
        )
        features = torch.randn(3, HISTORY_RETURN_COUNT)
        prediction = model(features)
        self.assertEqual(prediction.shape, (3,))
        prediction.square().mean().backward()
        self.assertTrue(all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        ))
        self.assertIsNotNone(model.core.output.weight.grad)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np
import torch

from next_return_dataset import (
    DAY_SECONDS,
    ExampleShard,
    HISTORY_RETURN_COUNT,
    SECOND_MS,
    daily_log_return_examples,
    example_span_ms,
    select_example_shards,
)
from next_return_sequence import (
    SUMMARY_NAMES,
    numpy_path_summaries,
    sequence_objective_components,
    sequence_objective_loss,
    torch_path_summaries,
)
from normalized_glu_next_return import NormalizedGluNextReturn
from train_normalized_glu_next_return import NextReturnDataset


def source_shard(split: str, start: int, count: int) -> dict:
    return {
        "split": split,
        "predictionTimeStart": start,
        "count": count,
        "date": "2024-01-01",
        "featureRowOffset": 0,
        "featureRowStride": 1,
    }


class NextReturnSequenceTest(unittest.TestCase):
    def test_batches_span_shard_boundaries_without_dropping_rows(self) -> None:
        class SyntheticDataset(NextReturnDataset):
            def __init__(self) -> None:
                self.horizon_return_count = 2
                self.shards = {
                    "train": [
                        ExampleShard("train", 0, 3, "2024-01-01", 0),
                        ExampleShard("train", 3_000, 4, "2024-01-02", 5),
                    ]
                }
                rows = np.arange(20, dtype=np.float32)
                self.synthetic_history = np.repeat(
                    rows[:, None], HISTORY_RETURN_COUNT, axis=1
                )
                self.synthetic_target = np.stack((rows, -rows), axis=1)

            def _component(self, day: str) -> tuple[np.ndarray, np.ndarray]:
                return self.synthetic_history, self.synthetic_target

        dataset = SyntheticDataset()
        batches = list(dataset.iter_batches(
            "train", 5, shuffle=False, seed=0
        ))
        self.assertEqual([len(batch[0]) for batch in batches], [5, 2])
        actual = torch.cat([batch[0][:, 0] for batch in batches]).numpy()
        np.testing.assert_array_equal(actual, [0, 1, 2, 5, 6, 7, 8])

    def test_daily_examples_return_fifteen_aligned_targets(self) -> None:
        step = 1e-6
        previous = np.exp(np.arange(-DAY_SECONDS, 0) * step)
        current = np.exp(np.arange(0, DAY_SECONDS) * step)
        following = np.exp(np.arange(DAY_SECONDS, 2 * DAY_SECONDS) * step)
        history, target = daily_log_return_examples(
            previous,
            current,
            following,
            horizon_return_count=15,
        )
        self.assertEqual(history.shape, (DAY_SECONDS, HISTORY_RETURN_COUNT))
        self.assertEqual(target.shape, (DAY_SECONDS, 15))
        np.testing.assert_allclose(history, step, rtol=0, atol=1e-8)
        np.testing.assert_allclose(target, step, rtol=0, atol=1e-8)

    def test_fifteen_second_split_uses_135_second_embargo(self) -> None:
        manifest = {"shards": [
            source_shard("train", 1_000 * SECOND_MS, 100),
            source_shard("validation", 1_400 * SECOND_MS, 100),
            source_shard("test", 2_000 * SECOND_MS, 300),
        ]}
        selected = select_example_shards(
            manifest,
            test_count=120,
            horizon_return_count=15,
        )
        self.assertTrue(selected["train"])
        self.assertTrue(selected["validation"])
        self.assertEqual(sum(value.count for value in selected["test"]), 120)
        self.assertEqual(example_span_ms(15), 135 * SECOND_MS)

    def test_test_tail_offset_selects_a_disjoint_earlier_window(self) -> None:
        manifest = {"shards": [
            source_shard("train", 1_000 * SECOND_MS, 100),
            source_shard("validation", 1_400 * SECOND_MS, 100),
            source_shard("test", 2_000 * SECOND_MS, 300),
        ]}
        selected = select_example_shards(
            manifest,
            test_count=120,
            test_tail_offset=50,
            horizon_return_count=15,
        )
        self.assertEqual(selected["test"][0].decision_time_start, 2_130 * SECOND_MS)
        self.assertEqual(selected["test"][0].decision_time_end, 2_249 * SECOND_MS)

    def test_numpy_and_torch_summaries_match(self) -> None:
        returns = np.array([
            [0.01, -0.02, 0.03],
            [-0.04, -0.01, 0.02],
        ], dtype=np.float32)
        actual = torch_path_summaries(torch.from_numpy(returns)).numpy()
        expected = numpy_path_summaries(returns)
        self.assertEqual(actual.shape, (2, len(SUMMARY_NAMES)))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
        np.testing.assert_allclose(
            actual[:, -1], np.expm1(returns.sum(axis=1)), rtol=1e-6
        )

    def test_sequence_objective_is_zero_for_an_exact_path(self) -> None:
        target = torch.randn(7, 15) * 1e-4
        candle, summaries = sequence_objective_components(
            target.clone(),
            target,
            target_std=torch.full((15,), 1e-4),
            summary_std=torch.ones(len(SUMMARY_NAMES)),
        )
        torch.testing.assert_close(candle, torch.zeros_like(candle))
        torch.testing.assert_close(summaries, torch.zeros_like(summaries))

    def test_cumulative_return_can_receive_extra_weight(self) -> None:
        target = torch.zeros(1, 15)
        prediction = target.clone()
        prediction[0, 0] = 1e-4
        common = {
            "target_std": torch.ones(15),
            "summary_std": torch.ones(len(SUMMARY_NAMES)),
            "candle_weight": 1.0,
            "summary_weight": 1.0,
        }
        equal = sequence_objective_loss(
            prediction,
            target,
            torch.ones(1),
            summary_metric_weights=torch.ones(len(SUMMARY_NAMES)),
            **common,
        )
        cumulative_heavy = sequence_objective_loss(
            prediction,
            target,
            torch.ones(1),
            summary_metric_weights=torch.tensor([1, 1, 1, 1, 2]),
            **common,
        )
        self.assertGreater(float(cumulative_heavy), float(equal))

    def test_cumulative_return_can_be_the_only_summary_loss(self) -> None:
        target = torch.zeros(2, 3)
        prediction = torch.tensor([
            [1e-4, -2e-4, 3e-4],
            [-1e-4, 1e-4, 2e-4],
        ])
        target_std = torch.ones(3)
        summary_std = torch.ones(len(SUMMARY_NAMES))
        candle, summaries = sequence_objective_components(
            prediction,
            target,
            target_std=target_std,
            summary_std=summary_std,
        )
        actual = sequence_objective_loss(
            prediction,
            target,
            torch.ones(2),
            target_std=target_std,
            summary_std=summary_std,
            summary_metric_weights=torch.tensor([0, 0, 0, 0, 1]),
            candle_weight=1,
            summary_weight=1,
        )
        expected = (candle + summaries[:, -1]).mean()
        torch.testing.assert_close(actual, expected)

    def test_summary_loss_can_be_disabled(self) -> None:
        target = torch.zeros(2, 3)
        prediction = torch.tensor([
            [1e-4, -2e-4, 3e-4],
            [-1e-4, 1e-4, 2e-4],
        ])
        target_std = torch.ones(3)
        summary_std = torch.ones(len(SUMMARY_NAMES))
        candle, _summaries = sequence_objective_components(
            prediction,
            target,
            target_std=target_std,
            summary_std=summary_std,
        )
        actual = sequence_objective_loss(
            prediction,
            target,
            torch.ones(2),
            target_std=target_std,
            summary_std=summary_std,
            summary_metric_weights=torch.tensor([0, 0, 0, 0, 1]),
            candle_weight=1,
            summary_weight=0,
        )
        torch.testing.assert_close(actual, candle.mean())

    def test_glu_outputs_the_configured_horizon(self) -> None:
        model = NormalizedGluNextReturn(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.linspace(-1e-6, 1e-6, 15),
            torch.full((15,), 1e-4),
            widths=(16,),
            dropout=0,
        )
        prediction = model(torch.randn(4, HISTORY_RETURN_COUNT))
        self.assertEqual(prediction.shape, (4, 15))
        torch.testing.assert_close(
            prediction,
            model.target_mean.unsqueeze(0).expand(4, -1),
        )
        prediction.square().mean().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()

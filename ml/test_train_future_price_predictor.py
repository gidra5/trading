from __future__ import annotations

import unittest
from pathlib import Path
import random
import tempfile

import numpy as np
import torch

from train_future_price_predictor import (
    HOUR_MS,
    PREDICTOR_EXAMPLE_SPAN_MS,
    compact_pair_rows,
    count_examples,
    restore_random_states,
    select_pair_segments,
    validate_predictor_split_disjointness,
)


def shard(
    split: str,
    start: int,
    count: int,
    *,
    date: str,
    row: int,
) -> dict:
    return {
        "split": split,
        "predictionTimeStart": start,
        "oracleTargetTimeStart": start - HOUR_MS,
        "count": count,
        "date": date,
        "featureRowOffset": row,
        "featureRowStride": 1,
        "oracleRowOffset": row - 3_600,
        "oracleRowStride": 1,
    }


class TrainFuturePricePredictorTest(unittest.TestCase):
    def test_rng_state_survives_save_reload_and_resume(self) -> None:
        random.seed(701)
        np.random.seed(702)
        torch.manual_seed(703)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(704)
        checkpoint = {
            "pythonRandomState": random.getstate(),
            "numpyRandomState": np.random.get_state(),
            "torchRandomState": torch.get_rng_state(),
            "cudaRandomState": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        }
        expected_python = random.random()
        expected_numpy = float(np.random.random())
        expected_torch = torch.rand(7)
        expected_cuda = (
            torch.rand(7, device="cuda").cpu()
            if torch.cuda.is_available()
            else None
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_file = Path(directory) / "resume.pt"
            torch.save(checkpoint, checkpoint_file)
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            reloaded = torch.load(
                checkpoint_file,
                map_location=device,
                weights_only=False,
            )
        random.seed(1)
        np.random.seed(2)
        torch.manual_seed(3)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(4)
        restore_random_states(reloaded, device)
        self.assertEqual(random.random(), expected_python)
        self.assertEqual(float(np.random.random()), expected_numpy)
        torch.testing.assert_close(torch.rand(7), expected_torch)
        if expected_cuda is not None:
            torch.testing.assert_close(
                torch.rand(7, device="cuda").cpu(),
                expected_cuda,
            )

    def test_compaction_preserves_exact_logical_measure(self) -> None:
        history, future, weights = compact_pair_rows(61, 3_661, 181)
        self.assertEqual(int(weights.sum()), 181)
        self.assertTrue(np.array_equal(np.diff(history), np.ones(3, dtype=np.int64)))
        self.assertTrue(np.array_equal(np.diff(future), np.ones(3, dtype=np.int64)))
        self.assertTrue(np.array_equal(future - history, np.full(4, 60)))

    def test_two_hour_purge_survives_a_fully_removed_bridge_shard(self) -> None:
        # The one-hour decoder selector may fully remove a short transition
        # shard. The predictor must carry its two-hour barrier into the next
        # shard of that split instead of accidentally admitting overlap.
        manifest = {
            "shards": [
                shard("train", 3 * HOUR_MS, 3_600, date="1970-01-01", row=10_800),
                shard("validation", 4 * HOUR_MS, 3_600, date="1970-01-01", row=14_400),
                shard("validation", 5 * HOUR_MS, 10_800, date="1970-01-01", row=18_000),
            ],
        }
        selected = select_pair_segments(
            manifest,
            cross_split_purge_ms=PREDICTOR_EXAMPLE_SPAN_MS,
        )
        validate_predictor_split_disjointness(selected)
        self.assertEqual(set(selected), {"train", "validation"})
        self.assertEqual(selected["validation"][0].prediction_time_start, 6 * HOUR_MS)
        self.assertEqual(count_examples(selected)["validation"], 7_200)

    def test_selector_never_exposes_a_test_split(self) -> None:
        manifest = {
            "shards": [
                shard("train", 3 * HOUR_MS, 3_600, date="1970-01-01", row=10_800),
                shard("validation", 7 * HOUR_MS, 3_600, date="1970-01-01", row=25_200),
                shard("test", 11 * HOUR_MS, 3_600, date="1970-01-01", row=39_600),
            ],
        }
        selected = select_pair_segments(
            manifest,
            cross_split_purge_ms=PREDICTOR_EXAMPLE_SPAN_MS,
        )
        self.assertEqual(set(selected), {"train", "validation"})
        self.assertNotIn("test", selected)


if __name__ == "__main__":
    unittest.main()

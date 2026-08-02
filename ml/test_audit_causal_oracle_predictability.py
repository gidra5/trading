from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from audit_causal_oracle_predictability import (
    ACTION_COUNT,
    DAY_ROWS,
    MINUTE_MS,
    load_lagged_target_split,
    minimum_safe_oracle_lag_minutes,
    select_convex_mixture_weight,
)
from train_joint_price_oracle import CausalSegment


def timestamp(day_value: str) -> int:
    return int(datetime.combine(
        date.fromisoformat(day_value),
        datetime.min.time(),
        timezone.utc,
    ).timestamp() * 1_000) + 999


def target_day(marker: int) -> np.ndarray:
    result = np.zeros((DAY_ROWS, ACTION_COUNT), dtype=np.float32)
    columns = (np.arange(DAY_ROWS) + marker) % ACTION_COUNT
    result[np.arange(DAY_ROWS), columns] = 1
    return result


class SafeLagTests(unittest.TestCase):
    def test_exact_hour_horizon_is_available_at_sixty_minute_lag(self) -> None:
        self.assertEqual(
            minimum_safe_oracle_lag_minutes(3_600, 1_000, 60_000),
            60,
        )
        self.assertEqual(
            minimum_safe_oracle_lag_minutes(3_601, 1_000, 60_000),
            61,
        )
        with self.assertRaisesRegex(ValueError, "positive"):
            minimum_safe_oracle_lag_minutes(0, 1_000, 60_000)

    def test_lag_alignment_crosses_only_into_past_target_days(self) -> None:
        paths = {
            day: Path(f"{day}.json")
            for day in ("2026-01-01", "2026-01-02", "2026-01-03")
        }
        arrays = {
            "2026-01-01": target_day(3),
            "2026-01-02": target_day(17),
            "2026-01-03": target_day(29),
        }
        opened: list[str] = []

        def fake_read(path: Path, *_args, **_kwargs):
            opened.append(path.stem)
            return None, arrays[path.stem]

        segment = CausalSegment(
            split="validation",
            prediction_time_start=timestamp("2026-01-02"),
            count=61,
            target_file=paths["2026-01-02"],
            target_row_offset=0,
            step_ms=MINUTE_MS,
        )
        with patch(
            "audit_causal_oracle_predictability.read_shard_array",
            side_effect=fake_read,
        ):
            result = load_lagged_target_split(
                [segment],
                paths,
                60,
                target_cache={},
            )

        self.assertTrue(bool(result.valid.all()))
        np.testing.assert_array_equal(
            result.probabilities[0],
            arrays["2026-01-01"][DAY_ROWS - 60],
        )
        np.testing.assert_array_equal(
            result.probabilities[-1],
            arrays["2026-01-02"][0],
        )
        self.assertEqual(set(opened), {"2026-01-01", "2026-01-02"})
        self.assertNotIn("2026-01-03", opened)
        self.assertEqual(
            {path.stem for path in result.opened_target_files},
            {"2026-01-01", "2026-01-02"},
        )

    def test_missing_past_reference_is_masked_and_future_is_never_opened(self) -> None:
        current = Path("2026-01-02.json")
        future = Path("2026-01-03.json")
        arrays = {
            current.stem: target_day(17),
            future.stem: target_day(29),
        }
        opened: list[str] = []

        def fake_read(path: Path, *_args, **_kwargs):
            opened.append(path.stem)
            return None, arrays[path.stem]

        segment = CausalSegment(
            split="train",
            prediction_time_start=timestamp(current.stem),
            count=61,
            target_file=current,
            target_row_offset=0,
            step_ms=MINUTE_MS,
        )
        with patch(
            "audit_causal_oracle_predictability.read_shard_array",
            side_effect=fake_read,
        ):
            result = load_lagged_target_split(
                [segment],
                {current.stem: current, future.stem: future},
                60,
                target_cache={},
            )

        self.assertEqual(int(result.valid.sum()), 1)
        self.assertFalse(bool(result.valid[:60].any()))
        self.assertEqual(opened, [current.stem])
        self.assertNotIn(future.stem, opened)

    def test_sub_hour_lag_is_rejected_before_any_payload_access(self) -> None:
        segment = CausalSegment(
            split="validation",
            prediction_time_start=timestamp("2026-01-02"),
            count=1,
            target_file=Path("2026-01-02.json"),
            target_row_offset=60,
            step_ms=MINUTE_MS,
        )
        with patch(
            "audit_causal_oracle_predictability.read_shard_array",
        ) as reader:
            with self.assertRaisesRegex(ValueError, "not causal"):
                load_lagged_target_split(
                    [segment],
                    {"2026-01-02": segment.target_file},
                    59,
                )
        reader.assert_not_called()


class MixtureSelectionTests(unittest.TestCase):
    def test_recovers_known_convex_mixture_weight(self) -> None:
        prior = np.asarray([0.5, 0.5], dtype=np.float64)
        lagged = np.asarray([
            [0.9, 0.1],
            [0.1, 0.9],
            [0.8, 0.2],
            [0.2, 0.8],
        ], dtype=np.float64)
        expected_weight = 0.75
        targets = (
            (1 - expected_weight) * prior[None, :]
            + expected_weight * lagged
        )
        weight, score = select_convex_mixture_weight(
            targets,
            lagged,
            prior,
        )
        self.assertAlmostEqual(weight, expected_weight, places=5)
        self.assertLess(score, 1e-10)


if __name__ == "__main__":
    unittest.main()

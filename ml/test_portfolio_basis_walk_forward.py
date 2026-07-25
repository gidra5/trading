import unittest

import numpy as np
import torch

from backtest_binance_multiscale_index import (
    apply_funding_cashflow,
    build_monthly_path,
    parse_day,
    precise,
    solve_relative_post_cost_value,
)
from portfolio_basis_walk_forward import (
    capped_proportional_weights_batch,
    select_basis_batch,
)


class PortfolioBasisWalkForwardTest(unittest.TestCase):
    def test_relative_rebalance_charges_buys_and_sells(self) -> None:
        current = np.array([1.0, 0.0])
        target = np.array([0.0, 1.0])
        costs = np.array([0.01, 0.01])
        relative = solve_relative_post_cost_value(current, target, costs)

        self.assertAlmostEqual(relative, 0.9801980198019802, places=12)
        traded = np.abs(target * relative - current).sum()
        self.assertAlmostEqual(1 - relative, 0.01 * traded, places=12)

    def test_relative_rebalance_can_leave_cash(self) -> None:
        relative = solve_relative_post_cost_value(
            np.array([0.0, 0.0]),
            np.array([0.25, 0.5]),
            np.array([0.0, 0.0]),
        )
        self.assertEqual(relative, 1)

    def test_positive_perpetual_funding_reduces_nav_before_rebalance(self) -> None:
        relative, post_funding_weights = apply_funding_cashflow(
            np.array([0.4, 0.6]),
            np.array([0.0, 0.0001]),
        )

        self.assertAlmostEqual(relative, 0.99994, places=12)
        self.assertAlmostEqual(post_funding_weights.sum(), 1 / 0.99994, places=12)

    def test_monthly_path_compounds_minute_returns(self) -> None:
        class Arguments:
            start_ms = parse_day("2026-01-31")
            finish_ms = parse_day("2026-02-02")

        count = (Arguments.finish_ms - Arguments.start_ms) // 60_000
        values = np.zeros(count)
        values[0] = 0.1
        values[24 * 60] = -0.1
        path = build_monthly_path(Arguments(), {"baseline": values})

        self.assertEqual([row["month"] for row in path], ["2026-01", "2026-02"])
        self.assertAlmostEqual(path[0]["baseline"]["endLevel"], 1_100)
        self.assertAlmostEqual(path[1]["baseline"]["endLevel"], 990)

    def test_precise_serialization_preserves_near_zero_index_levels(self) -> None:
        self.assertEqual(float(precise(4.374912711678468e-12)), 4.37491271167847e-12)

    def test_capped_weights_are_normalized_and_respect_cap(self) -> None:
        sizes = np.array(
            [[100, 10, 1, 1], [0, 0, 0, 0]],
            dtype=np.float64,
        )
        active = np.ones_like(sizes, dtype=bool)
        weights = capped_proportional_weights_batch(sizes, active, 0.5)
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-7)
        self.assertLessEqual(float(weights.max()), 0.5 + 1e-7)

    def test_amplitude_priority_uses_native_returns_before_normalization(self) -> None:
        shape = np.array([-2, -1, 0, 1, 2], dtype=np.float32)
        windows = np.array([[shape, shape * 10]], dtype=np.float32)
        result = select_basis_batch(
            windows,
            np.ones((1, 2), dtype=bool),
            anchor_index=None,
            min_size=1,
            max_size=1,
            target_median_r_squared=0,
            target_p10_r_squared=0,
            device="cpu",
        )

        self.assertEqual(result.selected[0, 0], 1)

    def test_batch_union_compaction_preserves_global_selection(self) -> None:
        random = np.random.default_rng(7)
        windows = random.normal(size=(3, 8, 40)).astype(np.float32)
        eligible = np.ones((3, 8), dtype=bool)
        eligible[:, [2, 6]] = False
        eligible[1, 4] = False
        windows[~eligible] = 0
        parameters = {
            "anchor_index": 0,
            "min_size": 3,
            "max_size": 5,
            "target_median_r_squared": 0.95,
            "target_p10_r_squared": 0.9,
            "device": "cpu",
        }
        full = select_basis_batch(windows, eligible, **parameters)
        union = np.flatnonzero(eligible.any(axis=0))
        compact_anchor = int(np.flatnonzero(union == 0)[0])
        compact = select_basis_batch(
            windows[:, union],
            eligible[:, union],
            **{**parameters, "anchor_index": compact_anchor},
        )
        compact_global = union[np.maximum(compact.selected, 0)]

        np.testing.assert_array_equal(compact.sizes, full.sizes)
        for row, size in enumerate(full.sizes):
            np.testing.assert_array_equal(
                compact_global[row, :size],
                full.selected[row, :size],
            )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_batch_selection_is_point_in_time_and_anchors_btc(self) -> None:
        random = np.random.default_rng(42)
        first = random.normal(size=(8, 40)).astype(np.float32)
        second = random.normal(size=(8, 40)).astype(np.float32)
        second[:, -1] = second[:, 0]
        windows = np.stack([first, second])
        eligible = np.ones((2, 8), dtype=bool)
        eligible[1, 3] = False

        result = select_basis_batch(
            windows,
            eligible,
            anchor_index=0,
            min_size=3,
            max_size=6,
            target_median_r_squared=0.5,
            target_p10_r_squared=0.2,
        )

        self.assertTrue(np.all(result.selected[:, 0] == 0))
        self.assertNotIn(3, result.selected[1, : result.sizes[1]])
        self.assertTrue(np.all(result.sizes >= 3))
        self.assertFalse(np.array_equal(result.selected[0], result.selected[1]))


if __name__ == "__main__":
    unittest.main()

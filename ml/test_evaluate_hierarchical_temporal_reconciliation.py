import math
import unittest

import numpy as np

from evaluate_hierarchical_temporal_reconciliation import (
    additive_bridge,
    coherent_bottom_residual_calibration,
    draw_fitted_ar1_paths,
    error_weighted_reconcile,
    fit_error_weighted_projection,
    hard_top_down_reconcile,
    hierarchy_matrix,
    online_calibrate_level,
    positive_allocation,
    rank_couple_hierarchy,
    select_projection_shrinkage,
)


class HierarchicalTemporalReconciliationTest(unittest.TestCase):
    def test_hierarchy_matrix_maps_four_hour_bottom_nodes(self) -> None:
        bottom = np.arange(1.0, 7.0)
        values = hierarchy_matrix() @ bottom
        np.testing.assert_allclose(
            values,
            np.array([21.0, 3.0, 7.0, 11.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        )

    def test_additive_bridge_meets_parent_target(self) -> None:
        bridged = additive_bridge(
            np.array([1.0, 2.0, 3.0]),
            12.0,
            np.array([1.0, 2.0, 3.0]),
        )
        self.assertAlmostEqual(float(np.sum(bridged)), 12.0)
        self.assertGreater(bridged[2] - 3.0, bridged[0] - 1.0)

    def test_positive_allocation_obeys_sum_and_caps(self) -> None:
        allocated = positive_allocation(
            np.array([10.0, 1.0, 1.0]),
            20.0,
            8.0,
        )
        self.assertAlmostEqual(float(np.sum(allocated)), 20.0)
        self.assertTrue(np.all(allocated >= 0.0))
        self.assertTrue(np.all(allocated <= 8.0))

    def test_hard_top_down_is_exactly_coherent(self) -> None:
        rng = np.random.default_rng(3)
        raw = rng.normal(size=(4, 10, 8))
        reconciled = hard_top_down_reconcile(
            raw,
            "periodReturnBps",
            np.eye(10),
        )
        np.testing.assert_allclose(reconciled[:, 0], np.sum(reconciled[:, 1:4], axis=1))
        np.testing.assert_allclose(reconciled[:, 0], np.sum(reconciled[:, 4:10], axis=1))
        for block in range(3):
            np.testing.assert_allclose(
                reconciled[:, 1 + block],
                np.sum(reconciled[:, 4 + 2 * block:6 + 2 * block], axis=1),
            )

    def test_error_weighted_projection_is_exactly_coherent(self) -> None:
        rng = np.random.default_rng(5)
        bottom = rng.normal(size=(100, 6))
        actual = bottom @ hierarchy_matrix().T
        raw = actual[:, :, None] + rng.normal(size=(100, 10, 16))
        projection, _ = fit_error_weighted_projection(raw, actual, 60)
        reconciled, clips = error_weighted_reconcile(
            raw,
            "periodReturnBps",
            projection,
        )
        self.assertEqual(clips, 0)
        np.testing.assert_allclose(reconciled[:, 0], np.sum(reconciled[:, 4:10], axis=1))
        np.testing.assert_allclose(reconciled[:, 1:4], np.stack([
            np.sum(reconciled[:, 4:6], axis=1),
            np.sum(reconciled[:, 6:8], axis=1),
            np.sum(reconciled[:, 8:10], axis=1),
        ], axis=1))

    def test_sequential_ar_draw_depends_on_previous_generated_state(self) -> None:
        fit = {"mean": 0.0, "phi": 0.5, "last": 2.0, "scale": 1.0, "degrees": 8.0}
        draws = np.zeros((1, 3))
        values = draw_fitted_ar1_paths(fit, draws)
        np.testing.assert_allclose(values, np.array([[1.0, 0.5, 0.25]]), atol=1e-10)

    def test_online_calibration_never_reads_current_or_future_outcomes(self) -> None:
        rng = np.random.default_rng(13)
        actual = rng.normal(size=(12, 1))
        ensemble = rng.normal(size=(12, 1, 16))
        baseline = online_calibrate_level(
            "1d",
            "periodReturnBps",
            actual,
            ensemble,
            start=5,
            history_days=5,
            method="residual",
            output_members=16,
        )
        changed = actual.copy()
        changed[8:] += 1_000.0
        perturbed = online_calibrate_level(
            "1d",
            "periodReturnBps",
            changed,
            ensemble,
            start=5,
            history_days=5,
            method="residual",
            output_members=16,
        )
        np.testing.assert_allclose(baseline[:9], perturbed[:9])
        self.assertFalse(np.allclose(baseline[9:], perturbed[9:]))

    def test_shrinkage_selection_returns_a_supported_value(self) -> None:
        rng = np.random.default_rng(21)
        bottom = rng.lognormal(size=(80, 6))
        actual = bottom @ hierarchy_matrix().T
        raw = actual[:, :, None] * np.exp(rng.normal(0.0, 0.2, (80, 10, 16)))
        selected, scores = select_projection_shrinkage(
            raw,
            actual,
            feature="oneSecondRealizedVarianceBpsSquared",
            fit_start=10,
            fit_end=45,
            selection_end=70,
        )
        self.assertIn(str(selected), scores)
        self.assertEqual(len(scores), 5)

    def test_rank_coupling_preserves_node_marginals_and_aligns_daily_total(self) -> None:
        raw = np.zeros((1, 10, 4), dtype=np.float64)
        raw[0, 0] = np.array([4.0, 1.0, 3.0, 2.0])
        raw[0, 1:4] = np.array([
            [0.1, 1.0, 0.5, 0.8],
            [0.1, 1.0, 0.5, 0.8],
            [0.1, 1.0, 0.5, 0.8],
        ])
        raw[0, 4:10] = np.arange(24.0).reshape(6, 4)
        coupled = rank_couple_hierarchy(raw)
        for node in range(10):
            np.testing.assert_allclose(
                np.sort(coupled[0, node]),
                np.sort(raw[0, node]),
            )
        np.testing.assert_array_equal(
            np.argsort(np.argsort(coupled[0, 0])),
            np.argsort(np.argsort(np.sum(coupled[0, 1:4], axis=0))),
        )

    def test_coherent_residual_calibration_is_causal_and_coherent(self) -> None:
        rng = np.random.default_rng(31)
        structure = hierarchy_matrix()
        actual_bottom = rng.normal(size=(12, 6))
        actual = actual_bottom @ structure.T
        ensemble_bottom = rng.normal(size=(12, 6, 16))
        ensemble = np.einsum("nb,dbp->dnp", structure, ensemble_bottom)
        baseline = coherent_bottom_residual_calibration(
            actual,
            ensemble,
            feature="periodReturnBps",
            start=5,
            history_days=5,
        )
        changed = actual.copy()
        changed[8:] += np.arange(10, dtype=np.float64)[None, :] * 1_000.0
        perturbed = coherent_bottom_residual_calibration(
            changed,
            ensemble,
            feature="periodReturnBps",
            start=5,
            history_days=5,
        )
        np.testing.assert_allclose(baseline[:9], perturbed[:9])
        self.assertFalse(np.allclose(baseline[9:], perturbed[9:]))
        np.testing.assert_allclose(baseline[:, 0], np.sum(baseline[:, 4:10], axis=1))
        np.testing.assert_allclose(baseline[5:, 0], ensemble[5:, 0])

    def test_variance_residual_calibration_bounds_multiplicative_correction(self) -> None:
        structure = hierarchy_matrix()
        actual_bottom = np.full((12, 6), 1e12)
        actual = actual_bottom @ structure.T
        ensemble_bottom = np.full((12, 6, 16), 1.0)
        ensemble = np.einsum("nb,dbp->dnp", structure, ensemble_bottom)
        calibrated = coherent_bottom_residual_calibration(
            actual,
            ensemble,
            feature="oneSecondRealizedVarianceBpsSquared",
            start=5,
            history_days=5,
        )
        self.assertLessEqual(float(np.max(calibrated[5:, 4:10])), math.exp(2.5) + 1e-9)


if __name__ == "__main__":
    unittest.main()

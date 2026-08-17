import json
from pathlib import Path
import unittest

import numpy as np

from ml.evaluate_full_hierarchical_one_second_process import (
    FEATURES,
    LEVEL_MINUTES,
    MINUTES_PER_DAY,
    actual_hierarchy,
    aggregate_leaves,
    chronological_folds,
    deserialize_fit,
    feasible_minute_targets,
    gaussian_member_ranks,
    hierarchy_from_leaves,
    joint_couple_leaf_features,
    match_leaf_marginals,
    positive_allocate_rows,
    rank_couple_parent,
    reconcile_immediate_children,
    reconcile_leaf_groups,
    restore_member_ranks,
    round_activity_counts,
    zero_gap_counts,
)


class FullHierarchyHelpersTest(unittest.TestCase):
    def test_actual_hierarchy_is_direct_sum_of_minute_leaves(self) -> None:
        minutes = np.arange(2 * MINUTES_PER_DAY, dtype=np.float64)
        hierarchy = actual_hierarchy(minutes, minutes + 1.0, np.ones_like(minutes))
        for level, width in LEVEL_MINUTES.items():
            expected = minutes.reshape(2, MINUTES_PER_DAY // width, width).sum(axis=2)
            np.testing.assert_allclose(hierarchy[level]["periodReturnBps"], expected)

    def test_chronological_folds_cover_interval_without_overlap(self) -> None:
        folds = chronological_folds(183, 274)
        self.assertEqual(folds[0][0], 183)
        self.assertEqual(folds[-1][1], 274)
        self.assertTrue(all(left[1] == right[0] for left, right in zip(folds, folds[1:])))

    def test_rank_coupling_preserves_parent_marginals_and_matches_order(self) -> None:
        parent = np.asarray([[[30.0, 10.0, 40.0, 20.0]]])
        children = np.asarray([[[4.0, 1.0, 3.0, 2.0]]])
        coupled = rank_couple_parent(parent, children)
        np.testing.assert_allclose(np.sort(coupled, axis=2), np.sort(parent, axis=2))
        self.assertEqual(
            np.argsort(coupled[0, 0]).tolist(),
            np.argsort(children[0, 0]).tolist(),
        )

    def test_calibrated_quantiles_restore_raw_member_ranks(self) -> None:
        raw = np.asarray([[3.0, 1.0, 4.0, 2.0], [0.0, 4.0, 2.0, 3.0]])
        calibrated = np.asarray([[10.0, 20.0, 30.0, 40.0], [11.0, 21.0, 31.0, 41.0]])
        restored = restore_member_ranks(raw, calibrated)
        np.testing.assert_allclose(np.sort(restored, axis=1), calibrated)
        np.testing.assert_array_equal(
            np.argsort(restored, axis=1),
            np.argsort(raw, axis=1),
        )

    def test_leaf_marginal_matching_preserves_values_and_dependence_ranks(self) -> None:
        dependence = np.asarray([[[3.0, 1.0, 2.0], [1.0, 3.0, 2.0]]])
        marginal = np.asarray([[[10.0, 30.0, 20.0], [60.0, 40.0, 50.0]]])
        matched = match_leaf_marginals(dependence, marginal)
        np.testing.assert_allclose(
            np.sort(matched, axis=2),
            np.sort(marginal, axis=2),
        )
        np.testing.assert_array_equal(
            np.argsort(matched, axis=2),
            np.argsort(dependence, axis=2),
        )

    def test_gaussian_member_ranks_are_rowwise_normal_scores(self) -> None:
        values = np.asarray([[[3.0, 1.0, 4.0, 2.0]]])
        scores = gaussian_member_ranks(values)
        np.testing.assert_array_equal(
            np.argsort(scores, axis=2),
            np.argsort(values, axis=2),
        )
        self.assertAlmostEqual(float(np.mean(scores)), 0.0, places=12)

    def test_joint_feature_coupling_preserves_each_feature_path_marginal(self) -> None:
        rng = np.random.default_rng(52)
        leaves = {
            "periodReturnBps": rng.normal(size=(2, MINUTES_PER_DAY, 4)).astype(np.float32),
            "oneSecondRealizedVarianceBpsSquared": np.exp(
                rng.normal(size=(2, MINUTES_PER_DAY, 4))
            ).astype(np.float32),
            "activeSeconds": rng.uniform(
                1.0, 59.0, size=(2, MINUTES_PER_DAY, 4)
            ).astype(np.float32),
        }
        coupled, details = joint_couple_leaf_features(leaves)
        np.testing.assert_allclose(coupled["periodReturnBps"], leaves["periodReturnBps"])
        for feature in (
            "oneSecondRealizedVarianceBpsSquared",
            "activeSeconds",
        ):
            before = np.sort(leaves[feature].sum(axis=1), axis=1)
            after = np.sort(coupled[feature].sum(axis=1), axis=1)
            np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-5)
        self.assertLessEqual(
            details["meanFeasibilityCostAfter"],
            details["meanFeasibilityCostBefore"] + 1e-12,
        )

    def test_positive_allocation_hits_targets_without_exceeding_cap(self) -> None:
        values = np.asarray([[1.0, 2.0, 100.0], [4.0, 3.0, 2.0]])
        targets = np.asarray([120.0, 7.5])
        result = positive_allocate_rows(values, targets, 60.0)
        np.testing.assert_allclose(result.sum(axis=1), targets, atol=1e-9)
        self.assertLessEqual(float(np.max(result)), 60.0)
        self.assertGreaterEqual(float(np.min(result)), 0.0)

    def test_reconcile_leaf_groups_is_exact_for_every_feature(self) -> None:
        rng = np.random.default_rng(3)
        leaves = rng.normal(size=(2, MINUTES_PER_DAY, 4))
        for feature in FEATURES:
            source = leaves
            if feature != "periodReturnBps":
                source = np.abs(leaves) + 0.1
            if feature == "activeSeconds":
                source = np.clip(source, 0.0, 60.0)
            target = aggregate_leaves(source, "15m")
            if feature == "periodReturnBps":
                target += 0.5
            else:
                target *= 0.8
            reconciled = reconcile_leaf_groups(
                source,
                target,
                level="15m",
                feature=feature,
                return_weights=np.abs(leaves) + 0.1,
            )
            np.testing.assert_allclose(
                aggregate_leaves(reconciled, "15m"),
                target,
                atol=1e-8,
            )
            if feature == "activeSeconds":
                self.assertLessEqual(float(np.max(reconciled)), 60.0)
                self.assertGreaterEqual(float(np.min(reconciled)), 0.0)

    def test_hierarchy_from_leaves_is_coherent(self) -> None:
        rng = np.random.default_rng(4)
        leaves = rng.normal(size=(3, MINUTES_PER_DAY, 2))
        hierarchy = hierarchy_from_leaves(leaves)
        np.testing.assert_allclose(
            hierarchy["1d"][:, 0],
            hierarchy["8h"].sum(axis=1),
            atol=1e-10,
        )
        np.testing.assert_allclose(
            hierarchy["8h"].reshape(3, 3, 1, 2),
            hierarchy["4h"].reshape(3, 3, 2, 2).sum(axis=2, keepdims=True),
            atol=1e-10,
        )

    def test_parent_correction_preserves_direct_child_inner_contrasts(self) -> None:
        rng = np.random.default_rng(41)
        leaves = rng.normal(size=(1, MINUTES_PER_DAY, 3))
        before = leaves.reshape(1, 48, 30, 3)
        before_contrast = before - before.mean(axis=2, keepdims=True)
        target = aggregate_leaves(leaves, "1h") + 6.0
        adjusted = reconcile_immediate_children(
            leaves,
            target,
            parent_level="1h",
            feature="periodReturnBps",
        )
        np.testing.assert_allclose(aggregate_leaves(adjusted, "1h"), target, atol=1e-9)
        after = adjusted.reshape(1, 48, 30, 3)
        after_contrast = after - after.mean(axis=2, keepdims=True)
        np.testing.assert_allclose(after_contrast, before_contrast, atol=1e-9)

    def test_activity_rounding_preserves_each_fifteen_minute_total(self) -> None:
        rng = np.random.default_rng(5)
        values = rng.uniform(0.0, 60.0, MINUTES_PER_DAY)
        result = round_activity_counts(values)
        self.assertTrue(np.all(result <= 60))
        expected = np.rint(values.reshape(-1, 15).sum(axis=1)).astype(np.int64)
        actual = result.reshape(-1, 15).sum(axis=1).astype(np.int64)
        np.testing.assert_array_equal(actual, expected)

    def test_feasible_targets_enforce_zero_and_single_activity_constraints(self) -> None:
        returns = np.linspace(-1.0, 1.0, 15)
        variance = np.full(15, 0.5)
        counts = np.asarray([0, 1] + [4] * 13, dtype=np.uint8)
        projected, q, details = feasible_minute_targets(returns, variance, counts)
        self.assertEqual(projected[0], 0.0)
        self.assertEqual(q[0], 0.0)
        self.assertAlmostEqual(abs(projected[1]), np.sqrt(q[1]))
        self.assertLess(details["maximumBlockReturnTargetErrorBps"], 1e-8)
        bounds = np.sqrt(counts.astype(np.float64) * q) + 1e-9
        self.assertTrue(np.all(np.abs(projected) <= bounds))

    def test_zero_gap_counts_censors_the_tail(self) -> None:
        values = np.asarray([1.0, 0.0, 0.0, 2.0, 0.0, 3.0] + [0.0] * 8)
        counts = zero_gap_counts(values, cap=5)
        self.assertEqual(int(counts[1]), 1)
        self.assertEqual(int(counts[2]), 1)
        self.assertEqual(int(counts[5]), 1)

    def test_serialized_repository_fit_can_be_deserialized(self) -> None:
        repo = Path(__file__).resolve().parents[1]
        report = json.loads(
            (repo / "data/benchmarks/parametric-one-second-process.json").read_text(
                encoding="utf-8"
            )
        )
        fitted = deserialize_fit(report["fittedParameters"])
        self.assertGreater(fitted.variance_quantile_knots.size, 100)
        self.assertEqual(fitted.activity_count_probabilities.size, 61)
        self.assertEqual(fitted.magnitude_mixture.component_weights.shape[-1], 3)


if __name__ == "__main__":
    unittest.main()

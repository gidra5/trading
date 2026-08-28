from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from global_feature_basis_search import (
    BasisCandidate,
    FeatureGroup,
    active_group_kkt_residuals,
    additive_logits,
    choose_lexicographic_basis,
    fit_group_lasso_additive,
    fit_multitask_group_lasso_additive,
    fit_multitask_group_lasso_active_set_torch,
    fit_multitask_group_lasso_torch,
    multitask_residual,
    multitask_probabilities,
    scan_excluded_kkt,
    scan_quantized_batches_kkt,
    scan_quantized_batches_multitask_kkt,
    scan_quantized_batches_residual,
    softmax,
)
from search_global_btc_per_horizon import (
    align_warm_coefficients,
    atomic_write_result_pair,
    choose_fraction,
)


class GlobalFeatureBasisSearchTest(unittest.TestCase):
    def test_atomic_result_pair_commits_model_before_json_marker(self) -> None:
        with TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "search.json"
            model = root / "model.npz"
            atomic_write_result_pair(
                output,
                {"status": "complete"},
                model,
                {"values": np.asarray([1.0, 2.0], dtype=np.float32)},
            )
            self.assertEqual(
                output.read_text(encoding="utf-8").strip(),
                '{\n  "status": "complete"\n}',
            )
            with np.load(model) as saved:
                np.testing.assert_array_equal(
                    saved["values"], np.asarray([1.0, 2.0], dtype=np.float32)
                )
            self.assertEqual(list(root.glob(".*.tmp*")), [])

    def test_warm_coefficients_align_into_expanded_working_set(self) -> None:
        warm = np.arange(2 * 4 * 3, dtype=np.float64).reshape(2, 4, 3)
        aligned = align_warm_coefficients(
            np.asarray(["old-b", "new", "old-a"]),
            np.asarray([3, 4, 2]),
            np.asarray(["old-a", "old-b"]),
            np.asarray([2, 3]),
            warm,
        )
        np.testing.assert_array_equal(aligned[0], warm[1, :2])
        np.testing.assert_array_equal(aligned[1], np.zeros((3, 3)))
        np.testing.assert_array_equal(aligned[2], warm[0, :1])

    def test_warm_alignment_rejects_dropped_coordinate_or_changed_arity(self) -> None:
        coefficients = np.zeros((1, 3, 2), dtype=np.float64)
        with self.assertRaises(ValueError):
            align_warm_coefficients(
                np.asarray(["new"]), np.asarray([2]),
                np.asarray(["old"]), np.asarray([2]), coefficients,
            )
        with self.assertRaises(ValueError):
            align_warm_coefficients(
                np.asarray(["old"]), np.asarray([3]),
                np.asarray(["old"]), np.asarray([2]), coefficients,
            )

    def test_paired_one_standard_error_rule_prefers_compact_equivalent_path(self) -> None:
        fold_bits = {
            1.0: (0.0, 0.0, 0.0),
            0.5: (0.028, 0.037, 0.044),
            0.25: (0.030, 0.040, 0.050),
        }
        folds = []
        for fold_index in range(3):
            path = []
            for fraction in (1.0, 0.5, 0.25):
                support_ids = [] if fraction == 1.0 else ["compact"]
                if fraction == 0.25:
                    support_ids.append("wide")
                path.append({
                    "lambdaFraction": fraction,
                    "validationBits": fold_bits[fraction][fold_index],
                    "supportSize": len(support_ids),
                    "supportIds": support_ids,
                    "converged": True,
                })
            folds.append({"path": path})
        policies = {
            feature_id: {"availability": 1.0, "acquisitionCost": 0}
            for feature_id in ("compact", "wide")
        }
        fixed, _ = choose_fraction(folds, [1.0, 0.5, 0.25], policies, "fixed")
        compact, summaries = choose_fraction(
            folds, [1.0, 0.5, 0.25], policies, "paired-one-se"
        )
        self.assertEqual(fixed, 0.25)
        self.assertEqual(compact, 0.5)
        selected = next(row for row in summaries if row["lambdaFraction"] == compact)
        self.assertTrue(selected["oneStandardErrorEligible"])

    def test_equivalent_path_does_not_reward_extra_always_available_inputs(self) -> None:
        folds = []
        dense_bits = (0.0190, 0.0201, 0.0212)
        for fold_index in range(3):
            folds.append({"path": [
                {
                    "lambdaFraction": 1.0, "validationBits": 0.0,
                    "supportSize": 0, "supportIds": [], "converged": True,
                },
                {
                    "lambdaFraction": 0.5, "validationBits": 0.02,
                    "supportSize": 1, "supportIds": ["bottleneck"], "converged": True,
                },
                {
                    "lambdaFraction": 0.25,
                    "validationBits": dense_bits[fold_index],
                    "supportSize": 3,
                    "supportIds": ["bottleneck", "easy-a", "easy-b"],
                    "converged": True,
                },
            ]})
        policies = {
            "bottleneck": {"availability": 0.9, "acquisitionCost": 1},
            "easy-a": {"availability": 1.0, "acquisitionCost": 0},
            "easy-b": {"availability": 1.0, "acquisitionCost": 0},
        }
        selected, _ = choose_fraction(
            folds, [1.0, 0.5, 0.25], policies, "paired-one-se"
        )
        self.assertEqual(selected, 0.5)

    def test_quality_precedes_cost_and_size(self) -> None:
        best = BasisCandidate("best", 0.0200, (0.0200, 0.0200), 0.5, 0.5, 5, 8)
        equivalent = BasisCandidate("easy", 0.0195, (0.0195, 0.0195), 1.0, 1.0, 0, 2)
        too_weak = BasisCandidate("weak", 0.0189, (0.0189, 0.0189), 1.0, 1.0, 0, 1)
        self.assertEqual(choose_lexicographic_basis([best, equivalent, too_weak]).id, "easy")

    def test_fold_quality_must_also_be_equivalent(self) -> None:
        stable = BasisCandidate("stable", 0.0200, (0.0200, 0.0200), 0.9, 0.9, 1, 3)
        unstable = BasisCandidate("unstable", 0.0201, (0.0230, 0.0172), 1.0, 1.0, 0, 1)
        self.assertEqual(choose_lexicographic_basis([stable, unstable]).id, "stable")

    def test_group_lasso_finds_predictive_feature_and_kkt_rejects_noise(self) -> None:
        random = np.random.default_rng(71391)
        rows = 4_000
        predictive = random.integers(0, 4, rows)
        labels = ((predictive >= 2) ^ (random.random(rows) < 0.08)).astype(np.int64)
        noise = random.integers(0, 4, rows)
        active = predictive[:, None]
        group = FeatureGroup("predictive", 4, 1.0)
        fit = fit_group_lasso_additive(active, labels, [group], 2, 0.01)
        self.assertTrue(fit.converged)
        self.assertGreater(np.linalg.norm(fit.coefficients[0]), 0.1)

        noise_group = FeatureGroup("noise", 4, 1.0)

        def provider():
            yield [noise_group], noise[:, None]

        scan = scan_excluded_kkt(labels, active, fit, provider, 0.01, {"predictive"})
        self.assertEqual(scan.scanned_groups, 1)
        self.assertEqual(scan.violating_groups, ())

    def test_active_group_stationarity_is_small_for_converged_fit(self) -> None:
        random = np.random.default_rng(812)
        states = random.integers(0, 4, (2_000, 1), dtype=np.uint8)
        labels = ((states[:, 0] >= 2) ^ (random.random(2_000) < 0.1)).astype(np.int64)
        group = FeatureGroup("active", 4, 1.0)
        fit = fit_multitask_group_lasso_additive(
            states, labels[:, None], [group], (2,), 0.01,
            max_iterations=4_000, tolerance=1e-10,
        )
        self.assertTrue(fit.converged)
        residual = multitask_residual(states, labels[:, None], fit)
        stationarity = active_group_kkt_residuals(
            states, residual, [group], fit.coefficients, 0.01, device="cpu"
        )
        self.assertLess(stationarity[0], 1e-4)
        torch_fit = fit_multitask_group_lasso_torch(
            states, labels[:, None], [group], (2,), 0.01,
            max_iterations=5_000, tolerance=1e-6,
            stationarity_tolerance=1e-4, device="cpu",
        )
        self.assertTrue(torch_fit.converged)
        self.assertLessEqual(torch_fit.stationarity_maximum, 1e-4)
        torch_residual = multitask_residual(states, labels[:, None], torch_fit)
        torch_stationarity = active_group_kkt_residuals(
            states, torch_residual, [group], torch_fit.coefficients,
            0.01, device="cpu",
        )
        self.assertLessEqual(torch_stationarity[0], 1.1e-4)

    def test_kkt_scan_detects_omitted_predictor(self) -> None:
        random = np.random.default_rng(42)
        rows = 3_000
        predictor = random.integers(0, 4, rows)
        labels = (predictor >= 2).astype(np.int64)
        intercept_state = np.zeros((rows, 0), dtype=np.int64)
        fit = fit_group_lasso_additive(intercept_state, labels, [], 2, 0.01)
        candidate = FeatureGroup("omitted", 4, 1.0)

        def provider():
            yield [candidate], predictor[:, None]

        scan = scan_excluded_kkt(labels, intercept_state, fit, provider, 0.01, set())
        self.assertEqual(scan.violating_groups[0][0], "omitted")
        self.assertGreater(scan.maximum_violation, 0)

    def test_batched_kkt_matches_scalar_scan(self) -> None:
        random = np.random.default_rng(7)
        rows = 1_000
        labels = random.integers(0, 3, rows)
        candidates = random.integers(0, 4, (rows, 5), dtype=np.uint8)
        active = np.zeros((rows, 0), dtype=np.uint8)
        fit = fit_group_lasso_additive(active, labels, [], 3, 0.002)
        groups = [FeatureGroup(f"candidate-{index}", 4, 1.0) for index in range(5)]

        def scalar_provider():
            yield groups, candidates

        class Batch:
            pass

        def batched_provider():
            batch = Batch()
            batch.groups = groups
            batch.states = candidates
            yield batch

        scalar = scan_excluded_kkt(labels, active, fit, scalar_provider, 0.002, set(), add_limit=5)
        batched = scan_quantized_batches_kkt(
            labels, active, fit, batched_provider, 0.002, set(), add_limit=5, device="cpu"
        )
        self.assertEqual([row[0] for row in scalar.violating_groups], [row[0] for row in batched.violating_groups])
        self.assertAlmostEqual(scalar.maximum_violation, batched.maximum_violation, places=6)

        probabilities = np.broadcast_to(softmax(fit.intercept[None, :]), (rows, 3)).copy()
        residual = probabilities
        residual[np.arange(rows), labels] -= 1
        direct = scan_quantized_batches_residual(
            residual, batched_provider, 0.002, set(), add_limit=5, device="cpu"
        )
        self.assertAlmostEqual(batched.maximum_violation, direct.maximum_violation, places=6)

    def test_multitask_fit_shares_a_predictive_group_across_softmax_heads(self) -> None:
        random = np.random.default_rng(181)
        rows = 5_000
        predictor = random.integers(0, 4, rows, dtype=np.uint8)
        labels = np.column_stack((
            ((predictor >= 2) ^ (random.random(rows) < 0.08)).astype(np.int64),
            ((predictor == 1) | (predictor == 3)).astype(np.int64),
        ))
        group = FeatureGroup("shared-predictor", 4, 1.0)
        fit = fit_multitask_group_lasso_additive(
            predictor[:, None], labels, [group], (2, 2), 0.01
        )
        self.assertTrue(fit.converged)
        self.assertEqual(fit.coefficients[0].shape, (3, 4))
        self.assertGreater(np.linalg.norm(fit.coefficients[0]), 0.1)

        noise = random.integers(0, 4, rows, dtype=np.uint8)
        noise_group = FeatureGroup("noise", 4, 1.0)

        class Batch:
            pass

        def batches():
            batch = Batch()
            batch.groups = [noise_group]
            batch.states = noise[:, None]
            yield batch

        residual = multitask_residual(predictor[:, None], labels, fit)
        scan = scan_quantized_batches_residual(
            residual, batches, 0.01, {group.id}, device="cpu"
        )
        self.assertEqual(scan.scanned_groups, 1)
        self.assertEqual(scan.violating_groups, ())

    def test_torch_multitask_solver_matches_support_on_cpu(self) -> None:
        random = np.random.default_rng(991)
        rows = 2_000
        predictor = random.integers(0, 4, rows, dtype=np.uint8)
        noise = random.integers(0, 4, rows, dtype=np.uint8)
        labels = np.column_stack(((predictor >= 2).astype(np.int64), (predictor == 3).astype(np.int64)))
        groups = [FeatureGroup("predictive", 4, 1.0), FeatureGroup("noise", 4, 1.0)]
        fit = fit_multitask_group_lasso_torch(
            np.column_stack((predictor, noise)),
            labels,
            groups,
            (2, 2),
            0.015,
            device="cpu",
            tolerance=1e-6,
        )
        self.assertTrue(fit.converged)
        self.assertGreater(np.linalg.norm(fit.coefficients[0]), 0.1)
        self.assertLess(np.linalg.norm(fit.coefficients[1]), 1e-5)

    def test_active_set_torch_solver_matches_full_convex_solution(self) -> None:
        random = np.random.default_rng(602)
        rows = 2_500
        predictor = random.integers(0, 4, rows, dtype=np.uint8)
        weak = random.integers(0, 4, rows, dtype=np.uint8)
        noise = random.integers(0, 4, (rows, 5), dtype=np.uint8)
        labels = ((predictor >= 2) ^ ((weak == 3) & (random.random(rows) < 0.55))).astype(np.int64)
        states = np.column_stack((predictor, weak, noise))
        groups = [FeatureGroup(f"group-{index}", 4, 1.0) for index in range(states.shape[1])]
        regularization = 0.012
        full = fit_multitask_group_lasso_torch(
            states, labels[:, None], groups, (2,), regularization,
            device="cpu", tolerance=1e-8, stationarity_tolerance=2e-4,
            max_iterations=5_000,
        )
        progress = []
        active = fit_multitask_group_lasso_active_set_torch(
            states, labels[:, None], groups, (2,), regularization,
            device="cpu", tolerance=1e-8, stationarity_tolerance=2e-4,
            kkt_tolerance=2e-4, add_limit=2, scan_chunk_size=3,
            max_iterations=5_000,
            progress_callback=progress.append,
        )
        self.assertTrue(full.converged)
        self.assertTrue(active.converged)
        full_probability = multitask_probabilities(
            additive_logits(states, full.intercept, full.coefficients), (2,)
        )
        active_probability = multitask_probabilities(
            additive_logits(states, active.intercept, active.coefficients), (2,)
        )
        np.testing.assert_allclose(active_probability, full_probability, atol=2e-3)
        self.assertEqual(
            [np.linalg.norm(value) > 1e-5 for value in active.coefficients],
            [np.linalg.norm(value) > 1e-5 for value in full.coefficients],
        )
        self.assertGreaterEqual(len(progress), 1)
        self.assertEqual(progress[-1]["additions"], 0)
        self.assertIn("omittedMaximumViolation", progress[-1])

    def test_task_specific_group_cannot_modify_other_softmax_head(self) -> None:
        random = np.random.default_rng(15)
        rows = 1_500
        predictor = random.integers(0, 4, rows, dtype=np.uint8)
        labels = np.column_stack(((predictor >= 2).astype(np.int64), random.integers(0, 2, rows)))
        group = FeatureGroup("task-zero-only", 4, 0.0, (0, 1))
        fit = fit_multitask_group_lasso_torch(
            predictor[:, None], labels, [group], (2, 2), 0.0, device="cpu", tolerance=1e-6
        )
        self.assertGreater(np.linalg.norm(fit.coefficients[0][:, :2]), 0.1)
        self.assertEqual(float(np.linalg.norm(fit.coefficients[0][:, 2:])), 0.0)

    def test_fixed_offset_nests_an_existing_probability_model_exactly(self) -> None:
        random = np.random.default_rng(22)
        rows = 500
        labels = random.integers(0, 2, (rows, 2))
        offset = random.normal(0, 0.5, (rows, 4))
        fit = fit_multitask_group_lasso_torch(
            np.empty((rows, 0), dtype=np.uint8),
            labels,
            [],
            (2, 2),
            0.1,
            offset_logits=offset,
            fit_intercept=False,
            device="cpu",
        )
        self.assertEqual(float(np.linalg.norm(fit.intercept)), 0.0)
        expected = multitask_probabilities(offset, (2, 2))
        residual = multitask_residual(
            np.empty((rows, 0), dtype=np.uint8), labels, fit, offset_logits=offset
        )
        reconstructed = residual.copy()
        reconstructed[np.arange(rows), labels[:, 0]] += 0.5
        reconstructed[np.arange(rows), 2 + labels[:, 1]] += 0.5
        reconstructed[:, :2] *= 2
        reconstructed[:, 2:] *= 2
        np.testing.assert_allclose(reconstructed, expected, atol=1e-6)

    def test_one_pass_multitask_kkt_matches_separate_scans(self) -> None:
        random = np.random.default_rng(812)
        rows = 700
        residual = random.normal(size=(rows, 5))
        residual[:, :2] -= residual[:, :2].mean(axis=0)
        residual[:, 2:] -= residual[:, 2:].mean(axis=0)
        states = random.integers(0, 4, (rows, 6), dtype=np.uint8)
        groups = [FeatureGroup(f"g-{index}", 4, 1.0) for index in range(6)]

        class Batch:
            pass

        def batches():
            batch = Batch()
            batch.groups = groups
            batch.states = states
            yield batch

        joint = scan_quantized_batches_multitask_kkt(
            residual,
            batches,
            (0.01, 0.02),
            (set(), set()),
            (slice(0, 2), slice(2, 5)),
            add_limit=6,
            device="cpu",
        )
        separate = (
            scan_quantized_batches_residual(
                residual[:, :2], batches, 0.01, set(), add_limit=6, device="cpu"
            ),
            scan_quantized_batches_residual(
                residual[:, 2:], batches, 0.02, set(), add_limit=6, device="cpu"
            ),
        )
        for combined, reference in zip(joint, separate):
            self.assertEqual(combined.scanned_groups, reference.scanned_groups)
            self.assertEqual(
                [row[0] for row in combined.violating_groups],
                [row[0] for row in reference.violating_groups],
            )
            self.assertAlmostEqual(combined.maximum_violation, reference.maximum_violation, places=6)

    def test_multitask_kkt_coalesces_exact_gradient_equivalents(self) -> None:
        random = np.random.default_rng(91)
        rows = 500
        state = random.integers(0, 4, rows, dtype=np.uint8)
        residual = random.normal(size=(rows, 2))
        groups = [
            FeatureGroup("asset/aia/test/1m/same", 4, 1.0),
            FeatureGroup("asset/btc/test/1m/same", 4, 1.0),
        ]

        class Batch:
            pass

        def batches():
            batch = Batch()
            batch.groups = groups
            batch.states = np.column_stack((state, state))
            yield batch

        scan = scan_quantized_batches_multitask_kkt(
            residual,
            batches,
            (0.0,),
            (set(),),
            (slice(0, 2),),
            add_limit=2,
            device="cpu",
        )[0]
        self.assertEqual(scan.scanned_groups, 2)
        self.assertEqual(len(scan.violating_groups), 1)
        self.assertEqual(scan.violating_groups[0][0], "asset/btc/test/1m/same")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from adaptive_curriculum import (
    LOSS_TERMS,
    GaussianProcessResidual,
    ProjectedQuadratic,
    enumerate_weight_candidates,
    initial_delay_continuation,
    interpolate_delay_reference,
    one_coordinate_neighbor_keys,
    propose_delay_seconds,
    recommended_schedule,
    schedule_pareto_front,
    schedule_point,
    select_gp_acquisition_indices,
    update_delay_continuation,
)
from run_adaptive_curriculum import (
    candidates_in_persisted_order,
    AdaptiveCurriculumRunner,
    constrained_parent_weight_candidate,
    delay_state_dict,
    delay_state_from_dict,
    merge_configuration,
    select_promotions,
    sweep_branch_models,
    unlink_branch_model,
)


class AdaptiveCurriculumTest(unittest.TestCase):
    def test_persisted_promotion_order_is_preserved_on_resume(self) -> None:
        candidates = [
            {"key": "third"},
            {"key": "first"},
            {"key": "second"},
        ]

        resumed = candidates_in_persisted_order(
            candidates,
            ["first", "second", "third"],
        )

        self.assertEqual(
            [candidate["key"] for candidate in resumed],
            ["first", "second", "third"],
        )

    def test_initial_parent_weights_apply_study_constraints(self) -> None:
        candidate = constrained_parent_weight_candidate(
            {
                "crossEntropy": 4,
                "probabilityMse": 1,
                "parameterMse": 0.25,
                "excessEntropy": 0.25,
                "temporalMutualInformation": 4,
                "oracleMutualInformation": 1,
            },
            {"excessEntropy": 0},
        )

        self.assertEqual(candidate.as_dict(), {
            "crossEntropy": 4,
            "probabilityMse": 1,
            "parameterMse": 0.25,
            "excessEntropy": 0,
            "temporalMutualInformation": 4,
            "oracleMutualInformation": 1,
        })

    def test_inherited_study_configuration_merges_nested_overrides(self) -> None:
        merged = merge_configuration(
            {
                "projection": {
                    "batchSize": 8,
                    "initialProbeCount": 48,
                    "adaptiveProbeCount": 16,
                },
            },
            {
                "projection": {
                    "initialProbeCount": 32,
                    "adaptiveProbeCount": 8,
                },
            },
        )
        self.assertEqual(merged["projection"], {
            "batchSize": 8,
            "initialProbeCount": 32,
            "adaptiveProbeCount": 8,
        })

    def test_absolute_space_requires_anchor_and_collapses_global_scale(self) -> None:
        candidates = enumerate_weight_candidates([0, 0.25, 1, 4])
        self.assertEqual(len(candidates), 3_330)
        values = {candidate.values for candidate in candidates}
        self.assertIn((4, 4, 4, 4, 4, 4), values)
        self.assertIn((4, 0, 0, 0, 0, 0), values)
        self.assertNotIn((0, 0, 0, 1, 1, 1), values)
        self.assertTrue(all(max(candidate.values) == 4 for candidate in candidates))
        self.assertTrue(all(
            set(candidate.values).issubset({0, 0.25, 1, 4})
            for candidate in candidates
        ))
        self.assertTrue(all(set(candidate.as_dict()) == set(LOSS_TERMS) for candidate in candidates))

    def test_fixed_entropy_removes_it_from_the_search_space(self) -> None:
        candidates = enumerate_weight_candidates(
            [0, 0.25, 1, 4],
            fixed_loss_weights={"excessEntropy": 0},
        )
        raw = enumerate_weight_candidates(
            [0, 0.25, 1, 4],
            canonicalize_global_scale=False,
            fixed_loss_weights={"excessEntropy": 0},
        )
        self.assertEqual(len(candidates), 774)
        self.assertEqual(len(raw), 1_008)
        self.assertTrue(all(candidate.values[3] == 0 for candidate in candidates))
        self.assertTrue(all(max(candidate.values) == 4 for candidate in candidates))

    def test_cross_entropy_can_be_constrained_to_positive_levels(self) -> None:
        candidates = enumerate_weight_candidates(
            [0, 0.25, 1, 4],
            fixed_loss_weights={"excessEntropy": 0},
            term_weight_levels={"crossEntropy": [0.25, 1, 4]},
        )
        raw = enumerate_weight_candidates(
            [0, 0.25, 1, 4],
            canonicalize_global_scale=False,
            fixed_loss_weights={"excessEntropy": 0},
            term_weight_levels={"crossEntropy": [0.25, 1, 4]},
        )
        self.assertEqual(len(candidates), 606)
        self.assertEqual(len(raw), 768)
        self.assertTrue(all(candidate.values[0] > 0 for candidate in candidates))
        self.assertTrue(all(candidate.values[3] == 0 for candidate in candidates))

    def test_one_coordinate_neighbors_respect_canonical_grid(self) -> None:
        candidates = enumerate_weight_candidates(
            [0, 0.25, 1, 4],
            fixed_loss_weights={"excessEntropy": 0},
            term_weight_levels={"crossEntropy": [0.25, 1, 4]},
        )
        by_values = {
            candidate.values: candidate
            for candidate in candidates
        }
        origin = by_values[(4.0, 1.0, 0.0, 0.0, 4.0, 0.0)]

        keys = set(one_coordinate_neighbor_keys(
            origin,
            candidates,
            [0, 0.25, 1, 4],
            fixed_loss_weights={"excessEntropy": 0},
            term_weight_levels={"crossEntropy": [0.25, 1, 4]},
        ))

        expected = {
            by_values[values].key
            for values in (
                (1.0, 1.0, 0.0, 0.0, 4.0, 0.0),
                (4.0, 0.25, 0.0, 0.0, 4.0, 0.0),
                (4.0, 4.0, 0.0, 0.0, 4.0, 0.0),
                (4.0, 1.0, 0.25, 0.0, 4.0, 0.0),
                (4.0, 1.0, 0.0, 0.0, 1.0, 0.0),
                (4.0, 1.0, 0.0, 0.0, 4.0, 0.25),
            )
        }
        self.assertEqual(keys, expected)
        self.assertTrue(all(
            by_values_key.values[3] == 0
            for by_values_key in candidates
            if by_values_key.key in keys
        ))

    def test_neighborhood_trials_expand_each_parent_without_duplicates(self) -> None:
        candidates = enumerate_weight_candidates(
            [0, 0.25, 1, 4],
            fixed_loss_weights={"excessEntropy": 0},
            term_weight_levels={"crossEntropy": [0.25, 1, 4]},
        )
        origin = next(
            candidate
            for candidate in candidates
            if candidate.values == (4.0, 1.0, 0.0, 0.0, 4.0, 0.0)
        )
        runner = object.__new__(AdaptiveCurriculumRunner)
        runner.config = {
            "absoluteWeightLevels": [0, 0.25, 1, 4],
            "canonicalizeGlobalScale": True,
            "fixedLossWeights": {"excessEntropy": 0},
            "termWeightLevels": {"crossEntropy": [0.25, 1, 4]},
            "neighborhoodExpansion": {"seedCountPerParent": 1},
        }
        runner.candidates = candidates
        runner.candidate_by_key = {
            candidate.key: candidate
            for candidate in candidates
        }
        parent = {"key": "parent-a"}
        seed = {
            "key": "seed-a",
            "searchParentKey": parent["key"],
            "weightCandidate": origin.key,
            "validation": {
                "klDivergence": 0.5,
                "klDivergenceStdDev": 1.0,
            },
            "_searchParent": parent,
        }

        trials = runner.neighborhood_trials([seed])

        self.assertEqual(len(trials), 6)
        self.assertTrue(all(
            trial["_initialize"] is seed
            and trial["_searchParent"] is parent
            and trial["_candidate"].key != origin.key
            for trial in trials
        ))
        self.assertEqual(
            len({trial["_candidate"].key for trial in trials}),
            len(trials),
        )

    def test_projected_quadratic_includes_clipped_full_tuple(self) -> None:
        model = ProjectedQuadratic(
            base_validation=2,
            linear=np.asarray([1, 2, 0, 0, 0, 0], dtype=np.float64),
            quadratic=np.eye(6, dtype=np.float64),
            gradient_gram=np.eye(6, dtype=np.float64),
            learning_rate=0.1,
            maximum_gradient_norm=1,
        )
        score = model.scores(np.asarray([[1, 1, 0, 0, 0, 0]], dtype=np.float64))[0]
        scale = 1 / np.sqrt(2)
        expected = 2 - 0.1 * scale * 3 + 0.5 * 0.1**2 * scale**2 * 2
        self.assertAlmostEqual(score, expected)

    def test_gp_residual_interpolates_and_acquisition_excludes_probes(self) -> None:
        inputs = np.asarray([[1, 0], [0.25, 1], [0, 1]], dtype=np.float64)
        gp = GaussianProcessResidual(noise=1e-8)
        gp.fit(inputs, np.asarray([0.2, -0.1, 0.3]))
        mean, uncertainty = gp.predict(inputs)
        np.testing.assert_allclose(mean, [0.2, -0.1, 0.3], atol=1e-5)
        self.assertTrue(np.all(uncertainty < 1e-3))

        candidates = np.pad(inputs, ((0, 0), (0, 4)))
        selected, prediction, spread = select_gp_acquisition_indices(
            candidates,
            np.asarray([1.0, 0.9, 1.2]),
            [0, 1],
            [1.1, 0.8],
            1,
        )
        self.assertEqual(selected, [2])
        self.assertEqual(prediction.shape, (3,))
        self.assertEqual(spread.shape, (3,))

    def test_delay_controller_dwells_advances_and_backtracks(self) -> None:
        self.assertEqual(
            propose_delay_seconds(3_600, 1_333, 1, 3),
            [2_267, 2_934, 3_267],
        )
        state = initial_delay_continuation(3_600, 1, 1_800)
        dwell = update_delay_continuation(
            state,
            1.5,
            1.0,
            minimum_delay_seconds=1,
            absolute_tolerance=0.02,
            relative_tolerance=0.02,
            minimum_improvement=0.01,
            patience=2,
            maximum_dwell_epochs=4,
        )
        self.assertEqual(dwell.action, "dwell")
        advance = update_delay_continuation(
            dwell.state,
            1.01,
            1.0,
            minimum_delay_seconds=1,
            absolute_tolerance=0.02,
            relative_tolerance=0.02,
            minimum_improvement=0.01,
            patience=2,
            maximum_dwell_epochs=4,
        )
        self.assertEqual(advance.action, "advance")

        stalled = state
        for _ in range(3):
            decision = update_delay_continuation(
                stalled,
                1.5,
                1.0,
                minimum_delay_seconds=1,
                absolute_tolerance=0.02,
                relative_tolerance=0.02,
                minimum_improvement=0.01,
                patience=2,
                maximum_dwell_epochs=4,
            )
            stalled = decision.state
        self.assertEqual(decision.action, "backtrack")
        self.assertEqual(decision.state.step_seconds, 900)

    def test_delay_controller_grows_from_one_second_after_success(self) -> None:
        state = initial_delay_continuation(3_600, 1, 1)
        decision = update_delay_continuation(
            state,
            1.0,
            1.0,
            minimum_delay_seconds=1,
            absolute_tolerance=0.02,
            relative_tolerance=0.02,
            minimum_improvement=0.01,
            patience=2,
            maximum_dwell_epochs=4,
            step_growth_factor=1.5,
        )
        self.assertEqual(decision.action, "advance")
        self.assertEqual(decision.state.anchor_delay_seconds, 3_599)
        self.assertEqual(decision.state.step_seconds, 2)
        self.assertEqual(decision.state.trial_delay_seconds, 3_597)

    def test_schedule_selection_uses_progress_then_effort_and_quality(self) -> None:
        def point(
            key: str,
            delay_seconds: int,
            batches: int,
            validation_kl: float,
            std: float,
        ):
            return schedule_point(
                {
                    "key": key,
                    "delayMs": delay_seconds * 1_000,
                    "validation": {
                        "klDivergence": validation_kl,
                        "klDivergenceStdDev": std,
                    },
                    "lineage": [{
                        "trainingBatches": batches,
                        "qualityGap": validation_kl - 0.66,
                    }],
                },
                initial_delay_seconds=3_600,
                reference_kl=0.66,
                absolute_quality_tolerance=0.05,
                relative_quality_tolerance=0,
                batch_size=256,
                gradient_accumulation=2,
            )

        near = point("near", 3_590, 256, 0.67, 1.2)
        efficient_far = point("efficient-far", 3_580, 512, 0.68, 1.3)
        dominated_far = point("dominated-far", 3_580, 768, 0.69, 1.4)
        points = [near, efficient_far, dominated_far]

        self.assertEqual(
            {point["key"] for point in schedule_pareto_front(points)},
            {"near", "efficient-far"},
        )
        self.assertEqual(
            recommended_schedule(points)["key"],
            "efficient-far",
        )

    def test_log_delay_reference(self) -> None:
        references = {0: 3.3, 60: 2, 1_800: 0.75, 3_600: 0.66}
        value = interpolate_delay_reference(600, references)
        self.assertGreater(value, 0.75)
        self.assertLess(value, 2)

    def test_untrained_delay_state_is_strict_json_compatible(self) -> None:
        state = initial_delay_continuation(3_600, 1, 1_800)
        serialized = delay_state_dict(state)
        self.assertIsNone(serialized["best_validation"])
        restored = delay_state_from_dict(serialized)
        self.assertTrue(np.isinf(restored.best_validation))
        self.assertEqual(restored.trial_delay_seconds, 1_800)

    def test_promotions_preserve_parent_coverage(self) -> None:
        candidates = [
            {
                "key": "a2",
                "searchParentKey": "a",
                "validation": {
                    "klDivergence": 2.0,
                    "klDivergenceStdDev": 1.0,
                },
            },
            {
                "key": "a1",
                "searchParentKey": "a",
                "validation": {
                    "klDivergence": 1.0,
                    "klDivergenceStdDev": 1.0,
                },
            },
            {
                "key": "b1",
                "searchParentKey": "b",
                "validation": {
                    "klDivergence": 3.0,
                    "klDivergenceStdDev": 1.0,
                },
            },
        ]
        selected = select_promotions(candidates, 2)
        self.assertEqual({item["searchParentKey"] for item in selected}, {"a", "b"})
        self.assertIn("a1", {item["key"] for item in selected})

    def test_retention_sweep_preserves_active_and_selected_models(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "branches"
            obsolete = root / "r001-obsolete" / "best-model.pt"
            selected = root / "r001-selected" / "best-model.pt"
            active = root / "r002-active" / "best-model.pt"
            outside = Path(temporary) / "best-model.pt"
            for model in (obsolete, selected, active, outside):
                model.parent.mkdir(parents=True, exist_ok=True)
                model.write_bytes(b"checkpoint")

            result = sweep_branch_models(root, [selected], "r002-")

            self.assertFalse(obsolete.exists())
            self.assertTrue(selected.exists())
            self.assertTrue(active.exists())
            self.assertTrue(outside.exists())
            self.assertEqual(result["removedModels"], 1)
            self.assertEqual(result["removedBytes"], len(b"checkpoint"))
            self.assertEqual(result["retainedModels"], 2)
            self.assertEqual(unlink_branch_model(root, outside), 0)


if __name__ == "__main__":
    unittest.main()

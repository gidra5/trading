from __future__ import annotations

import math
import unittest

import numpy as np
import torch

from joint_price_oracle_actions import (
    SwitchBalancedActionLossWeights,
    actionable_policy_metrics_numpy,
    exact_state_actionable_policy_metrics_numpy,
    execution_policy_native_scale,
    greedy_base_logit_rollout_tensor,
    greedy_teacher_rollout_numpy,
    greedy_teacher_rollout_tensor,
    rebalance_equity_factor_numpy,
    rebalance_equity_factor_tensor,
    resolve_execution_policy_config,
    self_conditioned_current_exposures_tensor,
    switch_balanced_action_objective,
    switch_balanced_weights_tensor,
    transition_conditioned_logits_numpy,
    transition_conditioned_logits_tensor,
    transition_conditioned_probabilities_numpy,
    transition_conditioned_probabilities_tensor,
    transition_logits_numpy,
    transition_logits_tensor,
)


class JointPriceOracleActionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
        self.tensor_grid = torch.tensor(self.grid, dtype=torch.float64)

    def test_rebalance_factor_matches_exact_buy_and_sell_formula(self) -> None:
        friction = 0.00175
        current = 0.4
        expected = []
        for target in self.grid:
            difference = target - current
            if difference > 0:
                expected.append(
                    1 - friction * difference
                    / (1 - friction + friction * target)
                )
            elif difference < 0:
                expected.append(
                    1 - friction * -difference / (1 - friction * target)
                )
            else:
                expected.append(1.0)
        actual = rebalance_equity_factor_numpy(
            current,
            self.grid,
            friction,
        )
        np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=0)
        tensor = rebalance_equity_factor_tensor(
            current,
            self.tensor_grid,
            friction,
        )
        np.testing.assert_allclose(
            tensor.numpy(),
            expected,
            atol=1e-15,
            rtol=0,
        )

    def test_numpy_and_tensor_transition_conditioning_are_identical(self) -> None:
        probabilities = np.asarray([
            [0.0, 0.4, 0.6],
            [0.7, 0.2, 0.1],
        ], dtype=np.float64)
        currents = np.asarray([0.0, 0.6], dtype=np.float64)
        numpy_transition = transition_logits_numpy(
            self.grid,
            currents,
        )
        tensor_transition = transition_logits_tensor(
            self.tensor_grid,
            torch.tensor(currents, dtype=torch.float64),
        )
        np.testing.assert_allclose(
            tensor_transition.numpy(),
            numpy_transition,
            atol=1e-13,
            rtol=1e-13,
        )
        numpy_conditioned = transition_conditioned_probabilities_numpy(
            probabilities,
            self.grid,
            currents,
        )
        tensor_conditioned = transition_conditioned_probabilities_tensor(
            torch.tensor(probabilities, dtype=torch.float64),
            self.tensor_grid,
            torch.tensor(currents, dtype=torch.float64),
        )
        np.testing.assert_allclose(
            tensor_conditioned.numpy(),
            numpy_conditioned,
            atol=1e-13,
            rtol=1e-13,
        )
        np.testing.assert_allclose(numpy_conditioned.sum(axis=-1), 1.0)
        self.assertEqual(numpy_conditioned[0, 0], 0.0)

        base_logits = np.log(np.maximum(probabilities, 1e-30))
        numpy_logits = transition_conditioned_logits_numpy(
            base_logits,
            self.grid,
            currents,
        )
        tensor_logits = transition_conditioned_logits_tensor(
            torch.tensor(base_logits, dtype=torch.float64),
            self.tensor_grid,
            torch.tensor(currents, dtype=torch.float64),
        )
        np.testing.assert_allclose(
            tensor_logits.numpy(),
            numpy_logits,
            atol=1e-13,
            rtol=1e-13,
        )

    def test_conditioning_preserves_base_logits_up_to_analytic_transition(self) -> None:
        logits = np.asarray([[2.0, -1.0, 0.5]], dtype=np.float64)
        shifted = logits + 17.0
        first = transition_conditioned_probabilities_numpy(
            np.exp(logits),
            self.grid,
            0.25,
        )
        second = transition_conditioned_probabilities_numpy(
            np.exp(shifted),
            self.grid,
            0.25,
        )
        np.testing.assert_allclose(first, second, atol=1e-14, rtol=0)

    def test_greedy_teacher_rollout_applies_friction_and_resets(self) -> None:
        target = np.asarray([
            [0.10, 0.43, 0.47],  # raw +1, but cost-aware hold at zero
            [0.01, 0.01, 0.98],  # enter +1
            [0.10, 0.47, 0.43],  # raw zero, but cost-aware hold at +1
            [0.98, 0.01, 0.01],  # flip to -1
        ], dtype=np.float64)
        rollout = greedy_teacher_rollout_numpy(target, self.grid)
        np.testing.assert_array_equal(rollout.target_indices, [1, 2, 2, 0])
        np.testing.assert_array_equal(
            rollout.current_exposures,
            [0.0, 0.0, 1.0, 1.0],
        )
        np.testing.assert_array_equal(
            rollout.switch_labels,
            [False, True, False, True],
        )
        np.testing.assert_array_equal(
            rollout.signed_transition_labels,
            [0, 1, 0, -1],
        )
        self.assertTrue(bool((rollout.conditional_margins >= 0).all()))
        self.assertTrue(bool((rollout.confidences >= 0).all()))
        self.assertTrue(bool((rollout.confidences <= 1).all()))

        reset = greedy_teacher_rollout_numpy(
            target,
            self.grid,
            reset_mask=np.asarray([False, False, True, False]),
        )
        np.testing.assert_array_equal(reset.target_indices, [1, 2, 1, 0])
        self.assertEqual(reset.current_exposures[2], 0.0)

        tensor = greedy_teacher_rollout_tensor(
            torch.tensor(target, dtype=torch.float64),
            self.tensor_grid,
        )
        np.testing.assert_array_equal(
            tensor.target_indices.numpy(),
            rollout.target_indices,
        )
        np.testing.assert_allclose(
            tensor.conditional_margins.numpy(),
            rollout.conditional_margins,
            atol=1e-13,
            rtol=1e-13,
        )

    def test_switch_balancing_assigns_equal_total_mass(self) -> None:
        labels = torch.tensor([True, False, False, False, True, False])
        weights = switch_balanced_weights_tensor(
            labels,
            target_switch_fraction=0.5,
        )
        self.assertAlmostEqual(float(weights.mean()), 1.0)
        self.assertAlmostEqual(
            float(weights[labels].sum()),
            float(weights[~labels].sum()),
        )
        no_switches = switch_balanced_weights_tensor(torch.zeros(4).bool())
        self.assertTrue(torch.equal(no_switches, torch.ones(4)))

        static = switch_balanced_weights_tensor(
            labels,
            target_switch_fraction=0.3,
            source_switch_fraction=0.1,
        )
        self.assertTrue(torch.equal(
            static[labels],
            torch.full((2,), 3.0),
        ))
        self.assertTrue(torch.allclose(
            static[~labels],
            torch.full((4,), 0.7 / 0.9),
        ))
        static_without_switches = switch_balanced_weights_tensor(
            torch.zeros(4).bool(),
            target_switch_fraction=0.3,
            source_switch_fraction=0.1,
        )
        self.assertTrue(torch.allclose(
            static_without_switches,
            torch.full((4,), 0.7 / 0.9),
        ))

    def test_model_rollout_state_is_detached_and_carries_forward(self) -> None:
        logits = torch.tensor([
            [0.0, 1.0, 5.0],
            [5.0, 1.0, 0.0],
            [0.0, 5.0, 1.0],
        ], dtype=torch.float64, requires_grad=True)
        rollout = greedy_base_logit_rollout_tensor(
            logits,
            self.tensor_grid,
            friction=0,
            initial_exposure=0.0,
        )
        torch.testing.assert_close(
            rollout.current_exposures,
            torch.tensor([0.0, 1.0, -1.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            rollout.target_exposures,
            torch.tensor([1.0, -1.0, 0.0], dtype=torch.float64),
        )
        self.assertFalse(rollout.current_exposures.requires_grad)

    def test_execution_policy_matches_scaled_bot_state_cap_gate_and_deadband(
        self,
    ) -> None:
        grid = np.asarray([-100.0, 0.0, 100.0])
        policy = resolve_execution_policy_config({
            "version": 2,
            "maximumLeverage": 1,
            "minimumConfidence": 0.05,
            "confidenceExposurePower": 0,
            "confidenceLeverageFloor": 0.75,
        })
        self.assertEqual(execution_policy_native_scale(grid, policy), 0.01)

        # This is the TypeScript parity case: after mapping a filled +1x
        # execution position to native +100x, transition friction keeps long.
        target = np.asarray([
            [0.0, 0.0, 1.0],
            [0.9, 0.0, 0.1],
        ])
        numpy_rollout = greedy_teacher_rollout_numpy(
            target,
            grid,
            friction=0.1,
            temperature=1,
            execution_policy=policy,
        )
        np.testing.assert_array_equal(
            numpy_rollout.current_exposures,
            [0.0, 100.0],
        )
        np.testing.assert_array_equal(
            numpy_rollout.target_exposures,
            [100.0, 100.0],
        )
        np.testing.assert_array_equal(
            numpy_rollout.switch_labels,
            [True, False],
        )
        tensor_rollout = greedy_teacher_rollout_tensor(
            torch.tensor(target, dtype=torch.float64),
            torch.tensor(grid, dtype=torch.float64),
            friction=0.1,
            temperature=1,
            execution_policy=policy,
        )
        np.testing.assert_allclose(
            tensor_rollout.target_exposures.numpy(),
            numpy_rollout.target_exposures,
        )
        self_states = self_conditioned_current_exposures_tensor(
            torch.tensor(target, dtype=torch.float64).clamp_min(1e-30).log(),
            torch.tensor(grid, dtype=torch.float64),
            friction=0.1,
            temperature=1,
            execution_policy=policy,
        )
        np.testing.assert_allclose(
            self_states.numpy(),
            numpy_rollout.current_exposures,
        )

        uniform = greedy_teacher_rollout_numpy(
            np.ones((1, 3)),
            grid,
            friction=0,
            temperature=1,
            execution_policy=policy,
        )
        self.assertEqual(uniform.target_exposures[0], 0)
        self.assertFalse(uniform.switch_labels[0])

        capped = greedy_teacher_rollout_numpy(
            np.asarray([[0.01, 0.09, 0.90]]),
            grid,
            friction=0,
            temperature=1,
            execution_policy=policy,
        )
        expected_cap = 100 * (0.75 + 0.25 * capped.confidences[0])
        self.assertAlmostEqual(capped.target_exposures[0], expected_cap)

        deadband = greedy_teacher_rollout_numpy(
            np.asarray([[0.0, 0.0, 1.0]]),
            grid,
            initial_exposure=80,
            friction=0,
            temperature=1,
            execution_policy=policy,
        )
        self.assertEqual(deadband.target_exposures[0], 80)
        self.assertFalse(deadband.switch_labels[0])

    def test_execution_policy_is_opt_in_and_rejects_invalid_versions(self) -> None:
        target = np.asarray([[1.0, 1.0, 1.0]])
        legacy = greedy_teacher_rollout_numpy(
            target,
            self.grid,
            friction=0,
            temperature=1,
        )
        # Legacy tie-breaking still moves to the first native grid cell; v2's
        # confidence gate holds.  This proves the default path was not changed.
        self.assertEqual(legacy.target_exposures[0], -1)
        with self.assertRaisesRegex(ValueError, "version must be 2"):
            resolve_execution_policy_config({"version": 1})

    def test_execution_aligned_objective_is_finite_and_rewards_ranking(self) -> None:
        target = torch.tensor([
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.98, 0.01, 0.01],
        ], dtype=torch.float64)
        rollout = greedy_teacher_rollout_tensor(target, self.tensor_grid)
        matching = target.log().detach().clone().requires_grad_(True)
        wrong = torch.flip(target, dims=(-1,)).log()
        loss_weights = SwitchBalancedActionLossWeights(
            hard_action=1,
            ranking=1,
            direction=1,
            conditional_kl=0.1,
        )
        matching_metrics = switch_balanced_action_objective(
            matching,
            target,
            self.tensor_grid,
            rollout.current_exposures,
            weights=loss_weights,
        )
        wrong_metrics = switch_balanced_action_objective(
            wrong,
            target,
            self.tensor_grid,
            rollout.current_exposures,
            weights=loss_weights,
        )
        self.assertTrue(bool(torch.isfinite(matching_metrics["loss"])))
        self.assertLess(
            float(matching_metrics["loss"].detach()),
            float(wrong_metrics["loss"].detach()),
        )
        self.assertAlmostEqual(
            float(matching_metrics["conditionalKlDivergence"].detach()),
            0.0,
            places=10,
        )
        matching_metrics["loss"].backward()
        self.assertIsNotNone(matching.grad)
        self.assertTrue(bool(torch.isfinite(matching.grad).all()))
        self.assertAlmostEqual(
            float(matching_metrics["meanExampleWeight"].detach()),
            1.0,
        )

    def test_actionable_metrics_distinguish_activity_sign_and_path(self) -> None:
        target_indices = np.asarray([1, 2, 2, 0, 0])
        predicted_indices = np.asarray([1, 2, 1, 0, 0])
        target = np.full((5, 3), 0.01, dtype=np.float64)
        target[np.arange(5), target_indices] = 0.98
        predicted = np.full((5, 3), math.log(0.01), dtype=np.float64)
        predicted[np.arange(5), predicted_indices] = math.log(0.98)
        metrics = actionable_policy_metrics_numpy(
            predicted,
            target,
            self.grid,
            friction=0,
        )
        self.assertEqual(metrics["targetSwitches"], 2)
        self.assertEqual(metrics["predictedSwitches"], 3)
        self.assertEqual(metrics["transitionTruePositive"], 2)
        self.assertAlmostEqual(metrics["transitionPrecision"], 2 / 3)
        self.assertAlmostEqual(metrics["transitionRecall"], 1.0)
        self.assertAlmostEqual(metrics["transitionF1"], 0.8)
        self.assertEqual(metrics["signedTransitionTruePositive"], 2)
        self.assertEqual(metrics["exactTransitionTruePositive"], 2)
        self.assertAlmostEqual(metrics["modeAccuracy"], 0.8)
        self.assertAlmostEqual(metrics["modeMeanAbsoluteError"], 0.2)
        self.assertAlmostEqual(metrics["pathMeanAbsoluteError"], 0.2)
        self.assertAlmostEqual(metrics["pathDirectionalAgreement"], 0.8)
        self.assertAlmostEqual(metrics["targetTurnover"], 3.0)
        self.assertAlmostEqual(metrics["predictedTurnover"], 3.0)
        self.assertAlmostEqual(metrics["turnoverRatio"], 1.0)
        self.assertGreater(metrics["meanConditionalRegret"], 0.0)

    def test_exact_state_metrics_compare_both_policies_at_same_current(self) -> None:
        target_indices = np.asarray([2, 0, 2, 1])
        target = np.full((4, 3), 0.01, dtype=np.float64)
        target[np.arange(4), target_indices] = 0.98
        matching = np.log(target)
        wrong = np.full((4, 3), math.log(0.01), dtype=np.float64)
        wrong[:, 1] = math.log(0.98)
        currents = np.asarray([0.0, 0.25, -0.25, 0.0])
        policy = {
            "version": 2,
            "maximumLeverage": 1,
            "minimumConfidence": 0,
            "confidenceExposurePower": 0,
            "confidenceLeverageFloor": 1,
        }
        exact = exact_state_actionable_policy_metrics_numpy(
            matching,
            target,
            self.grid,
            currents,
            friction=0,
            temperature=1,
            execution_policy=policy,
        )
        missed = exact_state_actionable_policy_metrics_numpy(
            wrong,
            target,
            self.grid,
            currents,
            friction=0,
            temperature=1,
            execution_policy=policy,
        )
        self.assertEqual(exact["signedTransitionF1"], 1.0)
        self.assertEqual(exact["executableTargetMeanAbsoluteError"], 0.0)
        self.assertEqual(exact["exactStateScore"], 0.0)
        self.assertTrue(math.isfinite(missed["exactStateScore"]))
        self.assertGreater(missed["exactStateScore"], exact["exactStateScore"])

        with self.assertRaisesRegex(ValueError, "one current exposure"):
            exact_state_actionable_policy_metrics_numpy(
                matching,
                target,
                self.grid,
                currents[:-1],
                execution_policy=policy,
            )

    def test_invalid_policy_parameters_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            transition_logits_numpy(self.grid, 0, temperature=0)
        with self.assertRaises(ValueError):
            rebalance_equity_factor_numpy(0, self.grid, -0.1)
        with self.assertRaises(ValueError):
            greedy_teacher_rollout_numpy(
                np.ones((2, 3)),
                self.grid,
                reset_mask=np.ones(3).astype(bool),
            )


if __name__ == "__main__":
    unittest.main()

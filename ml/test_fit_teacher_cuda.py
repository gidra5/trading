from __future__ import annotations

import unittest

import torch

from fit_teacher_cuda import (
    FitConfig,
    compact_visible_fit_config,
    cross_entropy_objective,
    fit_target,
    metric_surface,
    remap_cutoff_support,
    remap_score_support,
    transition_logits,
)
from mlp_model import (
    PolicySupport,
    conditional_policy_logits,
    materialize_raw_oracle_policy_map,
)


class VisibleMetricSurfaceTest(unittest.TestCase):
    def test_selects_only_usable_actions_and_current_exposures(self) -> None:
        actions = torch.linspace(-250, 250, 255)
        currents = torch.linspace(-250, 250, 255)
        base = torch.arange(510, dtype=torch.float32).reshape(2, 255)
        config = test_config(actions, currents)

        visible_base, visible_actions, visible_currents = metric_surface(
            base, actions, currents, config
        )

        expected = actions[(actions >= -100) & (actions <= 100)]
        self.assertEqual(tuple(visible_base.shape), (2, 101))
        self.assertEqual(visible_actions.numel(), 101)
        self.assertEqual(visible_currents.numel(), 101)
        torch.testing.assert_close(visible_actions, expected)
        torch.testing.assert_close(visible_currents, expected)
        self.assertGreater(float(visible_actions[0]), -100)
        self.assertLess(float(visible_actions[-1]), 100)

    def test_effective_only_values_cannot_change_metric_input(self) -> None:
        actions = torch.linspace(-250, 250, 255)
        currents = torch.linspace(-250, 250, 255)
        base = torch.ones((1, 255))
        changed = base.clone()
        changed[:, (actions < -100) | (actions > 100)] = 1_000_000
        config = test_config(actions, currents)

        visible, visible_actions, visible_currents = metric_surface(
            base, actions, currents, config
        )
        changed_visible, _, _ = metric_surface(changed, actions, currents, config)
        target, entropy = fit_target(
            visible, visible_actions, visible_currents, config
        )
        changed_target, changed_entropy = fit_target(
            changed_visible, visible_actions, visible_currents, config
        )

        torch.testing.assert_close(visible, changed_visible)
        torch.testing.assert_close(target, changed_target)
        torch.testing.assert_close(entropy, changed_entropy)


class RawOracleMapMaterializationTest(unittest.TestCase):
    def test_factorized_map_matches_production_teacher_transition(self) -> None:
        actions = torch.linspace(-250.0, 250.0, 255)
        currents = torch.linspace(-250.0, 250.0, 255)
        base = torch.softmax(torch.sin(actions / 17.0)[None, :], dim=-1)
        friction = 0.00175
        temperature = 0.01

        expected = torch.softmax(
            transition_logits(
                base,
                actions,
                currents,
                friction,
                1 / temperature,
            ),
            dim=-1,
        )
        actual = materialize_raw_oracle_policy_map(
            base,
            actions,
            currents,
            PolicySupport(-250.0, 250.0, -250.0, 250.0, friction, temperature),
        )

        self.assertEqual(tuple(actual.shape), (1, 255, 255))
        torch.testing.assert_close(actual, expected)


class CompactVisibleInitializationTest(unittest.TestCase):
    def test_score_remap_preserves_visible_probabilities(self) -> None:
        full_grid = torch.linspace(-250.0, 250.0, 255)
        config = test_config(full_grid, full_grid)
        config = FitConfig(**{
            **config.__dict__,
            "score_hinge_span": 200.0,
            "compact_visible_initialization": True,
        })
        compact = compact_visible_fit_config(config)
        raw = torch.tensor([
            [-1.0, 0.3, 2.0, -0.7, 1.2, -0.8, -14.0, 14.0],
            [0.1, -0.8, -1.5, 0.2, -2.0, 1.0, -14.0, 14.0],
        ])
        mapped = remap_score_support(raw[:, None, :], compact, config)[:, 0, :]
        actions = torch.linspace(-100.0, 100.0, 101)
        currents = torch.linspace(-100.0, 100.0, 31)
        current_rows = currents[None, :].expand(raw.shape[0], -1)
        compact_support = PolicySupport(
            -100.0, 100.0, -100.0, 100.0, 0.0015, 0.01, 200.0,
        )
        full_support = PolicySupport(
            -250.0, 250.0, -250.0, 250.0, 0.0015, 0.01, 200.0,
        )
        compact_logits = conditional_policy_logits(
            raw[:, None, :].expand(-1, currents.numel(), -1),
            actions,
            current_rows,
            compact_support,
        )
        full_logits = conditional_policy_logits(
            mapped[:, None, :].expand(-1, currents.numel(), -1),
            actions,
            current_rows,
            full_support,
        )
        torch.testing.assert_close(
            torch.log_softmax(compact_logits, -1),
            torch.log_softmax(full_logits, -1),
            atol=1e-5,
            rtol=1e-5,
        )

    def test_cutoff_remap_clips_only_outside_compact_support(self) -> None:
        grid = torch.linspace(-250.0, 250.0, 255)
        full = test_config(grid, grid)
        compact = compact_visible_fit_config(full)
        raw = torch.tensor([
            [-14.0, 14.0],
            [torch.logit(torch.tensor(0.68)), torch.logit(torch.tensor(0.32))],
        ])
        mapped = remap_cutoff_support(raw, full, compact)
        self.assertEqual(mapped[0, 0].item(), -14.0)
        self.assertEqual(mapped[0, 1].item(), 14.0)
        lower = -100.0 + 100.0 * torch.sigmoid(mapped[1, 0])
        upper = 100.0 * torch.sigmoid(mapped[1, 1])
        torch.testing.assert_close(lower, torch.tensor(-250.0 + 250.0 * 0.68))
        torch.testing.assert_close(upper, torch.tensor(250.0 * 0.32))


class QueuedOptimizerObjectiveTest(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_matches_shared_objective_with_padded_actions_and_custom_hinge_span(self) -> None:
        # 63 actions exercise the queued kernel's padded 64th lane. A hinge
        # span different from the 500-wide support verifies the compact-fit
        # calibration survives full-support adaptive refinement.
        from teacher_bfgs_triton import triton_bfgs

        torch.manual_seed(1337)
        device = torch.device("cuda")
        actions = torch.linspace(-250.0, 250.0, 63, device=device)
        currents = torch.linspace(-250.0, 250.0, 31, device=device)
        raw = torch.tensor([
            [-0.8, 0.3, 1.4, -0.2, 2.1, -1.2, -14.0, 14.0],
            [0.2, -0.5, -1.0, 0.4, -1.3, 0.7, -14.0, 14.0],
        ], device=device)
        target = torch.softmax(
            torch.randn(2, currents.numel(), actions.numel(), device=device),
            dim=-1,
        )
        support = PolicySupport(
            -250.0, 250.0, -250.0, 250.0, 0.00175, 0.01, 200.0,
        )
        parameter_mask = torch.ones(8, device=device)
        parameter_mask[6:] = 0

        _, queued_loss, _, _ = triton_bfgs(
            raw,
            target,
            actions,
            currents,
            support,
            maximum_iterations=0,
            tolerance=1e-8,
            parameter_mask=parameter_mask,
            line_search_candidates=8,
        )
        shared_loss = cross_entropy_objective(
            raw[:, None, :], target, actions, currents, support
        )[:, 0]

        self.assertTrue(bool(torch.isfinite(queued_loss).all()))
        torch.testing.assert_close(queued_loss, shared_loss, atol=2e-5, rtol=2e-5)


def test_config(actions: torch.Tensor, currents: torch.Tensor) -> FitConfig:
    return FitConfig(
        action_grid=actions.tolist(),
        current_grid=currents.tolist(),
        friction=0.0015,
        transition_log_scale=100,
        latent_lower=-250,
        latent_upper=250,
        visible_lower=-250,
        visible_upper=250,
        metric_visible_lower=-100,
        metric_visible_upper=100,
        sample_states=31,
        sample_actions=63,
        projection_iterations=1,
        iterations=1,
        adaptive_iterations=1,
        adaptive_rounds=0,
        restarts=1,
        batch_size=1,
        tolerance=1e-5,
        max_mean_kl=0.003,
        max_mean_mse=3e-6,
    )


if __name__ == "__main__":
    unittest.main()

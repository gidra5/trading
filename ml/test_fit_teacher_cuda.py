from __future__ import annotations

import unittest

import torch

from fit_teacher_cuda import FitConfig, fit_target, metric_surface


class VisibleMetricSurfaceTest(unittest.TestCase):
    def test_selects_only_usable_actions_and_current_exposures(self) -> None:
        actions = torch.linspace(-250, 250, 151)
        currents = torch.linspace(-250, 250, 151)
        base = torch.arange(302, dtype=torch.float32).reshape(2, 151)
        config = test_config(actions, currents)

        visible_base, visible_actions, visible_currents = metric_surface(
            base, actions, currents, config
        )

        self.assertEqual(tuple(visible_base.shape), (2, 61))
        self.assertEqual(visible_actions.numel(), 61)
        self.assertEqual(visible_currents.numel(), 61)
        self.assertAlmostEqual(float(visible_actions[0]), -100)
        self.assertAlmostEqual(float(visible_actions[-1]), 100)
        self.assertAlmostEqual(float(visible_currents[0]), -100)
        self.assertAlmostEqual(float(visible_currents[-1]), 100)

    def test_effective_only_values_cannot_change_metric_input(self) -> None:
        actions = torch.linspace(-250, 250, 151)
        currents = torch.linspace(-250, 250, 151)
        base = torch.ones((1, 151))
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
        sample_states=17,
        sample_actions=61,
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

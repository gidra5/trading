from __future__ import annotations

import unittest

import torch

from evaluate_autoregressive_density_episodes import (
    _randomized_sobol_uniforms,
    _sample_exact_tensor_returns,
    _sample_triangular_components,
    autoregressive_episode_metrics,
    evaluate_autoregressive_leads,
)
from exact_tensor_return_density import ExactTensorReturnDensity
from test_exact_tensor_return_density import density_contract


class AutoregressiveEpisodeMetricsTest(unittest.TestCase):
    def test_perfect_episode_forecasts_score_perfectly(self) -> None:
        target = torch.tensor([
            [1.0, -2.0, 3.0, -1.0],
            [-1.0, 4.0, -2.0, 2.0],
        ])
        result = autoregressive_episode_metrics(target, target)
        self.assertAlmostEqual(result["pooledCandles"]["correlation"], 1)
        self.assertAlmostEqual(result["pooledCandles"]["mseSkillVsZero"], 1)
        self.assertAlmostEqual(result["pooledCandles"]["directionAccuracy"], 1)
        self.assertAlmostEqual(result["episodeAverage"]["mseSkillVsZero"], 1)
        self.assertAlmostEqual(result["episodeAverage"]["directionAccuracy"], 1)
        self.assertAlmostEqual(result["episodeReturnCorrelation"], 1, places=5)
        self.assertAlmostEqual(
            result["episodeCumulativePathCorrelation"], 1, places=5
        )
        self.assertAlmostEqual(result["episodeEndpoint"]["correlation"], 1)

    def test_opposite_forecasts_have_negative_correlation(self) -> None:
        target = torch.tensor([
            [1.0, -2.0, 3.0, -1.0],
            [-2.0, 1.0, -4.0, 2.0],
        ])
        result = autoregressive_episode_metrics(-target, target)
        self.assertAlmostEqual(result["pooledCandles"]["correlation"], -1)
        self.assertAlmostEqual(result["pooledCandles"]["directionAccuracy"], 0)
        self.assertLess(result["pooledCandles"]["mseSkillVsZero"], 0)
        self.assertAlmostEqual(result["episodeReturnCorrelation"], -1, places=5)

    def test_zero_padding_is_excluded_from_all_metrics(self) -> None:
        target = torch.tensor([
            [1.0, -2.0, 3.0, 0.0],
            [-2.0, 1.0, -4.0, 2.0],
        ])
        prediction = target.clone()
        prediction[0, 3] = 1_000_000.0
        result = autoregressive_episode_metrics(
            prediction,
            target,
            source_episode_seconds=900,
        )
        self.assertEqual(result["activeCandles"], 7)
        self.assertEqual(result["sourceEpisodeSeconds"], 900)
        self.assertAlmostEqual(result["pooledCandles"]["mse"], 0)
        self.assertAlmostEqual(result["episodeEndpoint"]["mse"], 0)

    def test_triangular_component_samples_stay_inside_support(self) -> None:
        knots = torch.tensor([0.0, 0.2, 0.7, 1.0])
        component = torch.tensor([0, 1, 2, 3])
        sampled = _sample_triangular_components(
            component,
            torch.tensor([0.25, 0.5, 0.75, 0.25]),
            knots,
        )
        self.assertTrue(0 <= sampled[0] <= 0.2)
        self.assertTrue(0 <= sampled[1] <= 0.7)
        self.assertTrue(0.2 <= sampled[2] <= 1.0)
        self.assertTrue(0.7 <= sampled[3] <= 1.0)

    def test_randomized_sobol_uses_requested_replicates(self) -> None:
        values = _randomized_sobol_uniforms(
            3,
            replicates=4,
            trajectories_per_replicate=8,
            seed=17,
            device=torch.device("cpu"),
        )
        self.assertEqual(values.shape, (32, 6))
        self.assertTrue(bool(((values >= 0) & (values <= 1)).all()))

    def test_exact_tensor_sampling_returns_one_three_return_path(self) -> None:
        model = ExactTensorReturnDensity(
            torch.zeros(120), torch.ones(120), density_contract(),
            hidden_width=8, initial_radius=0.01, minimum_radius=0.0001,
            learnable_centering=False,
        )
        output = model(torch.zeros((5, 120)))
        sampled = _sample_exact_tensor_returns(
            output, torch.rand((5, 4)), model
        )
        self.assertEqual(sampled.shape, (5, 3))
        self.assertTrue(bool(torch.isfinite(sampled).all()))

    def test_single_return_model_is_rolled_out_for_each_requested_lead(self) -> None:
        class Model(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.register_buffer(
                    "density_component_return_means", torch.tensor([-1.0, 1.0])
                )

            def log_component_masses(self, history: torch.Tensor) -> torch.Tensor:
                positive = torch.sigmoid(history[:, -1])
                return torch.stack((1 - positive, positive), dim=1).log()

        class Dataset:
            return_count = 3

            def iter_batches(self, *_args, **_kwargs):
                yield (
                    torch.zeros((2, 120)),
                    torch.tensor([[0.1, 0.2, 0.3], [-0.1, -0.2, -0.3]]),
                    torch.ones(2),
                )

        result = evaluate_autoregressive_leads(
            Model(), Dataset(), "test", lead_count=3, batch_size=2,
            target_std=1.0, device=torch.device("cpu"),
        )
        self.assertEqual(len(result), 3)
        self.assertTrue(all(value["examples"] == 2 for value in result))


if __name__ == "__main__":
    unittest.main()

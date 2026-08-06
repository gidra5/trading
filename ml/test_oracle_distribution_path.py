from __future__ import annotations

import unittest

import torch

from oracle_distribution_path import (
    OracleDistributionMetricAccumulator,
    oracle_forward_kl_loss,
    oracle_mean_plus_p50_kl_loss,
)


class OracleDistributionPathTest(unittest.TestCase):
    def test_forward_kl_is_zero_for_identical_distributions(self) -> None:
        probabilities = torch.tensor([
            [0.0, 0.25, 0.75],
            [0.2, 0.3, 0.5],
        ])
        loss = oracle_forward_kl_loss(
            probabilities, probabilities, torch.ones(2),
            probability_floor=1e-8,
        )
        self.assertLess(abs(float(loss)), 1e-6)

    def test_forward_kl_remains_finite_across_different_hard_supports(self) -> None:
        logits = torch.tensor([
            [0.0, -torch.inf, -torch.inf],
            [-torch.inf, 0.0, -torch.inf],
        ], requires_grad=True)
        predicted = torch.softmax(logits, dim=-1)
        target = torch.tensor([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        loss = oracle_forward_kl_loss(
            predicted, target, torch.ones(2), probability_floor=1e-6
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad[torch.isfinite(logits)]).all())

    def test_metric_accumulator_reports_distribution_and_path_metrics(self) -> None:
        accumulator = OracleDistributionMetricAccumulator(
            torch.tensor([-1.0, 0.0, 1.0]),
            probability_floor=1e-8,
            track_kl_percentiles=True,
        )
        target_returns = torch.tensor([[0.1, -0.1], [0.2, 0.1]])
        probabilities = torch.tensor([[0.1, 0.8, 0.1], [0.0, 0.2, 0.8]])
        accumulator.add(
            target_returns,
            target_returns,
            probabilities,
            probabilities,
            torch.ones(2),
        )
        result = accumulator.result()
        self.assertLess(abs(float(result["klDivergence"])), 1e-6)
        self.assertEqual(result["modalActionAgreement"], 1.0)
        self.assertAlmostEqual(float(result["pathMse"]), 0.0)
        self.assertAlmostEqual(float(result["pathMseSkillVsZero"]), 1.0)
        self.assertLess(abs(float(result["klPercentiles"]["p95"])), 1e-6)

    def test_mean_plus_p50_loss_matches_explicit_batch_quantile(self) -> None:
        predicted = torch.tensor([
            [0.8, 0.2],
            [0.7, 0.3],
            [0.4, 0.6],
            [0.1, 0.9],
        ], requires_grad=True)
        target = torch.tensor([
            [0.6, 0.4],
            [0.2, 0.8],
            [0.5, 0.5],
            [0.9, 0.1],
        ])
        per_example = torch.stack([
            oracle_forward_kl_loss(
                predicted[index:index + 1],
                target[index:index + 1],
                torch.ones(1),
                probability_floor=1e-8,
            )
            for index in range(4)
        ])
        actual = oracle_mean_plus_p50_kl_loss(
            predicted, target, torch.ones(4),
            probability_floor=1e-8,
            mean_weight=1,
            p50_weight=1,
        )
        expected = per_example.mean() + torch.quantile(per_example, 0.5)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertGreater(float(predicted.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()

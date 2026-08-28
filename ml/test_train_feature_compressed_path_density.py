from __future__ import annotations

import unittest

import torch

from train_feature_compressed_path_density import PathMetrics


class PathMetricsTest(unittest.TestCase):
    def test_reports_per_lead_density_metrics_without_changing_joint_mean(self) -> None:
        metrics = PathMetrics(
            target_std=2.0,
            cumulative_std=3.0,
            steps=2,
            device=torch.device("cpu"),
        )
        metrics.add(
            prediction=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            target=torch.tensor([[1.5, 2.5], [2.0, 5.0]]),
            weights=torch.tensor([1.0, 0.5]),
            log_density=torch.tensor([[-2.0, -4.0], [-6.0, -8.0]]),
            crps=torch.tensor([[0.2, 0.4], [0.6, 0.8]]),
        )

        result = metrics.result()
        self.assertAlmostEqual(result["negativeLogLikelihood"], 13.0 / 3.0)
        expected_nll = ((2.0 + 0.5 * 6.0) / 1.5, (4.0 + 0.5 * 8.0) / 1.5)
        expected_crps = ((0.2 + 0.5 * 0.6) / 1.5, (0.4 + 0.5 * 0.8) / 1.5)
        for actual, expected in zip(
            result["perLeadNegativeLogLikelihood"], expected_nll, strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            result["perLeadMeanCrps"], expected_crps, strict=True
        ):
            self.assertAlmostEqual(actual, expected)
        for actual, expected in zip(
            result["perLeadNormalizedCrps"], expected_crps, strict=True
        ):
            self.assertAlmostEqual(actual, expected / 2.0)


if __name__ == "__main__":
    unittest.main()

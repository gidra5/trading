from __future__ import annotations

import unittest

import torch

from linear_next_return_path import LinearNextReturnPath, parameter_count
from next_return_dataset import HISTORY_RETURN_COUNT


class LinearNextReturnPathTest(unittest.TestCase):
    def test_linear_path_starts_at_each_lead_mean_and_backpropagates(self) -> None:
        target_mean = torch.linspace(-2e-6, 2e-6, 15)
        model = LinearNextReturnPath(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            target_mean,
            torch.full((15,), 1e-4),
        )
        prediction = model(torch.randn(7, HISTORY_RETURN_COUNT))
        self.assertEqual(prediction.shape, (7, 15))
        torch.testing.assert_close(
            prediction, target_mean.unsqueeze(0).expand(7, -1)
        )
        prediction.square().mean().backward()
        self.assertGreater(float(model.output.weight.grad.abs().sum()), 0)
        self.assertEqual(parameter_count(model), 1_815)

    def test_model_is_strictly_linear_in_standardized_feature_space(self) -> None:
        model = LinearNextReturnPath(
            torch.zeros(HISTORY_RETURN_COUNT),
            torch.ones(HISTORY_RETURN_COUNT),
            torch.zeros(3),
            torch.ones(3),
        )
        torch.nn.init.normal_(model.output.weight)
        torch.nn.init.normal_(model.output.bias)
        left = torch.randn(5, HISTORY_RETURN_COUNT)
        right = torch.randn(5, HISTORY_RETURN_COUNT)
        midpoint = (left + right) / 2
        torch.testing.assert_close(
            model.forward_standardized(midpoint),
            (
                model.forward_standardized(left)
                + model.forward_standardized(right)
            ) / 2,
        )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import torch

from compressed_path_return_density import CompressedPathOutput
from train_feature_compressed_path_density import (
    discrete_component_crps_terms,
    weighted_normalized_crps,
)


class _OneStepModel:
    return_count = 1


class DiscreteComponentCrpsTest(unittest.TestCase):
    def test_single_component_is_absolute_error(self) -> None:
        output = CompressedPathOutput(
            log_masses=(torch.zeros(2, 1),),
            expectations=torch.tensor([[0.5], [-0.25]]),
            component_means=(torch.tensor([[0.5], [-0.25]]),),
        )
        terms = discrete_component_crps_terms(
            output, torch.tensor([[0.0], [0.25]]), _OneStepModel()
        )
        torch.testing.assert_close(terms, torch.tensor([[0.5], [0.5]]))

    def test_two_point_identity_includes_pairwise_spread_credit(self) -> None:
        probabilities = torch.tensor([[0.5, 0.5]])
        values = torch.tensor([[0.0, 2.0]])
        output = CompressedPathOutput(
            log_masses=(torch.log(probabilities),),
            expectations=(probabilities * values).sum(dim=1, keepdim=True),
            component_means=(values,),
        )
        terms = discrete_component_crps_terms(
            output, torch.tensor([[0.0]]), _OneStepModel()
        )
        torch.testing.assert_close(terms, torch.tensor([[0.5]]))

    def test_weighted_normalized_score_has_finite_live_gradients(self) -> None:
        logits = torch.tensor(
            [[-0.4, 0.2, 0.1], [0.3, -0.1, 0.5]], requires_grad=True
        )
        values = torch.tensor(
            [[-0.2, 0.0, 0.4], [-0.1, 0.1, 0.5]], requires_grad=True
        )
        probabilities = torch.softmax(logits, dim=1)
        output = CompressedPathOutput(
            log_masses=(torch.log(probabilities),),
            expectations=(probabilities * values).sum(dim=1, keepdim=True),
            component_means=(values,),
        )
        score, terms = weighted_normalized_crps(
            output,
            torch.tensor([[0.05], [0.2]]),
            torch.ones(2),
            _OneStepModel(),
            target_std=0.5,
        )
        score.backward()
        self.assertEqual(terms.shape, (2, 1))
        self.assertTrue(bool(torch.isfinite(score)))
        self.assertTrue(bool(torch.isfinite(logits.grad).all()))
        self.assertTrue(bool(torch.isfinite(values.grad).all()))
        self.assertGreater(float(logits.grad.abs().sum()), 0)
        self.assertGreater(float(values.grad.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()

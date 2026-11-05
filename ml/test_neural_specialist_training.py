"""Independent checks of specialist targets, masked losses and accumulation."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.nn import functional as F

from stochastic_price_forecast import TimelineSplit
from train_neural_sign_size_forecast import (
    HORIZONS, constants_from_training, examples, loss_terms, score_predictions, select_origins,
)


class SpecialistTrainingTest(unittest.TestCase):
    def test_sign_loss_ignores_magnitude_and_exact_zeros(self):
        logits = torch.tensor([[.3, -.4], [-.9, .5], [.1, -.2]], dtype=torch.float64, requires_grad=True)
        targets = torch.tensor([[.002, -.001], [-.1, .08], [0., 0.]], dtype=torch.float64)
        scales = torch.ones(2, 2, dtype=torch.float64)
        first, mask = loss_terms(logits, targets, "sign", scales)
        second, _ = loss_terms(logits, targets*torch.tensor([[100., 1.], [.01, .2], [1., 1.]]), "sign", scales)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        torch.testing.assert_close(first[:2], F.binary_cross_entropy_with_logits(
            logits[:2], (targets[:2] > 0).float(), reduction="none"))
        self.assertFalse(bool(mask[-1].any()))
        first.sum().backward()
        torch.testing.assert_close(logits.grad[-1], torch.zeros(2, dtype=torch.float64), rtol=0, atol=0)

    def test_size_targets_use_the_correct_polarity_scale_and_ignore_other_heads(self):
        targets = torch.tensor([[.004, -.009], [-.012, .015], [0., 0.]], dtype=torch.float64)
        scales = torch.tensor([[.002, .004], [.005, .003]], dtype=torch.float64)
        prediction = torch.ones(3, 2, 2, dtype=torch.float64, requires_grad=True)
        loss, mask = loss_terms(prediction, targets, "size", scales)
        expected = torch.tensor([[[1., 0.], [0., 4.]], [[0., 4.], [4., 0.]], [[0., 0.], [0., 0.]]], dtype=torch.float64)
        torch.testing.assert_close(loss, expected)
        unit_changed, _ = loss_terms(prediction, targets*1e4, "size", scales*1e4)
        torch.testing.assert_close(unit_changed, expected)
        loss.sum().backward()
        torch.testing.assert_close(prediction.grad[~mask], torch.zeros_like(prediction.grad[~mask]), rtol=0, atol=0)

    def test_uneven_microbatch_accumulation_matches_full_objective_and_gradient(self):
        y = torch.tensor([[0., .002], [.004, -.012], [-.008, 0.], [.002, .009],
                          [0., -.003], [-.012, .015], [.008, -.006]], dtype=torch.float64)
        scales = torch.tensor([[.004, .008], [.003, .006]], dtype=torch.float64)
        for task in ("sign", "size"):
            with self.subTest(task=task):
                shape = (7, 2) if task == "sign" else (7, 2, 2)
                initial = torch.linspace(.1, 1.1, int(np.prod(shape)), dtype=torch.float64).reshape(shape)
                whole = initial.clone().requires_grad_()
                chunks = initial.clone().requires_grad_()
                if task == "sign":
                    full = torch.stack([F.binary_cross_entropy_with_logits(
                        whole[y[:, j] != 0, j], (y[y[:, j] != 0, j] > 0).float()) for j in range(2)]).mean()
                    denominators = (y != 0).sum(0).clamp_min(1)
                else:
                    full = torch.stack([((whole[(y[:, j] > 0) if k == 0 else (y[:, j] < 0), j, k]
                        -y[(y[:, j] > 0) if k == 0 else (y[:, j] < 0), j].abs()/scales[j, k])**2).mean()
                        for j in range(2) for k in range(2)]).mean()
                    denominators = torch.stack((y > 0, y < 0), -1).sum(0).clamp_min(1)
                full.backward()
                total = 0.
                for start, stop in ((0, 2), (2, 3), (3, 7)):
                    values, _ = loss_terms(chunks[start:stop], y[start:stop], task, scales)
                    part = (values.sum(0)/denominators).mean()
                    total += float(part.detach())
                    part.backward()
                self.assertAlmostEqual(total, float(full.detach()), places=7)
                torch.testing.assert_close(chunks.grad, whole.grad, rtol=1e-7, atol=1e-9)
                scores = score_predictions(initial.numpy(), y.numpy(), task, scales.numpy())
                self.assertAlmostEqual(float(scores.mean()), float(full.detach()), places=7)

    def test_price_endpoints_and_observed_inputs_are_causal_and_inside_split(self):
        length, axis = 8200, 123000
        timeline = np.arange(length*59, dtype=np.float32).reshape(length, 59)/1e5
        source = TimelineSplit(timeline, np.arange(1, length-1), axis)
        origins = select_origins(source, stride=7, limit=8)
        self.assertEqual(len(origins), 8)
        self.assertTrue(np.all(np.diff(origins) > 0))
        self.assertTrue(np.all(origins+max(HORIZONS) < length))
        prices = 100+np.sin(np.arange(length)/400)
        prices[8] = prices[7]
        def observed_prices(times, _):
            indices = (times-axis)//1000
            self.assertTrue(np.all((indices >= 0) & (indices < length)))
            return prices[indices]
        with patch("train_neural_sign_size_forecast.load_close_prices", side_effect=observed_prices):
            data = examples(source, np.array([7, 31, 80]))
            expected = np.array([[prices[o+h]/prices[o]-1 for h in HORIZONS] for o in (7, 31, 80)])
            np.testing.assert_array_equal(data.targets, expected)
            self.assertEqual(data.targets[0, 0], 0.)
            np.testing.assert_array_equal(data.features[0], timeline[[6, 7]])
            future_changed = timeline.copy()
            future_changed[8:] = -999
            changed_source = TimelineSplit(future_changed, source.origins, axis)
            np.testing.assert_array_equal(examples(changed_source, np.array([7])).features[0], data.features[0])
            for invalid in (np.array([0]), np.array([length-max(HORIZONS)])):
                with self.assertRaises(ValueError): examples(source, invalid)

    def test_training_constants_are_sign_conditional_simple_units(self):
        y = np.array([[.002, .004, .008, .016], [-.006, -.012, -.024, -.048], [0., 0., 0., 0.]])
        constants = constants_from_training(y)
        for j, value in enumerate(constants):
            self.assertEqual(value["horizonSeconds"], HORIZONS[j])
            self.assertEqual(value["pZero"], .375)
            self.assertEqual(value["pUpGivenActive"], .5)
            self.assertEqual(value["upMeanSimple"], y[0, j])
            self.assertEqual(value["downMeanSimple"], -y[1, j])
            self.assertEqual(value["activeTrainingExamples"], 2)


if __name__ == "__main__":
    unittest.main()

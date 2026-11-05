"""Independent checks of the ternary training objective and frozen transfer."""
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import torch

from neural_sign_size_forecast import StructuredSpecialist
from ternary_sign_size_forecast import CombinedSignSize, TernarySignSpecialist
from train_ternary_sign_size_forecast import (
    HORIZONS, loss_elements, make_pair, scores, target_classes,
)


class TernaryTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_three_class_cross_entropy_includes_exact_zeros_without_weights(self):
        y = torch.tensor([[-.002, 0.], [0., .005], [.02, -.03]], dtype=torch.float64)
        expected_classes = torch.tensor([[0, 1], [1, 2], [2, 0]])
        torch.testing.assert_close(target_classes(y), expected_classes)
        np.testing.assert_array_equal(target_classes(y.numpy()), expected_classes.numpy())
        logits = torch.tensor([[[.2, -.4, .7], [-.3, .8, .1]],
                               [[.4, -.2, .5], [-.7, .1, .3]],
                               [[.8, -.4, .2], [.6, -.5, .1]]], dtype=torch.float64, requires_grad=True)
        selected = logits.gather(-1, expected_classes[..., None])[..., 0]
        expected = logits.logsumexp(-1)-selected
        actual = loss_elements(logits, y, "ternary", torch.tensor([1., 20.]))
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(loss_elements(logits, y*100, "ternary", torch.tensor([100., 1.])), actual)
        actual.mean().backward()
        self.assertTrue(bool((logits.grad[y == 0] != 0).all()))
        self.assertTrue(bool((logits.grad[..., 1][y == 0] < 0).all()))
        np.testing.assert_allclose(scores(logits.detach().numpy(), y.numpy(), "ternary", np.ones(2)),
                                   expected.detach().numpy().mean(0), rtol=1e-14, atol=1e-14)

    def test_composed_mse_keeps_zero_targets_and_only_uses_signed_mean(self):
        y = torch.tensor([[0., .006], [-.002, 0.], [.003, -.004]], dtype=torch.float64)
        prediction = torch.tensor([[.001, .004], [-.001, .002], [.003, -.008]], dtype=torch.float64,
                                  requires_grad=True)
        second = torch.tensor([2e-6, 8e-6], dtype=torch.float64)
        actual = loss_elements(prediction, y, "combined", second)
        torch.testing.assert_close(actual, torch.square(prediction-y)/second, rtol=0, atol=0)
        actual.mean().backward()
        torch.testing.assert_close(prediction.grad, 2*(prediction.detach()-y)/(second*y.numel()))
        self.assertTrue(bool((prediction.grad[y == 0] != 0).all()))
        unit_changed = loss_elements(prediction.detach()*1e4, y*1e4, "combined", second*1e8)
        torch.testing.assert_close(unit_changed, actual)
        np.testing.assert_allclose(scores(prediction.detach().numpy(), y.numpy(), "combined", second.numpy()),
                                   actual.detach().numpy().mean(0), rtol=1e-14, atol=1e-14)

    def test_uneven_microbatches_match_whole_effective_batch_objective_and_gradient(self):
        y = torch.tensor([[0., .002], [.004, -.012], [-.008, 0.], [.002, .009],
                          [0., -.003], [-.012, .015], [.008, -.006]], dtype=torch.float64)
        second = torch.tensor([8e-5, 2e-4], dtype=torch.float64)
        for task in ("ternary", "combined"):
            with self.subTest(task=task):
                shape = (7, 2, 3) if task == "ternary" else (7, 2)
                initial = torch.linspace(-.1, .1, int(np.prod(shape)), dtype=torch.float64).reshape(shape)
                whole = initial.clone().requires_grad_()
                chunks = initial.clone().requires_grad_()
                if task == "ternary":
                    labels = target_classes(y)
                    expected = (whole.logsumexp(-1)-whole.gather(-1, labels[..., None])[..., 0]).mean()
                else:
                    expected = ((whole-y).square()/second).mean()
                expected.backward()
                total = 0.
                for start, stop in ((0, 2), (2, 3), (3, 7)):
                    part = loss_elements(chunks[start:stop], y[start:stop], task, second).sum()/y.numel()
                    total += float(part.detach())
                    part.backward()
                self.assertAlmostEqual(total, float(expected.detach()), places=12)
                torch.testing.assert_close(chunks.grad, whole.grad, rtol=1e-13, atol=1e-13)

    def test_trainer_singleton_transfer_preserves_selected_prediction_and_source(self):
        torch.manual_seed(271)
        mean, std = np.linspace(-.2, .2, 59, dtype=np.float32), np.linspace(.8, 1.2, 59, dtype=np.float32)
        sign = TernarySignSpecialist(torch.from_numpy(mean), torch.from_numpy(std), HORIZONS).eval()
        size = StructuredSpecialist("size", torch.from_numpy(mean), torch.from_numpy(std), HORIZONS).eval()
        scales = np.array([[.0002, .0003], [.002, .003], [.005, .004], [.01, .012]])
        sign_saved = {"horizonsSeconds": list(HORIZONS), "inputMean": torch.from_numpy(mean),
                      "inputStd": torch.from_numpy(std), "modelState": sign.state_dict()}
        size_saved = {"horizonsSeconds": list(HORIZONS), "inputMean": torch.from_numpy(mean),
                      "inputStd": torch.from_numpy(std), "modelState": size.state_dict(),
                      "sizeScales": torch.from_numpy(scales)}
        j = 2
        with patch("train_ternary_sign_size_forecast.load_verified", side_effect=[sign_saved, size_saved]):
            selected = make_pair({}, {}, j, mean, std, scales[j:j+1]).eval()
        full = CombinedSignSize(sign, size, torch.from_numpy(scales)).eval()
        x = torch.randn(3, 2, 59)
        with torch.no_grad():
            expected, actual = full.components(x), selected.components(x)
        self.assertEqual(selected.horizons, (HORIZONS[j],))
        self.assertEqual(selected.scales.dtype, torch.float64)
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a[:, 0], b[:, j], rtol=1e-6, atol=1e-9)
        self.assertFalse({p.data_ptr() for p in full.parameters()} & {p.data_ptr() for p in selected.parameters()})
        old_weight = size.readout.weight.detach().clone()
        with torch.no_grad(): selected.size.readout.weight.add_(1)
        torch.testing.assert_close(size.readout.weight, old_weight, rtol=0, atol=0)
        self.assertTrue(all(p.grad is None for p in full.parameters()))


if __name__ == "__main__":
    unittest.main()

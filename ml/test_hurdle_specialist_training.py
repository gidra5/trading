"""Independent objective and frozen-expert transfer checks for hurdle training."""
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch.distributions import Beta, Gamma
from torch.nn import functional as F

from hurdle_sign_size_forecast import HurdleSignSize
from export_hurdle_sign_size_forecast import load_combined as export_load_combined
from neural_sign_size_forecast import StructuredSpecialist
from train_hurdle_sign_size_forecast import (
    HORIZONS, checkpoint_payload, load_combined, loss_elements, make_shared_triplet, scores, sha,
)


class HurdleTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_zero_expert_uses_all_rows_and_exact_endpoint_equality(self):
        y = torch.tensor([[0., .001], [-.02, -0.], [.003, -.004]], dtype=torch.float64)
        logits = torch.tensor([[.1, -.2], [-.4, .8], [.2, -.6]], dtype=torch.float64, requires_grad=True)
        expected = -torch.where(y == 0, F.logsigmoid(logits), F.logsigmoid(-logits))
        actual = loss_elements(logits, y, "zero")
        torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)
        torch.testing.assert_close(loss_elements(logits, y*1e4, "zero"), actual, rtol=0, atol=0)
        actual.mean().backward()
        self.assertTrue(bool((logits.grad[y == 0] < 0).all()))
        self.assertTrue(bool((logits.grad[y != 0] > 0).all()))
        np.testing.assert_allclose(scores(logits.detach().numpy(), y.numpy(), "zero"),
                                   expected.detach().numpy().mean(0), rtol=1e-14, atol=1e-14)
        with self.assertRaises(ValueError): loss_elements(logits, y, "unknown")

    def test_full_joint_likelihood_and_uneven_microbatch_gradients_have_all_row_denominator(self):
        y = torch.tensor([[0., .002], [0., -.004], [-.003, 0.], [0., .001],
                          [.005, -.002], [0., 0.], [-.001, .003]], dtype=torch.float64)
        initial = torch.empty(7, 2, 6, dtype=torch.float64)
        initial[..., 0] = torch.linspace(-.6, .7, 14).reshape(7, 2)
        initial[..., 1] = torch.linspace(.8, -.3, 14).reshape(7, 2)
        initial[..., 2:] = torch.tensor([.002, .003, .8, 1.2])
        for task in ("zero", "combined"):
            with self.subTest(task=task):
                values = initial[..., 0] if task == "zero" else initial
                whole, chunks = values.clone().requires_grad_(), values.clone().requires_grad_()
                if task == "zero":
                    expected = -torch.where(y == 0, F.logsigmoid(whole), F.logsigmoid(-whole)).mean()
                else:
                    zero, sign, up, down, au, ad = whole.unbind(-1)
                    positive, negative, active = y > 0, y < 0, y != 0
                    class_sum = -torch.where(active, F.logsigmoid(-zero), F.logsigmoid(zero)).sum()
                    class_sum -= torch.where(positive[active], F.logsigmoid(sign[active]), F.logsigmoid(-sign[active])).sum()
                    density_sum = -Gamma(au[positive], au[positive]/up[positive]).log_prob(y[positive]).sum()
                    density_sum -= Beta(ad[negative], ad[negative]*(1-down[negative])/down[negative]).log_prob(-y[negative]).sum()
                    expected = (class_sum+density_sum)/y.numel()
                expected.backward()
                accumulated = 0.
                for start, stop in ((0, 1), (1, 4), (4, 7)):
                    part = loss_elements(chunks[start:stop], y[start:stop], task).sum()/y.numel()
                    accumulated += float(part.detach())
                    part.backward()
                self.assertAlmostEqual(accumulated, float(expected.detach()), delta=2e-10)
                torch.testing.assert_close(chunks.grad, whole.grad, rtol=1e-10, atol=1e-9)
                self.assertAlmostEqual(float(scores(values.numpy(), y.numpy(), task).mean()),
                                       float(expected.detach()), delta=2e-10)

    def test_shared_three_expert_transfer_preserves_all_horizons_and_owns_parameters(self):
        torch.manual_seed(522)
        mean, std = np.linspace(-.1, .1, 59, dtype=np.float32), np.linspace(.9, 1.1, 59, dtype=np.float32)
        experts = [StructuredSpecialist(task, torch.from_numpy(mean), torch.from_numpy(std), HORIZONS).eval()
                   for task in ("sign", "sign", "size")]
        scales = np.array([[.0001, .0002], [.002, .003], [.004, .005], [.008, .009]])
        saved = [{"horizonsSeconds": list(HORIZONS), "inputMean": torch.from_numpy(mean),
                  "inputStd": torch.from_numpy(std), "modelState": expert.state_dict(),
                  "sizeScales": torch.from_numpy(scales)} for expert in experts]
        with patch("train_hurdle_sign_size_forecast.load_verified", side_effect=saved):
            selected = make_shared_triplet({"zero": {}, "sign": {}, "size": {}}, mean, std, scales).eval()
        original = HurdleSignSize(*experts, torch.from_numpy(scales)).eval()
        x = torch.randn(3, 2, 59)
        with torch.no_grad():
            expected, actual = original(x), selected(x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(selected.sign(x), experts[1](x), rtol=0, atol=0)
        torch.testing.assert_close(selected.size(x), experts[2](x), rtol=0, atol=0)
        self.assertEqual(selected.horizons, HORIZONS)
        self.assertEqual(selected.scales.dtype, torch.float64)
        source_ptrs = {p.data_ptr() for p in original.parameters()}
        self.assertFalse(source_ptrs & {p.data_ptr() for p in selected.parameters()})
        old = [expert.readout.weight.detach().clone() for expert in experts]
        with torch.no_grad():
            for expert in (selected.zero, selected.sign, selected.size): expert.readout.weight.add_(1)
        for before, expert in zip(old, experts): torch.testing.assert_close(expert.readout.weight, before, rtol=0, atol=0)

    def test_training_and_export_loaders_preserve_entire_likelihood_checkpoint(self):
        torch.manual_seed(271)
        mean, std = np.zeros(59, dtype=np.float32), np.ones(59, dtype=np.float32)
        scales = np.array([[.002, .003]])
        experts = [StructuredSpecialist(task, torch.from_numpy(mean), torch.from_numpy(std), (900,))
                   for task in ("sign", "sign", "size")]
        model = HurdleSignSize(*experts, torch.from_numpy(scales), torch.tensor([[.6, 1.3]])).eval()
        with TemporaryDirectory() as directory:
            file = Path(directory)/"combined.pt"
            torch.save(checkpoint_payload(model, "combined", (900,), 2, mean, std, scales,
                                          "test-plan-sha", {"test": True}), file)
            record = {"checkpoint": str(file), "sha256": sha(file), "epoch": 2}
            training = load_combined(record, torch.device("cpu"))
            exported = export_load_combined(record, 900, "test-plan-sha", torch.device("cpu"))
            x = torch.randn(4, 2, 59)
            with torch.no_grad():
                torch.testing.assert_close(training(x), model(x), rtol=0, atol=0)
                torch.testing.assert_close(exported(x), model(x), rtol=0, atol=0)
            with self.assertRaises(ValueError):
                export_load_combined(record, 900, "wrong-plan-sha", torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()

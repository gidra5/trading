from __future__ import annotations

import io
import unittest

import numpy as np
import torch
from torch.nn import functional as F

from neural_sign_size_forecast import StructuredSpecialist
from normalized_glu_next_return import optimizer_parameter_groups
from ternary_sign_size_forecast import (
    CONTRACT, DOWN, ZERO, UP, CombinedSignSize, TernarySignSpecialist,
    physical_downside_cap, select_horizon_pair, ternary_sign_labels,
)


class TernarySignSizeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, dtype=torch.float32):
        torch.manual_seed(837)
        mean, std = torch.linspace(-.2, .2, 59), torch.linspace(.7, 1.3, 59)
        sign = TernarySignSpecialist(mean, std, (1, 900))
        size = StructuredSpecialist("size", mean, std, (1, 900))
        return CombinedSignSize(sign, size, torch.tensor([[.002, .003], [.02, .03]], dtype=dtype))

    def test_ternary_order_exact_zero_labels_and_learned_zero_mass(self):
        values = torch.tensor([[-.1, 0., .1], [-1e-12, -0., 1e-12]])
        torch.testing.assert_close(ternary_sign_labels(values), torch.tensor([[DOWN, ZERO, UP]]*2))
        model = self.model()
        with torch.no_grad():
            model.sign.readout.weight.zero_()
            model.sign.readout.bias.copy_(torch.tensor([[.2, .6, .2], [.4, .1, .5]]).log().flatten())
        logits = model.sign(torch.randn(3, 2, 59))
        self.assertEqual(logits.shape, (3, 2, 3))
        probabilities = logits.softmax(-1)
        torch.testing.assert_close(probabilities[0], torch.tensor([[.2, .6, .2], [.4, .1, .5]]))
        labels = torch.tensor([[DOWN, ZERO], [ZERO, UP], [UP, DOWN]])
        loss = F.cross_entropy(logits.reshape(-1, 3), labels.flatten())
        self.assertTrue(bool(torch.isfinite(loss)))

    def test_same_physical_mean_as_three_point_law_including_representable_cap(self):
        inputs = torch.randn(3, 2, 59)
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                model = self.model(dtype)
                with torch.no_grad():
                    model.size.readout.weight.zero_()
                    model.size.readout.bias.zero_()
                    model.scales[:, 1] = 2
                    probabilities, up, down = model.components(inputs)
                    mean = model(inputs)
                self.assertEqual(probabilities.shape, (3, 2, 3))
                self.assertEqual(up.shape, (3, 2))
                self.assertEqual(down.dtype, dtype)
                self.assertTrue(bool((1-down > 0).all()))
                self.assertTrue(bool((1+up > 0).all()))
                self.assertEqual(float(down[0, 0]), physical_downside_cap(dtype))
                self.assertEqual(model.downside_cap, physical_downside_cap(dtype))
                returns = np.stack((-down.double().numpy(), np.zeros((3, 2)), up.double().numpy()), -1)
                numerical_mean = np.sum(probabilities.double().numpy()*returns, axis=-1)
                np.testing.assert_allclose(mean.double().numpy(), numerical_mean, atol=3e-8, rtol=2e-7)
                np.testing.assert_allclose(probabilities.sum(-1).numpy(), 1, atol=1e-7)

    def test_signed_mse_reaches_all_three_logits_and_entire_experts(self):
        model = self.model()
        logits = []
        def retain_logits(_, __, output):
            output.retain_grad()
            logits.append(output)
        handle = model.sign.register_forward_hook(retain_logits)
        inputs = torch.randn(5, 2, 59, requires_grad=True)
        loss = F.mse_loss(model(inputs), torch.full((5, 2), .006))
        loss.backward()
        handle.remove()
        self.assertTrue(bool(torch.isfinite(inputs.grad).all()))
        for state in (DOWN, ZERO, UP):
            self.assertTrue(bool((logits[0].grad[..., state] != 0).any()))
        for expert in (model.sign, model.size):
            blocks = [getattr(expert.backbone, f"layer{i}") for i in range(1, 12)]
            blocks += [expert.backbone.initial_market, expert.backbone.initial_prefix,
                       *expert.backbone.layer4_source_branches]
            for block in blocks:
                for parameter in block.muon_parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                    self.assertTrue(bool((parameter.grad != 0).any()), type(block).__name__)
            self.assertTrue(bool((expert.readout.weight.grad != 0).any()))

    def test_optimizer_routing_covers_both_models_once(self):
        model = self.model()
        muon, adamw = optimizer_parameter_groups(model)
        ids = [id(p) for p in (*muon, *adamw)]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in model.parameters() if p.requires_grad})
        for expert in (model.sign, model.size):
            self.assertIn(id(expert.readout.weight), {id(p) for p in muon})
            self.assertIn(id(expert.readout.bias), {id(p) for p in adamw})
        self.assertNotIn(id(model.scales), set(ids))

    def test_checkpoint_reconstruction_preserves_both_experts_scales_and_cap(self):
        original = self.model(torch.float64).eval()
        inputs = torch.randn(3, 2, 59)
        with torch.no_grad(): expected = original.components(inputs), original(inputs)
        file = io.BytesIO()
        torch.save({"contract": CONTRACT, "horizons": original.horizons, "state": original.state_dict()}, file)
        file.seek(0)
        saved = torch.load(file, weights_only=True)
        state, horizons = saved["state"], saved["horizons"]
        sign = TernarySignSpecialist(state["sign.backbone.input_mean"], state["sign.backbone.input_std"], horizons)
        size = StructuredSpecialist("size", state["size.backbone.input_mean"], state["size.backbone.input_std"], horizons)
        restored = CombinedSignSize(sign, size, state["scales"]).eval()
        restored.load_state_dict(state)
        with torch.no_grad(): actual = restored.components(inputs), restored(inputs)
        for a, b in zip(actual[0], expected[0]): torch.testing.assert_close(a, b, rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)
        self.assertEqual(restored.downside_cap, original.downside_cap)

    def test_horizon_selection_preserves_predictions_without_parameter_sharing(self):
        source = self.model(torch.float64).eval()
        inputs = torch.randn(3, 2, 59)
        selected = select_horizon_pair(source.sign, source.size, source.scales, 900).eval()
        self.assertEqual(selected.horizons, (900,))
        with torch.no_grad():
            before = source.components(inputs), source(inputs)
            after = selected.components(inputs), selected(inputs)
        for a, b in zip(after[0], before[0]): torch.testing.assert_close(a[:, 0], b[:, 1], rtol=1e-6, atol=1e-8)
        torch.testing.assert_close(after[1][:, 0], before[1][:, 1], rtol=1e-6, atol=1e-8)
        self.assertFalse({id(p) for p in source.parameters()} & {id(p) for p in selected.parameters()})
        with torch.no_grad(): selected.sign.readout.bias.add_(1)
        self.assertFalse(torch.equal(selected.sign.readout.bias, source.sign.readout.bias[3:6]))

    def test_rejects_invalid_horizon_scales_and_targets(self):
        model = self.model()
        for scales in (torch.ones(2), torch.zeros(2, 2), torch.ones(2, 2, dtype=torch.int64),
                       torch.full((2, 2), float("inf"))):
            with self.assertRaises(ValueError): CombinedSignSize(model.sign, model.size, scales)
        with self.assertRaises(ValueError): ternary_sign_labels(torch.tensor([float("nan")]))
        with self.assertRaises(ValueError): select_horizon_pair(model.sign, model.size, model.scales, 123)


if __name__ == "__main__":
    unittest.main()

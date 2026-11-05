"""Shared-horizon ownership, objective weighting and selection regression checks."""
from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from hurdle_sign_size_forecast import HurdleSignSize, hurdle_nll_elements
from neural_sign_size_forecast import StructuredSpecialist
from train_hurdle_sign_size_forecast import aggregate_validation_score, fit, initialize_shapes
from train_neural_sign_size_forecast import HORIZONS


class FixedForecast(torch.nn.Module):
    def __init__(self, output: torch.Tensor):
        super().__init__()
        self.register_buffer("output", output)
        self.horizons = HORIZONS
        self.initialized_shapes = None

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.output[indices.long()]

    def set_shapes_(self, value: torch.Tensor) -> None:
        self.initialized_shapes = value.detach().clone()


class TinyHurdle(torch.nn.Module):
    """Small trainable four-horizon graph for the actual trainer control flow."""
    def __init__(self):
        super().__init__()
        self.readout = torch.nn.Linear(1, 4*4, dtype=torch.float64)

    def forward(self, x):
        raw = self.readout(x).reshape(-1, 4, 4)
        means = torch.nn.functional.softplus(raw[..., 2:])*0.002
        shapes = torch.ones_like(means)
        return torch.cat((raw[..., :2], means, shapes), dim=-1)


class SharedHurdleTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_aggregate_selects_global_score_instead_of_per_horizon_winners(self):
        # Candidate B wins the first horizon decisively, but A has better total NLL.
        a = np.array([-.2, -5., -5., -5.])
        b = np.array([-2., -4., -4., -4.])
        self.assertLess(b[0], a[0])
        self.assertLess(aggregate_validation_score(a), aggregate_validation_score(b))
        self.assertEqual(aggregate_validation_score(a), float(a.mean()))
        # Return-density unit changes add fixed per-horizon constants. Ranking
        # must stay unchanged; dividing by each horizon's NLL would not preserve it.
        constants = np.array([8., 2., -3., 11.])
        self.assertLess(aggregate_validation_score(a+constants), aggregate_validation_score(b+constants))
        for value in ([], [np.nan], [np.inf], [-np.inf]):
            with self.subTest(value=value), self.assertRaises(ValueError):
                aggregate_validation_score(np.asarray(value))

    def test_fit_retains_one_global_ema_checkpoint_when_horizon_winners_disagree(self):
        torch.manual_seed(287)
        model = TinyHurdle()
        x = torch.arange(5, dtype=torch.float64).unsqueeze(-1)/5
        y = torch.tensor([[0., .002, -.003, .004], [.001, -.002, 0., -.004],
                          [-.001, 0., .003, .004], [0., .002, -.003, -.004],
                          [.001, -.002, .003, 0.]], dtype=torch.float64)
        per_epoch = [np.array([-.2, -5., -5., -5.]), np.array([-2., -4., -4., -4.]),
                     np.array([-.1, -5.2, -5.2, -5.2])]
        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root/"plan.json").write_text("{}", encoding="utf-8")
            args = SimpleNamespace(batch_size=5, microbatch_size=2, no_compile=True, output=root)
            optimizers = lambda module: (torch.optim.SGD(module.parameters(), lr=.01),)
            with patch("train_hurdle_sign_size_forecast.make_optimizers", side_effect=optimizers), \
                 patch("train_hurdle_sign_size_forecast.scores", side_effect=per_epoch), \
                 patch("train_hurdle_sign_size_forecast.torch.save", wraps=torch.save) as save, \
                 redirect_stdout(io.StringIO()):
                record = fit(model, "combined", HORIZONS, x, y, x, y.numpy(),
                    np.square(y.numpy()).mean(0), np.zeros(59, np.float32), np.ones(59, np.float32),
                    np.full((4, 2), .002), {"test": True}, args, maximum=2, minimum=2,
                    patience=10, learning_rate=.001, half_life=1., prefix="finetuned")
            self.assertIsInstance(record, dict)
            self.assertEqual(record["epoch"], 2)
            self.assertEqual(record["validationScoresByHorizon"], per_epoch[2].tolist())
            self.assertEqual([call.args[0]["epoch"] for call in save.call_args_list], [0, 2])
            self.assertEqual([file.name for file in root.glob("*.pt")], ["finetuned-shared.pt"])
            saved = torch.load(root/"finetuned-shared.pt", weights_only=True)
            self.assertEqual(saved["horizonsSeconds"], list(HORIZONS))
            self.assertEqual(saved["validationScoresByHorizon"], per_epoch[2].tolist())
            self.assertEqual(saved["epoch"], 2)
            rows = [json.loads(line) for line in (root/"training.jsonl").read_text().splitlines()]
            epochs = [row for row in rows if row["event"] == "epoch"]
            self.assertEqual([row["bestEpoch"] for row in epochs], [0, 0, 2])

    def test_each_horizon_updates_the_same_three_shared_backbones(self):
        torch.manual_seed(222)
        mean, std = torch.zeros(59), torch.ones(59)
        experts = [StructuredSpecialist(task, mean, std, HORIZONS)
                   for task in ("sign", "sign", "size")]
        model = HurdleSignSize(*experts, torch.tensor(
            [[.0001, .0002], [.002, .003], [.004, .005], [.008, .009]], dtype=torch.float64))
        self.assertEqual(len([module for module in model.modules() if isinstance(module, StructuredSpecialist)]), 3)
        self.assertEqual([expert.readout.out_features for expert in experts], [4, 4, 8])
        ids = [{id(p) for p in expert.parameters()} for expert in experts]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        inputs = torch.randn(3, 2, 59)
        target = torch.tensor([[0., 0., 0., 0.], [.00012, .0022, .0042, .0082],
                               [-.00017, -.0027, -.0047, -.0087]], dtype=torch.float64)
        output = model(inputs)
        self.assertEqual(output.shape, (3, 4, 6))
        for horizon_index in range(4):
            model.zero_grad(set_to_none=True)
            hurdle_nll_elements(output, target)[:, horizon_index].mean().backward(retain_graph=True)
            for expert in experts:
                params = tuple(expert.backbone.parameters())
                self.assertTrue(any(p.grad is not None and bool((p.grad != 0).any()) for p in params))
                self.assertTrue(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in params))
                width = 2 if expert.task == "size" else 1
                rows = expert.readout.weight.grad.reshape(4, width, -1)
                self.assertTrue(bool((rows[horizon_index] != 0).any()))
                other = [j for j in range(4) if j != horizon_index]
                torch.testing.assert_close(rows[other], torch.zeros_like(rows[other]), rtol=0, atol=0)
            self.assertTrue(bool((model.raw_shapes.grad[horizon_index] != 0).all()))
            other = [j for j in range(4) if j != horizon_index]
            torch.testing.assert_close(model.raw_shapes.grad[other], torch.zeros_like(model.raw_shapes.grad[other]), rtol=0, atol=0)

    def test_shape_initialization_uses_every_horizon_and_its_actual_polarity(self):
        up = np.array([.0001, .002, .004, .008])
        down = np.array([.0002, .003, .005, .009])
        y = np.stack((up*.5, -down*.4, np.zeros(4), up*np.array([1.5, 2., 2.5, 3.]),
                      -down*np.array([1.2, 1.5, 2., 2.5])))
        output = torch.ones(5, 4, 6, dtype=torch.float64)
        output[..., 2] = torch.from_numpy(up)
        output[..., 3] = torch.from_numpy(down)
        model = FixedForecast(output)
        original_output, original_y = output.clone(), y.copy()
        metadata = initialize_shapes(model, torch.arange(5), y, batch=2)
        expected = np.empty((4, 2))
        for j in range(4):
            for k, condition in enumerate((y[:, j] > 0, y[:, j] < 0)):
                scale = up[j] if k == 0 else down[j]
                expected[j, k] = np.clip(1/np.square(np.abs(y[condition, j])/scale-1).mean(), .01, 50.)
        np.testing.assert_allclose(model.initialized_shapes.numpy(), expected, rtol=1e-14, atol=1e-14)
        self.assertFalse(metadata["testUsed"])
        torch.testing.assert_close(output, original_output, rtol=0, atol=0)
        np.testing.assert_array_equal(y, original_y)
        # A missing DOWN sample in just H4 must fail; checking H1 alone misses it.
        invalid = y.copy()
        invalid[:, 3] = np.abs(invalid[:, 3])
        with self.assertRaises(ValueError):
            initialize_shapes(FixedForecast(output), torch.arange(5), invalid, batch=2)


if __name__ == "__main__":
    unittest.main()

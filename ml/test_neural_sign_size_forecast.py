from __future__ import annotations

import io
import json
import math
from pathlib import Path
import unittest

import torch
from torch.nn import functional as F

from neural_sign_size_forecast import (
    CONTRACT, HORIZONS, StructuredSpecialist, architecture_config,
)
from normalized_glu_next_return import optimizer_parameter_groups


class StructuredSpecialistTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def model(self, task, horizons=HORIZONS):
        torch.manual_seed(731)
        return StructuredSpecialist(task, torch.linspace(-.1, .1, 59),
                                   torch.linspace(.7, 1.3, 59), horizons)

    def test_historical_architectures_and_parameter_counts_are_preserved(self):
        for task, count in (("sign", 4130613), ("size", 9067378)):
            with self.subTest(task=task):
                config = architecture_config(task)
                source = Path(__file__).parent / "training-plans" / f"{config['sourcePlanId']}.json"
                original = json.loads(source.read_text())["architecture"]
                for key in ("inputSteps", "outputSteps", "inputFeatures", "outputFeatures", "featureWidth",
                            "marketWidth", "prefixWidth", "featureDistributionWidth", "extendedPrefixWidth",
                            "nextFeatureDistributionWidth", "initialRadius", "minimumRadius", "learnableCentering",
                            "layer8Attention", "layer4ParallelBranches"):
                    self.assertEqual(config.get(key), original.get(key), key)
                model = self.model(task)
                self.assertEqual(sum(p.numel() for p in model.backbone.parameters()), count)
                self.assertEqual(model.backbone.layer4_branch_count, 5 if task == "size" else 2)

    def test_shapes_conditional_order_positive_domain_and_normalization(self):
        inputs = torch.randn(3, 2, 59)
        sign = self.model("sign", (1, 900))
        size = self.model("size", (1, 900))
        self.assertEqual(sign(inputs).shape, (3, 2))
        result = size(inputs).detach()
        self.assertEqual(result.shape, (3, 2, 2))
        self.assertTrue(bool(torch.isfinite(result).all() and (result > 0).all()))
        with torch.no_grad():
            size.readout.weight.zero_()
            size.readout.bias.copy_(torch.tensor([0., 1., 2., -1000.]))
        result = size(inputs).detach()
        self.assertEqual(float(result[0, 0, 0]), 1.)
        self.assertAlmostEqual(float(result[0, 0, 1]), float(F.softplus(torch.tensor(1.))/math.log(2)), places=6)
        self.assertAlmostEqual(float(result[0, 1, 0]), float(F.softplus(torch.tensor(2.))/math.log(2)), places=6)
        self.assertGreater(float(result[0, 1, 1]), 0.)
        captured = []
        handle = sign.backbone.layer1.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
        sign(inputs)
        handle.remove()
        expected = (inputs[:, 0]-sign.backbone.input_mean)/sign.backbone.input_std
        torch.testing.assert_close(captured[0], expected)

    def test_gradients_reach_every_branch_attention_and_task_readout(self):
        for task in ("sign", "size"):
            with self.subTest(task=task):
                model = self.model(task)
                inputs = torch.randn(5, 2, 59, requires_grad=True)
                result = model(inputs)
                loss = F.binary_cross_entropy_with_logits(result, torch.rand_like(result)) if task == "sign" \
                    else F.mse_loss(result, torch.rand_like(result)*2)
                loss.backward()
                self.assertTrue(bool(torch.isfinite(inputs.grad).all() and (inputs.grad != 0).any()))
                blocks = [getattr(model.backbone, f"layer{i}") for i in range(1, 12)]
                blocks += [model.backbone.initial_market, model.backbone.initial_prefix,
                           *model.backbone.layer4_source_branches]
                for block in blocks:
                    for parameter in block.muon_parameters():
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                        self.assertTrue(bool((parameter.grad != 0).any()), type(block).__name__)
                self.assertTrue(bool((model.readout.weight.grad != 0).any()))

    def test_optimizer_partition_covers_backbone_and_head_exactly_once(self):
        for task in ("sign", "size"):
            model = self.model(task)
            muon, adamw = optimizer_parameter_groups(model)
            self.assertIn(id(model.readout.weight), {id(p) for p in muon})
            self.assertIn(id(model.readout.bias), {id(p) for p in adamw})
            self.assertTrue(all(p.ndim == 2 for p in muon))
            all_ids = [id(p) for p in (*muon, *adamw)]
            self.assertEqual(len(all_ids), len(set(all_ids)))
            self.assertEqual(set(all_ids), {id(p) for p in model.parameters() if p.requires_grad})

    def test_saved_model_reconstructs_exactly_with_frozen_statistics_and_horizons(self):
        for task in ("sign", "size"):
            model = self.model(task, (900, 1)).eval()
            inputs = torch.randn(3, 2, 59)
            with torch.no_grad(): expected = model(inputs)
            buffer = io.BytesIO()
            torch.save({"contract": CONTRACT, "task": task, "horizons": model.horizons,
                        "architecture": architecture_config(task), "state": model.state_dict()}, buffer)
            buffer.seek(0)
            saved = torch.load(buffer, weights_only=True)
            reconstructed = StructuredSpecialist(saved["task"], saved["state"]["backbone.input_mean"],
                saved["state"]["backbone.input_std"], saved["horizons"]).eval()
            reconstructed.load_state_dict(saved["state"])
            with torch.no_grad(): actual = reconstructed(inputs)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_rejects_incompatible_contract_inputs(self):
        for task, mean, std, horizons in (("other", torch.zeros(59), torch.ones(59), HORIZONS),
                ("sign", torch.zeros(58), torch.ones(58), HORIZONS),
                ("sign", torch.zeros(59), torch.zeros(59), HORIZONS),
                ("sign", torch.zeros(59), torch.full((59,), float("nan")), HORIZONS),
                ("sign", torch.zeros(59), torch.ones(59), (1, 1)),
                ("sign", torch.zeros(59), torch.ones(59), (True,))):
            with self.assertRaises(ValueError): StructuredSpecialist(task, mean, std, horizons)
        with self.assertRaises(ValueError): self.model("sign")(torch.zeros(3, 59))


if __name__ == "__main__":
    unittest.main()

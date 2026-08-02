from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from joint_price_oracle import JointLossWeights, parameter_count
from joint_price_oracle_sequence import (
    BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
    BOUNDARY_SEQUENCE_PROTOTYPE_MIXTURE_ARCHITECTURE_CONTRACT,
    BoundaryPrototypeMixtureMinutePolicyModel,
)
from train_joint_price_oracle import (
    SEQUENCE_REUSE_TRAINING_CONTRACT,
    architecture_contract_for_model_config,
    build_model,
    build_training_checkpoint,
    configuration_fingerprint,
    policy_only_forward_objective,
    resolve_training_config,
    validate_plan,
)


def contiguous_closes(core_count: int) -> torch.Tensor:
    close_count = (59 + core_count) * 60 + 1
    time = torch.arange(close_count, dtype=torch.float32)
    log_close = 10.5 + time * 1e-6 + torch.sin(time / 37.0) * 2e-4
    return torch.exp(log_close).view(1, -1, 1)


def fixed_windows(closes: torch.Tensor, core_count: int) -> torch.Tensor:
    windows = [
        closes[:, index * 60:index * 60 + 3_601, :]
        for index in range(core_count)
    ]
    return torch.cat(windows, dim=0)


class PrototypeMixturePolicyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        cls.plan_file = (
            repo_root
            / "ml/training-plans/"
            "joint-price-oracle-kl-minute-sequence-boundary-"
            "prototype-mixture-tcn-v27-reuse.json"
        )
        cls.plan = json.loads(cls.plan_file.read_text(encoding="utf-8"))
        cls.v18_plan = json.loads((
            repo_root
            / "ml/training-plans/"
            "joint-price-oracle-kl-minute-sequence-boundary-tcn-"
            "v18-reuse.json"
        ).read_text(encoding="utf-8"))

    def small_model(self) -> BoundaryPrototypeMixtureMinutePolicyModel:
        config = copy.deepcopy(self.plan["model"])
        config["tokenWidth"] = 8
        config["policyHiddenWidth"] = 12
        torch.manual_seed(101)
        model = build_model(config)
        self.assertIsInstance(
            model,
            BoundaryPrototypeMixtureMinutePolicyModel,
        )
        return model

    def test_projection_is_continuous_weights_times_fixed_prototypes(
        self,
    ) -> None:
        model = self.small_model()
        mixture_logits = torch.linspace(-2, 2, 48).reshape(3, 16)
        weights = model.policy_mixture_weights(mixture_logits)
        probabilities = model.policy_probabilities_from_mixture_logits(
            mixture_logits
        )
        action_logits = model._project_policy_output(mixture_logits)

        self.assertEqual(tuple(weights.shape), (3, 16))
        self.assertEqual(tuple(probabilities.shape), (3, 101))
        torch.testing.assert_close(
            weights.sum(dim=-1),
            torch.ones(3),
        )
        torch.testing.assert_close(
            probabilities,
            weights @ model.policy_prototypes,
        )
        torch.testing.assert_close(
            torch.softmax(action_logits, dim=-1),
            probabilities,
            atol=2e-7,
            rtol=2e-6,
        )

    def test_epoch_zero_exactly_reconstructs_train_prior_mixture(self) -> None:
        model = self.small_model().eval()
        closes = contiguous_closes(2)
        expected = (
            model.initial_train_mixture_weights @ model.policy_prototypes
        )
        with torch.no_grad():
            probabilities = torch.softmax(
                model.forward_policy_logits(closes[:, :3_601]),
                dim=-1,
            )
        torch.testing.assert_close(
            probabilities[0],
            expected,
            atol=2e-7,
            rtol=2e-6,
        )
        self.assertTrue(bool((probabilities == probabilities[0]).all()))
        final_projection = model.policy_head[-1]
        self.assertTrue(bool((final_projection.weight == 0).all()))
        torch.testing.assert_close(
            final_projection.bias,
            model.initial_train_mixture_weights.log(),
        )
        self.assertTrue(bool((model.hour_scale_bypass.weight == 0).all()))
        self.assertTrue(bool((model.hour_scale_bypass.bias == 0).all()))

    def test_direct_soft_target_kl_backpropagates_into_mixture_weights_only(
        self,
    ) -> None:
        model = self.small_model().train()
        closes = contiguous_closes(2)[:, :3_601]
        encoded, hour_scale = model.encode_history(closes)
        mixture_logits = (
            model.policy_head(encoded)
            + model.hour_scale_bypass(hour_scale)
        )
        mixture_logits.retain_grad()
        action_logits = model._project_policy_output(mixture_logits)
        target = torch.softmax(
            torch.linspace(2, -2, 101).reshape(1, 101),
            dim=-1,
        )
        metrics = policy_only_forward_objective(
            model,
            closes,
            target,
            JointLossWeights(forecast=0, soft_layer_norm=0),
        )[1]
        manual_log_probability = torch.log_softmax(
            model.forward_policy_logits(closes),
            dim=-1,
        )
        manual_cross_entropy = -(target * manual_log_probability).sum()
        target_log = target.log()
        manual_kl = (target * (target_log - manual_log_probability)).sum()
        torch.testing.assert_close(metrics["loss"], manual_cross_entropy)
        torch.testing.assert_close(metrics["klDivergence"], manual_kl)

        basis_before = model.policy_prototypes.clone()
        direct_loss = -(target * torch.log_softmax(
            action_logits,
            dim=-1,
        )).sum()
        direct_loss.backward()
        assert mixture_logits.grad is not None
        self.assertGreater(float(mixture_logits.grad.abs().sum()), 0)
        self.assertIsNotNone(model.policy_head[-1].bias.grad)
        self.assertGreater(
            float(model.policy_head[-1].bias.grad.abs().sum()),
            0,
        )
        self.assertIsNone(model.policy_prototypes.grad)
        self.assertNotIn(
            "policy_prototypes",
            dict(model.named_parameters()),
        )
        self.assertIn("policy_prototypes", model.state_dict())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        optimizer.step()
        torch.testing.assert_close(
            model.policy_prototypes,
            basis_before,
            atol=0,
            rtol=0,
        )

    def test_reused_core_probabilities_equal_overlapping_fixed_windows(
        self,
    ) -> None:
        core_count = 5
        closes = contiguous_closes(core_count)
        windows = fixed_windows(closes, core_count)
        model = self.small_model().eval()
        # Exercise the learned, input-dependent path rather than the constant
        # epoch-zero initialization.
        torch.manual_seed(103)
        with torch.no_grad():
            model.policy_head[-1].weight.normal_(mean=0, std=0.1)
            model.hour_scale_bypass.weight.normal_(mean=0, std=0.1)
        with torch.no_grad():
            reused = model.forward_sequence_core(closes)[0]
            fixed = model.forward_policy_logits(windows)
        torch.testing.assert_close(reused, fixed, atol=5e-7, rtol=1e-7)

    def test_fixed_basis_action_logits_export_to_onnx(self) -> None:
        model = self.small_model().eval()
        example = contiguous_closes(1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "prototype-mixture.onnx"
            torch.onnx.export(
                model,
                example,
                output,
                input_names=["closes"],
                output_names=["action_logits"],
                dynamic_axes={
                    "closes": {0: "batch"},
                    "action_logits": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
                external_data=False,
                dynamo=False,
            )
            exported = onnx.load(output, load_external_data=True)
            onnx.checker.check_model(exported, full_check=True)
            actual = ReferenceEvaluator(exported).run(
                ["action_logits"],
                {"closes": example.numpy()},
            )[0]
        with torch.no_grad():
            expected = model(example).numpy()
        self.assertTrue(np.isfinite(actual).all())
        self.assertEqual(actual.shape, (1, 101))
        self.assertLessEqual(float(np.max(np.abs(actual - expected))), 1e-4)

    def test_v27_plan_seals_test_and_verifies_basis_provenance(self) -> None:
        validate_plan(self.plan)
        self.assertEqual(self.plan["version"], 27)
        self.assertEqual(self.plan["testPolicy"], "sealed-never-load")
        self.assertNotIn("evaluateTest", self.plan["training"])
        self.assertEqual(
            self.plan["training"]["targetRepresentation"],
            "verifiedOracleProbabilities",
        )
        self.assertEqual(
            self.plan["training"]["selectionMetric"],
            "klDivergence",
        )
        self.assertEqual(
            self.plan["training"]["sequenceCoreTraining"],
            {
                "contract": SEQUENCE_REUSE_TRAINING_CONTRACT,
                "coreRows": 1_440,
            },
        )
        self.assertNotIn("prototypeLabels", self.plan["training"])
        for key in ("seed", "learningRate", "betas", "epsilon"):
            self.assertEqual(
                self.plan["training"][key],
                self.v18_plan["training"][key],
            )
        self.assertEqual(
            self.plan["training"]["learningRateSchedule"],
            self.v18_plan["training"]["learningRateSchedule"],
        )
        self.assertEqual(
            self.plan["training"]["patience"],
            self.v18_plan["training"]["patience"],
        )
        self.assertEqual(
            architecture_contract_for_model_config(self.plan["model"]),
            BOUNDARY_SEQUENCE_PROTOTYPE_MIXTURE_ARCHITECTURE_CONTRACT,
        )
        v18 = copy.deepcopy(self.plan["model"])
        v18["variant"] = "minute_sequence_boundary_tcn"
        for key in tuple(v18):
            if key.startswith("prototype"):
                del v18[key]
        self.assertEqual(
            architecture_contract_for_model_config(v18),
            BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
        )
        self.assertNotEqual(
            architecture_contract_for_model_config(self.plan["model"]),
            architecture_contract_for_model_config(v18),
        )

        for key, replacement, message in (
            (
                "prototypeBasisContentSha256",
                "0" * 64,
                "content hash",
            ),
            (
                "prototypeSourceTrainFingerprintSha256",
                "0" * 64,
                "source-train fingerprint",
            ),
        ):
            with self.subTest(key=key):
                changed = copy.deepcopy(self.plan)
                changed["model"][key] = replacement
                with self.assertRaisesRegex(ValueError, message):
                    validate_plan(changed)

    def test_checkpoint_records_unique_basis_bound_contract(self) -> None:
        config = copy.deepcopy(self.plan["model"])
        config["tokenWidth"] = 8
        config["policyHiddenWidth"] = 12
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=48,
        )
        training_fingerprint = configuration_fingerprint(
            resolve_training_config(self.plan["training"])
        )
        checkpoint = build_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=0,
            global_step=0,
            best_validation=1.0,
            best_epoch=0,
            stale_epochs=0,
            validation={"klDivergence": 1.0},
            model_parameters=parameter_count(model),
            dataset_fingerprint="v27-prototype-mixture-dataset",
            model_config=config,
            plan_id=self.plan["id"],
            device=torch.device("cpu"),
            interrupted=False,
            training_config_fingerprint=training_fingerprint,
        )
        self.assertEqual(
            checkpoint["architectureContract"],
            BOUNDARY_SEQUENCE_PROTOTYPE_MIXTURE_ARCHITECTURE_CONTRACT,
        )
        self.assertEqual(checkpoint["modelConfig"], config)
        self.assertEqual(
            checkpoint["modelConfig"]["prototypeBasisContentSha256"],
            self.plan["model"]["prototypeBasisContentSha256"],
        )
        self.assertIn("policy_prototypes", checkpoint["model"])


if __name__ == "__main__":
    unittest.main()

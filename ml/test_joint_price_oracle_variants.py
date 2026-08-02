from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from export_joint_price_oracle import validate_export_checkpoint
from joint_price_oracle import (
    ARCHITECTURE_CONTRACT,
    OUTPUT_ACTION_COUNT,
    FixedDctOrdinalLogitProjection,
    architecture_contract_with_policy_decoder,
    fixed_dct_ordinal_logit_basis,
    parameter_count,
)
from joint_price_oracle_variants import (
    LONG_CONTEXT_PATCH_MIXER_CONTRACT,
    PATCH_TRANSFORMER_CONTRACT,
    RESIDUAL_MIXER_CONTRACT,
    TCN_CONTRACT,
    CausalReversibleStreamNormalizer,
    LearnedMultiScaleTrendResidual,
    LongContextMultiScalePatchMixer,
    MultiResolutionPatchTransformer,
    MultiScaleDilatedTcn,
    MultiScaleResidualMixer,
    build_variant_model,
    causal_standardized_close_features,
)
from train_joint_price_oracle import (
    DATA_CONTRACT,
    architecture_contract_for_model_config,
    atomic_torch_save,
    build_model,
    build_training_checkpoint,
    configuration_fingerprint,
    load_resume_checkpoint,
    resolve_training_config,
    validate_plan,
)


class JointPriceOracleVariantTest(unittest.TestCase):
    @staticmethod
    def closes(batch: int = 2, length: int = 3_600) -> torch.Tensor:
        generator = torch.Generator().manual_seed(71)
        returns = torch.randn(batch, length, generator=generator) * 1e-4
        return (60_000 * torch.exp(returns.cumsum(dim=1))).unsqueeze(-1)

    @staticmethod
    def patch_model() -> MultiResolutionPatchTransformer:
        return MultiResolutionPatchTransformer(
            patch_sizes=(60, 300),
            model_width=32,
            attention_heads=4,
            layer_count=2,
            feed_forward_width=64,
            policy_hidden_width=32,
            dropout=0,
            forecast_coarse_steps=60,
            forecast_rank=8,
        )

    @staticmethod
    def tcn_model() -> MultiScaleDilatedTcn:
        return MultiScaleDilatedTcn(
            scales=(30, 60),
            tcn_width=24,
            dilations=(1, 2, 4, 8, 16, 32, 64),
            fusion_width=32,
            policy_hidden_width=32,
            dropout=0,
            forecast_coarse_steps=60,
            forecast_rank=8,
        )

    @staticmethod
    def residual_mixer_model(
        policy_logit_rank: int | None = None,
    ) -> MultiScaleResidualMixer:
        return MultiScaleResidualMixer(
            aggregate_scales=(60, 300, 900),
            stream_width=16,
            stream_mixer_layers=1,
            stream_feed_forward_width=24,
            fusion_width=24,
            policy_hidden_width=24,
            policy_logit_rank=policy_logit_rank,
            dropout=0,
            forecast_coarse_steps=60,
            forecast_rank=4,
        )

    @staticmethod
    def long_context_model() -> LongContextMultiScalePatchMixer:
        return LongContextMultiScalePatchMixer(
            context_length=3_600,
            forecast_horizon=60,
            patch_sizes=(60, 300),
            encoder_width=16,
            kernel_size=3,
            dilations=(1, 2, 4, 8, 16, 32),
            query_count=2,
            fusion_width=24,
            policy_hidden_width=24,
            dropout=0,
            forecast_coarse_steps=10,
            forecast_rank=4,
        )

    def test_causal_features_do_not_read_suffix(self) -> None:
        closes = self.closes(batch=1)
        changed = closes.clone()
        changed[:, 2_400:] *= 1.25
        original = causal_standardized_close_features(closes)
        modified = causal_standardized_close_features(changed)
        self.assertTrue(torch.equal(
            original[:, :, :2_400],
            modified[:, :, :2_400],
        ))

    def test_causal_features_preserve_absolute_movement_scale(self) -> None:
        returns = torch.sin(torch.linspace(0, 20, 3_600)) * 5e-5
        low_volatility = (60_000 * torch.exp(
            returns.cumsum(dim=0)
        )).view(1, -1, 1)
        high_volatility = (60_000 * torch.exp(
            (returns * 4).cumsum(dim=0)
        )).view(1, -1, 1)
        low_features = causal_standardized_close_features(low_volatility)
        high_features = causal_standardized_close_features(high_volatility)
        self.assertEqual(tuple(low_features.shape), (1, 5, 3_600))
        # Shape-normalized channels can be similar, but fixed-scale return,
        # path, and RMS channels must expose the fourfold volatility change.
        self.assertGreater(float(
            (low_features[:, 2:] - high_features[:, 2:]).abs().mean()
        ), 0.05)

    def test_patch_transformer_tokens_are_strictly_causal(self) -> None:
        model = self.patch_model().eval()
        closes = self.closes(batch=1)
        changed = closes.clone()
        changed[:, 3_000:] *= 0.8
        with torch.no_grad():
            original = model.encode_history(closes)
            modified = model.encode_history(changed)
        unaffected = model.patch_end_times < 3_000
        self.assertTrue(torch.allclose(
            original[:, :-1, :][:, unaffected, :],
            modified[:, :-1, :][:, unaffected, :],
            atol=2e-6,
            rtol=2e-6,
        ))

    def test_tcn_branch_sequences_are_causal(self) -> None:
        model = self.tcn_model().eval()
        closes = self.closes(batch=1)
        changed = closes.clone()
        changed[:, 3_000:] *= 1.1
        with torch.no_grad():
            original = model.encode_branches(closes)
            modified = model.encode_branches(changed)
        for scale, original_branch, modified_branch in zip(
            model.scales,
            original,
            modified,
            strict=True,
        ):
            unaffected_tokens = 3_000 // scale
            self.assertTrue(torch.allclose(
                original_branch[:, :, :unaffected_tokens],
                modified_branch[:, :, :unaffected_tokens],
                atol=2e-6,
                rtol=2e-6,
            ))

    def test_long_context_patch_sequences_are_strictly_causal(self) -> None:
        model = self.long_context_model().eval()
        closes = self.closes(batch=1)
        changed = closes.clone()
        changed[:, 3_000:] *= 1.15
        with torch.no_grad():
            original = model.encode_branches(closes)
            modified = model.encode_branches(changed)
        for patch_size, original_branch, modified_branch in zip(
            model.patch_sizes,
            original,
            modified,
            strict=True,
        ):
            unaffected_tokens = 3_000 // patch_size
            self.assertTrue(torch.allclose(
                original_branch[:, :unaffected_tokens],
                modified_branch[:, :unaffected_tokens],
                atol=2e-6,
                rtol=2e-6,
            ))

    def test_long_context_policy_logits_do_not_execute_forecast_head(
        self,
    ) -> None:
        class ForecastTrap(torch.nn.Module):
            def forward(self, _summary):
                raise AssertionError("forecast head executed")

        model = self.long_context_model().eval()
        checkpoint_keys = tuple(model.state_dict())
        closes = self.closes(batch=1)
        with torch.no_grad():
            expected = model.forward_policy_logits(closes)
        self.assertEqual(checkpoint_keys, tuple(model.state_dict()))
        model.forecast_head = ForecastTrap()
        with torch.no_grad():
            logits = model.forward_policy_logits(closes)
        self.assertEqual(tuple(logits.shape), (1, OUTPUT_ACTION_COUNT))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        torch.testing.assert_close(logits, expected)
        with self.assertRaisesRegex(AssertionError, "forecast head executed"):
            model(closes)

    def test_tcn_prunes_each_scale_to_a_full_receptive_field(self) -> None:
        model = MultiScaleDilatedTcn(dropout=0)
        self.assertEqual(
            [branch.dilations[-1] for branch in model.branches],
            [256, 64, 32],
        )
        for scale, branch in zip(model.scales, model.branches, strict=True):
            self.assertGreaterEqual(
                branch.receptive_field,
                model.context_length // scale,
            )
        with self.assertRaisesRegex(ValueError, "does not cover"):
            MultiScaleDilatedTcn(
                scales=(5,),
                dilations=(1, 2, 4, 8, 16, 32, 64),
            )

    def test_learned_multiscale_decomposition_starts_as_moving_average(
        self,
    ) -> None:
        decomposition = LearnedMultiScaleTrendResidual((5, 20))
        values = torch.randn(2, 100)
        trends, residuals = decomposition(values)
        for scale, weights, trend, residual in zip(
            decomposition.scales,
            decomposition.learned_weights(),
            trends,
            residuals,
            strict=True,
        ):
            self.assertTrue(torch.allclose(
                weights,
                torch.full_like(weights, 1.0 / scale),
            ))
            self.assertTrue(torch.allclose(
                trend + residual,
                values,
                atol=1e-7,
                rtol=1e-7,
            ))

    def test_causal_stream_normalization_is_reversible_and_prefix_safe(
        self,
    ) -> None:
        normalizer = CausalReversibleStreamNormalizer(1e-8)
        values = torch.randn(2, 300).cumsum(dim=1) * 1e-4 + 11.0
        changed = values.clone()
        changed[:, 200:] += 2.0
        normalized, mean, scale = normalizer.normalize(values)
        modified, modified_mean, modified_scale = normalizer.normalize(changed)
        self.assertTrue(torch.allclose(
            normalizer.denormalize(normalized, mean, scale),
            values,
            atol=2e-6,
            rtol=2e-6,
        ))
        self.assertTrue(torch.equal(normalized[:, :200], modified[:, :200]))
        self.assertTrue(torch.equal(mean[:, :200], modified_mean[:, :200]))
        self.assertTrue(torch.equal(scale[:, :200], modified_scale[:, :200]))

    def test_residual_mixer_stream_tokens_are_strictly_causal(self) -> None:
        model = self.residual_mixer_model(policy_logit_rank=16).eval()
        closes = self.closes(batch=1)
        changed = closes.clone()
        changed[:, 3_000:] *= 1.2
        with torch.no_grad():
            original = model.encode_streams(closes)
            modified = model.encode_streams(changed)
        for scale, original_stream, modified_stream in zip(
            model.stream_scales,
            original,
            modified,
            strict=True,
        ):
            unaffected_tokens = 3_000 // scale
            self.assertTrue(torch.allclose(
                original_stream[:, :unaffected_tokens],
                modified_stream[:, :unaffected_tokens],
                atol=2e-6,
                rtol=2e-6,
            ))

    def test_fixed_dct_policy_decoder_is_ordered_low_rank_and_finite(
        self,
    ) -> None:
        basis = fixed_dct_ordinal_logit_basis(101, 16)
        self.assertEqual(tuple(basis.shape), (16, 101))
        self.assertTrue(torch.allclose(
            basis @ basis.transpose(0, 1),
            torch.eye(16),
            atol=2e-6,
            rtol=2e-6,
        ))
        self.assertLess(float(basis.mean(dim=1).abs().max()), 2e-7)

        decoder = FixedDctOrdinalLogitProjection(24, 101, 16)
        hidden = torch.randn(3, 24, requires_grad=True)
        logits = decoder(hidden)
        self.assertEqual(tuple(logits.shape), (3, 101))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        logits.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(hidden.grad).all()))

        legacy = self.residual_mixer_model()
        structured = self.residual_mixer_model(policy_logit_rank=16)
        self.assertEqual(
            legacy.architecture_contract,
            RESIDUAL_MIXER_CONTRACT,
        )
        expected_contract = architecture_contract_with_policy_decoder(
            RESIDUAL_MIXER_CONTRACT,
            16,
        )
        self.assertEqual(structured.architecture_contract, expected_contract)
        self.assertEqual(
            architecture_contract_for_model_config({
                "variant": "multiscale_residual_mixer",
                "contextLength": 3_600,
                "forecastHorizon": 3_600,
                "variableCount": 1,
                "actionCount": 101,
                "policyLogitRank": 16,
            }),
            expected_contract,
        )

    def test_variants_return_compatible_finite_outputs_and_gradients(
        self,
    ) -> None:
        for model in (
            self.patch_model(),
            self.tcn_model(),
            self.residual_mixer_model(),
        ):
            with self.subTest(model=type(model).__name__):
                model.train()
                output = model.forward_with_forecast(self.closes())
                self.assertEqual(tuple(output.policy_logits.shape), (2, 101))
                self.assertEqual(tuple(output.predicted_closes.shape), (2, 3_600, 1))
                self.assertEqual(
                    tuple(output.predicted_log_movements.shape),
                    (2, 3_600, 1),
                )
                for value in output:
                    self.assertTrue(bool(torch.isfinite(value).all()))
                loss = (
                    output.policy_logits.square().mean()
                    + output.predicted_log_movements.mean()
                    + output.soft_layer_norm_mean
                    + output.soft_layer_norm_variance
                )
                loss.backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(all(gradient is not None for gradient in gradients))
                self.assertTrue(all(
                    bool(torch.isfinite(gradient).all())
                    for gradient in gradients
                    if gradient is not None
                ))

        long_context = self.long_context_model().train()
        output = long_context.forward_with_forecast(self.closes())
        self.assertEqual(tuple(output.policy_logits.shape), (2, 101))
        self.assertEqual(tuple(output.predicted_closes.shape), (2, 60, 1))
        loss = (
            output.policy_logits.square().mean()
            + output.predicted_log_movements.square().mean()
            + output.soft_layer_norm_mean
            + output.soft_layer_norm_variance
        )
        loss.backward()
        self.assertTrue(all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in long_context.parameters()
            if parameter.requires_grad
        ))

    def test_factory_validates_contract_and_default_models_are_small(self) -> None:
        common = {
            "contextLength": 3_600,
            "forecastHorizon": 3_600,
            "variableCount": 1,
            "actionCount": 101,
        }
        for variant in (
            "patch_transformer",
            "dilated_tcn",
            "multiscale_residual_mixer",
        ):
            model = build_variant_model({**common, "variant": variant})
            parameter_count = sum(
                parameter.numel()
                for parameter in model.parameters()
                if parameter.requires_grad
            )
            self.assertLess(parameter_count, 20_000_000)
            if variant == "multiscale_residual_mixer":
                self.assertLess(parameter_count, 2_000_000)
        with self.assertRaisesRegex(ValueError, "3,600"):
            build_variant_model({
                **common,
                "variant": "patch_transformer",
                "contextLength": 1_800,
            })
        with self.assertRaisesRegex(ValueError, "divide"):
            build_variant_model({
                **common,
                "variant": "patch_transformer",
                "patchSizes": [17],
            })

        long_context = build_variant_model({
            **common,
            "variant": "long_context_patch_mixer",
            "contextLength": 21_600,
            "patchSizes": [60, 300],
        })
        self.assertEqual(
            long_context.architecture_contract,
            LONG_CONTEXT_PATCH_MIXER_CONTRACT,
        )
        self.assertLess(sum(
            parameter.numel()
            for parameter in long_context.parameters()
            if parameter.requires_grad
        ), 2_000_000)

    def test_trainer_factory_and_plan_validation_are_variant_aware(self) -> None:
        common = {
            "contextLength": 3_600,
            "forecastHorizon": 3_600,
            "variableCount": 1,
            "actionCount": 101,
        }
        expected = {
            "patch_transformer": PATCH_TRANSFORMER_CONTRACT,
            "dilated_tcn": TCN_CONTRACT,
            "multiscale_residual_mixer": RESIDUAL_MIXER_CONTRACT,
        }
        for variant, contract in expected.items():
            config = {**common, "variant": variant}
            model = build_model(config)
            self.assertEqual(model.architecture_contract, contract)
            self.assertEqual(
                architecture_contract_for_model_config(config),
                contract,
            )

        repo_root = Path(__file__).resolve().parents[1]
        plan = json.loads((
            repo_root
            / "ml"
            / "training-plans"
            / "joint-price-oracle-decision-conditioned-v3.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(
            architecture_contract_for_model_config(plan["model"]),
            ARCHITECTURE_CONTRACT,
        )
        plan["id"] = "joint-price-oracle-variant-validation-test"
        plan["model"] = {
            **common,
            "variant": "patch_transformer",
            "patchSizes": [60, 300],
        }
        validate_plan(plan)
        plan["model"]["patchSizes"] = [17]
        with self.assertRaisesRegex(ValueError, "divide"):
            validate_plan(plan)

        plan["model"] = {
            **common,
            "variant": "long_context_patch_mixer",
            "contextLength": 21_600,
            "patchSizes": [60, 300],
        }
        plan["training"]["lossWeights"] = {
            "policyCrossEntropy": 1,
            "conditionedPolicyCrossEntropy": 0,
            "forecast": 0,
            "softLayerNorm": 0,
        }
        validate_plan(plan)
        self.assertEqual(
            architecture_contract_for_model_config(plan["model"]),
            LONG_CONTEXT_PATCH_MIXER_CONTRACT,
        )

    def test_residual_mixer_exports_with_matching_onnx_logits(self) -> None:
        model = self.residual_mixer_model(policy_logit_rank=16).eval()
        example = self.closes(batch=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "residual-mixer.onnx"
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
            with torch.no_grad():
                torch_logits = model(example).numpy()
            onnx_logits = ReferenceEvaluator(exported).run(
                ["action_logits"],
                {"closes": example.numpy()},
            )[0]
            self.assertTrue(np.isfinite(onnx_logits).all())
            self.assertLess(
                float(np.max(np.abs(torch_logits - onnx_logits))),
                1e-3,
            )

    def test_long_context_patch_mixer_exports_with_matching_onnx_logits(
        self,
    ) -> None:
        model = self.long_context_model().eval()
        example = self.closes(batch=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "long-context-patch-mixer.onnx"
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
            with torch.no_grad():
                torch_logits = model(example).numpy()
            onnx_logits = ReferenceEvaluator(exported).run(
                ["action_logits"],
                {"closes": example.numpy()},
            )[0]
            self.assertTrue(np.isfinite(onnx_logits).all())
            self.assertLess(
                float(np.max(np.abs(torch_logits - onnx_logits))),
                1e-3,
            )

    def test_variant_checkpoint_contract_is_resume_isolated(self) -> None:
        model_config = {
            "variant": "dilated_tcn",
            "contextLength": 3_600,
            "forecastHorizon": 3_600,
            "variableCount": 1,
            "actionCount": 101,
            "scales": [60, 300],
            "tcnWidth": 16,
            "dilations": [1, 2, 4, 8, 16, 32],
            "fusionWidth": 24,
            "policyHiddenWidth": 24,
            "forecastRank": 4,
            "dropout": 0,
        }
        model = build_model(model_config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        training_fingerprint = "training-fingerprint"
        checkpoint = build_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=2,
            global_step=9,
            best_validation=0.7,
            best_epoch=2,
            stale_epochs=0,
            validation={"klDivergence": 0.7},
            model_parameters=parameter_count(model),
            dataset_fingerprint="dataset-fingerprint",
            model_config=model_config,
            plan_id="variant-resume-test",
            device=torch.device("cpu"),
            interrupted=False,
            training_config_fingerprint=training_fingerprint,
        )
        self.assertEqual(
            checkpoint["architectureContract"],
            TCN_CONTRACT,
        )
        with tempfile.TemporaryDirectory() as directory:
            file = (
                Path(directory)
                / "data"
                / "training"
                / "runs"
                / "variant-resume-test"
                / "checkpoints"
                / "last.json"
            )
            atomic_torch_save(checkpoint, file)
            resumed = load_resume_checkpoint(
                file,
                model,
                optimizer,
                scheduler,
                {"id": "variant-resume-test"},
                model_config,
                "dataset-fingerprint",
                parameter_count(model),
                torch.device("cpu"),
                training_config_fingerprint=training_fingerprint,
            )
            self.assertEqual(resumed, (3, 9, 0.7, 2, 0))
            checkpoint["architectureContract"] = ARCHITECTURE_CONTRACT
            atomic_torch_save(checkpoint, file)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": "variant-resume-test"},
                    model_config,
                    "dataset-fingerprint",
                    parameter_count(model),
                    torch.device("cpu"),
                    training_config_fingerprint=training_fingerprint,
                )

    def test_export_rejects_training_configuration_drift(self) -> None:
        model_config = {
            "variant": "multiscale_residual_mixer",
            "contextLength": 3_600,
            "forecastHorizon": 3_600,
            "variableCount": 1,
            "actionCount": 101,
        }
        training = {
            "lossWeights": {},
            "selectionMetric": "rolloutScore",
            "actionObjective": {},
        }
        plan = {
            "id": "residual-mixer-export-isolation",
            "model": model_config,
            "training": training,
        }
        checkpoint = {
            "planId": plan["id"],
            "architectureContract": RESIDUAL_MIXER_CONTRACT,
            "dataContract": DATA_CONTRACT,
            "modelConfig": model_config,
            "trainingConfigFingerprint": configuration_fingerprint(
                resolve_training_config(training)
            ),
        }
        validate_export_checkpoint(checkpoint, plan)
        plan["training"]["selectionMetric"] = "actionLoss"
        with self.assertRaisesRegex(ValueError, "does not match"):
            validate_export_checkpoint(checkpoint, plan)


if __name__ == "__main__":
    unittest.main()

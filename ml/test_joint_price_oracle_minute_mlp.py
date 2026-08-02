from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from joint_price_oracle_minute_mlp import (
    ARCHITECTURE_CONTRACT,
    FIXED_MINUTE_RETURN_SCALE,
    INPUT_CLOSE_COUNT,
    INPUT_FEATURE_COUNT,
    MINUTE_RETURN_COUNT,
    OUTPUT_ACTION_COUNT,
    SCALE_FEATURE_COUNT,
    MinuteReturnMlpConfig,
    MinuteReturnOracleMlp,
    causal_minute_log_returns,
    causal_minute_mlp_features,
)


def closes_from_log_returns(log_returns: torch.Tensor) -> torch.Tensor:
    if log_returns.ndim != 2 or log_returns.shape[1] != INPUT_CLOSE_COUNT - 1:
        raise ValueError("test log-return tensor has an invalid shape")
    initial = torch.full(
        (log_returns.shape[0], 1),
        math.log(100.0),
        dtype=log_returns.dtype,
        device=log_returns.device,
    )
    log_closes = torch.cat((
        initial,
        initial + log_returns.cumsum(dim=1),
    ), dim=1)
    return log_closes.exp().unsqueeze(-1)


class JointPriceOracleMinuteMlpTest(unittest.TestCase):
    def test_minute_aggregation_has_exact_nonoverlapping_alignment(self) -> None:
        second_returns = torch.zeros(2, INPUT_CLOSE_COUNT - 1)
        second_returns[0, 59] = 0.01
        second_returns[0, 60] = -0.02
        second_returns[0, 3_599] = 0.03
        second_returns[1] = torch.arange(
            INPUT_CLOSE_COUNT - 1,
            dtype=torch.float32,
        ) * 1e-8
        actual = causal_minute_log_returns(
            closes_from_log_returns(second_returns)
        )
        expected = second_returns.reshape(2, 60, 60).sum(dim=-1)
        self.assertEqual(tuple(actual.shape), (2, MINUTE_RETURN_COUNT))
        self.assertTrue(torch.allclose(actual, expected, atol=2e-5, rtol=1e-4))
        self.assertAlmostEqual(float(actual[0, 0]), 0.01, places=5)
        self.assertAlmostEqual(float(actual[0, 1]), -0.02, places=5)
        self.assertAlmostEqual(float(actual[0, -1]), 0.03, places=5)

    def test_final_close_only_changes_the_final_minute_features(self) -> None:
        base_returns = torch.linspace(-2e-5, 2e-5, INPUT_CLOSE_COUNT - 1) \
            .unsqueeze(0)
        changed_returns = base_returns.clone()
        changed_returns[:, -1] += 0.01
        base = causal_minute_log_returns(
            closes_from_log_returns(base_returns)
        )
        changed = causal_minute_log_returns(
            closes_from_log_returns(changed_returns)
        )
        self.assertTrue(torch.equal(base[:, :-1], changed[:, :-1]))
        self.assertFalse(torch.equal(base[:, -1:], changed[:, -1:]))

    def test_features_preserve_fixed_scale_and_are_price_scale_invariant(
        self,
    ) -> None:
        minute_returns = torch.linspace(-0.003, 0.004, 60).repeat_interleave(
            60
        ).unsqueeze(0) / 60
        closes = closes_from_log_returns(minute_returns)
        features, scale_features = causal_minute_mlp_features(closes)
        scaled_minutes = causal_minute_log_returns(closes) \
            / FIXED_MINUTE_RETURN_SCALE
        self.assertEqual(tuple(features.shape), (1, INPUT_FEATURE_COUNT))
        self.assertEqual(
            tuple(scale_features.shape),
            (1, SCALE_FEATURE_COUNT),
        )
        self.assertTrue(torch.allclose(
            features[:, :MINUTE_RETURN_COUNT],
            scaled_minutes,
            atol=1e-5,
            rtol=1e-5,
        ))
        scaled_features, scaled_scale_features = causal_minute_mlp_features(
            closes * 1_000.0
        )
        self.assertTrue(torch.allclose(
            features,
            scaled_features,
            atol=3e-3,
            rtol=3e-4,
        ))
        self.assertTrue(torch.allclose(
            scale_features,
            scaled_scale_features,
            atol=3e-3,
            rtol=3e-4,
        ))

    def test_default_model_is_dense_mid_size_and_outputs_only_logits(self) -> None:
        model = MinuteReturnOracleMlp().eval()
        closes = closes_from_log_returns(torch.randn(3, 3_600) * 2e-5)
        with torch.no_grad():
            logits = model(closes)
        self.assertEqual(tuple(logits.shape), (3, OUTPUT_ACTION_COUNT))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        self.assertEqual(model.architecture_contract, ARCHITECTURE_CONTRACT)
        self.assertGreaterEqual(model.parameter_count(), 2_000_000)
        self.assertLessEqual(model.parameter_count(), 8_000_000)

    def test_width_layers_and_dropout_are_configurable(self) -> None:
        config = MinuteReturnMlpConfig(
            hidden_width=48,
            layer_count=3,
            dropout=0.2,
            residual_gain=0.5,
        )
        model = MinuteReturnOracleMlp(config)
        self.assertEqual(model.config, config)
        self.assertEqual(len(model.blocks), 2)
        self.assertEqual(model.input_value_gate.output_width, 48)
        with self.assertRaisesRegex(ValueError, "either config"):
            MinuteReturnOracleMlp(config, hidden_width=64)
        for invalid in (
            {"hidden_width": 0},
            {"layer_count": 0},
            {"dropout": 1.0},
            {"residual_gain": 0.0},
        ):
            with self.assertRaises(ValueError):
                MinuteReturnMlpConfig(**invalid)

    def test_policy_loss_has_finite_gradients_through_closes_and_model(
        self,
    ) -> None:
        torch.manual_seed(27)
        model = MinuteReturnOracleMlp(
            hidden_width=64,
            layer_count=4,
            dropout=0,
        ).train()
        closes = closes_from_log_returns(
            torch.randn(4, 3_600) * 3e-5
        ).requires_grad_()
        target = torch.softmax(torch.randn(4, OUTPUT_ACTION_COUNT), dim=-1)
        logits = model(closes)
        loss = -(target * torch.log_softmax(logits, dim=-1)).sum(dim=-1).mean()
        loss.backward()
        self.assertTrue(bool(torch.isfinite(loss)))
        self.assertIsNotNone(closes.grad)
        self.assertTrue(bool(torch.isfinite(closes.grad).all()))
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
        self.assertGreater(
            sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None),
            0.0,
        )

    def test_input_contract_rejects_misaligned_or_invalid_closes(self) -> None:
        with self.assertRaisesRegex(ValueError, "shape"):
            causal_minute_log_returns(torch.ones(2, 3_600, 1))
        with self.assertRaisesRegex(ValueError, "shape"):
            causal_minute_log_returns(torch.ones(2, INPUT_CLOSE_COUNT, 2))
        with self.assertRaisesRegex(TypeError, "floating"):
            causal_minute_log_returns(torch.ones(
                1,
                INPUT_CLOSE_COUNT,
                1,
                dtype=torch.int64,
            ))
        invalid = torch.ones(1, INPUT_CLOSE_COUNT, 1)
        invalid[:, 100] = 0
        with self.assertRaisesRegex(ValueError, "positive"):
            causal_minute_log_returns(invalid)
        invalid[:, 100] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            causal_minute_log_returns(invalid)

    def test_onnx_logits_match_pytorch(self) -> None:
        torch.manual_seed(81)
        model = MinuteReturnOracleMlp(
            hidden_width=32,
            layer_count=3,
            dropout=0,
        ).eval()
        example = closes_from_log_returns(
            torch.randn(2, 3_600) * 2e-5
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "minute-return-mlp.onnx"
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
                expected = model(example).numpy()
            actual = ReferenceEvaluator(exported).run(
                ["action_logits"],
                {"closes": example.numpy()},
            )[0]
        self.assertTrue(np.isfinite(actual).all())
        self.assertLess(float(np.max(np.abs(expected - actual))), 1e-4)


if __name__ == "__main__":
    unittest.main()

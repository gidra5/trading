from __future__ import annotations

import unittest

import torch

from future_price_predictor import (
    CausalPatchTcnCore,
    ForecastNormalization,
    build_close_return_predictor,
    normalized_forecast_loss,
    parameter_count,
)


def normalization() -> ForecastNormalization:
    return ForecastNormalization(
        torch.linspace(-1e-5, 1e-5, 60),
        torch.linspace(5e-4, 8e-4, 60),
        torch.linspace(-2e-5, 2e-5, 60),
        torch.linspace(6e-4, 9e-4, 60),
    )


class FuturePricePredictorTest(unittest.TestCase):
    def test_both_compute_matched_families_train(self) -> None:
        definitions = (
            ("rlinear_dlinear", {"movingAverageKernel": 15}),
            (
                "causal_patch_tcn",
                {
                    "patchSize": 5,
                    "width": 24,
                    "dilations": [1, 2, 4, 4],
                    "kernelSize": 3,
                    "dropout": 0.05,
                },
            ),
        )
        counts: list[int] = []
        for architecture, config in definitions:
            model = build_close_return_predictor(
                architecture,
                normalization(),
                config,
            )
            history = torch.randn(11, 60) * 7e-4
            target = torch.randn(11, 60) * 7e-4
            prediction = model(history)
            self.assertEqual(prediction.shape, target.shape)
            loss = normalized_forecast_loss(
                prediction,
                target,
                torch.arange(1, 12, dtype=torch.float32),
                model.target_std,
                objective="huber",
                huber_delta=1.0,
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ))
            counts.append(parameter_count(model))
        self.assertEqual(counts, [7_320, 8_844])
        self.assertLess(max(counts) / min(counts), 1.25)

    def test_patch_tokens_are_strictly_causal(self) -> None:
        model = CausalPatchTcnCore(
            patch_size=5,
            width=12,
            dilations=(1, 2, 4),
            dropout=0,
        ).eval()
        original = torch.randn(3, 60)
        changed = original.clone()
        changed[:, 30:] = torch.randn_like(changed[:, 30:]) * 100
        with torch.inference_mode():
            first = model.encode_tokens(original)
            second = model.encode_tokens(changed)
        torch.testing.assert_close(first[:, :, :6], second[:, :, :6])
        self.assertFalse(torch.equal(first[:, :, 6:], second[:, :, 6:]))

    def test_training_scaling_is_embedded_and_not_batch_dependent(self) -> None:
        model = build_close_return_predictor(
            "causal_patch_tcn",
            normalization(),
            {
                "patchSize": 5,
                "width": 8,
                "dilations": [1],
                "kernelSize": 3,
                "dropout": 0,
            },
        ).eval()
        example = torch.randn(1, 60) * 1e-3
        companions = torch.randn(7, 60) * 0.1
        with torch.inference_mode():
            alone = model(example)
            together = model(torch.cat((example, companions)))[0:1]
        torch.testing.assert_close(alone, together)


if __name__ == "__main__":
    unittest.main()

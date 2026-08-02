from __future__ import annotations

import unittest

import numpy as np
import torch

from future_price_feature_screen import (
    BASE_CHANNEL_COUNT,
    FeatureNormalization,
    FeatureSpec,
    build_causal_features,
    build_feature_predictor,
    parameter_count,
)
from train_future_price_feature_screen import (
    EXAMPLE_SPAN_MS,
    aggregate_minute_base,
)
from train_future_price_predictor import (
    HOUR_MS,
    select_pair_segments,
    validate_predictor_split_disjointness,
)


def base_window(spec: FeatureSpec, batch: int = 3) -> np.ndarray:
    generator = np.random.default_rng(71)
    values = np.zeros(
        (batch, spec.required_base_minutes, BASE_CHANNEL_COUNT),
        dtype=np.float32,
    )
    values[:, :, 0] = generator.normal(0, 7e-4, values.shape[:2])
    values[:, :, 1] = generator.normal(0, 5e-4, values.shape[:2])
    values[:, :, 2:4] = generator.uniform(0, 4e-4, (*values.shape[:2], 2))
    values[:, :, 4] = generator.normal(2, 0.3, values.shape[:2])
    return values


def normalization(spec: FeatureSpec) -> FeatureNormalization:
    return FeatureNormalization(
        torch.zeros(360, spec.channel_count),
        torch.ones(360, spec.channel_count),
        torch.zeros(60),
        torch.ones(60),
    )


class FuturePriceFeatureScreenTest(unittest.TestCase):
    def test_additive_ma_bands_exactly_reconstruct_anchored_path(self) -> None:
        spec = FeatureSpec.from_config({
            "format": "close_ma_bands",
            "historyMinutes": 360,
            "maPeriods": [5, 15, 60],
        })
        features = build_causal_features(base_window(spec), spec)
        self.assertEqual(features.shape, (3, 360, 5))
        reconstructed_path = features[:, :, 1:].sum(axis=-1)
        expected_path = np.cumsum(features[:, :, 0], axis=-1)
        np.testing.assert_allclose(
            reconstructed_path,
            expected_path,
            rtol=2e-5,
            atol=2e-7,
        )

    def test_volume_residual_uses_only_prior_sixty_minutes(self) -> None:
        spec = FeatureSpec.from_config({
            "format": "close_volume_residual",
            "historyMinutes": 360,
            "volumeLookback": 60,
            "volumeEpsilon": 1e-6,
        })
        original = base_window(spec, batch=1)
        changed = original.copy()
        desired_start = spec.required_base_minutes - 360
        changed[:, desired_start + 100:, 4] += 20
        changed[:, desired_start + 100:, 5] = 1
        first = build_causal_features(original, spec)
        second = build_causal_features(changed, spec)
        np.testing.assert_array_equal(first[:, :100], second[:, :100])
        self.assertFalse(np.array_equal(first[:, 100:], second[:, 100:]))
        self.assertEqual(
            spec.identity_normalized_channels,
            ("volumeZeroMask",),
        )

    def test_minute_geometry_reconstructs_aggregated_ohlc(self) -> None:
        second = np.arange(86_400, dtype=np.float64)
        open_value = 100 + second * 1e-5
        close_value = open_value + np.sin(second / 17) * 2e-4
        high_value = np.maximum(open_value, close_value) + 3e-4
        low_value = np.minimum(open_value, close_value) - 4e-4
        volume = 1 + second % 7
        previous_close = np.full(86_400, 99.5, dtype=np.float64)
        base = aggregate_minute_base(
            previous_close,
            open_value,
            high_value,
            low_value,
            close_value,
            volume,
        )
        minute_open = open_value.reshape(1_440, 60)[:, 0]
        minute_close = close_value.reshape(1_440, 60)[:, -1]
        minute_high = high_value.reshape(1_440, 60).max(axis=1)
        minute_low = low_value.reshape(1_440, 60).min(axis=1)
        prior = np.concatenate((previous_close[-1:], minute_close[:-1]))
        reconstructed_open = prior * np.exp(base[:, 1])
        reconstructed_close = prior * np.exp(base[:, 0])
        reconstructed_high = np.maximum(
            reconstructed_open,
            reconstructed_close,
        ) * np.exp(base[:, 2])
        reconstructed_low = np.minimum(
            reconstructed_open,
            reconstructed_close,
        ) / np.exp(base[:, 3])
        np.testing.assert_allclose(
            reconstructed_open,
            minute_open,
            rtol=2e-7,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            reconstructed_close,
            minute_close,
            rtol=2e-7,
            atol=2e-5,
        )
        np.testing.assert_allclose(
            reconstructed_high,
            minute_high,
            rtol=3e-7,
            atol=3e-5,
        )
        np.testing.assert_allclose(
            reconstructed_low,
            minute_low,
            rtol=3e-7,
            atol=3e-5,
        )

    def test_models_cover_all_feature_formats(self) -> None:
        expected_parameters = {
            "close": (43_500, 33_228),
            "close_ma_bands": (216_780, 35_148),
            "close_ohlc_geometry": (173_460, 34_668),
            "close_volume_residual": (130_140, 34_188),
        }
        for feature_format, expected in expected_parameters.items():
            spec = FeatureSpec.from_config({"format": feature_format})
            models = (
                build_feature_predictor(
                    "rlinear_dlinear",
                    {"movingAverageKernel": 15},
                    spec,
                    normalization(spec),
                ),
                build_feature_predictor(
                    "causal_patch_tcn",
                    {
                        "patchSize": 10,
                        "width": 48,
                        "dilations": [1, 2, 4, 8, 4],
                        "kernelSize": 3,
                        "dropout": 0.05,
                    },
                    spec,
                    normalization(spec),
                ),
            )
            self.assertEqual(tuple(parameter_count(model) for model in models), expected)
            features = torch.randn(2, 360, spec.channel_count)
            for model in models:
                prediction = model(features)
                self.assertEqual(prediction.shape, (2, 60))
                prediction.square().mean().backward()
                self.assertTrue(all(
                    parameter.grad is None or torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                ))

    def test_patch_tokens_do_not_see_later_patches(self) -> None:
        spec = FeatureSpec.from_config({"format": "close_ohlc_geometry"})
        model = build_feature_predictor(
            "causal_patch_tcn",
            {
                "patchSize": 10,
                "width": 16,
                "dilations": [1, 2, 4],
                "kernelSize": 3,
                "dropout": 0,
            },
            spec,
            normalization(spec),
        ).eval()
        original = torch.randn(2, 360, 4)
        changed = original.clone()
        changed[:, 180:] = torch.randn_like(changed[:, 180:]) * 100
        with torch.inference_mode():
            first = model.core.encode_tokens(original)
            second = model.core.encode_tokens(changed)
        torch.testing.assert_close(first[:, :, :18], second[:, :, :18])
        self.assertFalse(torch.equal(first[:, :, 18:], second[:, :, 18:]))

    def test_patch_tide_preserves_positions_and_has_finite_gradients(self) -> None:
        spec = FeatureSpec.from_config({"format": "close"})
        model = build_feature_predictor(
            "patch_tide",
            {
                "patchSize": 10,
                "width": 24,
                "hiddenWidth": 96,
                "depth": 2,
                "dropout": 0.05,
            },
            spec,
            normalization(spec),
        )
        self.assertEqual(parameter_count(model), 231_204)
        history = torch.randn(3, 360, 1)
        patches = model.core.encode_patches(history)
        self.assertEqual(patches.shape, (3, 24, 36))
        prediction = model(history)
        self.assertEqual(prediction.shape, (3, 60))
        (prediction - torch.randn_like(prediction)).square().mean().backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_six_hour_selector_enforces_seven_hour_span(self) -> None:
        def source(split: str, start: int, count: int) -> dict:
            return {
                "split": split,
                "predictionTimeStart": start,
                "oracleTargetTimeStart": start - HOUR_MS,
                "count": count,
                "date": "1970-01-01",
                "featureRowOffset": start // 1_000,
                "featureRowStride": 1,
                "oracleRowOffset": start // 1_000 - 3_600,
                "oracleRowStride": 1,
            }
        manifest = {"shards": [
            source("train", 8 * HOUR_MS, 3_600),
            source("validation", 9 * HOUR_MS, 3_600),
            source("validation", 10 * HOUR_MS, 28_800),
        ]}
        selected = select_pair_segments(
            manifest,
            cross_split_purge_ms=EXAMPLE_SPAN_MS,
        )
        validate_predictor_split_disjointness(
            selected,
            example_span_ms=EXAMPLE_SPAN_MS,
        )
        self.assertEqual(
            selected["validation"][0].prediction_time_start,
            16 * HOUR_MS,
        )


if __name__ == "__main__":
    unittest.main()

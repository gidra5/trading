from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from future_price_resolution_screen import (
    ResolutionNormalization,
    ResolutionSpec,
    aggregate_minute_log_returns,
    build_resolution_examples,
    build_resolution_predictor,
    parameter_count,
)
from train_future_price_predictor import HOUR_MS
from train_future_price_resolution_screen import (
    MINUTE_MS,
    ResolutionMetricAccumulator,
    compute_training_normalization,
    compute_validation_baselines,
    select_resolution_segments,
)


def normalization(spec: ResolutionSpec) -> ResolutionNormalization:
    return ResolutionNormalization(
        torch.zeros(spec.history_steps),
        torch.ones(spec.history_steps),
        torch.zeros(spec.target_steps),
        torch.ones(spec.target_steps),
    )


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


class _Reporter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(event)


class _Dataset:
    def __init__(self, spec: ResolutionSpec) -> None:
        self.spec = spec
        self.values = {
            "train": (
                torch.stack((
                    torch.linspace(-0.01, 0.01, spec.history_steps),
                    torch.linspace(0.02, -0.01, spec.history_steps),
                )),
                torch.stack((
                    torch.linspace(-0.02, 0.03, spec.target_steps),
                    torch.linspace(0.01, -0.01, spec.target_steps),
                )),
                torch.tensor([2.0, 3.0]),
            ),
            "validation": (
                torch.full((2, spec.history_steps), 100.0),
                torch.stack((
                    torch.linspace(0.04, -0.03, spec.target_steps),
                    torch.linspace(-0.02, 0.05, spec.target_steps),
                )),
                torch.tensor([1.0, 4.0]),
            ),
        }

    def logical_count(self, split: str) -> int:
        return int(self.values[split][2].sum())

    def iter_batches(self, split: str, batch_size: int, **_: object):
        history, target, weights = self.values[split]
        for start in range(0, weights.shape[0], batch_size):
            end = min(start + batch_size, weights.shape[0])
            yield history[start:end], target[start:end], weights[start:end]


class FuturePriceResolutionScreenTest(unittest.TestCase):
    def test_block_sums_losslessly_reconstruct_resolution_closes(self) -> None:
        generator = np.random.default_rng(17)
        minute = generator.normal(0, 0.001, (3, 60)).astype(np.float64)
        aggregate = aggregate_minute_log_returns(minute, 5)
        expected = np.log(
            np.exp(np.cumsum(minute, axis=-1))[:, 4::5]
        )
        np.testing.assert_allclose(
            np.cumsum(aggregate, axis=-1),
            expected,
            rtol=2e-6,
            atol=2e-8,
        )
        np.testing.assert_allclose(
            aggregate,
            minute.reshape(3, 12, 5).sum(axis=-1),
            rtol=2e-7,
            atol=2e-9,
        )

    def test_history_and_target_aggregation_have_no_gap_or_overlap(self) -> None:
        spec = ResolutionSpec.from_config({
            "historyMinutes": 360,
            "candleMinutes": 5,
            "targetSteps": 3,
        })
        history_minute = np.arange(-360, 0, dtype=np.float32).reshape(1, 360)
        future_hour = np.arange(0, 60, dtype=np.float32).reshape(1, 60)
        history, target = build_resolution_examples(
            history_minute,
            future_hour,
            spec,
        )
        np.testing.assert_array_equal(
            history,
            history_minute.reshape(1, 72, 5).sum(axis=-1),
        )
        np.testing.assert_array_equal(
            target,
            future_hour[:, :15].reshape(1, 3, 5).sum(axis=-1),
        )
        self.assertEqual(history[0, -1], sum(range(-5, 0)))
        self.assertEqual(target[0, 0], sum(range(0, 5)))

    def test_resolution_selector_uses_history_plus_actual_horizon_purge(self) -> None:
        spec = ResolutionSpec.from_config({
            "historyMinutes": 360,
            "candleMinutes": 5,
            "targetSteps": 1,
        })
        manifest = {"shards": [
            source("train", 8 * HOUR_MS, 3_600),
            source("validation", 9 * HOUR_MS, 3_600),
            source("validation", 16 * HOUR_MS, 28_800),
            source("test", 30 * HOUR_MS, 3_600),
        ]}
        selected = select_resolution_segments(manifest, spec)
        self.assertEqual(set(selected), {"train", "validation"})
        train_end = selected["train"][-1].prediction_time_end
        validation_start = selected["validation"][0].prediction_time_start
        self.assertGreater(
            validation_start,
            train_end + spec.example_span_minutes * MINUTE_MS,
        )
        # The one-hour source assignment is shifted to the actual five-minute
        # target end before the strict 6h05m interval is checked.
        self.assertEqual(
            selected["train"][0].prediction_time_start,
            8 * HOUR_MS - 55 * MINUTE_MS,
        )

    def test_compute_matched_models_have_finite_forward_and_backward(self) -> None:
        cases = (
            ({"historyMinutes": 360, "candleMinutes": 5, "targetSteps": 1}, 3, 26_593),
            ({"historyMinutes": 360, "candleMinutes": 5, "targetSteps": 3}, 3, 26_787),
            ({"historyMinutes": 360, "candleMinutes": 5, "targetSteps": 12}, 3, 27_660),
            ({"historyMinutes": 1440, "candleMinutes": 15, "targetSteps": 4}, 4, 26_932),
        )
        for config, patch_size, expected_parameters in cases:
            spec = ResolutionSpec.from_config(config)
            model = build_resolution_predictor(
                "causal_patch_tcn",
                {
                    "patchSize": patch_size,
                    "width": 48,
                    "dilations": [1, 2, 4, 8, 4],
                    "kernelSize": 3,
                    "dropout": 0.05,
                },
                spec,
                normalization(spec),
            )
            self.assertEqual(parameter_count(model), expected_parameters)
            self.assertEqual(model.core.token_count, 24)
            history = torch.randn(4, spec.history_steps)
            prediction = model(history)
            self.assertEqual(prediction.shape, (4, spec.target_steps))
            (prediction - torch.randn_like(prediction)).square().mean().backward()
            self.assertTrue(all(
                parameter.grad is None or torch.isfinite(parameter.grad).all()
                for parameter in model.parameters()
            ))

    def test_primary_patch_tide_is_modest_and_finite(self) -> None:
        spec = ResolutionSpec.from_config({
            "historyMinutes": 360,
            "candleMinutes": 5,
            "targetSteps": 1,
        })
        model = build_resolution_predictor(
            "patch_tide",
            {
                "patchSize": 3,
                "width": 16,
                "hiddenWidth": 64,
                "depth": 2,
                "dropout": 0.05,
            },
            spec,
            normalization(spec),
        )
        self.assertEqual(parameter_count(model), 75_905)
        prediction = model(torch.randn(3, spec.history_steps))
        self.assertEqual(prediction.shape, (3, 1))
        (prediction - 1).square().mean().backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))

    def test_patch_tokens_cannot_see_later_history_patches(self) -> None:
        spec = ResolutionSpec.from_config({
            "historyMinutes": 360,
            "candleMinutes": 5,
            "targetSteps": 1,
        })
        model = build_resolution_predictor(
            "causal_patch_tcn",
            {
                "patchSize": 3,
                "width": 16,
                "dilations": [1, 2, 4],
                "kernelSize": 3,
                "dropout": 0,
            },
            spec,
            normalization(spec),
        ).eval()
        original = torch.randn(2, spec.history_steps)
        changed = original.clone()
        changed[:, 36:] = torch.randn_like(changed[:, 36:]) * 100
        with torch.inference_mode():
            first = model.core.encode_tokens(original)
            second = model.core.encode_tokens(changed)
        torch.testing.assert_close(first[:, :, :12], second[:, :, :12])
        self.assertFalse(torch.equal(first[:, :, 12:], second[:, :, 12:]))

    def test_normalization_is_train_only_and_baselines_are_exact(self) -> None:
        spec = ResolutionSpec.from_config({
            "historyMinutes": 360,
            "candleMinutes": 5,
            "targetSteps": 3,
        })
        dataset = _Dataset(spec)
        reporter = _Reporter()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            actual = compute_training_normalization(
                dataset,
                root / "normalization.npz",
                fingerprint="resolution-data",
                batch_size=1,
                reporter=reporter,
            )
            baselines = compute_validation_baselines(
                dataset,
                actual,
                root / "baselines.json",
                data_fingerprint="resolution-data",
                batch_size=1,
                objective="mse",
                huber_delta=1.0,
                reporter=reporter,
            )
            repeated = compute_validation_baselines(
                dataset,
                actual,
                root / "baselines.json",
                data_fingerprint="resolution-data",
                batch_size=2,
                objective="mse",
                huber_delta=1.0,
                reporter=reporter,
            )
        train_target = dataset.values["train"][1]
        train_weights = dataset.values["train"][2]
        expected_mean = (
            train_target * train_weights[:, None]
        ).sum(dim=0) / train_weights.sum()
        torch.testing.assert_close(actual.target_mean, expected_mean)
        self.assertTrue(bool((actual.input_mean.abs() < 1).all()))
        self.assertEqual(baselines, repeated)
        self.assertEqual(
            baselines["zeroReturn"]["examples"],
            dataset.logical_count("validation"),
        )
        direct = ResolutionMetricAccumulator(
            actual,
            huber_delta=1.0,
            target_steps=spec.target_steps,
            device=torch.device("cpu"),
        )
        _, target, weights = dataset.values["validation"]
        direct.add(torch.zeros_like(target), target, weights)
        self.assertAlmostEqual(
            baselines["zeroReturn"]["normalizedMse"],
            direct.result()["normalizedMse"],
        )


if __name__ == "__main__":
    unittest.main()

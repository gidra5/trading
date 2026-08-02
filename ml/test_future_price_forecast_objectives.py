from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from future_price_feature_screen import FeatureNormalization
from future_price_forecast_objectives import (
    ForecastObjectiveStatistics,
    forecast_objective_loss,
    multiscale_decompose,
    multiscale_reconstruct,
    objective_for_epoch,
    target_structure,
)
from train_future_price_feature_screen import (
    compute_validation_structural_baselines,
    compute_training_objective_statistics,
    evaluate,
    validation_baselines_for_objective,
)


def unit_statistics() -> ForecastObjectiveStatistics:
    return ForecastObjectiveStatistics(
        path_std=torch.ones(60),
        hour_std=torch.ones(1),
        quarter_hour_contrast_std=torch.ones(4),
        five_minute_contrast_std=torch.ones(12),
        minute_residual_std=torch.ones(60),
    )


def objective(
    objective_type: str,
    weights: dict[str, float],
    *,
    loss: str = "mse",
) -> dict:
    return {
        "type": objective_type,
        "loss": loss,
        "huberDelta": 1.0,
        "weights": weights,
    }


class _Reporter:
    def __init__(self) -> None:
        self.events: list[dict] = []

    def emit(self, event: dict) -> None:
        self.events.append(event)


class _TargetDataset:
    def __init__(self, targets: torch.Tensor, weights: torch.Tensor) -> None:
        self.targets = targets
        self.weights = weights

    def logical_count(self, split: str) -> int:
        if split not in {"train", "validation"}:
            raise ValueError(split)
        return int(self.weights.sum())

    def iter_batches(self, split: str, batch_size: int, **_: object):
        if split not in {"train", "validation"}:
            raise ValueError(split)
        for start in range(0, self.targets.shape[0], batch_size):
            end = min(start + batch_size, self.targets.shape[0])
            yield (
                torch.empty(end - start, 360, 1),
                self.targets[start:end],
                self.weights[start:end],
            )


class FuturePriceForecastObjectivesTest(unittest.TestCase):
    def test_multiscale_decomposition_is_exact_and_hierarchical(self) -> None:
        generator = torch.Generator().manual_seed(19)
        returns = torch.randn(7, 60, generator=generator, dtype=torch.float64)
        components = multiscale_decompose(returns)
        torch.testing.assert_close(
            multiscale_reconstruct(components),
            returns,
            rtol=1e-12,
            atol=1e-12,
        )
        torch.testing.assert_close(
            components["quarterHourContrast"].sum(dim=-1),
            torch.zeros(7, dtype=torch.float64),
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            components["fiveMinuteContrast"].reshape(7, 4, 3).sum(dim=-1),
            torch.zeros(7, 4, dtype=torch.float64),
            rtol=0,
            atol=1e-12,
        )
        torch.testing.assert_close(
            components["minuteResidual"].reshape(7, 12, 5).sum(dim=-1),
            torch.zeros(7, 12, dtype=torch.float64),
            rtol=0,
            atol=1e-12,
        )

    def test_block_alignment_matches_future_return_boundaries(self) -> None:
        returns = torch.arange(1, 61, dtype=torch.float64).reshape(1, 60)
        components = multiscale_decompose(returns)
        five_sums = returns.reshape(1, 12, 5).sum(dim=-1)
        quarter_sums = returns.reshape(1, 4, 15).sum(dim=-1)
        torch.testing.assert_close(
            components["hour"],
            returns.sum(dim=-1, keepdim=True),
        )
        torch.testing.assert_close(
            components["quarterHourContrast"] + components["hour"] / 4,
            quarter_sums,
        )
        torch.testing.assert_close(
            components["fiveMinuteContrast"]
            + quarter_sums.repeat_interleave(3, dim=-1) / 3,
            five_sums,
        )
        self.assertEqual(float(target_structure(returns)["path"][0, 14]), 120.0)

    def test_cumulative_path_objective_is_variance_normalized(self) -> None:
        prediction = torch.ones(2, 60)
        target = torch.zeros_like(prediction)
        weights = torch.tensor([1.0, 3.0])
        statistics = ForecastObjectiveStatistics(
            path_std=torch.arange(1, 61, dtype=torch.float32),
            hour_std=torch.ones(1),
            quarter_hour_contrast_std=torch.ones(4),
            five_minute_contrast_std=torch.ones(12),
            minute_residual_std=torch.ones(60),
        )
        loss, metrics, stage = forecast_objective_loss(
            prediction,
            target,
            weights,
            torch.full((60,), 2.0),
            statistics,
            objective("cumulative_path", {"path": 2.0, "return": 1.0}),
            epoch=1,
            validation=False,
        )
        self.assertEqual(stage, "cumulative_path")
        self.assertAlmostEqual(float(metrics["normalizedPathMse"]), 1.0)
        self.assertAlmostEqual(float(metrics["normalizedReturnMse"]), 0.25)
        self.assertAlmostEqual(float(loss), 0.75)

    def test_multiscale_loss_uses_explicit_normalized_component_weights(self) -> None:
        prediction = torch.zeros(2, 60)
        prediction[0, :15] = 0.1
        prediction[1, 30:35] = -0.2
        target = torch.zeros_like(prediction)
        weights = torch.tensor([2.0, 1.0])
        config = objective(
            "multiscale_reconstruction",
            {
                "hour": 1.0,
                "quarterHourContrast": 0.75,
                "fiveMinuteContrast": 0.5,
                "minuteResidual": 0.25,
                "path": 0.5,
                "return": 0.1,
            },
        )
        loss, metrics, _ = forecast_objective_loss(
            prediction,
            target,
            weights,
            torch.ones(60),
            unit_statistics(),
            config,
            epoch=1,
            validation=False,
        )
        numerator = sum(
            float(metrics[
                "normalized" + name[0].upper() + name[1:] + "Mse"
            ]) * component_weight
            for name, component_weight in config["weights"].items()
        )
        self.assertAlmostEqual(
            float(loss),
            numerator / sum(config["weights"].values()),
            places=6,
        )

    def test_curriculum_validation_objective_never_changes(self) -> None:
        coarse = objective("cumulative_path", {"path": 1.0})
        medium = objective(
            "multiscale_reconstruction",
            {"hour": 1.0, "quarterHourContrast": 1.0},
        )
        full = objective(
            "multiscale_reconstruction",
            {
                "hour": 1.0,
                "quarterHourContrast": 1.0,
                "fiveMinuteContrast": 1.0,
                "minuteResidual": 1.0,
            },
        )
        curriculum = {
            "type": "curriculum",
            "stages": [
                {"throughEpoch": 4, "objective": coarse},
                {"throughEpoch": 12, "objective": medium},
                {"throughEpoch": 200, "objective": full},
            ],
            "validationObjective": full,
        }
        self.assertIs(objective_for_epoch(
            curriculum, 2, validation=False
        )[0], coarse)
        self.assertIs(objective_for_epoch(
            curriculum, 8, validation=False
        )[0], medium)
        first_validation = objective_for_epoch(
            curriculum, 1, validation=True
        )
        late_validation = objective_for_epoch(
            curriculum, 199, validation=True
        )
        self.assertIs(first_validation[0], full)
        self.assertIs(late_validation[0], full)
        self.assertEqual(first_validation[1], "validationObjective")

    def test_objective_statistics_use_weighted_training_targets_and_cache(self) -> None:
        targets = torch.stack((
            torch.linspace(-0.01, 0.02, 60),
            torch.linspace(0.03, -0.02, 60),
            torch.sin(torch.arange(60) / 7) * 0.01,
        ))
        weights = torch.tensor([2.0, 1.0, 3.0])
        dataset = _TargetDataset(targets, weights)
        reporter = _Reporter()
        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "statistics.npz"
            actual = compute_training_objective_statistics(
                dataset,
                cache,
                fingerprint="training-corpus",
                batch_size=2,
                reporter=reporter,
            )
            repeated = compute_training_objective_statistics(
                dataset,
                cache,
                fingerprint="training-corpus",
                batch_size=3,
                reporter=reporter,
            )
        structure = target_structure(targets.to(dtype=torch.float64))
        expected: dict[str, torch.Tensor] = {}
        normalized_weights = weights.to(dtype=torch.float64) / weights.sum()
        for name, values in structure.items():
            mean = (values * normalized_weights[:, None]).sum(dim=0)
            variance = (
                values.square() * normalized_weights[:, None]
            ).sum(dim=0) - mean.square()
            expected[name] = variance.clamp_min(1e-12).sqrt().float()
        torch.testing.assert_close(actual.path_std, expected["path"])
        torch.testing.assert_close(actual.hour_std, expected["hour"])
        torch.testing.assert_close(
            actual.quarter_hour_contrast_std,
            expected["quarterHourContrast"],
        )
        torch.testing.assert_close(
            actual.five_minute_contrast_std,
            expected["fiveMinuteContrast"],
        )
        torch.testing.assert_close(
            actual.minute_residual_std,
            expected["minuteResidual"],
        )
        self.assertEqual(actual.as_json(), repeated.as_json())
        self.assertEqual([event["hit"] for event in reporter.events], [False, True])

    def test_full_validation_baselines_match_direct_structural_losses(self) -> None:
        generator = torch.Generator().manual_seed(83)
        targets = torch.randn(5, 60, generator=generator) * 0.01
        weights = torch.tensor([1.0, 3.0, 2.0, 4.0, 1.0])
        dataset = _TargetDataset(targets, weights)
        target_mean = torch.linspace(-0.002, 0.003, 60)
        normalization = FeatureNormalization(
            torch.zeros(360, 1),
            torch.ones(360, 1),
            target_mean,
            torch.full((60,), 0.02),
        )
        statistics = unit_statistics()
        reporter = _Reporter()
        with tempfile.TemporaryDirectory() as directory:
            first = compute_validation_structural_baselines(
                dataset,
                normalization,
                statistics,
                Path(directory),
                data_fingerprint="validation-corpus",
                batch_size=2,
                huber_delta=1.0,
                reporter=reporter,
            )
            repeated = compute_validation_structural_baselines(
                dataset,
                normalization,
                statistics,
                Path(directory),
                data_fingerprint="validation-corpus",
                batch_size=5,
                huber_delta=1.0,
                reporter=reporter,
            )
        full_objective = objective(
            "multiscale_reconstruction",
            {
                "hour": 1.0,
                "quarterHourContrast": 0.75,
                "fiveMinuteContrast": 0.5,
                "minuteResidual": 0.25,
                "path": 0.5,
                "return": 0.1,
            },
            loss="huber",
        )
        surfaced = validation_baselines_for_objective(first, full_objective)
        for forecast_name, prediction in {
            "zeroReturn": torch.zeros_like(targets),
            "trainingMean": target_mean.unsqueeze(0).expand_as(targets),
        }.items():
            direct_loss, direct_metrics, _ = forecast_objective_loss(
                prediction,
                targets,
                weights,
                normalization.target_std,
                statistics,
                full_objective,
                epoch=1,
                validation=True,
            )
            self.assertEqual(
                surfaced[forecast_name]["examples"],
                int(weights.sum()),
            )
            for metric, expected in direct_metrics.items():
                self.assertAlmostEqual(
                    surfaced[forecast_name][metric],
                    float(expected),
                    places=6,
                )
            self.assertAlmostEqual(
                surfaced[forecast_name]["forecastObjectiveLoss"],
                float(direct_loss),
                places=6,
            )
        self.assertEqual(first, repeated)
        self.assertEqual([event["hit"] for event in reporter.events], [False, True])

        class ZeroModel(torch.nn.Module):
            def forward(self, features: torch.Tensor) -> torch.Tensor:
                return torch.zeros(features.shape[0], 60)

        validation = evaluate(
            ZeroModel(),
            dataset,
            normalization,
            batch_size=2,
            device=torch.device("cpu"),
            amp_dtype=torch.float32,
            objective="huber",
            huber_delta=1.0,
            forecast_objective=full_objective,
            objective_statistics=statistics,
            epoch=1,
            validation_baselines=surfaced,
        )
        self.assertEqual(validation["structuralBaselines"], surfaced)
        self.assertIn("rawMse", validation)
        self.assertIn("cumulativePathRmse", validation)


if __name__ == "__main__":
    unittest.main()

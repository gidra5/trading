from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch import Tensor, nn

from diagnose_joint_price_oracle_overfit import (
    FixedTrainingSubset,
    OverfitSettings,
    chronological_training_head,
    evaluate_fixed_subset,
    fixed_subset_source_switch_fraction,
    materialize_fixed_training_subset,
    run_fixed_subset_overfit,
)
from joint_price_oracle import JointPriceOracleOutput
from train_joint_price_oracle import CausalSegment


class TinyJointModel(nn.Module):
    def __init__(self, context: int, horizon: int, actions: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(context, 16)
        self.policy = nn.Linear(16, actions)
        self.forecast = nn.Linear(16, horizon)

    def forward_with_forecast(self, closes: Tensor) -> JointPriceOracleOutput:
        normalized = closes.squeeze(-1) / 100.0 - 1.0
        hidden = torch.tanh(self.encoder(normalized))
        logits = self.policy(hidden)
        log_movements = (0.02 * torch.tanh(self.forecast(hidden))).unsqueeze(-1)
        predicted = closes[:, -1:, :] * torch.exp(log_movements)
        mean = hidden.mean(dim=-1)
        variance = (hidden - mean[:, None]).square().mean(dim=-1)
        return JointPriceOracleOutput(
            policy_logits=logits,
            predicted_closes=predicted,
            predicted_log_movements=log_movements,
            predicted_movements=torch.expm1(log_movements),
            soft_layer_norm_mean=mean.square().mean(),
            soft_layer_norm_variance=(variance - 1).square().mean(),
        )


class RecordingDataset:
    def __init__(self, batches: list[tuple[Tensor, Tensor, Tensor]]) -> None:
        self.batches = batches
        self.requested_splits: list[str] = []

    def iter_batches(
        self,
        split: str,
        _batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        maximum_batches: int | None,
    ):
        self.requested_splits.append(split)
        if split != "train" or shuffle or seed != 0 or maximum_batches is not None:
            raise AssertionError("diagnostic requested non-training data")
        yield from self.batches


class JointPriceOracleOverfitDiagnosticTest(unittest.TestCase):
    def test_chronological_head_never_selects_other_splits(self) -> None:
        segments = [
            CausalSegment("train", 999, 5, Path("a"), 3, 60_000),
            CausalSegment("train", 300_999, 5, Path("b"), 0, 60_000),
        ]
        selected = chronological_training_head(segments, 7)
        self.assertEqual([segment.count for segment in selected], [5, 2])
        self.assertEqual(selected[1].target_file, Path("b"))
        with self.assertRaisesRegex(ValueError, "only train"):
            chronological_training_head([
                CausalSegment("validation", 999, 2, Path("v"), 0),
            ], 1)

    def test_materialization_reads_only_train_and_builds_teacher_state(self) -> None:
        grid = np.array([-1.0, 0.0, 1.0], dtype=np.float64)
        first_targets = torch.tensor([
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ])
        second_targets = torch.tensor([[1.0, 0.0, 0.0]])
        batches = [
            (
                torch.full((2, 4, 1), 100.0),
                torch.full((2, 2, 1), 101.0),
                first_targets,
            ),
            (
                torch.full((1, 4, 1), 100.0),
                torch.full((1, 2, 1), 99.0),
                second_targets,
            ),
        ]
        dataset = RecordingDataset(batches)
        segments = [
            CausalSegment("train", 999, 2, Path("a"), 0, 60_000),
            CausalSegment("train", 500_999, 1, Path("b"), 0, 60_000),
        ]
        subset = materialize_fixed_training_subset(
            dataset,  # type: ignore[arg-type]
            segments,
            grid,
            friction=0.0,
            temperature=1.0,
        )
        self.assertEqual(dataset.requested_splits, ["train"])
        self.assertEqual(subset.count, 3)
        self.assertEqual(subset.reset_mask.tolist(), [True, False, True])
        self.assertTrue(torch.equal(
            subset.teacher_current_exposures,
            torch.tensor([0.0, 1.0, 0.0]),
        ))

    @staticmethod
    def learnable_subset() -> tuple[FixedTrainingSubset, Tensor]:
        generator = torch.Generator().manual_seed(7)
        examples, context, horizon, actions = 16, 8, 4, 5
        signal = torch.linspace(-1.0, 1.0, examples)
        closes = 100 + signal[:, None] * torch.linspace(
            0.1,
            1.0,
            context,
        )[None, :]
        closes += torch.randn(
            examples,
            context,
            generator=generator,
        ) * 0.005
        future = closes[:, -1:] + signal[:, None] * torch.linspace(
            0.1,
            0.4,
            horizon,
        )[None, :]
        targets = torch.zeros(examples, actions)
        target_indexes = torch.where(
            signal < -0.2,
            torch.zeros(examples, dtype=torch.long),
            torch.where(
                signal > 0.2,
                torch.full((examples,), actions - 1, dtype=torch.long),
                torch.full((examples,), actions // 2, dtype=torch.long),
            ),
        )
        targets.scatter_(1, target_indexes[:, None], 1.0)
        grid = torch.linspace(-1.0, 1.0, actions)
        from joint_price_oracle_actions import greedy_teacher_rollout_numpy
        reset = np.zeros(examples, dtype=np.bool_)
        reset[0] = True
        rollout = greedy_teacher_rollout_numpy(
            targets.numpy(),
            grid.numpy(),
            reset_mask=reset,
            friction=0.0,
            temperature=1.0,
        )
        return FixedTrainingSubset(
            input_closes=closes.unsqueeze(-1),
            future_closes=future.unsqueeze(-1),
            target_policy=targets,
            teacher_current_exposures=torch.from_numpy(
                rollout.current_exposures.astype(np.float32)
            ),
            reset_mask=reset,
        ), grid

    def test_bounded_overfit_reports_loss_and_action_improvement(self) -> None:
        subset, grid = self.learnable_subset()
        training = {
            "mixedPrecision": "float32",
            "forecastHuberDelta": 1.0,
            "volatilityFloor": 1e-5,
            "lossWeights": {
                "policyCrossEntropy": 1.0,
                "conditionedPolicyCrossEntropy": 0.0,
                "forecast": 0.05,
                "softLayerNorm": 0.0,
            },
        }
        action_objective = {
            "lossWeight": 1.0,
            "targetSwitchFraction": 0.5,
            "rankingMargin": 0.1,
            "weights": {
                "hardAction": 1.0,
                "ranking": 0.5,
                "direction": 0.5,
                "conditionalKl": 0.1,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            before = set(Path(directory).iterdir())
            report = run_fixed_subset_overfit(
                TinyJointModel(8, 4, 5),
                subset,
                grid,
                training,
                action_objective,
                OverfitSettings(
                    steps=80,
                    batch_size=16,
                    learning_rate=0.02,
                    weight_decay=0.0,
                    gradient_clip=10.0,
                    minimum_relative_improvement=0.1,
                    seed=11,
                    device=torch.device("cpu"),
                ),
                friction=0.0,
                temperature=1.0,
            )
            self.assertEqual(before, set(Path(directory).iterdir()))
        self.assertGreater(report["relativeLossImprovement"], 0.1)
        self.assertLess(
            report["final"]["components"]["loss"],
            report["initial"]["components"]["loss"],
        )
        self.assertIn("transitionF1", report["final"]["actions"])
        self.assertIn("pathDirectionalAgreement", report["final"]["actions"])

    def test_global_weighting_uses_fixed_subset_switch_fraction(self) -> None:
        subset, grid = self.learnable_subset()
        training = {
            "mixedPrecision": "float32",
            "forecastHuberDelta": 1.0,
            "volatilityFloor": 1e-5,
            "lossWeights": {
                "policyCrossEntropy": 1.0,
                "conditionedPolicyCrossEntropy": 0.0,
                "forecast": 0.05,
                "softLayerNorm": 0.0,
            },
        }
        action_objective = {
            "lossWeight": 1.0,
            "targetSwitchFraction": 0.5,
            "rankingMargin": 0.1,
            "switchWeighting": "global",
            "weights": {
                "hardAction": 1.0,
                "ranking": 0.5,
                "direction": 0.5,
                "conditionalKl": 0.1,
            },
            "executionPolicy": {
                "version": 2,
                "maximumLeverage": 100.0,
                "minimumConfidence": 0.05,
                "confidenceExposurePower": 0.0,
                "confidenceLeverageFloor": 0.75,
            },
        }
        source_fraction = fixed_subset_source_switch_fraction(
            subset,
            grid,
            action_objective,
            friction=0.0,
            temperature=1.0,
        )
        self.assertIsNotNone(source_fraction)
        assert source_fraction is not None
        self.assertGreater(source_fraction, 0.0)
        self.assertLess(source_fraction, 1.0)

        # This used to fail before the first batch because the diagnostic did
        # not supply the source fraction required by global weighting.
        report = evaluate_fixed_subset(
            TinyJointModel(8, 4, 5),
            subset,
            grid,
            training,
            action_objective,
            device=torch.device("cpu"),
            batch_size=3,
            friction=0.0,
            temperature=1.0,
        )
        self.assertTrue(np.isfinite(report["components"]["loss"]))
        self.assertIn("signedTransitionF1", report["actions"])

    def test_v2_action_report_uses_same_teacher_switches_as_loss(self) -> None:
        from joint_price_oracle_actions import greedy_teacher_rollout_numpy

        grid = torch.tensor([-200.0, -100.0, 0.0, 100.0, 200.0])
        targets = torch.tensor([
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.21, 0.20, 0.20, 0.20, 0.19],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.19, 0.20, 0.20, 0.20, 0.21],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.21, 0.20, 0.20, 0.20, 0.19],
        ])
        execution_policy = {
            "version": 2,
            "maximumLeverage": 100.0,
            "minimumConfidence": 0.05,
            "confidenceExposurePower": 0.0,
            "confidenceLeverageFloor": 0.75,
        }
        reset_mask = np.array([True, False, False, False, False, False])
        expected = greedy_teacher_rollout_numpy(
            targets.numpy(),
            grid.numpy(),
            reset_mask=reset_mask,
            friction=0.0,
            temperature=1.0,
            execution_policy=execution_policy,
        )
        raw = greedy_teacher_rollout_numpy(
            targets.numpy(),
            grid.numpy(),
            reset_mask=reset_mask,
            friction=0.0,
            temperature=1.0,
        )
        expected_switches = int(expected.switch_labels.sum())
        self.assertNotEqual(expected_switches, int(raw.switch_labels.sum()))
        subset = FixedTrainingSubset(
            input_closes=torch.full((6, 8, 1), 100.0),
            future_closes=torch.full((6, 4, 1), 100.0),
            target_policy=targets,
            teacher_current_exposures=torch.from_numpy(
                expected.current_exposures.astype(np.float32)
            ),
            reset_mask=reset_mask,
        )
        report = evaluate_fixed_subset(
            TinyJointModel(8, 4, 5),
            subset,
            grid,
            {
                "mixedPrecision": "float32",
                "forecastHuberDelta": 1.0,
                "volatilityFloor": 1e-5,
                "lossWeights": {
                    "policyCrossEntropy": 1.0,
                    "conditionedPolicyCrossEntropy": 0.0,
                    "forecast": 0.05,
                    "softLayerNorm": 0.0,
                },
            },
            {
                "lossWeight": 1.0,
                "targetSwitchFraction": 0.5,
                "rankingMargin": 0.1,
                "switchWeighting": "global",
                "weights": {
                    "hardAction": 1.0,
                    "ranking": 0.5,
                    "direction": 0.5,
                    "conditionalKl": 0.1,
                },
                "executionPolicy": execution_policy,
            },
            device=torch.device("cpu"),
            batch_size=2,
            friction=0.0,
            temperature=1.0,
        )
        self.assertEqual(report["actions"]["targetSwitches"], expected_switches)
        self.assertAlmostEqual(
            report["components"]["actionSwitchRate"],
            expected_switches / subset.count,
        )

    def test_nonfinite_input_fails_sanity_gate(self) -> None:
        subset, grid = self.learnable_subset()
        bad = FixedTrainingSubset(
            input_closes=subset.input_closes.clone(),
            future_closes=subset.future_closes,
            target_policy=subset.target_policy,
            teacher_current_exposures=subset.teacher_current_exposures,
            reset_mask=subset.reset_mask,
        )
        bad.input_closes[0, 0, 0] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "non-finite"):
            run_fixed_subset_overfit(
                TinyJointModel(8, 4, 5),
                bad,
                grid,
                {
                    "mixedPrecision": "float32",
                    "forecastHuberDelta": 1.0,
                    "volatilityFloor": 1e-5,
                    "lossWeights": {
                        "policyCrossEntropy": 1.0,
                        "forecast": 1.0,
                        "softLayerNorm": 0.0,
                    },
                },
                {},
                OverfitSettings(
                    steps=1,
                    batch_size=16,
                    learning_rate=0.01,
                    weight_decay=0.0,
                    gradient_clip=10.0,
                    minimum_relative_improvement=0.0,
                    seed=1,
                    device=torch.device("cpu"),
                ),
                friction=0.0,
                temperature=1.0,
            )


if __name__ == "__main__":
    unittest.main()

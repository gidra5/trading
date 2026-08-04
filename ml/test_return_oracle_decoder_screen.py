from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from return_oracle_decoder_screen import (  # noqa: E402
    FINAL_DATASET_COUNTS,
    LEARNED_RADIUS_ARCHITECTURE,
    LearnedRadiusShrinkingDecoder,
    next_equal_entropy_step_training_temperature,
    next_stale_learning_rate,
    PRODUCTION_ORACLE_TEMPERATURE,
    production_entropy_confidence_weights,
    ResidualGluDecoder,
    canonical_plan_fingerprint,
    smooth_oracle_probabilities,
    target_temperature,
    training_temperature,
    validate_screen_plan,
    weighted_policy_metrics,
    weighted_target_entropy_log_temperature_derivative,
)
from trading_storage import load_torch_checkpoint  # noqa: E402
from train_return_oracle_decoder_screen import (  # noqa: E402
    batch_metrics,
    completed_epoch_limit_reached,
    early_stopping_limit_reached,
    persist_validated_epoch,
    validate_runtime_options,
)


ROOT = Path(__file__).resolve().parent.parent
PLAN_FILES = (
    ROOT / "ml/training-plans/return-oracle-decoder-screen-residual-glu-direct-v1.json",
    ROOT / "ml/training-plans/return-oracle-decoder-screen-learned-radius-direct-v1.json",
    ROOT / "ml/training-plans/return-oracle-decoder-screen-learned-radius-curriculum-v1.json",
    ROOT / "ml/training-plans/return-oracle-decoder-learned-radius-kl-gated-temperature-v1.json",
    ROOT / "ml/training-plans/return-oracle-decoder-learned-radius-direct-long-v1.json",
)
LONG_PLAN_FILE = PLAN_FILES[-1]
PRESERVED_BEST = (
    ROOT / "data/training/runs/"
    "return-oracle-ce-shrinking-v1-learned-radius-shrinking-baseline-"
    "20260730-104920/checkpoints/best.json"
)


class TemperatureCurriculumTests(unittest.TestCase):
    def test_production_entropy_confidence_weighting_prefers_sharper_targets(
        self,
    ) -> None:
        targets = torch.tensor([
            [1.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
            [1 / 3, 1 / 3, 1 / 3],
        ])
        weights = production_entropy_confidence_weights(
            targets, torch.ones(3), 0.05
        )
        self.assertEqual(float(weights[0]), 1.0)
        self.assertGreater(float(weights[0]), float(weights[1]))
        self.assertGreater(float(weights[1]), float(weights[2]))
        self.assertGreater(float(weights[2]), 0.0)

    def test_production_temperature_is_exact_identity(self) -> None:
        probabilities = torch.softmax(torch.randn(7, 255), dim=-1)
        transformed = smooth_oracle_probabilities(
            probabilities,
            source_temperature=PRODUCTION_ORACLE_TEMPERATURE,
            target_temperature=PRODUCTION_ORACLE_TEMPERATURE,
        )
        torch.testing.assert_close(transformed, probabilities)

    def test_smoothing_increases_entropy_and_preserves_zero_support(self) -> None:
        probabilities = torch.zeros(1, 255)
        probabilities[0, 10] = 0.8
        probabilities[0, 20] = 0.2
        smoothed = smooth_oracle_probabilities(
            probabilities,
            source_temperature=0.01,
            target_temperature=0.04,
        )
        original_entropy = -(probabilities[probabilities > 0].log()
                             * probabilities[probabilities > 0]).sum()
        smooth_entropy = -(smoothed[smoothed > 0].log()
                           * smoothed[smoothed > 0]).sum()
        self.assertGreater(float(smooth_entropy), float(original_entropy))
        self.assertEqual(int(torch.count_nonzero(smoothed)), 2)
        self.assertEqual(float(smoothed[0, 0]), 0.0)

    def test_curriculum_is_determined_only_by_absolute_epoch(self) -> None:
        plan = json.loads(PLAN_FILES[2].read_text(encoding="utf-8"))
        self.assertEqual(training_temperature(plan, 0), 0.04)
        self.assertEqual(training_temperature(plan, 3), 0.04)
        self.assertEqual(training_temperature(plan, 4), 0.02)
        self.assertEqual(training_temperature(plan, 7), 0.02)
        self.assertEqual(training_temperature(plan, 8), 0.01)
        self.assertEqual(training_temperature(plan, 100), 0.01)

    def test_kl_gated_curriculum_takes_equal_entropy_steps(self) -> None:
        plan = json.loads(PLAN_FILES[3].read_text(encoding="utf-8"))
        current = plan["curriculum"]["startTemperature"]
        held, goal = next_equal_entropy_step_training_temperature(
            plan, current, 0.050001, 5.2, 3.3, 5.2, 0.8
        )
        self.assertEqual(held, current)
        self.assertEqual(goal, 5.2)
        decreased, first_goal = next_equal_entropy_step_training_temperature(
            plan, current, 0.05, 5.2, 3.3, 5.2, 0.8
        )
        self.assertLess(decreased, current)
        entropy_step = (5.2 - 3.3) / (
            plan["curriculum"]["entropySchedulePoints"] - 1
        )
        self.assertAlmostEqual(first_goal, 5.2 - entropy_step)
        decreased_again, second_goal = (
            next_equal_entropy_step_training_temperature(
                plan, decreased, 0.0, first_goal, 3.3, 5.2, 0.8
            )
        )
        self.assertLess(decreased_again, decreased)
        self.assertAlmostEqual(second_goal, first_goal - entropy_step)
        final, _ = next_equal_entropy_step_training_temperature(
            plan, 0.01, 0.0, 3.3, 3.3, 5.2, 0.8
        )
        self.assertEqual(final, 0.01)

    def test_entropy_temperature_derivative_matches_finite_difference(self) -> None:
        probabilities = torch.softmax(torch.randn(4, 9), dim=-1)
        weights = torch.tensor([1.0, 2.0, 3.0, 4.0])
        temperature = 0.08
        target = smooth_oracle_probabilities(
            probabilities,
            source_temperature=0.01,
            target_temperature=temperature,
        )
        derivative = weighted_target_entropy_log_temperature_derivative(
            probabilities,
            target,
            weights,
            source_temperature=0.01,
            target_temperature=temperature,
        )

        def entropy(log_temperature: float) -> Tensor:
            transformed = smooth_oracle_probabilities(
                probabilities,
                source_temperature=0.01,
                target_temperature=float(torch.exp(torch.tensor(log_temperature))),
            )
            per_example = -(
                transformed * transformed.clamp_min(1e-30).log()
            ).sum(dim=-1)
            return (per_example * weights).sum() / weights.sum()

        center = float(torch.log(torch.tensor(temperature)))
        epsilon = 1e-3
        numerical = (entropy(center + epsilon) - entropy(center - epsilon)) \
            / (2 * epsilon)
        torch.testing.assert_close(derivative, numerical, rtol=2e-3, atol=2e-4)

    def test_learning_rate_decays_after_each_stale_prod_kl_block(self) -> None:
        plan = json.loads(PLAN_FILES[3].read_text(encoding="utf-8"))
        schedule = plan["training"]["learningRateSchedule"]
        current = schedule["startLearningRate"]
        stale_block = schedule["staleEpochsPerReduction"]
        for stale_epoch in range(1, stale_block):
            self.assertEqual(
                next_stale_learning_rate(schedule, current, stale_epoch),
                current,
            )
        first_reduction = next_stale_learning_rate(
            schedule, current, stale_block
        )
        self.assertAlmostEqual(
            first_reduction,
            current * schedule["decayFactor"],
        )
        current = schedule["startLearningRate"]
        first_floor_epoch = schedule["qualifyingStaleEpochs"]
        for stale_epoch in range(1, first_floor_epoch + 1):
            current = next_stale_learning_rate(schedule, current, stale_epoch)
        self.assertAlmostEqual(
            current,
            schedule["finalLearningRate"],
            places=15,
        )
        self.assertEqual(
            next_stale_learning_rate(
                schedule,
                current,
                first_floor_epoch + stale_block,
            ),
            schedule["finalLearningRate"],
        )


class ArchitectureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.mean = torch.zeros(60)
        self.std = torch.ones(60)

    def test_residual_glu_baseline_shape_and_optimizer_routing(self) -> None:
        model = ResidualGluDecoder(self.mean, self.std)
        self.assertEqual(model(torch.randn(5, 60)).shape, (5, 255))
        self.assertEqual(model.architecture_contract, "pre-ln-residual-glu-8x256-v1")

    def test_recovered_architecture_matches_preserved_parameter_contract(self) -> None:
        model = LearnedRadiusShrinkingDecoder(self.mean, self.std)
        self.assertEqual(model.architecture_contract, LEARNED_RADIUS_ARCHITECTURE)
        self.assertEqual(sum(value.numel() for value in model.parameters()), 15_090_191)
        self.assertEqual(len(model.state_dict()), 164)
        self.assertEqual(model(torch.randn(3, 60)).shape, (3, 255))
        logits, mean_penalty, variance_penalty = (
            model.forward_with_regularizers(torch.randn(3, 60))
        )
        self.assertEqual(logits.shape, (3, 255))
        self.assertEqual(mean_penalty.shape, (3,))
        self.assertEqual(variance_penalty.shape, (3,))

    @unittest.skipUnless(PRESERVED_BEST.is_file(), "preserved checkpoint unavailable")
    def test_recovered_architecture_loads_preserved_best_checkpoint_exactly(self) -> None:
        checkpoint = load_torch_checkpoint(
            PRESERVED_BEST, map_location="cpu", weights_only=False
        )
        state = checkpoint["model"]
        model = LearnedRadiusShrinkingDecoder(
            state["feature_mean"], state["feature_std"]
        )
        incompatible = model.load_state_dict(state)
        self.assertEqual(incompatible.missing_keys, [])
        self.assertEqual(incompatible.unexpected_keys, [])
        self.assertEqual(checkpoint["validation"]["baseKlDivergence"],
                         0.0711950351548919)


class SelectionAndPlanTests(unittest.TestCase):
    def test_restored_regularizers_ignore_disabled_centering_loss(self) -> None:
        plan = json.loads(PLAN_FILES[3].read_text(encoding="utf-8"))
        logits = torch.randn(2, 255)
        metrics = batch_metrics(
            (
                logits,
                torch.tensor([1.0, 3.0]),
                torch.tensor([2.0, 4.0]),
                torch.tensor(5.0),
                torch.tensor(0.25),
                torch.tensor(0.5),
            ),
            torch.softmax(torch.randn(2, 255), dim=-1),
            torch.tensor([1.0, 3.0]),
            0.05,
            plan["objective"]["regularizers"],
        )
        self.assertAlmostEqual(float(metrics["softLayerNorm"]), 6.0)
        self.assertEqual(float(metrics["centeringConstraint"]), 0.0)
        self.assertAlmostEqual(float(metrics["regularizationLoss"]), 6.05, places=6)
        self.assertAlmostEqual(
            float(metrics["loss"]),
            float(metrics["trainingCrossEntropy"]) + 6.05,
            places=5,
        )

    def test_raw_metric_is_zero_for_an_exact_uncalibrated_prediction(self) -> None:
        logits = torch.randn(4, 255)
        targets = torch.softmax(logits, dim=-1)
        metrics = weighted_policy_metrics(
            logits, targets, torch.tensor([1.0, 2.0, 3.0, 4.0])
        )
        self.assertAlmostEqual(float(metrics["rawBaseActionKl"]), 0.0, places=6)
        self.assertAlmostEqual(
            float(metrics["rawPredictedEntropy"]),
            float(metrics["rawTargetEntropy"]),
            places=5,
        )

    def test_all_screen_plans_are_frozen_raw_kl_validation_plans(self) -> None:
        fingerprints: set[str] = set()
        for plan_file in PLAN_FILES:
            plan = json.loads(plan_file.read_text(encoding="utf-8"))
            validate_screen_plan(plan)
            fingerprints.add(canonical_plan_fingerprint(plan))
            self.assertEqual(plan["dataset"]["expectedCounts"], FINAL_DATASET_COUNTS)
            self.assertEqual(plan["dataset"]["testPolicy"], "sealed-never-load")
            self.assertEqual(plan["selection"]["metric"], "rawBaseActionKl")
            self.assertEqual(plan["selection"]["predictionCalibration"], "none")
            self.assertEqual(plan["selection"]["oracleTemperature"], 0.01)
        self.assertEqual(len(fingerprints), len(PLAN_FILES))

    def test_plan_fingerprint_changes_with_any_training_change(self) -> None:
        plan = json.loads(PLAN_FILES[0].read_text(encoding="utf-8"))
        original = canonical_plan_fingerprint(plan)
        plan["training"]["learningRate"] *= 0.5
        self.assertNotEqual(canonical_plan_fingerprint(plan), original)

    def test_long_plan_is_fresh_direct_point01_and_patience_complete(self) -> None:
        long_plan = json.loads(LONG_PLAN_FILE.read_text(encoding="utf-8"))
        screen_plan = json.loads(PLAN_FILES[1].read_text(encoding="utf-8"))
        validate_screen_plan(long_plan)
        self.assertNotEqual(long_plan["id"], screen_plan["id"])
        self.assertNotEqual(long_plan["runDir"], screen_plan["runDir"])
        self.assertNotIn("initialCheckpoint", long_plan)
        self.assertEqual(long_plan["architecture"], screen_plan["architecture"])
        self.assertEqual(long_plan["dataset"], screen_plan["dataset"])
        self.assertEqual(long_plan["curriculum"]["stages"], [
            {"startEpoch": 0, "temperature": 0.01},
        ])
        self.assertEqual(long_plan["training"]["epochs"], 1_600)
        self.assertGreaterEqual(
            long_plan["training"]["learningRateSchedule"]["patience"],
            48,
        )
        self.assertGreaterEqual(
            long_plan["training"]["earlyStoppingPatience"],
            160,
        )
        self.assertEqual(long_plan["selection"]["metric"], "rawBaseActionKl")
        self.assertEqual(long_plan["dataset"]["testPolicy"], "sealed-never-load")

    def test_early_stopping_counts_validation_epochs_and_can_be_disabled(self) -> None:
        self.assertFalse(early_stopping_limit_reached(None, 10_000))
        self.assertFalse(early_stopping_limit_reached(160, 159))
        self.assertTrue(early_stopping_limit_reached(160, 160))
        with self.assertRaisesRegex(ValueError, "positive"):
            early_stopping_limit_reached(0, 0)
        with self.assertRaisesRegex(ValueError, "negative"):
            early_stopping_limit_reached(160, -1)

    def test_plan_cannot_finish_before_production_temperature_stage(self) -> None:
        plan = json.loads(PLAN_FILES[2].read_text(encoding="utf-8"))
        plan["training"]["epochs"] = 8
        with self.assertRaisesRegex(ValueError, "production temperature"):
            validate_screen_plan(plan)

    def test_stop_after_epoch_is_one_based_and_only_limits_training(self) -> None:
        validate_runtime_options(validate_only=False, stop_after_epoch=1)
        self.assertFalse(completed_epoch_limit_reached(1, -1))
        self.assertTrue(completed_epoch_limit_reached(1, 0))
        self.assertFalse(completed_epoch_limit_reached(3, 1))
        self.assertTrue(completed_epoch_limit_reached(3, 2))
        self.assertFalse(completed_epoch_limit_reached(None, 100))
        with self.assertRaisesRegex(ValueError, "must be positive"):
            validate_runtime_options(validate_only=False, stop_after_epoch=0)
        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            validate_runtime_options(validate_only=True, stop_after_epoch=1)

    def test_stop_boundary_is_returned_only_after_durable_checkpoint_calls(self) -> None:
        checkpoint = {"validation": {"rawBaseActionKl": 0.5}}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_root = Path(directory) / "checkpoints"
            last = checkpoint_root / "last.json"
            best = checkpoint_root / "best.json"
            with patch(
                "train_return_oracle_decoder_screen.save_torch_checkpoint"
            ) as save:
                should_pause = persist_validated_epoch(
                    checkpoint,
                    last_checkpoint=last,
                    best_checkpoint=best,
                    improved=True,
                    stop_after_epoch=1,
                    completed_epoch=0,
                )
            self.assertTrue(should_pause)
            self.assertEqual(
                [call.args[1] for call in save.call_args_list],
                [last, best],
            )
        with self.assertRaisesRegex(ValueError, "lacks validation"):
            persist_validated_epoch(
                {},
                last_checkpoint=Path("last.json"),
                best_checkpoint=Path("best.json"),
                improved=False,
                stop_after_epoch=1,
                completed_epoch=0,
            )


if __name__ == "__main__":
    unittest.main()

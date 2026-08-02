from __future__ import annotations

import io
import json
import math
import random
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from joint_price_oracle import JointLossWeights
from joint_price_oracle_actions import teacher_actions_at_current_exposures_numpy

from train_joint_price_oracle import (
    METRIC_NAMES,
    CausalOracleDataset,
    CausalSegment,
    MetricAccumulator,
    RunReporter,
    add_action_objective,
    build_training_checkpoint,
    chronological_reset_mask,
    configuration_fingerprint,
    evaluate,
    ordered_rollout_metrics,
    precompute_teacher_current_exposures,
    resolve_action_objective_config,
    resolve_training_config,
)


class FakeTargetCache:
    rows_per_file = 2
    action_count = 3

    def __init__(self, rows):
        self.rows = rows
        self.loaded = []

    def load(self, file):
        self.loaded.append(file)
        return self.rows[file]


class FakeCloseCache:
    def range(self, first_close_time, last_close_time):
        count = (last_close_time - first_close_time) // 1_000 + 1
        return np.arange(1, count + 1, dtype=np.float32)


class FakeOrderedEvaluationDataset:
    def __init__(self, segments, targets, teacher_rows):
        self.segments = {"validation": segments}
        self.targets = targets
        self.teacher_rows = teacher_rows

    def iter_batches(
        self,
        split,
        batch_size,
        *,
        shuffle,
        seed,
        maximum_batches,
    ):
        del split, seed
        if shuffle:
            raise AssertionError("ordered validation must not shuffle")
        emitted = 0
        cursor = 0
        for start in range(0, len(self.targets), batch_size):
            end = min(start + batch_size, len(self.targets))
            indices = torch.arange(start, end, dtype=torch.float32)
            input_closes = (indices + 1).view(-1, 1, 1).repeat(1, 2, 1)
            future_closes = (indices + 1).view(-1, 1, 1)
            yield (
                input_closes,
                future_closes,
                self.targets[start:end],
                self.teacher_rows[start:end],
            )
            emitted += 1
            cursor = end
            if maximum_batches is not None and emitted >= maximum_batches:
                break
        if maximum_batches is None and cursor != len(self.targets):
            raise AssertionError("validation rows were truncated")


class FakePolicyModel:
    def __init__(self, logits):
        self.logits = logits

    def eval(self):
        return self

    def forward_with_forecast(self, input_closes):
        indices = input_closes[:, 0, 0].long() - 1
        return SimpleNamespace(policy_logits=self.logits[indices])


class TrainJointPriceOracleActionsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.grid = np.asarray([-1.0, 0.0, 1.0], dtype=np.float64)
        self.files = {
            name: Path(f"{name}.json")
            for name in (
                "train-1",
                "train-2",
                "train-gap",
                "validation",
                "test",
            )
        }
        self.target_rows = {
            self.files["train-1"]: torch.tensor([
                [0.10, 0.80, 0.10],
                [0.05, 0.10, 0.85],
            ]),
            self.files["train-2"]: torch.tensor([
                [0.04, 0.10, 0.86],
                [0.03, 0.10, 0.87],
            ]),
            self.files["train-gap"]: torch.tensor([
                [0.02, 0.10, 0.88],
                [0.10, 0.89, 0.01],
            ]),
            self.files["validation"]: torch.tensor([
                [0.10, 0.80, 0.10],
                [0.10, 0.80, 0.10],
            ]),
            self.files["test"]: torch.tensor([
                [0.10, 0.80, 0.10],
                [0.10, 0.80, 0.10],
            ]),
        }
        self.segments = {
            "train": [
                CausalSegment(
                    "train", 999, 2, self.files["train-1"], 0, 60_000
                ),
                # Continuous despite crossing an immutable daily file.
                CausalSegment(
                    "train", 120_999, 2, self.files["train-2"], 0, 60_000
                ),
                # A true timestamp gap must reset exposure to zero.
                CausalSegment(
                    "train", 360_999, 2, self.files["train-gap"], 0, 60_000
                ),
            ],
            "validation": [CausalSegment(
                "validation", 999, 2, self.files["validation"], 0, 60_000
            )],
            "test": [CausalSegment(
                "test", 999, 2, self.files["test"], 0, 60_000
            )],
        }

    def teacher_rows(self):
        return precompute_teacher_current_exposures(
            self.segments,
            FakeTargetCache(self.target_rows),
            self.grid,
            friction=0,
            temperature=0.01,
        )

    def test_teacher_rollout_carries_daily_state_and_resets_real_gaps(self) -> None:
        rows = self.teacher_rows()
        torch.testing.assert_close(
            rows[self.files["train-1"]],
            torch.tensor([0.0, 0.0]),
        )
        torch.testing.assert_close(
            rows[self.files["train-2"]],
            torch.tensor([1.0, 1.0]),
        )
        torch.testing.assert_close(
            rows[self.files["train-gap"]],
            torch.tensor([0.0, 1.0]),
        )

    def test_teacher_rollout_keeps_test_payload_sealed_by_default(self) -> None:
        cache = FakeTargetCache(self.target_rows)
        statistics = {}
        rows = precompute_teacher_current_exposures(
            self.segments,
            cache,
            self.grid,
            friction=0,
            temperature=0.01,
            switch_statistics=statistics,
        )
        self.assertNotIn(self.files["test"], cache.loaded)
        self.assertNotIn(self.files["test"], rows)
        self.assertEqual(statistics["train"]["decisions"], 6)
        self.assertGreater(statistics["train"]["switchFraction"], 0)

        explicit_cache = FakeTargetCache(self.target_rows)
        explicit_rows = precompute_teacher_current_exposures(
            self.segments,
            explicit_cache,
            self.grid,
            splits=("train", "validation", "test"),
            friction=0,
            temperature=0.01,
        )
        self.assertIn(self.files["test"], explicit_cache.loaded)
        self.assertIn(self.files["test"], explicit_rows)

    def test_exact_teacher_state_provider_controls_rows_and_switch_statistics(
        self,
    ) -> None:
        exact_by_file = {
            self.files["train-1"]: np.asarray([0.25, -0.25]),
            self.files["train-2"]: np.asarray([0.20, 0.30]),
            self.files["train-gap"]: np.asarray([-1.0, 1.0]),
        }
        calls = []

        def provider(split, segment):
            calls.append((split, segment.target_file, segment.target_row_offset))
            return exact_by_file[segment.target_file]

        execution_policy = {
            "version": 2,
            "maximumLeverage": 1,
            "minimumConfidence": 0.05,
            "confidenceExposurePower": 0,
            "confidenceLeverageFloor": 0.75,
        }
        statistics = {}
        rows = precompute_teacher_current_exposures(
            self.segments,
            FakeTargetCache(self.target_rows),
            self.grid,
            splits=("train",),
            friction=0,
            temperature=0.01,
            execution_policy=execution_policy,
            teacher_current_exposure_provider=provider,
            switch_statistics=statistics,
        )
        expected_switches = 0
        for segment in self.segments["train"]:
            expected = exact_by_file[segment.target_file]
            np.testing.assert_allclose(
                rows[segment.target_file].numpy(),
                expected,
            )
            actions = teacher_actions_at_current_exposures_numpy(
                self.target_rows[segment.target_file].numpy(),
                self.grid,
                expected,
                friction=0,
                temperature=0.01,
                execution_policy=execution_policy,
            )
            expected_switches += int(actions.switch_labels.sum())
        self.assertEqual(len(calls), 3)
        self.assertEqual(statistics["train"]["switches"], expected_switches)
        self.assertEqual(statistics["train"]["decisions"], 6)

    def test_split_specific_exact_provider_keeps_train_surrogate(self) -> None:
        exact_validation = np.asarray([0.25, -0.25])
        calls = []

        def provider(split, segment):
            calls.append((split, segment.target_file))
            return exact_validation

        rows = precompute_teacher_current_exposures(
            self.segments,
            FakeTargetCache(self.target_rows),
            self.grid,
            splits=("train", "validation"),
            friction=0,
            temperature=0.01,
            teacher_current_exposure_providers={
                "validation": provider,
            },
        )
        np.testing.assert_allclose(
            rows[self.files["validation"]].numpy(),
            exact_validation,
        )
        self.assertEqual(calls, [
            ("validation", self.files["validation"]),
        ])
        # The unconfigured train split still follows its chronological target.
        np.testing.assert_allclose(
            rows[self.files["train-1"]].numpy(),
            [0.0, 0.0],
        )

    def test_teacher_rows_remain_aligned_when_batches_shuffle(self) -> None:
        teacher_rows = self.teacher_rows()
        dataset = CausalOracleDataset(
            Path("unused"),
            self.segments,
            context_length=2,
            forecast_horizon=1,
            target_rows_per_file=2,
            action_count=3,
            teacher_current_exposures=teacher_rows,
        )
        dataset.close_cache = FakeCloseCache()
        dataset.target_cache = FakeTargetCache(self.target_rows)
        expected = {
            tuple(target.tolist()): float(teacher_rows[file][row])
            for file, targets in self.target_rows.items()
            for row, target in enumerate(targets)
            if file in {
                self.files["train-1"],
                self.files["train-2"],
                self.files["train-gap"],
            }
        }
        observed = {}
        batches = dataset.iter_batches(
            "train",
            1,
            shuffle=True,
            seed=19,
        )
        for batch in batches:
            self.assertEqual(len(batch), 4)
            target = tuple(batch[2][0].tolist())
            observed[target] = float(batch[3][0])
        self.assertEqual(observed, expected)

        legacy = CausalOracleDataset(
            Path("unused"),
            self.segments,
            context_length=2,
            forecast_horizon=1,
            target_rows_per_file=2,
            action_count=3,
        )
        legacy.close_cache = FakeCloseCache()
        legacy.target_cache = FakeTargetCache(self.target_rows)
        self.assertEqual(len(next(legacy.iter_batches(
            "train", 1, shuffle=False, seed=1
        ))), 3)

    def test_action_objective_adds_finite_gradient_and_metrics(self) -> None:
        target = torch.tensor([
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.98, 0.01, 0.01],
        ])
        current = torch.tensor([0.0, 0.0, 1.0, 1.0])
        predicted = target.log().detach().clone().requires_grad_(True)
        base = {
            name: torch.tensor(2.0 if name == "loss" else 0.0)
            for name in METRIC_NAMES
        }
        config = resolve_action_objective_config({
            "lossWeight": 0.5,
            "targetSwitchFraction": 0.5,
            "rankingMargin": 0.1,
        })
        metrics = add_action_objective(
            base,
            predicted,
            target,
            current,
            torch.tensor([-1.0, 0.0, 1.0]),
            config,
            policy_friction=0,
            policy_temperature=0.01,
        )
        self.assertTrue(bool(torch.isfinite(metrics["loss"])))
        self.assertGreater(float(metrics["loss"].detach()), 2.0)
        self.assertIn("actionLoss", metrics)
        accumulator = MetricAccumulator()
        accumulator.add(metrics, len(target))
        self.assertIn("actionLoss", accumulator.result())
        metrics["loss"].backward()
        self.assertIsNotNone(predicted.grad)
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_mixed_self_conditioned_objective_is_finite_and_differentiable(
        self,
    ) -> None:
        target = torch.tensor([
            [0.01, 0.98, 0.01],
            [0.01, 0.01, 0.98],
            [0.01, 0.01, 0.98],
            [0.98, 0.01, 0.01],
        ])
        teacher_current = torch.tensor([0.0, 0.0, 1.0, 1.0])
        predicted = torch.flip(target, dims=(-1,)).log().requires_grad_(True)
        base = {
            name: torch.tensor(2.0 if name == "loss" else 0.0)
            for name in METRIC_NAMES
        }
        metrics = add_action_objective(
            base,
            predicted,
            target,
            teacher_current,
            torch.tensor([-1.0, 0.0, 1.0]),
            {
                "teacherStateWeight": 1,
                "selfStateWeight": 1,
                "switchWeighting": "global",
            },
            source_switch_fraction=0.25,
            policy_friction=0,
            policy_temperature=0.01,
        )
        self.assertIn("mixedActionLoss", metrics)
        self.assertIn("selfActionLoss", metrics)
        self.assertTrue(bool(torch.isfinite(metrics["loss"])))
        accumulator = MetricAccumulator()
        accumulator.add(metrics, len(target))
        self.assertIn("selfActionLoss", accumulator.result())
        metrics["loss"].backward()
        self.assertTrue(bool(torch.isfinite(predicted.grad).all()))

    def test_ordered_rollout_resets_only_at_true_gaps(self) -> None:
        resets = chronological_reset_mask(self.segments["train"])
        np.testing.assert_array_equal(
            resets,
            [True, False, False, False, True, False],
        )
        target_indices = np.asarray([1, 2, 2, 2, 1, 0])
        target = np.full((6, 3), 0.01, dtype=np.float64)
        target[np.arange(6), target_indices] = 0.98
        perfect = np.log(target)
        metrics = ordered_rollout_metrics(
            perfect,
            target,
            self.grid,
            self.segments["train"],
            friction=0,
            temperature=0.01,
        )
        self.assertAlmostEqual(metrics["rolloutTransitionPrecision"], 1)
        self.assertAlmostEqual(metrics["rolloutTransitionRecall"], 1)
        self.assertAlmostEqual(metrics["rolloutTransitionF1"], 1)
        self.assertAlmostEqual(metrics["rolloutScore"], 0)

        wrong = np.flip(perfect, axis=-1).copy()
        wrong_metrics = ordered_rollout_metrics(
            wrong,
            target,
            self.grid,
            self.segments["train"],
            friction=0,
            temperature=0.01,
        )
        self.assertGreater(wrong_metrics["rolloutScore"], 0)

    def test_rollout_score_v2_is_signed_transition_dominated(self) -> None:
        target_indices = np.asarray([1, 2, 2, 0, 0, 1])
        target = np.full((6, 3), 0.01, dtype=np.float64)
        target[np.arange(6), target_indices] = 0.98
        wrong = np.flip(np.log(target), axis=-1).copy()
        execution_policy = {
            "version": 2,
            "maximumLeverage": 1,
            "minimumConfidence": 0.05,
            "confidenceExposurePower": 0,
            "confidenceLeverageFloor": 0.75,
        }
        metrics = ordered_rollout_metrics(
            wrong,
            target,
            self.grid,
            self.segments["train"],
            friction=0,
            temperature=0.01,
            execution_policy=execution_policy,
            rollout_score_version=2,
        )
        expected = (
            0.6 * (1 - metrics["rolloutSignedTransitionF1"])
            + 0.15 * (1 - metrics["rolloutPathDirectionalAgreement"])
            + 0.15 * metrics["rolloutPathMeanAbsoluteError"] / 2
            + 0.1 * metrics["rolloutTurnoverRelativeError"]
        )
        self.assertAlmostEqual(metrics["rolloutScore"], expected)
        self.assertEqual(metrics["rolloutScoreVersion"], 2)

    def test_zero_teacher_mass_regret_is_null_but_score_stays_finite(
        self,
    ) -> None:
        segments = [CausalSegment(
            "validation",
            999,
            2,
            Path("validation.json"),
            0,
            60_000,
        )]
        target = np.asarray([
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        predicted = np.asarray([
            [10.0, 0.0, -10.0],
            [10.0, 0.0, -10.0],
        ])
        metrics = ordered_rollout_metrics(
            predicted,
            target,
            self.grid,
            segments,
            friction=0,
            temperature=0.01,
        )
        self.assertIsNone(metrics["rolloutMeanConditionalRegret"])
        self.assertFalse(metrics["rolloutMeanConditionalRegretFinite"])
        self.assertIsNone(metrics["rolloutSwitchConditionalRegret"])
        self.assertFalse(metrics["rolloutSwitchConditionalRegretFinite"])
        self.assertTrue(math.isfinite(metrics["rolloutScore"]))
        json.dumps(metrics, allow_nan=False)

    def test_reporter_and_checkpoint_metadata_replace_nonfinite_scalars(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            reporter = RunReporter(run_dir, "json-safe-test")
            with redirect_stdout(io.StringIO()):
                reporter.emit({
                    "event": "diagnostic",
                    "infinite": math.inf,
                    "nan": math.nan,
                })
            event = json.loads(
                reporter.log_file.read_text(encoding="utf-8")
            )
            self.assertIsNone(event["infinite"])
            self.assertIsNone(event["nan"])
            reporter.status("training", bestValidation=math.inf)
            status = json.loads(
                reporter.status_file.read_text(encoding="utf-8")
            )
            self.assertIsNone(status["bestValidation"])

        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
        checkpoint = build_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=1,
            global_step=1,
            best_validation=0.5,
            best_epoch=1,
            stale_epochs=0,
            validation={"regret": math.inf, "other": math.nan},
            model_parameters=sum(p.numel() for p in model.parameters()),
            dataset_fingerprint="dataset",
            model_config={},
            plan_id="json-safe-test",
            device=torch.device("cpu"),
            interrupted=False,
            training_config_fingerprint="training",
        )
        self.assertIsNone(checkpoint["validation"]["regret"])
        self.assertIsNone(checkpoint["validation"]["other"])
        json.dumps(checkpoint["validation"], allow_nan=False)
        with self.assertRaises(ValueError):
            build_training_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch=1,
                global_step=1,
                best_validation=math.inf,
                best_epoch=0,
                stale_epochs=0,
                validation=None,
                model_parameters=2,
                dataset_fingerprint="dataset",
                model_config={},
                plan_id="json-safe-test",
                device=torch.device("cpu"),
                interrupted=False,
                training_config_fingerprint="training",
            )

    def test_validation_rollout_metrics_are_independent_of_batch_boundaries(
        self,
    ) -> None:
        files = (
            self.files["train-1"],
            self.files["train-2"],
            self.files["train-gap"],
        )
        targets = torch.cat([self.target_rows[file] for file in files])
        teacher = self.teacher_rows()
        teacher_rows = torch.cat([teacher[file] for file in files])
        dataset = FakeOrderedEvaluationDataset(
            self.segments["train"],
            targets,
            teacher_rows,
        )
        model = FakePolicyModel(targets.log())
        training = {
            "seed": 7,
            "prefetchBatches": 1,
            "mixedPrecision": "float32",
            "forecastHuberDelta": 1,
            "volatilityFloor": 1e-5,
        }

        def zero_base_metrics(output, *_args, **_kwargs):
            zero = output.policy_logits.float().sum() * 0
            return {name: zero for name in METRIC_NAMES}

        results = []
        with patch(
            "train_joint_price_oracle.joint_price_oracle_objective",
            side_effect=zero_base_metrics,
        ):
            for batch_size in (2, 4):
                results.append(evaluate(
                    model,
                    dataset,
                    "validation",
                    batch_size,
                    torch.device("cpu"),
                    training,
                    JointLossWeights(),
                    torch.tensor([-1.0, 0.0, 1.0]),
                    0,
                    0.01,
                    action_objective={},
                    maximum_batches=None,
                ))
        for result in results:
            self.assertAlmostEqual(result["rolloutTransitionPrecision"], 1)
            self.assertAlmostEqual(result["rolloutTransitionRecall"], 1)
            self.assertAlmostEqual(result["rolloutTransitionF1"], 1)
            self.assertAlmostEqual(result["rolloutScore"], 0)
        for name in (
            "rolloutTransitionPrecision",
            "rolloutTransitionRecall",
            "rolloutTransitionF1",
            "rolloutScore",
        ):
            self.assertAlmostEqual(results[0][name], results[1][name])

    def test_action_config_is_optional_canonical_and_validated(self) -> None:
        legacy = resolve_training_config({"lossWeights": {}})
        self.assertNotIn("actionObjective", legacy)
        action = resolve_training_config({
            "lossWeights": {},
            "actionObjective": {},
        })["actionObjective"]
        self.assertEqual(action["lossWeight"], 1.0)
        self.assertEqual(action["weights"]["conditionalKl"], 0.1)
        self.assertNotIn("teacherStateWeight", action)
        self.assertNotIn("selfStateWeight", action)
        self.assertNotIn("switchWeighting", action)
        self.assertNotIn("executionPolicy", action)
        self.assertNotIn("rolloutScoreVersion", action)
        mixed = resolve_action_objective_config({
            "teacherStateWeight": 0.5,
            "selfStateWeight": 1.0,
            "switchWeighting": "global",
        })
        self.assertEqual(mixed["teacherStateWeight"], 0.5)
        self.assertEqual(mixed["selfStateWeight"], 1.0)
        self.assertEqual(mixed["switchWeighting"], "global")
        execution = resolve_action_objective_config({
            "rolloutScoreVersion": 2,
            "executionPolicy": {"version": 2},
        })
        self.assertEqual(execution["executionPolicy"]["maximumLeverage"], 100)
        self.assertEqual(execution["rolloutScoreVersion"], 2)
        trace_declaration = {
            "validation": {
                "path": "data/training/derived/joint-price-oracle/teacher-traces/v.json",
                "sha256": "a" * 64,
                "schemaVersion": 1,
                "maximumLeverage": 100,
                "rows": 43200,
            },
        }
        with self.assertRaisesRegex(
            ValueError,
            "allowValidationExactWithTrainSurrogate",
        ):
            resolve_action_objective_config({
                "executionPolicy": {"version": 2},
                "exactStateTraces": trace_declaration,
            })
        exact = resolve_action_objective_config({
            "executionPolicy": {"version": 2},
            "exactStateTraces": trace_declaration,
            "allowValidationExactWithTrainSurrogate": True,
        })
        self.assertEqual(
            exact["exactStateTraces"]["validation"]["sha256"],
            "a" * 64,
        )
        self.assertEqual(
            exact["exactStateTraces"]["validation"]["rows"],
            43200,
        )
        with self.assertRaises(ValueError):
            resolve_action_objective_config({"targetSwitchFraction": 1})
        with self.assertRaises(ValueError):
            resolve_action_objective_config({"weights": {"typo": 1}})
        with self.assertRaises(ValueError):
            resolve_action_objective_config({
                "teacherStateWeight": 0,
                "selfStateWeight": 0,
            })
        with self.assertRaises(ValueError):
            resolve_action_objective_config({"switchWeighting": "epoch"})
        with self.assertRaises(ValueError):
            resolve_action_objective_config({"rolloutScoreVersion": 2})

        repo_root = Path(__file__).resolve().parents[1]
        legacy_plan = json.loads((
            repo_root
            / "ml"
            / "training-plans"
            / "joint-price-oracle-action-tcn-v4.json"
        ).read_text(encoding="utf-8"))
        self.assertEqual(
            configuration_fingerprint(resolve_training_config(
                legacy_plan["training"]
            )),
            "960a076142f7d94b7c17674699e21d58206828fc8807aeceaddadf9fed238094",
        )
        rollout_fingerprints = {}
        expected_fingerprints = {
            "joint-price-oracle-action-patch-transformer-rollout-v6.json": (
                "e066d7743fa5682879d159d9622d0c0cc48bd6e134608d5fbab7f639a7d535df"
            ),
            "joint-price-oracle-action-residual-mixer-rollout-v7.json": (
                "b0b642b019c151558e4494efdf0500785d9091f332468b89b2b3c8aeebdd8c0b"
            ),
            "joint-price-oracle-action-tcn-rollout-v8.json": (
                "39cde458d1eedaa5fb662fbba65b643347a773dc70082637bf6bbe90ad04a1ca"
            ),
        }
        for name in expected_fingerprints:
            rollout_plan = json.loads((
                repo_root / "ml" / "training-plans" / name
            ).read_text(encoding="utf-8"))
            resolved = resolve_training_config(rollout_plan["training"])
            self.assertEqual(
                resolved["actionObjective"]["executionPolicy"]["version"],
                2,
            )
            self.assertEqual(
                resolved["actionObjective"]["rolloutScoreVersion"],
                2,
            )
            self.assertEqual(
                resolved["selectionMetric"],
                "exactStateScore",
            )
            self.assertEqual(
                resolved["actionObjective"]["targetSwitchFraction"],
                0.2,
            )
            self.assertEqual(
                resolved["actionObjective"]["exactStateTraces"]
                ["validation"]["sha256"],
                "49c6c6328479f971e1e33752b60fe74da2d39f187635450833bc7a589a9097aa",
            )
            rollout_fingerprints[name] = configuration_fingerprint(resolved)
        self.assertEqual(
            rollout_fingerprints,
            expected_fingerprints,
        )


if __name__ == "__main__":
    random.seed(0)
    unittest.main()

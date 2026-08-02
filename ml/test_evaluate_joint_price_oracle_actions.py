from __future__ import annotations

import unittest
from itertools import permutations
from pathlib import Path

import numpy as np
import torch
from torch import nn

from evaluate_joint_price_oracle_actions import (
    chronological_reset_mask,
    collect_policy_rows,
    finite_json_value,
    parse_positive_temperatures,
    raw_distribution_metrics,
    resolved_action_execution_policy,
    select_action_temperature,
    validate_checkpoint,
    validate_split_access,
)
from train_joint_price_oracle import (
    DATA_CONTRACT,
    CausalSegment,
    architecture_contract_for_model_config,
)


class FakeDataset:
    def __init__(self, batches):
        self.batches = batches

    def iter_batches(
        self,
        split,
        batch_size,
        *,
        shuffle,
        seed,
        maximum_batches,
    ):
        if split != "validation" or batch_size != 2 or shuffle \
                or seed != 13 or maximum_batches is not None:
            raise AssertionError("evaluator changed chronological batch options")
        yield from self.batches

    def batch_count(self, split, batch_size):
        if split != "validation" or batch_size != 2:
            raise AssertionError("unexpected batch-count request")
        return len(self.batches)


class FirstTimeStepModel(nn.Module):
    def forward(self, closes):
        return closes[:, 0, :]


class PolicyOnlyFirstTimeStepModel(nn.Module):
    def forward_policy_logits(self, closes):
        return closes[:, 0, :]

    def forward(self, _closes):
        raise AssertionError("legacy forecast-producing forward executed")


class EvaluateJointPriceOracleActionsTest(unittest.TestCase):
    def test_reset_mask_ignores_shard_boundaries_but_marks_time_gaps(self) -> None:
        segments = [
            CausalSegment(
                "validation",
                999,
                2,
                Path("day-1.json"),
                0,
                60_000,
            ),
            CausalSegment(
                "validation",
                120_999,
                2,
                Path("day-2.json"),
                0,
                60_000,
            ),
            CausalSegment(
                "validation",
                360_999,
                2,
                Path("day-3.json"),
                0,
                60_000,
            ),
        ]
        np.testing.assert_array_equal(
            chronological_reset_mask(segments),
            [True, False, False, False, True, False],
        )

    def test_reset_mask_rejects_overlapping_segments(self) -> None:
        with self.assertRaises(ValueError):
            chronological_reset_mask([
                CausalSegment("validation", 999, 3, Path("a"), 0, 60_000),
                CausalSegment("validation", 60_999, 2, Path("b"), 0, 60_000),
            ])

    def test_raw_metrics_are_exact_for_matching_distribution(self) -> None:
        targets = np.asarray([
            [0.7, 0.2, 0.1],
            [0.1, 0.4, 0.5],
        ], dtype=np.float64)
        metrics = raw_distribution_metrics(np.log(targets), targets * 3)
        self.assertAlmostEqual(metrics["klDivergence"], 0.0, places=14)
        self.assertAlmostEqual(
            metrics["crossEntropy"],
            metrics["targetEntropy"],
            places=14,
        )
        self.assertAlmostEqual(
            metrics["predictedEntropy"],
            metrics["targetEntropy"],
            places=14,
        )

    def test_collect_policy_rows_preserves_batch_and_row_order(self) -> None:
        first_logits = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        second_logits = torch.tensor([[7.0, 8.0, 9.0]])
        first_target = torch.softmax(first_logits, dim=-1)
        second_target = torch.softmax(second_logits, dim=-1)
        batches = [
            (
                first_logits[:, None, :],
                torch.zeros(2, 1, 3),
                first_target,
            ),
            (
                second_logits[:, None, :],
                torch.zeros(1, 1, 3),
                second_target,
            ),
        ]
        logits, targets = collect_policy_rows(
            FirstTimeStepModel(),
            FakeDataset(batches),
            "validation",
            2,
            torch.device("cpu"),
            {
                "seed": 13,
                "mixedPrecision": "float32",
                "prefetchBatches": 1,
            },
        )
        np.testing.assert_array_equal(
            logits,
            torch.cat((first_logits, second_logits)).numpy(),
        )
        np.testing.assert_allclose(
            targets,
            torch.cat((first_target, second_target)).numpy(),
        )

    def test_collect_policy_rows_uses_compact_policy_only_batches(self) -> None:
        first_logits = torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ])
        second_logits = torch.tensor([[7.0, 8.0, 9.0]])
        first_target = torch.softmax(first_logits, dim=-1)
        second_target = torch.softmax(second_logits, dim=-1)
        logits, targets = collect_policy_rows(
            PolicyOnlyFirstTimeStepModel(),
            FakeDataset([
                (first_logits[:, None, :], first_target),
                (second_logits[:, None, :], second_target),
            ]),
            "validation",
            2,
            torch.device("cpu"),
            {
                "policyOnly": True,
                "seed": 13,
                "mixedPrecision": "float32",
                "prefetchBatches": 1,
            },
        )
        np.testing.assert_array_equal(
            logits,
            torch.cat((first_logits, second_logits)).numpy(),
        )
        np.testing.assert_allclose(
            targets,
            torch.cat((first_target, second_target)).numpy(),
        )

    def test_test_split_requires_explicit_access(self) -> None:
        validate_split_access("validation", False)
        validate_split_access("test", True)
        with self.assertRaises(PermissionError):
            validate_split_access("test", False)
        with self.assertRaises(ValueError):
            validate_split_access("train", True)

    def test_checkpoint_validation_uses_variant_architecture_contract(self) -> None:
        model_config = {"variant": "patch_transformer"}
        checkpoint = {
            "planId": "variant-plan",
            "architectureContract": architecture_contract_for_model_config(
                model_config
            ),
            "dataContract": DATA_CONTRACT,
            "datasetFingerprint": "dataset",
            "modelConfig": model_config,
            "parameterCount": 17,
            "trainingConfigFingerprint": "training",
            "model": {},
        }
        validate_checkpoint(
            checkpoint,
            {"id": "variant-plan"},
            model_config,
            "dataset",
            17,
            "training",
        )
        checkpoint["architectureContract"] = "legacy-contract"
        with self.assertRaises(ValueError):
            validate_checkpoint(
                checkpoint,
                {"id": "variant-plan"},
                model_config,
                "dataset",
                17,
                "training",
            )

    def test_nonfinite_metrics_become_strict_json_nulls(self) -> None:
        result = finite_json_value({
            "finite": 1.0,
            "infinite": float("inf"),
            "nested": [np.float64(float("nan"))],
        })
        self.assertEqual(result, {
            "finite": 1.0,
            "infinite": None,
            "nested": [None],
        })

    def test_temperature_grid_is_positive_unique_and_ordered(self) -> None:
        self.assertEqual(
            parse_positive_temperatures("1, 0.5,1,2"),
            (1.0, 0.5, 2.0),
        )
        for invalid in ("", "0", "-1", "nan", "one"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                parse_positive_temperatures(invalid)

    def test_execution_policy_is_resolved_for_the_temperature_sweep(self) -> None:
        self.assertIsNone(resolved_action_execution_policy({}))
        self.assertEqual(
            resolved_action_execution_policy({
                "actionObjective": {
                    "executionPolicy": {
                        "version": 2,
                        "maximumLeverage": 100,
                    },
                },
            }),
            {
                "version": 2,
                "maximumLeverage": 100.0,
                "minimumConfidence": 0.05,
                "confidenceExposurePower": 0.0,
                "confidenceLeverageFloor": 0.75,
            },
        )

    def test_action_temperature_selection_is_order_independent(self) -> None:
        candidates = [
            temperature_item(0.5, f1=0.4, precision=0.6, recall=0.3),
            temperature_item(1.0, f1=0.4, precision=0.7, recall=0.2),
            temperature_item(2.0, f1=0.3, precision=0.9, recall=0.9),
        ]
        for ordered in permutations(candidates):
            with self.subTest(order=[item["logitTemperature"] for item in ordered]):
                selection = select_action_temperature(list(ordered))
                self.assertEqual(selection["selectedLogitTemperature"], 1.0)
                self.assertEqual(
                    selection["selectedMetrics"],
                    {
                        "raw": candidates[1]["raw"],
                        "actions": candidates[1]["actions"],
                    },
                )
                self.assertEqual(selection["objective"], {
                    "metric": "actions.signedTransitionF1",
                    "direction": "maximize",
                })

    def test_action_temperature_exact_tie_prefers_identity_then_lower(self) -> None:
        shared = {
            "f1": 0.0,
            "precision": 0.0,
            "recall": 0.0,
        }
        selection = select_action_temperature([
            temperature_item(2.0, **shared),
            temperature_item(0.5, **shared),
            temperature_item(1.0, **shared),
        ])
        self.assertEqual(selection["selectedLogitTemperature"], 1.0)
        self.assertEqual(selection["identityIndex"], 2)

    def test_exact_state_temperature_selection_uses_executable_f1(self) -> None:
        surrogate_winner = temperature_item(
            0.5,
            f1=0.9,
            precision=0.9,
            recall=0.9,
        )
        exact_winner = temperature_item(
            1.0,
            f1=0.2,
            precision=0.2,
            recall=0.2,
        )
        surrogate_winner["exactStateActions"] = exact_state_item(
            f1=0.3,
            precision=0.4,
            recall=0.2,
        )
        exact_winner["exactStateActions"] = exact_state_item(
            f1=0.8,
            precision=0.75,
            recall=0.85,
        )
        selection = select_action_temperature([
            surrogate_winner,
            exact_winner,
        ])
        self.assertEqual(selection["selectedLogitTemperature"], 1.0)
        self.assertEqual(selection["objective"], {
            "metric": "exactStateActions.signedTransitionF1",
            "direction": "maximize",
        })
        self.assertEqual(
            selection["selectedMetrics"]["exactStateActions"],
            exact_winner["exactStateActions"],
        )

    def test_action_temperature_selection_rejects_duplicate_candidates(self) -> None:
        with self.assertRaises(ValueError):
            select_action_temperature([
                temperature_item(1.0),
                temperature_item(1.0),
            ])


def temperature_item(
    temperature: float,
    *,
    f1: float = 0.2,
    precision: float = 0.2,
    recall: float = 0.2,
) -> dict:
    return {
        "logitTemperature": temperature,
        "raw": {"klDivergence": 123.0},
        "actions": {
            "signedTransitionF1": f1,
            "signedTransitionPrecision": precision,
            "signedTransitionRecall": recall,
            "exactTransitionF1": 0.1,
            "pathDirectionalAgreement": 0.5,
            "pathMeanAbsoluteError": 20.0,
            "turnoverRatio": 1.0,
        },
    }


def exact_state_item(
    *,
    f1: float,
    precision: float,
    recall: float,
) -> dict:
    return {
        "signedTransitionF1": f1,
        "signedTransitionPrecision": precision,
        "signedTransitionRecall": recall,
        "exactTransitionF1": 0.1,
        "executableTargetDirectionalAgreement": 0.5,
        "executableTargetMeanAbsoluteError": 20.0,
        "turnoverRelativeError": 0.2,
        "exactStateScore": 0.5,
    }


if __name__ == "__main__":
    unittest.main()

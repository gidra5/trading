from __future__ import annotations

import itertools
import unittest

import numpy as np
import torch

from eiil import (
    binary_eiil_objective,
    infer_binary_environment_ids,
    irm_v1_regression_objective,
    regression_scale_gradients,
)
from train_next_return_memorization import (
    eiil_irm_variant,
    validate_plan,
    with_validation_curve,
)
from train_normalized_glu_next_return import NextReturnDataset


class EiilInferenceTest(unittest.TestCase):
    def test_binary_assignment_reaches_global_hard_objective_maximum(self) -> None:
        gradients = np.array([-3.0, -0.5, 0.25, 2.0], dtype=np.float64)
        inferred = infer_binary_environment_ids(gradients)
        inferred_objective = binary_eiil_objective(gradients, inferred)
        possible = (
            binary_eiil_objective(gradients, np.array(ids, dtype=np.uint8))
            for ids in itertools.product((0, 1), repeat=gradients.size)
            if len(set(ids)) == 2
        )
        self.assertAlmostEqual(inferred_objective, max(possible))
        self.assertTrue(np.array_equal(inferred, np.array([1, 1, 0, 0])))

    def test_regression_scale_gradient_matches_autograd(self) -> None:
        predictions = torch.tensor([0.5, -0.25])
        targets = torch.tensor([0.1, 0.2])
        scale = torch.ones((), requires_grad=True)
        individual = ((predictions * scale - targets) / 2.0).square()
        expected = torch.stack([
            torch.autograd.grad(value, scale, retain_graph=True)[0]
            for value in individual
        ])
        actual = regression_scale_gradients(
            predictions, targets, target_std=2.0
        )
        self.assertTrue(torch.allclose(actual, expected))

    def test_assigned_batches_preserve_flattened_example_alignment(self) -> None:
        dataset = NextReturnDataset.__new__(NextReturnDataset)
        dataset.horizon_return_count = 1
        dataset._daily_group_rows = lambda _split: [
            ("first", np.array([0, 2], dtype=np.int64)),
            ("second", np.array([1], dtype=np.int64)),
        ]
        components = {
            "first": (
                np.arange(3 * 120, dtype=np.float32).reshape(3, 120),
                np.array([10.0, 11.0, 12.0], dtype=np.float32),
            ),
            "second": (
                np.arange(3 * 120, 6 * 120, dtype=np.float32).reshape(3, 120),
                np.array([20.0, 21.0, 22.0], dtype=np.float32),
            ),
        }
        dataset._component = lambda day: components[day]
        batches = list(dataset.iter_assigned_group_batches(
            "train",
            2,
            np.array([1, 0, 1], dtype=np.uint8),
            shuffle=False,
            seed=0,
        ))
        self.assertTrue(torch.equal(
            torch.cat([batch[1] for batch in batches]),
            torch.tensor([10.0, 12.0, 21.0]),
        ))
        self.assertTrue(torch.equal(
            torch.cat([batch[3] for batch in batches]),
            torch.tensor([1, 0, 1]),
        ))


class IrmV1RegressionTest(unittest.TestCase):
    def test_uniform_environment_risk_and_scale_penalty(self) -> None:
        predictions = torch.tensor([1.0, -1.0], requires_grad=True)
        objective, risk, penalty = irm_v1_regression_objective(
            predictions,
            torch.zeros(2),
            torch.ones(2),
            torch.tensor([0, 1]),
            target_std=1.0,
            penalty_weight=10.0,
        )
        self.assertAlmostEqual(float(risk.detach()), 1.0)
        self.assertAlmostEqual(float(penalty.detach()), 4.0)
        self.assertAlmostEqual(float(objective.detach()), 4.1, places=6)
        objective.backward()
        self.assertTrue(bool(torch.isfinite(predictions.grad).all()))

    def test_variant_has_clean_eiil_irm_contract(self) -> None:
        source = {
            "id": "base",
            "label": "Base",
            "datasetDir": "data/training/datasets/base",
            "runDir": "data/training/runs/base",
            "historyDir": "data/market/immutable/refs/candles/spot/btc/1s",
            "subset": {
                "type": "fixed-contiguous",
                "date": "2026-04-01",
                "examples": 16,
            },
            "architecture": {
                "widths": [8],
                "dropout": 0.0,
                "dropoutRate": 0.0,
                "initialRadius": 0.01,
                "minimumRadius": 0.0001,
                "learnableCentering": False,
            },
            "training": {
                "epochs": 2,
                "batchSize": 4,
                "evaluationBatchSize": 8,
                "learningRate": 0.0001,
                "targetNormalizedMse": 0.0001,
                "mixedPrecision": "float32",
                "device": "cpu",
            },
        }
        variant = eiil_irm_variant(
            source,
            reference_plan="data/training/runs/reference/state/plan.json",
            penalty_weight=10_000.0,
            penalty_anneal_steps=100,
            suffix="nonzero-eiil-irmv1-v1",
        )
        validate_plan(variant)
        self.assertEqual(
            variant["training"]["environmentInference"]["environmentCount"],
            2,
        )
        self.assertEqual(
            variant["training"]["invariantRiskMinimization"]["penaltyWeight"],
            10_000.0,
        )
        with_curve = with_validation_curve(
            variant,
            split_plan=(
                "ml/training-plans/"
                "direct-glu-to-next-1s-recent-4m-1-layer-long-v2.json"
            ),
            examples=65_536,
        )
        validate_plan(with_curve)
        self.assertEqual(with_curve["validationCurve"]["frequency"], "every-epoch")


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

import numpy as np
import torch

from joint_price_oracle import JointLossWeights
from train_joint_price_oracle import (
    DATA_CONTRACT,
    METRIC_NAMES,
    CausalOracleDataset,
    CausalSegment,
    causal_input_close_windows,
    evaluate,
    resolve_training_config,
    validate_plan,
)


class FakeCloseCache:
    def __init__(self) -> None:
        self.requests: list[tuple[int, int]] = []

    def range(self, first_close_time: int, last_close_time: int) -> np.ndarray:
        self.requests.append((first_close_time, last_close_time))
        count = (last_close_time - first_close_time) // 1_000 + 1
        return np.arange(1, count + 1, dtype=np.float32)


class FakeTargetCache:
    rows_per_file = 4
    action_count = 101

    def __init__(self, target_file: Path) -> None:
        self.target_file = target_file
        self.rows = torch.full((4, 101), 1 / 101)

    def load(self, target_file: Path) -> torch.Tensor:
        if target_file != self.target_file:
            raise AssertionError("unexpected target file")
        return self.rows


class PolicyOnlyEvaluationDataset:
    def __init__(self) -> None:
        self.targets = torch.tensor([
            [0.1, 0.2, 0.7],
            [0.2, 0.6, 0.2],
            [0.7, 0.2, 0.1],
        ])

    def iter_batches(
        self,
        split,
        batch_size,
        *,
        shuffle,
        seed,
        maximum_batches,
    ):
        del split, batch_size, seed, maximum_batches
        if shuffle:
            raise AssertionError("validation cannot shuffle")
        closes = torch.arange(1, 13, dtype=torch.float32).view(3, 4, 1)
        yield closes, self.targets


class PolicyOnlyEvaluationModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = torch.nn.Parameter(torch.tensor([-0.2, 0.1, 0.3]))
        self.forecast_calls = 0

    def forward_policy_logits(self, closes: torch.Tensor) -> torch.Tensor:
        return self.bias.unsqueeze(0).expand(closes.shape[0], -1)

    def forward_with_forecast(self, _closes: torch.Tensor):
        self.forecast_calls += 1
        raise AssertionError("forecast path executed")


class TrainJointPriceOraclePolicyOnlyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.plan_file = (
            cls.repo_root
            / "ml"
            / "training-plans"
            / "joint-price-oracle-kl-fast-patch-mixer-v15.json"
        )
        cls.plan = json.loads(cls.plan_file.read_text(encoding="utf-8"))

    def test_policy_only_loader_never_requests_future_closes(self) -> None:
        target_file = Path("targets.json")
        segment = CausalSegment(
            split="train",
            prediction_time_start=10_000,
            count=3,
            target_file=target_file,
            target_row_offset=0,
        )
        dataset = CausalOracleDataset(
            Path("unused-history"),
            {"train": [segment], "validation": [], "test": []},
            context_length=4,
            forecast_horizon=3_600,
            target_rows_per_file=4,
            action_count=101,
            include_future_closes=False,
        )
        close_cache = FakeCloseCache()
        dataset.close_cache = close_cache
        dataset.target_cache = FakeTargetCache(target_file)

        batch = next(dataset.iter_batches(
            "train",
            3,
            shuffle=False,
            seed=1,
            maximum_batches=1,
        ))
        self.assertEqual(len(batch), 2)
        input_closes, target_policy = batch
        self.assertEqual(tuple(input_closes.shape), (3, 4, 1))
        self.assertEqual(tuple(target_policy.shape), (3, 101))
        self.assertEqual(close_cache.requests, [(7_000, 12_000)])
        self.assertEqual(dataset.forecast_horizon, 3_600)
        torch.testing.assert_close(
            input_closes[:, -1, 0],
            torch.tensor([4.0, 5.0, 6.0]),
        )

    def test_policy_only_windows_are_prefix_causal(self) -> None:
        closes = np.arange(6, dtype=np.float32)
        original = causal_input_close_windows(
            closes,
            count=2,
            context_length=4,
            sample_step_seconds=2,
        )
        changed = closes.copy()
        changed[-2:] += 1_000
        modified = causal_input_close_windows(
            changed,
            count=2,
            context_length=4,
            sample_step_seconds=2,
        )
        np.testing.assert_array_equal(original[0], modified[0])
        self.assertFalse(np.array_equal(original[1], modified[1]))

    def test_policy_only_validation_skips_forecast_and_keeps_metric_schema(
        self,
    ) -> None:
        model = PolicyOnlyEvaluationModel()
        result = evaluate(
            model,
            PolicyOnlyEvaluationDataset(),
            "validation",
            3,
            torch.device("cpu"),
            {
                "policyOnly": True,
                "seed": 1,
                "prefetchBatches": 1,
                "mixedPrecision": "float32",
            },
            JointLossWeights(
                policy_cross_entropy=1,
                conditioned_policy_cross_entropy=0,
                forecast=0,
                soft_layer_norm=0,
            ),
            torch.tensor([-1.0, 0.0, 1.0]),
            0.00175,
            0.01,
            maximum_batches=None,
        )
        self.assertEqual(model.forecast_calls, 0)
        self.assertEqual(set(result), set(METRIC_NAMES))
        for name in (
            "forecastLoss",
            "nextMovementRmse",
            "directionAccuracy",
            "softLayerNorm",
        ):
            self.assertEqual(result[name], 0.0)

    def test_v15_plan_is_fast_policy_only_and_contract_preserving(self) -> None:
        validate_plan(self.plan)
        self.assertTrue(self.plan["training"]["policyOnly"])
        self.assertEqual(self.plan["model"]["contextLength"], 3_600)
        self.assertEqual(self.plan["model"]["forecastHorizon"], 3_600)
        self.assertEqual(self.plan["training"]["batchSize"], 1_440)
        self.assertEqual(self.plan["training"]["evaluationBatchSize"], 1_440)
        self.assertEqual(self.plan["training"]["epochs"], 192)
        self.assertEqual(self.plan["training"]["patience"], 64)
        self.assertEqual(
            DATA_CONTRACT,
            "causal-1s-close-context-ending-at-minute-t-future-closes-t-plus-1-"
            "through-1h-verified-oracle-policy-at-t-hold-60-delay-60-v2",
        )

    def test_policy_only_plan_rejects_every_incompatible_objective(self) -> None:
        for weight in (
            "conditionedPolicyCrossEntropy",
            "forecast",
            "softLayerNorm",
        ):
            with self.subTest(weight=weight):
                plan = copy.deepcopy(self.plan)
                plan["training"]["lossWeights"][weight] = 0.1
                with self.assertRaisesRegex(ValueError, "zero incompatible"):
                    validate_plan(plan)

        plan = copy.deepcopy(self.plan)
        plan["training"]["actionObjective"] = {}
        with self.assertRaisesRegex(ValueError, "actionObjective"):
            validate_plan(plan)

        plan = copy.deepcopy(self.plan)
        plan["training"]["selectionMetric"] = "loss"
        with self.assertRaisesRegex(ValueError, "raw klDivergence"):
            validate_plan(plan)

        plan = copy.deepcopy(self.plan)
        plan["training"]["policyOnly"] = "true"
        with self.assertRaisesRegex(ValueError, "must be a boolean"):
            validate_plan(plan)

        plan = copy.deepcopy(self.plan)
        plan["model"]["variant"] = "multiscale_residual_mixer"
        with self.assertRaisesRegex(ValueError, "forward_policy_logits"):
            validate_plan(plan)

    def test_policy_only_default_does_not_change_existing_run_fingerprints(
        self,
    ) -> None:
        resolved = resolve_training_config({
            "lossWeights": {
                "policyCrossEntropy": 1,
                "forecast": 0,
                "softLayerNorm": 0,
            },
        })
        self.assertNotIn("policyOnly", resolved)


if __name__ == "__main__":
    unittest.main()

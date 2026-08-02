from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from joint_price_oracle import parameter_count
from joint_price_oracle_minute_mlp import (
    ARCHITECTURE_CONTRACT as MINUTE_RETURN_MLP_CONTRACT,
    INPUT_CLOSE_COUNT,
    MinuteReturnOracleMlp,
)
from joint_price_oracle_variants import build_variant_model
from train_joint_price_oracle import (
    DATA_CONTRACT,
    CausalOracleDataset,
    CausalSegment,
    architecture_contract_for_model_config,
    atomic_torch_save,
    build_model,
    build_training_checkpoint,
    load_resume_checkpoint,
    validate_plan,
)


class _CloseCache:
    def __init__(self) -> None:
        self.requests: list[tuple[int, int]] = []

    def range(self, first_close_time: int, last_close_time: int) -> np.ndarray:
        self.requests.append((first_close_time, last_close_time))
        count = (last_close_time - first_close_time) // 1_000 + 1
        return np.arange(1, count + 1, dtype=np.float32)


class _TargetCache:
    rows_per_file = 1_440
    action_count = 101

    def __init__(self, target_file: Path) -> None:
        self.target_file = target_file
        self.rows = torch.full((self.rows_per_file, 101), 1 / 101)

    def load(self, target_file: Path) -> torch.Tensor:
        if target_file != self.target_file:
            raise AssertionError("unexpected target file")
        return self.rows


class MinuteReturnMlpIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.plan_file = (
            cls.repo_root
            / "ml"
            / "training-plans"
            / "joint-price-oracle-kl-minute-return-mlp-v16.json"
        )
        cls.plan = json.loads(cls.plan_file.read_text(encoding="utf-8"))

    @staticmethod
    def model_config(**overrides) -> dict:
        config = {
            "variant": "minute_return_mlp",
            "contextLength": 3_601,
            "forecastHorizon": 3_600,
            "variableCount": 1,
            "actionCount": 101,
            "hiddenWidth": 32,
            "layerCount": 3,
            "dropout": 0,
            "residualGain": 0.75,
        }
        config.update(overrides)
        return config

    @staticmethod
    def closes(batch: int = 2) -> torch.Tensor:
        generator = torch.Generator().manual_seed(161)
        returns = torch.randn(batch, 3_600, generator=generator) * 2e-5
        log_start = torch.full((batch, 1), 11.0)
        log_closes = torch.cat((
            log_start,
            log_start + returns.cumsum(dim=1),
        ), dim=1)
        return log_closes.exp().unsqueeze(-1)

    def test_factory_builds_explicit_logits_only_policy_variant(self) -> None:
        config = self.model_config()
        model = build_model(config).eval()
        self.assertIsInstance(model, MinuteReturnOracleMlp)
        self.assertEqual(model.context_length, INPUT_CLOSE_COUNT)
        self.assertEqual(model.forecast_horizon, 3_600)
        self.assertEqual(model.variable_count, 1)
        self.assertEqual(model.action_count, 101)
        self.assertEqual(model.config.hidden_width, 32)
        self.assertEqual(model.config.layer_count, 3)
        self.assertEqual(model.config.dropout, 0)
        self.assertEqual(model.config.residual_gain, 0.75)
        self.assertEqual(
            architecture_contract_for_model_config(config),
            MINUTE_RETURN_MLP_CONTRACT,
        )
        self.assertEqual(model.architecture_contract, MINUTE_RETURN_MLP_CONTRACT)
        self.assertFalse(hasattr(model, "forward_with_forecast"))
        closes = self.closes()
        with torch.no_grad():
            forward_logits = model(closes)
            policy_logits = model.forward_policy_logits(closes)
        self.assertEqual(tuple(forward_logits.shape), (2, 101))
        torch.testing.assert_close(forward_logits, policy_logits)

    def test_factory_enforces_the_exact_data_and_output_contract(self) -> None:
        cases = (
            ({"contextLength": 3_600}, "3,601 closes"),
            ({"forecastHorizon": 60}, "3,600 forecast"),
            ({"variableCount": 2}, "one close"),
            ({"actionCount": 100}, "101 actions"),
            ({"hiddenWidth": 0}, "hiddenWidth"),
            ({"layerCount": 0}, "layerCount"),
            ({"dropout": 1.0}, "dropout"),
            ({"residualGain": 0.0}, "residual_gain"),
            ({"policyLogitRank": 16}, "direct 101 logits"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    build_variant_model(self.model_config(**overrides))

    def test_v16_plan_is_raw_kl_and_long_enough_to_cross_plateaus(self) -> None:
        validate_plan(self.plan)
        model = self.plan["model"]
        training = self.plan["training"]
        schedule = training["learningRateSchedule"]
        self.assertEqual(self.plan["version"], 16)
        self.assertEqual(model["variant"], "minute_return_mlp")
        self.assertEqual(model["contextLength"], 3_601)
        self.assertEqual(model["forecastHorizon"], 3_600)
        self.assertEqual(model["variableCount"], 1)
        self.assertEqual(model["actionCount"], 101)
        self.assertTrue(training["policyOnly"])
        self.assertEqual(training["selectionMetric"], "klDivergence")
        self.assertEqual(training["batchSize"], 1_440)
        self.assertEqual(training["evaluationBatchSize"], 1_440)
        self.assertGreaterEqual(training["epochs"], 256)
        self.assertGreaterEqual(schedule["patience"], 40)
        self.assertLessEqual(schedule["patience"], 50)
        self.assertGreaterEqual(training["patience"], 120)
        self.assertGreater(training["patience"], 2 * schedule["patience"])
        self.assertEqual(training["lossWeights"], {
            "policyCrossEntropy": 1,
            "conditionedPolicyCrossEntropy": 0,
            "forecast": 0,
            "softLayerNorm": 0,
        })

        incompatible = copy.deepcopy(self.plan)
        incompatible["training"]["policyOnly"] = False
        with self.assertRaisesRegex(ValueError, "policy-only variant"):
            validate_plan(incompatible)

    def test_policy_only_loader_reads_3601_closes_through_t_and_no_future(
        self,
    ) -> None:
        target_file = Path("targets.json")
        prediction_time = 10_000_000
        segment = CausalSegment(
            split="train",
            prediction_time_start=prediction_time,
            count=2,
            target_file=target_file,
            target_row_offset=0,
            step_ms=60_000,
        )
        dataset = CausalOracleDataset(
            Path("unused-history"),
            {"train": [segment], "validation": [], "test": []},
            context_length=3_601,
            forecast_horizon=3_600,
            target_rows_per_file=1_440,
            action_count=101,
            include_future_closes=False,
        )
        close_cache = _CloseCache()
        dataset.close_cache = close_cache
        dataset.target_cache = _TargetCache(target_file)

        batch = next(dataset.iter_batches(
            "train",
            2,
            shuffle=False,
            seed=16,
            maximum_batches=1,
        ))
        self.assertEqual(len(batch), 2)
        closes, targets = batch
        self.assertEqual(tuple(closes.shape), (2, 3_601, 1))
        self.assertEqual(tuple(targets.shape), (2, 101))
        self.assertEqual(close_cache.requests, [(
            prediction_time - 3_600_000,
            prediction_time + 60_000,
        )])
        torch.testing.assert_close(
            closes[:, -1, 0],
            torch.tensor([3_601.0, 3_661.0]),
        )
        self.assertEqual(
            DATA_CONTRACT,
            "causal-1s-close-context-ending-at-minute-t-future-closes-t-plus-1-"
            "through-1h-verified-oracle-policy-at-t-hold-60-delay-60-v2",
        )

    def test_checkpoint_resume_is_bound_to_minute_mlp_config_and_contract(
        self,
    ) -> None:
        config = self.model_config()
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=48,
        )
        fingerprint = "v16-training-fingerprint"
        count = parameter_count(model)
        checkpoint = build_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=7,
            global_step=91,
            best_validation=0.2,
            best_epoch=7,
            stale_epochs=0,
            validation={"klDivergence": 0.2},
            model_parameters=count,
            dataset_fingerprint="v16-dataset-fingerprint",
            model_config=config,
            plan_id="minute-return-mlp-resume-test",
            device=torch.device("cpu"),
            interrupted=False,
            training_config_fingerprint=fingerprint,
        )
        self.assertEqual(
            checkpoint["architectureContract"],
            MINUTE_RETURN_MLP_CONTRACT,
        )
        self.assertEqual(checkpoint["dataContract"], DATA_CONTRACT)
        self.assertEqual(checkpoint["modelConfig"], config)

        with tempfile.TemporaryDirectory() as directory:
            file = (
                Path(directory)
                / "data"
                / "training"
                / "runs"
                / "minute-return-mlp-resume-test"
                / "checkpoints"
                / "last.json"
            )
            atomic_torch_save(checkpoint, file)
            resumed = load_resume_checkpoint(
                file,
                model,
                optimizer,
                scheduler,
                {"id": "minute-return-mlp-resume-test"},
                config,
                "v16-dataset-fingerprint",
                count,
                torch.device("cpu"),
                training_config_fingerprint=fingerprint,
            )
            self.assertEqual(resumed, (8, 91, 0.2, 7, 0))

            drifted = copy.deepcopy(checkpoint)
            drifted["modelConfig"]["hiddenWidth"] = 48
            atomic_torch_save(drifted, file)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": "minute-return-mlp-resume-test"},
                    config,
                    "v16-dataset-fingerprint",
                    count,
                    torch.device("cpu"),
                    training_config_fingerprint=fingerprint,
                )


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from joint_price_oracle import parameter_count
from joint_price_oracle_sequence import (
    INPUT_CLOSE_COUNT,
    SEQUENCE_ARCHITECTURE_CONTRACT,
    ChronologicalMinutePolicyModel,
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


class MinuteSequenceTcnIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        cls.plan_file = (
            cls.repo_root
            / "ml"
            / "training-plans"
            / "joint-price-oracle-kl-minute-sequence-tcn-v17.json"
        )
        cls.plan = json.loads(cls.plan_file.read_text(encoding="utf-8"))

    @staticmethod
    def model_config(**overrides) -> dict:
        config = {
            "variant": "minute_sequence_tcn",
            "contextLength": 3_600,
            "forecastHorizon": 3_600,
            "variableCount": 1,
            "actionCount": 101,
            "receptiveFieldMinutes": 60,
            "tokenWidth": 8,
            "policyHiddenWidth": 12,
            "dropout": 0,
            "featureEpsilon": 1e-8,
        }
        config.update(overrides)
        return config

    @staticmethod
    def closes(batch: int = 2) -> torch.Tensor:
        time = torch.arange(3_600, dtype=torch.float32).view(1, -1, 1)
        offset = torch.arange(batch, dtype=torch.float32).view(-1, 1, 1)
        return torch.exp(10.5 + time * 1e-6 + offset * 1e-4)

    def test_factory_maps_exact_policy_only_contract(self) -> None:
        config = self.model_config()
        model = build_model(config).eval()
        self.assertIsInstance(model, ChronologicalMinutePolicyModel)
        self.assertEqual(model.context_length, INPUT_CLOSE_COUNT)
        self.assertEqual(model.receptive_field_minutes, 60)
        self.assertEqual(model.forecast_horizon, 3_600)
        self.assertEqual(model.variable_count, 1)
        self.assertEqual(model.action_count, 101)
        self.assertEqual(
            architecture_contract_for_model_config(config),
            SEQUENCE_ARCHITECTURE_CONTRACT,
        )
        self.assertEqual(
            model.architecture_contract,
            SEQUENCE_ARCHITECTURE_CONTRACT,
        )
        self.assertFalse(hasattr(model, "forward_with_forecast"))
        closes = self.closes()
        with torch.no_grad():
            direct = model(closes)
            policy = model.forward_policy_logits(closes)
            single = model.forward_single(closes)
        self.assertEqual(tuple(policy.shape), (2, 101))
        torch.testing.assert_close(direct, policy)
        torch.testing.assert_close(single, policy)

    def test_factory_rejects_contract_drift(self) -> None:
        cases = (
            ({"contextLength": 3_601}, "3,600 closes"),
            ({"forecastHorizon": 60}, "3,600 forecast"),
            ({"variableCount": 2}, "one close"),
            ({"actionCount": 100}, "101 actions"),
            ({"receptiveFieldMinutes": 61}, "60-minute"),
            ({"tokenWidth": 0}, "tokenWidth"),
            ({"policyHiddenWidth": 0}, "policyHiddenWidth"),
            ({"dropout": 1}, "dropout"),
            ({"featureEpsilon": 0}, "epsilon"),
            ({"policyLogitRank": 16}, "direct 101 logits"),
        )
        for overrides, message in cases:
            with self.subTest(overrides=overrides):
                with self.assertRaisesRegex(ValueError, message):
                    build_variant_model(self.model_config(**overrides))

    def test_v17_plan_is_raw_kl_and_saturation_sized(self) -> None:
        validate_plan(self.plan)
        model = self.plan["model"]
        training = self.plan["training"]
        schedule = training["learningRateSchedule"]
        self.assertEqual(self.plan["version"], 17)
        self.assertEqual(model["variant"], "minute_sequence_tcn")
        self.assertEqual(model["contextLength"], 3_600)
        self.assertEqual(model["receptiveFieldMinutes"], 60)
        self.assertTrue(training["policyOnly"])
        self.assertEqual(training["selectionMetric"], "klDivergence")
        self.assertEqual(training["batchSize"], 1_440)
        self.assertEqual(training["evaluationBatchSize"], 1_440)
        self.assertEqual(training["epochs"], 384)
        self.assertEqual(schedule["patience"], 48)
        self.assertEqual(training["patience"], 160)
        self.assertEqual(self.plan["oracleTarget"]["options"]["temperature"], 0.01)
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

    def test_policy_only_loader_reads_current_3600_closes_and_no_future(
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
            context_length=3_600,
            forecast_horizon=3_600,
            target_rows_per_file=1_440,
            action_count=101,
            include_future_closes=False,
        )
        close_cache = _CloseCache()
        dataset.close_cache = close_cache
        dataset.target_cache = _TargetCache(target_file)

        closes, targets = next(dataset.iter_batches(
            "train",
            2,
            shuffle=False,
            seed=17,
            maximum_batches=1,
        ))
        self.assertEqual(tuple(closes.shape), (2, 3_600, 1))
        self.assertEqual(tuple(targets.shape), (2, 101))
        self.assertEqual(close_cache.requests, [(
            prediction_time - 3_599_000,
            prediction_time + 60_000,
        )])
        torch.testing.assert_close(
            closes[:, -1, 0],
            torch.tensor([3_600.0, 3_660.0]),
        )
        self.assertEqual(
            DATA_CONTRACT,
            "causal-1s-close-context-ending-at-minute-t-future-closes-t-plus-1-"
            "through-1h-verified-oracle-policy-at-t-hold-60-delay-60-v2",
        )

    def test_factory_model_exports_with_direct_logits(self) -> None:
        model = build_model(self.model_config()).eval()
        example = self.closes(batch=1)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "minute-sequence-tcn.onnx"
            torch.onnx.export(
                model,
                example,
                output,
                input_names=["closes"],
                output_names=["action_logits"],
                dynamic_axes={
                    "closes": {0: "batch"},
                    "action_logits": {0: "batch"},
                },
                opset_version=18,
                do_constant_folding=True,
                external_data=False,
                dynamo=False,
            )
            exported = onnx.load(output, load_external_data=True)
            onnx.checker.check_model(exported, full_check=True)
            exported_logits = ReferenceEvaluator(exported).run(
                ["action_logits"],
                {"closes": example.numpy()},
            )[0]
        with torch.no_grad():
            expected_logits = model.forward_policy_logits(example).numpy()
        self.assertLessEqual(
            float(np.max(np.abs(expected_logits - exported_logits))),
            1e-4,
        )

    def test_checkpoint_resume_is_bound_to_v17_config_and_contract(self) -> None:
        config = self.model_config()
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=48,
        )
        fingerprint = "v17-training-fingerprint"
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
            dataset_fingerprint="v17-dataset-fingerprint",
            model_config=config,
            plan_id="minute-sequence-tcn-resume-test",
            device=torch.device("cpu"),
            interrupted=False,
            training_config_fingerprint=fingerprint,
        )
        self.assertEqual(
            checkpoint["architectureContract"],
            SEQUENCE_ARCHITECTURE_CONTRACT,
        )
        self.assertEqual(checkpoint["dataContract"], DATA_CONTRACT)
        self.assertEqual(checkpoint["modelConfig"], config)

        with tempfile.TemporaryDirectory() as directory:
            file = (
                Path(directory)
                / "data"
                / "training"
                / "runs"
                / "minute-sequence-tcn-resume-test"
                / "checkpoints"
                / "last.json"
            )
            atomic_torch_save(checkpoint, file)
            resumed = load_resume_checkpoint(
                file,
                model,
                optimizer,
                scheduler,
                {"id": "minute-sequence-tcn-resume-test"},
                config,
                "v17-dataset-fingerprint",
                count,
                torch.device("cpu"),
                training_config_fingerprint=fingerprint,
            )
            self.assertEqual(resumed, (8, 91, 0.2, 7, 0))

            drifted = copy.deepcopy(checkpoint)
            drifted["modelConfig"]["tokenWidth"] = 16
            atomic_torch_save(drifted, file)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": "minute-sequence-tcn-resume-test"},
                    config,
                    "v17-dataset-fingerprint",
                    count,
                    torch.device("cpu"),
                    training_config_fingerprint=fingerprint,
                )


if __name__ == "__main__":
    unittest.main()

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
    BOUNDARY_INPUT_CLOSE_COUNT,
    BOUNDARY_MA_SEQUENCE_ARCHITECTURE_CONTRACT,
    BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
    BoundaryCompleteMinutePolicyModel,
    BoundaryMaMinutePolicyModel,
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


class CorrectedMinuteSequenceIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.repo_root = Path(__file__).resolve().parents[1]
        plan_root = cls.repo_root / "ml" / "training-plans"
        cls.plans = (
            json.loads((
                plan_root
                / "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18.json"
            ).read_text(encoding="utf-8")),
            json.loads((
                plan_root
                / "joint-price-oracle-kl-minute-sequence-boundary-ma-tcn-v19.json"
            ).read_text(encoding="utf-8")),
        )

    @staticmethod
    def model_config(variant: str, **overrides) -> dict:
        config = {
            "variant": variant,
            "contextLength": BOUNDARY_INPUT_CLOSE_COUNT,
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
    def closes(batch_size: int = 1) -> torch.Tensor:
        time = torch.arange(
            BOUNDARY_INPUT_CLOSE_COUNT,
            dtype=torch.float32,
        ).view(1, -1, 1)
        offset = torch.arange(
            batch_size,
            dtype=torch.float32,
        ).view(-1, 1, 1)
        return torch.exp(10.5 + time * 1e-6 + offset * 1e-4)

    def test_factories_map_new_immutable_direct_contracts(self) -> None:
        cases = (
            (
                "minute_sequence_boundary_tcn",
                BoundaryCompleteMinutePolicyModel,
                BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
            ),
            (
                "minute_sequence_boundary_ma_tcn",
                BoundaryMaMinutePolicyModel,
                BOUNDARY_MA_SEQUENCE_ARCHITECTURE_CONTRACT,
            ),
        )
        for variant, model_type, contract in cases:
            with self.subTest(variant=variant):
                config = self.model_config(variant)
                model = build_model(config).eval()
                self.assertIsInstance(model, model_type)
                self.assertEqual(model.context_length, 3_601)
                self.assertEqual(model.architecture_contract, contract)
                self.assertEqual(
                    architecture_contract_for_model_config(config),
                    contract,
                )
                self.assertFalse(hasattr(model, "forward_with_forecast"))
                with torch.no_grad():
                    direct = model(self.closes())
                    policy = model.forward_policy_logits(self.closes())
                self.assertEqual(tuple(policy.shape), (1, 101))
                torch.testing.assert_close(direct, policy)

    def test_factories_reject_contract_drift(self) -> None:
        for variant in (
            "minute_sequence_boundary_tcn",
            "minute_sequence_boundary_ma_tcn",
        ):
            cases = (
                ({"contextLength": 3_600}, "3,601 closes"),
                ({"receptiveFieldMinutes": 61}, "60-minute"),
                ({"policyLogitRank": 16}, "direct 101 logits"),
                ({"tokenWidth": 0}, "tokenWidth"),
                ({"policyHiddenWidth": 0}, "policyHiddenWidth"),
                ({"dropout": 1}, "dropout"),
                ({"featureEpsilon": 0}, "epsilon"),
            )
            for overrides, message in cases:
                with self.subTest(variant=variant, overrides=overrides):
                    with self.assertRaisesRegex(ValueError, message):
                        build_variant_model(self.model_config(
                            variant,
                            **overrides,
                        ))

    def test_plans_are_direct_production_temperature_raw_kl(self) -> None:
        expected = (
            (18, "minute_sequence_boundary_tcn"),
            (19, "minute_sequence_boundary_ma_tcn"),
        )
        for plan, (version, variant) in zip(self.plans, expected):
            with self.subTest(variant=variant):
                validate_plan(plan)
                self.assertEqual(plan["version"], version)
                self.assertEqual(plan["model"]["variant"], variant)
                self.assertEqual(plan["model"]["contextLength"], 3_601)
                self.assertEqual(
                    plan["oracleTarget"]["options"]["temperature"],
                    0.01,
                )
                training = plan["training"]
                self.assertTrue(training["policyOnly"])
                self.assertEqual(
                    training["selectionMetric"],
                    "klDivergence",
                )
                self.assertEqual(training["epochs"], 384)
                self.assertEqual(training["batchSize"], 1_440)
                self.assertEqual(training["evaluationBatchSize"], 1_440)
                self.assertEqual(
                    training["learningRateSchedule"]["patience"],
                    48,
                )
                self.assertEqual(training["patience"], 160)
                self.assertNotIn("calibration", training)
                self.assertEqual(training["lossWeights"], {
                    "policyCrossEntropy": 1,
                    "conditionedPolicyCrossEntropy": 0,
                    "forecast": 0,
                    "softLayerNorm": 0,
                })

                incompatible = copy.deepcopy(plan)
                incompatible["training"]["policyOnly"] = False
                with self.assertRaisesRegex(ValueError, "policy-only variant"):
                    validate_plan(incompatible)

    def test_policy_only_loader_reads_boundary_close_and_no_future(self) -> None:
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
        closes, targets = next(dataset.iter_batches(
            "train",
            2,
            shuffle=False,
            seed=18,
            maximum_batches=1,
        ))
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

    def test_new_contracts_round_trip_durable_resume(self) -> None:
        cases = (
            (
                "minute_sequence_boundary_tcn",
                BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
            ),
            (
                "minute_sequence_boundary_ma_tcn",
                BOUNDARY_MA_SEQUENCE_ARCHITECTURE_CONTRACT,
            ),
        )
        for variant, contract in cases:
            with self.subTest(variant=variant):
                config = self.model_config(variant)
                model = build_model(config)
                optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    patience=48,
                )
                fingerprint = f"{variant}-training-fingerprint"
                dataset_fingerprint = (
                    "a9e75c9edd2660190d77b0933b1f7ccc93d107531fdd6d04fd90357b6d8a3783"
                )
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
                    dataset_fingerprint=dataset_fingerprint,
                    model_config=config,
                    plan_id=variant,
                    device=torch.device("cpu"),
                    interrupted=False,
                    training_config_fingerprint=fingerprint,
                )
                self.assertEqual(checkpoint["architectureContract"], contract)
                self.assertEqual(checkpoint["dataContract"], DATA_CONTRACT)
                with tempfile.TemporaryDirectory() as directory:
                    file = (
                        Path(directory)
                        / "data"
                        / "training"
                        / "runs"
                        / variant
                        / "checkpoints"
                        / "last.json"
                    )
                    atomic_torch_save(checkpoint, file)
                    resumed = load_resume_checkpoint(
                        file,
                        model,
                        optimizer,
                        scheduler,
                        {"id": variant},
                        config,
                        dataset_fingerprint,
                        count,
                        torch.device("cpu"),
                        training_config_fingerprint=fingerprint,
                    )
                self.assertEqual(resumed, (8, 91, 0.2, 7, 0))

    def test_both_new_contracts_export_direct_logits(self) -> None:
        for variant in (
            "minute_sequence_boundary_tcn",
            "minute_sequence_boundary_ma_tcn",
        ):
            with self.subTest(variant=variant):
                model = build_model(self.model_config(variant)).eval()
                example = self.closes()
                with tempfile.TemporaryDirectory() as directory:
                    output = Path(directory) / f"{variant}.onnx"
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
                    expected = model.forward_policy_logits(example).numpy()
                self.assertLessEqual(
                    float(np.max(np.abs(expected - exported_logits))),
                    1e-4,
                )


if __name__ == "__main__":
    unittest.main()

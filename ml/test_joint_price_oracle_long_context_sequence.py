from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from joint_price_oracle import JointLossWeights, parameter_count
from joint_price_oracle_minute_mlp import fixed_scale_minute_features
from joint_price_oracle_sequence import (
    BOUNDARY_LONG_SEQUENCE_ARCHITECTURE_CONTRACT,
    BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
    BoundaryCompleteLongContextMinutePolicyModel,
    BoundaryCompleteMinutePolicyModel,
    BoundaryCompleteMinuteTokenizer,
    boundary_sequence_core_alignment,
    exact_receptive_field_dilations,
    last_hour_scale_minute_features,
)
from train_joint_price_oracle import (
    DATA_CONTRACT,
    SEQUENCE_REUSE_TRAINING_CONTRACT,
    architecture_contract_for_model_config,
    atomic_torch_save,
    build_model,
    build_training_checkpoint,
    causal_input_close_windows,
    configuration_fingerprint,
    load_resume_checkpoint,
    policy_only_forward_objective,
    resolve_training_config,
    sequence_core_policy_forward_objective,
    validate_plan,
)


LONG_CONTEXT_MINUTES = 360
LONG_CONTEXT_CLOSE_COUNT = LONG_CONTEXT_MINUTES * 60 + 1


def closes_from_second_returns(second_returns: torch.Tensor) -> torch.Tensor:
    boundary = torch.full(
        (second_returns.shape[0], 1),
        10.5,
        dtype=second_returns.dtype,
    )
    return torch.exp(torch.cat((
        boundary,
        boundary + second_returns.cumsum(dim=1),
    ), dim=1)).unsqueeze(-1)


def contiguous_long_closes(core_count: int) -> torch.Tensor:
    token_count = LONG_CONTEXT_MINUTES - 1 + core_count
    seconds = torch.arange(token_count * 60, dtype=torch.float32)
    returns = (
        torch.sin(seconds / 41.0) * 8e-6
        + torch.cos(seconds / 173.0) * 3e-6
    ).view(1, -1)
    return closes_from_second_returns(returns)


def fixed_long_windows(
    closes: torch.Tensor,
    core_count: int,
) -> torch.Tensor:
    values = causal_input_close_windows(
        closes[0, :, 0].numpy(),
        core_count,
        LONG_CONTEXT_CLOSE_COUNT,
        sample_step_seconds=60,
    )
    return torch.from_numpy(values[:, :, None])


class LongContextBoundarySequenceTest(unittest.TestCase):
    @staticmethod
    def model() -> BoundaryCompleteLongContextMinutePolicyModel:
        torch.manual_seed(281)
        return BoundaryCompleteLongContextMinutePolicyModel(
            token_width=4,
            policy_hidden_width=6,
            dropout=0,
        )

    def test_six_hour_boundary_alignment_preserves_every_transition(
        self,
    ) -> None:
        start = 1_800_000_000_000 + 999
        start -= start % 60_000 - 999
        alignment = boundary_sequence_core_alignment(
            start,
            7,
            LONG_CONTEXT_MINUTES,
        )
        self.assertEqual(alignment.halo_minutes, 359)
        self.assertEqual(alignment.token_count, 366)
        self.assertEqual(alignment.close_count, 366 * 60 + 1)
        self.assertEqual(alignment.sequence_index(0), 359)
        self.assertEqual(alignment.sequence_index(6), 365)
        self.assertEqual(
            alignment.input_close_time_start,
            start - 21_600_000,
        )
        self.assertEqual(
            alignment.input_close_time_start
            + (alignment.close_count - 1) * 1_000,
            alignment.input_close_time_end,
        )

    def test_long_encoder_scale_bypass_uses_only_the_last_hour(self) -> None:
        torch.manual_seed(283)
        returns = torch.randn(2, LONG_CONTEXT_MINUTES) * 1e-4
        changed = returns.clone()
        changed[:, :-60] += torch.linspace(
            -8e-4,
            8e-4,
            LONG_CONTEXT_MINUTES - 60,
        )
        _scaled, baseline = last_hour_scale_minute_features(returns)
        _changed_scaled, perturbed = last_hour_scale_minute_features(changed)
        _expected_scaled, expected = fixed_scale_minute_features(
            returns[:, -60:]
        )
        torch.testing.assert_close(baseline, expected, atol=0, rtol=0)
        torch.testing.assert_close(perturbed, expected, atol=0, rtol=0)

        model = self.model()
        self.assertEqual(model.scale_window_minutes, 60)
        self.assertEqual(model.receptive_field_minutes, 360)

    def test_reused_logits_equal_all_overlapping_fixed_windows(self) -> None:
        core_count = 4
        closes = contiguous_long_closes(core_count)
        windows = fixed_long_windows(closes, core_count)
        model = self.model().eval()
        with torch.no_grad():
            reused = model.forward_sequence_core(closes)[0]
            fixed = model.forward_policy_logits(windows)
        self.assertEqual(tuple(reused.shape), (core_count, 101))
        torch.testing.assert_close(reused, fixed, atol=0, rtol=0)

    def test_reused_objective_and_gradients_match_fixed_windows(self) -> None:
        core_count = 3
        closes = contiguous_long_closes(core_count)
        windows = fixed_long_windows(closes, core_count)
        torch.manual_seed(287)
        targets = torch.softmax(
            torch.randn(core_count, 101),
            dim=-1,
        )
        reused_model = self.model().train()
        fixed_model = copy.deepcopy(reused_model).train()
        weights = JointLossWeights(forecast=0, soft_layer_norm=0)
        reused_logits, reused_metrics = (
            sequence_core_policy_forward_objective(
                reused_model,
                closes,
                targets.unsqueeze(0),
                weights,
            )
        )
        fixed_logits, fixed_metrics = policy_only_forward_objective(
            fixed_model,
            windows,
            targets,
            weights,
        )
        reused_metrics["loss"].backward()
        fixed_metrics["loss"].backward()

        torch.testing.assert_close(
            reused_logits[0],
            fixed_logits,
            atol=0,
            rtol=0,
        )
        for name in (
            "loss",
            "crossEntropy",
            "klDivergence",
            "probabilityMse",
        ):
            torch.testing.assert_close(
                reused_metrics[name],
                fixed_metrics[name],
                atol=0,
                rtol=0,
            )
        for (reused_name, reused), (fixed_name, fixed) in zip(
            reused_model.named_parameters(),
            fixed_model.named_parameters(),
        ):
            self.assertEqual(reused_name, fixed_name)
            assert reused.grad is not None and fixed.grad is not None
            torch.testing.assert_close(
                reused.grad,
                fixed.grad,
                atol=2e-7,
                rtol=2e-5,
            )

    def test_long_model_reuses_v18_tokenizer_and_exact_tcn_topology(
        self,
    ) -> None:
        model = self.model()
        self.assertIsInstance(model.tokenizer, BoundaryCompleteMinuteTokenizer)
        self.assertEqual(
            model.dilations,
            exact_receptive_field_dilations(360),
        )
        self.assertEqual(1 + sum(model.dilations), 360)
        self.assertTrue(model.supports_sequence_core_reuse)
        self.assertEqual(model.context_length, LONG_CONTEXT_CLOSE_COUNT)
        self.assertEqual(
            model.architecture_contract,
            BOUNDARY_LONG_SEQUENCE_ARCHITECTURE_CONTRACT,
        )


class LongContextPlanAndCheckpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parent / "training-plans"
        cls.plan = json.loads((root /
            "joint-price-oracle-kl-minute-sequence-boundary-long-tcn-v28-reuse.json"
        ).read_text(encoding="utf-8"))
        cls.v18 = json.loads((root /
            "joint-price-oracle-kl-minute-sequence-boundary-tcn-v18-reuse.json"
        ).read_text(encoding="utf-8"))

    def test_v28_is_sealed_raw_kl_and_optimizer_matched_to_v18(self) -> None:
        validate_plan(self.plan)
        self.assertEqual(self.plan["version"], 28)
        self.assertEqual(self.plan["testPolicy"], "sealed-never-load")
        self.assertNotEqual(self.plan["id"], self.v18["id"])
        self.assertNotEqual(self.plan["runDir"], self.v18["runDir"])
        self.assertEqual(self.plan["training"], self.v18["training"])
        self.assertNotIn("evaluateTest", self.plan["training"])
        self.assertNotIn("calibration", self.plan["training"])
        self.assertEqual(
            self.plan["training"]["selectionMetric"],
            "klDivergence",
        )
        self.assertEqual(
            self.plan["training"]["sequenceCoreTraining"],
            {
                "contract": SEQUENCE_REUSE_TRAINING_CONTRACT,
                "coreRows": 1_440,
            },
        )
        self.assertEqual(
            self.plan["oracleTarget"]["options"]["temperature"],
            0.01,
        )
        model_config = self.plan["model"]
        self.assertEqual(model_config["contextLength"], 21_601)
        self.assertEqual(model_config["receptiveFieldMinutes"], 360)
        self.assertEqual(model_config["scaleWindowMinutes"], 60)
        self.assertEqual(
            model_config["tokenWidth"],
            self.v18["model"]["tokenWidth"],
        )
        self.assertEqual(
            model_config["policyHiddenWidth"],
            self.v18["model"]["policyHiddenWidth"],
        )
        self.assertEqual(model_config["dropout"], 0)
        self.assertEqual(
            architecture_contract_for_model_config(model_config),
            BOUNDARY_LONG_SEQUENCE_ARCHITECTURE_CONTRACT,
        )

    def test_v28_rejects_context_receptive_field_and_scale_drift(self) -> None:
        cases = (
            ("contextLength", 3_601, "21,601"),
            ("receptiveFieldMinutes", 60, "360-minute"),
            ("scaleWindowMinutes", 360, "exactly 60"),
        )
        for key, value, message in cases:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.plan)
                changed["model"][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    validate_plan(changed)

    def test_v28_checkpoint_isolated_from_legacy_and_config_drift(self) -> None:
        config = copy.deepcopy(self.plan["model"])
        config["tokenWidth"] = 4
        config["policyHiddenWidth"] = 6
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=48,
        )
        training_fingerprint = configuration_fingerprint(
            resolve_training_config(self.plan["training"])
        )
        dataset_fingerprint = "v28-six-hour-dataset-fingerprint"
        count = parameter_count(model)
        checkpoint = build_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=2,
            global_step=744,
            best_validation=0.95,
            best_epoch=2,
            stale_epochs=0,
            validation={"klDivergence": 0.95},
            model_parameters=count,
            dataset_fingerprint=dataset_fingerprint,
            model_config=config,
            plan_id=self.plan["id"],
            device=torch.device("cpu"),
            interrupted=False,
            training_config_fingerprint=training_fingerprint,
        )
        self.assertEqual(checkpoint["dataContract"], DATA_CONTRACT)
        self.assertEqual(
            checkpoint["architectureContract"],
            BOUNDARY_LONG_SEQUENCE_ARCHITECTURE_CONTRACT,
        )
        with tempfile.TemporaryDirectory() as directory:
            file = (
                Path(directory)
                / "data"
                / "training"
                / "runs"
                / "v28"
                / "checkpoints"
                / "last.json"
            )
            atomic_torch_save(checkpoint, file)
            resumed = load_resume_checkpoint(
                file,
                model,
                optimizer,
                scheduler,
                {"id": self.plan["id"]},
                config,
                dataset_fingerprint,
                count,
                torch.device("cpu"),
                training_config_fingerprint=training_fingerprint,
            )
            self.assertEqual(resumed, (3, 744, 0.95, 2, 0))

            changed = copy.deepcopy(config)
            changed["scaleWindowMinutes"] = 59
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": self.plan["id"]},
                    changed,
                    dataset_fingerprint,
                    count,
                    torch.device("cpu"),
                    training_config_fingerprint=training_fingerprint,
                )

            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_resume_checkpoint(
                    file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": self.v18["id"]},
                    config,
                    dataset_fingerprint,
                    count,
                    torch.device("cpu"),
                    training_config_fingerprint=training_fingerprint,
                )

    def test_legacy_sixty_minute_contract_and_scale_are_unchanged(self) -> None:
        config = copy.deepcopy(self.v18["model"])
        config["tokenWidth"] = 8
        config["policyHiddenWidth"] = 12
        model = build_model(config)
        self.assertIsInstance(model, BoundaryCompleteMinutePolicyModel)
        self.assertEqual(model.context_length, 3_601)
        self.assertEqual(model.receptive_field_minutes, 60)
        self.assertEqual(model.scale_window_minutes, 60)
        self.assertEqual(
            model.dilations,
            exact_receptive_field_dilations(60),
        )
        self.assertEqual(
            model.architecture_contract,
            BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
        )
        self.assertEqual(
            architecture_contract_for_model_config(config),
            BOUNDARY_SEQUENCE_ARCHITECTURE_CONTRACT,
        )

        torch.manual_seed(293)
        minute_returns = torch.randn(3, 60) * 1e-4
        expected = fixed_scale_minute_features(minute_returns)
        actual = last_hour_scale_minute_features(minute_returns)
        torch.testing.assert_close(actual[0], expected[0], atol=0, rtol=0)
        torch.testing.assert_close(actual[1], expected[1], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()

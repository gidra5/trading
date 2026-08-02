from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

import torch

from joint_price_oracle import parameter_count
from joint_price_oracle_variants import (
    LEARNED_AGGREGATE_PATCH_MIXER_CONTRACT,
)
from train_joint_price_oracle import (
    atomic_torch_save,
    architecture_contract_for_model_config,
    build_model,
    build_training_checkpoint,
    configuration_fingerprint,
    load_resume_checkpoint,
    resolve_training_config,
    validate_plan,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_FILE = (
    ROOT
    / "ml"
    / "training-plans"
    / "joint-price-oracle-kl-learned-aggregate-patch-mixer-v20.json"
)


def _plan() -> dict:
    return json.loads(PLAN_FILE.read_text(encoding="utf-8"))


def _closes(batch: int = 2) -> torch.Tensor:
    time = torch.arange(3_600, dtype=torch.float32)
    returns = (
        torch.sin(time / 37) * 1e-4
        + torch.cos(time / 131) * 5e-5
    )
    path = torch.exp(10.5 + returns.cumsum(dim=0))
    return path.view(1, -1, 1).repeat(batch, 1, 1)


class LearnedAggregateArchitectureTest(unittest.TestCase):
    def test_compute_matched_shape_contract_and_policy_only_path(self) -> None:
        plan = _plan()
        config = copy.deepcopy(plan["model"])
        config["dropout"] = 0
        model = build_model(config).eval()
        self.assertEqual(parameter_count(model), 342_045)
        self.assertEqual(
            model.architecture_contract,
            LEARNED_AGGREGATE_PATCH_MIXER_CONTRACT,
        )
        self.assertEqual(model.patch_sizes, (30, 120, 600))
        self.assertEqual(
            [branch.token_count for branch in model.branches],
            [120, 30, 6],
        )
        self.assertEqual(
            [branch.dilations for branch in model.branches],
            [
                (1, 2, 4, 8, 16, 32),
                (1, 2, 4, 8),
                (1, 2),
            ],
        )
        self.assertEqual(
            [branch.receptive_field for branch in model.branches],
            [127, 31, 7],
        )

        class ForecastTrap(torch.nn.Module):
            def forward(self, _values):
                raise AssertionError("forecast head executed")

        model.forecast_head = ForecastTrap()
        closes = _closes()
        logits = model.forward_policy_logits(closes)
        self.assertEqual(tuple(logits.shape), (2, 101))
        self.assertTrue(bool(torch.isfinite(logits).all()))
        logits.square().mean().backward()
        self.assertTrue(all(
            parameter.grad is not None
            and bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
            if parameter.requires_grad
        ))
        with self.assertRaisesRegex(AssertionError, "forecast head executed"):
            model(closes)

    def test_every_patch_stream_is_prefix_causal(self) -> None:
        config = copy.deepcopy(_plan()["model"])
        config["dropout"] = 0
        model = build_model(config).eval()
        original = _closes(batch=1)
        changed = original.clone()
        cut = 1_800
        changed[:, cut:] *= torch.linspace(
            1.01,
            1.03,
            3_600 - cut,
        ).view(1, -1, 1)
        with torch.no_grad():
            original_branches = model.encode_branches(original)
            changed_branches = model.encode_branches(changed)
        for patch_size, original_values, changed_values in zip(
            model.patch_sizes,
            original_branches,
            changed_branches,
            strict=True,
        ):
            safe_tokens = cut // patch_size
            torch.testing.assert_close(
                original_values[:, :safe_tokens],
                changed_values[:, :safe_tokens],
                atol=0,
                rtol=0,
            )
            self.assertFalse(torch.equal(
                original_values[:, safe_tokens:],
                changed_values[:, safe_tokens:],
            ))

    def test_exact_dilation_schedule_must_cover_finest_patch_stream(self) -> None:
        config = copy.deepcopy(_plan()["model"])
        config["dilations"] = [1, 2, 4, 8, 16]
        with self.assertRaisesRegex(ValueError, "does not cover"):
            build_model(config)


class LearnedAggregatePlanTest(unittest.TestCase):
    def test_v20_is_direct_point01_raw_kl_and_seals_test(self) -> None:
        plan = _plan()
        validate_plan(plan)
        self.assertEqual(plan["version"], 20)
        self.assertIn("v20", plan["id"])
        self.assertEqual(plan["testPolicy"], "sealed-never-load")
        self.assertEqual(plan["oracleTarget"]["options"]["temperature"], 0.01)
        self.assertEqual(plan["training"]["selectionMetric"], "klDivergence")
        self.assertTrue(plan["training"]["policyOnly"])
        self.assertEqual(plan["training"]["epochs"], 384)
        self.assertGreaterEqual(
            plan["training"]["learningRateSchedule"]["patience"],
            48,
        )
        self.assertGreaterEqual(plan["training"]["patience"], 160)
        self.assertEqual(parameter_count(build_model(plan["model"])), 342_045)
        self.assertEqual(
            architecture_contract_for_model_config(plan["model"]),
            LEARNED_AGGREGATE_PATCH_MIXER_CONTRACT,
        )
        changed = copy.deepcopy(plan)
        changed.pop("testPolicy")
        with self.assertRaisesRegex(ValueError, "sealed-never-load"):
            validate_plan(changed)

    def test_v20_checkpoint_is_durably_resume_isolated(self) -> None:
        plan = _plan()
        config = copy.deepcopy(plan["model"])
        config["encoderWidth"] = 8
        config["fusionWidth"] = 16
        config["policyHiddenWidth"] = 16
        model = build_model(config)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            patience=48,
        )
        fingerprint = configuration_fingerprint(resolve_training_config(
            plan["training"]
        ))
        checkpoint = build_training_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch=2,
            global_step=1_116,
            best_validation=0.4,
            best_epoch=2,
            stale_epochs=0,
            validation={"klDivergence": 0.4},
            model_parameters=parameter_count(model),
            dataset_fingerprint="v20-dataset",
            model_config=config,
            plan_id=plan["id"],
            device=torch.device("cpu"),
            interrupted=False,
            training_config_fingerprint=fingerprint,
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint_file = (
                Path(directory)
                / "data"
                / "training"
                / "runs"
                / "v20"
                / "checkpoints"
                / "last.json"
            )
            atomic_torch_save(checkpoint, checkpoint_file)
            resumed = load_resume_checkpoint(
                checkpoint_file,
                model,
                optimizer,
                scheduler,
                {"id": plan["id"]},
                config,
                "v20-dataset",
                parameter_count(model),
                torch.device("cpu"),
                training_config_fingerprint=fingerprint,
            )
            self.assertEqual(resumed, (3, 1_116, 0.4, 2, 0))
            with self.assertRaisesRegex(ValueError, "incompatible"):
                load_resume_checkpoint(
                    checkpoint_file,
                    model,
                    optimizer,
                    scheduler,
                    {"id": "different-plan"},
                    config,
                    "v20-dataset",
                    parameter_count(model),
                    torch.device("cpu"),
                    training_config_fingerprint=fingerprint,
                )


if __name__ == "__main__":
    unittest.main()

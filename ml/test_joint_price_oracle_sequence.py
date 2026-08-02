from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import onnx
from onnx.reference import ReferenceEvaluator
import torch

from joint_price_oracle_sequence import (
    ACTION_COUNT,
    DECISION_INTERVAL_MS,
    DECISION_PHASE_MS,
    SECONDS_PER_MINUTE,
    ChronologicalMinutePolicyModel,
    exact_receptive_field_dilations,
    masked_soft_target_policy_objective,
    sequence_core_alignment,
)


def deterministic_closes(
    batch_size: int,
    minute_count: int,
) -> torch.Tensor:
    seconds = minute_count * SECONDS_PER_MINUTE
    time = torch.arange(seconds, dtype=torch.float32).view(1, -1, 1)
    batch_offset = torch.arange(
        batch_size,
        dtype=torch.float32,
    ).view(-1, 1, 1)
    log_close = 10.5 + time * 1e-6 + batch_offset * 1e-4
    log_close = log_close + torch.sin(time / 37.0) * 2e-4
    return torch.exp(log_close)


class SequenceAlignmentTest(unittest.TestCase):
    def test_core_alignment_maps_targets_to_halo_rows(self) -> None:
        start = 1_800_000_000_000 + DECISION_PHASE_MS
        # Force the synthetic timestamp onto the verified phase.
        start -= start % DECISION_INTERVAL_MS - DECISION_PHASE_MS
        alignment = sequence_core_alignment(start, 1_440, 360)
        self.assertEqual(alignment.halo_minutes, 359)
        self.assertEqual(alignment.token_count, 1_799)
        self.assertEqual(alignment.close_count, 1_799 * 60)
        self.assertEqual(alignment.sequence_index(0), 359)
        self.assertEqual(alignment.sequence_index(1_439), 1_798)
        self.assertEqual(alignment.prediction_time(1), start + 60_000)
        self.assertEqual(
            alignment.input_close_time_start,
            start - (360 * 60 - 1) * 1_000,
        )
        self.assertEqual(
            alignment.input_close_time_start
            + (alignment.close_count - 1) * 1_000,
            alignment.input_close_time_end,
        )

    def test_exact_receptive_field_schedules(self) -> None:
        self.assertEqual(
            exact_receptive_field_dilations(60),
            (1, 2, 4, 8, 16, 28),
        )
        self.assertEqual(
            1 + sum(exact_receptive_field_dilations(360)),
            360,
        )


class ChronologicalMinutePolicyModelTest(unittest.TestCase):
    @staticmethod
    def model(receptive_field_minutes: int = 60) \
            -> ChronologicalMinutePolicyModel:
        torch.manual_seed(7)
        return ChronologicalMinutePolicyModel(
            receptive_field_minutes=receptive_field_minutes,
            token_width=8,
            policy_hidden_width=12,
            dropout=0.0,
        )

    def test_sequence_and_single_windows_have_matching_logits(self) -> None:
        receptive_field = 60
        minute_count = 67
        model = self.model(receptive_field).eval()
        closes = deterministic_closes(2, minute_count)
        with torch.no_grad():
            sequence = model.forward_sequence(closes)
            for minute in (59, 62, 66):
                start = (
                    minute - receptive_field + 1
                ) * SECONDS_PER_MINUTE
                end = (minute + 1) * SECONDS_PER_MINUTE
                single = model.forward_single(closes[:, start:end, :])
                torch.testing.assert_close(
                    single,
                    sequence[:, minute, :],
                    atol=1e-6,
                    rtol=1e-6,
                )

    def test_360_minute_configuration_has_single_sequence_parity(self) -> None:
        receptive_field = 360
        model = self.model(receptive_field).eval()
        closes = deterministic_closes(1, 362)
        with torch.no_grad():
            sequence = model.forward_sequence(closes)
            single = model.forward_single(
                closes[:, -receptive_field * SECONDS_PER_MINUTE:, :]
            )
        torch.testing.assert_close(
            single,
            sequence[:, -1, :],
            atol=1e-6,
            rtol=1e-6,
        )

    def test_future_patch_perturbation_cannot_change_prior_logits(self) -> None:
        model = self.model(60).eval()
        closes = deterministic_closes(1, 65)
        changed = closes.clone()
        changed[:, 62 * SECONDS_PER_MINUTE:, :] *= 1.03
        with torch.no_grad():
            baseline = model.forward_sequence(closes)
            perturbed = model.forward_sequence(changed)
        torch.testing.assert_close(
            baseline[:, :62, :],
            perturbed[:, :62, :],
            atol=0,
            rtol=0,
        )

    def test_final_logit_ignores_minutes_older_than_exact_rf(self) -> None:
        model = self.model(60).eval()
        closes = deterministic_closes(1, 65)
        changed = closes.clone()
        changed[:, :5 * SECONDS_PER_MINUTE, :] *= 0.97
        with torch.no_grad():
            baseline = model.forward_sequence(closes)[:, -1, :]
            perturbed = model.forward_sequence(changed)[:, -1, :]
        torch.testing.assert_close(
            baseline,
            perturbed,
            atol=0,
            rtol=0,
        )

    def test_direct_masked_objective_has_finite_gradients(self) -> None:
        model = self.model(60).train()
        closes = deterministic_closes(2, 63)
        logits = model.forward_sequence(closes)[:, -3:, :]
        raw_target = torch.rand(2, 3, ACTION_COUNT)
        target = raw_target / raw_target.sum(dim=-1, keepdim=True)
        mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        result = masked_soft_target_policy_objective(logits, target, mask)
        self.assertEqual(int(result["validRows"]), 4)
        self.assertTrue(bool(torch.isfinite(result["loss"])))
        result["loss"].backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(
            bool(torch.isfinite(gradient).all())
            for gradient in gradients
            if gradient is not None
        ))

    def test_masked_rows_do_not_affect_direct_objective(self) -> None:
        torch.manual_seed(9)
        logits = torch.randn(2, 4, ACTION_COUNT)
        target = torch.softmax(torch.randn(2, 4, ACTION_COUNT), dim=-1)
        mask = torch.tensor([
            [True, False, True, False],
            [False, True, False, True],
        ])
        baseline = masked_soft_target_policy_objective(
            logits,
            target,
            mask,
        )
        changed_logits = logits.clone()
        changed_target = target.clone()
        changed_logits[~mask] += 100
        changed_target[~mask] = torch.softmax(
            torch.randn_like(changed_target[~mask]),
            dim=-1,
        )
        changed = masked_soft_target_policy_objective(
            changed_logits,
            changed_target,
            mask,
        )
        torch.testing.assert_close(
            baseline["crossEntropy"],
            changed["crossEntropy"],
        )
        torch.testing.assert_close(
            baseline["klDivergence"],
            changed["klDivergence"],
        )

    def test_single_decision_wrapper_exports_to_onnx(self) -> None:
        model = self.model(60).eval()
        example = deterministic_closes(1, 60)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sequence-single.onnx"
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
            onnx_logits = ReferenceEvaluator(exported).run(
                ["action_logits"],
                {"closes": example.numpy()},
            )[0]
        with torch.no_grad():
            torch_logits = model(example).numpy()
        self.assertTrue(np.isfinite(onnx_logits).all())
        self.assertLessEqual(
            float(np.max(np.abs(torch_logits - onnx_logits))),
            1e-4,
        )


if __name__ == "__main__":
    unittest.main()

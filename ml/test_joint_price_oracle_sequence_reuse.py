from __future__ import annotations

import copy
import unittest

import torch

from joint_price_oracle import JointLossWeights
from joint_price_oracle_sequence import (
    DECISION_INTERVAL_MS,
    DECISION_PHASE_MS,
    BoundaryCompleteMinutePolicyModel,
    BoundaryMaMinutePolicyModel,
    boundary_sequence_core_alignment,
)
from train_joint_price_oracle import (
    causal_input_close_windows,
    policy_only_forward_objective,
    sequence_core_policy_forward_objective,
)


def contiguous_closes(core_count: int) -> torch.Tensor:
    close_count = (59 + core_count) * 60 + 1
    time = torch.arange(close_count, dtype=torch.float32)
    log_close = 10.5 + time * 1e-6 + torch.sin(time / 37.0) * 2e-4
    return torch.exp(log_close).view(1, -1, 1)


def fixed_windows(closes: torch.Tensor, core_count: int) -> torch.Tensor:
    values = causal_input_close_windows(
        closes[0, :, 0].numpy(),
        core_count,
        3_601,
        sample_step_seconds=60,
    )
    return torch.from_numpy(values[:, :, None])


class BoundarySequenceCoreModelTest(unittest.TestCase):
    @staticmethod
    def model() -> BoundaryCompleteMinutePolicyModel:
        torch.manual_seed(71)
        return BoundaryCompleteMinutePolicyModel(
            token_width=8,
            policy_hidden_width=12,
            dropout=0,
        )

    def test_corrected_core_timestamp_and_close_alignment(self) -> None:
        start = 1_800_000_000_000 + DECISION_PHASE_MS
        start -= start % DECISION_INTERVAL_MS - DECISION_PHASE_MS
        alignment = boundary_sequence_core_alignment(start, 7, 60)
        self.assertEqual(alignment.halo_minutes, 59)
        self.assertEqual(alignment.token_count, 66)
        self.assertEqual(alignment.close_count, 66 * 60 + 1)
        self.assertEqual(alignment.sequence_index(0), 59)
        self.assertEqual(alignment.sequence_index(6), 65)
        self.assertEqual(
            alignment.input_close_time_start,
            start - 3_600_000,
        )
        self.assertEqual(
            alignment.input_close_time_start
            + (alignment.close_count - 1) * 1_000,
            alignment.input_close_time_end,
        )

    def test_contiguous_core_logits_equal_every_fixed_window(self) -> None:
        core_count = 9
        closes = contiguous_closes(core_count)
        windows = fixed_windows(closes, core_count)
        model = self.model().eval()
        with torch.no_grad():
            reused = model.forward_sequence_core(closes)[0]
            fixed = model.forward_policy_logits(windows)
        torch.testing.assert_close(reused, fixed, atol=0, rtol=0)

    def test_raw_loss_and_gradients_equal_fixed_window_batch(self) -> None:
        core_count = 7
        closes = contiguous_closes(core_count)
        windows = fixed_windows(closes, core_count)
        torch.manual_seed(73)
        target = torch.softmax(torch.randn(core_count, 101), dim=-1)
        reused_model = self.model().train()
        fixed_model = copy.deepcopy(reused_model).train()
        weights = JointLossWeights(forecast=0, soft_layer_norm=0)

        reused_logits, reused_metrics = (
            sequence_core_policy_forward_objective(
                reused_model,
                closes,
                target.unsqueeze(0),
                weights,
            )
        )
        fixed_logits, fixed_metrics = policy_only_forward_objective(
            fixed_model,
            windows,
            target,
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
        for name in ("loss", "crossEntropy", "klDivergence", "probabilityMse"):
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
                atol=1e-7,
                rtol=1e-5,
            )

    def test_window_anchored_ma_variant_rejects_multirow_reuse(self) -> None:
        model = BoundaryMaMinutePolicyModel(
            token_width=8,
            policy_hidden_width=12,
            dropout=0,
        )
        with self.assertRaisesRegex(ValueError, "window-anchored"):
            model.forward_sequence_core(contiguous_closes(2))
        with torch.no_grad():
            logits = model.forward_sequence_core(contiguous_closes(1))
        self.assertEqual(tuple(logits.shape), (1, 1, 101))


if __name__ == "__main__":
    unittest.main()

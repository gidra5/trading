from __future__ import annotations

import unittest

import torch

from joint_price_oracle import parameter_count
from joint_price_oracle_sequence import (
    ADDITIVE_MA_FEATURE_COUNT,
    BOUNDARY_INPUT_CLOSE_COUNT,
    BoundaryCompleteMinutePolicyModel,
    BoundaryMaMinutePolicyModel,
    boundary_complete_minute_features,
    causal_additive_minute_path_band_features,
    causal_additive_minute_path_bands,
)


def closes_from_second_returns(second_returns: torch.Tensor) -> torch.Tensor:
    boundary = torch.full(
        (second_returns.shape[0], 1),
        10.5,
        dtype=second_returns.dtype,
    )
    log_closes = torch.cat((
        boundary,
        boundary + second_returns.cumsum(dim=1),
    ), dim=1)
    return torch.exp(log_closes).unsqueeze(-1)


class BoundaryCompleteMinuteFeatureTest(unittest.TestCase):
    def test_all_sixty_returns_include_each_minute_boundary(self) -> None:
        seconds = torch.arange(3_600, dtype=torch.float32)
        second_returns = (torch.sin(seconds / 17.0) * 4e-4).view(1, -1)
        # Every one of these is the first transition of a completed minute.
        # Legacy v17 patch-local differencing replaced positions 60, 120, ...
        # with zero after splitting its 3,600 closes into minute patches.
        second_returns[0, ::60] = torch.linspace(1e-4, 6e-4, 60)
        features, minute_returns = boundary_complete_minute_features(
            closes_from_second_returns(second_returns)
        )
        expected_returns = second_returns.reshape(1, 60, 60)
        torch.testing.assert_close(
            minute_returns,
            expected_returns.sum(dim=-1),
            atol=4e-6,
            rtol=0,
        )
        expected_fixed = torch.tanh(
            expected_returns * 10_000.0 / 8.0
        ) * 8.0
        torch.testing.assert_close(
            features[:, :, 2, :],
            expected_fixed,
            atol=0.02,
            rtol=0,
        )
        self.assertEqual(tuple(features.shape), (1, 60, 5, 60))
        self.assertTrue(bool((features[0, :, 2, 0] > 0).all()))

    def test_future_second_cannot_change_prior_minute_features(self) -> None:
        torch.manual_seed(41)
        second_returns = torch.randn(1, 180) * 1e-4
        closes = closes_from_second_returns(second_returns)
        changed_returns = second_returns.clone()
        changed_returns[:, 120:] += 2e-4
        changed = closes_from_second_returns(changed_returns)
        baseline, baseline_minutes = boundary_complete_minute_features(closes)
        perturbed, perturbed_minutes = boundary_complete_minute_features(
            changed
        )
        torch.testing.assert_close(
            baseline[:, :2],
            perturbed[:, :2],
            atol=0,
            rtol=0,
        )
        torch.testing.assert_close(
            baseline_minutes[:, :2],
            perturbed_minutes[:, :2],
            atol=0,
            rtol=0,
        )


class AdditiveMinutePathBandTest(unittest.TestCase):
    def test_bands_and_deltas_reconstruct_exact_causal_path(self) -> None:
        torch.manual_seed(43)
        minute_returns = torch.randn(3, 60) * 4e-4
        bands, deltas = causal_additive_minute_path_bands(minute_returns)
        path = minute_returns.cumsum(dim=1)
        torch.testing.assert_close(
            bands.sum(dim=-1),
            path,
            atol=2e-9,
            rtol=2e-6,
        )
        torch.testing.assert_close(
            deltas.cumsum(dim=1),
            bands,
            atol=2e-9,
            rtol=2e-6,
        )
        features = causal_additive_minute_path_band_features(minute_returns)
        self.assertEqual(tuple(features.shape), (3, 60, 8))
        torch.testing.assert_close(
            features[:, :, :4].sum(dim=-1) / 100.0,
            path,
            atol=2e-9,
            rtol=2e-6,
        )

    def test_ma_bands_are_strictly_causal(self) -> None:
        torch.manual_seed(47)
        minute_returns = torch.randn(2, 60) * 2e-4
        changed = minute_returns.clone()
        changed[:, 35:] += 0.01
        baseline = causal_additive_minute_path_band_features(minute_returns)
        perturbed = causal_additive_minute_path_band_features(changed)
        torch.testing.assert_close(
            baseline[:, :35],
            perturbed[:, :35],
            atol=0,
            rtol=0,
        )


class CorrectedMinutePolicyModelTest(unittest.TestCase):
    @staticmethod
    def closes(batch_size: int = 2) -> torch.Tensor:
        torch.manual_seed(53)
        returns = torch.randn(
            batch_size,
            BOUNDARY_INPUT_CLOSE_COUNT - 1,
        ) * 1e-4
        return closes_from_second_returns(returns)

    def test_default_variants_are_compute_matched_and_direct(self) -> None:
        base = BoundaryCompleteMinutePolicyModel()
        bands = BoundaryMaMinutePolicyModel()
        self.assertEqual(parameter_count(base), 385_266)
        self.assertEqual(parameter_count(bands), 386_290)
        self.assertIsNone(base.band_projection)
        self.assertIsNotNone(bands.band_projection)
        assert bands.band_projection is not None
        self.assertEqual(
            bands.band_projection.in_features,
            ADDITIVE_MA_FEATURE_COUNT,
        )
        self.assertEqual(base.hour_scale_bypass.in_features, 8)
        self.assertEqual(bands.hour_scale_bypass.in_features, 8)

    def test_both_variants_emit_finite_logits_and_gradients(self) -> None:
        closes = self.closes()
        for model_type in (
            BoundaryCompleteMinutePolicyModel,
            BoundaryMaMinutePolicyModel,
        ):
            with self.subTest(model=model_type.__name__):
                torch.manual_seed(59)
                model = model_type(
                    token_width=8,
                    policy_hidden_width=12,
                    dropout=0,
                ).train()
                logits = model.forward_policy_logits(closes)
                self.assertEqual(tuple(logits.shape), (2, 101))
                self.assertTrue(bool(torch.isfinite(logits).all()))
                logits.square().mean().backward()
                gradients = [
                    parameter.grad
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ]
                self.assertTrue(all(value is not None for value in gradients))
                self.assertTrue(all(
                    bool(torch.isfinite(value).all())
                    for value in gradients
                    if value is not None
                ))

    def test_exact_input_contract_rejects_legacy_3600_close_window(self) -> None:
        model = BoundaryCompleteMinutePolicyModel(
            token_width=8,
            policy_hidden_width=12,
            dropout=0,
        )
        with self.assertRaisesRegex(ValueError, "3601"):
            model(torch.ones(1, 3_600, 1))


if __name__ == "__main__":
    unittest.main()

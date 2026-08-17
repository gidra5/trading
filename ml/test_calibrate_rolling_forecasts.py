import unittest

import numpy as np

from calibrate_rolling_forecasts import (
    calibrated_ensemble,
    feature_domain,
    fit_calibrator,
    forward_transform,
    inverse_transform,
    walk_forward_calibrated_ensemble,
)


class CalibrateRollingForecastsTest(unittest.TestCase):
    def test_feature_transforms_round_trip(self) -> None:
        cases = {
            "periodReturnBps": np.array([-2.0, 0.0, 3.0]),
            "oneSecondRealizedVarianceBpsSquared": np.array([1.0, 4.0, 9.0]),
            "activeSecondFraction": np.array([0.1, 0.5, 0.9]),
            "minuteLag1ReturnCorrelation": np.array([-0.7, 0.0, 0.8]),
        }
        for feature, values in cases.items():
            with self.subTest(domain=feature_domain(feature)):
                np.testing.assert_allclose(
                    inverse_transform(feature, forward_transform(feature, values)),
                    values,
                    rtol=1e-10,
                    atol=1e-10,
                )

    def test_residual_calibration_corrects_multiplicative_positive_bias(self) -> None:
        rng = np.random.default_rng(7)
        latent = rng.normal(4.0, 0.25, 160)
        actual = np.exp(latent)
        ensemble = np.exp(
            latent[:, None] - 0.5 + rng.normal(0.0, 0.12, (160, 16))
        )
        model = fit_calibrator(
            "oneSecondRealizedVarianceBpsSquared",
            actual[:100],
            ensemble[:100],
            "affineResidual",
        )
        calibrated, parameters = calibrated_ensemble(
            "oneSecondRealizedVarianceBpsSquared",
            ensemble[100:],
            model,
            16,
        )
        raw_error = np.mean(np.abs(np.median(ensemble[100:], axis=1) - actual[100:]))
        calibrated_error = np.mean(
            np.abs(np.median(calibrated, axis=1) - actual[100:])
        )
        self.assertLess(calibrated_error, raw_error * 0.2)
        self.assertEqual(parameters["domain"], "log")
        self.assertEqual(calibrated.shape, (60, 16))

    def test_standardized_calibration_scales_residual_quantiles_per_row(self) -> None:
        rng = np.random.default_rng(11)
        centers = np.linspace(-1.0, 1.0, 80)
        scales = np.linspace(0.2, 1.0, 80)
        ensemble = centers[:, None] + scales[:, None] * rng.normal(size=(80, 16))
        actual = centers + scales * rng.normal(size=80)
        model = fit_calibrator(
            "periodReturnBps",
            actual[:50],
            ensemble[:50],
            "standardizedResidual",
        )
        calibrated, _ = calibrated_ensemble(
            "periodReturnBps",
            ensemble[50:],
            model,
            16,
        )
        output_spread = np.std(calibrated, axis=1)
        input_spread = np.std(ensemble[50:], axis=1)
        self.assertGreater(np.corrcoef(output_spread, input_spread)[0, 1], 0.99)

    def test_walk_forward_calibration_never_reads_current_or_future_outcome(self) -> None:
        rng = np.random.default_rng(19)
        actual = rng.normal(size=50)
        ensemble = rng.normal(size=(50, 16))
        baseline = walk_forward_calibrated_ensemble(
            "periodReturnBps",
            actual,
            ensemble,
            start=40,
            method="residual",
            history_days=30,
            output_members=16,
        )
        changed = actual.copy()
        changed[45:] += 1_000.0
        perturbed = walk_forward_calibrated_ensemble(
            "periodReturnBps",
            changed,
            ensemble,
            start=40,
            method="residual",
            history_days=30,
            output_members=16,
        )
        np.testing.assert_allclose(baseline[:6], perturbed[:6])
        self.assertFalse(np.allclose(baseline[6:], perturbed[6:]))


if __name__ == "__main__":
    unittest.main()

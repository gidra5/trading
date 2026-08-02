from __future__ import annotations

import unittest

import numpy as np

from audit_v18_ohlcv_fusion import log_ratio_feature_fusion
from audit_v18_six_hour_close_fusion import (
    FEATURE_INDEX,
    HISTORY_MINUTES,
    causal_six_hour_features,
    completed_close_windows,
)


class SixHourCloseFeatureTests(unittest.TestCase):
    def test_completed_windows_exclude_current_unfinished_minute(self) -> None:
        previous = np.arange(1, 1_441, dtype=np.float64)
        current = np.arange(10_001, 11_441, dtype=np.float64)
        windows = completed_close_windows(previous, current)

        self.assertEqual(windows.shape, (1_440, HISTORY_MINUTES + 1))
        np.testing.assert_array_equal(windows[0], previous[-361:])
        self.assertEqual(windows[1, -1], current[0])
        self.assertEqual(windows[-1, -1], current[-2])
        self.assertNotIn(current[-1], windows[-1])

    def test_features_are_scale_invariant_and_bands_are_additive(self) -> None:
        log_returns = np.linspace(-0.001, 0.0015, HISTORY_MINUTES)
        closes = 25_000 * np.exp(np.concatenate(([0.0], np.cumsum(log_returns))))
        windows = np.stack((closes, closes * 7.25))
        features = causal_six_hour_features(windows)

        np.testing.assert_allclose(features[0], features[1], atol=2e-7, rtol=2e-5)
        bands = features[:, 8:].sum(axis=1)
        np.testing.assert_allclose(
            bands,
            features[:, FEATURE_INDEX["return6h"]],
            atol=2e-7,
            rtol=2e-5,
        )
        self.assertAlmostEqual(
            float(features[0, FEATURE_INDEX["return1h"]]),
            float(log_returns[-60:].sum()),
            places=7,
        )

    def test_constant_return_rms_and_horizons_are_exact(self) -> None:
        value = 0.0002
        closes = np.exp(np.arange(HISTORY_MINUTES + 1) * value)[None, :]
        features = causal_six_hour_features(closes)[0]

        for hours, minutes in ((1, 60), (2, 120), (3, 180), (6, 360)):
            self.assertAlmostEqual(
                float(features[FEATURE_INDEX[f"return{hours}h"]]),
                minutes * value,
                places=7,
            )
            self.assertAlmostEqual(
                float(features[FEATURE_INDEX[f"rmsReturn{hours}h"]]),
                value,
                places=7,
            )

    def test_zero_log_ratio_weight_is_exact_frozen_identity(self) -> None:
        base = np.asarray([[0.2, 0.3, 0.5]], dtype=np.float64)
        long = np.asarray([[0.5, 0.25, 0.25]], dtype=np.float64)
        short = np.asarray([[0.25, 0.5, 0.25]], dtype=np.float64)
        fused = log_ratio_feature_fusion(base, long, short, 0.0)

        np.testing.assert_allclose(fused, base, atol=1e-15, rtol=1e-15)


if __name__ == "__main__":
    unittest.main()

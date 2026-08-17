from __future__ import annotations

import unittest

import numpy as np

from analyze_fourier_return_features import fractional_dft, spectral_features


class SpectralFeaturesTest(unittest.TestCase):
    def test_zero_window_is_finite(self) -> None:
        result = spectral_features(np.zeros((3, 16), dtype=np.float64))
        self.assertEqual(result.shape, (3, 34))
        self.assertTrue(np.isfinite(result).all())
        self.assertTrue(np.all(result == 0))

    def test_low_and_high_frequency_are_separated(self) -> None:
        size = 64
        time = np.arange(size, dtype=np.float64)
        low = np.sin(2 * np.pi * 2 * time / size)
        high = np.sin(2 * np.pi * 24 * time / size)
        result = spectral_features(np.stack([low, high]))
        self.assertGreater(result[0, 1], result[1, 1])
        self.assertGreater(result[1, 2], result[0, 2])

    def test_output_is_scale_invariant_except_energy(self) -> None:
        values = np.sin(2 * np.pi * 3 * np.arange(64) / 64)
        result = spectral_features(np.stack([values, values * 7]))
        self.assertGreater(result[1, 0], result[0, 0])
        np.testing.assert_allclose(result[0, 1:], result[1, 1:], atol=1e-12)

    def test_fractional_dft_endpoints_and_composition(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(size=(2, 16))
        np.testing.assert_allclose(fractional_dft(values, 0), values, atol=1e-12)
        expected = np.fft.fft(values, axis=1, norm="ortho")
        np.testing.assert_allclose(fractional_dft(values, 1), expected, atol=1e-12)
        half_twice = fractional_dft(fractional_dft(values, 0.5), 0.5)
        np.testing.assert_allclose(half_twice, expected, atol=1e-11)

    def test_efficiency_ratios_distinguish_trend_from_noise(self) -> None:
        trend = np.ones(16, dtype=np.float64)
        alternating = np.tile([1.0, -1.0], 8)
        result = spectral_features(np.stack([trend, alternating]))
        self.assertAlmostEqual(result[0, -3], 1.0)
        self.assertAlmostEqual(result[0, -2], 1.0)
        self.assertAlmostEqual(result[0, -1], 4.0)
        self.assertAlmostEqual(result[1, -3], 0.0)
        self.assertAlmostEqual(result[1, -2], 0.0)
        self.assertAlmostEqual(result[1, -1], 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import unittest

import numpy as np

from forecast_candle_transforms import decode_quantile_paths, encode_contexts


class ForecastCandleTransformTests(unittest.TestCase):
    def setUp(self) -> None:
        close = np.asarray([100.0, 101.0, 100.5, 102.0])
        open_ = np.asarray([99.8, 100.2, 101.1, 100.7])
        high = np.maximum(open_, close) + 0.4
        low = np.minimum(open_, close) - 0.3
        volume = np.asarray([10.0, 12.0, 11.0, 13.0])
        self.context = np.stack((open_, high, low, close, volume), axis=-1)[None]

    def test_all_representations_round_trip_shapes(self) -> None:
        for representation in ("raw", "anchored-log", "candle-returns"):
            encoded, state = encode_contexts(self.context, representation)
            self.assertEqual(encoded.shape, (1, 5, 4))
            forecast = np.repeat(encoded[:, :, -1:, None], 9, axis=3)
            decoded = decode_quantile_paths(forecast, state)
            self.assertEqual(decoded.shape, (1, 9, 1, 5))
            self.assertTrue(np.isfinite(decoded).all())
            self.assertTrue((decoded > 0).all())

    def test_candle_return_decode_always_produces_valid_ohlc(self) -> None:
        _, state = encode_contexts(self.context, "candle-returns")
        rng = np.random.default_rng(7)
        encoded = rng.normal(0, 0.05, size=(2, 5, 15, 9))
        state = type(state)(state.representation, np.asarray([100.0, 200.0]), np.asarray([10.0, 20.0]))
        decoded = decode_quantile_paths(encoded, state)
        self.assertTrue((decoded[..., 1] >= np.maximum(decoded[..., 0], decoded[..., 3])).all())
        self.assertTrue((decoded[..., 2] <= np.minimum(decoded[..., 0], decoded[..., 3])).all())

    def test_raw_accepts_zero_volume(self) -> None:
        context = self.context.copy()
        context[:, 2, 4] = 0.0
        encoded, _state = encode_contexts(context, "raw")
        self.assertEqual(encoded[0, 4, 2], 0.0)

    def test_log_representations_condition_zero_volume(self) -> None:
        context = self.context.copy()
        context[:, 2, 4] = 0.0
        for representation in ("anchored-log", "candle-returns"):
            encoded, _state = encode_contexts(context, representation)
            self.assertTrue(np.isfinite(encoded).all())
            self.assertGreater(encoded.min(), -20.0)


if __name__ == "__main__":
    unittest.main()

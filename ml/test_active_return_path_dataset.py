from __future__ import annotations

import unittest

import numpy as np

from active_return_path_dataset import next_active_return_paths


class ActiveReturnPathDatasetTest(unittest.TestCase):
    def test_selects_next_nonzero_returns_from_each_start(self) -> None:
        returns = np.asarray(
            [0.0, 0.1, 0.0, 0.2, 0.3, 0.0, 0.4], dtype=np.float32
        )
        actual = next_active_return_paths(returns, 4, 2)
        expected = np.asarray(
            [[0.1, 0.2], [0.1, 0.2], [0.2, 0.3], [0.2, 0.3]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(actual, expected)

    def test_requires_enough_future_active_returns(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not cover"):
            next_active_return_paths(
                np.asarray([0.1, 0.0, 0.2], dtype=np.float32), 3, 2
            )


if __name__ == "__main__":
    unittest.main()

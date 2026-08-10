from __future__ import annotations

import unittest

import numpy as np

from kronos_probabilistic import (
    DEFAULT_QUANTILE_LEVELS,
    empirical_ohlc_quantiles,
    generate_sample_paths,
    isotonic_projection,
    kline_valid_mask,
    kqsp,
    point_estimators,
    project_kline_rows,
)


class _FakePredictor:
    def generate(
        self,
        context,
        context_stamps,
        target_stamps,
        horizon,
        temperature,
        top_k,
        top_p,
        sample_count,
        verbose,
    ):
        del context_stamps, target_stamps, temperature, top_k, top_p, verbose
        if sample_count != 1:
            raise AssertionError("the retained-path adapter must disable averaging")
        row = np.arange(context.shape[0], dtype=np.float32)[:, None, None]
        return np.broadcast_to(row, (context.shape[0], horizon, 6)).copy()


class KronosProbabilisticTest(unittest.TestCase):
    def test_generate_sample_paths_retains_sample_axis(self) -> None:
        context = np.zeros((2, 4, 6), dtype=np.float32)
        stamps = np.zeros((2, 4, 5), dtype=np.float32)
        targets = np.zeros((2, 2, 5), dtype=np.float32)
        means = np.zeros((2, 6), dtype=np.float32)
        stds = np.ones((2, 6), dtype=np.float32) - 1e-5
        paths = generate_sample_paths(
            _FakePredictor(),
            context,
            stamps,
            targets,
            means,
            stds,
            horizon=2,
            temperature=0.6,
            top_p=0.9,
            sample_count=3,
        )
        self.assertEqual(paths.shape, (2, 3, 2, 6))
        np.testing.assert_allclose(paths[0, :, 0, 0], (0, 1, 2))
        np.testing.assert_allclose(paths[1, :, 0, 0], (3, 4, 5))

    def test_kline_projection_matches_paper_example(self) -> None:
        projected = project_kline_rows(np.asarray((100, 99, 98, 101)))
        np.testing.assert_allclose(projected, (100, 100, 98, 100))
        self.assertTrue(bool(kline_valid_mask(projected)))

    def test_isotonic_projection_matches_paper_example(self) -> None:
        projected = isotonic_projection(np.asarray((99, 100, 99)))
        np.testing.assert_allclose(projected, (99, 99.5, 99.5))

    def test_kqsp_removes_both_constraint_families(self) -> None:
        quantiles = np.asarray([[[
            (100, 99, 98, 101),
            (99, 101, 98, 100),
            (101, 100, 99, 102),
        ]]], dtype=np.float64)
        repaired = kqsp(quantiles)
        self.assertTrue(np.all(kline_valid_mask(repaired)))
        self.assertTrue(np.all(np.diff(repaired, axis=2) >= -1e-10))

    def test_kqsp_random_property_is_valid_and_idempotent(self) -> None:
        generator = np.random.default_rng(1_337)
        quantiles = generator.normal(100, 8, size=(8, 15, 9, 4))

        repaired = kqsp(quantiles)

        self.assertTrue(np.all(kline_valid_mask(repaired)))
        self.assertTrue(np.all(np.diff(repaired, axis=2) >= -1e-10))
        np.testing.assert_allclose(kqsp(repaired), repaired, atol=1e-9)

    def test_kline_projection_never_moves_more_than_one_sided_clipping(self) -> None:
        generator = np.random.default_rng(7)
        values = generator.normal(100, 10, size=(1_000, 4))
        projected = project_kline_rows(values)
        clipped = values.copy()
        clipped[:, 1] = np.maximum.reduce((values[:, 1], values[:, 0], values[:, 3]))
        clipped[:, 2] = np.minimum.reduce((values[:, 2], values[:, 0], values[:, 3]))

        projection_error = np.square(projected - values).sum(axis=1)
        clipping_error = np.square(clipped - values).sum(axis=1)
        self.assertTrue(np.all(projection_error <= clipping_error + 1e-9))

    def test_empirical_quantiles_and_estimators(self) -> None:
        paths = np.zeros((1, 10, 2, 6), dtype=np.float64)
        for sample in range(10):
            paths[0, sample, :, :4] = (
                100 + sample,
                99 + sample,
                98 + sample,
                101 + sample,
            )
        raw = empirical_ohlc_quantiles(paths, DEFAULT_QUANTILE_LEVELS)
        repaired = kqsp(raw)
        estimators = point_estimators(paths, repaired)
        self.assertEqual(set(estimators), {
            "ensembleMean", "projectedMean", "ensembleMedian", "kqspMedian"
        })
        self.assertTrue(np.all(kline_valid_mask(estimators["projectedMean"])))
        self.assertTrue(np.all(kline_valid_mask(estimators["kqspMedian"])))


if __name__ == "__main__":
    unittest.main()

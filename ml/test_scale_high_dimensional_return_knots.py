"""Focused tests for scalable high-dimensional return point clouds."""

from __future__ import annotations

import numpy as np

from scale_high_dimensional_return_knots import (
    fit_point_cloud,
    prune_adaptive_centers,
    projection_directions,
    sliced_js_metrics,
)


def test_projection_directions_include_axes_and_are_normalized() -> None:
    directions = projection_directions(6, 32, 123)

    assert directions.shape == (32, 6)
    np.testing.assert_allclose(directions[:6], np.eye(6))
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1.0)


def test_sliced_js_is_zero_for_identical_samples_and_detects_shift() -> None:
    rng = np.random.default_rng(7)
    values = rng.uniform(0.05, 0.95, size=(4_096, 5))
    directions = projection_directions(5, 24, 321)

    identical = sliced_js_metrics(values, values.copy(), directions, 64)
    shifted = sliced_js_metrics(
        values,
        np.clip(values + 0.08, 0.0, 1.0),
        directions,
        64,
    )

    assert identical["meanSlicedJsBits"] == 0.0
    assert shifted["meanSlicedJsBits"] > 0.01


def test_fast_point_cloud_supports_more_than_three_dimensions() -> None:
    rng = np.random.default_rng(19)
    values = np.clip(
        rng.normal(0.5, 0.16, size=(2_048, 4)),
        1e-5,
        1.0 - 1e-5,
    )
    directions = projection_directions(4, 16, 456)
    result = fit_point_cloud(
        values,
        count=24,
        directions=directions,
        true_metric_sample=values,
        projection_bins=64,
        metric_draws=2_048,
        cloud_sample=2_048,
        high_fidelity=False,
        seed=789,
    )

    assert result["knotCount"] == 24
    assert len(result["centersUnit"]) == 24
    assert len(result["centersUnit"][0]) == 4
    assert np.isfinite(result["meanSlicedJsBits"])
    np.testing.assert_allclose(sum(result["componentWeights"]), 1.0)

    pruned = prune_adaptive_centers(result, 17)
    assert pruned.shape == (16, 4)
    nested = fit_point_cloud(
        values,
        count=17,
        directions=directions,
        true_metric_sample=values,
        projection_bins=64,
        metric_draws=2_048,
        cloud_sample=2_048,
        high_fidelity=True,
        seed=790,
        metric_seed=991,
        initial_adaptive_centers=pruned,
    )
    assert nested["knotCount"] == 17
    assert np.isfinite(nested["meanSlicedJsBits"])

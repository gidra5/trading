from __future__ import annotations

from dataclasses import dataclass

import numpy as np


REPRESENTATIONS = ("raw", "anchored-log", "candle-returns")


@dataclass(frozen=True)
class TransformState:
    representation: str
    anchor_close: np.ndarray
    anchor_volume: np.ndarray


def encode_contexts(contexts: np.ndarray, representation: str) -> tuple[np.ndarray, TransformState]:
    values = np.asarray(contexts, dtype=np.float64)
    if values.ndim != 3 or values.shape[-1] != 5:
        raise ValueError(f"contexts must have shape [batch, time, 5], got {values.shape}")
    if representation not in REPRESENTATIONS:
        raise ValueError(f"unknown representation: {representation}")
    if not np.isfinite(values).all() or np.any(values[..., :4] <= 0) or np.any(values[..., 4] < 0):
        raise ValueError("OHLC prices must be positive and volume must be non-negative")
    anchor_close = values[:, -1, 3].copy()
    volume_floor = np.finfo(np.float64).eps
    anchor_volume = np.asarray([
        np.median(row[row > 0]) if np.any(row > 0) else 1.0
        for row in values[..., 4]
    ], dtype=np.float64)
    # A true zero-volume minute is valid market data, but log(epsilon) creates
    # an artificial ~35-sigma input that can destabilize large FP16 decoders.
    # Preserve raw zeros and use a context-relative six-order floor only for
    # representations that take logarithms.
    safe_volume = np.maximum(values[..., 4], anchor_volume[:, None] * 1e-6)
    state = TransformState(representation, anchor_close, anchor_volume)
    if representation == "raw":
        encoded = values
    elif representation == "anchored-log":
        encoded = np.empty_like(values)
        encoded[..., :4] = np.log(values[..., :4] / anchor_close[:, None, None])
        encoded[..., 4] = np.log(safe_volume / anchor_volume[:, None])
    else:
        previous_close = np.concatenate((values[:, :1, 3], values[:, :-1, 3]), axis=1)
        previous_volume = np.concatenate((safe_volume[:, :1], safe_volume[:, :-1]), axis=1)
        encoded = np.empty_like(values)
        encoded[..., 0] = np.log(values[..., 0] / previous_close)
        encoded[..., 1] = np.log(values[..., 1] / np.maximum(values[..., 0], values[..., 3]))
        encoded[..., 2] = np.log(np.minimum(values[..., 0], values[..., 3]) / values[..., 2])
        encoded[..., 3] = np.log(values[..., 3] / previous_close)
        encoded[..., 4] = np.log(safe_volume / previous_volume)
    return np.moveaxis(encoded.astype(np.float32), -1, 1), state


def decode_quantile_paths(encoded: np.ndarray, state: TransformState) -> np.ndarray:
    """Decode [batch, variate, horizon, quantile] into [batch, quantile, horizon, OHLCV]."""
    values = np.asarray(encoded, dtype=np.float64)
    if values.ndim != 4 or values.shape[1] != 5:
        raise ValueError(f"encoded forecast must have shape [batch, 5, horizon, quantile], got {values.shape}")
    trajectories = values.transpose(0, 3, 2, 1)
    if state.representation == "raw":
        return np.maximum(trajectories, np.finfo(np.float64).tiny)
    if state.representation == "anchored-log":
        decoded = np.empty_like(trajectories)
        decoded[..., :4] = state.anchor_close[:, None, None, None] * np.exp(
            np.clip(trajectories[..., :4], -0.5, 0.5)
        )
        decoded[..., 4] = state.anchor_volume[:, None, None] * np.exp(
            np.clip(trajectories[..., 4], -20.0, 20.0)
        )
        return decoded

    decoded = np.empty_like(trajectories)
    previous_close = np.broadcast_to(
        state.anchor_close[:, None], trajectories.shape[:2]
    ).copy()
    previous_volume = np.broadcast_to(
        state.anchor_volume[:, None], trajectories.shape[:2]
    ).copy()
    for step in range(trajectories.shape[2]):
        row = trajectories[:, :, step]
        open_price = previous_close * np.exp(np.clip(row[..., 0], -0.2, 0.2))
        close = previous_close * np.exp(np.clip(row[..., 3], -0.2, 0.2))
        high = np.maximum(open_price, close) * np.exp(np.clip(row[..., 1], 0.0, 0.2))
        low = np.minimum(open_price, close) * np.exp(-np.clip(row[..., 2], 0.0, 0.2))
        volume = previous_volume * np.exp(np.clip(row[..., 4], -10.0, 10.0))
        decoded[:, :, step] = np.stack((open_price, high, low, close, volume), axis=-1)
        previous_close = close
        previous_volume = volume
    return decoded

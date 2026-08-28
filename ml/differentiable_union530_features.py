from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Mapping, Sequence

import torch


EPSILON = 1e-12


def safe_divide(numerator: torch.Tensor, denominator: torch.Tensor) -> torch.Tensor:
    return torch.where(denominator != 0, numerator / denominator, torch.zeros_like(numerator))


def imbalance(positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
    return safe_divide(positive - negative, positive + negative)


def lag(values: torch.Tensor, steps: int, fill: float = 0.0) -> torch.Tensor:
    if steps < 0:
        raise ValueError("lag must be nonnegative")
    if steps == 0:
        return values
    output = torch.full_like(values, fill)
    if steps < values.shape[0]:
        output[steps:] = values[:-steps]
    return output


def rolling_sum(values: torch.Tensor, window: int) -> torch.Tensor:
    if window < 1:
        raise ValueError("rolling window must be positive")
    prefix = torch.cat((torch.zeros_like(values[:1]), torch.cumsum(values, dim=0)))
    result = prefix[1:].clone()
    if window < values.shape[0]:
        result[window:] -= prefix[:-window - 0][1:values.shape[0] - window + 1]
    return result


def rolling_sum_exact(values: torch.Tensor, window: int) -> torch.Tensor:
    """Trailing sum with a shortened causal window at the beginning."""
    if window < 1:
        raise ValueError("rolling window must be positive")
    prefix = torch.cat((torch.zeros_like(values[:1]), torch.cumsum(values, dim=0)))
    indices = torch.arange(values.shape[0], device=values.device)
    start = torch.clamp(indices + 1 - window, min=0)
    return prefix[indices + 1] - prefix[start]


def ema_filter_fft(
    values: torch.Tensor,
    alpha: float,
    *,
    initial: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Differentiable finite-axis EMA matching scipy.signal.lfilter seeding.

    The recurrence is y[t]=(1-alpha)y[t-1]+alpha*x[t], with the state before
    t=0 equal to ``initial``. FFT convolution avoids a Python/autograd node per
    history row and keeps multi-million-row feature histories practical.
    """
    if values.ndim != 1 or values.numel() < 1:
        raise ValueError("EMA input must be one nonempty timeline")
    if not 0 < alpha <= 1:
        raise ValueError("EMA alpha must be in (0, 1]")
    if initial is None:
        initial = values[0]
    initial_tensor = torch.as_tensor(initial, dtype=values.dtype, device=values.device)
    scale = torch.clamp_min(
        values.detach().abs().max(), torch.finfo(values.dtype).tiny
    )
    normalized = values / scale
    normalized_initial = initial_tensor / scale
    count = values.shape[0]
    fft_count = 1 << max(0, (2 * count - 1).bit_length())
    decay = 1.0 - float(alpha)
    powers = torch.pow(
        torch.as_tensor(decay, dtype=values.dtype, device=values.device),
        torch.arange(count, dtype=values.dtype, device=values.device),
    )
    kernel = float(alpha) * powers
    convolved = torch.fft.irfft(
        torch.fft.rfft(normalized, n=fft_count)
        * torch.fft.rfft(kernel, n=fft_count),
        n=fft_count,
    )[:count]
    return (convolved + normalized_initial * decay * powers) * scale


def log_rms(returns: torch.Tensor, window: int) -> torch.Tensor:
    count = torch.minimum(
        torch.arange(1, returns.shape[0] + 1, device=returns.device),
        torch.as_tensor(window, device=returns.device),
    ).to(returns.dtype)
    mean_square = torch.clamp_min(
        rolling_sum_exact(returns.square(), window) / count, 0
    )
    return 0.5 * torch.log(1e-16 + mean_square)


def rolling_rms(returns: torch.Tensor, window: int) -> torch.Tensor:
    count = torch.minimum(
        torch.arange(1, returns.shape[0] + 1, device=returns.device),
        torch.as_tensor(window, device=returns.device),
    ).to(returns.dtype)
    return torch.sqrt(torch.clamp_min(rolling_sum_exact(returns.square(), window) / count, 0))


def zero_run_age(returns: torch.Tensor) -> torch.Tensor:
    indices = torch.arange(returns.shape[0], device=returns.device)
    last_active = torch.where(returns != 0, indices, torch.full_like(indices, -1))
    last_active = torch.cummax(last_active, dim=0).values
    return (indices - last_active).to(returns.dtype) * (returns == 0).to(returns.dtype)


def carry_completed_minutes(values: torch.Tensor, second_rows: int) -> torch.Tensor:
    """Carry minute m from second m*60+59 through the following 59 seconds."""
    output_shape = (second_rows,) + tuple(values.shape[1:])
    output = torch.zeros(output_shape, dtype=values.dtype, device=values.device)
    repeated = values.repeat_interleave(60, dim=0)
    if second_rows > 59:
        output[59:] = repeated[:second_rows - 59]
    return output


def candle_columns_to_tensor(columns: Mapping[str, torch.Tensor]) -> torch.Tensor:
    return torch.stack(
        tuple(columns[name] for name in ("open", "high", "low", "close", "volume")),
        dim=1,
    )


@dataclass(frozen=True)
class Production59Base:
    # [second, open/high/low/close/baseVolume]
    btc_second_candles: torch.Tensor
    # Primitive aggregate-trade columns, each [second].
    btc_trade_flow: Mapping[str, torch.Tensor]
    # [minute, open/high/low/close/baseVolume/quoteVolume/tradeCount/
    #  takerBuyBaseVolume/takerBuyQuoteVolume]
    btc_minute_candles: torch.Tensor
    eth_minute_candles: torch.Tensor
    btc_futures_minute: torch.Tensor
    start_ms: int


@dataclass(frozen=True)
class Global471Base:
    """Primitive histories used by the selected 471-channel feature graph."""

    # Raw numeric tensors keyed by the source path in dataset.json.  Entries
    # are either canonical OHLCV(/quote/count/taker/observed) matrices,
    # one-second close timelines, native 5m metrics, depth snapshots, or the
    # recovered raw ETH trade-count timeline.
    tensors: Mapping[str, torch.Tensor]
    # Availability masks keyed by their explicit source path.
    observed: Mapping[str, torch.Tensor]
    # Aggregate-trade primitives keyed by immutable reference directory.
    trade_flows: Mapping[str, Mapping[str, torch.Tensor]]
    # Funding settlement primitives keyed by immutable/cache JSON path.
    funding_events: Mapping[str, tuple[torch.Tensor, torch.Tensor]]
    # Rows/times of the 42,901-point working feature axis.
    minute_origin_rows: torch.Tensor
    minute_origin_times_ms: torch.Tensor
    minute_rows: int
    second_rows: int


def _rsi(close: torch.Tensor, period: int) -> torch.Tensor:
    changes = torch.diff(close, prepend=close[:1])
    gain = torch.clamp_min(
        ema_filter_fft(torch.clamp_min(changes, 0), 1.0 / period, initial=0.0), 0
    )
    loss = torch.clamp_min(
        ema_filter_fft(torch.clamp_min(-changes, 0), 1.0 / period, initial=0.0), 0
    )
    result = torch.full_like(close, 50.0)
    positive_loss = loss > 0
    result = torch.where(positive_loss, 100.0 - 100.0 / (1.0 + gain / loss), result)
    result = torch.where(
        (~positive_loss) & (gain > 0), torch.full_like(result, 100.0), result
    )
    # Gain and loss have the same decay.  Their ratio is therefore exactly
    # constant through a zero-change run, while a very long FFT convolution
    # can introduce tiny independent roundoff into the two decaying states.
    # Carrying the last active ratio is both mathematically exact and stable.
    indices = torch.arange(close.shape[0], device=close.device)
    active_indices = torch.where(
        changes != 0, indices, torch.full_like(indices, -1)
    )
    latest_active = torch.cummax(active_indices, dim=0).values
    safe_active = torch.clamp_min(latest_active, 0)
    carried = result[safe_active]
    smallest = 5e-324 if close.dtype == torch.float64 else 1.401298464e-45
    zero_age = indices - latest_active
    carried_magnitude = torch.maximum(gain, loss)[safe_active]
    still_representable = (
        torch.log(torch.clamp_min(carried_magnitude, smallest))
        + zero_age.to(close.dtype) * math.log(1.0 - 1.0 / period)
        >= math.log(smallest)
    )
    valid_carry = (latest_active >= 0) & still_representable
    return torch.where(valid_carry, carried, torch.full_like(carried, 50.0))


def _one_hot_side(side: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = side.dtype
    return (
        (side < 0).to(dtype),
        (side == 0).to(dtype),
        (side > 0).to(dtype),
    )


def _trade_flow_features(flow: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    buy_quote = flow["aggressiveBuyQuoteVolume"]
    sell_quote = flow["aggressiveSellQuoteVolume"]
    buy_aggregate = flow["aggressiveBuyAggregateTradeCount"]
    sell_aggregate = flow["aggressiveSellAggregateTradeCount"]
    quote_imbalance = imbalance(buy_quote, sell_quote)
    quote_ema2 = imbalance(
        ema_filter_fft(buy_quote, 2.0 / 3.0),
        ema_filter_fft(sell_quote, 2.0 / 3.0),
    )
    quote_ema8 = imbalance(
        ema_filter_fft(buy_quote, 2.0 / 9.0),
        ema_filter_fft(sell_quote, 2.0 / 9.0),
    )
    return {
        "lastSide": flow["lastAggressorSide"],
        "lastSideLag2": lag(flow["lastAggressorSide"], 1),
        "quoteImbalance": quote_imbalance,
        "quoteImbalanceEma2": quote_ema2,
        "quoteImbalanceEma8": quote_ema8,
        "aggregateCountImbalance": imbalance(buy_aggregate, sell_aggregate),
    }


def _futures_minute_features(
    futures: torch.Tensor,
    spot: torch.Tensor,
) -> dict[str, torch.Tensor]:
    # Both tensors use the canonical nine numeric kline columns.
    future_close = futures[:, 3]
    spot_close = spot[:, 3]
    basis = torch.log(future_close / spot_close) * 10_000.0
    future_return = torch.diff(torch.log(future_close), prepend=torch.log(future_close[:1])) * 10_000.0
    spot_return = torch.diff(torch.log(spot_close), prepend=torch.log(spot_close[:1])) * 10_000.0
    return {
        "tradeCount": torch.log1p(futures[:, 6]),
        "range": torch.log(futures[:, 1] / futures[:, 2]) * 10_000.0,
        "basis": basis,
        "basisChange": basis - lag(basis, 1, fill=float(basis[0].detach().cpu())),
        "relativeReturn": future_return - spot_return,
    }


def _completed_hour_values(volume: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    count = volume.shape[0]
    output = torch.zeros_like(volume)
    age = torch.zeros_like(volume)
    complete_hours = count // 3_600
    if complete_hours == 0:
        return output, age
    hour_values = torch.log1p(volume[:complete_hours * 3_600].reshape(-1, 3_600).sum(dim=1))
    repeated = hour_values.repeat_interleave(3_600)
    output[3_599:] = repeated[:count - 3_599]
    indices = torch.arange(count, dtype=volume.dtype, device=volume.device)
    completed_boundary = torch.floor((indices + 1) / 3_600) * 3_600
    completed_boundary = torch.clamp_min(completed_boundary, 3_600)
    age = torch.log1p(torch.clamp_min(indices + 1 - completed_boundary, 0))
    return output, age


def production59_features(
    base: Production59Base,
    origins: torch.Tensor | None = None,
) -> torch.Tensor:
    """Rebuild the exact 59 production channels from primitive histories."""
    candle = base.btc_second_candles
    if candle.ndim != 2 or candle.shape[1] != 5:
        raise ValueError("BTC second candle base must have five columns")
    count = candle.shape[0]
    open_, high, low, close, volume = candle.unbind(dim=1)
    returns = torch.diff(torch.log(close), prepend=torch.log(close[:1]))
    rms = {window: rolling_rms(returns, window) for window in (5, 15, 60)}
    rms60 = torch.clamp_min(rms[60], 1e-8)
    active = (returns != 0).to(returns.dtype)
    channels: list[torch.Tensor] = [returns]
    channels.extend((
        returns / rms60,
        lag(returns, 1) / rms60,
        (returns == 0).to(returns.dtype),
        (lag(returns, 1) == 0).to(returns.dtype),
        rolling_sum_exact(active, 10) / 10.0,
        rolling_sum_exact(active, 60) / 60.0,
        torch.log1p(torch.clamp_max(zero_run_age(returns), 3_600)),
    ))
    anchor = log_rms(returns, 3_600)
    channels.append(anchor)
    for window in (5, 15, 60, 300, 900, 1_800, 14_400):
        channels.append(log_rms(returns, window) - anchor)
    for window in (5, 15, 60, 300, 900, 3_600):
        divisor = torch.minimum(
            torch.arange(1, count + 1, device=returns.device),
            torch.as_tensor(window, device=returns.device),
        ).to(returns.dtype)
        channels.append(
            torch.log(1e-12 + rolling_sum_exact(torch.abs(returns), window) / divisor)
        )

    rsi2 = _rsi(close, 2)
    ema2 = ema_filter_fft(close, 2.0 / 3.0)
    ema2_log = torch.log(ema2) * 10_000.0
    slope2 = ema2_log - lag(ema2_log, 1, fill=float(ema2_log[0].detach().cpu()))
    acceleration2 = slope2 - lag(slope2, 1, fill=float(slope2[0].detach().cpu()))
    ema8 = ema_filter_fft(close, 2.0 / 9.0)
    ema8_log = torch.log(ema8) * 10_000.0
    slope8 = (
        ema8_log - lag(ema8_log, 8, fill=float(ema8_log[0].detach().cpu()))
    ) / 8.0
    channels.extend((
        (rsi2 - 50.0) / 50.0,
        acceleration2 / torch.clamp_min(rms[5] * 10_000.0, 1e-4),
        slope8 / torch.clamp_min(rms[15] * 10_000.0, 1e-4),
    ))

    completed_hour_volume, completed_hour_age = _completed_hour_values(volume)
    channels.extend((
        completed_hour_volume,
        torch.ones_like(returns),
        completed_hour_age,
        torch.log1p(torch.log(high / low) / rms60),
        torch.where(high > low, (2 * close - high - low) / (high - low), torch.zeros_like(close)),
    ))
    sum16 = rolling_sum_exact(returns, 16)
    absolute16 = rolling_sum_exact(torch.abs(returns), 16)
    square16 = torch.clamp_min(rolling_sum_exact(returns.square(), 16), 0)
    square16_root = torch.sqrt(square16 + 1e-24)
    channels.extend((
        (lag(returns, 1) - returns) / torch.clamp_min(
            math.sqrt(2.0) * square16_root, 1e-12
        ),
        safe_divide(sum16, absolute16),
        safe_divide(sum16, square16_root),
    ))

    trade = _trade_flow_features(base.btc_trade_flow)
    channels.extend((*_one_hot_side(trade["lastSide"]), *_one_hot_side(trade["lastSideLag2"])))
    channels.extend((
        trade["quoteImbalance"],
        trade["quoteImbalanceEma2"],
        trade["quoteImbalanceEma8"],
        trade["aggregateCountImbalance"],
    ))

    minute = _futures_minute_features(base.btc_futures_minute, base.btc_minute_candles)
    channels.extend((
        carry_completed_minutes(minute["tradeCount"], count),
        carry_completed_minutes(minute["range"], count),
        torch.ones_like(returns),
    ))
    minute_indices = torch.arange(count, dtype=returns.dtype, device=returns.device)
    channels.append(torch.log1p(torch.remainder(minute_indices + 1, 60)))

    btc_minute_return = torch.diff(
        torch.log(base.btc_minute_candles[:, 3]),
        prepend=torch.log(base.btc_minute_candles[:1, 3]),
    )
    eth_minute_return = torch.diff(
        torch.log(base.eth_minute_candles[:, 3]),
        prepend=torch.log(base.eth_minute_candles[:1, 3]),
    )
    for window in (30, 60):
        difference = log_rms(eth_minute_return, window) - log_rms(btc_minute_return, window)
        channels.append(carry_completed_minutes(difference, count))
    channels.extend((torch.ones_like(returns), torch.log1p(torch.remainder(minute_indices + 1, 60))))

    origin_ms = base.start_ms + (torch.arange(count, device=returns.device) + 1) * 1_000
    seconds = torch.remainder(origin_ms // 1_000, 60).to(returns.dtype)
    minutes = torch.remainder(origin_ms // 60_000, 60).to(returns.dtype)
    hours = torch.remainder(origin_ms // 3_600_000, 24).to(returns.dtype)
    # 1970-01-01 was Thursday (3 when Sunday=0).
    days = torch.remainder(origin_ms // 86_400_000 + 4, 7).to(returns.dtype)
    for value, period in ((seconds, 60), (minutes, 60), (hours, 24), (days, 7)):
        angle = 2 * math.pi * value / period
        channels.extend((torch.sin(angle), torch.cos(angle)))

    if len(channels) != 59:
        raise AssertionError(f"production feature graph emitted {len(channels)} channels")
    if origins is not None:
        selected = origins.to(device=returns.device, dtype=torch.long)
        channels = [channel[selected] for channel in channels]
    return torch.stack(channels, dim=1)


_DENSE_RE = re.compile(
    r"^(?P<kind>rsi|ema-distance|ema-slope|ema-acceleration)-(?P<period>\d+)m"
    r"(?:-(?P<horizon>\d+)m)?(?:-lag-(?P<lag>\d+)m)?$"
)


def dense_minute_feature(close: torch.Tensor, formula: str) -> torch.Tensor:
    """Differentiable implementation of every selected dense-minute formula."""
    match = _DENSE_RE.fullmatch(formula)
    if match is None:
        raise ValueError(f"unsupported dense-minute formula {formula!r}")
    kind = match.group("kind")
    period = int(match.group("period"))
    horizon = int(match.group("horizon") or 0)
    feature_lag = int(match.group("lag") or 0)
    if kind == "rsi":
        result = _rsi(close, period)
    else:
        average = ema_filter_fft(close, 2.0 / (period + 1.0))
        if kind == "ema-distance":
            result = 10_000.0 * torch.log(close / average)
        else:
            if horizon < 1:
                raise ValueError(f"EMA dynamics formula lacks a horizon: {formula}")
            slope = torch.zeros_like(average)
            slope[horizon:] = (
                10_000.0
                * torch.log(average[horizon:] / average[:-horizon])
                / horizon
            )
            if kind == "ema-slope":
                result = slope
            else:
                result = torch.zeros_like(slope)
                result[2 * horizon:] = (
                    slope[2 * horizon:] - slope[horizon:-horizon]
                )
    return lag(result, feature_lag, fill=float("nan")) if feature_lag else result


def causal_fill(values: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Forward-fill a primitive timeline while preserving gradients to sources."""
    if values.ndim != 1 or observed.shape != values.shape:
        raise ValueError("causal fill expects matching one-dimensional timelines")
    indices = torch.arange(values.shape[0], device=values.device)
    valid = observed.to(torch.bool) & torch.isfinite(values) & (values > 0)
    if not bool(valid.any()):
        raise ValueError("causal source has no observed positive value")
    source = torch.where(valid, indices, torch.full_like(indices, -1))
    source = torch.cummax(source, dim=0).values
    first = torch.nonzero(valid, as_tuple=False)[0, 0]
    source = torch.where(source >= 0, source, first)
    return values[source]


_TECHNICAL_RE = re.compile(
    r"^(?P<kind>rsi|ema-distance|ema-slope|ema-acceleration)-(?P<period>\d+)s"
    r"(?:-(?P<horizon>\d+)s)?$"
)
_MACD_RE = re.compile(
    r"^macd-(?P<kind>line|histogram)-(?P<fast>\d+)-(?P<slow>\d+)-"
    r"(?P<signal>\d+)-1s-cadence$"
)


def technical_second_feature(close: torch.Tensor, formula: str) -> torch.Tensor:
    match = _TECHNICAL_RE.fullmatch(formula)
    if match is not None:
        kind = match.group("kind")
        period = int(match.group("period"))
        horizon = int(match.group("horizon") or 0)
        if kind == "rsi":
            return _rsi(close, period)
        average = ema_filter_fft(close, 2.0 / (period + 1.0))
        if kind == "ema-distance":
            return 10_000.0 * torch.log(close / average)
        if horizon < 1:
            raise ValueError(f"technical EMA dynamics lacks a horizon: {formula}")
        previous = lag(average, horizon, fill=float(average[0].detach().cpu()))
        slope = 10_000.0 * torch.log(average / previous) / horizon
        return slope if kind == "ema-slope" else (
            slope - lag(slope, horizon, fill=float(slope[0].detach().cpu()))
        )
    macd = _MACD_RE.fullmatch(formula)
    if macd is None:
        raise ValueError(f"unsupported one-second technical formula {formula!r}")
    fast = int(macd.group("fast"))
    slow = int(macd.group("slow"))
    signal_period = int(macd.group("signal"))
    line = (
        ema_filter_fft(close, 2.0 / (fast + 1.0))
        - ema_filter_fft(close, 2.0 / (slow + 1.0))
    )
    value = line
    if macd.group("kind") == "histogram":
        value = line - ema_filter_fft(line, 2.0 / (signal_period + 1.0), initial=0.0)
    return 10_000.0 * value / close


def history_windows(
    values: torch.Tensor,
    origins: torch.Tensor,
    window: int,
) -> torch.Tensor:
    offsets = torch.arange(window - 1, -1, -1, device=values.device)
    indices = origins[:, None] - offsets[None, :]
    if bool((indices < 0).any()) or bool((indices >= values.shape[0]).any()):
        raise ValueError("feature window escapes its base timeline")
    return values[indices]


def _fractional_dft(values: torch.Tensor, order: float) -> torch.Tensor:
    values = values.to(torch.complex128 if values.dtype == torch.float64 else torch.complex64)
    powers = [values]
    for _ in range(3):
        powers.append(torch.fft.fft(powers[-1], dim=1, norm="ortho"))
    result = torch.zeros_like(powers[0])
    for power, transformed in enumerate(powers):
        coefficient = sum(
            complex(
                math.cos(-0.5 * math.pi * order * eigen + 0.5 * math.pi * power * eigen),
                math.sin(-0.5 * math.pi * order * eigen + 0.5 * math.pi * power * eigen),
            )
            for eigen in range(4)
        ) / 4.0
        result = result + coefficient * transformed
    return result


def _haar_components(windows: torch.Tensor) -> dict[str, torch.Tensor]:
    approximation = windows
    detail_energies: list[torch.Tensor] = []
    latest: list[torch.Tensor] = []
    norm = torch.sqrt(torch.clamp_min(windows.square().sum(dim=1), torch.finfo(windows.dtype).tiny))
    while approximation.shape[1] >= 2:
        left = approximation[:, 0::2]
        right = approximation[:, 1::2]
        detail = (left - right) / math.sqrt(2.0)
        approximation = (left + right) / math.sqrt(2.0)
        detail_energies.append(detail.square().sum(dim=1))
        latest.append(detail[:, -1] / norm)
    total = windows.square().sum(dim=1)
    safe_total = torch.clamp_min(total, torch.finfo(windows.dtype).tiny)
    used = sum(detail_energies[:4])
    return {
        "haar-fine-energy-share": sum(detail_energies[:2]) / safe_total,
        "haar-mid-energy-share": sum(detail_energies[2:4]) / safe_total,
        "haar-coarse-energy-share": (total - used) / safe_total,
        "haar-latest-detail-l1": latest[0],
        "haar-latest-detail-l2": latest[1],
        "haar-latest-detail-l3": latest[2],
    }


_SPECTRAL_WINDOW_RE = re.compile(r"-(?P<window>\d+)(?P<unit>[sm])$")


def spectral_feature_from_returns(
    returns: torch.Tensor,
    origins: torch.Tensor,
    formula: str,
) -> torch.Tensor:
    """Differentiable counterpart of analyze_fourier_return_features.py."""
    match = _SPECTRAL_WINDOW_RE.search(formula)
    if match is None:
        raise ValueError(f"spectral formula lacks a window: {formula}")
    window = int(match.group("window"))
    stem = formula[:match.start()]
    valid_history = (origins >= window - 1) & (origins < returns.shape[0])
    safe_origins = torch.clamp(origins, min=window - 1, max=returns.shape[0] - 1)
    windows = history_windows(returns, safe_origins, window)
    weights = torch.hann_window(
        window, periodic=False, dtype=windows.dtype, device=windows.device
    )
    windowed = windows * weights
    transformed = torch.fft.rfft(windowed, dim=1)
    positive = transformed[:, 1:].abs().square()
    total = positive.sum(dim=1)
    empty = total <= torch.finfo(windows.dtype).tiny
    raw_norm = torch.sqrt(torch.clamp_min(windows.square().sum(dim=1), 0))
    if stem == "fft-log-energy":
        result = torch.log1p(raw_norm)
        return torch.where(
            valid_history & ~empty, result, torch.zeros_like(result)
        )
    if stem.startswith("haar-"):
        result = _haar_components(windows)[stem]
        return torch.where(
            valid_history & ~empty, result, torch.zeros_like(result)
        )
    safe_total = torch.clamp_min(total, torch.finfo(windows.dtype).tiny)
    proportions = positive / safe_total[:, None]
    bins = positive.shape[1]
    if stem == "fft-low-power-share":
        result = proportions[:, :max(1, window // 8)].sum(dim=1)
    elif stem == "fft-high-power-share":
        result = proportions[:, max(max(1, window // 8), window // 4):].sum(dim=1)
    elif stem == "fft-entropy":
        terms = torch.where(
            proportions > 0,
            proportions * torch.log(torch.clamp_min(proportions, torch.finfo(windows.dtype).tiny)),
            torch.zeros_like(proportions),
        )
        result = -terms.sum(dim=1) / math.log(bins)
    elif stem == "fft-centroid":
        frequencies = torch.arange(1, bins + 1, dtype=windows.dtype, device=windows.device) / bins
        result = proportions @ frequencies
    elif stem == "fft-dominant-frequency":
        result = (positive.argmax(dim=1).to(windows.dtype) + 1) / bins
    elif re.fullmatch(r"fft-k[124]-(real|imag)", stem):
        index = int(stem.split("-")[1][1:])
        value = transformed[:, index] / torch.sqrt(safe_total)
        result = value.real if stem.endswith("real") else value.imag
    elif stem.startswith("frft-"):
        frft = re.fullmatch(r"frft-(0p25|0p5|0p75)-(k1-real|k1-imag|entropy)", stem)
        if frft is None:
            raise ValueError(f"unsupported fractional Fourier feature {formula}")
        order = {"0p25": 0.25, "0p5": 0.5, "0p75": 0.75}[frft.group(1)]
        fractional = _fractional_dft(windowed, order)
        if frft.group(2) == "k1-real":
            result = fractional[:, 1].real / torch.sqrt(torch.clamp_min(windowed.square().sum(dim=1), torch.finfo(windows.dtype).tiny))
        elif frft.group(2) == "k1-imag":
            result = fractional[:, 1].imag / torch.sqrt(torch.clamp_min(windowed.square().sum(dim=1), torch.finfo(windows.dtype).tiny))
        else:
            power = fractional.abs().square()
            probability = power / torch.clamp_min(power.sum(dim=1), torch.finfo(windows.dtype).tiny)[:, None]
            terms = torch.where(
                probability > 0,
                probability * torch.log(torch.clamp_min(probability, torch.finfo(windows.dtype).tiny)),
                torch.zeros_like(probability),
            )
            result = -terms.sum(dim=1) / math.log(window)
    elif stem.startswith("morlet-"):
        morlet = re.fullmatch(r"morlet-(fast|slow)-(real|imag)", stem)
        if morlet is None:
            raise ValueError(f"unsupported Morlet feature {formula}")
        scale = window / (8 if morlet.group(1) == "fast" else 2)
        history_lag = torch.arange(window - 1, -1, -1, dtype=windows.dtype, device=windows.device)
        kernel = torch.exp(-0.5 * (history_lag / scale).square()).to(
            torch.complex128 if windows.dtype == torch.float64 else torch.complex64
        ) * torch.exp(-6j * history_lag / scale)
        kernel = kernel / torch.sqrt(kernel.abs().square().sum())
        norm = torch.sqrt(torch.clamp_min(windowed.square().sum(dim=1), torch.finfo(windows.dtype).tiny))
        value = windows.to(kernel.dtype) @ kernel / norm
        result = value.real if morlet.group(2) == "real" else value.imag
    else:
        raise ValueError(f"unsupported spectral formula {formula!r}")
    return torch.where(
        valid_history & ~empty, result, torch.zeros_like(result)
    )


def minute_market_feature(values: torch.Tensor, formula: str) -> torch.Tensor:
    """Primitive one-minute OHLCV/trade transforms used by selected features."""
    if values.ndim != 2 or values.shape[1] < 5:
        raise ValueError("minute market base must contain at least OHLCV")
    open_, high, low, close, base_volume = (values[:, index] for index in range(5))
    returns = torch.diff(torch.log(close), prepend=torch.log(close[:1])) * 10_000.0
    match = re.fullmatch(r"realized-volatility-(\d+)m", formula)
    if match:
        return torch.sqrt(torch.clamp_min(rolling_sum_exact(returns.square(), int(match.group(1))), 0))
    match = re.fullmatch(r"return-(\d+)m", formula)
    if match:
        horizon = int(match.group(1))
        return torch.log(close / lag(close, horizon, fill=float(close[0].detach().cpu()))) * 10_000.0
    match = re.fullmatch(r"active-fraction-(\d+)m", formula)
    if match:
        window = int(match.group(1))
        return rolling_sum_exact((returns != 0).to(returns.dtype), window) / window
    if formula in {"zero-run-age", "zero-run-age-1s"}:
        return torch.log1p(zero_run_age(returns))
    if formula == "range-1m":
        return torch.log(high / low) * 10_000.0
    if formula == "log-quote-volume-1m":
        quote_volume = values[:, 5]
        return torch.log1p(quote_volume)
    if formula == "log-trade-count-1m":
        trade_count = values[:, 6]
        return torch.log1p(trade_count)
    if formula == "log-mean-trade-notional-1m":
        quote_volume, trade_count = values[:, 5], values[:, 6]
        return torch.log((quote_volume + 1) / (trade_count + 1))
    if formula == "taker-imbalance-ema-60m":
        quote_volume, taker_buy_quote = values[:, 5], values[:, 8]
        current = safe_divide(2 * taker_buy_quote - quote_volume, quote_volume)
        return ema_filter_fft(current, 2.0 / 61.0)
    match = re.fullmatch(r"completed-(5|15|60)m-log-volume", formula)
    if match:
        window = int(match.group(1))
        result = torch.zeros_like(base_volume)
        buckets = base_volume[:base_volume.shape[0] // window * window].reshape(-1, window).sum(dim=1)
        carried = torch.log1p(buckets).repeat_interleave(window)
        result[window - 1:] = carried[:result.shape[0] - window + 1]
        return result
    raise ValueError(f"unsupported minute market formula {formula!r}")


def book_depth_feature(values: torch.Tensor, formula: str) -> torch.Tensor:
    if values.ndim != 2 or values.shape[1] < 24:
        raise ValueError("book-depth base must contain twelve depth and twelve notional bands")
    observed = values[:, 24] == 1 if values.shape[1] > 24 else torch.ones(
        values.shape[0], dtype=torch.bool, device=values.device
    )
    missing = torch.full_like(values[:, :12], float("nan"))
    depth = torch.where(observed[:, None], values[:, :12], missing)
    notional = torch.where(observed[:, None], values[:, 12:24], missing)
    if formula == "book-log-depth-5pct":
        return torch.log(depth[:, 0] + depth[:, 11])
    if formula == "book-log-notional-5pct":
        return torch.log(notional[:, 0] + notional[:, 11])
    match = re.fullmatch(r"book-depth-imbalance-([1-5])pct", formula)
    if match:
        band = int(match.group(1))
        bid_index = 5 - band
        ask_index = 6 + band
        return imbalance(depth[:, bid_index], depth[:, ask_index])
    raise ValueError(f"unsupported selected book formula {formula!r}")


def metrics_feature(values: torch.Tensor, formula: str) -> torch.Tensor:
    """Five-minute USD-M metrics, returned at their native 5m cadence."""
    names = {
        "open-interest": 0,
        "open-interest-value": 1,
        "top-account-long-short": 2,
        "top-position-long-short": 3,
        "global-long-short": 4,
        "taker-buy-sell": 5,
    }
    if values.ndim != 2 or values.shape[1] < 6:
        raise ValueError("metrics base must contain six primitive ratio columns")
    observed = values[:, 6] == 1 if values.shape[1] > 6 else torch.ones(
        values.shape[0], dtype=torch.bool, device=values.device
    )
    transformed = {
        name: torch.where(
            observed,
            torch.log(values[:, index]),
            torch.full_like(values[:, index], float("nan")),
        )
        for name, index in names.items()
    }
    if formula == "top-position-minus-account-long-short":
        return transformed["top-position-long-short"] - transformed["top-account-long-short"]
    level = re.fullmatch(r"(.+)-log-level", formula)
    if level and level.group(1) in transformed:
        return transformed[level.group(1)]
    change = re.fullmatch(r"(.+)-log-change-(5|15|60|240)m", formula)
    if change and change.group(1) in transformed:
        buckets = int(change.group(2)) // 5
        series = transformed[change.group(1)]
        return series - lag(series, buckets, fill=float(series[0].detach().cpu()))
    raise ValueError(f"unsupported selected metrics formula {formula!r}")


def funding_feature(
    event_times_ms: torch.Tensor,
    rates: torch.Tensor,
    origin_times_ms: torch.Tensor,
    formula: str,
) -> torch.Tensor:
    available = event_times_ms + 60_000
    latest = torch.searchsorted(available, origin_times_ms, right=True) - 1
    match = re.fullmatch(r"funding-(mean|absolute-mean)-(\d+)", formula)
    if match is None:
        raise ValueError(f"unsupported selected funding formula {formula!r}")
    window = int(match.group(2))
    enough = latest >= window - 1
    safe_latest = torch.clamp(latest, min=window - 1, max=rates.shape[0] - 1)
    selected = history_windows(
        torch.abs(rates) if match.group(1) == "absolute-mean" else rates,
        safe_latest,
        window,
    )
    result = selected.mean(dim=1)
    return torch.where(enough, result, torch.full_like(result, float("nan")))


def _minute_close(values: torch.Tensor, *, causal: bool) -> tuple[torch.Tensor, torch.Tensor]:
    if values.ndim != 2 or values.shape[1] < 10:
        raise ValueError("global minute source must have ten canonical columns")
    observed = values[:, 9] == 1
    close = values[:, 3]
    if causal:
        close = causal_fill(close, observed)
    else:
        close = torch.where(observed, close, torch.full_like(close, float("nan")))
    return close, observed


def _log_returns(close: torch.Tensor) -> torch.Tensor:
    result = torch.full_like(close, float("nan"))
    valid = (
        torch.isfinite(close[1:]) & torch.isfinite(close[:-1])
        & (close[1:] > 0) & (close[:-1] > 0)
    )
    current = torch.log(close[1:] / close[:-1]) * 10_000.0
    result[1:] = torch.where(valid, current, torch.full_like(current, float("nan")))
    return result


def _rolling_sum_full(values: torch.Tensor, window: int) -> torch.Tensor:
    """Full trailing window sum with NaN when any member is unavailable."""
    finite = torch.isfinite(values)
    clean = torch.where(finite, values, torch.zeros_like(values))
    sums = rolling_sum_exact(clean, window)
    counts = rolling_sum_exact(finite.to(values.dtype), window)
    rows = torch.arange(values.shape[0], device=values.device)
    valid = (rows >= window - 1) & (counts == window)
    return torch.where(valid, sums, torch.full_like(sums, float("nan")))


def _completed_bucket_log_volume(volume: torch.Tensor, window: int) -> torch.Tensor:
    result = torch.full_like(volume, float("nan"))
    complete = volume.shape[0] // window
    if complete:
        buckets = torch.log1p(
            volume[:complete * window].reshape(complete, window).sum(dim=1)
        )
        carried = buckets.repeat_interleave(window)
        result[window - 1:] = carried[:result.shape[0] - window + 1]
    return result


def _representative_minute_feature(values: torch.Tensor, formula: str) -> torch.Tensor:
    close, _observed = _minute_close(values, causal=False)
    returns = _log_returns(close)
    if match := re.fullmatch(r"realized-volatility-(\d+)m", formula):
        return torch.sqrt(torch.clamp_min(
            _rolling_sum_full(returns.square(), int(match.group(1))), 0
        ))
    if match := re.fullmatch(r"active-fraction-(\d+)m", formula):
        window = int(match.group(1))
        return _rolling_sum_full((returns != 0).to(returns.dtype), window) / window
    if formula == "zero-run-age":
        return torch.log1p(zero_run_age(returns))
    return minute_market_feature(values, formula)


def _long_minute_feature(values: torch.Tensor, formula: str) -> torch.Tensor:
    close, observed = _minute_close(values, causal=True)
    returns = torch.diff(torch.log(close), prepend=torch.log(close[:1])) * 10_000.0
    if formula == "realized-volatility-2m":
        result = torch.full_like(close, float("nan"))
        result[1:] = torch.sqrt(returns[1:].square() + returns[:-1].square())
        return result
    match = re.fullmatch(r"completed-(5|15|60)m-log-volume", formula)
    if match:
        volume = torch.where(observed, values[:, 4], torch.zeros_like(values[:, 4]))
        return _completed_bucket_log_volume(volume, int(match.group(1)))
    raise ValueError(f"unsupported selected long-minute formula {formula!r}")


def _existing_trade_features(flow: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    buy_trade = flow["aggressiveBuyTradeCount"]
    sell_trade = flow["aggressiveSellTradeCount"]
    total_trade = buy_trade + sell_trade
    buy_aggregate = flow["aggressiveBuyAggregateTradeCount"]
    sell_aggregate = flow["aggressiveSellAggregateTradeCount"]
    total_aggregate = buy_aggregate + sell_aggregate
    trade_activity = torch.log1p(total_trade)
    activity_ema = ema_filter_fft(trade_activity, 2.0 / 33.0)
    prior_activity = lag(
        activity_ema, 1, fill=float(activity_ema[0].detach().cpu())
    )
    return {
        "spot-flow-trade-imbalance-ema-2": imbalance(
            ema_filter_fft(buy_trade, 2.0 / 3.0),
            ema_filter_fft(sell_trade, 2.0 / 3.0),
        ),
        "spot-flow-first-side-1s": flow["firstAggressorSide"],
        "spot-flow-flip-rate-1s": torch.where(
            total_aggregate > 1,
            flow["aggressorSideFlipCount"] / (total_aggregate - 1),
            torch.zeros_like(total_aggregate),
        ),
        "spot-flow-last-side-1s": flow["lastAggressorSide"],
        "spot-flow-last-side-lag-2s": lag(flow["lastAggressorSide"], 1),
        "spot-flow-log-trade-count-1s": trade_activity,
        "spot-flow-raw-per-aggregate-1s": torch.log(
            (total_trade + 1) / (total_aggregate + 1)
        ),
        "spot-flow-trade-surprise-1s": trade_activity - prior_activity,
    }


def _existing_futures_features(
    futures: torch.Tensor,
    spot: torch.Tensor,
) -> dict[str, torch.Tensor]:
    close = futures[:, 3]
    spot_close = spot[:, 3]
    basis = torch.log(close / spot_close) * 10_000.0
    future_return = torch.diff(torch.log(close), prepend=torch.log(close[:1])) * 10_000.0
    spot_return = torch.diff(
        torch.log(spot_close), prepend=torch.log(spot_close[:1])
    ) * 10_000.0
    result = {
        "futures-basis-change-1m": basis - lag(
            basis, 1, fill=float(basis[0].detach().cpu())
        ),
        "futures-close-location-1m": torch.where(
            futures[:, 1] > futures[:, 2],
            (close - futures[:, 2]) / (futures[:, 1] - futures[:, 2]),
            torch.full_like(close, 0.5),
        ),
        "futures-log-quote-volume-1m": torch.log1p(futures[:, 5]),
        "futures-log-trade-count-1m": torch.log1p(futures[:, 6]),
        "futures-range-1m": torch.log(futures[:, 1] / futures[:, 2]) * 10_000.0,
        "futures-relative-return-1m": future_return - spot_return,
    }
    for period in (5, 15, 60):
        updated = ema_filter_fft(basis, 2.0 / (period + 1.0))
        prior = lag(updated, 1, fill=float(updated[0].detach().cpu()))
        result[f"futures-basis-deviation-{period}m"] = basis - prior
    return result


def _fast_second_feature(
    close: torch.Tensor,
    formula: str,
    *,
    candles: torch.Tensor | None = None,
    trade_count: torch.Tensor | None = None,
) -> torch.Tensor:
    returns = torch.diff(torch.log(close), prepend=torch.log(close[:1])) * 10_000.0
    if formula == "realized-volatility-5s":
        squares = returns.square()
        result = torch.zeros_like(squares)
        result[4:] = squares.unfold(0, 5, 1).sum(dim=1)
        return torch.sqrt(torch.clamp_min(result, 0))
    if formula == "zero-run-age-1s":
        return torch.log1p(zero_run_age(returns))
    if formula == "rsi-2s-ema-alpha-2-over-3":
        gain = torch.clamp_min(
            ema_filter_fft(torch.clamp_min(returns, 0), 2.0 / 3.0, initial=0.0), 0
        )
        loss = torch.clamp_min(
            ema_filter_fft(torch.clamp_min(-returns, 0), 2.0 / 3.0, initial=0.0), 0
        )
        ratio = torch.where(
            gain + loss > 0, gain / (gain + loss), torch.full_like(gain, 0.5)
        )
        indices = torch.arange(close.shape[0], device=close.device)
        active_indices = torch.where(
            returns != 0, indices, torch.full_like(indices, -1)
        )
        latest_active = torch.cummax(active_indices, dim=0).values
        carried = ratio[torch.clamp_min(latest_active, 0)]
        smallest = 5e-324 if close.dtype == torch.float64 else 1.401298464e-45
        zero_age = indices - latest_active
        carried_magnitude = torch.maximum(gain, loss)[torch.clamp_min(latest_active, 0)]
        still_representable = (
            torch.log(torch.clamp_min(carried_magnitude, smallest))
            + zero_age.to(close.dtype) * math.log(1.0 / 3.0)
            >= math.log(smallest)
        )
        return torch.where(
            (latest_active >= 0) & still_representable,
            carried,
            torch.full_like(carried, 0.5),
        )
    if formula == "ema-log-change-8s-ema-over-8s":
        average = ema_filter_fft(close, 2.0 / 9.0)
        levels = torch.log(average) * 10_000.0
        return levels - lag(levels, 8, fill=float(levels[0].detach().cpu()))
    if formula == "range-1s":
        if candles is None:
            raise ValueError("one-second range requires raw candles")
        return torch.log(candles[:, 1] / candles[:, 2]) * 10_000.0
    if formula == "log-trade-count-1s":
        if trade_count is None:
            raise ValueError("one-second trade count requires its raw count timeline")
        return torch.log1p(trade_count)
    raise ValueError(f"unsupported selected fast-second formula {formula!r}")


def reconstruct_global470(
    base: Global471Base,
    specs: Sequence[Mapping[str, Any]],
) -> torch.Tensor:
    """Reconstruct the 470 minute-axis channels (spread is example-specific)."""
    rows = base.minute_origin_rows.to(torch.long)
    seconds = rows * 60 + 59
    outputs: list[torch.Tensor] = []
    close_cache: dict[tuple[str, str], tuple[torch.Tensor, torch.Tensor]] = {}
    ema_cache: dict[tuple[str, str, int], torch.Tensor] = {}
    rsi_cache: dict[tuple[str, str, int], torch.Tensor] = {}
    trade_cache: dict[str, dict[str, torch.Tensor]] = {}
    futures_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

    def close_for(source: str, cadence: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = (source, cadence)
        if key not in close_cache:
            tensor = base.tensors[source]
            if tensor.ndim == 1:
                observed_source = next(
                    str(spec["observedSource"]) for spec in specs
                    if (spec.get("baseSource") == source or source in spec.get("baseSources", ()))
                    and spec.get("observedSource") is not None
                )
                observed = base.observed[observed_source].to(torch.bool)
                close_cache[key] = (causal_fill(tensor, observed), observed)
            elif cadence == "1s" and tensor.ndim == 2 and tensor.shape[1] >= 5:
                close_cache[key] = (
                    tensor[:, 3],
                    torch.ones(tensor.shape[0], dtype=torch.bool, device=tensor.device),
                )
            else:
                close_cache[key] = _minute_close(tensor, causal=True)
        return close_cache[key]

    def average(source: str, cadence: str, period: int) -> torch.Tensor:
        key = (source, cadence, period)
        if key not in ema_cache:
            close, _ = close_for(source, cadence)
            ema_cache[key] = ema_filter_fft(close, 2.0 / (period + 1.0))
        return ema_cache[key]

    def rsi_value(source: str, cadence: str, period: int) -> torch.Tensor:
        key = (source, cadence, period)
        if key not in rsi_cache:
            close, _ = close_for(source, cadence)
            rsi_cache[key] = _rsi(close, period)
        return rsi_cache[key]

    for spec in specs:
        provider = str(spec["provider"])
        if provider == "top-of-book":
            continue
        formula = str(spec["formula"])
        cadence = str(spec["cadence"])
        value: torch.Tensor
        if provider == "dense-minute":
            source = str(spec["baseSource"])
            match = _DENSE_RE.fullmatch(formula)
            if match is None:
                raise ValueError(f"invalid selected dense formula {formula}")
            kind = match.group("kind")
            period = int(match.group("period"))
            horizon = int(match.group("horizon") or 0)
            feature_lag = int(match.group("lag") or 0)
            close, observed = close_for(source, "1m")
            if kind == "rsi":
                signal = rsi_value(source, "1m", period)
            elif kind == "ema-distance":
                signal = 10_000.0 * torch.log(close / average(source, "1m", period))
            else:
                mean = average(source, "1m", period)
                slope = torch.zeros_like(mean)
                slope[horizon:] = 10_000.0 * torch.log(
                    mean[horizon:] / mean[:-horizon]
                ) / horizon
                if kind == "ema-slope":
                    signal = slope
                else:
                    signal = torch.zeros_like(slope)
                    signal[2 * horizon:] = slope[2 * horizon:] - slope[horizon:-horizon]
            indices = rows - feature_lag
            value = signal[indices]
            value = torch.where(
                observed[indices], value, torch.full_like(value, float("nan"))
            )
        elif provider == "technical-1s":
            source = str(spec["baseSource"])
            match = _TECHNICAL_RE.fullmatch(formula)
            if match:
                kind = match.group("kind")
                period = int(match.group("period"))
                horizon = int(match.group("horizon") or 0)
                close, observed = close_for(source, "1s")
                if kind == "rsi":
                    signal = rsi_value(source, "1s", period)
                elif kind == "ema-distance":
                    signal = 10_000.0 * torch.log(close / average(source, "1s", period))
                else:
                    mean = average(source, "1s", period)
                    current = mean[seconds]
                    previous = mean[seconds - horizon]
                    slope = 10_000.0 * torch.log(current / previous) / horizon
                    if kind == "ema-slope":
                        value = slope
                    else:
                        prior = 10_000.0 * torch.log(
                            previous / mean[seconds - 2 * horizon]
                        ) / horizon
                        value = slope - prior
                    value = torch.where(
                        observed[seconds], value, torch.full_like(value, float("nan"))
                    )
                    outputs.append(value)
                    continue
                value = signal[seconds]
                value = torch.where(
                    observed[seconds], value, torch.full_like(value, float("nan"))
                )
            else:
                macd = _MACD_RE.fullmatch(formula)
                if macd is None:
                    raise ValueError(f"invalid selected technical formula {formula}")
                close, observed = close_for(source, "1s")
                fast = average(source, "1s", int(macd.group("fast")))
                slow = average(source, "1s", int(macd.group("slow")))
                line = fast - slow
                signal = line
                if macd.group("kind") == "histogram":
                    signal = line - ema_filter_fft(
                        line, 2.0 / (int(macd.group("signal")) + 1.0), initial=0.0
                    )
                value = 10_000.0 * signal[seconds] / close[seconds]
                value = torch.where(
                    observed[seconds], value, torch.full_like(value, float("nan"))
                )
        elif provider in {"spectral-1s", "spectral-1m"}:
            source = str(spec["baseSource"])
            close, observed = close_for(source, cadence)
            returns = torch.diff(torch.log(close), prepend=torch.log(close[:1])) * 10_000.0
            origins = seconds if provider == "spectral-1s" else rows
            value = spectral_feature_from_returns(returns, origins, formula)
            if provider == "spectral-1s":
                value = torch.where(
                    observed[seconds], value, torch.full_like(value, float("nan"))
                )
        elif provider == "long-unique":
            source = str(spec["baseSource"])
            value = _long_minute_feature(base.tensors[source], formula)[rows]
        elif provider == "funding-grid":
            source = str(spec["baseSource"])
            event_times, rates = base.funding_events[source]
            value = funding_feature(
                event_times, rates, base.minute_origin_times_ms, formula
            )
        elif provider == "representative-cross-asset":
            sources = tuple(str(v) for v in spec["baseSources"])
            if formula.startswith("book-"):
                value = book_depth_feature(base.tensors[sources[0]], formula)[rows]
            elif cadence == "1m":
                value = _representative_minute_feature(
                    base.tensors[sources[0]], formula
                )[rows]
            elif cadence == "5m":
                native = metrics_feature(base.tensors[sources[0]], formula)
                metric_index = ((torch.arange(base.minute_rows, device=rows.device) + 1) // 5) - 1
                expanded = torch.full(
                    (base.minute_rows,), float("nan"), dtype=native.dtype, device=native.device
                )
                valid = metric_index >= 0
                expanded[valid] = native[metric_index[valid]]
                value = expanded[rows]
            elif cadence == "1s":
                close, observed = close_for(sources[0], "1s")
                candles = next(
                    (base.tensors[s] for s in sources if base.tensors[s].ndim == 2),
                    None,
                )
                count = next(
                    (base.tensors[s] for s in sources
                     if base.tensors[s].ndim == 1 and s != sources[0]),
                    None,
                )
                signal = _fast_second_feature(
                    close, formula, candles=candles, trade_count=count
                )
                sample_indices = seconds if signal.shape[0] == base.second_rows else rows
                value = signal[sample_indices]
                value = torch.where(
                    observed[seconds], value, torch.full_like(value, float("nan"))
                )
            else:
                raise ValueError(f"unsupported representative cadence {cadence}")
        elif provider == "existing-recent":
            sources = tuple(str(v) for v in spec["baseSources"])
            if formula.startswith("spot-flow-"):
                source = sources[0]
                if source not in trade_cache:
                    trade_cache[source] = _existing_trade_features(base.trade_flows[source])
                value = trade_cache[source][formula][seconds]
            elif formula.startswith("futures-"):
                key = (sources[0], sources[1])
                if key not in futures_cache:
                    futures_cache[key] = _existing_futures_features(
                        base.tensors[sources[0]], base.tensors[sources[1]]
                    )
                value = futures_cache[key][formula][rows]
            elif cadence == "1s":
                candles = base.tensors[sources[0]]
                close = candles[:, 3]
                if formula in {"ema-acceleration-2s-1s", "ema-slope-8s-8s"}:
                    value = technical_second_feature(close, formula)[seconds]
                elif formula == "range-1s":
                    value = torch.log(candles[:, 1] / candles[:, 2])[seconds] * 10_000.0
                elif formula == "realized-volatility-60s":
                    returns = torch.diff(
                        torch.log(close), prepend=torch.log(close[:1])
                    ) * 10_000.0
                    value = torch.sqrt(torch.clamp_min(
                        rolling_sum_exact(returns.square(), 60), 0
                    ))[seconds]
                else:
                    raise ValueError(f"unsupported existing 1s formula {formula}")
            elif formula == "completed-1h-log-volume":
                completed, _age = _completed_hour_values(base.tensors[sources[0]][:, 4])
                value = completed[seconds]
            else:
                value = minute_market_feature(base.tensors[sources[0]], formula)[rows]
        else:
            raise ValueError(f"unsupported selected provider {provider!r}")
        outputs.append(value)
    if len(outputs) != 470:
        raise AssertionError(f"global graph emitted {len(outputs)} minute channels")
    return torch.stack(outputs, dim=1)


def spread_from_top_of_book(top_of_book: torch.Tensor) -> torch.Tensor:
    if top_of_book.ndim != 2 or top_of_book.shape[1] != 2:
        raise ValueError("top of book must contain bid and ask")
    bid, ask = top_of_book.unbind(dim=1)
    return 10_000.0 * (ask - bid) / ((ask + bid) / 2.0)

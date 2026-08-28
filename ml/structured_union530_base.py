from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import torch
from scipy.signal import lfilter
from torch import Tensor

from differentiable_union530_features import (
    production59_features,
    reconstruct_global470,
    spread_from_top_of_book,
)
from union530_base_dataset import Union530BaseHistoryDataset


DATASET_CONTRACT = "structured-union530-base-rollout-v1"
OBJECTIVE_CONTRACT = "mean-standardized-derived-union530-feature-mse-v1"
PRODUCTION59_DATASET_CONTRACT = "structured-production59-base-rollout-v1"
PRODUCTION59_OBJECTIVE_CONTRACT = (
    "mean-standardized-derived-production59-feature-mse-v1"
)
PRODUCTION59_BALANCED_OBJECTIVE_CONTRACT = (
    "balanced-return-derived-primitive-production59-mse-v1"
)
BASE_COORDINATE_IDS = (
    "btc-log-return-1s",
    "btc-return-active-logit",
    "btc-range-active-logit",
    "btc-open-gap-log-return-1s",
    "btc-high-excess-log-bps",
    "btc-low-excess-log-bps",
    "spot-buy-quote-volume-log",
    "spot-sell-quote-volume-log",
    "spot-buy-aggregate-count-log",
    "spot-sell-aggregate-count-log",
    "spot-last-side-sell-logit",
    "spot-last-side-none-logit",
    "spot-last-side-buy-logit",
    "spot-book-spread-log-bps",
)
BASE_COORDINATE_COUNT = len(BASE_COORDINATE_IDS)
PRODUCTION_BASE_COORDINATE_IDS = BASE_COORDINATE_IDS[:-1] + (
    "spot-base-volume-log",
    "btc-futures-minute-log-trade-count",
    "btc-futures-minute-range-log-bps",
    "eth-minute-log-return",
)
PRODUCTION_BASE_COORDINATE_COUNT = len(PRODUCTION_BASE_COORDINATE_IDS)
PRODUCTION_FAST_BASE_COORDINATE_COUNT = 14
PRODUCTION_BOUNDARY_BASE_COORDINATE_PERIODS = {
    "btc-futures-minute-log-trade-count": 60,
    "btc-futures-minute-range-log-bps": 60,
    "eth-minute-log-return": 60,
}
SQUARE_WINDOWS = (5, 15, 16, 60, 300, 900, 1_800, 3_600, 14_400)
ABSOLUTE_WINDOWS = (5, 15, 16, 60, 300, 900, 3_600)
ACTIVE_WINDOWS = (10, 60)
SIDE_LOGIT = 20.0
EPSILON = 1e-12


def _ema(values: np.ndarray, alpha: float, *, initial: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    result, _ = lfilter(
        [float(alpha)], [1.0, -(1.0 - float(alpha))], values,
        zi=np.asarray([(1.0 - float(alpha)) * float(initial)], dtype=np.float64),
    )
    return np.asarray(result, dtype=np.float64)


def _prefix(values: np.ndarray) -> np.ndarray:
    return np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(
        np.asarray(values, dtype=np.float64), dtype=np.float64,
    )))


def _safe_divide(numerator: Tensor, denominator: Tensor) -> Tensor:
    valid = denominator != 0
    safe_denominator = torch.where(
        valid, denominator, torch.ones_like(denominator)
    )
    return torch.where(
        valid,
        numerator / safe_denominator,
        torch.zeros_like(numerator),
    )


@dataclass(frozen=True)
class StructuredUnion530Batch:
    inputs: Tensor
    targets: Tensor
    weights: Tensor
    context: Mapping[str, Tensor]


class Production59BaseRollout:
    """Differentiate future production states from sufficient causal state.

    The 470 global channels are carried only for examples whose complete K1/K2
    sequence shares one immutable minute snapshot.  The remaining 59 production
    channels and live spread are rebuilt from the decoder's primitive causal
    coordinates.  No observed future feature is used in the predicted state.
    """

    def __init__(self, second_start_ms: int) -> None:
        self.second_start_ms = int(second_start_ms)

    def __call__(
        self,
        base_coordinates: Tensor,
        current_features: Tensor,
        context: Mapping[str, Tensor],
    ) -> Tensor:
        if base_coordinates.ndim != 3 \
                or base_coordinates.shape[1] < 1 \
                or base_coordinates.shape[2] not in (
                    PRODUCTION_BASE_COORDINATE_COUNT, BASE_COORDINATE_COUNT,
                ):
            raise ValueError(
                "base rollout expects [batch,steps,17] production "
                "coordinates or [batch,steps,14] union coordinates"
            )
        if current_features.ndim != 2 or current_features.shape[1] not in (59, 530):
            raise ValueError("current feature state must contain 59 or 530 channels")
        union_output = current_features.shape[1] == 530
        if union_output and base_coordinates.shape[2] != BASE_COORDINATE_COUNT:
            raise ValueError("union rollout requires the spot-spread coordinate")

        output_dtype = base_coordinates.dtype
        base_coordinates = base_coordinates.double()
        current_features = current_features.double()
        context = {
            name: value if name == "secondIndex" else value.double()
            for name, value in context.items()
        }
        square = context["squareSums"].clone()
        absolute = context["absoluteSums"].clone()
        active_sums = context["activeSums"].clone()
        previous_return = context["previousReturn"]
        previous_active = context["previousActive"]
        zero_age = context["zeroAge"]
        previous_close = context["previousClose"]
        rsi_gain = context["rsiGain"]
        rsi_loss = context["rsiLoss"]
        ema2 = context["ema2"]
        ema2_previous = context["ema2Previous"]
        ema8 = context["ema8"]
        quote_buy_ema2 = context["quoteBuyEma2"]
        quote_sell_ema2 = context["quoteSellEma2"]
        quote_buy_ema8 = context["quoteBuyEma8"]
        quote_sell_ema8 = context["quoteSellEma8"]
        previous_side = context["previousSide"]
        second_index = context["secondIndex"].to(torch.long)
        signed16 = context["signed16"]
        if not union_output:
            hour_volume_sum = context["hourVolumeSum"]
            completed_hour_volume = current_features[:, 25]
            futures_trade_count = current_features[:, 43]
            futures_range = current_features[:, 44]
            minute_volatility_30 = current_features[:, 47]
            minute_volatility_60 = current_features[:, 48]
            btc_minute_square = context["btcMinuteSquareSums"].clone()
            eth_minute_square = context["ethMinuteSquareSums"].clone()
            previous_btc_minute_close = context["previousBtcMinuteClose"]

        square_index = {window: index for index, window in enumerate(SQUARE_WINDOWS)}
        absolute_index = {
            window: index for index, window in enumerate(ABSOLUTE_WINDOWS)
        }
        active_index = {window: index for index, window in enumerate(ACTIVE_WINDOWS)}
        outputs: list[Tensor] = []
        predicted_returns: list[Tensor] = []
        predicted_active: list[Tensor] = []
        predicted_ema8: list[Tensor] = []

        for step in range(base_coordinates.shape[1]):
            coordinates = base_coordinates[:, step]
            active_probability = torch.sigmoid(coordinates[:, 1])
            range_probability = torch.sigmoid(coordinates[:, 2])
            current_return = coordinates[:, 0] * active_probability
            current_close = previous_close * torch.exp(current_return)
            current_open = previous_close * torch.exp(coordinates[:, 3])
            upper_bps = torch.exp(torch.clamp(coordinates[:, 4], -30.0, 20.0))
            lower_bps = torch.exp(torch.clamp(coordinates[:, 5], -30.0, 20.0))
            high = torch.maximum(current_open, current_close) * torch.exp(
                upper_bps / 10_000.0
            )
            low = torch.minimum(current_open, current_close) * torch.exp(
                -lower_bps / 10_000.0
            )
            buy_quote = torch.exp(torch.clamp(coordinates[:, 6], -30.0, 30.0))
            sell_quote = torch.exp(torch.clamp(coordinates[:, 7], -30.0, 30.0))
            buy_count = torch.exp(torch.clamp(coordinates[:, 8], -30.0, 20.0))
            sell_count = torch.exp(torch.clamp(coordinates[:, 9], -30.0, 20.0))
            side = torch.softmax(coordinates[:, 10:13], dim=1)
            spread = (
                torch.exp(torch.clamp(coordinates[:, 13], -30.0, 20.0))
                if union_output else None
            )
            spot_volume = (
                None if union_output
                else torch.exp(torch.clamp(coordinates[:, 13], -30.0, 30.0))
            )

            # Once the forecast extends past a rolling window, the outgoing
            # value is an earlier prediction from this same causal rollout,
            # not the observed future row stored in the source timeline.
            outgoing_square = torch.stack([
                context["outgoingSquare"][:, step, index]
                if step < window
                else predicted_returns[step - window].square()
                for index, window in enumerate(SQUARE_WINDOWS)
            ], dim=1)
            outgoing_absolute = torch.stack([
                context["outgoingAbsolute"][:, step, index]
                if step < window
                else predicted_returns[step - window].abs()
                for index, window in enumerate(ABSOLUTE_WINDOWS)
            ], dim=1)
            outgoing_active = torch.stack([
                context["outgoingActive"][:, step, index]
                if step < window
                else predicted_active[step - window]
                for index, window in enumerate(ACTIVE_WINDOWS)
            ], dim=1)
            square = square - outgoing_square + current_return.square()[:, None]
            absolute = absolute - outgoing_absolute + current_return.abs()[:, None]
            active_sums = active_sums - outgoing_active + active_probability[:, None]
            signed16 = (
                signed16
                - (
                    context["outgoingReturn16"][:, step]
                    if step < 16
                    else predicted_returns[step - 16]
                )
                + current_return
            )

            rms = {
                window: torch.sqrt(
                    torch.clamp_min(
                        square[:, square_index[window]] / float(window),
                        0.0,
                    )
                    + 1e-24
                )
                for window in (5, 15, 60)
            }
            rms60 = torch.clamp_min(rms[60], 1e-8)
            channels: list[Tensor] = [
                current_return,
                current_return / rms60,
                previous_return / rms60,
                1.0 - active_probability,
                1.0 - previous_active,
                active_sums[:, active_index[10]] / 10.0,
                active_sums[:, active_index[60]] / 60.0,
            ]
            zero_age = (zero_age + 1.0) * (1.0 - active_probability)
            channels.append(torch.log1p(torch.clamp_max(zero_age, 3_600.0)))

            anchor = 0.5 * torch.log(
                1e-16 + square[:, square_index[3_600]] / 3_600.0
            )
            channels.append(anchor)
            for window in (5, 15, 60, 300, 900, 1_800, 14_400):
                value = 0.5 * torch.log(
                    1e-16 + square[:, square_index[window]] / float(window)
                )
                channels.append(value - anchor)
            for window in (5, 15, 60, 300, 900, 3_600):
                channels.append(torch.log(
                    1e-12
                    + absolute[:, absolute_index[window]] / float(window)
                ))

            change = current_close - previous_close
            rsi_gain = 0.5 * rsi_gain + 0.5 * torch.clamp_min(change, 0.0)
            rsi_loss = 0.5 * rsi_loss + 0.5 * torch.clamp_min(-change, 0.0)
            channels.append(_safe_divide(
                rsi_gain - rsi_loss, rsi_gain + rsi_loss
            ))
            new_ema2 = (1.0 / 3.0) * ema2 + (2.0 / 3.0) * current_close
            level2 = torch.log(new_ema2) * 10_000.0
            previous_level2 = torch.log(ema2) * 10_000.0
            prior_level2 = torch.log(ema2_previous) * 10_000.0
            acceleration2 = (level2 - previous_level2) - (
                previous_level2 - prior_level2
            )
            channels.append(
                acceleration2 / torch.clamp_min(rms[5] * 10_000.0, 1e-4)
            )
            new_ema8 = (7.0 / 9.0) * ema8 + (2.0 / 9.0) * current_close
            old_ema8 = (
                context["ema8Outgoing"][:, step]
                if step < 8
                else predicted_ema8[step - 8]
            )
            slope8 = (
                torch.log(new_ema8) - torch.log(old_ema8)
            ) * (10_000.0 / 8.0)
            channels.append(
                slope8 / torch.clamp_min(rms[15] * 10_000.0, 1e-4)
            )

            next_second = second_index + step + 1
            if union_output:
                next_completed_hour_volume = current_features[:, 25]
            else:
                assert spot_volume is not None
                hour_volume_sum = hour_volume_sum + spot_volume
                hour_boundary_mask = torch.remainder(
                    next_second + 1, 3_600
                ) == 0
                completed_hour_volume = torch.where(
                    hour_boundary_mask,
                    torch.log1p(hour_volume_sum),
                    completed_hour_volume,
                )
                next_completed_hour_volume = completed_hour_volume
                hour_volume_sum = torch.where(
                    hour_boundary_mask,
                    torch.zeros_like(hour_volume_sum),
                    hour_volume_sum,
                )
            channels.extend((
                next_completed_hour_volume,
                torch.ones_like(current_return),
            ))
            hour_boundary = torch.div(
                next_second + 1, 3_600, rounding_mode="floor"
            ) * 3_600
            hour_boundary = torch.clamp_min(hour_boundary, 3_600)
            channels.append(torch.log1p(torch.clamp_min(
                (next_second + 1 - hour_boundary).to(current_return.dtype), 0.0
            )))
            channels.extend((
                torch.log1p(torch.log(high / low) / rms60),
                range_probability * _safe_divide(
                    2.0 * current_close - high - low, high - low
                ),
            ))
            square16 = torch.clamp_min(
                square[:, square_index[16]], 0.0
            )
            square16_root = torch.sqrt(square16 + 1e-24)
            channels.extend((
                (previous_return - current_return)
                / torch.clamp_min(np.sqrt(2.0) * square16_root, 1e-12),
                _safe_divide(
                    signed16,
                    absolute[:, absolute_index[16]],
                ),
                _safe_divide(signed16, square16_root),
            ))

            channels.extend(tuple(side.unbind(dim=1)))
            channels.extend(tuple(previous_side.unbind(dim=1)))
            channels.append(_safe_divide(
                buy_quote - sell_quote, buy_quote + sell_quote
            ))
            quote_buy_ema2 = (1.0 / 3.0) * quote_buy_ema2 + (2.0 / 3.0) * buy_quote
            quote_sell_ema2 = (1.0 / 3.0) * quote_sell_ema2 + (2.0 / 3.0) * sell_quote
            channels.append(_safe_divide(
                quote_buy_ema2 - quote_sell_ema2,
                quote_buy_ema2 + quote_sell_ema2,
            ))
            quote_buy_ema8 = (7.0 / 9.0) * quote_buy_ema8 + (2.0 / 9.0) * buy_quote
            quote_sell_ema8 = (7.0 / 9.0) * quote_sell_ema8 + (2.0 / 9.0) * sell_quote
            channels.append(_safe_divide(
                quote_buy_ema8 - quote_sell_ema8,
                quote_buy_ema8 + quote_sell_ema8,
            ))
            channels.append(_safe_divide(
                buy_count - sell_count, buy_count + sell_count
            ))

            if union_output:
                next_futures_trade_count = current_features[:, 43]
                next_futures_range = current_features[:, 44]
                next_minute_volatility_30 = current_features[:, 47]
                next_minute_volatility_60 = current_features[:, 48]
            else:
                minute_boundary_mask = torch.remainder(next_second + 1, 60) == 0
                futures_trade_count = torch.where(
                    minute_boundary_mask, coordinates[:, 14], futures_trade_count
                )
                futures_range = torch.where(
                    minute_boundary_mask, coordinates[:, 15], futures_range
                )
                btc_minute_return = torch.log(
                    current_close / previous_btc_minute_close
                )
                eth_minute_return = coordinates[:, 16]
                btc_minute_square = torch.where(
                    minute_boundary_mask[:, None],
                    btc_minute_square
                    - context["outgoingBtcMinuteSquare"][:, step]
                    + btc_minute_return.square()[:, None],
                    btc_minute_square,
                )
                eth_minute_square = torch.where(
                    minute_boundary_mask[:, None],
                    eth_minute_square
                    - context["outgoingEthMinuteSquare"][:, step]
                    + eth_minute_return.square()[:, None],
                    eth_minute_square,
                )
                minute_differences = []
                for minute_window_index, minute_window in enumerate((30, 60)):
                    btc_log_rms = 0.5 * torch.log(
                        1e-16
                        + torch.clamp_min(
                            btc_minute_square[:, minute_window_index], 0.0
                        ) / float(minute_window)
                    )
                    eth_log_rms = 0.5 * torch.log(
                        1e-16
                        + torch.clamp_min(
                            eth_minute_square[:, minute_window_index], 0.0
                        ) / float(minute_window)
                    )
                    minute_differences.append(eth_log_rms - btc_log_rms)
                minute_volatility_30 = torch.where(
                    minute_boundary_mask,
                    minute_differences[0],
                    minute_volatility_30,
                )
                minute_volatility_60 = torch.where(
                    minute_boundary_mask,
                    minute_differences[1],
                    minute_volatility_60,
                )
                previous_btc_minute_close = torch.where(
                    minute_boundary_mask,
                    current_close,
                    previous_btc_minute_close,
                )
                next_futures_trade_count = futures_trade_count
                next_futures_range = futures_range
                next_minute_volatility_30 = minute_volatility_30
                next_minute_volatility_60 = minute_volatility_60

            channels.extend((
                next_futures_trade_count,
                next_futures_range,
                torch.ones_like(current_return),
                torch.log1p(torch.remainder(
                    (next_second + 1).to(current_return.dtype), 60.0
                )),
                next_minute_volatility_30,
                next_minute_volatility_60,
                torch.ones_like(current_return),
                torch.log1p(torch.remainder(
                    (next_second + 1).to(current_return.dtype), 60.0
                )),
            ))

            origin_ms = self.second_start_ms + (next_second + 1) * 1_000
            seconds = torch.remainder(origin_ms // 1_000, 60).to(current_return.dtype)
            minutes = torch.remainder(origin_ms // 60_000, 60).to(current_return.dtype)
            hours = torch.remainder(origin_ms // 3_600_000, 24).to(current_return.dtype)
            days = torch.remainder(origin_ms // 86_400_000 + 4, 7).to(
                current_return.dtype
            )
            for value, period in ((seconds, 60), (minutes, 60), (hours, 24), (days, 7)):
                angle = 2.0 * np.pi * value / float(period)
                channels.extend((torch.sin(angle), torch.cos(angle)))
            if len(channels) != 59:
                raise AssertionError(f"base rollout emitted {len(channels)} channels")
            production = torch.stack(channels, dim=1)
            if union_output:
                assert spread is not None
                outputs.append(torch.cat((
                    production,
                    current_features[:, 59:529],
                    spread[:, None],
                ), dim=1))
            else:
                outputs.append(production)

            previous_return = current_return
            previous_active = active_probability
            previous_close = current_close
            previous_side = side
            ema2_previous = ema2
            ema2 = new_ema2
            ema8 = new_ema8
            predicted_returns.append(current_return)
            predicted_active.append(active_probability)
            predicted_ema8.append(new_ema8)

        return torch.stack(outputs, dim=1).to(output_dtype)


class StructuredUnion530BaseDataset:
    """K1=K2=2 union sequences with primitive future-output targets."""

    dataset_contract = DATASET_CONTRACT
    feature_count = 530
    base_coordinate_count = BASE_COORDINATE_COUNT
    input_steps = 2
    output_steps = 2

    def __init__(
        self,
        root: Path,
        *,
        train_examples: int,
        validation_examples: int = 65_536,
        test_examples: int = 65_536,
    ) -> None:
        self.root = root.resolve()
        self.base = Union530BaseHistoryDataset(self.root)
        self.rollout = Production59BaseRollout(self.base.second_start_ms)
        limits = {
            "train": int(train_examples),
            "validation": int(validation_examples),
            "test": int(test_examples),
        }
        origins = np.asarray(self.base.origins, dtype=np.int64)
        split_codes = np.asarray(self.base.split_codes, dtype=np.uint8)
        source_rows = np.asarray(self.base.minute_source_rows, dtype=np.int64)
        nonzero = np.asarray(self.base.nonzero, dtype=np.uint8) != 0
        self.current_rows: dict[str, np.ndarray] = {}
        for split, code in (("train", 0), ("validation", 1), ("test", 2)):
            physical = np.flatnonzero(split_codes == code)
            values = origins[physical]
            current = physical[1:-2]
            valid = (
                (values[1:-2] - values[:-3] == 1_000)
                & (values[2:-1] - values[1:-2] == 1_000)
                & (values[3:] - values[2:-1] == 1_000)
                & (source_rows[physical[:-3]] == source_rows[physical[3:]])
                & nonzero[current]
            )
            candidates = current[valid]
            if candidates.size < limits[split]:
                raise ValueError(
                    f"{split} has {candidates.size:,} exact union sequences, "
                    f"below requested {limits[split]:,}"
                )
            self.current_rows[split] = candidates[:limits[split]]
        self.counts = {
            split: int(rows.size) for split, rows in self.current_rows.items()
        }

        production_base = self.base.production_base(
            torch.device("cpu"), torch.float64
        )
        all_origins = torch.as_tensor(
            self.base.second_indices(np.arange(self.base.rows, dtype=np.int64)),
            dtype=torch.long,
        )
        self.production = production59_features(
            production_base, all_origins
        ).detach().float().numpy()
        global_base = self.base.global_base(torch.device("cpu"), torch.float64)
        self.global470 = reconstruct_global470(
            global_base, self.base.global_specs
        ).detach().float().numpy()

        self._initialize_causal_history(production_base)

    def _initialize_causal_history(self, production_base) -> None:
        candles = np.asarray(production_base.btc_second_candles, dtype=np.float64)
        flow = {
            name: np.asarray(values, dtype=np.float64)
            for name, values in production_base.btc_trade_flow.items()
        }
        close = candles[:, 3]
        self.returns = np.diff(np.log(close), prepend=np.log(close[:1]))
        self.active = (self.returns != 0).astype(np.float64)
        self.return_prefix = _prefix(self.returns)
        self.return_square_prefix = _prefix(self.returns * self.returns)
        self.return_absolute_prefix = _prefix(np.abs(self.returns))
        self.return_active_prefix = _prefix(self.active)
        self.volume_prefix = _prefix(candles[:, 4])
        changes = np.diff(close, prepend=close[:1])
        self.rsi_gain = _ema(np.maximum(changes, 0), 0.5, initial=0.0)
        self.rsi_loss = _ema(np.maximum(-changes, 0), 0.5, initial=0.0)
        self.ema2 = _ema(close, 2.0 / 3.0, initial=float(close[0]))
        self.ema8 = _ema(close, 2.0 / 9.0, initial=float(close[0]))
        buy_quote = flow["aggressiveBuyQuoteVolume"]
        sell_quote = flow["aggressiveSellQuoteVolume"]
        self.quote_buy_ema2 = _ema(
            buy_quote, 2.0 / 3.0, initial=float(buy_quote[0])
        )
        self.quote_sell_ema2 = _ema(
            sell_quote, 2.0 / 3.0, initial=float(sell_quote[0])
        )
        self.quote_buy_ema8 = _ema(
            buy_quote, 2.0 / 9.0, initial=float(buy_quote[0])
        )
        self.quote_sell_ema8 = _ema(
            sell_quote, 2.0 / 9.0, initial=float(sell_quote[0])
        )
        last_active = np.maximum.accumulate(np.where(
            self.active != 0, np.arange(self.active.size), -1,
        ))
        self.zero_age = np.arange(self.active.size) - last_active
        btc_minute = np.asarray(
            production_base.btc_minute_candles, dtype=np.float64
        )
        eth_minute = np.asarray(
            production_base.eth_minute_candles, dtype=np.float64
        )
        futures_minute = np.asarray(
            production_base.btc_futures_minute, dtype=np.float64
        )
        self.btc_minute_close = btc_minute[:, 3]
        self.eth_minute_return = np.diff(
            np.log(eth_minute[:, 3]), prepend=np.log(eth_minute[:1, 3])
        )
        self.btc_minute_return = np.diff(
            np.log(self.btc_minute_close),
            prepend=np.log(self.btc_minute_close[:1]),
        )
        self.btc_minute_square_prefix = _prefix(
            np.square(self.btc_minute_return)
        )
        self.eth_minute_square_prefix = _prefix(
            np.square(self.eth_minute_return)
        )
        self.futures_minute = futures_minute
        self.candles = candles
        self.flow = flow

    def close(self) -> None:
        self.base._numpy_cache.clear()

    def _feature_rows(self, physical: np.ndarray) -> np.ndarray:
        flat = np.asarray(physical, dtype=np.int64).reshape(-1)
        output = np.empty((flat.size, 530), dtype=np.float32)
        output[:, :59] = self.production[flat]
        source = np.asarray(self.base.minute_source_rows[flat], dtype=np.int64)
        output[:, 59:529] = self.global470[source]
        top = np.asarray(self.base.top_of_book[flat], dtype=np.float64)
        output[:, 529] = (
            10_000.0 * (top[:, 1] - top[:, 0])
            / ((top[:, 1] + top[:, 0]) / 2.0)
        ).astype(np.float32)
        return output.reshape(tuple(physical.shape) + (530,))

    def _base_coordinates(self, physical: np.ndarray) -> np.ndarray:
        physical = np.asarray(physical, dtype=np.int64)
        second = self.base.second_indices(physical.reshape(-1)).reshape(physical.shape)
        core = self._production_base_coordinates(second)
        top = np.asarray(self.base.top_of_book[physical], dtype=np.float64)
        spread = 10_000.0 * (top[..., 1] - top[..., 0]) / (
            (top[..., 1] + top[..., 0]) / 2.0
        )
        return np.concatenate((
            core,
            np.log(np.maximum(spread, EPSILON))[..., None].astype(np.float32),
        ), axis=-1)

    def _production_base_coordinates(self, second: np.ndarray) -> np.ndarray:
        second = np.asarray(second, dtype=np.int64)
        previous_close = self.candles[second - 1, 3]
        candle = self.candles[second]
        close = candle[..., 3]
        open_ = candle[..., 0]
        upper = 10_000.0 * np.log(
            candle[..., 1] / np.maximum(open_, close)
        )
        lower = 10_000.0 * np.log(
            np.minimum(open_, close) / candle[..., 2]
        )
        side_value = self.flow["lastAggressorSide"][second]
        side = np.stack((side_value < 0, side_value == 0, side_value > 0), axis=-1)
        side_logits = np.where(side, SIDE_LOGIT, -SIDE_LOGIT)
        result = np.stack((
            self.returns[second],
            np.where(self.active[second] != 0, SIDE_LOGIT, -SIDE_LOGIT),
            np.where(candle[..., 1] > candle[..., 2], SIDE_LOGIT, -SIDE_LOGIT),
            np.log(open_ / previous_close),
            np.log(np.maximum(upper, EPSILON)),
            np.log(np.maximum(lower, EPSILON)),
            np.log(np.maximum(
                self.flow["aggressiveBuyQuoteVolume"][second], EPSILON
            )),
            np.log(np.maximum(
                self.flow["aggressiveSellQuoteVolume"][second], EPSILON
            )),
            np.log(np.maximum(
                self.flow["aggressiveBuyAggregateTradeCount"][second], EPSILON
            )),
            np.log(np.maximum(
                self.flow["aggressiveSellAggregateTradeCount"][second], EPSILON
            )),
            side_logits[..., 0], side_logits[..., 1], side_logits[..., 2],
        ), axis=-1)
        return result.astype(np.float32)

    def _production59_base_coordinates(self, second: np.ndarray) -> np.ndarray:
        second = np.asarray(second, dtype=np.int64)
        core = self._production_base_coordinates(second)
        minute = second // 60
        futures = self.futures_minute[minute]
        additions = np.stack((
            np.log(np.maximum(self.candles[second, 4], EPSILON)),
            np.log1p(futures[..., 6]),
            10_000.0 * np.log(futures[..., 1] / futures[..., 2]),
            self.eth_minute_return[minute],
        ), axis=-1).astype(np.float32)
        return np.concatenate((core, additions), axis=-1)

    @staticmethod
    def _rolling(prefix: np.ndarray, origins: np.ndarray, windows: tuple[int, ...]):
        return np.stack([
            prefix[origins + 1] - prefix[origins + 1 - window]
            for window in windows
        ], axis=1)

    def _context(self, current_physical: np.ndarray) -> dict[str, Tensor]:
        return self._context_seconds(self.base.second_indices(current_physical))

    def _context_seconds(self, second: np.ndarray) -> dict[str, Tensor]:
        second = np.asarray(second, dtype=np.int64)
        output_steps = int(self.output_steps)
        outgoing_square = np.stack([
            np.stack([
                self.returns[second - window + 1 + step] ** 2
                for window in SQUARE_WINDOWS
            ], axis=1)
            for step in range(output_steps)
        ], axis=1)
        outgoing_absolute = np.stack([
            np.stack([
                np.abs(self.returns[second - window + 1 + step])
                for window in ABSOLUTE_WINDOWS
            ], axis=1)
            for step in range(output_steps)
        ], axis=1)
        outgoing_active = np.stack([
            np.stack([
                self.active[second - window + 1 + step]
                for window in ACTIVE_WINDOWS
            ], axis=1)
            for step in range(output_steps)
        ], axis=1)
        side_value = self.flow["lastAggressorSide"][second]
        previous_side = np.stack(
            (side_value < 0, side_value == 0, side_value > 0), axis=1
        ).astype(np.float32)
        completed_minute = (second + 1) // 60 - 1
        minute_windows = (30, 60)
        btc_minute_square_sums = self._rolling(
            self.btc_minute_square_prefix, completed_minute, minute_windows
        )
        eth_minute_square_sums = self._rolling(
            self.eth_minute_square_prefix, completed_minute, minute_windows
        )
        future_second = second[:, None] + np.arange(
            1, output_steps + 1, dtype=np.int64
        )[None, :]
        future_minute = future_second // 60
        outgoing_btc_minute_square = np.stack([
            np.stack([
                np.square(self.btc_minute_return[future_minute[:, step] - window])
                for window in minute_windows
            ], axis=1)
            for step in range(output_steps)
        ], axis=1)
        outgoing_eth_minute_square = np.stack([
            np.stack([
                np.square(self.eth_minute_return[future_minute[:, step] - window])
                for window in minute_windows
            ], axis=1)
            for step in range(output_steps)
        ], axis=1)
        hour_start = second - np.remainder(second, 3_600)
        values: dict[str, np.ndarray] = {
            "squareSums": self._rolling(
                self.return_square_prefix, second, SQUARE_WINDOWS
            ),
            "absoluteSums": self._rolling(
                self.return_absolute_prefix, second, ABSOLUTE_WINDOWS
            ),
            "activeSums": self._rolling(
                self.return_active_prefix, second, ACTIVE_WINDOWS
            ),
            "outgoingSquare": outgoing_square,
            "outgoingAbsolute": outgoing_absolute,
            "outgoingActive": outgoing_active,
            "signed16": self._rolling(
                self.return_prefix, second, (16,)
            )[:, 0],
            "outgoingReturn16": np.stack([
                self.returns[second - 15 + step]
                for step in range(output_steps)
            ], axis=1),
            "previousReturn": self.returns[second],
            "previousActive": self.active[second],
            "zeroAge": self.zero_age[second],
            "previousClose": self.candles[second, 3],
            "rsiGain": self.rsi_gain[second],
            "rsiLoss": self.rsi_loss[second],
            "ema2": self.ema2[second],
            "ema2Previous": self.ema2[second - 1],
            "ema8": self.ema8[second],
            "ema8Outgoing": np.stack([
                self.ema8[second - 7 + step]
                for step in range(output_steps)
            ], axis=1),
            "quoteBuyEma2": self.quote_buy_ema2[second],
            "quoteSellEma2": self.quote_sell_ema2[second],
            "quoteBuyEma8": self.quote_buy_ema8[second],
            "quoteSellEma8": self.quote_sell_ema8[second],
            "previousSide": previous_side,
            "secondIndex": second,
            "hourVolumeSum": (
                self.volume_prefix[second + 1] - self.volume_prefix[hour_start]
            ),
            "btcMinuteSquareSums": btc_minute_square_sums,
            "ethMinuteSquareSums": eth_minute_square_sums,
            "outgoingBtcMinuteSquare": outgoing_btc_minute_square,
            "outgoingEthMinuteSquare": outgoing_eth_minute_square,
            "previousBtcMinuteClose": self.btc_minute_close[completed_minute],
        }
        return {
            name: torch.from_numpy(np.asarray(value, dtype=(
                np.int64 if name == "secondIndex" else np.float64
            )).copy())
            for name, value in values.items()
        }

    def _examples(
        self, split: str, logical: np.ndarray
    ) -> StructuredUnion530Batch:
        current = self.current_rows[split][logical]
        inputs = self._feature_rows(np.stack((current - 1, current), axis=1))
        future = np.stack((current + 1, current + 2), axis=1)
        targets = self._feature_rows(future)
        context = self._context(current)
        context["baseTargets"] = torch.from_numpy(self._base_coordinates(future))
        return StructuredUnion530Batch(
            inputs=torch.from_numpy(inputs),
            targets=torch.from_numpy(targets),
            weights=torch.ones(current.size, dtype=torch.float32),
            context=context,
        )

    def iter_batches(
        self,
        split: str,
        batch_size: int,
        *,
        shuffle: bool,
        seed: int,
        limit: int | None = None,
        pad: bool = True,
    ) -> Iterator[StructuredUnion530Batch]:
        count = self.counts[split]
        if limit is not None:
            count = min(count, int(limit))
        order = np.arange(count, dtype=np.int64)
        if shuffle:
            np.random.default_rng(seed).shuffle(order)
        for start in range(0, count, int(batch_size)):
            selected = order[start:start + int(batch_size)]
            batch = self._examples(split, selected)
            if pad and selected.size < int(batch_size):
                missing = int(batch_size) - selected.size
                batch = StructuredUnion530Batch(
                    inputs=torch.cat((batch.inputs, torch.zeros(
                        missing, self.input_steps, self.feature_count,
                        dtype=batch.inputs.dtype
                    ))),
                    targets=torch.cat((batch.targets, torch.zeros(
                        missing, self.output_steps, self.feature_count,
                        dtype=batch.targets.dtype
                    ))),
                    weights=torch.cat((batch.weights, torch.zeros(missing))),
                    context={
                        name: torch.cat((value, torch.zeros(
                            (missing,) + value.shape[1:], dtype=value.dtype
                        )))
                        for name, value in batch.context.items()
                    },
                )
            yield batch

    def statistics(self, batch_size: int) -> dict[str, np.ndarray]:
        input_sum = np.zeros(self.feature_count, dtype=np.float64)
        input_square = np.zeros(self.feature_count, dtype=np.float64)
        target_sum = np.zeros(self.feature_count, dtype=np.float64)
        target_square = np.zeros(self.feature_count, dtype=np.float64)
        base_sum = np.zeros(self.base_coordinate_count, dtype=np.float64)
        base_square = np.zeros(self.base_coordinate_count, dtype=np.float64)
        input_count = target_count = base_count = 0
        for batch in self.iter_batches(
            "train", batch_size, shuffle=False, seed=0, pad=False,
        ):
            inputs = batch.inputs.numpy().astype(np.float64, copy=False)
            targets = batch.targets.numpy().astype(np.float64, copy=False)
            base = batch.context["baseTargets"].numpy().astype(np.float64, copy=False)
            input_sum += inputs.sum(axis=(0, 1))
            input_square += np.square(inputs).sum(axis=(0, 1))
            target_sum += targets.sum(axis=(0, 1))
            target_square += np.square(targets).sum(axis=(0, 1))
            base_sum += base.sum(axis=(0, 1))
            base_square += np.square(base).sum(axis=(0, 1))
            input_count += inputs.shape[0] * inputs.shape[1]
            target_count += targets.shape[0] * targets.shape[1]
            base_count += base.shape[0] * base.shape[1]

        def finish(total: np.ndarray, square: np.ndarray, count: int):
            mean = total / count
            variance = np.maximum(square / count - np.square(mean), 0.0)
            std = np.sqrt(variance)
            return (
                mean.astype(np.float32),
                np.where(std > 1e-8, std, 1.0).astype(np.float32),
            )

        input_mean, input_std = finish(input_sum, input_square, input_count)
        target_mean, target_std = finish(target_sum, target_square, target_count)
        base_mean, base_std = finish(base_sum, base_square, base_count)
        self.derived_output_mean = torch.from_numpy(target_mean.copy())
        self.derived_output_std = torch.from_numpy(target_std.copy())
        return {
            "inputMean": input_mean,
            "inputStd": input_std,
            "outputMean": base_mean,
            "outputStd": base_std,
            "derivedOutputMean": target_mean,
            "derivedOutputStd": target_std,
        }

    def derive(
        self,
        base_coordinates: Tensor,
        inputs: Tensor,
        context: Mapping[str, Tensor],
    ) -> Tensor:
        return self.rollout(base_coordinates, inputs[:, -1], context)


class StructuredProduction59BaseDataset(StructuredUnion530BaseDataset):
    """Original dense production59 population with causal primitive outputs."""

    dataset_contract = PRODUCTION59_DATASET_CONTRACT
    feature_count = 59
    base_coordinate_count = PRODUCTION_BASE_COORDINATE_COUNT
    input_steps = 2
    output_steps = 2

    def __init__(
        self,
        timeline_root: Path,
        base_history_root: Path,
        *,
        train_examples: int,
        input_steps: int = 2,
        output_steps: int = 2,
    ) -> None:
        if min(int(input_steps), int(output_steps)) <= 0:
            raise ValueError("production sequence lengths must be positive")
        self.input_steps = int(input_steps)
        self.output_steps = int(output_steps)
        self.root = timeline_root.resolve()
        self.base_history_root = base_history_root.resolve()
        self.base = Union530BaseHistoryDataset(self.base_history_root)
        self.rollout = Production59BaseRollout(self.base.second_start_ms)
        self.manifest = json.loads(
            (self.root / "manifest.json").read_text(encoding="utf-8")
        )
        if self.manifest.get("storageLayout") \
                != "temporal-channel-timeline-v1" \
                or int(self.manifest.get("temporalChannelCount", 0)) != 59:
            raise ValueError(
                "production59 base rollout requires the dense 59-channel timeline"
            )

        self.timeline: dict[str, np.memmap] = {}
        self.origins: dict[str, np.ndarray] = {}
        self.second_offsets: dict[str, int] = {}
        self.counts: dict[str, int] = {}
        files = self.manifest["files"]
        examples = self.manifest["examplesBySplit"]
        rows = self.manifest["timelineRowsBySplit"]
        for split in ("train", "validation", "test"):
            physical_count = int(examples[split])
            timeline_rows = int(rows[split])
            timeline = np.memmap(
                self.root / files[split]["timelineFeatures"],
                dtype="<f4",
                mode="r",
                shape=(timeline_rows, self.feature_count),
            )
            physical_origins = np.memmap(
                self.root / files[split]["origins"],
                dtype="<i4",
                mode="r",
                shape=(physical_count,),
            )
            times = np.memmap(
                self.root / files[split]["times"],
                dtype="<f8",
                mode="r",
                shape=(physical_count,),
            )
            all_origins = np.asarray(physical_origins, dtype=np.int64)
            constructible = all_origins[
                (all_origins >= self.input_steps - 1)
                & (all_origins + self.output_steps < timeline_rows)
            ]
            logical_count = min(constructible.size, int(train_examples)) \
                if split == "train" else int(constructible.size)
            selected = constructible[:logical_count]
            if selected.size != logical_count or (
                split == "train" and logical_count != int(train_examples)
            ) or not np.all(np.diff(selected) > 0):
                raise ValueError(
                    f"{split} cannot supply the requested production sequences"
                )
            timeline_start_ms = int(round(float(times[0]))) \
                - (int(all_origins[0]) + 1) * 1_000
            relative_start_ms = timeline_start_ms - self.base.second_start_ms
            if relative_start_ms % 1_000 != 0:
                raise ValueError("production timeline is not second aligned")
            second_offset = relative_start_ms // 1_000
            if second_offset < 0 \
                    or second_offset + timeline_rows > self.base.second_rows:
                raise ValueError("production timeline lies outside base history")
            self.timeline[split] = timeline
            self.origins[split] = selected
            self.second_offsets[split] = int(second_offset)
            self.counts[split] = int(logical_count)

        production_base = self.base.production_base(
            torch.device("cpu"), torch.float64
        )
        self._initialize_causal_history(production_base)

    def close(self) -> None:
        for values in self.timeline.values():
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.timeline.clear()
        self.base._numpy_cache.clear()

    def _examples(
        self, split: str, logical: np.ndarray
    ) -> StructuredUnion530Batch:
        origins = self.origins[split][logical]
        input_offsets = np.arange(-self.input_steps + 1, 1, dtype=np.int64)
        output_offsets = np.arange(1, self.output_steps + 1, dtype=np.int64)
        inputs = np.asarray(
            self.timeline[split][origins[:, None] + input_offsets[None, :]],
            dtype=np.float32,
        )
        targets = np.asarray(
            self.timeline[split][origins[:, None] + output_offsets[None, :]],
            dtype=np.float32,
        )
        current_second = self.second_offsets[split] + origins
        future_second = current_second[:, None] + output_offsets[None, :]
        context = self._context_seconds(current_second)
        context["baseTargets"] = torch.from_numpy(
            self._production59_base_coordinates(future_second)
        )
        return StructuredUnion530Batch(
            inputs=torch.from_numpy(inputs),
            targets=torch.from_numpy(targets),
            weights=torch.ones(origins.size, dtype=torch.float32),
            context=context,
        )

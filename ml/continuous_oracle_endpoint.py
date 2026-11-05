"""Continuous market trades against the full saved log-hurdle price law.

Scope: gamma=0, q,a>=0, unit funding, no orders, entry [0,1], maintenance
[0,1]. Under these assumptions all positive intermediate marks survive.
Terminal holdings must be exactly zero after a valid complete close.
Every positive-probability outcome must admit that close.
This is endpoint MPC, not a multi-period Bellman solution.

Every central PRICE TICK is integrated by CDF differences, including the zero
atom. Infinite tails are bounded using their mass and analytic first moments;
they are neither dropped nor replaced by conditional-mean scenarios. The
optimizer covers continuous signed quote trades, including the minimum-order
gap. Complete-close wealth is affine on each feasible interval, so endpoints
suffice. Its gap is numerical, not directed-rounding or a Lean certificate.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import expit, gammainc, gammaincc
from scipy.stats import gamma


@dataclass(frozen=True)
class Rules:
    fee: float = .0005
    minimum: float = 5.
    maximum: float = 10000.
    price_step: float = .01

    def __post_init__(self):
        if not (0 <= self.fee < 1 and 0 <= self.minimum <= self.maximum
                and self.maximum > 0 and self.price_step > 0):
            raise ValueError('invalid execution rules')


def terminal_close(quote, asset, price, rules):
    """Full physical close or explicit infeasibility; never mark leftover assets."""
    flow = (1-rules.fee)*asset*price
    if asset < 0 or not (asset == 0 or (flow > 0 and rules.minimum <= flow <= rules.maximum)):
        raise ValueError('terminal-exposure: full close violates order size limits')
    return quote+flow, 0., flow


def close_cost(asset, price, rules):
    terminal_close(0., asset, price, rules)
    return rules.fee*asset*price


@dataclass(frozen=True)
class Tail:
    mass: float
    first_moment: float  # unrounded price, NOT a representative execution price
    lower: float        # bounds on the quantized price
    upper: float
    moment_error: float = 0.  # absolute error in the unrounded first moment


class PriceLaw:
    def __init__(self, prices, weights, tails=(), price_step=.01, support_bounds=None):
        prices, weights = np.asarray(prices, dtype=float), np.asarray(weights, dtype=float)
        if (len(prices) != len(weights) or len(prices) == 0 or
                np.any(prices <= 0) or np.any(np.diff(prices) < 0) or
                np.any(weights < -1e-15) or not np.isfinite(prices).all() or
                not np.isfinite(weights).all()):
            raise ValueError('invalid price law')
        weights = np.maximum(weights, 0)
        if abs(weights.sum()+sum(t.mass for t in tails)-1) > 2e-12:
            raise ValueError('probability mass was lost')
        self.prices = prices
        self.mass = np.r_[0., np.cumsum(weights)]
        self.moment = np.r_[0., np.cumsum(weights*prices)]
        self.tails = tuple(tails)
        self.mean = self.moment[-1]+sum(t.first_moment for t in tails)
        # Saturating nearest-tick projection max(dp, round(P/dp)*dp).
        # It differs from raw P by at most dp, including P<dp/2.
        self.mean_error = price_step*sum(t.mass for t in tails)+sum(t.moment_error for t in tails)
        positive = prices[weights > 0]
        lows = positive.tolist()+[t.lower for t in tails if t.mass > 0]
        highs = positive.tolist()+[t.upper for t in tails if t.mass > 0]
        self.support_lower, self.support_upper = (min(lows), max(highs)) if support_bounds is None else support_bounds
        for array in (self.prices, self.mass, self.moment):
            array.flags.writeable = False

    def inventory_bounds(self, rules):
        # Positive holdings must satisfy minimum and maximum at every outcome.
        return (rules.minimum/((1-rules.fee)*self.support_lower),
                rules.maximum/((1-rules.fee)*self.support_upper))

    def can_close(self, asset, rules):
        lower, upper = self.inventory_bounds(rules)
        return asset == 0 or (asset > 0 and lower <= asset <= upper
                             and (1-rules.fee)*asset*self.support_lower > 0)

    def cost_bounds(self, asset, rules):
        if not self.can_close(asset, rules):
            raise ValueError('terminal-exposure: law admits an uncloseable outcome')
        scale = rules.fee*asset
        return scale*(self.mean-self.mean_error), scale*(self.mean+self.mean_error)

    def value_bounds(self, quote, asset, rules):
        if not self.can_close(asset, rules):
            return -math.inf, -math.inf
        value = quote+(1-rules.fee)*asset*self.mean
        error = (1-rules.fee)*asset*self.mean_error
        rounding = 2e-10*max(1., abs(value))
        return value-error-rounding, value+error+rounding


def log_hurdle_law(packed, price, price_step=.01, tail_probability=1e-11,
                   maximum_ticks=8_000_000):
    """Integrate the existing six-parameter forecast, without refitting it."""
    z, sign, up, down, au, ad = np.asarray(packed, dtype=float)
    if not np.isfinite(packed).all() or min(up, down, au, ad, price, price_step) <= 0:
        raise ValueError('invalid log-hurdle parameters')
    tu, td = up/au, down/ad
    if tu >= 1 or not 0 < tail_probability < .01:
        raise ValueError('finite price expectation and a small tail tolerance required')
    p0, pu = expit(z), expit(sign)
    wu, wd = (1-p0)*pu, (1-p0)*(1-pu)
    lower = price*np.exp(-gamma.isf(tail_probability, ad, scale=td))
    upper = price*np.exp(gamma.isf(tail_probability, au, scale=tu))
    first = max(1, int(np.floor(lower/price_step)))
    last = max(first, int(np.ceil(upper/price_step)))
    if last-first+1 > maximum_ticks:
        raise ValueError('price law exceeds integration budget; widen budget explicitly')
    ticks = np.arange(first, last+1, dtype=np.float64)
    boundaries = (np.arange(first, last+2, dtype=np.float64)-.5)*price_step
    x = np.log(boundaries/price)
    cdf = np.empty_like(x)
    negative = x < 0
    cdf[negative] = wd*gammaincc(ad, -x[negative]/td)
    cdf[~negative] = wd+p0+wu*gammainc(au, x[~negative]/tu)
    weights = np.diff(cdf)
    # Quantile range contains the observed price and hence the exact zero atom.
    if not boundaries[0] < price < boundaries[-1]:
        raise ValueError('central range must contain the zero atom')
    left_mass = float(cdf[0])
    right_mass = float(wu*gammaincc(au, x[-1]/tu))
    # E[exp(-M) 1(M>b)] and E[exp(M) 1(M>b)] for Gamma M.
    left_moment = price*wd*(1+td)**(-ad)*gammaincc(ad, -x[0]*(1/td+1))
    right_moment = price*wu*(1-tu)**(-au)*gammaincc(au, x[-1]*(1/tu-1))
    tails = (Tail(left_mass, float(left_moment), price_step, max(price_step, (first-1)*price_step)),
             Tail(right_mass, float(right_moment), (last+1)*price_step, math.inf))
    # Structural support must survive numerical tail underflow.
    bounds = (price_step if wd > 0 else price, math.inf if wu > 0 else price)
    return PriceLaw(ticks*price_step, weights, tails, price_step, support_bounds=bounds)


def capacities(quote, asset, price, rules):
    if min(quote, asset) < -1e-9 or price <= 0 or not np.isfinite([quote, asset, price]).all():
        raise ValueError('requires q,a>=0; leveraged multi-tick accounts need a path law')
    return min(max(0., quote), rules.maximum), min(max(0., (1-rules.fee)*asset*price), rules.maximum)


def execute(quote, asset, price, trade, rules):
    """Oracle signed actual quote flow: negative buys, positive sells, zero holds."""
    if trade == 0:
        return quote, asset
    b = 1-rules.fee
    if trade > 0 and trade == b*asset*price:
        return quote+trade, 0.  # exact complete-close request, not dust rounding
    return quote+trade, asset-trade*(b if trade < 0 else 1/b)/price


def continuous_decision(quote, asset, price, law, rules=Rules(), tolerance=1e-4):
    """Optimize affine complete-close wealth over the continuous feasible set.

    Intersect each buy/sell interval with inventory that can close at ALL prices
    in the law's support. Flat inventory is a separate feasible point. Unbounded
    upward support plus a finite terminal order cap permits only this flat point
    in this no-liquidation long-only endpoint model. No tail mass is ignored.
    """
    if tolerance <= 0:
        raise ValueError('positive wealth tolerance required')
    buy, sell = capacities(quote, asset, price, rules)
    b = 1-rules.fee
    lower, upper = law.inventory_bounds(rules)
    trades = {0.}
    full_close = b*asset*price
    if full_close == 0 or rules.minimum <= full_close <= rules.maximum:
        trades.add(full_close)
    if 0 <= lower <= upper and upper > 0:
        for side, cap in ((-1, buy), (1, sell)):
            if cap <= 0 or cap < rules.minimum:
                continue
            left, right = sorted((side*rules.minimum, side*cap))
            slope = (b if side < 0 else 1/b)/price
            left, right = max(left, (asset-upper)/slope), min(right, (asset-lower)/slope)
            if left <= right:
                # Nudge inward by one ulp where inversion rounded outside the
                # supported inventory interval. Never accept an illegal residual.
                for trade, toward in ((left, right), (right, left)):
                    for _ in range(4):
                        q, a = execute(quote, asset, price, trade, rules)
                        if law.can_close(a, rules):
                            trades.add(trade)
                            break
                        trade = float(np.nextafter(trade, toward))
                    else:
                        raise RuntimeError('floating-point resolution cannot resolve a feasible interval endpoint')
    evaluated = []
    for trade in sorted(trades):
        q, a = execute(quote, asset, price, trade, rules)
        lo, hi = law.value_bounds(q, a, rules)
        if math.isfinite(lo):
            evaluated.append((lo, hi, trade, q, a))
    if not evaluated:
        raise ValueError('No feasible flat-terminal continuation in the full price law')
    lo, hi, trade, q, a = max(evaluated, key=lambda x: (x[0], -abs(x[2])))
    global_upper = max(x[1] for x in evaluated)
    gap = global_upper-lo
    if gap > tolerance:
        raise RuntimeError(f'price-moment or rounding uncertainty exceeds requested value gap: {gap}')
    return {'marketQuote': trade, 'quote': q, 'asset': a,
            'valueLower': lo, 'valueUpper': hi, 'globalUpper': global_upper,
            'optimalityGap': gap, 'evaluations': len(evaluated),
            'terminalRequirement': 'zero-asset-and-no-orders', 'valueTolerance': tolerance}

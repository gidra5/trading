"""Complete Gaussian-coordinate knot laws projected to Oracle price ticks."""
from __future__ import annotations
import math
import numpy as np
from scipy.special import ndtr, ndtri, ndtri_exp, expit, gammainc, gammaincc
from continuous_oracle_endpoint import PriceLaw, Tail, log_hurdle_law


def knot_cdf(knots, heights, units):
    widths = np.diff(knots)
    prefix = np.r_[0., np.cumsum((heights[:-1]+heights[1:])*widths/2)]
    i = np.clip(np.searchsorted(knots, units, side='right')-1, 0, len(knots)-2)
    d = units-knots[i]
    values = prefix[i]+heights[i]*d+(heights[i+1]-heights[i])*d*d/(2*widths[i])
    return np.clip(values, 0, 1)


def knot_price_law(zero, masses, knots, areas, sigma, price, price_step=.01, radius=8., maximum_ticks=200000):
    """Exact central CDF cell masses and bounded first moments in BOTH tails.

    In Gaussian units z the active density is h(Phi(z))*phi(z). Therefore
    its exponential tail moment is bounded by max(h) times the shifted normal
    tail moment. No tail probability is discarded or renormalized away.
    """
    if not (0 <= zero <= 1 and sigma > 0 and price > 0 and radius >= 6):
        raise ValueError('invalid knot price law')
    masses = np.asarray(masses, dtype=float)
    if np.any(masses<0) or abs(masses.sum()-1)>2e-12:
        raise ValueError('invalid knot masses')
    heights = masses/areas
    # The learned probability-coordinate density can be enormous on a tiny
    # endpoint hat. A fixed normal-sigma range does NOT bound its tail mass.
    # Use the density envelope to bound omitted mass and price first moments.
    log_tail = np.log(1e-12)-np.log(max(float(heights.max()),1.))-np.log(max(1-zero,1e-300))
    normal_cut = float(ndtri_exp(min(log_tail,np.log(1e-12))))
    lower_z, upper_z = min(-radius,normal_cut), max(radius,sigma-normal_cut)
    first = max(1, int(np.floor(price*np.exp(lower_z*sigma)/price_step)))
    last = int(np.ceil(price*np.exp(upper_z*sigma)/price_step))
    if last-first+1 > maximum_ticks:
        raise ValueError('central price-grid budget exceeded')
    boundaries = (np.arange(first, last+2)-.5)*price_step
    gaussian = np.log(boundaries/price)/sigma
    units = ndtr(gaussian)
    active_cdf = knot_cdf(knots, heights, units)
    cdf = (1-zero)*active_cdf+zero*(boundaries>=price)
    weights = np.diff(cdf)
    # Tiny negative differences can arise through roundoff in saturated tails.
    if weights.min() < -2e-14:
        raise ValueError('nonmonotone knot CDF')
    left_mass, right_mass = float(cdf[0]), float(1-cdf[-1])
    # Piecewise linear h attains its maximum at a knot or the tail boundary.
    left_height = max(float(np.interp(units[0], knots, heights)), float(heights[knots<=units[0]].max()))
    right_height = max(float(np.interp(units[-1], knots, heights)), float(heights[knots>=units[-1]].max()))
    factor = price*(1-zero)*np.exp(sigma*sigma/2)
    left_upper = float(factor*left_height*ndtr(gaussian[0]-sigma))
    right_upper = float(factor*right_height*ndtr(sigma-gaussian[-1]))
    # CDF subtraction rounding must not make the lower support moment exceed
    # the analytic upper bound. Widen upwards to cover that numerical residue.
    left_upper = max(left_upper, left_mass*price_step)
    right_upper = max(right_upper, right_mass*boundaries[-1])
    tails = (Tail(left_mass, left_upper/2, price_step, max(price_step,(first-1)*price_step), left_upper/2),
        Tail(right_mass, right_upper/2, (last+1)*price_step, math.inf, right_upper/2))
    # Use structural hat support, including tails too small for CDF subtraction.
    active = np.flatnonzero(heights > 0)
    left_unit = knots[max(0, int(active[0])-1)]
    right_unit = knots[min(len(knots)-1, int(active[-1])+1)]
    def bound(u):
        raw = price*np.exp(sigma*ndtri(u))
        return math.inf if math.isinf(raw) else max(price_step, math.floor(raw/price_step+.5)*price_step)
    bounds = (price, price) if zero == 1 else (bound(left_unit), bound(right_unit))
    if zero > 0:
        bounds = (min(bounds[0], price), max(bounds[1], price))
    return PriceLaw(np.arange(first,last+1)*price_step, weights, tails, price_step, support_bounds=bounds)


def mixture_law(laws, weights, price_step=.01):
    weights = np.asarray(weights, dtype=float)
    if np.any(weights<0) or abs(weights.sum()-1)>2e-12:
        raise ValueError('mixture weights must sum to one')
    return MixturePriceLaw(tuple(laws),weights)


class MixturePriceLaw(PriceLaw):
    """Expectation linearity avoids merging identical price lattices."""
    def __init__(self,laws,weights):
        self.laws,self.weights=laws,weights
        self.mean=float(sum(w*law.mean for law,w in zip(laws,weights)))
        self.mean_error=float(sum(w*law.mean_error for law,w in zip(laws,weights)))

        self.support_lower = min(law.support_lower for law,w in zip(laws,weights) if w > 0)
        self.support_upper = max(law.support_upper for law,w in zip(laws,weights) if w > 0)


class AnalyticGammaPriceLaw(PriceLaw):
    """Gamma price moments with explicit unbounded structural support."""
    def __init__(self, packed, price, price_step=.01):
        central = log_hurdle_law(packed, price, price_step, tail_probability=.001)
        self.mean, self.mean_error = central.mean, central.mean_error
        self.support_lower, self.support_upper = central.support_lower, central.support_upper

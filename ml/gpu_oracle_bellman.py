"""Batched finite-tree Bellman search, gamma=0, float64 Oracle execution.

This is a numerical counterpart, not a replacement for the rational certificate
checker. Supports one pending limit order (the experiment's configured bound).
Actions are nonanticipative: maximize AFTER averaging each action's children.
Chunking bounds memory, not the exponential amount of work.
"""
from dataclasses import dataclass
from decimal import Decimal
import math
import torch

# State: quote, asset, current price, previous price, order side (+buy/-sell/0),
# order price, notional, placement time, current time. All tensors are private.
Q, A, P, PREV, SIDE, LP, N, PLACED, TIME = range(9)


@dataclass(frozen=True)
class Rules:
    maker: float = .0002
    taker: float = .0005
    reserve_leverage: float = 2.
    lower: float = -2.
    upper: float = 2.
    maintenance_lower: float = -3.
    maintenance_upper: float = 3.
    minimum: float = 5.
    maximum: float = 10000.
    tick: float = .01


@dataclass(frozen=True)
class Menu:
    step: float = .01
    limits: bool = True
    limit_fractions: tuple[float, ...] = (.25,)
    limit_offsets: tuple[int, ...] = (1,)
    joint: bool = False


def equity(s):
    return s[..., Q] + s[..., A] * s[..., P]


def trade(s, amount, fee, price=None):
    out = s.clone()
    p = s[..., P] if price is None else price
    out[..., Q] += amount
    out[..., A] -= torch.where(amount <= 0, (1-fee)*amount/p, amount/((1-fee)*p))
    return out


class TensorOracle:
    def __init__(self, rules=Rules()):
        self.r = rules
        r = rules
        if not (0 <= r.maker < r.taker < 1 or r.maker == r.taker == 0):
            raise ValueError('Invalid fees')
        if not (r.reserve_leverage > 0 and r.tick > 0 and 0 <= r.minimum <= r.maximum
                and r.lower <= 0 <= r.upper and r.maintenance_lower <= 0 and r.maintenance_upper >= 1):
            raise ValueError('Invalid Oracle rules')

    # Only absorb floating-point roundoff near exact constraint boundaries.
    # This tolerance is NOT a formal feasibility or value-error certificate.
    def le(self, a, b):
        bound = torch.as_tensor(b, device=a.device, dtype=a.dtype)
        return a <= bound + 1e-11 * (1 + bound.abs())

    def size(self, x):
        return (x == 0) | ((x.abs() >= self.r.minimum) & self.le(x.abs(), self.r.maximum))

    def effective(self, s):
        v = s[..., A]*s[..., P]
        return torch.where(v >= 0, (1-self.r.taker)*v, v/(1-self.r.taker))

    def advance(self, s, price, funding=1.):
        out = s.clone()
        out[..., PREV] = s[..., P]
        out[..., P] = price
        out[..., A] *= funding
        out[..., TIME] += 1
        eff = self.effective(out)
        w = out[..., Q] + eff
        e = eff/torch.where(w != 0, w, torch.ones_like(w))
        survives = (equity(out) > 0) & (w > 0) & self.le(-e, -self.r.maintenance_lower) & self.le(e, self.r.maintenance_upper)
        for col in (Q, A, SIDE, LP, N, PLACED):
            out[..., col] = torch.where(survives, out[..., col], 0.)
        return out

    def arrival(self, s):
        r = self.r
        # These are grid prices: compare their integer tick coordinates. Binary
        # subtraction p - .01 need not equal the float for the neighboring tick.
        p, previous, limit = ((s[..., col]/r.tick).round() for col in (P, PREV, LP))
        buy = (s[..., SIDE] > 0) & (p < previous) & (p <= limit) & (limit <= previous)
        sell = (s[..., SIDE] < 0) & (previous < p) & (previous <= limit) & (limit <= p)
        fill = (buy | sell) & (s[..., PLACED] < s[..., TIME])
        amount = torch.where(fill, torch.where(buy, -s[..., N], (1-r.maker)*s[..., N]), 0.)
        out = trade(s, amount, r.maker, torch.where(fill, s[..., LP], s[..., P]))
        for col in (SIDE, LP, N, PLACED):
            out[..., col] = torch.where(fill, 0., out[..., col])
        return out, amount.abs(), fill, self.size(amount)

    def terminal(self, s):
        s, consumed, _, valid = self.arrival(s)
        eff = self.effective(s)
        # Full closing flow only. A capped/dust residual is an infeasible leaf,
        # so the recursion must arrange earlier reductions on every branch.
        valid &= self.size(eff) & self.le(consumed+eff.abs(), self.r.maximum)
        valid &= (s[..., A] == 0) | (eff != 0)  # underflow must not erase inventory
        # Executing exactly the full effective flow makes inventory zero.
        # Evaluate cash directly to avoid manufacturing floating-point dust.
        w = s[..., Q]+eff
        return torch.where(valid & (w >= 0), w, -torch.inf)

    def descriptors(self, menu, device):
        r = self.r
        if not math.isfinite(menu.step) or menu.step <= 0:
            raise ValueError('Positive finite exposure step required')
        count = round((r.upper-r.lower)/menu.step)
        if menu.step <= 0 or count < 1 or not math.isclose(count*menu.step, r.upper-r.lower):
            raise ValueError('Exposure step must divide the entry interval')
        if any(x <= 0 for x in menu.limit_fractions) or any(x <= 0 or int(x) != x for x in menu.limit_offsets):
            raise ValueError('Invalid limit grid')
        targets = [float('nan')] + [float(Decimal(str(r.lower))+i*Decimal(str(menu.step))) for i in range(count+1)]
        # target NaN means zero trade, not target zero exposure.
        placements = [(0., 0., 0.)]
        if menu.limits:
            placements += [(side, float(offset), fraction) for side in (1., -1.)
                           for offset in menu.limit_offsets for fraction in menu.limit_fractions]
        rows = [(target, float(cancel), side, offset, fraction)
                for cancel in (False, True) for side, offset, fraction in placements
                for target in targets if menu.joint or side == 0 or math.isnan(target)]
        return torch.tensor(rows, dtype=torch.float64, device=device)

    def actions(self, s, descriptors):
        """Return [accounts, actions, state], feasibility, actual quote movements."""
        r = self.r
        arrived, consumed, filled, arrival_ok = self.arrival(s)
        x = arrived[:, None, :].expand(-1, len(descriptors), -1)
        target, cancel, side, offset, fraction = descriptors.unbind(-1)
        w = equity(x)
        v = x[..., A]*x[..., P]
        e = v/torch.where(w != 0, w, 1.)
        t = torch.nan_to_num(target)[None, :]
        den = torch.where(e < t, 1-r.taker+r.taker*t, 1-r.taker*t)
        request = torch.where(e < t, -w*(t-e)/den, (1-r.taker)*w*(e-t)/den)
        available = (r.maximum-consumed)[:, None]
        amount = request.sign()*torch.minimum(request.abs(), available)
        amount = torch.where(torch.isnan(target)[None, :] | (w <= 0), 0., amount)
        pending = (x[..., SIDE] != 0) & (cancel[None, :] == 0)
        reserve = torch.where(pending, x[..., N]/r.reserve_leverage, 0.)
        valid = arrival_ok[:, None] & self.size(amount) & self.le(consumed[:, None]+amount.abs(), r.maximum)
        # Canceling an empty book duplicates an already included physical action.
        valid &= (cancel[None, :] == 0) | (x[..., SIDE] != 0)
        # Capped targets often execute exactly the same quote amount. Remove
        # consecutive duplicates within each cancel/place group, and requests
        # that duplicate its explicit zero-trade action. No value approximation.
        valid &= torch.isnan(target)[None, :] | (amount != 0)
        same_group = (descriptors[1:, 1:] == descriptors[:-1, 1:]).all(1)
        finite_pair = torch.isfinite(target[1:]) & torch.isfinite(target[:-1])
        valid[:, 1:] &= ~((amount[:, 1:] == amount[:, :-1]) & (den[:, :-1] > 0) & (same_group & finite_pair)[None, :])
        valid &= self.le(reserve, w) & ((den > 0) | torch.isnan(target)[None, :])
        out = trade(x, amount, r.taker)
        # An uncapped target-zero execution is algebraically a full close.
        # Do not manufacture floating-point dust by subtracting nearly equal
        # asset quantities; genuine undersized or capped residuals remain.
        full_close = (target[None, :] == 0) & (request.abs() <= available) & self.size(request)
        out[..., A] = torch.where(full_close, 0., out[..., A])
        new = side[None, :] != 0
        valid &= ~(pending & new)
        order_price = ((x[..., P]/r.tick).round()-side[None, :]*offset[None, :])*r.tick
        notional = torch.minimum(w*fraction[None, :], torch.where(side[None, :] > 0, r.maximum, r.maximum/(1-r.maker)))
        flow = torch.where(side[None, :] > 0, -notional, (1-r.maker)*notional)
        valid &= ~new | ((order_price > 0) & (notional > 0) & self.size(flow))
        for col, value in ((SIDE, side[None, :]), (LP, order_price), (N, notional), (PLACED, x[..., TIME])):
            out[..., col] = torch.where(new, value, torch.where(pending, x[..., col], 0.))
        reserve = torch.where(out[..., SIDE] != 0, out[..., N]/r.reserve_leverage, 0.)
        post_w = equity(out)
        valid &= self.le(reserve, post_w)
        valid &= (out[..., SIDE] == 0) | self.le((out[..., A]*out[..., P]).abs()/r.reserve_leverage+reserve, post_w)
        valid &= (post_w > 0) | ((out[..., Q] == 0) & (out[..., A] == 0))
        post_e = out[..., A]*out[..., P]/torch.where(post_w != 0, post_w, 1.)
        physical = (amount != 0) | filled[:, None]
        valid &= ~physical | (self.le(-post_e, -r.lower) & self.le(post_e, r.upper))
        return out, valid, amount


class Bellman:
    """Regular observable trees, batched account search with bounded working sets.

    prices[d] and weights[d] have [market nodes at d, children] shape. Distinct
    nodes MUST correspond to distinct observable histories, not hidden IDs.
    Every level has a positive normalized law. Funding is a known per-level
    factor. State.time counts modeled transitions, matching the rational solver.
    """
    def __init__(self, prices, weights, menus, rules=Rules(), chunk=32768, funding=None, max_work=200_000_000, destinations=None):
        self.prices, self.weights, self.menus = prices, weights, menus
        self.oracle = TensorOracle(rules)
        self.chunk = chunk
        self.work = 0
        self.max_work = max_work
        self.destinations = destinations
        self.funding = [1.]*len(prices) if funding is None else funding
        if not prices or len(prices) != len(weights) or len(menus) != len(prices) or len(self.funding) != len(prices) or chunk < 1:
            raise ValueError('Invalid tree or chunk')
        if destinations is not None and len(destinations) != len(prices):
            raise ValueError('A graph needs destination indices for every transition')
        nodes = 1
        for depth, (p, w, f) in enumerate(zip(prices, weights, self.funding)):
            if p.ndim != 2 or p.shape[0] != nodes or p.shape != w.shape or p.shape[1] < 1:
                raise ValueError('Nonregular tree')
            if not bool(torch.isfinite(p).all() & (p > 0).all() & torch.isfinite(w).all() & (w > 0).all()):
                raise ValueError('Invalid law')
            if not torch.allclose(w.sum(1), torch.ones_like(w[:, 0]), atol=1e-12, rtol=0):
                raise ValueError('Law not normalized')
            if not torch.allclose(p/rules.tick, (p/rules.tick).round(), atol=1e-7, rtol=0) or not math.isfinite(f) or f <= 0:
                raise ValueError('Invalid price grid or funding')
            if destinations is None:
                nodes *= p.shape[1]
            else:
                dest = destinations[depth]
                if dest.shape != p.shape or dest.dtype != torch.long or not bool((dest >= 0).all()):
                    raise ValueError('Invalid graph destinations')
                nodes = int(dest.max().item())+1
                if not torch.equal(torch.unique(dest), torch.arange(nodes, device=dest.device)):
                    raise ValueError('Graph destination indices must be dense')
        self.descriptors = [self.oracle.descriptors(m, prices[0].device) for m in menus]

    @torch.inference_mode()
    def solve(self, state):
        if state.ndim != 2 or state.shape[1] != 9 or not len(state) or state.dtype != torch.float64:
            raise ValueError('Require a nonempty float64 account batch with nine coordinates')
        if not bool(torch.isfinite(state).all()) or not bool((state[:, P] > 0).all() & (state[:, PREV] > 0).all()):
            raise ValueError('Invalid account domain')
        self.work = 0
        value, choice = self._value(state, torch.zeros(len(state), dtype=torch.long, device=state.device), 0)
        executed, valid, amount = self.oracle.actions(state, self.descriptors[0])
        rows = torch.arange(len(state), device=state.device)
        if not bool(torch.isfinite(value).all() & valid[rows, choice].all()):
            raise ValueError('No feasible continuation in numerical menu')
        return dict(value=value, choice=choice, state=executed[rows, choice], amount=amount[rows, choice],
                    descriptor=self.descriptors[0][choice], work=self.work)

    def _value(self, state, nodes, depth):
        if depth == len(self.prices):
            self.work += len(state)
            if self.work > self.max_work:
                raise RuntimeError('Bellman work budget exceeded; no partial decision returned')
            return self.oracle.terminal(state), None
        desc = self.descriptors[depth]
        batch_size = max(1, self.chunk//len(desc))
        values, choices = [], []
        for start in range(0, len(state), batch_size):
            s, n = state[start:start+batch_size], nodes[start:start+batch_size]
            out, valid, _ = self.oracle.actions(s, desc)
            row, action = valid.nonzero(as_tuple=True)
            scores = torch.full(valid.shape, -torch.inf, device=s.device, dtype=torch.float64)
            if len(row):
                candidate = out[row, action]
                market = n[row]
                branches = self.prices[depth].shape[1]
                expanded = candidate[:, None, :].expand(-1, branches, -1)
                moved = self.oracle.advance(expanded, self.prices[depth][market], self.funding[depth])
                child_nodes = (market[:, None]*branches+torch.arange(branches, device=s.device)[None, :]
                               if self.destinations is None else self.destinations[depth][market])
                continuation, _ = self._value(moved.reshape(-1, 9), child_nodes.reshape(-1), depth+1)
                expected = (self.weights[depth][market]*continuation.reshape(-1, branches)).sum(1)
                scores[row, action] = expected
            best, choice = scores.max(dim=1)
            values.append(best); choices.append(choice)
        return torch.cat(values), torch.cat(choices)

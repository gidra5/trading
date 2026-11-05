"""Differentiable restricted Oracle execution for realized-wealth training.

Unleveraged long/cash, positive marks, unit signed-asset funding, no orders.
All such marks survive maintenance [0,1]. Actual quote minima/caps and the
valid capped final close remain active. No straight-through estimator is used
at minimum-size boundaries. This is a learned controller, not a Bellman solver.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from scipy.special import expit
from continuous_oracle_endpoint import Rules, execute


@dataclass(frozen=True)
class TrainingRules:
    fee: float = .0005
    minimum: float = 5.
    maximum: float = 10000.
    initial: float = 10000.


class DistributionController(nn.Module):
    """Full atom/knot masses plus current account and remaining horizon.

Initial target ~4.7%, identically for every experimental arm. This is an
initialization, not an activity floor. Cash is a separate selection baseline.
"""
    def __init__(self, components):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(components+1, 16), nn.Tanh())
        self.output = nn.Linear(19, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.constant_(self.output.bias, -3.)

    def coefficients(self, zero, masses):
        embedding = self.encoder(torch.cat((zero[..., None], masses), dim=-1))
        w = self.output.weight[0].double()
        static = embedding.double()@w[:16]+self.output.bias[0].double()
        return static, w[16:]


def quote_request(q, a, price, target, rules):
    wealth = q+a*price
    current = a*price/wealth
    requested = torch.where(target > current,
        -wealth*(target-current)/(1-rules.fee+rules.fee*target),
        (1-rules.fee)*wealth*(current-target)/(1-rules.fee*target))
    buy_cap = torch.minimum(q, torch.full_like(q, rules.maximum))
    sell_cap = torch.minimum((1-rules.fee)*a*price, torch.full_like(q, rules.maximum))
    requested = torch.maximum(-buy_cap, torch.minimum(requested, sell_cap))
    return torch.where(abs(requested) >= rules.minimum, requested, torch.zeros_like(requested))


def execute_torch(q, a, price, dq, rules):
    asset_change = torch.where(dq <= 0, (1-rules.fee)*dq/price, dq/((1-rules.fee)*price))
    return q+dq, a-asset_change


def rollout_torch(static, state_weights, prices, rules=TrainingRules(), trace=False):
    """Batch of complete episodes: N issue prices plus one final close mark.

Future prices enter only after actions are selected at their current marks.
No close occurs between observations. Exposure/wealth state is not detached.
"""
    batch, length = static.shape
    if prices.shape != (batch, length+1):
        raise ValueError('one terminal price is required after all issue prices')
    q = torch.full((batch,), rules.initial, dtype=torch.float64, device=static.device)
    a = torch.zeros_like(q)
    trades = []
    turnover = torch.zeros_like(q)
    for t in range(length):
        p = prices[:, t]
        wealth = q+a*p
        state = torch.stack((a*p/wealth, wealth/rules.initial-1,
                             torch.full_like(q, (length-t)/length)), dim=1)
        target = torch.sigmoid(static[:, t]+state@state_weights)
        dq = quote_request(q, a, p, target, rules)
        q, a = execute_torch(q, a, p, dq, rules)
        turnover = turnover+abs(dq)
        if trace:
            trades.append(dq)
    p = prices[:, -1]
    close = torch.minimum((1-rules.fee)*a*p, torch.full_like(q, rules.maximum))
    close = torch.where(close >= rules.minimum, close, torch.zeros_like(close))
    q, a = execute_torch(q, a, p, close, rules)
    result = {'wealth': q+a*p, 'quote': q, 'asset': a, 'close': close,
              'turnover': turnover+close}
    if trace:
        result['trades'] = torch.stack(trades, dim=1)
    return result


def rollout_numpy(static, state_weights, prices, rules=TrainingRules(), trace=False):
    """Independent scalar replay using the existing Oracle execute function."""
    if len(prices) != len(static)+1 or np.any(prices <= 0):
        raise ValueError('invalid price tape')
    q, a, peak = rules.initial, 0., rules.initial
    turnover = fees = drawdown = exposure_sum = 0.
    count = active = 0
    records = []
    raw_rules = Rules(fee=rules.fee, minimum=rules.minimum, maximum=rules.maximum)
    for t, base in enumerate(static):
        p = prices[t]
        wealth = q+a*p
        peak = max(peak, wealth); drawdown = max(drawdown, 1-wealth/peak)
        current = a*p/wealth
        state = np.array([current, wealth/rules.initial-1, (len(static)-t)/len(static)])
        target = expit(base+state@state_weights)
        request = (-wealth*(target-current)/(1-rules.fee+rules.fee*target) if target > current else
                   (1-rules.fee)*wealth*(current-target)/(1-rules.fee*target))
        dq = float(np.clip(request, -min(q, rules.maximum), min((1-rules.fee)*a*p, rules.maximum)))
        if abs(dq) < rules.minimum:
            dq = 0.
        q, a = execute(q, a, p, dq, raw_rules)
        if min(q, a) < -1e-8:
            raise ValueError('account left the unleveraged domain')
        turnover += abs(dq); count += dq != 0
        fees += rules.fee*abs(dq)*(1 if dq <= 0 else 1/(1-rules.fee))
        wealth = q+a*p
        drawdown = max(drawdown, 1-wealth/peak)
        exposure = a*p/wealth; exposure_sum += exposure; active += exposure > 1e-8
        if trace:
            records.append([dq, q, a, exposure])
    p = prices[-1]
    wealth = q+a*p; peak = max(peak, wealth); drawdown = max(drawdown, 1-wealth/peak)
    close = min((1-rules.fee)*a*p, rules.maximum)
    if close < rules.minimum:
        close = 0.
    q, a = execute(q, a, p, close, raw_rules)
    turnover += close; count += close != 0; fees += rules.fee/(1-rules.fee)*close
    wealth = q+a*p; drawdown = max(drawdown, 1-wealth/peak)
    result = {'wealth': float(wealth), 'returnPercent': float(100*(wealth/rules.initial-1)),
        'quote': float(q), 'asset': float(a), 'close': float(close),
        'turnover': float(turnover), 'fees': float(fees), 'trades': int(count),
        'maximumDecisionMarkDrawdownPercent': float(100*drawdown),
        'averageExposure': float(exposure_sum/len(static)), 'activeDecisions': int(active)}
    if trace:
        result['trace'] = np.asarray(records)
    return result

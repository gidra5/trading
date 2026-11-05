"""Amortized finite-horizon log Bellman values for original Oracle accounts.

The network approximates continuation, not execution or the market kernel.
Each backup still averages each action's one-second outcomes before choosing.
No residual measured on sampled states is called a uniform certificate.
"""
from dataclasses import asdict
import math
import numpy as np
import torch
from torch import nn
from gpu_oracle_bellman import TensorOracle, Rules, Menu, equity, Q, A, P, PREV, SIDE, LP, N, PLACED, TIME
from latent_neural_stochastic_flow import clock_features

CONTRACT = 'oracle-log-continuation-one-second-v1'


def market_coordinates(features, prices, times, mean, std):
    """Two completed observable rows; no hidden latent sample or scenario ID."""
    if features.shape[-2:] != (2, 59) or prices.shape[-1] != 2 or times.shape[-1] != 2:
        raise ValueError('Require two complete market observations')
    f = ((features-mean)/std).reshape(*features.shape[:-2], 118)
    relative = torch.log(prices[..., 0]/prices[..., 1]).float()[..., None]/.01
    price = torch.log(prices[..., 1]/60000).float()[..., None]
    clock = clock_features(times[..., -1]/21600, f.dtype)
    return torch.cat((f, relative, price, clock), -1)


class ContinuationValue(nn.Module):
    def __init__(self, feature_mean, feature_std, rules=Rules(), horizon=3600, width=64):
        super().__init__()
        self.rules, self.horizon, self.width = rules, horizon, width
        self.oracle = TensorOracle(rules)
        self.register_buffer('feature_mean', feature_mean.detach().clone())
        self.register_buffer('feature_std', feature_std.detach().clone())
        self.market = nn.Sequential(nn.Linear(124, width), nn.SiLU(), nn.Linear(width, 32), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(46, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.constant_(self.head[-1].bias, .001)

    def encode_market(self, features, prices, times):
        return self.market(market_coordinates(features, prices, times, self.feature_mean, self.feature_std))

    def account_coordinates(self, s, horizon):
        # Size ratios retain the absolute order-size contract. Account magnitude
        # cannot be discarded merely because gamma one is scale invariant.
        w = equity(s).clamp_min(1e-100)
        pending = s[..., SIDE] != 0
        tau = horizon.to(s.dtype)
        return torch.stack((s[..., Q]/w, s[..., A]*s[..., P]/w,
            torch.log(w/10000), self.rules.minimum/w, self.rules.maximum/w,
            s[..., SIDE], torch.where(pending, torch.log(s[..., LP].clamp_min(self.rules.tick)/s[..., P]), 0.)/.01,
            torch.where(pending, s[..., N]/w, 0.),
            torch.where(pending, s[..., N]/self.rules.reserve_leverage/w, 0.),
            torch.where(pending, (s[..., TIME]-s[..., PLACED]).clamp(0, 2), 0.),
            torch.log(s[..., PREV]/s[..., P])/.01,
            tau/self.horizon, torch.log1p(tau)/math.log1p(self.horizon), (tau/self.horizon).tanh()), -1).float()

    def value_encoded(self, encoded, state, horizon):
        if bool(((horizon < 0) | (horizon > self.horizon)).any()):
            raise ValueError('Continuation horizon outside trained contract')
        terminal = self.oracle.terminal(state)
        terminal_log = torch.where(terminal > 0, torch.log(terminal.clamp_min(1e-100)/10000), -torch.inf)
        # Unrestricted fee-close is only a smooth initialization, NOT a valid
        # terminal close: at tau=0 the actual capped close above is compulsory.
        arrived, _, _, arrival_valid = self.oracle.arrival(state)
        effective = arrived[..., Q]+self.oracle.effective(arrived)
        baseline = torch.log(effective.clamp_min(1e-100)/10000)
        inputs = self.account_coordinates(state, horizon)
        residual = self.head(torch.cat((encoded, inputs), -1)).squeeze(-1).double()
        predicted = baseline + .01*(horizon.double()/self.horizon)*residual
        # A valid immediate close followed by cash is an attainable lower policy.
        predicted = torch.maximum(predicted, terminal_log)
        predicted = torch.where((equity(state) > 0) & arrival_valid, predicted, -torch.inf)
        return torch.where(horizon == 0, terminal_log, predicted)

    def forward(self, features, prices, times, state, horizon):
        return self.value_encoded(self.encode_market(features, prices, times), state, horizon)


@torch.no_grad()
def bellman_backup(value, state, next_features, next_prices, next_times, horizon,
                   menu=Menu(step=.01), funding=1., chunk=65536):
    """One-second sample backup with exact terminal formula and full account.

    next_* describe the same conditional samples for EVERY candidate action.
    A finite sampling law is not a certificate for unsampled Gaussian tails.
    """
    if bool(((horizon < 1) | (horizon > value.horizon)).any()):
        raise ValueError('Backup needs a positive supported remaining horizon')
    oracle = value.oracle
    desc = oracle.descriptors(menu, state.device)
    out, valid, amounts = oracle.actions(state, desc)
    row, action = valid.nonzero(as_tuple=True)
    scores = torch.full(valid.shape, -torch.inf, device=state.device, dtype=torch.float64)
    branches = next_prices.shape[1]
    encoded = value.encode_market(next_features, next_prices, next_times)
    for start in range(0, len(row), max(1, chunk//branches)):
        r, a = row[start:start+max(1, chunk//branches)], action[start:start+max(1, chunk//branches)]
        moved = oracle.advance(out[r, a, None, :].expand(-1, branches, -1), next_prices[r, :, -1], funding)
        future = value.value_encoded(encoded[r].reshape(-1, 32), moved.reshape(-1, 9),
            (horizon[r, None]-1).expand(-1, branches).reshape(-1)).reshape(-1, branches)
        scores[r, a] = future.mean(1)
    best, choice = scores.max(1)
    batch = torch.arange(len(state), device=state.device)
    # False means no finite log continuation was found, not a clipped utility.
    return dict(value=best, choice=choice, valid=torch.isfinite(best),
        state=out[batch, choice], amount=amounts[batch, choice], descriptor=desc[choice], scores=scores)


def observed_inputs(data, indices, device):
    return timeline_inputs(data, data['indices'][np.asarray(indices)], device)


def timeline_inputs(data, offsets, device):
    """Offsets are physical seconds, independent of the archived origin stride."""
    pi = np.asarray(offsets)
    origin = int(data['origins'][0])+pi
    features = np.stack((np.asarray(data['timeline'][origin-1]), np.asarray(data['timeline'][origin])), 1).astype(np.float32)
    prices = np.stack((data['pricesWithPast'][pi], data['pricesWithPast'][pi+1]), 1)
    times = float(data['timestamps'][0])/1000+pi
    times = np.stack((times-1, times), 1)
    return tuple(torch.as_tensor(x, device=device) for x in (features, prices, times))


@torch.no_grad()
def conditional_children(nsf, features, prices, times, branches, seed):
    if branches < 2 or branches % 2:
        raise ValueError('Require an even positive conditional sample count')
    # Shared implementation preserves the trained NSF law and observation filter.
    from export_nsf_planning_trees import next_observations
    nf, np_ = next_observations(nsf, features.cpu().numpy(), prices.cpu().numpy(), times.cpu().numpy(),
        np.ones(len(features)), branches, seed)
    f = torch.stack((features[:, -1, None].expand(-1, branches, -1), torch.as_tensor(nf, device=features.device)), -2)
    p = torch.stack((prices[:, -1, None].expand(-1, branches), torch.as_tensor(np_, device=prices.device)), -1)
    t = torch.stack((times[:, -1, None].expand(-1, branches), (times[:, -1, None]+1).expand(-1, branches)), -1)
    return f, p, t


def save_value(path, value, metadata):
    torch.save(dict(contract=CONTRACT, state=value.state_dict(), rules=asdict(value.rules),
        horizon=value.horizon, width=value.width, metadata=metadata), path)


def load_value(path, device):
    saved = torch.load(path, map_location=device, weights_only=True)
    if saved['contract'] != CONTRACT:
        raise ValueError('Unknown continuation contract')
    weights = saved['state']
    model = ContinuationValue(weights['feature_mean'], weights['feature_std'], Rules(**saved['rules']), saved['horizon'], saved['width']).to(device)
    model.load_state_dict(weights)
    return model.eval(), saved['metadata']

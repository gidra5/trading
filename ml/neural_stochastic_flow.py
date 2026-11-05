"""Neural Stochastic Flows (Kiyohara et al., NeurIPS 2025), PyTorch port.

Equations 5--10: time-scaled Gaussian base, alternating affine couplings,
and both variational bridge KL bounds. Context is fixed for a queried path.
Market price marginal is a pushforward of this joint state law, not its mean.
"""
from __future__ import annotations
import math
from dataclasses import dataclass
import torch
from torch import nn, Tensor

CONTRACT = 'neural-stochastic-flow-price-volatility-v1'


def network(inputs, outputs, width=64):
    return nn.Sequential(nn.Linear(inputs, width), nn.SiLU(), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, outputs))


def normal_log_prob(value, mean, std):
    return (-.5*((value-mean)/std).square()-std.log()-.5*math.log(2*math.pi)).sum(-1)


class Couplings(nn.Module):
    def __init__(self, condition_size, width=64, layers=4):
        super().__init__()
        self.nets = nn.ModuleList([network(condition_size+1, 2, width) for _ in range(layers)])
        for net in self.nets:
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def transform(self, value, condition, duration, inverse=False):
        logdet = torch.zeros_like(duration)
        indices = range(len(self.nets)-1, -1, -1) if inverse else range(len(self.nets))
        for i in indices:
            a, b = i % 2, 1-i % 2
            raw = self.nets[i](torch.cat((value[:, a:a+1], condition), -1))
            # Bounded scale/shift networks retain identity and invertibility.
            scale = duration*.25*torch.tanh(raw[:, 0])
            shift = duration*torch.tanh(raw[:, 1])
            transformed = (value[:, b]-shift)*torch.exp(-scale) if inverse else value[:, b]*torch.exp(scale)+shift
            value = torch.stack((value[:, a], transformed), -1) if a == 0 else torch.stack((transformed, value[:, a]), -1)
            logdet = logdet+(-scale if inverse else scale)
        return value, logdet


class NeuralStochasticFlow(nn.Module):
    def __init__(self, feature_mean, feature_std, width=64, time_scale=21600., price_scale=.01):
        super().__init__()
        self.register_buffer('feature_mean', feature_mean.clone())
        self.register_buffer('feature_std', feature_std.clone())
        self.width, self.time_scale, self.price_scale = width, time_scale, price_scale
        condition_size = len(feature_mean)+3
        self.base = network(condition_size, 4, width)
        self.couplings = Couplings(condition_size, width)
        nn.init.zeros_(self.base[-1].weight); nn.init.zeros_(self.base[-1].bias)

    def condition(self, state, features, duration, start):
        local = torch.stack((torch.zeros_like(state[:, 0]), state[:, 1]), -1)
        return torch.cat(((features-self.feature_mean)/self.feature_std,
            local, torch.tanh(duration)[:, None]), -1)

    def parameters_at(self, state, features, duration, start):
        condition = self.condition(state, features, duration, start)
        raw = self.base(condition)
        local = torch.stack((torch.zeros_like(state[:, 0]), state[:, 1]), -1)
        mean = local+duration[:, None]*2*torch.tanh(raw[:, :2])
        std = duration[:, None].sqrt()*(torch.nn.functional.softplus(raw[:, 2:])+.001)
        return condition, mean, std

    def rsample(self, state, features, duration, start, noise=None):
        """Durations are physical seconds divided by time_scale, not tanh(dt)."""
        if bool((duration < 0).any()):
            raise ValueError('negative duration')
        condition, mean, std = self.parameters_at(state, features, duration, start)
        noise = torch.randn_like(state) if noise is None else noise
        out = self.couplings.transform(mean+std*noise, condition, duration)[0]
        return out+torch.stack((state[:, 0], torch.zeros_like(state[:, 0])), -1)

    def log_prob(self, target, state, features, duration, start):
        if bool((duration <= 0).any()):
            raise ValueError('zero-duration law is a point mass, not a Lebesgue density')
        condition, mean, std = self.parameters_at(state, features, duration, start)
        local = target-torch.stack((state[:, 0], torch.zeros_like(state[:, 0])), -1)
        base, inverse_logdet = self.couplings.transform(local, condition, duration, inverse=True)
        # Includes the Gaussian base scale Jacobian as well as coupling Jacobians.
        return normal_log_prob(base, mean, std)+inverse_logdet

    def distribution(self, features, log_volatility, current_price, horizon_seconds):
        h = torch.as_tensor(horizon_seconds, dtype=features.dtype, device=features.device)
        if h.ndim == 0:
            h = h.expand(len(features))
        if h.shape != (len(features),) or not bool(torch.all(torch.isfinite(h) & (h >= 0) & (h <= 86400))):
            raise ValueError('horizon must be finite seconds in [0,86400], one per row or scalar')
        if current_price.shape != h.shape or not bool(torch.all(torch.isfinite(current_price) & (current_price > 0))):
            raise ValueError('positive current prices required')
        state = torch.stack((torch.zeros_like(log_volatility), log_volatility), -1)
        return PriceMarginal(self, state, features, current_price, h/self.time_scale)


class Bridge(nn.Module):
    def __init__(self, feature_mean, feature_std, width=64):
        super().__init__()
        self.register_buffer('feature_mean', feature_mean.clone())
        self.register_buffer('feature_std', feature_std.clone())
        condition_size = len(feature_mean)+6
        self.base = network(condition_size, 4, width)
        self.couplings = Couplings(condition_size, width)
        nn.init.zeros_(self.base[-1].weight); nn.init.zeros_(self.base[-1].bias)

    def parameters_at(self, left, right, features, d1, d2, start):
        offset = torch.stack((left[:, 0], torch.zeros_like(left[:, 0])), -1)
        left, right = left-offset, right-offset
        condition = torch.cat(((features-self.feature_mean)/self.feature_std, left, right,
            torch.tanh(d1)[:, None], torch.tanh(d2)[:, None]), -1)
        raw = self.base(condition)
        duration = d1+d2
        interpolation = left+(right-left)*(d1/duration)[:, None]
        mean = interpolation+duration[:, None]*torch.tanh(raw[:, :2])
        std = (d1*d2/duration)[:, None].sqrt()*(torch.nn.functional.softplus(raw[:, 2:])+.001)
        return condition, mean, std

    def rsample(self, left, right, features, d1, d2, start, noise=None):
        c, mu, std = self.parameters_at(left, right, features, d1, d2, start)
        z = mu+std*(torch.randn_like(mu) if noise is None else noise)
        return self.couplings.transform(z, c, d1+d2)[0]+torch.stack((left[:, 0], torch.zeros_like(left[:, 0])), -1)

    def log_prob(self, mid, left, right, features, d1, d2, start):
        c, mu, std = self.parameters_at(left, right, features, d1, d2, start)
        local = mid-torch.stack((left[:, 0], torch.zeros_like(left[:, 0])), -1)
        z, jac = self.couplings.transform(local, c, d1+d2, inverse=True)
        return normal_log_prob(z, mu, std)+jac


def flow_loss(flow, bridge, state, features, d1, d2, start):
    """Unclipped MC estimators of Eq. 8 and Eq. 9, differentiable samples."""
    end = flow.rsample(state, features, d1+d2, start)
    mid = bridge.rsample(state, end, features, d1, d2, start)
    forward = (flow.log_prob(end, state, features, d1+d2, start)
        +bridge.log_prob(mid, state, end, features, d1, d2, start)
        -flow.log_prob(mid, state, features, d1, start)
        -flow.log_prob(end, mid, features, d2, start+d1))
    mid = flow.rsample(state, features, d1, start)
    end = flow.rsample(mid, features, d2, start+d1)
    reverse = (flow.log_prob(mid, state, features, d1, start)
        +flow.log_prob(end, mid, features, d2, start+d1)
        -bridge.log_prob(mid, state, end, features, d1, d2, start)
        -flow.log_prob(end, state, features, d1+d2, start))
    return forward.mean()+reverse.mean()


@dataclass(frozen=True)
class PriceMarginal:
    flow: NeuralStochasticFlow
    state: Tensor
    features: Tensor
    current_price: Tensor
    duration: Tensor

    @torch.inference_mode()
    def sample_log_returns(self, samples=256, seed=1):
        # Scrambled Sobol integration for the 2D Gaussian pushforward.
        if not isinstance(samples, int) or samples < 1:
            raise ValueError('positive integer sample count required')
        count = samples//2 if samples % 2 == 0 else samples
        u = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(count).to(self.state)
        noise = torch.special.ndtri(u.clamp(1e-7, 1-1e-7))
        if samples % 2 == 0:
            noise = torch.cat((noise, -noise), 0)
        batch = len(self.state)
        x = self.state[:, None, :].expand(batch, samples, 2).reshape(-1, 2)
        f = self.features[:, None, :].expand(batch, samples, -1).reshape(batch*samples, -1)
        d = self.duration[:, None].expand(batch, samples).reshape(-1)
        eps = noise[None].expand(batch, samples, 2).reshape(-1, 2)
        out = self.flow.rsample(x, f, d, torch.zeros_like(d), eps)
        return out[:, 0].reshape(batch, samples)*self.flow.price_scale

    @torch.inference_mode()
    def sample_prices(self, samples=256, seed=1, tick_size=.01):
        price = self.current_price[:, None]*self.sample_log_returns(samples, seed).double().exp()
        if tick_size is None:
            return price
        if tick_size <= 0:
            raise ValueError('positive tick size required')
        return torch.clamp(torch.round(price/tick_size), min=1)*tick_size

    @torch.inference_mode()
    def cdf(self, price, samples=1024, seed=1, tick_size=.01):
        """Numerical marginal CDF, not an exact analytic integration."""
        return (self.sample_prices(samples, seed, tick_size) <= price[:, None]).double().mean(1)


def load_nsf(file, device):
    saved = torch.load(file, map_location='cpu', weights_only=False)
    if saved.get('contract') != CONTRACT:
        raise ValueError('unsupported NSF checkpoint')
    model = NeuralStochasticFlow(saved['featureMean'], saved['featureStd'], saved['width'],
        saved['timeScale'], saved['priceScale'])
    model.load_state_dict(saved['model'])
    return model.to(device).eval(), saved

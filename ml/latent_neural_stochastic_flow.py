"""Causal variational encoder, latent NSF, and a distributional observation decoder."""
from __future__ import annotations
from dataclasses import dataclass
import math
import torch
from torch import nn
from neural_stochastic_flow import network, normal_log_prob, flow_loss

CONTRACT = 'latent-nsf-calendar-multiscale-v2'
TIME_SCALE = 21600.

def horizon_features(duration):
    return torch.stack((duration, duration.tanh(), torch.log1p(duration*TIME_SCALE)), -1)

def clock_features(start, dtype):
    # start is Unix time / TIME_SCALE, kept in float64 until after phase reduction.
    seconds = start.double()*TIME_SCALE
    periods = seconds.new_tensor([86400., 604800.])
    phase = 2*math.pi*torch.remainder(seconds[..., None], periods)/periods
    return torch.cat((phase.sin(), phase.cos()), -1).to(dtype)

def transition_time_features(duration, start, dtype):
    return torch.cat((horizon_features(duration), clock_features(start, dtype),
        clock_features(start+duration.double(), dtype)), -1)



def sample_skip_pairs(elapsed, max_skip=4, uniforms=None):
    """One uniform nonadjacent target per eligible distinct source time.

    max_skip counts distinct observation intervals, not seconds. At sequence
    boundaries the target window is truncated. Duplicate timestamps are padding.
    """
    if max_skip < 2:
        raise ValueError('skip span must include at least two observation intervals')
    active = elapsed > 0
    active = torch.cat((torch.ones_like(active[:, :1]), active[:, 1:]), 1)
    rank = active.long().cumsum(1)
    gap = rank[:, None, :]-rank[:, :, None]
    valid = active[:, :, None] & active[:, None, :] & (gap >= 2) & (gap <= max_skip)
    counts = valid.sum(-1)
    if uniforms is None:
        uniforms = torch.rand(counts.shape, device=elapsed.device)
    choice = (uniforms*counts).long()
    selected = valid & (valid.long().cumsum(-1)-1 == choice[..., None])
    batch, source = (counts > 0).nonzero(as_tuple=True)
    target = selected.long().argmax(-1)[batch, source]
    return batch, source, target


class LatentCouplings(nn.Module):
    def __init__(self, dimensions, conditions, width=64):
        super().__init__()
        if dimensions % 2:
            raise ValueError('even latent dimension required')
        self.dimensions = dimensions
        self.nets = nn.ModuleList([network(conditions+dimensions//2, dimensions, width) for _ in range(4)])
        for net in self.nets:
            nn.init.zeros_(net[-1].weight); nn.init.zeros_(net[-1].bias)

    def transform(self, value, condition, duration, inverse=False):
        logdet = torch.zeros_like(duration)
        for i in (range(3, -1, -1) if inverse else range(4)):
            a, b = i % 2, 1-i % 2
            fixed, changed = value[:, a::2], value[:, b::2]
            scale, shift = self.nets[i](torch.cat((fixed, condition), -1)).chunk(2, -1)
            scale = duration[:, None]*.25*torch.tanh(scale)
            shift = duration[:, None]*torch.tanh(shift)
            changed = (changed-shift)*torch.exp(-scale) if inverse else changed*torch.exp(scale)+shift
            out = torch.empty_like(value)
            out[:, a::2], out[:, b::2] = fixed, changed
            value = out
            logdet = logdet+(-scale if inverse else scale).sum(-1)
        return value, logdet


class LatentTransition(nn.Module):
    def __init__(self, dimensions=256, width=256):
        super().__init__()
        self.base = network(dimensions+11, dimensions*2, width)
        self.couplings = LatentCouplings(dimensions, dimensions+11, width)
        nn.init.zeros_(self.base[-1].weight); nn.init.zeros_(self.base[-1].bias)

    def parameters_at(self, state, duration, start):
        condition = torch.cat((state, transition_time_features(duration, start, state.dtype)), -1)
        drift, diffusion = self.base(condition).chunk(2, -1)
        return condition, state+duration[:, None]*2*drift.tanh(), duration[:, None].sqrt()*(nn.functional.softplus(diffusion)+.001)

    def rsample(self, state, features, duration, start, noise=None):
        c, mu, sigma = self.parameters_at(state, duration, start)
        return self.couplings.transform(mu+sigma*(torch.randn_like(mu) if noise is None else noise), c, duration)[0]

    def log_prob(self, target, state, features, duration, start):
        if bool((duration <= 0).any()):
            raise ValueError('positive duration required for a latent density')
        c, mu, sigma = self.parameters_at(state, duration, start)
        base, jac = self.couplings.transform(target, c, duration, inverse=True)
        return normal_log_prob(base, mu, sigma)+jac


class LatentBridge(nn.Module):
    def __init__(self, dimensions=256, width=256):
        super().__init__()
        self.base = network(2*dimensions+18, 2*dimensions, width)
        self.couplings = LatentCouplings(dimensions, 2*dimensions+18, width)
        nn.init.zeros_(self.base[-1].weight); nn.init.zeros_(self.base[-1].bias)

    def parameters_at(self, left, right, d1, d2, start):
        c = torch.cat((left, right, horizon_features(d1), horizon_features(d2),
            clock_features(start, left.dtype), clock_features(start+d1.double(), left.dtype),
            clock_features(start+d1.double()+d2.double(), left.dtype)), -1)
        mu, sigma = self.base(c).chunk(2, -1)
        duration = d1+d2
        mean = left+(right-left)*(d1/duration)[:, None]+duration[:, None]*mu.tanh()
        std = (d1*d2/duration)[:, None].sqrt()*(nn.functional.softplus(sigma)+.001)
        return c, mean, std

    def rsample(self, left, right, features, d1, d2, start, noise=None):
        c, mu, sigma = self.parameters_at(left, right, d1, d2, start)
        return self.couplings.transform(mu+sigma*(torch.randn_like(mu) if noise is None else noise), c, d1+d2)[0]

    def log_prob(self, mid, left, right, features, d1, d2, start):
        c, mu, sigma = self.parameters_at(left, right, d1, d2, start)
        base, jac = self.couplings.transform(mid, c, d1+d2, inverse=True)
        return normal_log_prob(base, mu, sigma)+jac


class LatentNSF(nn.Module):
    def __init__(self, feature_mean, feature_std, dimensions=256, width=256):
        super().__init__()
        self.dimensions, self.width, self.time_scale, self.price_scale = dimensions, width, 21600., .01
        self.register_buffer('feature_mean', feature_mean.clone())
        self.register_buffer('feature_std', feature_std.clone())
        self.encoder = nn.GRUCell(67, width)
        self.posterior = nn.Linear(width, dimensions*2)
        self.transition = LatentTransition(dimensions, width)
        self.decoder = network(dimensions, 120, width)
        nn.init.zeros_(self.posterior.weight); nn.init.zeros_(self.posterior.bias)
        nn.init.zeros_(self.decoder[-1].weight); nn.init.zeros_(self.decoder[-1].bias)
        with torch.no_grad():
            self.posterior.bias[dimensions:] = math.log(math.expm1(1.))
            self.decoder[-1].bias[60:] = math.log(math.expm1(1.))

    def observations(self, features, prices):
        # Prices are anchored to the known forecast-origin price for this window.
        return torch.cat((prices.to(features.dtype)[..., None]/self.price_scale,
            (features-self.feature_mean)/self.feature_std), -1)

    def encode(self, observations, elapsed_seconds, timestamps_seconds):
        hidden = observations.new_zeros((len(observations), self.width))
        means, stds = [], []
        for i in range(observations.shape[1]):
            dt = elapsed_seconds[:, i]/self.time_scale
            candidate = self.encoder(torch.cat((observations[:, i], horizon_features(dt),
                clock_features(timestamps_seconds[:, i].double()/self.time_scale, observations.dtype)), -1), hidden)
            active = (dt > 0) if i else torch.ones_like(dt, dtype=torch.bool)
            hidden = torch.where(active[:, None], candidate, hidden)
            mu, raw = self.posterior(hidden).chunk(2, -1)
            # Price skip connection initializes one latent coordinate to observed price.
            # It remains learned through the posterior residual and decoder residual.
            mu = mu+torch.cat((observations[:, i, :1], torch.zeros_like(mu[:, 1:])), -1)
            scales = torch.cat((raw.new_full((len(raw), 1), .001), raw.new_full((len(raw), self.dimensions-1), .02)), -1)
            std = nn.functional.softplus(raw)*scales+1e-5
            if i:
                mu = torch.where(active[:, None], mu, means[-1])
                std = torch.where(active[:, None], std, stds[-1])
            means.append(mu); stds.append(std)
        return torch.stack(means, 1), torch.stack(stds, 1)

    def emission(self, state):
        mu, raw = self.decoder(state).chunk(2, -1)
        mu = mu+torch.cat((state[:, :1], torch.zeros_like(mu[:, 1:])), -1)
        scale = torch.cat((raw.new_full((len(raw), 1), .001), torch.ones_like(raw[:, 1:])), -1)
        return mu, nn.functional.softplus(raw)*scale+1e-5

    def variational_loss(self, obs, elapsed, timestamps_seconds, beta=.1, beta_skip=.1, max_skip=4):
        mu, sigma = self.encode(obs, elapsed, timestamps_seconds)
        samples = mu+sigma*torch.randn_like(mu)
        zs = [samples[:, 0]]
        for i in range(1, samples.shape[1]):
            zs.append(torch.where((elapsed[:, i] > 0)[:, None], samples[:, i], zs[-1]))
        z = torch.stack(zs, 1)
        emu, esigma = self.emission(z.flatten(0, 1))
        recon = -normal_log_prob(obs.flatten(0, 1), emu, esigma).reshape(obs.shape[:2])
        active = (elapsed > 0).to(obs.dtype); active[:, 0] = 1
        recon = (recon*active).sum(1)/active.sum(1)
        prior_kl = .5*(mu[:, 0].square()+sigma[:, 0].square()-1-2*sigma[:, 0].log()).sum(-1)
        terms = [prior_kl]
        empty = obs.new_empty((len(obs), 0))
        starts = timestamps_seconds.double()/self.time_scale
        for i in range(1, obs.shape[1]):
            duration = elapsed[:, i].clamp_min(1)/self.time_scale
            kl = normal_log_prob(z[:, i], mu[:, i], sigma[:, i])-self.transition.log_prob(z[:, i], z[:, i-1], empty, duration, starts[:, i-1])
            terms.append(kl*(elapsed[:, i] > 0))
        kl = torch.stack(terms, 1).sum(1)/active.sum(1)
        # Direct transition from every eligible source to a uniformly sampled
        # nonadjacent target. Both posteriors remain differentiable and causal.
        skip = obs.new_zeros(len(obs))
        if beta_skip:
            batch, source, target = sample_skip_pairs(elapsed, max_skip)
            if len(batch):
                duration = ((timestamps_seconds[batch, target]-timestamps_seconds[batch, source])/self.time_scale).to(obs.dtype)
                pair_kl = normal_log_prob(z[batch, target], mu[batch, target], sigma[batch, target])-self.transition.log_prob(
                    z[batch, target], z[batch, source], obs.new_empty((len(batch), 0)), duration, starts[batch, source])
                skip = skip.index_add(0, batch, pair_kl)
                counts = torch.bincount(batch, minlength=len(obs)).clamp_min(1).to(obs.dtype)
                skip = skip/counts
        loss = recon.mean()+beta*kl.mean()+beta_skip*skip.mean()
        return loss, {'reconstruction': recon.mean(), 'adjacentKl': kl.mean(), 'skipKl': skip.mean()}, z[:, 1]

    @torch.inference_mode()
    def distribution(self, observed_features, observed_prices, elapsed_seconds, horizon_seconds, timestamps_seconds, components=256, seed=13):
        if observed_features.ndim != 3 or observed_features.shape[-1] != 59:
            raise ValueError('observed features must be [batch,history,59]')
        if observed_prices.shape != observed_features.shape[:2] or elapsed_seconds.shape != observed_prices.shape:
            raise ValueError('price and elapsed-time history shapes must match features')
        if not bool(torch.all(torch.isfinite(observed_features))) or not bool(torch.all(torch.isfinite(observed_prices) & (observed_prices > 0))) or not bool(torch.all(torch.isfinite(elapsed_seconds) & (elapsed_seconds >= 0))):
            raise ValueError('invalid observed history')
        if timestamps_seconds.shape != observed_prices.shape or timestamps_seconds.dtype != torch.float64:
            raise ValueError('timestamps must be float64 Unix seconds matching history shape')
        if not bool(torch.isfinite(timestamps_seconds).all()) or not torch.allclose(
                timestamps_seconds[:, 1:]-timestamps_seconds[:, :-1], elapsed_seconds[:, 1:].double(), atol=1e-6, rtol=0):
            raise ValueError('timestamps and elapsed seconds disagree')
        current = observed_prices[:, -1]
        obs = self.observations(observed_features, (observed_prices/current[:, None]).log())
        qmu, qstd = self.encode(obs, elapsed_seconds, timestamps_seconds)
        h = torch.as_tensor(horizon_seconds, dtype=obs.dtype, device=obs.device)
        if h.ndim == 0:
            h = h.expand(len(obs))
        if h.shape != (len(obs),) or not bool(torch.all(torch.isfinite(h) & (h >= 0) & (h <= 86400))):
            raise ValueError('horizon must be finite seconds in [0,86400]')
        if not isinstance(components, int) or components < 2 or components % 2:
            raise ValueError('positive even mixture component count required')
        u = torch.quasirandom.SobolEngine(self.dimensions*2, scramble=True, seed=seed).draw(components//2).to(obs)
        eps = torch.special.ndtri(u.clamp(1e-7, 1-1e-7)); eps = torch.cat((eps, -eps), 0)
        initial = qmu[:, -1, None]+qstd[:, -1, None]*eps[None, :, :self.dimensions]
        duration = (h/self.time_scale)[:, None].expand(-1, components).reshape(-1)
        noise = eps[None, :, self.dimensions:].expand(len(obs), -1, -1).reshape(-1, self.dimensions)
        evolved = self.transition.rsample(initial.reshape(-1, self.dimensions), obs.new_empty((len(duration), 0)), duration,
            (timestamps_seconds[:, -1]/self.time_scale)[:, None].expand(-1, components).reshape(-1), noise)
        mu, std = self.emission(evolved)
        mu = mu[:, 0].reshape(len(obs), components)*self.price_scale
        std = std[:, 0].reshape(len(obs), components)*self.price_scale
        # The current observed price is known, not a fresh noisy observation.
        mu = torch.where((h == 0)[:, None], 0., mu)
        std = torch.where((h == 0)[:, None], 0., std)
        return LatentPriceMixture(current, mu.double(), std.double())


@dataclass(frozen=True)
class LatentPriceMixture:
    current_price: torch.Tensor
    log_means: torch.Tensor
    log_stds: torch.Tensor

    def mean_log_return(self):
        return self.log_means.mean(1)

    def mean_price(self):
        return self.current_price*(self.log_means+.5*self.log_stds.square()).exp().mean(1)

    def cdf(self, price, tick_size=.01):
        boundary = price if tick_size is None else (torch.floor(price/tick_size+1e-8)+.5)*tick_size
        y = (boundary/self.current_price).log()[:, None]
        gaussian = torch.special.ndtr((y-self.log_means)/self.log_stds.clamp_min(1e-30))
        return torch.where(self.log_stds == 0, (y >= self.log_means).double(), gaussian).mean(1)

    def tick_probability(self, price, tick_size=.01):
        return (self.cdf(price, tick_size)-self.cdf(price-tick_size, tick_size)).clamp_min(0)

    def log_return_density(self, returns):
        if bool((self.log_stds == 0).any()):
            raise ValueError('zero-horizon observation is a point mass')
        z = (returns[:, None]-self.log_means)/self.log_stds
        ll = -.5*z.square()-self.log_stds.log()-.5*math.log(2*math.pi)
        return torch.logsumexp(ll, 1)-math.log(self.log_means.shape[1])

    def sample_prices(self, samples=512, seed=37, tick_size=.01):
        u = torch.quasirandom.SobolEngine(2, scramble=True, seed=seed).draw(samples).to(self.log_means)
        index = (u[:, 0]*self.log_means.shape[1]).long().clamp_max(self.log_means.shape[1]-1)
        eps = torch.special.ndtri(u[:, 1].clamp(1e-9, 1-1e-9))
        prices = self.current_price[:, None]*(self.log_means[:, index]+self.log_stds[:, index]*eps).exp()
        return prices if tick_size is None else (prices/tick_size).round().clamp_min(1)*tick_size


def load_latent_nsf(file, device):
    saved = torch.load(file, map_location='cpu', weights_only=False)
    if saved.get('contract') != CONTRACT:
        raise ValueError('unsupported latent NSF checkpoint')
    model = LatentNSF(saved['featureMean'], saved['featureStd'], saved['dimensions'], saved['width'])
    model.load_state_dict(saved['model'])
    return model.to(device).eval(), saved

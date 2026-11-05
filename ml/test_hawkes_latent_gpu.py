"""Independent, small float64 CPU oracle for the addressed CUDA random law."""
import math
import unittest
import numpy as np
import torch
from scipy.stats import poisson
from hawkes_latent import LatentModel, Diffusion, integrated_count, physical_arrays
from hawkes_modulation import Anchor
from hawkes_research import Mode
from hawkes_latent_gpu import filter_batch


def uniform(seed, offset):
    mask = 0xffffffff
    c0, c1, c2, c3 = offset & mask, (offset >> 32) & mask, 0, 0
    k0, k1 = seed & mask, (seed >> 32) & mask
    for _ in range(10):
        p0, p1 = 0xD2511F53 * c0, 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0) & mask, p1 & mask, ((p0 >> 32) ^ c3 ^ k1) & mask, p0 & mask
        k0, k1 = (k0 + 0x9E3779B9) & mask, (k1 + 0xBB67AE85) & mask
    return max(((c0 << 21) | (c1 >> 11)) / 9007199254740992., 1. / 18014398509481984.)


def normal(seed, offset):
    return math.sqrt(-2. * math.log(uniform(seed, offset * 2))) * math.cos(2. * math.pi * uniform(seed, offset * 2 + 1))


def reference(model, y, valid, observed_mean, particles, seed, first_fast=None, first_slow=None):
    bg, target, source, mass, decay, anchor, adecay, beta, initial, bounded = physical_arrays(model, observed_mean)
    k, a, q = len(target), len(anchor), len(model.diffusions)
    zd = np.exp(-1. / np.array([z.tau for z in model.diffusions]))
    c = np.array([z.loading for z in model.diffusions]).reshape(q, 3)
    sd, variance = np.sqrt(1 - zd * zd), np.sum(c * c, axis=0)
    fast, slope = np.zeros((particles, k)), np.zeros((particles, k))
    slow, sslope = np.zeros((particles, a)), np.zeros((particles, a))
    pending, z = np.zeros(particles), np.zeros((particles, q))
    result = np.zeros((len(y), 7))
    reset = True
    for t, observed in enumerate(y):
        if not valid[t]:
            reset = True
            continue
        if reset:
            fast[:] = first_fast if t == 0 and first_fast is not None else initial[source]
            slow[:] = first_slow if t == 0 and first_slow is not None else initial[anchor]
            slope[:] = 0.; sslope[:] = 0.; pending[:] = 0.
            for p in range(particles):
                for j in range(q):
                    z[p, j] = normal(seed, (t * particles + p) * 64 + j)
            reset = False
        bases, slopes = np.zeros((particles, 2)), np.zeros((particles, 2))
        immigration = np.repeat(bg[None, :], particles, axis=0)
        correction = np.zeros(particles)
        for side in range(2):
            bases[:, side] = (fast[:, target == side] * mass[target == side]).sum(axis=1)
            slopes[:, side] = (slope[:, target == side] * mass[target == side]).sum(axis=1)
        prediction = bases + slopes * pending[:, None]
        if q:
            prediction += bg[None, :2] * np.exp((z * zd[None, :]) @ c[:, :2] - .5 * np.sum((c[:, :2] * zd[:, None]) ** 2, axis=0))
        else:
            prediction += bg[None, :2]
        result[t, :2] = prediction.mean(axis=0)
        for p in range(particles):
            offset = (t * particles + p) * 64
            if q:
                pm, center, precision = zd * z[p], np.zeros(q), np.ones(q)
                kernel = bases[p] + slopes[p] * pending[p]
                for iteration in range(9):
                    gradient, precision = -center.copy(), np.ones(q)
                    for side in range(2):
                        immigration_side = bg[side] * np.exp(-.5 * variance[side] + np.dot(c[:, side], pm + sd * center))
                        rate = max(kernel[side] + immigration_side, 1e-300)
                        coefficient = c[:, side] * sd
                        gradient += coefficient * immigration_side * (observed[side] / rate - 1)
                        precision += coefficient * coefficient * immigration_side * immigration_side / rate
                    if iteration == 8:
                        break
                    delta = np.clip(gradient / precision, -2., 2.) * (.5 if q > 1 else 1.)
                    center += delta
                    if np.max(np.abs(delta)) < 1e-4:
                        break
                psd = 1 / np.sqrt(precision)
                noise = np.array([normal(seed, offset + 8 + j) for j in range(q)])
                innovation = noise if uniform(seed, offset + 24) < .25 else center + psd * noise
                z[p] = pm + sd * innovation
                lp = -.5 * np.sum(innovation * innovation)
                lg = -np.sum(.5 * ((innovation - center) / psd) ** 2 + np.log(psd))
                correction[p] = lp - np.logaddexp(np.log(.25) + lp, np.log(.75) + lg)
                immigration[p] = bg * np.exp(-.5 * variance + z[p] @ c)
        bases += immigration[:, :2]
        integrals = [integrated_count(pending[p], bases[p], slopes[p], observed) for p in range(particles)]
        ll = np.array([v[0] for v in integrals]) + correction
        weights = np.exp(ll - ll.max()); normalization = weights.sum(); weights /= normalization
        result[t, 2] = -(ll.max() + np.log(normalization / particles))
        result[t, 3] = 1. / np.sum(weights * weights)
        result[t, 5] = weights @ np.array([v[3] for v in integrals])
        result[t, 6] = weights @ np.array([v[4] for v in integrals])
        thresholds = uniform(seed, t * particles * 64 + 63) / particles + np.arange(particles) / particles
        ancestors = np.minimum(np.searchsorted(np.cumsum(weights), thresholds), particles - 1)
        counts = np.zeros(particles)
        for p, ancestor in enumerate(ancestors):
            normalizer, lo, hi, mean, _ = integrals[ancestor]
            u = uniform(seed, (t * particles + p) * 64 + 25)
            if lo == -1:
                continue
            if lo == -2:
                counts[p] = poisson.ppf(u, mean)
            else:
                ns = np.arange(lo, hi + 1)
                from scipy.special import gammaln
                rates = bases[ancestor] + ns[:, None] * slopes[ancestor]
                logs = ns * np.log(pending[ancestor]) - pending[ancestor] - gammaln(ns + 1) + np.sum(observed * np.log(rates) - rates, axis=1)
                counts[p] = ns[min(len(ns) - 1, np.searchsorted(np.cumsum(np.exp(logs - normalizer)), u))]
        full_fast = fast[ancestors] + slope[ancestors] * counts[:, None]
        full_slow = slow[ancestors] + sslope[ancestors] * counts[:, None]
        pending = immigration[ancestors, 2].copy()
        for j in range(k):
            if target[j] == 2:
                pending += mass[j] * full_fast[:, j]
        result[t, 4] = pending.mean()
        z = z[ancestors]
        normalized = full_slow / np.maximum(initial[anchor], 1e-8)
        factor = 1 + (np.tanh(normalized) if bounded else normalized) @ beta.T
        fast = decay[None, :] * full_fast
        slope[:] = 0.
        for j in range(k):
            if source[j] == 2:
                slope[:, j] = (1 - decay[j]) * factor[:, j]
            else:
                fast[:, j] += (1 - decay[j]) * factor[:, j] * observed[source[j]]
        slow = adecay[None, :] * full_slow
        sslope[:] = 0.
        for j in range(a):
            if anchor[j] == 2:
                sslope[:, j] = 1 - adecay[j]
            else:
                slow[:, j] += (1 - adecay[j]) * observed[anchor[j]]
    return result


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA device required')
class GpuFilterTests(unittest.TestCase):
    def test_cpu_oracle_hidden_counts_resampling_anchors_and_ou(self):
        modes = (Mode(0, 0, .2, 3.), Mode(1, 1, .25, 20.), Mode(0, 2, 4., 1.5),
                 Mode(1, 2, 3., 5.), Mode(2, 0, .01, 4.), Mode(2, 2, .15, 8.))
        anchors = (Anchor(0, 30.), Anchor(2, 50.))
        beta = tuple((.02, .01) for _ in modes)
        models = [LatentModel((1., 2., .03), modes),
                  LatentModel((1., 2., .03), modes, anchors, beta),
                  LatentModel((1., 2., .03), modes, diffusions=(Diffusion(2., (.5, -.3, .2)),)),
                  LatentModel((1., 2., .03), modes, anchors, beta,
                              (Diffusion(2., (.5, -.3, .2)), Diffusion(300., (.1, .2, -.1))), 'bounded'),
                  LatentModel((1., 2., .03), modes, diffusions=tuple(Diffusion(2. + j * 3., (.2, -.1, .03)) for j in range(4)))]
        y = np.random.default_rng(137).poisson(3, (48, 2)).astype(float)
        y[19] = (40, 30)
        valid = np.ones(len(y), bool); valid[27:30] = False
        means = np.array([3., 3.])
        gpu = filter_batch(models, y, valid, means, 16, [71] * len(models), chunk_seconds=7)
        for i, model in enumerate(models):
            cpu = reference(model, y, valid, means, 16, 71)
            actual = np.column_stack([gpu[0][i], *(field[i] for field in gpu[1:])])
            np.testing.assert_allclose(actual, cpu, atol=2e-8, rtol=2e-9)

    def test_chunk_batch_invariance_and_causality(self):
        model = LatentModel((1., 2., .1), (Mode(0, 2, 5., 2.), Mode(2, 0, .01, 5.)),
                            diffusions=(Diffusion(3., (.6, .2, .1)),))
        y = np.random.default_rng(95).poisson(3, (36, 2)).astype(float)
        valid, mean = np.ones(len(y), bool), np.array([3., 3.])
        first = filter_batch([model], y, valid, mean, 16, [71], chunk_seconds=5)
        second = filter_batch([model, model], y, valid, mean, 16, [99, 71], chunk_seconds=17)
        altered = y.copy(); altered[21:] = 500.
        future = filter_batch([model], altered, valid, mean, 16, [71], chunk_seconds=9)
        for j in range(6):
            np.testing.assert_array_equal(first[j][0], second[j][1])
        np.testing.assert_array_equal(first[0][0, :22], future[0][0, :22])

    def test_rare_posterior_impulse_and_large_poisson_counts(self):
        models = [LatentModel((1., 1., .01), (Mode(0, 2, 50., .1),)),
                  LatentModel((1., 1., 850.), (Mode(2, 2, .1, 3.),))]
        y = np.array([[1., 1.], [10000., 2.], [2., 1.], [0., 0.]])
        valid, mean = np.ones(len(y), bool), np.array([2., 2.])
        gpu = filter_batch(models, y, valid, mean, 8, [91, 91], chunk_seconds=2)
        for i, model in enumerate(models):
            actual = np.column_stack([gpu[0][i], *(field[i] for field in gpu[1:])])
            np.testing.assert_allclose(actual, reference(model, y, valid, mean, 8, 91), atol=2e-7, rtol=1e-9)

    def test_variable_input_segments_and_custom_initial_traces(self):
        model = LatentModel((1., 2., .03), (Mode(0, 0, .2, 4.), Mode(0, 2, 5., 2.)),
                            (Anchor(0, 20.),), ((.01,), (.03,)), (Diffusion(3., (.3, -.1, .1)),))
        y = np.random.default_rng(105).poisson(3, (48, 2)).astype(float)
        valid = np.ones(len(y), bool); valid[27:30] = False
        segments = np.array([[3, 29], [10, 40]])
        fast, slow = [np.array([7., .2]), np.array([4., .1])], [np.array([6.]), np.array([9.])]
        gpu = filter_batch([model] * 2, y, valid, np.array([3., 3.]), 8, [75, 76],
                           segments=segments, initial_fast=fast, initial_slow=slow, chunk_seconds=5)
        for i, (begin, end) in enumerate(segments):
            actual = np.column_stack([gpu[0][i, :end-begin], *(field[i, :end-begin] for field in gpu[1:])])
            expected = reference(model, y[begin:end], valid[begin:end], np.array([3., 3.]), 8, 75+i, fast[i], slow[i])
            np.testing.assert_allclose(actual, expected, atol=2e-8, rtol=2e-9)

    def test_compact_weighted_loss_matches_full_filter_with_duplicate_windows(self):
        model = LatentModel((1., 2., .1), (Mode(0, 2, 5., 2.), Mode(2, 0, .01, 5.)),
                            diffusions=(Diffusion(3., (.6, .2, .1)),))
        y = np.random.default_rng(98).poisson(3, (40, 2)).astype(float)
        valid, mean = np.ones(len(y), bool), np.array([3., 3.])
        segments = np.array([[0, 40], [7, 35]])
        weights = np.zeros((2, 40), np.int8)
        weights[:, 10:20] = 2
        weights[:, 23:27] = 1
        full = filter_batch([model] * 2, y, valid, mean, 16, [71, 72], segments=segments, chunk_seconds=5)
        compact = filter_batch([model] * 2, y, valid, mean, 16, [71, 72], segments=segments,
                               loss_weights=weights, chunk_seconds=7)
        np.testing.assert_allclose(compact, np.sum(full[1] * weights, axis=1), atol=1e-12, rtol=1e-13)
        self.assertFalse(compact.flags.writeable)


if __name__ == '__main__':
    unittest.main()

"""Paired local market-state refinement against an unchanged distant tail.

Both refinements use identical random draws at the next hypothetical histories.
One queries the matched retained histories; the other queries the sampled real
histories before projection. This isolates a local sensitivity, not the global
error against an uncompressed optimal policy.
"""
import numpy as np
from audit_oracle_market_projection import input_coordinates, nearest, law_errors
from export_nsf_planning_trees import next_observations
from oracle_book_mc import backup_layer, decide, post_value


def candidate_schedule(m, depth, seed, count):
    E = len(m.a['exposures'])
    C = len(m.a['coordinates'])
    rng = np.random.default_rng(seed + int(m.cloud.origin_seconds) + depth)
    extra = max(0, min(C-E, count-E))
    return np.r_[np.arange(E), rng.choice(np.arange(E, C), extra, replace=False)].astype(np.int64)


def backup(m, parents, prices, destinations, child, candidates, next_factors):
    a, g = m.a, m.grid
    B = prices.shape[1]
    return backup_layer(parents, prices, destinations, np.tile(np.arange(B), (len(parents), 1)),
        np.full(len(a['coordinates']), B), child, a['wealth'], a['exposures'], a['books'], a['book_counts'],
        a['code_offsets'], a['book_lookup'], a['size_edges'], g.price_bins, g.price_scale,
        m.rules, candidates, False, next_factors)[0]


def local_fields(m, model, features, prices, times, offset, seed, candidate_count,
                 sample_seed, branches=8, predict=next_observations):
    if not 0 <= offset <= m.shape[0]-4:
        raise ValueError('Local refresh requires at least four remaining transitions')
    f, p, t = map(np.asarray, (features, prices, times))
    if t[-1] != m.cloud.origin_seconds + offset:
        raise ValueError('Observation timestamp differs from the fixed tail')
    if branches < 2 or branches % 2:
        raise ValueError('Even conditional sample count required')
    mean = model.feature_mean.cpu().numpy().astype(np.float64)
    std = model.feature_std.cpu().numpy().astype(np.float64)
    def coords(features, prices):
        return input_coordinates(np.asarray(features, dtype=np.float64), np.asarray(prices, dtype=np.float64), mean, std, p[-1])
    c = m.cloud
    def match(features, prices, depth):
        n = c.counts[depth]
        return nearest(coords(features, prices), coords(c.features[depth, :n], c.price_pairs[depth, :n]))
    B = branches
    root_f, root_p = predict(model, f[None], p[None], t[None], np.ones(1), B, sample_seed)
    raw_f = np.stack((np.repeat(f[-1][None], B, axis=0), root_f[0]), 1)
    raw_p = np.stack((np.full(B, p[-1]), root_p[0]), 1)
    indices = match(raw_f, raw_p, offset+1)
    retained_f, retained_p = c.features[offset+1, indices], c.price_pairs[offset+1, indices]
    child_times = np.tile([t[-1], t[-1]+1], (B, 1))
    root_candidates = candidate_schedule(m, offset, seed, candidate_count)
    child_candidates = candidate_schedule(m, offset+1, seed, candidate_count)
    fields = {'archived': backup(m, p[-1:], root_p, indices[None], m.values[offset+1],
                               root_candidates, c.factors(offset+1))}
    execution_samples = {}
    for name, child_f, child_p in (('retained_resampled', retained_f, retained_p), ('fresh_history', raw_f, raw_p)):
        future_f, future_p = predict(model, child_f, child_p, child_times, np.ones(B), B, sample_seed+1)
        # The retained law predicts relative moves; it must never pull an
        # actual account's mark back to the retained node's absolute price.
        execution = np.maximum(1., np.rint(root_p[0, :, None]*(future_p/child_p[:, -1, None])/m.rules[9]))*m.rules[9]
        target_f = np.stack((np.repeat(child_f[:, -1], B, axis=0), future_f.reshape(-1, 59)), 1)
        target_p = np.stack((np.repeat(child_p[:, -1], B), future_p.ravel()), 1)
        target_indices = match(target_f, target_p, offset+2).reshape(B, B)
        child_field = backup(m, root_p[0], execution, target_indices, m.values[offset+2],
                             child_candidates, c.factors(offset+2))
        fields[name] = backup(m, p[-1:], root_p, np.arange(B)[None], child_field,
                              root_candidates, execution/root_p[0, :, None])
        execution_samples[name] = execution
    diagnostics = law_errors(execution_samples['fresh_history'], execution_samples['retained_resampled'], root_p[0])
    diagnostics['rootProjectionRMSEBps'] = float(np.sqrt(np.mean(np.log(c.prices[offset+1, indices]/root_p[0])**2))*10000)
    return fields, root_p/p[-1], diagnostics


def root_decision(m, state, consumed, filled, field, factors):
    a, g = m.a, m.grid
    return decide(state, consumed, filled, 0, field, a['wealth'], a['exposures'], a['books'],
        a['book_counts'], a['code_offsets'], a['book_lookup'], a['size_edges'], g.price_bins,
        g.price_scale, m.rules, np.arange(len(a['coordinates']), dtype=np.int64), factors)


def score(m, state, field, factors):
    a, g = m.a, m.grid
    return float(post_value(state, 0, field, a['wealth'], a['exposures'], a['code_offsets'],
        a['book_lookup'], a['size_edges'], g.price_bins, g.price_scale, m.rules, factors))

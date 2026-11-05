"""Stateful, one-second Oracle paper execution using the dense value map.

The controller owns the post-decision account. Each observation applies exactly
one arrival before choosing another post-state. A caller must not execute that
returned account again. This is the finite numerical log-utility planner, not
a certificate of optimality for continuous actions or the true market law.
"""
from dataclasses import dataclass
import time
import numpy as np
from oracle_book_mc import move_arrive, close_terminal, effective, valid_post
from oracle_dense_rollout import roll_market_cloud
from oracle_dense_value_map import DenseValueMap
from oracle_dense_metadata import prepared


@dataclass(frozen=True)
class DenseControllerConfig:
    terminal_seconds: float
    seed: int
    candidates: int
    forecast_origin_index: int = 0
    refresh_every: int = 0
    repair_tolerance: float = 1.e-6
    repair_interval: int = 8
    backend: str = 'cuda'

    def __post_init__(self):
        if not np.isfinite(self.terminal_seconds):
            raise ValueError('A finite terminal timestamp is required')
        for name, minimum in (('seed', 0), ('candidates', 1), ('forecast_origin_index', 0),
                              ('refresh_every', 0), ('repair_interval', 1)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError('Invalid ' + name)
        if not np.isfinite(self.repair_tolerance) or self.repair_tolerance < 0:
            raise ValueError('Invalid repair tolerance')
        if self.backend not in ('cpu', 'cuda'):
            raise ValueError('Explicit CPU or CUDA backend required')


class DenseController:
    def __init__(self, current, model, state, config, *, front=None,
                 episode_origin=None, maximum_horizon=None, predict=None,
                 consumed_quote=0., filled=False):
        self.config = config
        self.model = model
        self.current = current
        self.predict = predict
        origin = float(current.cloud.origin_seconds)
        self.episode_origin = origin if episode_origin is None else float(episode_origin)
        self.maximum_horizon = current.shape[0] if maximum_horizon is None else maximum_horizon
        elapsed = origin - self.episode_origin
        remaining = config.terminal_seconds - origin
        if (not np.isfinite(elapsed) or elapsed < 0 or elapsed != int(elapsed)
                or remaining < 1 or remaining != int(remaining)
                or not isinstance(self.maximum_horizon, int) or self.maximum_horizon < 2
                or current.shape[0] != min(self.maximum_horizon, remaining)):
            raise ValueError('Map, episode and terminal timestamps disagree')
        self.update = int(elapsed)
        self._validate_state(state)
        if (not np.isfinite(consumed_quote) or consumed_quote < 0 or consumed_quote > current.rules[8]
                or not isinstance(filled, bool) or (consumed_quote > 0 and not filled)):
            raise ValueError('Invalid initial arrival quote budget')
        if front is not None and (consumed_quote != 0 or filled):
            raise ValueError('A restored post-decision account has no new arrival')
        if front is None:
            front = self._initial_front(state, consumed_quote, filled)
        elif not np.array_equal(front['state'], state):
            raise ValueError('Restored front and post-decision account disagree')
        if not np.isfinite(front['value']):
            raise ValueError('No feasible initial action under the finite forecast law')
        self._validate_state(front['state'])
        self.front = front
        self.state = front['state']
        self.phase = 'active'
        self.failure = None
        self.terminal_wealth = None
        self.arrival_filled = False

    def _initial_front(self, state, consumed, filled):
        return prepared(self.current).front(state, consumed, filled,
            interior_samples=1, seed=self.config.seed+self.update)

    def _validate_state(self, state):
        s = np.asarray(state)
        if (s.shape != (4 + 3*self.current.grid.max_orders,) or not np.isfinite(s).all()
                or s[2] != self.current.cloud.prices[0, 0] or s[3] <= 0
                or not valid_post(s, self.current.rules, False, False)):
            raise ValueError('Invalid account at the current decision phase')
        for i in range(4, len(s), 3):
            side, price, size = s[i:i+3]
            if (side not in (-1., 0., 1.) or (side == 0 and (price != 0 or size != 0))
                    or (side != 0 and (size <= 0 or price <= 0
                        or abs(price/self.current.rules[9]-np.rint(price/self.current.rules[9])) > 1.e-7))):
                raise ValueError('Invalid pending-order state')

    def _validate_observation(self, features, prices, timestamps):
        f, p, t = map(np.asarray, (features, prices, timestamps))
        old = self.current.cloud
        if (f.shape != (2, 59) or p.shape != (2,) or t.shape != (2,)
                or not np.isfinite(f).all() or not np.isfinite(p).all()
                or not np.isfinite(t).all() or np.any(p <= 0)):
            raise ValueError('Invalid observed market pair')
        if t[0] != old.origin_seconds or t[1] != old.origin_seconds + 1:
            raise ValueError('Expected the next observed second')
        if p[0] != self.state[2] or not np.array_equal(f[0], old.features[0, 0, -1]):
            raise ValueError('Observed pair does not extend the accepted history')
        tick = self.current.rules[9]
        if np.any(np.abs(p/tick - np.rint(p/tick)) > 1.e-7):
            raise ValueError('Observed price is outside the configured tick lattice')
        return f, p, t

    def advance(self, features, prices, timestamps):
        """Accept one real observation; return its execution and planning record.

        Invalid input is rejected before changing state. Once arrival starts,
        a numerical failure stops the controller: retrying must never apply
        funding or fills twice. Recovery uses a verified external checkpoint.
        """
        if self.phase != 'active':
            raise RuntimeError('Controller is ' + self.phase)
        f, p, t = self._validate_observation(features, prices, timestamps)
        started = time.perf_counter()
        self.phase = 'advancing'
        self.arrival_filled = False
        self.update += 1
        try:
            return self._advance(f, p, t, started)
        except Exception as error:
            self.phase = 'failed'
            self.failure = str(error)
            raise

    def _advance(self, f, p, t, started):
        old = self.current
        cfg = self.config
        r = old.rules
        self.state, consumed, filled, valid = move_arrive(self.state, float(p[-1]), r)
        self.arrival_filled = bool(filled)
        if not valid:
            raise RuntimeError('Actual liquidation or invalid aggregate committed fills')
        marked = float(self.state[0] + self.state[1]*self.state[2])
        if t[-1] == cfg.terminal_seconds:
            value = close_terminal(self.state, consumed, r)
            if not np.isfinite(value):
                raise RuntimeError('No valid terminal full close')
            traded = bool(self.state[1] != 0)
            cash = float(self.state[0] + effective(self.state, r))
            self.state = np.r_[cash, 0., p[-1], p[-1], np.zeros(3*old.grid.max_orders)]
            self.terminal_wealth = cash
            self.phase = 'complete'
            return dict(update=self.update, originSeconds=float(t[-1]), markedEquity=marked,
                        terminalWealth=cash, state=self.state.tolist(), filled=bool(filled),
                        marketTrade=traded, stepSeconds=time.perf_counter()-started)
        arrived_quote = float(self.state[0])
        current, front, diagnostics = self._plan(f, p, t, consumed, filled)
        if not np.isfinite(front['value']):
            raise RuntimeError('No feasible action under the finite forecast law')
        self.current, self.front, self.state = current, front, front['state']
        equity = float(self.state[0]+self.state[1]*self.state[2])
        self.phase = 'active'
        return dict(update=self.update, originSeconds=float(t[-1]), price=float(p[-1]),
            markedEquity=equity, exposure=float(self.state[1]*self.state[2]/equity),
            orders=int(np.count_nonzero(self.state[4::3])), filled=bool(filled),
            marketTrade=bool(self.state[0] != arrived_quote), tradeQuote=float(self.state[0]-arrived_quote),
            estimatedLogValue=float(front['value']), state=self.state.tolist(),
            stepSeconds=time.perf_counter()-started, **diagnostics)

    def _plan(self, f, p, t, consumed, filled):
        old, cfg = self.current, self.config
        r = old.rules
        observed = prepared(old).observe(self.front, float(p[-1]), f, p, self.model.feature_std.cpu().numpy())
        remaining = min(self.maximum_horizon, int(cfg.terminal_seconds-t[-1]))
        kwargs = {} if self.predict is None else dict(predict=self.predict)
        cloud, market = roll_market_cloud(old.cloud, self.model, f, p, t,
            cfg.seed + cfg.forecast_origin_index + self.update,
            append_tail=remaining == old.shape[0], **kwargs)
        if len(cloud.counts)-1 != remaining:
            raise AssertionError('Incorrect terminal horizon')
        current = DenseValueMap(old.grid, cloud, r, None, old.errors)
        reset = bool(cfg.refresh_every and self.update % cfg.refresh_every == 0
                     and remaining == self.maximum_horizon)
        run = current.sweep(cfg.seed, cfg.candidates, cloud.destinations.shape[-1], previous=old,
            backend=cfg.backend, estimator='sampled' if reset or cfg.backend == 'cpu' else 'repair',
            repair_tolerance=cfg.repair_tolerance, repair_interval=cfg.repair_interval)
        front = prepared(current).front(self.state, consumed, filled, interior_samples=1, seed=cfg.seed+self.update)
        return current, front, dict(horizon=remaining, sweepSeconds=run['seconds'],
            repairedLayers=run['repairedLayers'], reusedLayers=run['reusedLayers'],
            errorAllowance=run['rootErrorAllowance'], fullRefresh=reset, market=market,
            tdRMS=observed.get('rmsResidual'))

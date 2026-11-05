"""Persistent JSONL paper controller for the NSF dense Bellman planner.

Configuration and checkpoint are fixed for the process lifetime. Initialize
with the last two observed seconds and an already-arrived account, then send
one next observed pair per tick. The returned account is already executed.
There is no exchange API, order-book simulator or future observation input.
"""
import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch
from numba import get_num_threads, set_num_threads
from gpu_oracle_bellman import Rules
from latent_neural_stochastic_flow import load_latent_nsf
from nsf_scenario_graph import build_graph
from oracle_account_grid import AccountGrid
from oracle_book_mc import rule_vector
from oracle_dense_controller import DenseController, DenseControllerConfig
from oracle_dense_grid_config import future_candidate_count
from oracle_dense_value_map import DenseValueMap, MarketCloud
from oracle_incremental_table import TableBudget


@dataclass(frozen=True)
class WorkerSettings:
    grid: AccountGrid
    rules: Rules
    funding: float = 1.
    horizon: int = 3600
    width: int = 8
    branches: int = 8
    candidates: int = 16
    seed: int = 20260916
    forecast_origin_index: int = 0
    refresh_every: int = 60
    repair_tolerance: float = 1.e-5
    repair_interval: int = 8
    backend: str = 'cuda'
    planner: str = 'repair'
    table_budget: TableBudget = TableBudget()
    root_draws: int | None = None

    def __post_init__(self):
        for name, lo in (('horizon', 2), ('width', 1), ('branches', 2)):
            x = getattr(self, name)
            if isinstance(x, bool) or not isinstance(x, int) or x < lo:
                raise ValueError('Invalid ' + name)
        if self.branches % 2:
            raise ValueError('Even conditional branch count required')
        if self.planner not in ('repair','incremental') or not isinstance(self.table_budget,TableBudget):
            raise ValueError('Invalid planner or table budget')
        if self.root_draws is not None and (isinstance(self.root_draws,bool)
                or not isinstance(self.root_draws,int) or not 1<=self.root_draws<=self.branches):
            raise ValueError('Invalid root draw count')
        if self.planner=='incremental' and (self.backend!='cuda' or self.refresh_every!=0
                or self.table_budget.draws>self.branches):
            raise ValueError('Incremental planner requires CUDA, refresh_every=0 and draws<=branches')
        future_candidate_count(self.grid, self.candidates)
        rule_vector(self.rules, self.funding)
        self.controller_config(float(self.horizon))

    def controller_config(self, terminal):
        return DenseControllerConfig(terminal, self.seed, self.candidates,
            self.forecast_origin_index, self.refresh_every, self.repair_tolerance,
            self.repair_interval, self.backend)


def number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value):
        raise ValueError(name + ' must be a finite number')
    return float(value)


def observed_pair(value):
    if not isinstance(value, dict) or set(value) != {'features', 'prices', 'timestamps'}:
        raise ValueError('Observation requires features, prices and timestamps')
    f, p, t = (np.asarray(value[key]) for key in ('features', 'prices', 'timestamps'))
    if (f.shape != (2, 59) or p.shape != (2,) or t.shape != (2,)
            or any(a.dtype.kind not in 'fiu' for a in (f, p, t))
            or not all(np.isfinite(a).all() for a in (f, p, t)) or np.any(p <= 0)
            or t[1]-t[0] != 1 or np.any(np.abs(f) > np.finfo(np.float32).max)):
        raise ValueError('Require two consecutive seconds, positive prices and 59 finite features each')
    return f.astype(np.float32), p.astype(np.float64), t.astype(np.float64)


def account_array(value, price, max_orders):
    if not isinstance(value, dict) or set(value) != {'quote', 'asset', 'previousPrice', 'orders'}:
        raise ValueError('Account requires quote, asset, previousPrice and orders')
    orders = value['orders']
    if not isinstance(orders, list) or len(orders) > max_orders:
        raise ValueError('Too many pending orders')
    s = np.zeros(4+3*max_orders)
    s[:4] = (number(value['quote'], 'quote'), number(value['asset'], 'asset'),
             price, number(value['previousPrice'], 'previousPrice'))
    for j, order in enumerate(orders):
        if not isinstance(order, dict) or set(order) != {'side', 'price', 'notional'}:
            raise ValueError('Order requires side, price and notional')
        if order['side'] not in ('buy', 'sell'):
            raise ValueError('Order side must be buy or sell')
        s[4+3*j:7+3*j] = (1. if order['side'] == 'buy' else -1.,
            number(order['price'], 'order price'), number(order['notional'], 'order notional'))
    return s


def account_json(state):
    return dict(quote=float(state[0]), asset=float(state[1]), price=float(state[2]),
        previousPrice=float(state[3]), orders=[dict(side='buy' if state[i] > 0 else 'sell',
            price=float(state[i+1]), notional=float(state[i+2]))
            for i in range(4, len(state), 3) if state[i] != 0])


def public_diagnostics(diagnostics):
    """Bound wire detail even when support repair crosses a whole horizon.

    Keep aggregate work counts and both ends of the update sequence. The
    controller/backtest still owns the complete record; no planning data change.
    """
    table = diagnostics.get('table')
    if table is None or len(table.get('updates',()))<=32: return diagnostics
    updates = table['updates']
    return dict(diagnostics,table=dict(table,updates=updates[:16]+updates[-16:],
        omittedUpdateRecords=len(updates)-32))


class DenseWorker:
    def __init__(self, model, settings, *, graph_builder=build_graph, predict=None,
                 state_directory=None, model_identity=None):
        self.model, self.settings = model, settings
        self.graph_builder, self.predict = graph_builder, predict
        self.controller = None
        self.failed = False
        self.state_directory = state_directory
        self.model_identity = model_identity
        from oracle_dense_worker_checkpoint import sources
        self.source_identity = sources()

    def process(self, request):
        if not isinstance(request, dict) or not isinstance(request.get('id'), str) or not request['id']:
            raise ValueError('Nonempty string request ID required')
        kind = request.get('kind')
        if kind in ('checkpoint', 'restore'):
            if set(request) != {'id', 'kind'} or self.state_directory is None:
                raise ValueError('Checkpoint request needs a configured state directory and no extra fields')
            from oracle_dense_worker_checkpoint import save_worker, restore_worker
            metadata = (save_worker if kind == 'checkpoint' else restore_worker)(self, self.state_directory)
            c = self.controller
            cloud = c.current.cloud
            return dict(id=request['id'], phase=c.phase, update=c.update,
                timestamp=float(cloud.origin_seconds), terminalSeconds=c.config.terminal_seconds,
                account=account_json(c.state), terminalWealth=None, diagnostics=dict(checkpoint=metadata),
                observation=dict(features=cloud.features[0, 0].tolist(), prices=cloud.price_pairs[0, 0].tolist(),
                    timestamps=[cloud.origin_seconds-1, cloud.origin_seconds]))
        keys = {'id', 'kind', 'observation'}
        if kind == 'initialize':
            keys |= {'account', 'terminalSeconds', 'consumedQuote', 'filled'}
        elif kind != 'tick':
            raise ValueError('Request kind must be initialize or tick')
        if set(request) != keys:
            raise ValueError('Unexpected or missing request fields')
        if self.failed:
            raise RuntimeError('Worker failed; restore a verified checkpoint before continuing')
        f, p, t = observed_pair(request['observation'])
        if kind == 'initialize':
            if self.controller is not None:
                raise ValueError('Worker is already initialized')
            s = self.settings
            terminal = number(request['terminalSeconds'], 'terminalSeconds')
            remaining = terminal-t[-1]
            if remaining < s.horizon or remaining != int(remaining):
                raise ValueError('Terminal timestamp must cover the configured horizon in whole seconds')
            state = account_array(request['account'], p[-1], s.grid.max_orders)
            if state[3] != p[0]:
                raise ValueError('Account previous price differs from the observed pair')
            consumed = number(request['consumedQuote'], 'consumedQuote')
            filled = request['filled']
            if not isinstance(filled, bool) or consumed < 0 or consumed > s.rules.maximum or (consumed > 0 and not filled):
                raise ValueError('Invalid arrival quote budget')
            tick = s.rules.tick
            if np.any(np.abs(p/tick-np.rint(p/tick)) > 1.e-7):
                raise ValueError('Observed price is outside the configured tick lattice')
            # Build from this origin's observations only, never saved later forecasts.
            try:
                graph = self.graph_builder(self.model, f, p, t, list(range(1, s.horizon+1)),
                    width=s.width, samples=s.branches, seed=s.seed)
                cloud = MarketCloud.from_graph(graph)
                current = DenseValueMap(s.grid, cloud, rule_vector(s.rules, s.funding), None)
                run = current.sweep(s.seed, s.candidates, s.branches, backend=s.backend,
                                    estimator='sampled' if s.backend == 'cpu' else 'repair')
                controller = DenseController; options = {}
                if s.planner=='incremental':
                    from oracle_incremental_controller import IncrementalController
                    controller = IncrementalController
                    options = dict(budget=s.table_budget,root_draws=s.root_draws)
                self.controller = controller(current, self.model, state, s.controller_config(terminal),
                    predict=self.predict, consumed_quote=consumed, filled=filled,**options)
            except Exception:
                self.failed = True
                raise
            diagnostics = dict(initialSweepSeconds=run['seconds'], horizon=s.horizon,planner=s.planner)
        else:
            if self.controller is None:
                raise ValueError('Initialize before sending ticks')
            try:
                diagnostics = self.controller.advance(f, p, t)
            except Exception:
                self.failed = self.controller.phase == 'failed'
                raise
            diagnostics = public_diagnostics({k: v for k, v in diagnostics.items() if k != 'state'})
        c = self.controller
        return dict(id=request['id'], phase=c.phase, update=c.update,
            timestamp=float(t[-1]), account=account_json(c.state), diagnostics=diagnostics,
            terminalWealth=c.terminal_wealth,
            interpretation='Executed Oracle paper account; finite-grid log-utility planner')


def process_line(worker, line):
    request_id = None
    try:
        def reject(value):
            raise ValueError('Nonfinite JSON constant: ' + value)
        request = json.loads(line, parse_constant=reject)
        if isinstance(request, dict) and isinstance(request.get('id'), str):
            request_id = request['id']
        response = worker.process(request)
        json.dumps(response, allow_nan=False)
        return response
    except Exception as error:
        return dict(id=request_id, error=dict(message=str(error),
            failed=worker.failed or (worker.controller is not None and worker.controller.phase == 'failed')))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=Path, required=True)
    ap.add_argument('--state-directory', type=Path)
    args = ap.parse_args()
    config = json.loads(args.config.read_text())
    checkpoint = Path(config.pop('checkpoint'))
    digest = config.pop('checkpointSha256')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != digest:
        raise ValueError('Checkpoint hash mismatch')
    grid = config.pop('grid')
    for name in ('exposure_points', 'wealth_points'):
        if name in grid:
            grid[name] = tuple(grid[name])
    if 'table_budget' in config: config['table_budget'] = TableBudget(**config['table_budget'])
    settings = WorkerSettings(AccountGrid(**grid), Rules(**config.pop('rules')), **config)
    torch.set_num_threads(2)
    set_num_threads(min(4, get_num_threads()))
    torch.set_float32_matmul_precision('high')
    model, _ = load_latent_nsf(checkpoint, settings.backend)
    worker = DenseWorker(model, settings, state_directory=args.state_directory, model_identity=digest)
    while True:
        line = sys.stdin.readline(65537)
        if not line:
            break
        if len(line) > 65536:
            raise ValueError('Request exceeds 64 KiB')
        print(json.dumps(process_line(worker, line), allow_nan=False), flush=True)


if __name__ == '__main__':
    main()

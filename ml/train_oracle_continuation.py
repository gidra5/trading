"""Fitted finite-horizon log Bellman iteration against frozen one-second NSF."""
import argparse
import copy
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from gpu_oracle_bellman import Rules, Menu
from latent_neural_stochastic_flow import load_latent_nsf
from train_latent_neural_stochastic_flow import latent_data
from target_horizon_forecast import HorizonRange
from oracle_continuation_value import ContinuationValue, observed_inputs, conditional_children, bellman_backup, save_value


def accounts(prices, rng, device):
    p = prices[:, -1].cpu().numpy(); n = len(p)
    wealth = np.exp(rng.uniform(np.log(2500), np.log(20000), n))
    e = rng.uniform(-2.5, 2.5, n); e[rng.random(n) < .25] = 0
    s = np.zeros((n, 9)); s[:, 0] = wealth*(1-e); s[:, 1] = wealth*e/p
    s[:, 2] = p; s[:, 3] = prices[:, -2].cpu().numpy(); s[:, 8] = 1
    pending = (rng.random(n) < .25) & (abs(e) < 1.5)
    s[:, 4] = pending*rng.choice([-1, 1], n)
    s[:, 5] = np.where(pending, (np.round(p/.01)-s[:, 4]*rng.choice([1, 5, 100], n))*.01, 0)
    s[:, 6] = np.where(pending, np.minimum(wealth*.1, 10000), 0)
    return torch.as_tensor(s, dtype=torch.float64, device=device)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--samples', type=int, default=8)
    ap.add_argument('--action-step', type=float, default=.01)
    ap.add_argument('--validate-every', type=int, default=100)
    args = ap.parse_args()
    if args.output.exists(): raise ValueError('Experiment already exists')
    args.output.mkdir(parents=True)
    device = torch.device('cuda'); torch.set_num_threads(4); torch.set_float32_matmul_precision('high')
    torch.manual_seed(20260918); rng = np.random.default_rng(20260918)
    checkpoint = Path('data/backtests/oracle-latent-nsf-sampled-skips/latentSampledSkipFlow.pt')
    source = json.loads((checkpoint.parent/'plan.json').read_text())
    nsf, _ = load_latent_nsf(checkpoint, device); nsf.requires_grad_(False)
    train, validation = [latent_data(Path(source['nativeDirectory']), Path(source['dataset']), split, HorizonRange(21600., 1)) for split in ['train', 'validation']]
    value = ContinuationValue(nsf.feature_mean, nsf.feature_std).to(device)
    target = copy.deepcopy(value).eval().requires_grad_(False)
    menu = Menu(step=args.action_step)
    optimizer = torch.optim.AdamW(value.parameters(), lr=3e-4, weight_decay=1e-5)
    sources = ['ml/oracle_continuation_value.py', 'ml/train_oracle_continuation.py', 'ml/gpu_oracle_bellman.py', 'ml/export_nsf_planning_trees.py', 'ml/latent_neural_stochastic_flow.py']
    metadata = dict(gamma=1, horizonSeconds=3600, forecastStepSeconds=1, trainSteps=args.steps, batchSize=args.batch,
        trainConditionalSamples=args.samples, actionExposureStep=args.action_step, limitSlots=1, limitNotionalFractions=[.25], limitOffsetsTicks=[1],
        jointMarketPlacement=False, rules=value.rules.__dict__, checkpoint=str(checkpoint),
        checkpointSha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        sourceHashes={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in sources},
        training='one-second sampled Bellman targets with EMA target network, real train origins and recursive synthetic observations',
        selection='minimum mean absolute one-step Bellman residual on fixed validation scenarios; NOT a uniform value-error certificate',
        terminal='actual valid flat close, log ruin is negative infinity; nonfinite targets excluded and counted',
        testUsedForTrainingOrSelection=False, uniformResidualCertificate=None, continuousMenuCertificate=None)
    (args.output/'plan.json').write_text(json.dumps(metadata, indent=2))
    (args.output/'source').mkdir()
    for path in sources: (args.output/'source'/Path(path).name).write_bytes(Path(path).read_bytes())
    vf, vp, vt = observed_inputs(validation, np.linspace(0, len(validation['origins'])-1, 96, dtype=int), device)
    vs = accounts(vp, np.random.default_rng(20260919), device)
    vh = torch.tensor(np.tile([1, 2, 8, 60, 600, 3600], 16), device=device)
    vc = conditional_children(nsf, vf, vp, vt, 32, 20260919)
    f, p, t = observed_inputs(train, rng.integers(0, len(train['origins']), args.batch), device)
    started = time.perf_counter(); history = []; best = float('inf'); best_step = 0
    skipped = 0
    for step in range(args.steps+1):
        loss_number = None
        if step:
            # Half the batch repeatedly restarts at actual training origins;
            # the other half follows uninterrupted synthetic one-second paths.
            fresh = observed_inputs(train, rng.integers(0, len(train['origins']), args.batch//2), device)
            f, p, t = [torch.cat((new, old[len(new):]), 0) for new, old in zip(fresh, (f, p, t))]
            state = accounts(p, rng, device)
            h = rng.integers(1, 3601, args.batch); h[:args.batch//2] = rng.integers(1, 65, args.batch//2)
            horizon = torch.as_tensor(h, device=device)
            children = conditional_children(nsf, f, p, t, args.samples, 20260920+step)
            target_value = bellman_backup(target, state, *children, horizon, menu)['value']
            predicted = value(f, p, t, state, horizon)
            finite = torch.isfinite(target_value) & torch.isfinite(predicted)
            skipped += int((~finite).sum())
            if finite.any():
                difference = (predicted[finite]-target_value[finite])*1000
                loss = torch.nn.functional.smooth_l1_loss(difference, torch.zeros_like(difference))
                optimizer.zero_grad(set_to_none=True); loss.backward()
                torch.nn.utils.clip_grad_norm_(value.parameters(), 5); optimizer.step()
                loss_number = float(loss.detach())
                with torch.no_grad():
                    for dst, src in zip(target.parameters(), value.parameters()): dst.lerp_(src, .02)
            branch = step % args.samples
            f, p, t = [x[:, branch].detach() for x in children]
        if step % args.validate_every == 0 or step == args.steps:
            with torch.no_grad():
                exact_backup = bellman_backup(value, vs, *vc, vh, menu)['value']
                prediction = value(vf, vp, vt, vs, vh)
                finite = torch.isfinite(exact_backup) & torch.isfinite(prediction)
                residual = (prediction[finite]-exact_backup[finite]).abs()
                score = float(residual.mean())
                record = dict(step=step, loss=loss_number, validationMeanAbsResidual=score,
                    validationMaxAbsResidual=float(residual.max()), validationFinite=int(finite.sum()),
                    excludedTrainingTargets=skipped, seconds=time.perf_counter()-started)
            if score < best:
                best, best_step = score, step
                save_value(args.output/'value.pt', value, dict(**metadata, selectedStep=step, validation=record))
            history.append(record)
            (args.output/'history.json').write_text(json.dumps(history, indent=2))
            print(json.dumps(record), flush=True)
    save_value(args.output/'last.pt', value, dict(**metadata, selectedStep=args.steps))
    (args.output/'selection.json').write_text(json.dumps(dict(step=best_step, validationMeanAbsResidual=best,
        seconds=time.perf_counter()-started, checkpointSha256=hashlib.sha256((args.output/'value.pt').read_bytes()).hexdigest()), indent=2))


if __name__ == '__main__': main()

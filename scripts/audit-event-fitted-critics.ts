/** Prior-only test of separate action selection/evaluation and critic disagreement. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { chooseEventTrade, eventHolding, restoreEventPolicy, type EventAccount } from "../packages/bot-algo/src/event-log-policy.js";
import { fittedEventHolding, predictEventHoldingGrid, trainEventFittedValue, type EventValueTransition } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";
import { eventSecondDynamicsAt, loadEventSecondDynamics } from "./event-second-dynamics.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000, BLOCK = 14 * DAY;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify a single-setting fitted screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json");
if (config.contract !== "event-fitted-value-screen-v1" || config.settings.length !== 1) throw new Error("Requires one frozen forecast setting");
const setting = config.settings[0], name = eventFittedSettingName(setting), phases = config.phases;
const jc = read(config.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const fits = phases.map((phase: any) => read(jc.source, `${jc.window.id}-${phase.id}-model.json`));
const c = loadEventCandles(fits[0].trainStart - DAY, phases.at(-1).endTime);
assert.ok(c.at(-1)!.openTime + 60000 <= jc.window.startTime);
const external = loadEventFuturesRows(fits[0].trainStart - DAY, phases.at(-1).endTime);
const seconds = setting.secondDynamics ? loadEventSecondDynamics(fits[0].trainStart - DAY, phases.at(-1).endTime) : undefined;
const inputCache = new Map<number, number[] | null>();
const inputs = (i: number): number[] | null => {
  if (inputCache.has(i)) return inputCache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t));
  const d = ["deviation", "centered"].includes(setting.basis) || setting.historyMinutes === 240
    ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  const shape = setting.candleShape ? eventCandleShapes(c, i, t => external.rows.get(t)) : [];
  const values = e && d && shape ? [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(setting.basis, e, d), ...(setting.secondDynamics ? eventSecondDynamicsAt(seconds!.rows, c[i]) : []), ...shape] : null;
  inputCache.set(i, values); return values;
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-fitted-critic-audit-v1", source, setting, phases,
  blockDays: 14, sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(external.fingerprint).update(seconds?.fingerprint ?? "").digest("hex"),
  split: "UTC-epoch anchored alternating 14-day blocks. Retain a transition only when its entire 1440-minute feature support through next-event close lies inside one block. Train a pooled control on the same retained rows.",
  evaluation: "One-event values only. Each critic chooses an action; both itself and the other critic value that same feasible trade. Compare both with observed holding plus terminal settlement, including entry fees, borrowing and liquidation. Common accounts use equity 10000 and exposures -L,0,+L.",
  caveat: "These separately fitted historical blocks are not statistically independent market samples. This is a selection/evaluation bias diagnostic, not Double Q-learning, a multi-event payoff or a strategy backtest. Reused prior origins only; no final outcomes loaded." }, null, 2));
const files = ["scripts/audit-event-fitted-critics.ts", "scripts/research-event-policy.ts", "scripts/event-futures-basis.ts", "scripts/event-second-dynamics.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const mean = (xs: number[]) => xs.reduce((sum, x) => sum + x, 0) / Math.max(1, xs.length);
const errors = (pairs: Array<{ predicted: number; actual: number }>) => ({ samples: pairs.length,
  biasBps: mean(pairs.map(p => p.predicted - p.actual)) * 10000,
  mseBpsSquared: mean(pairs.map(p => (p.predicted - p.actual) ** 2)) * 1e8 });
const results = [], started = performance.now();
for (const [index, phase] of phases.entries()) {
  const begin = performance.now(), base = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
  const allTrain = makeSamples(c, sc.clock, fits[index].trainStart, fits[index].trainEnd, [jc.window], sc.stride, "chain", sc.featureNames);
  const allTest = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames);
  const train = allTrain.filter(r => inputs(r.start) && inputs(r.end)), test = allTest.filter(r => inputs(r.start) && inputs(r.end));
  if (train.length / allTrain.length < 0.98 || test.length !== allTest.length) throw new Error("Incomplete matched observations");
  const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, base.model.clock));
  const transition = (r: MoveSample): EventValueTransition => ({ features: inputs(r.start)!, nextFeatures: inputs(r.end)!, leaf: state(r.start), nextLeaf: state(r.end),
    move: { return: r.return, low: r.low, high: r.high, duration: r.duration } });
  const original = read(source, `${phase.id}-${name}-policy.json`);
  // Resume at the existing first depth solely to validate the full transition
  // identity against the saved model. No backup is recomputed for this control.
  const full = trainEventFittedValue(base, train.map(transition), setting.penalty, 1, setting.boost, { ...original, tables: original.tables.slice(0, 1) });
  const assignments = train.map(r => {
    const supportStart = c[r.start - 1440].openTime, supportEnd = c[r.end].openTime + 60000;
    assert.ok(supportEnd < phase.startTime);
    const block = Math.floor(supportStart / BLOCK), last = Math.floor((supportEnd - 1) / BLOCK);
    return { row: r, supportStart, supportEnd, block, fold: block === last ? ((block % 2) + 2) % 2 : null };
  });
  const retained = assignments.filter(r => r.fold !== null), grouped = [0, 1].map(f => retained.filter(r => r.fold === f));
  if (grouped.some(rows => rows.length < 100)) throw new Error("Too few rows per critic");
  for (const rows of grouped) for (const row of rows) {
    assert.ok(row.supportStart >= row.block * BLOCK && row.supportEnd <= (row.block + 1) * BLOCK);
  }
  const pooled = trainEventFittedValue(base, retained.map(r => transition(r.row)), setting.penalty, 1, setting.boost);
  const critics = grouped.map(rows => trainEventFittedValue(base, rows.map(r => transition(r.row)), setting.penalty, 1, setting.boost));
  const policies = [full, pooled, ...critics], labels = ["full", "pooled", "A", "B", "mean"];
  for (const [i, policy] of policies.entries()) fs.writeFileSync(path.join(output, `${phase.id}-${labels[i]}-policy.json`), JSON.stringify(policy));
  fs.writeFileSync(path.join(output, `${phase.id}-assignments.json`), JSON.stringify(assignments.map(({ row, ...r }) => ({ ...r, time: c[row.start].openTime + 60000 }))));
  const holding: Array<{ model: string; predicted: number; actual: number }> = [], decisions: any[] = [];
  const fee = (base.costs.feeBps + base.costs.slippageBps) / 10000;
  const actualValue = (before: EventAccount, after: EventAccount, sample: MoveSample) => {
    const h = eventHolding(after.exposure, sample, base.costs);
    const value = h.liquidated ? -Infinity : Math.log(after.equity / before.equity) + Math.log(h.factor) + Math.log(1 - Math.abs(h.exposure) * fee);
    if (!Number.isFinite(value)) throw new Error("A diagnostic action has non-finite realized utility");
    return value;
  };
  for (const row of test) {
    const time = c[row.start].openTime + 60000, features = inputs(row.start)!, leaf = state(row.start);
    const grids = policies.map(p => predictEventHoldingGrid(p, features, 1));
    grids.push(grids[2].map((v, i) => (v + grids[3][i]) / 2));
    for (const side of [-1, 0, 1]) {
      const account = { equity: 10000, price: c[row.start].close, exposure: side * base.costs.maxLeverage };
      if (side) {
        const actual = actualValue(account, account, row);
        for (const [i, grid] of grids.entries()) holding.push({ model: labels[i], predicted: fittedEventHolding(full, grid, account, leaf), actual });
        holding.push({ model: "zero", predicted: Math.log(1 - Math.abs(account.exposure) * fee), actual });
      }
      const trades = grids.map(grid => chooseEventTrade(full, account, after => fittedEventHolding(full, grid, after, leaf)));
      for (const [i, trade] of trades.entries()) {
        const actual = actualValue(account, trade, row);
        const other = i === 2 ? 3 : i === 3 ? 2 : i;
        const cross = Math.log(trade.equity / account.equity) + fittedEventHolding(full, grids[other], trade, leaf);
        if (![trade.value, cross, actual].every(Number.isFinite)) throw new Error("Non-finite critic evaluation");
        decisions.push({ time, availableAt: c[row.end].openTime + 60000, side, model: labels[i], exposure: trade.exposure, quantity: trade.quantity,
          self: trade.value, cross, actual, oppositeExposure: trades[other].exposure });
      }
    }
  }
  fs.writeFileSync(path.join(output, `${phase.id}-decisions.json`), JSON.stringify(decisions));
  const forecast = labels.concat("zero").map(model => ({ model, ...errors(holding.filter(r => r.model === model)) }));
  const actionEvaluation = [-1, 0, 1].map(side => {
    const rows = decisions.filter(r => r.side === side && ["A", "B"].includes(r.model));
    const entries = rows.filter(r => Math.abs(r.exposure) > 1e-4);
    return { side, self: errors(rows.map(r => ({ predicted: r.self, actual: r.actual }))),
      cross: errors(rows.map(r => ({ predicted: r.cross, actual: r.actual }))),
      selfMinusCrossBps: mean(rows.map(r => r.self - r.cross)) * 10000,
      changedExposureFraction: mean(rows.map(r => Number(Math.abs(r.exposure - r.oppositeExposure) > 1e-4))),
      ...(side === 0 ? { entryDecisions: entries.length, crossRejectsEntry: entries.filter(r => r.cross <= 0).length,
        entrySelf: errors(entries.map(r => ({ predicted: r.self, actual: r.actual }))), entryCross: errors(entries.map(r => ({ predicted: r.cross, actual: r.actual }))) } : {}) };
  });
  const policyProbes = labels.map(model => {
    const rows = decisions.filter(r => r.model === model && r.side === 0), active = rows.filter(r => Math.abs(r.exposure) > 1e-4);
    return { model, entries: active.length, meanRealizedEntryBps: mean(active.map(r => r.actual)) * 10000,
      sumRealizedUtility: active.reduce((s, r) => s + r.actual, 0), caveat: "Independent fixed-account one-event settlement probes; not a continuous execution backtest." };
  });
  const result = { phase, training: train.length, retained: retained.length, folds: grouped.map(rows => rows.length), validation: test.length,
    forecast, actionEvaluation, policyProbes, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, `${phase.id}-summary.json`), JSON.stringify(result, null, 2));
  console.log(JSON.stringify(result));
}
const result = { results, elapsedSec: (performance.now() - started) / 1000 };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
console.log(JSON.stringify({ elapsedSec: result.elapsedSec }));

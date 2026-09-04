/** Prior-origin diagnostics for depth stability and out-of-sample Bellman residuals. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, observeMove } from "../packages/bot-algo/src/event-distribution.js";
import { chooseEventTrade, eventHolding, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { fittedEventHolding, predictEventHoldingGrid, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { loadEventCandles } from "./research-event-policy.js";
import { eventSecondDynamicsAt, loadEventSecondDynamics } from "./event-second-dynamics.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify fitted screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json");
if (config.contract !== "event-fitted-value-screen-v1" || config.settings.length !== 1) throw new Error("Requires a single-setting prior-origin depth screen");
const setting = config.settings[0], name = eventFittedSettingName(setting), phases = config.phases;
const jc = read(config.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const c = loadEventCandles(phases[0].startTime - 2 * DAY, phases.at(-1).endTime), external = loadEventFuturesRows(phases[0].startTime - 2 * DAY, phases.at(-1).endTime);
const seconds = setting.secondDynamics ? loadEventSecondDynamics(phases[0].startTime - 2 * DAY, phases.at(-1).endTime) : undefined;
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i])), inputCache = new Map<number, number[]>();
const inputs = (i: number): number[] => {
  if (inputCache.has(i)) return inputCache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t)); if (!e) throw new Error("Missing completed audit input");
  const d = ["deviation", "centered"].includes(setting.basis) ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  if (!d) throw new Error("Missing completed basis deviation");
  const shape = setting.candleShape ? eventCandleShapes(c, i, t => external.rows.get(t)) : [];
  if (!shape) throw new Error("Missing completed candle shape");
  const values = [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(setting.basis, e, d),
    ...(setting.secondDynamics ? eventSecondDynamicsAt(seconds!.rows, c[i]) : []), ...shape];
  inputCache.set(i, values); return values;
};
fs.mkdirSync(output, { recursive: true });
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(external.fingerprint);
for (const phase of phases) hash.update(fs.readFileSync(path.join(source, `${phase.id}-${name}-policy.json`)));
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, setting, phases, sourceHash: hash.digest("hex"),
  probes: "At every saved decision time use common cash, -L and +L accounts at equity 10000 and observed price. This prevents endogenous exposure from changing the comparison population.",
  residual: "H_d minus [observed log holding factor + the previous fitted optimal value at the observed next features and marked account]. This is a Bellman residual, not an error against a realized multi-event payoff.",
  caveat: "Prior research origins only. Changes and value decreases diagnose approximation behavior, not a convergence proof. No final inspector outcome is loaded. Terminal settlement and bounded account interpolation differ from an exact infinite-horizon operator." }, null, 2));
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename));
const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / Math.max(1, v.length);
const statistics = (v: number[]) => ({ count: v.length, mean: mean(v), minimum: Math.min(...v), maximum: Math.max(...v) });
const results = [], started = performance.now();
for (const phase of phases) {
  const p: FittedEventValue = read(source, `${phase.id}-${name}-policy.json`), base = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
  const trace = read(source, `${phase.id}-${name}-d1-trades.json`);
  const rows: Array<{ depth: number; side: number; time: number; exposure: number; value: number; holding: number; valueIncrement: number | null; holdingIncrement: number | null;
    exposureChange: number | null; residual: number | null; entryAdvantageBps: number | null }> = [];
  for (const raw of trace) {
    const i = byTime.get(raw.time); if (i === undefined) throw new Error("Missing audit decision candle");
    const leaf = eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, base.model.clock));
    const grids = p.tables.map(t => predictEventHoldingGrid(p, inputs(i), t.depth));
    const move = observeMove(c, i, base.model.clock, base.model.featureNames);
    const complete = move && c[move.end].openTime + 60000 < phase.endTime;
    const nextLeaf = complete ? eventLeaf(base.model, eventFeatures(c, move.end, base.model.featureNames, base.model.clock)) : 0;
    const nextGrids = complete ? p.tables.slice(0, -1).map(t => predictEventHoldingGrid(p, inputs(move.end), t.depth)) : [];
    for (const side of [-1, 0, 1]) {
      const account = { equity: 10000, price: c[i].close, exposure: side * p.costs.maxLeverage };
      let previous: (typeof rows)[number] | undefined;
      for (const [d, grid] of grids.entries()) {
        const decision = chooseEventTrade(p, account, a => fittedEventHolding(p, grid, a, leaf)), holding = fittedEventHolding(p, grid, account, leaf);
        let residual: number | null = null;
        if (complete) {
          const h = eventHolding(account.exposure, move, p.costs);
          const next = { equity: account.equity * h.factor, price: account.price * (1 + move.return), exposure: h.exposure };
          const continuation = d === 0 ? Math.log(1 - Math.abs(h.exposure) * (p.costs.feeBps + p.costs.slippageBps) / 10000)
            : chooseEventTrade(p, next, a => fittedEventHolding(p, nextGrids[d - 1], a, nextLeaf)).value;
          if (!h.liquidated && Number.isFinite(holding) && Number.isFinite(continuation)) residual = holding - Math.log(h.factor) - continuation;
        }
        const row: (typeof rows)[number] = { depth: d + 1, side, time: raw.time, exposure: decision.exposure, value: decision.value, holding,
          valueIncrement: previous ? decision.value - previous.value : null, holdingIncrement: previous ? holding - previous.holding : null,
          exposureChange: previous ? Math.abs(decision.exposure - previous.exposure) : null, residual,
          entryAdvantageBps: side === 0 && decision.quantity ? (decision.value - holding) * 10000 : null };
        rows.push(row); previous = row;
      }
    }
  }
  const summary = p.tables.map(t => {
    const selected = rows.filter(r => r.depth === t.depth), cash = selected.filter(r => r.side === 0), changed = selected.filter(r => r.exposureChange !== null);
    return { depth: t.depth, cashEntries: cash.filter(r => r.entryAdvantageBps !== null).length,
      cashLongEntries: cash.filter(r => r.exposure > 1e-8).length, cashShortEntries: cash.filter(r => r.exposure < -1e-8).length,
      entryAdvantageBps: statistics(cash.flatMap(r => r.entryAdvantageBps === null ? [] : [r.entryAdvantageBps])),
      changedExposureFraction: changed.length ? changed.filter(r => r.exposureChange! > 1e-4).length / changed.length : null,
      valueIncrementBps: statistics(changed.map(r => r.valueIncrement! * 10000)),
      holdingDecreaseFraction: changed.length ? changed.filter(r => r.holdingIncrement! < -1e-10).length / changed.length : null,
      residuals: [-1, 0, 1].map(side => { const errors = selected.filter(r => r.side === side && r.residual !== null).map(r => r.residual!);
        return { side, samples: errors.length, biasBps: mean(errors) * 10000, mseBpsSquared: mean(errors.map(e => e * e)) * 1e8 }; }) };
  });
  fs.writeFileSync(path.join(output, `${phase.id}-rows.json`), JSON.stringify(rows));
  results.push({ phase, decisions: trace.length, rows: summary });
  console.log(JSON.stringify({ phase: phase.id, rows: summary }));
}
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));

/** Post-selection diagnosis: distinguish refit changes from changes in market inputs. */
import fs from "node:fs";
import path from "node:path";
import { eventFeatures, eventLeaf, observeMove } from "../packages/bot-algo/src/event-distribution.js";
import { eventHolding, eventTrade, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { fittedEventHolding, predictEventHoldingGrid, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";
import { eventSecondDynamicsAt, loadEventSecondDynamics } from "./event-second-dynamics.js";
const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify fitted-policy comparison and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), selection = read(source, "selection.json"), chosen = selection.diagnosticChoice;
if (config.contract !== "event-fitted-value-policy-v1") throw new Error("Requires fitted policy source");
const name = chosen.choice.slice("fitted-value-".length);
const screen = [...config.screens].reverse().find((dir: string) => {
  const config = read(dir, "config.json");
  return config.depths >= chosen.depth && fs.existsSync(path.join(dir, `${config.phases[0].id}-${name}-policy.json`));
});
if (!screen) throw new Error("Missing selected fitted calibration policy");
const phases = read(screen, "config.json").phases, setting = config.settings[chosen.choice];
const jc = read(config.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const c = loadEventCandles(phases[0].startTime - 2 * DAY, config.window.endTime + DAY), external = loadEventFuturesRows(phases[0].startTime - 2 * DAY, config.window.endTime);
const seconds = setting.secondDynamics ? loadEventSecondDynamics(phases[0].startTime - 2 * DAY, config.window.endTime) : undefined;
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i]));
const inputs = (i: number) => {
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t)); if (!e) throw new Error("Missing audit input");
  const deviation = ["deviation", "centered"].includes(setting.basis) ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  if (!deviation) throw new Error("Missing completed basis deviation");
  const shape = setting.candleShape ? eventCandleShapes(c, i, t => external.rows.get(t)) : [];
  if (!shape) throw new Error("Missing completed candle shape");
  return [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(setting.basis, e, deviation),
    ...(setting.secondDynamics ? eventSecondDynamicsAt(seconds!.rows, c[i]) : []), ...shape];
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, screen, chosen, setting,
  caveat: "Post-selection diagnosis only. The older fit on the final window is a counterfactual, not a selection candidate. One-event forecasts include terminal fees; deeper fitted values are not realized multi-event returns." }, null, 2));
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename));
const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / Math.max(1, v.length);
const range = (v: number[]) => ({ mean: mean(v), min: Math.min(...v), max: Math.max(...v) });
const results = [];
const cases = [...phases.map((phase: any) => ({ phase, fit: phase.id, id: phase.id })),
  { phase: { ...config.window, id: "final" }, fit: "final", id: "final" },
  { phase: { ...config.window, id: "final" }, fit: phases.at(-1).id, id: "final-with-previous-fit" }];
for (const entry of cases) {
  const base = restoreEventPolicy(read(config.source, `${entry.fit}-policy.json`));
  const policy: FittedEventValue = entry.fit === "final" ? read(source, "final-fitted-policy.json")
    : read(screen, `${entry.fit}-${name}-policy.json`);
  const control = replayEventPolicy(c, base, entry.phase.startTime, entry.phase.endTime, 1, { trace: true });
  const observations = new Map(control.trace.map((r: any) => [r.time, { availableAt: r.time, values: inputs(byTime.get(r.time)!) }]));
  const replay = replayEventPolicy(c, base, entry.phase.startTime, entry.phase.endTime, chosen.depth, { fitted: { policy, observations }, trace: true });
  const audit = replay.trace.map((raw: any) => {
    const i = byTime.get(raw.time)!, features = observations.get(raw.time)!.values, leaf = eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, base.model.clock));
    const account = { equity: 10000, price: c[i].close, exposure: 0 }, move = observeMove(c, i, base.model.clock, base.model.featureNames)!;
    const grids = [1, chosen.depth].map(d => predictEventHoldingGrid(policy, features, d));
    const cash = grids.map(g => fittedEventHolding(policy, g, account, leaf));
    const probes = [-1, 1].map(side => {
      const trade = eventTrade(account, side * base.costs.maxLeverage, base.costs)!;
      const values = grids.map((g, d) => Math.log(trade.equity / account.equity) + fittedEventHolding(policy, g, trade, leaf) - cash[d]);
      const held = eventHolding(trade.exposure, move, base.costs);
      const actual = Math.log(trade.equity / account.equity) + Math.log(held.factor) + Math.log(1 - Math.abs(held.exposure) * (base.costs.feeBps + base.costs.slippageBps) / 10000);
      return { side, oneEventAdvantageBps: values[0] * 10000, deeperAdvantageBps: values[1] * 10000,
        actualOneEventBps: c[move.end].openTime + 60000 < entry.phase.endTime && !held.liquidated ? actual * 10000 : null };
    });
    return { time: raw.time, equityBefore: raw.equityBefore, equityAfter: raw.equityAfter, price: account.price,
      features, clippedInputs: features.map((v, f) => Math.abs((v - policy.means[f]) / policy.scales[f]) > 5),
      cashValueBps: cash[1] * 10000, probes, executedQuantity: raw.orderQuantity, proposedQuantity: raw.order.quantity,
      exposureBefore: raw.exposureBefore, exposureAfter: raw.exposureAfter, realizedReturnBps: raw.realizedReturnBps };
  });
  const valid = audit.filter(r => r.probes.every(p => p.actualOneEventBps !== null));
  const mse = mean(valid.flatMap(r => r.probes.map(p => (p.actualOneEventBps! - p.oneEventAdvantageBps) ** 2)));
  const { trace, ...metrics } = replay;
  const result = { id: entry.id, fit: entry.fit, ...metrics, oneEventMseBpsSquared: mse,
    longOneEventBps: range(audit.map(r => r.probes[1].oneEventAdvantageBps)), shortOneEventBps: range(audit.map(r => r.probes[0].oneEventAdvantageBps)),
    longDeeperBps: range(audit.map(r => r.probes[1].deeperAdvantageBps)), shortDeeperBps: range(audit.map(r => r.probes[0].deeperAdvantageBps)),
    cashValueBps: range(audit.map(r => r.cashValueBps)),
    externalFeatures: Array.from({ length: Math.max(0, policy.means.length - 23) }, (_, i) => i + 23).map(f => ({ index: f, trainingMean: policy.means[f], trainingScale: policy.scales[f],
      evaluation: range(audit.map(r => r.features[f])), clipped: audit.filter(r => r.clippedInputs[f]).length })),
    tradesAudit: audit.filter(r => r.executedQuantity), canceledAudit: audit.filter(r => r.proposedQuantity && !r.executedQuantity) };
  results.push(result); fs.writeFileSync(path.join(output, `${entry.id}-events.json`), JSON.stringify(audit));
  fs.writeFileSync(path.join(output, `${entry.id}-trades.json`), JSON.stringify(trace));
  console.log(JSON.stringify({ ...result, daily: undefined, tradesAudit: undefined, canceledAudit: undefined }));
}
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));

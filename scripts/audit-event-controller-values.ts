/** Paired out-of-fit value errors for hold and sampled two-event controllers. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy, type EventAccount } from "../packages/bot-algo/src/event-log-policy.js";
import { decideFittedEventControllers, fittedEventHolding, predictEventHoldingGrid, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventControllerHolding, eventSignHorizonPaths } from "./event-paths.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000, seed = 20260904, replicates = 2000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify composed screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json");
assert.equal(config.contract, "event-controller-composition-screen-v1");
const hc = read(config.source, "config.json"), pc = read(hc.source, "config.json"), joint = pc.source;
const jc = read(joint, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const setting = config.sampledSetting, phases = config.phases;
assert.deepEqual(phases, hc.phases);
const c = loadEventCandles(phases[0].startTime - 2 * DAY, phases.at(-1).endTime);
const external = loadEventFuturesRows(phases[0].startTime - 2 * DAY, phases.at(-1).endTime), cache = new Map<number, number[]>();
const inputs = (i: number) => {
  if (cache.has(i)) return cache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t)), d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  if (!e || !d) throw new Error("Missing completed controller input");
  const values = [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i), ...eventFittedFuturesInputs(setting.basis, e, d)];
  cache.set(i, values); return values;
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-controller-value-audit-v1", source, phases, setting, seed, replicates,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(JSON.stringify(c)).update(external.fingerprint).digest("hex"),
  method: "On every complete prior-window two-event path, compare predicted sampled-minus-hold H2 value with the observed difference. Both controllers receive the same post-order account and first event. At the intermediate observed state, choose minimum turnover or the saved full-fit sampled H1 before consuming event two. Use fixed full-short/cash/full-long probes and reproduce actual composed planning actions on their original accounts. No new fitting, future-outcome maximization or final-window labels.",
  inference: "Report paired MSE versus a zero-difference forecast and realized value of selecting the predicted controller versus always hold and always sampled. Dense fixed-exposure cohorts use 2000 circular two-day block resamples within each origin, with equal weight per origin. Sparse selected-action cohorts are descriptive only.",
  caveat: "Observed-path forecast convention: close-price intermediate orders, aggregate event borrowing/risk and proportional terminal settlement. Not next-open minute execution or a portfolio replay. Paths overlap; windows were repeatedly inspected. Finite-horizon diagnostic, not a confidence guarantee or evidence of live profitability." }, null, 2));
const files = ["scripts/audit-event-controller-values.ts", "scripts/event-paths.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts",
  "scripts/research-event-policy.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }));
const mean = (v: number[]) => v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
const stats = (rows: any[]) => {
  const mse = mean(rows.map(r => (r.predictedDifferenceBps - r.actualDifferenceBps) ** 2)), zeroMse = mean(rows.map(r => r.actualDifferenceBps ** 2));
  const nonzero = rows.filter(r => Math.abs(r.actualDifferenceBps) > 1e-8);
  return { count: rows.length, distinctDays: new Set(rows.map(r => Math.floor(r.time / DAY))).size,
    mseBpsSquared: mse, zeroMseBpsSquared: zeroMse, skill: zeroMse ? 1 - mse! / zeroMse : null,
    meanPredictedDifferenceBps: mean(rows.map(r => r.predictedDifferenceBps)), meanActualDifferenceBps: mean(rows.map(r => r.actualDifferenceBps)),
    sampledChosenFraction: mean(rows.map(r => Number(r.sampledChosen))), differingActions: rows.filter(r => r.differingActions).length,
    meanGainVsHoldBps: mean(rows.map(r => r.gainVsHoldBps)), meanGainVsSampledBps: mean(rows.map(r => r.gainVsSampledBps)),
    nonzeroDifferences: nonzero.length,
    winnerAccuracy: mean(nonzero.map(r => Number(r.sampledChosen === (r.actualDifferenceBps > 0)))) };
};
const cohorts: Record<string, (r: any) => boolean> = {
  short: r => r.kind === "short", cash: r => r.kind === "cash", long: r => r.kind === "long",
  selectedExposed: r => r.kind === "selected" && Math.abs(r.account.exposure) > 1e-9,
  executedEntries: r => r.kind === "selected" && r.executedEntry,
};
const results: any[] = [], raw: any[] = [], started = performance.now();
for (const phase of phases) {
  const begin = performance.now(), option: FittedEventValue = read(config.source, `${phase.id}-policy.json`);
  const sampled: FittedEventValue = read(config.sampledSource, `${phase.id}-${eventFittedSettingName(setting)}-policy.json`);
  for (const key of ["costs", "targets", "exposures", "equities", "prices", "means", "scales", "penalty", "samples", "limits", "trainingSignature", "pathHorizon"] as const)
    assert.deepEqual(option[key], sampled[key]);
  assert.equal(option.targetMode, "minimum-turnover"); assert.equal(sampled.targetMode, "sampled-path"); assert.deepEqual(option.tables[0], sampled.tables[0]);
  const base = restoreEventPolicy(read(joint, `${phase.id}-policy.json`)), original: any[] = read(source, `${phase.id}-trades.json`);
  const byTime = new Map(original.map(r => [r.time, r]));
  const validation = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames);
  const paths = eventSignHorizonPaths(validation, 2), rows: any[] = []; let exactActions = 0;
  for (const p of paths) {
    const time = c[p.start].openTime + 60000, availableAt = c[p.end].openTime + 60000, originalRow = byTime.get(time);
    assert.ok(originalRow && availableAt < phase.endTime);
    const features = inputs(p.start), leaf = eventLeaf(base.model, eventFeatures(c, p.start, base.model.featureNames, sc.clock));
    const next = p.steps[1].start, nextLeaf = eventLeaf(base.model, eventFeatures(c, next, base.model.featureNames, sc.clock));
    const grids = [option, sampled].map(m => predictEventHoldingGrid(m, features, 2));
    const nextGrid = predictEventHoldingGrid(sampled, inputs(next), 1);
    const accounts = [-1, 0, 1].map(side => ({ kind: side < 0 ? "short" : side ? "long" : "cash",
      account: { equity: 10000, price: c[p.start].close, exposure: side * option.costs.maxLeverage }, executedEntry: false }));
    if (!originalRow.optionHolding) {
      const before = { equity: originalRow.equityBefore, price: c[p.start].close, exposure: originalRow.exposureBefore };
      const decision = decideFittedEventControllers([option, sampled], features, before, 2, leaf);
      assert.equal(JSON.stringify(decision.decision), JSON.stringify(originalRow.order), "Composed planning action changed");
      assert.equal(decision.controller ? "sampled" : "hold", originalRow.optionController); exactActions++;
      accounts.push({ kind: "selected", account: { equity: decision.decision.equity, price: decision.decision.price, exposure: decision.decision.exposure },
        executedEntry: Math.abs(originalRow.previousQuantity) < 1e-9 && originalRow.orderQuantity !== 0 });
    }
    for (const query of accounts) {
      const predicted = [option, sampled].map((m, i) => fittedEventHolding(m, grids[i], query.account, leaf));
      const actual = [eventControllerHolding(option, query.account, p.steps[0], p.steps[1], nextLeaf),
        eventControllerHolding(sampled, query.account, p.steps[0], p.steps[1], nextLeaf, nextGrid)];
      assert.ok([...predicted, ...actual.map(v => v.value)].every(Number.isFinite), "Nonfinite/ruin probe must not be silently dropped");
      const predictedDifferenceBps = (predicted[1] - predicted[0]) * 10000, actualDifferenceBps = (actual[1].value - actual[0].value) * 10000;
      const sampledChosen = predicted[1] > predicted[0] + 1e-12;
      const differingActions = Math.abs(actual[0].trade!.quantity - actual[1].trade!.quantity) > 1e-12;
      if (!differingActions) assert.ok(Math.abs(actualDifferenceBps) < 1e-8);
      rows.push({ time, availableAt, ...query, predictedBps: predicted.map(v => v * 10000), actualBps: actual.map(v => v.value * 10000),
        predictedDifferenceBps, actualDifferenceBps, sampledChosen, differingActions,
        continuationOrders: actual.map(v => ({ quantity: v.trade!.quantity, exposure: v.trade!.exposure })),
        gainVsHoldBps: sampledChosen ? actualDifferenceBps : 0, gainVsSampledBps: sampledChosen ? 0 : -actualDifferenceBps });
    }
  }
  const result = { phase, completePaths: paths.length, exactPlanningActions: exactActions,
    cohorts: Object.fromEntries(Object.entries(cohorts).map(([key, predicate]) => [key, stats(rows.filter(predicate))])), elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result); raw.push({ phase, rows });
  fs.writeFileSync(path.join(output, `${phase.id}-rows.json`), JSON.stringify(rows));
  console.log(JSON.stringify(result));
}
const paired = ["short", "cash", "long"].map(cohort => {
  const metrics = ["mseGainBpsSquared", "gainVsHoldBps", "gainVsSampledBps"];
  const origins = raw.map(({ phase, rows }) => {
    const firstDay = Math.floor(phase.startTime / DAY), lastDay = Math.ceil(phase.endTime / DAY);
    const days = Array.from({ length: lastDay - firstDay }, () => ({ count: 0, sums: [0, 0, 0] }));
    for (const r of rows.filter(cohorts[cohort])) {
      const d = days[Math.floor(r.time / DAY) - firstDay]; d.count++;
      d.sums[0] += r.actualDifferenceBps ** 2 - (r.predictedDifferenceBps - r.actualDifferenceBps) ** 2;
      d.sums[1] += r.gainVsHoldBps; d.sums[2] += r.gainVsSampledBps;
    }
    assert.ok(days.reduce((s, d) => s + d.count, 0) > 0);
    return { phase: phase.id, days };
  });
  let state = seed >>> 0;
  const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; };
  const draws = Array.from({ length: replicates }, () => {
    const values = origins.map(o => {
      const totals = [0, 0, 0]; let count = 0;
      for (let i = 0; i < o.days.length; i += 2) {
        const at = Math.floor(random() * o.days.length);
        for (let j = 0; j < 2 && i + j < o.days.length; j++) {
          const d = o.days[(at + j) % o.days.length]; count += d.count;
          for (let k = 0; k < 3; k++) totals[k] += d.sums[k];
        }
      }
      assert.ok(count); return totals.map(v => v / count);
    });
    return metrics.map((_, k) => mean(values.map(v => v[k]))!);
  });
  return { cohort, metrics: metrics.map((metric, k) => {
    const gains = origins.map(o => o.days.reduce((s, d) => s + d.sums[k], 0) / o.days.reduce((s, d) => s + d.count, 0));
    const samples = draws.map(v => v[k]).sort((a, b) => a - b);
    return { metric, originGains: gains, meanGain: mean(gains), interval95: [samples[Math.floor(replicates * .025)], samples[Math.floor(replicates * .975)]] };
  }) };
});
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, paired, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
console.log(JSON.stringify({ paired, elapsedSec: (performance.now() - started) / 1000 }));

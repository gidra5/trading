/** Exact short rollouts of the deployed frozen controller on observed accounts. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, observeMove } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { predictEventHoldingGrid, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventControllerHolding } from "./event-paths.js";
import { loadEventCandles, replayEventPolicy, settleEventReplayAccount } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000, horizon = 4;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify composed screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), sourceSummary = read(source, "summary.json");
assert.equal(config.contract, "event-controller-composition-screen-v1");
const hc = read(config.source, "config.json"), pc = read(hc.source, "config.json"), joint = pc.source;
const jc = read(joint, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const phases = config.phases, setting = config.sampledSetting;
assert.deepEqual(phases, hc.phases); assert.deepEqual(sourceSummary.results.map((r: any) => r.phase), phases);
const c = loadEventCandles(phases[0].startTime - 2 * DAY, phases.at(-1).endTime);
const external = loadEventFuturesRows(phases[0].startTime - 2 * DAY, phases.at(-1).endTime);
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i])), cache = new Map<number, number[]>();
const inputs = (i: number) => {
  if (cache.has(i)) return cache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t)), d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  if (!e || !d) throw new Error("Missing completed rollout input");
  const values = [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i), ...eventFittedFuturesInputs(setting.basis, e, d)];
  cache.set(i, values); return values;
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-deployed-rollout-audit-v1", source, phases, setting,
  replanCash: config.replanCash, quoteEntries: config.quoteEntries, horizon,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")))
    .update(fs.readFileSync(path.join(source, "summary.json"))).update(JSON.stringify(c)).update(external.fingerprint).digest("hex"),
  method: "At every original account with four complete future events inside its prior origin, resume the exact replay engine with original equity, quantity and option phase. Keep its frozen models, cash cadence and minute execution. Require every local decision and marked-account trace to equal the original suffix. Label each prefix by the replay's terminal market settlement without inserting settlements into subsequent prefixes. Independently replay H2 for the first origin row and every executed entry.",
  comparison: "At H2 planning states compare the forecast with both its original fixed-controller observed-path target and the exact deployed two-event payoff. Report the realized H4-minus-H2 continuation separately; an H2 forecast is not an H4 forecast. No new value head or action is selected from realized returns.",
  caveat: "Conditional on accounts visited by the existing policy; this is not a counterfactual action grid or a new profitable policy. Overlapping, repeatedly inspected prior paths; no final outcomes or training. Zero-return MSE is a forecast baseline, not an executable alternative from an invested account." }, null, 2));
const files = ["scripts/audit-event-policy-rollouts.ts", "scripts/event-paths.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts",
  "scripts/research-event-policy.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }));
const mean = (v: number[]) => v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
const describe = (rows: any[]) => {
  const predictions = rows.filter(r => r.predictedH2Bps !== null);
  return { count: rows.length, distinctDays: new Set(rows.map(r => Math.floor(r.time / DAY))).size, predictedH2Count: predictions.length,
    meanPredictedH2Bps: mean(predictions.map(r => r.predictedH2Bps)),
    meanActualH2Bps: mean(rows.map(r => r.prefixes[1].valueBps)), meanActualH4Bps: mean(rows.map(r => r.prefixes[3].valueBps)),
    meanH4MinusH2Bps: mean(rows.map(r => r.prefixes[3].valueBps - r.prefixes[1].valueBps)),
    h2ForecastMseBpsSquared: mean(predictions.map(r => (r.predictedH2Bps - r.prefixes[1].valueBps) ** 2)),
    h2ZeroMseBpsSquared: mean(predictions.map(r => r.prefixes[1].valueBps ** 2)),
    originalTargetMseBpsSquared: mean(predictions.map(r => (r.predictedH2Bps - r.originalTargetH2Bps) ** 2)),
    meanExactMinusTargetBps: mean(predictions.map(r => r.prefixes[1].valueBps - r.originalTargetH2Bps)),
    meanAbsoluteExactMinusTargetBps: mean(predictions.map(r => Math.abs(r.prefixes[1].valueBps - r.originalTargetH2Bps))),
    positiveH2ToNegativeH4: rows.filter(r => r.prefixes[1].valueBps > 0 && r.prefixes[3].valueBps < 0).length,
    negativeH2ToPositiveH4: rows.filter(r => r.prefixes[1].valueBps < 0 && r.prefixes[3].valueBps > 0).length };
};
const results: any[] = [], started = performance.now();
for (const [phaseIndex, phase] of phases.entries()) {
  const begin = performance.now(), sourceResult = sourceSummary.results[phaseIndex];
  for (const model of sourceResult.modelHashes)
    assert.equal(createHash("sha256").update(fs.readFileSync(model.file)).digest("hex"), model.sha256, "Frozen model changed");
  const fit = read(jc.source, `${jc.window.id}-${phase.id}-model.json`); assert.ok(fit.trainEnd <= phase.startTime);
  const policy: FittedEventValue = read(config.source, `${phase.id}-policy.json`);
  const sampled: FittedEventValue = read(config.sampledSource, `${phase.id}-${eventFittedSettingName(setting)}-policy.json`);
  const base = restoreEventPolicy(read(joint, `${phase.id}-policy.json`)), original: any[] = read(source, `${phase.id}-trades.json`);
  const observations = new Map<number, { availableAt: number; values: number[] }>(original.map(r => {
    const i = byTime.get(r.time); assert.notEqual(i, undefined);
    return [r.time, { availableAt: r.time, values: inputs(i!) }];
  }));
  const fitted = { policy, observations, replanEvery: 2, sampledAlternative: sampled, replanCash: config.replanCash || undefined };
  const execution = { quoteEntries: config.quoteEntries || undefined };
  const { trace: control, ...metrics } = replayEventPolicy(c, base, phase.startTime, phase.endTime, 2, { ...execution, fitted, trace: true });
  assert.equal(JSON.stringify(control), JSON.stringify(original), "Original deployed trace changed");
  assert.equal(JSON.stringify(metrics), JSON.stringify(sourceResult.candidate), "Original deployed metrics changed");
  assert.equal(metrics.liquidations, 0, "This audit must not censor a liquidated path");
  const moves = new Map<number, NonNullable<ReturnType<typeof observeMove>>>();
  const move = (r: any) => {
    if (moves.has(r.time)) return moves.get(r.time)!;
    const observed = observeMove(c, byTime.get(r.time)!, sc.clock, sc.featureNames);
    assert.ok(observed && c[observed.end].openTime + 60000 === r.endTime && r.endTime < phase.endTime);
    moves.set(r.time, observed); return observed;
  };
  const rows: any[] = []; let exactEvents = 0, independentH2Checks = 0;
  for (let index = 0; index + horizon <= original.length; index++) {
    const expected = original.slice(index, index + horizon), first = expected[0], last = expected.at(-1)!;
    if (last.endTime >= phase.endTime) continue;
    assert.ok(expected.every((r, i) => !i || r.time === expected[i - 1].endTime));
    const initialState = { remaining: first.optionHolding ? first.optionRemaining : 0, controller: first.optionController === "sampled" ? 1 as const : 0 as const };
    const options = { ...execution, equity: first.equityBefore, initialQuantity: first.previousQuantity, fitted: { ...fitted, initialState }, trace: true };
    const replay = replayEventPolicy(c, base, first.time, last.endTime, 2, options);
    assert.equal(JSON.stringify(replay.trace), JSON.stringify(expected), "Resumed local rollout differs from original suffix");
    exactEvents += replay.trace.length;
    const prefixes = replay.trace.map((r: any, i) => {
      const price = c[byTime.get(r.endTime)!].close;
      const quantity = Math.round((r.previousQuantity + r.orderQuantity) / policy.costs.quantityStep) * policy.costs.quantityStep;
      const settled = settleEventReplayAccount(r.equityAfter, quantity, price, policy.costs);
      assert.ok(settled.equity > 0);
      return { events: i + 1, availableAt: r.endTime, equity: settled.equity, valueBps: Math.log(settled.equity / first.equityBefore) * 10000,
        terminalFee: settled.fee, terminalOrders: settled.orders, terminalDust: settled.dust };
    });
    assert.equal(prefixes.at(-1)!.equity, replay.finalEquity);
    const executedEntry = Math.abs(first.previousQuantity) < 1e-9 && first.orderQuantity !== 0;
    if (!index || executedEntry) {
      const two = replayEventPolicy(c, base, first.time, expected[1].endTime, 2, options);
      assert.equal(JSON.stringify(two.trace), JSON.stringify(expected.slice(0, 2)));
      assert.equal(two.finalEquity, prefixes[1].equity); independentH2Checks++;
    }
    let originalTargetH2Bps: number | null = null;
    if (!first.optionHolding) {
      const next = byTime.get(expected[1].time)!, nextLeaf = eventLeaf(base.model, eventFeatures(c, next, base.model.featureNames, sc.clock));
      const actual = eventControllerHolding(first.optionController === "sampled" ? sampled : policy,
        { equity: first.order.equity, price: first.order.price, exposure: first.order.exposure }, move(first), move(expected[1]), nextLeaf,
        first.optionController === "sampled" ? predictEventHoldingGrid(sampled, inputs(next), 1) : undefined);
      originalTargetH2Bps = (Math.log(first.order.equity / first.equityBefore) + actual.value) * 10000;
      assert.ok(Number.isFinite(originalTargetH2Bps));
    }
    rows.push({ time: first.time, date: new Date(first.time).toISOString(), account: { equity: first.equityBefore,
      quantity: first.previousQuantity, exposure: first.exposureBefore, price: c[byTime.get(first.time)!].close },
      initialState, features: inputs(byTime.get(first.time)!), planning: !first.optionHolding, controller: first.optionController,
      proposedExposure: first.order.exposure, executedEntry,
      canceledEntry: Math.abs(first.previousQuantity) < 1e-9 && first.order.quantity !== 0 && !first.orderQuantity,
      predictedH2Bps: first.optionHolding ? null : first.order.value * 10000, originalTargetH2Bps, prefixes,
      orders: replay.trace.map((r: any) => ({ time: r.time, controller: r.optionController, remaining: r.optionRemaining, intermediate: r.optionHolding,
        plannedQuantity: r.order.quantity, executedQuantity: r.orderQuantity, equityAfter: r.equityAfter })) });
  }
  fs.writeFileSync(path.join(output, `${phase.id}-rows.json`), JSON.stringify(rows));
  const result = { phase, completeRollouts: rows.length, originalDecisions: original.length, exactEvents, independentH2Checks,
    originalReplayExact: true, allRolloutTracesExact: true,
    cohorts: { planning: describe(rows.filter(r => r.planning)), planningExposed: describe(rows.filter(r => r.planning && Math.abs(r.proposedExposure) > 1e-9)),
      executedEntries: describe(rows.filter(r => r.executedEntry)), canceledEntries: describe(rows.filter(r => r.canceledEntry)),
      intermediate: describe(rows.filter(r => !r.planning)) }, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result); console.log(JSON.stringify(result));
}
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
console.log(JSON.stringify({ elapsedSec: (performance.now() - started) / 1000 }));

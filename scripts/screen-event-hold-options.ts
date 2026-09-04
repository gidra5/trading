/** Two-event holding option with mandatory cap reduction and real replay execution. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { trainEventFittedValue, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventSignHorizonPaths } from "./event-paths.js";
import { eventOriginScore } from "./research-event-refits.js";
import { loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify sampled-path screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), choices = config.settings.filter((s: any) => s.sampledPath);
if (config.contract !== "event-fitted-value-screen-v1" || choices.length !== 1 || choices[0].boost || choices[0].secondDynamics
  || choices[0].candleShape || choices[0].continuationFolds || choices[0].pathHorizon !== 3) throw new Error("Requires one plain three-event sampled setting");
const setting = choices[0], jc = read(config.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const fits = config.phases.map((p: any) => read(jc.source, `${jc.window.id}-${p.id}-model.json`));
const c = loadEventCandles(fits[0].trainStart - DAY, config.phases.at(-1).endTime);
const external = loadEventFuturesRows(fits[0].trainStart - DAY, config.phases.at(-1).endTime), cache = new Map<number, number[] | null>();
const inputs = (i: number) => {
  if (cache.has(i)) return cache.get(i)!;
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t));
  const d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  const values = e && d ? [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
    ...eventFittedFuturesInputs(setting.basis, e, d)] : null;
  cache.set(i, values); return values;
};
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i]));
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-hold-option-screen-v1", source, phases: config.phases, setting, depths: [1, 2],
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(JSON.stringify(c)).update(external.fingerprint).digest("hex"),
  training: "Same complete three-event cohort and ridge features/penalty as the reference. At depth two, future orders minimize turnover subject to the existing feasible-action contract. Targets include the order cost, marked account, borrowing and terminal settlement. Fit the full equity/price/exposure grid; H1 is unchanged. No fitted future head or observed-return maximization chooses the continuation.",
  replay: "Choose the best H2 action, then hold for two completed events, making mandatory cap reductions at intermediate decisions. Replan at option expiry; cash may replan at every event. Replanning can retain inventory instead of closing/reopening it. Existing next-open fills, lot/notional constraints, fees, borrowing, intrabar risks and terminal settlement apply.",
  caveat: "Restricted two-event option experiment, not Bellman-optimal policy iteration or an optimality claim. Terminal training settlement uses the existing proportional forecast convention. Reused prior origins; no final inspector outcome is loaded and no candidate is promoted here." }, null, 2));
const files = ["scripts/screen-event-hold-options.ts", "scripts/event-paths.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts",
  "scripts/research-event-policy.ts", "scripts/research-event-refits.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }));
const results: any[] = [], started = performance.now();
for (const [index, phase] of config.phases.entries()) {
  const begin = performance.now(), fit = fits[index], base = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
  const reference: FittedEventValue = read(source, `${phase.id}-${eventFittedSettingName(setting)}-policy.json`);
  const state = (i: number) => eventLeaf(base.model, eventFeatures(c, i, base.model.featureNames, sc.clock));
  const transition = (r: MoveSample) => ({ features: inputs(r.start)!, nextFeatures: inputs(r.end)!, leaf: state(r.start), nextLeaf: state(r.end),
    move: { return: r.return, low: r.low, high: r.high, duration: r.duration } });
  const training = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [jc.window], sc.stride, "chain", sc.featureNames).filter(r => inputs(r.start) && inputs(r.end));
  const train = eventSignHorizonPaths(training, 3);
  assert.ok(train.every(p => c[p.end].openTime + 60000 < phase.startTime));
  const rows = train.map(p => transition(p.steps[0])), following = train.map(p => p.steps.slice(1).map(transition));
  trainEventFittedValue(base, rows, setting.penalty, 1, undefined, { ...reference, tables: reference.tables.slice(0, 1) }, { following });
  const policy = trainEventFittedValue(base, rows, setting.penalty, 2, undefined, undefined, { following, minimumTurnover: true });
  assert.deepEqual(policy.tables[0], reference.tables[0]);
  assert.equal(policy.trainingSignature, reference.trainingSignature); assert.deepEqual(policy.means, reference.means); assert.deepEqual(policy.scales, reference.scales);
  fs.writeFileSync(path.join(output, `${phase.id}-policy.json`), JSON.stringify(policy));
  const oldTrace = read(source, `${phase.id}-${eventFittedSettingName(setting)}-d1-trades.json`);
  const observations = new Map<number, { availableAt: number; values: number[] }>(oldTrace.map((r: any) => {
    const values = inputs(byTime.get(r.time)!); if (!values) throw new Error("Missing completed replay input");
    return [r.time, { availableAt: r.time, values }];
  }));
  const controls = replayEventPolicy(c, base, phase.startTime, phase.endTime, 1, { fitted: { policy, observations }, trace: true });
  assert.equal(JSON.stringify(controls.trace), JSON.stringify(oldTrace), "Serialized H1 control trace changed");
  const { trace: controlTrace, ...control } = controls;
  const { trace, ...candidate } = replayEventPolicy(c, base, phase.startTime, phase.endTime, 2,
    { fitted: { policy, observations, replanEvery: 2 }, trace: true });
  for (const row of trace) if (row.optionHolding) {
    assert.ok(!row.order.quantity || Math.abs(row.exposureBefore) > policy.costs.maxLeverage + 1e-9);
    assert.ok(!row.order.quantity || Math.abs(row.order.exposure) < Math.abs(row.exposureBefore));
  }
  fs.writeFileSync(path.join(output, `${phase.id}-trades.json`), JSON.stringify(trace));
  const result = { phase, training: train.length, referenceH1Exact: true, controlReplayExact: true, control, candidate,
    optionSteps: trace.filter((r: any) => r.optionHolding).length, forcedReductionSignals: trace.filter((r: any) => r.optionHolding && r.order.quantity).length,
    elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result);
  console.log(JSON.stringify({ phase: phase.id, training: train.length, control: { returnPct: control.returnPct, trades: control.trades },
    candidate: { returnPct: candidate.returnPct, drawdown: candidate.maxDrawdownPct, trades: candidate.trades, fees: candidate.fees },
    optionSteps: result.optionSteps, forcedReductionSignals: result.forcedReductionSignals, elapsedSec: result.elapsedSec }));
}
const ranking = ["control", "candidate"].map(choice => ({ choice, ...eventOriginScore(results.map(r => r[choice]), sc.riskPenalty) }))
  .sort((a, b) => b.score - a.score);
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, ranking, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
console.log(JSON.stringify({ ranking, elapsedSec: (performance.now() - started) / 1000 }));

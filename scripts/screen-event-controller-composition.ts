/** Replay a fixed two-event choice between saved hold and sampled controllers. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import type { FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { eventOriginScore } from "./research-event-refits.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86400000;
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify holding-option screen and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json");
if (config.contract !== "event-hold-option-screen-v1") throw new Error("Requires holding-option screen");
const setting = config.setting, parent = read(config.source, "config.json"), jc = read(parent.source, "config.json");
const sampledSource = arg("sampled-source") ? path.resolve(root, "data/benchmarks", arg("sampled-source")) : config.source;
const sampledConfig = read(sampledSource, "config.json"), choices = sampledConfig.settings.filter((s: any) => s.sampledPath);
if (sampledConfig.contract !== "event-fitted-value-screen-v1" || sampledConfig.source !== parent.source || choices.length !== 1
  || eventFittedSettingName({ ...choices[0], continuationFolds: undefined }) !== eventFittedSettingName(setting)
  || setting.secondDynamics || setting.candleShape || setting.boost || setting.pathHorizon !== 3)
  throw new Error("Requires compatible saved plain three-event controllers");
assert.deepEqual(sampledConfig.phases, config.phases);
const sampledSetting = choices[0], oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json"), phases = config.phases;
const reference = arg("reference") ? path.resolve(root, "data/benchmarks", arg("reference")) : undefined;
const replanCash = process.argv.includes("--replan-cash");
const quoteEntries = process.argv.includes("--quote-entries");
if ((replanCash || quoteEntries) && !reference) throw new Error("Timing or execution ablation requires the original composition reference");
if (reference) {
  const rc = read(reference, "config.json");
  assert.equal(rc.contract, "event-controller-composition-screen-v1"); assert.equal(rc.source, source); assert.equal(rc.sampledSource, sampledSource);
  assert.deepEqual(rc.phases, phases);
  assert.ok(!rc.quoteEntries, "The execution reference must use fixed base quantity");
  assert.equal(Boolean(rc.replanCash), quoteEntries && replanCash, "Reference cash timing differs from the control");
}
const c = loadEventCandles(phases[0].startTime - 2 * DAY, phases.at(-1).endTime);
const external = loadEventFuturesRows(phases[0].startTime - 2 * DAY, phases.at(-1).endTime);
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i]));
const inputs = (i: number) => {
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t));
  const d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  if (!e || !d) throw new Error("Missing completed composition input");
  return [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i), ...eventFittedFuturesInputs(setting.basis, e, d)];
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-controller-composition-screen-v1", source, sampledSource, reference,
  replanCash: replanCash || undefined, quoteEntries: quoteEntries || undefined,
  phases, setting, sampledSetting, depth: 2,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")))
    .update(fs.readFileSync(path.join(sampledSource, "config.json"))).update(JSON.stringify(c)).update(external.fingerprint).digest("hex"),
  method: "At each option expiry compare the complete interpolated H2 holding values of the saved minimum-turnover and sampled-path models for each feasible action. Commit to the maximizing controller for two events: the hold controller makes only mandatory cap reductions at the intermediate event; the sampled controller uses its shared H1, including after choosing cash. Exact ties retain the hold controller. Hold-controller cash can replan immediately. No new training.",
  controls: "Reproduce both the original minimum-turnover replay and the original rolling sampled-H2 replay exactly. Add a fixed two-event sampled controller that advances from H2 to H1 even in cash, isolating horizon advancement from composition. Compare with the composed replay on the same prior origins; when a reference is supplied, require its composed trace to reproduce exactly.",
  cashTimingAblation: replanCash ? "Keep the same action-value maximum but replan at every cash event, rather than committing sampled cash to H1. Invested two-event options still advance. Reproduce the original committed composition first. The cash H2 target still assumes H1 continuation, so this ablation adds a target/execution mismatch; it is not exact policy evaluation or an improvement guarantee." : undefined,
  entryExecutionAblation: quoteEntries ? "Freeze the selected flat entry's quote turnover at its completed decision close. At the next open, floor quote turnover divided by fill price to the base lot step, then apply the same fees, order bounds and leverage cap. Long and short entries use the same rule. Existing-inventory orders keep their fixed base quantity. Reproduce the reference with identical cash timing and fixed base orders first. This is a single-price quote-order approximation; it does not simulate exchange liquidity, commission assets, borrowing inventory or all symbol filters. Saved action-value models are unchanged." : undefined,
  executionReferences: quoteEntries ? ["https://developers.binance.com/en/docs/catalog/core-trading-margin-trading/api/rest-api/trade", "https://developers.binance.com/en/docs/catalog/core-trading-spot-trading/api/rest-api/trade"] : undefined,
  caveat: "Approximate finite-horizon controllers with no established uniform value-error bound; no generalized-policy-improvement or profitability guarantee. Reused prior origins only, no final-window outcomes or promotion." }, null, 2));
const files = ["scripts/screen-event-controller-composition.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts",
  "scripts/research-event-policy.ts", "scripts/research-event-refits.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }));
const results: any[] = [], started = performance.now();
for (const phase of phases) {
  const begin = performance.now(), optionFile = `${phase.id}-policy.json`, sampledFile = `${phase.id}-${eventFittedSettingName(sampledSetting)}-policy.json`;
  const option: FittedEventValue = read(source, optionFile), sampled: FittedEventValue = read(sampledSource, sampledFile);
  const base = restoreEventPolicy(read(parent.source, `${phase.id}-policy.json`)), originalTrace = read(source, `${phase.id}-trades.json`);
  assert.equal(option.targetMode, "minimum-turnover"); assert.equal(sampled.targetMode, "sampled-path");
  for (const key of ["costs", "targets", "exposures", "equities", "prices", "means", "scales", "penalty", "samples", "limits", "trainingSignature", "pathHorizon"] as const)
    assert.deepEqual(option[key], sampled[key]);
  assert.deepEqual(option.tables[0], sampled.tables[0]);
  const observations = new Map<number, { availableAt: number; values: number[] }>(originalTrace.map((r: any) => {
    const i = byTime.get(r.time); assert.notEqual(i, undefined);
    return [r.time, { availableAt: r.time, values: inputs(i!) }];
  }));
  const { trace: controlTrace, ...control } = replayEventPolicy(c, base, phase.startTime, phase.endTime, 2,
    { fitted: { policy: option, observations, replanEvery: 2 }, trace: true });
  assert.equal(JSON.stringify(controlTrace), JSON.stringify(originalTrace), "Original hold control trace changed");
  const { trace: sampledTrace, ...sampledControl } = replayEventPolicy(c, base, phase.startTime, phase.endTime, 2,
    { fitted: { policy: sampled, observations }, trace: true });
  assert.equal(JSON.stringify(sampledTrace), JSON.stringify(read(sampledSource, `${phase.id}-${eventFittedSettingName(sampledSetting)}-d2-trades.json`)),
    "Original sampled H2 control trace changed");
  const { trace: fixedSampledTrace, ...fixedSampledControl } = replayEventPolicy(c, base, phase.startTime, phase.endTime, 2,
    { fitted: { policy: sampled, observations, replanEvery: 2 }, trace: true });
  fs.writeFileSync(path.join(output, `${phase.id}-sampled-fixed-trades.json`), JSON.stringify(fixedSampledTrace));
  const { trace: committedTrace, ...committedControl } = replayEventPolicy(c, base, phase.startTime, phase.endTime, 2,
    { fitted: { policy: option, observations, replanEvery: 2, sampledAlternative: sampled }, trace: true });
  const { trace: timingTrace, ...timingControl } = replanCash ? replayEventPolicy(c, base, phase.startTime, phase.endTime, 2,
    { fitted: { policy: option, observations, replanEvery: 2, sampledAlternative: sampled, replanCash: true }, trace: true })
    : { trace: committedTrace, ...committedControl };
  if (reference) {
    assert.equal(JSON.stringify(quoteEntries ? timingTrace : committedTrace), JSON.stringify(read(reference, `${phase.id}-trades.json`)), "Composed reference trace changed");
    assert.deepEqual(quoteEntries ? timingControl : committedControl,
      read(reference, "summary.json").results.find((r: any) => r.phase.id === phase.id).candidate, "Composed reference metrics changed");
  }
  const { trace, ...candidate } = quoteEntries ? replayEventPolicy(c, base, phase.startTime, phase.endTime, 2,
    { fitted: { policy: option, observations, replanEvery: 2, sampledAlternative: sampled, replanCash: replanCash || undefined }, quoteEntries: true, trace: true })
    : { trace: timingTrace, ...timingControl };
  for (const row of trace as any[]) if (row.optionHolding && row.optionController === "hold") {
    assert.ok(!row.order.quantity || Math.abs(row.exposureBefore) > option.costs.maxLeverage + 1e-9);
    assert.ok(!row.order.quantity || Math.abs(row.order.exposure) < Math.abs(row.exposureBefore));
  }
  fs.writeFileSync(path.join(output, `${phase.id}-trades.json`), JSON.stringify(trace));
  const count = (controller: string, intermediate: boolean) => trace.filter(r => r.optionController === controller && r.optionHolding === intermediate).length;
  const result = { phase, holdControlExact: true, sampledControlExact: true, composedReferenceExact: reference ? true : undefined,
    control, sampledControl, fixedSampledControl, ...(replanCash ? { committedControl } : {}),
    ...(quoteEntries ? { executionControl: timingControl } : {}), candidate,
    modelHashes: [path.join(source, optionFile), path.join(sampledSource, sampledFile)].map(file => ({ file,
      sha256: createHash("sha256").update(fs.readFileSync(file)).digest("hex") })),
    decisions: { holdPlanning: count("hold", false), holdIntermediate: count("hold", true),
      sampledPlanning: count("sampled", false), sampledIntermediate: count("sampled", true) },
    elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result);
  console.log(JSON.stringify({ phase: phase.id, control: control.returnPct, sampledControl: sampledControl.returnPct,
    fixedSampledControl: { returnPct: fixedSampledControl.returnPct, drawdown: fixedSampledControl.maxDrawdownPct, trades: fixedSampledControl.trades },
    candidate: { returnPct: candidate.returnPct, drawdown: candidate.maxDrawdownPct, trades: candidate.trades, fees: candidate.fees },
    decisions: result.decisions, elapsedSec: result.elapsedSec }));
}
const ranking = ["control", "sampledControl", "fixedSampledControl", ...(replanCash ? ["committedControl"] : []),
  ...(quoteEntries ? ["executionControl"] : []), "candidate"]
  .map(choice => ({ choice, ...eventOriginScore(results.map(r => r[choice]), sc.riskPenalty) }))
  .sort((a, b) => b.score - a.score);
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, ranking, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
console.log(JSON.stringify({ ranking, elapsedSec: (performance.now() - started) / 1000 }));

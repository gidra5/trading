/** Diagnose cash opportunity value on the original holding-option accounts. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { eventFeatures, eventLeaf } from "../packages/bot-algo/src/event-distribution.js";
import { chooseEventTrade, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { decideFittedEvent, fittedEventHolding, predictEventHoldingGrid, type FittedEventValue } from "../packages/bot-algo/src/event-fitted-value.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { loadEventCandles } from "./research-event-policy.js";

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
const sampledConfig = read(sampledSource, "config.json"), sampledChoices = sampledConfig.settings.filter((s: any) => s.sampledPath);
if (sampledConfig.source !== parent.source || sampledChoices.length !== 1
  || eventFittedSettingName({ ...sampledChoices[0], continuationFolds: undefined }) !== eventFittedSettingName(setting))
  throw new Error("Incompatible sampled continuation source");
assert.deepEqual(sampledConfig.phases, config.phases);
const sampledSetting = sampledChoices[0];
const oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json"), phases = config.phases;
const c = loadEventCandles(phases[0].startTime - 2 * DAY, phases.at(-1).endTime), external = loadEventFuturesRows(phases[0].startTime - 2 * DAY, phases.at(-1).endTime);
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i]));
const inputs = (i: number) => {
  const e = eventFuturesFeatures(c, i, t => external.rows.get(t)), d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i, t => external.rows.get(t)) : [];
  if (!e || !d) throw new Error("Missing completed opportunity input");
  return [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i), ...eventFittedFuturesInputs(setting.basis, e, d)];
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-hold-opportunity-audit-v1", source, sampledSource, phases, setting, sampledSetting,
  sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "config.json"))).update(fs.readFileSync(path.join(sampledSource, "config.json"))).update(JSON.stringify(c)).update(external.fingerprint).digest("hex"),
  method: "On every original option planning account, compare minimum-turnover H2 with sampled-H1-continuation H2. Attribute the original entry's advantage over sampled cash waiting. Also maximize over the two complete interpolated holding functions for each feasible action. Preserve original accounts and timings; this is not a replay of the combined controller.",
  caveat: "Reused prior windows and selected trades. Both are approximate finite-horizon value functions; the maximum can amplify model errors. No new training, final outcomes, policy selection or generalized-policy-improvement guarantee." }, null, 2));
const files = ["scripts/audit-event-hold-opportunity.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts", "scripts/research-event-policy.ts",
  "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [], started = performance.now();
for (const phase of phases) {
  const option: FittedEventValue = read(source, `${phase.id}-policy.json`), sampled: FittedEventValue = read(sampledSource, `${phase.id}-${eventFittedSettingName(sampledSetting)}-policy.json`);
  const base = restoreEventPolicy(read(parent.source, `${phase.id}-policy.json`)), trace = read(source, `${phase.id}-trades.json`);
  assert.equal(option.targetMode, "minimum-turnover"); assert.equal(sampled.targetMode, "sampled-path");
  for (const key of ["costs", "targets", "exposures", "equities", "prices", "means", "scales", "penalty", "samples", "limits", "trainingSignature", "pathHorizon"] as const)
    assert.deepEqual(option[key], sampled[key]);
  assert.deepEqual(option.tables[0], sampled.tables[0]);
  const rows = [];
  for (const original of trace) {
    if (original.optionHolding) continue;
    const i = byTime.get(original.time); assert.notEqual(i, undefined);
    const features = inputs(i!), account = { equity: original.equityBefore, price: c[i!].close, exposure: original.exposureBefore };
    const leaf = eventLeaf(base.model, eventFeatures(c, i!, base.model.featureNames, sc.clock));
    const originalDecision = decideFittedEvent(option, features, account, 2, leaf);
    assert.equal(JSON.stringify(originalDecision), JSON.stringify(original.order), "Original option planning action changed");
    const grids = [option, sampled].map(p => predictEventHoldingGrid(p, features, 2));
    const values = (a: typeof account) => [option, sampled].map((p, j) => fittedEventHolding(p, grids[j], a, leaf));
    const combined = chooseEventTrade(option, account, a => Math.max(...values(a))), sampledDecision = decideFittedEvent(sampled, features, account, 2, leaf);
    const cash = { ...account, exposure: 0 }, cashValues = values(cash), chosenValues = values(combined);
    const flat = Math.abs(account.exposure) < 1e-9, entry = flat && original.orderQuantity !== 0;
    rows.push({ time: original.time, date: new Date(original.time).toISOString(), flat, originalEntry: entry,
      originalOrder: originalDecision, sampledOrder: sampledDecision, combinedOrder: combined,
      winningContinuation: chosenValues[0] >= chosenValues[1] - 1e-12 ? "hold" : "sampled",
      cashValuesBps: cashValues.map(v => v * 10000),
      originalEntryAdvantageVsWaitBps: entry ? (originalDecision.value - cashValues[1]) * 10000 : null,
      originalEntryKept: entry ? Math.sign(combined.exposure) === Math.sign(originalDecision.exposure) : null,
      originalEntryReplacedByCash: entry ? Math.abs(combined.exposure) < 1e-9 : null });
  }
  fs.writeFileSync(path.join(output, `${phase.id}-rows.json`), JSON.stringify(rows));
  const entries = rows.filter(r => r.originalEntry), result = { phase, originalPlanningActionsExact: true, planningDecisions: rows.length,
    originalEntries: entries.length, keptEntrySide: entries.filter(r => r.originalEntryKept).length,
    replacedByCash: entries.filter(r => r.originalEntryReplacedByCash).length,
    entries: entries.map(r => ({ date: r.date, side: Math.sign(r.originalOrder.exposure), optionNetValueBps: r.originalOrder.value * 10000,
      sampledWaitValueBps: r.cashValuesBps[1], advantageVsWaitBps: r.originalEntryAdvantageVsWaitBps,
      combinedExposure: r.combinedOrder.exposure, winningContinuation: r.winningContinuation })) };
  results.push(result); console.log(JSON.stringify(result));
}
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));

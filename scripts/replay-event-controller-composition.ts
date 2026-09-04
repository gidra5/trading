/** Freeze prior-origin selection, then replay compositions of saved final models. */
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
if (!["sources", "sampled-finals", "hold-final", "incumbent", "output"].every(k => arg(k)))
  throw new Error("Specify composition screens, paired sampled finals, hold final, incumbent and new output");
const dir = (s: string) => path.resolve(root, "data/benchmarks", s), read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const sources = arg("sources").split(",").map(dir), sampledFinals = arg("sampled-finals").split(",").map(dir);
const holdFinal = dir(arg("hold-final")), incumbent = dir(arg("incumbent")), output = dir(arg("output"));
assert.equal(sources.length, sampledFinals.length);
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const hc = read(holdFinal, "config.json"), joint = hc.source, jc = read(joint, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
const window = jc.window, phases = read(sources[0], "config.json").phases, previous = read(incumbent, "selection.json");
assert.equal(hc.contract, "event-hold-option-policy-v1"); assert.equal(read(incumbent, "config.json").source, joint);
assert.ok(!window.id.startsWith("fit-") && phases.every((p: any) => p.endTime <= window.startTime));
const option: FittedEventValue = read(holdFinal, "final-fitted-policy.json");
const screens = sources.map((source, i) => {
  const config = read(source, "config.json"), summary = read(source, "summary.json"), final = sampledFinals[i];
  const fc = read(final, "config.json"), selection = read(final, "selection.json"), sampled: FittedEventValue = read(final, "final-fitted-policy.json");
  assert.equal(config.contract, "event-controller-composition-screen-v1"); assert.equal(config.source, hc.screen);
  assert.deepEqual(config.phases, phases); assert.deepEqual(summary.results.map((r: any) => r.phase), phases);
  assert.equal(fc.source, joint); assert.ok(fc.screens.includes(config.sampledSource));
  assert.equal(selection.diagnosticChoice.choice, `fitted-value-${eventFittedSettingName(config.sampledSetting)}`); assert.equal(selection.diagnosticChoice.depth, 2);
  assert.equal(option.targetMode, "minimum-turnover"); assert.equal(sampled.targetMode, "sampled-path");
  for (const key of ["costs", "targets", "exposures", "equities", "prices", "means", "scales", "penalty", "samples", "limits", "trainingSignature", "pathHorizon"] as const)
    assert.deepEqual(option[key], sampled[key]);
  assert.deepEqual(option.tables[0], sampled.tables[0]);
  const candidates = (config.replanCash || config.quoteEntries ? ["candidate"] : ["candidate", "fixedSampledControl"]).map(key => {
    assert.ok(summary.results.every((r: any) => r[key]));
    return { choice: `${key === "candidate" ? "controller-composition" : "fixed-sampled"}-${eventFittedSettingName(config.sampledSetting)}${config.replanCash ? "-cash-replan" : ""}${config.quoteEntries ? "-quote-entry" : ""}`,
      depth: 2, ...eventOriginScore(summary.results.map((r: any) => r[key]), sc.riskPenalty),
      trades: summary.results.reduce((s: number, r: any) => s + r[key].trades, 0) };
  });
  return { source, config, summary, final, sampled, candidates };
});
const candidates = screens.flatMap(s => s.candidates), names = candidates.map(c => c.choice);
assert.equal(new Set(names).size, names.length); assert.ok(!previous.ranking.some((r: any) => names.includes(r.choice)));
const ranking = [...previous.ranking, ...candidates].sort((a, b) => b.score - a.score), chosen = ranking[0], cash = chosen.score <= 0;
assert.deepEqual(ranking.filter(r => !names.includes(r.choice)), previous.ranking);
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ chosen, cash, diagnosticChoices: candidates, ranking }, null, 2));
for (const [i, phase] of phases.entries()) {
  const saved = read(incumbent, `${phase.id}-scores.json`);
  const rows = screens.flatMap(s => s.candidates.map((candidate, j) => ({ choice: candidate.choice, depth: 2,
    ...s.summary.results[i][j ? "fixedSampledControl" : "candidate"] })));
  fs.writeFileSync(path.join(output, `${phase.id}-scores.json`), JSON.stringify({ ...saved, rows: [...saved.rows, ...rows] }));
}
const modelFiles = [path.join(holdFinal, "final-fitted-policy.json"), ...sampledFinals.map(d => path.join(d, "final-fitted-policy.json"))];
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-controller-composition-policy-v1", source: joint,
  screens: sources, sampledFinals, holdFinal, incumbent, window, phases,
  modelHashes: modelFiles.map(file => ({ file, sha256: createHash("sha256").update(fs.readFileSync(file)).digest("hex") })),
  sourceHash: createHash("sha256").update(JSON.stringify(screens.map(s => ({ config: s.config, summary: s.summary }))))
    .update(fs.readFileSync(path.join(incumbent, "selection.json"))).digest("hex"),
  selection: "Retain every incumbent and append the declared composition and, for the original committed mode, its fixed sampled control. Cash-replanning or quote-entry screens add only their new composition; their fixed sampled control is unchanged. Write the prior-origin ranking before loading final outcomes. Use compatible saved final models without retraining. Each diagnostic is gated only by its own prior score; if the incumbent wins, retain its exact final result.",
  caveat: "Repeated research windows, no sealed holdout, established value-error bound, optimality or profitability guarantee. Controller timing is part of the policy. Cash-replanning is an explicit target/execution mismatch: the saved cash H2 targets assume H1 continuation." }, null, 2));
const files = ["scripts/replay-event-controller-composition.ts", "scripts/screen-event-controller-composition.ts", "scripts/event-fitted-settings.ts", "scripts/event-futures-basis.ts",
  "scripts/research-event-policy.ts", "scripts/research-event-refits.ts", "packages/bot-algo/src/event-fitted-value.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
console.log(JSON.stringify({ chosen, diagnosticChoices: candidates, selectionSaved: true, retainedCandidates: ranking.length }));
const started = performance.now(), base = restoreEventPolicy(read(joint, "final-policy.json"));
const c = loadEventCandles(window.startTime - 2 * DAY, window.endTime), external = loadEventFuturesRows(window.startTime - 2 * DAY, window.endTime);
const byTime = new Map(c.map((r, i) => [r.openTime + 60000, i])), originalTrace = read(holdFinal, "fitted-diagnostic-trades.json");
const setting = hc.setting;
const observations = new Map<number, { availableAt: number; values: number[] }>(originalTrace.map((r: any) => {
  const i = byTime.get(r.time); assert.notEqual(i, undefined);
  const e = eventFuturesFeatures(c, i!, t => external.rows.get(t)), d = setting.historyMinutes ? eventFuturesBasisDeviations(c, i!, t => external.rows.get(t)) : [];
  if (!e || !d) throw new Error("Missing completed final composition input");
  return [r.time, { availableAt: r.time, values: [...eventFeatures(c, i!, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i!),
    ...eventFittedFuturesInputs(setting.basis, e, d)] }];
}));
const control = replayEventPolicy(c, base, window.startTime, window.endTime, 2, { fitted: { policy: option, observations, replanEvery: 2 }, trace: true });
assert.equal(JSON.stringify(control.trace), JSON.stringify(originalTrace), "Original final hold replay changed");
const diagnostics = []; let test;
for (const screen of screens) {
  const rolling = replayEventPolicy(c, base, window.startTime, window.endTime, 2, { fitted: { policy: screen.sampled, observations }, trace: true });
  assert.equal(JSON.stringify(rolling.trace), JSON.stringify(read(screen.final, "fitted-diagnostic-trades.json")), "Original final sampled replay changed");
  for (const [i, candidate] of screen.candidates.entries()) {
    const { trace, ...metrics } = replayEventPolicy(c, base, window.startTime, window.endTime, 2, { fitted: { observations, replanEvery: 2,
      policy: i ? screen.sampled : option, sampledAlternative: i ? undefined : screen.sampled,
      replanCash: screen.config.replanCash || undefined }, quoteEntries: screen.config.quoteEntries || undefined, cash: candidate.score <= 0, trace: true });
    fs.writeFileSync(path.join(output, `${candidate.choice}-trades.json`), JSON.stringify(trace));
    diagnostics.push({ candidate, metrics });
    if (chosen.choice === candidate.choice) { test = metrics; fs.writeFileSync(path.join(output, "trades.json"), JSON.stringify(trace)); }
    console.log(JSON.stringify({ choice: candidate.choice, returnPct: metrics.returnPct, trades: metrics.trades, drawdown: metrics.maxDrawdownPct }));
  }
}
if (!test) {
  const old = read(incumbent, "summary.json"); assert.deepEqual(chosen, old.chosen); assert.equal(cash, old.cash);
  test = old.test; fs.copyFileSync(path.join(incumbent, "trades.json"), path.join(output, "trades.json"));
}
fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }));
fs.writeFileSync(path.join(output, "reproduction-check.json"), JSON.stringify({ incumbentRankingExact: true, originalFinalHoldReplayExact: true,
  originalFinalSampledReplaysExact: true, incumbentCandidates: previous.ranking.length, totalCandidates: ranking.length }, null, 2));
const result = { window, chosen, cash, test, diagnostics, elapsedSec: (performance.now() - started) / 1000 };
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
console.log(JSON.stringify({ chosen, test: { returnPct: test.returnPct, trades: test.trades }, elapsedSec: result.elapsedSec }));

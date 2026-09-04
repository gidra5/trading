/** Paired approximation audit: same data/partitions/shrinkage, complete priors.
 * Exact training laws are references, not oracle evaluation distributions. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, eventMoveLabel, type EventDistribution, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { trainEventRunDistribution } from "../packages/bot-algo/src/event-run-model.js";
import { trainEventVolatilityLaw, retainQuietEventLaw } from "../packages/bot-algo/src/event-volatility-law.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { eventHolding } from "../packages/bot-algo/src/event-log-policy.js";
import { eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("forecast") || !arg("output")) throw new Error("Specify saved joint forecast and new audit output");
const forecast = path.resolve(root, "data/benchmarks", arg("forecast")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fc = read(forecast, "config.json"), source = fc.source, config = read(source, "config.json"), sc = read(config.source, "config.json");
if (fc.contract !== "event-volatility-law-screen-v1") throw new Error("Requires saved joint forecast");
const window = fc.window, pool = read(fc.forecast, "config.json"), setting = config.settings.find((s: any) => s.window.id === window.id);
const phases = [...eventRefitOrigins(window.startTime, config.foldCount, config.foldDays), { ...window, id: "final" }];
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const phase of phases) hash.update(fs.readFileSync(path.join(source, `${window.id}-${phase.id}-model.json`)));
if (hash.digest("hex") !== fc.sourceHash) throw new Error("Origin source changed");
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-prior-audit-v1", forecast, source, sourceHash: fc.sourceHash, window,
  selection: "No selection: exact-prior reference under the same frozen partition and shrinkage; no policy return inferred from forecast scores",
  caveat: "Same repeatedly examined research windows. Exact means the finite training prior is not compressed, not that forecasts are correct." }, null, 2));
const files = ["scripts/audit-event-prior.ts", "packages/bot-algo/src/event-run-model.ts", "packages/bot-algo/src/event-volatility-law.ts",
  "packages/bot-algo/src/event-distribution.ts", "scripts/research-event-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const started = performance.now(), c = loadEventCandles(pool.poolStart - DAY, window.endTime + DAY), results: any[] = [];
const history = makeSamples(c, sc.clock, pool.poolStart, window.startTime, [window], sc.stride, "chain", sc.featureNames);
const available = (r: MoveSample) => c[r.end].openTime + 60_000;
const fastCache = new Map<number, number>();
const fast = (i: number) => { if (!fastCache.has(i)) fastCache.set(i, eventFastVolatilityFeatures(c, i)[1]); return fastCache.get(i)!; };
const augment = (rows: MoveSample[]) => rows.map(r => ({ ...r, features: [...r.features, fast(r.start)], nextFeatures: [...r.nextFeatures, fast(r.end)] }));
const olderInputs = augment(history);
const moments = (model: EventDistribution) => model.kernels.map(kernel => ({
  mean: kernel.reduce((s, a) => s + a.probability * a.return, 0), duration: kernel.reduce((s, a) => s + a.probability * a.duration, 0),
  logDuration: kernel.reduce((s, a) => s + a.probability * Math.log1p(a.duration), 0),
  low: Math.min(...kernel.map(a => a.low)), high: Math.max(...kernel.map(a => a.high)),
  classes: Array.from({ length: 15 }, (_, label) => kernel.reduce((s, a) => s + a.probability * Number(eventMoveLabel(a.return, a.duration, model.clock) === label), 0)),
  successor: Array.from({ length: model.kernels.length }, (_, next) => kernel.reduce((s, a) => s + a.probability * Number(a.next === next), 0)),
  holding: [-1, 0, 1].map(x => kernel.reduce((s, a) => { const held = eventHolding(x, a, config.costs); return s + a.probability * (held.liquidated ? -Infinity : Math.log(held.factor)); }, 0)),
}));
for (const phase of phases) {
  const fit = read(source, `${window.id}-${phase.id}-model.json`), saved = read(forecast, `${phase.id}-models.json`).models;
  const rows = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [window], sc.stride, "chain", sc.featureNames);
  const recent = augment(rows), older = olderInputs.filter(r => available(r) < fit.trainStart);
  if (rows.length !== fit.support.samples || [...recent, ...older].some(r => available(r) >= phase.startTime)) throw new Error("Training chronology mismatch");
  const options = { maxDepth: setting.treeDepth, minLeaf: sc.minLeaf, prior: sc.prior, criterion: sc.criterion, directionPrior: sc.runDirectionPrior };
  const control = trainEventRunDistribution(rows, sc.clock, options);
  if (JSON.stringify(control) !== JSON.stringify(fit.base.model)) throw new Error("Original run law does not reproduce exactly");
  const base = trainEventRunDistribution(rows, sc.clock, { ...options, priorLimit: rows.length });
  if (JSON.stringify(base.nodes) !== JSON.stringify(control.nodes)) throw new Error("Exact prior changed partition");
  const compressed = trainEventVolatilityLaw(control, recent, older, { prior: fc.prior, quantile: fc.quantile, globalPriorShare: fc.globalPriorShare });
  const hybrid = retainQuietEventLaw(control, compressed, recent);
  if (JSON.stringify(hybrid) !== JSON.stringify(saved.hybrid)) throw new Error("Original hybrid law does not reproduce exactly");
  const pooled = trainEventVolatilityLaw(control, recent, older, { prior: fc.prior, quantile: fc.quantile, globalPriorShare: fc.globalPriorShare, compressPrior: false });
  const models: Record<string, EventDistribution> = { "base-control": control, "base-exact": base, "hybrid-control": hybrid,
    "hybrid-exact-high": retainQuietEventLaw(control, pooled, recent), "hybrid-exact": retainQuietEventLaw(base, pooled, recent) };
  fs.writeFileSync(path.join(output, `${phase.id}-models.json`), JSON.stringify({ phase, models }));
  const ms = Object.fromEntries(Object.entries(models).map(([name, model]) => [name, moments(model)]));
  const evaluation = augment(makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames));
  const scores = Object.entries(models).map(([choice, model]) => {
    let nll = 0, mse = 0, logDurationMse = 0, unsupported = 0;
    for (const r of evaluation) {
      const m = ms[choice][eventLeaf(model, r.features.slice(0, model.featureNames.length))];
      nll -= Math.log(Math.max(1e-12, m.classes[r.label])); mse += (r.return - m.mean) ** 2;
      logDurationMse += (Math.log1p(r.duration) - m.logDuration) ** 2; unsupported += Number(m.classes[r.label] === 0);
    }
    return { choice, events: evaluation.length, nll: nll / evaluation.length, mse: mse / evaluation.length,
      logDurationMse: logDurationMse / evaluation.length, unsupported, atoms: model.kernels.reduce((s, k) => s + k.length, 0) };
  });
  const pairs = [["base-control", "base-exact"], ["hybrid-control", "hybrid-exact-high"], ["hybrid-control", "hybrid-exact"]].map(([a, b]) => {
    const states = ms[a].map((m, i) => { const exact = ms[b][i]; return { state: i, count: models[a].counts[i],
      meanErrorBps: (m.mean - exact.mean) * 1e4, durationError: m.duration - exact.duration,
      logDurationError: m.logDuration - exact.logDuration,
      classTV: m.classes.reduce((s, p, j) => s + Math.abs(p - exact.classes[j]), 0) / 2,
      successorTV: m.successor.reduce((s, p, j) => s + Math.abs(p - exact.successor[j]), 0) / 2,
      holdingErrorBps: m.holding.map((v, j) => (v - exact.holding[j]) * 1e4),
      omittedWorstLow: m.low > exact.low + 1e-12, omittedWorstHigh: m.high < exact.high - 1e-12,
    }; });
    const visits = evaluation.map(r => eventLeaf(models[a], r.features.slice(0, models[a].featureNames.length)));
    return { compressed: a, exact: b, states,
      maximumMeanErrorBps: Math.max(...states.map(s => Math.abs(s.meanErrorBps))),
      visitedMeanAbsoluteErrorBps: visits.reduce((s, i) => s + Math.abs(states[i].meanErrorBps), 0) / visits.length,
      maximumHoldingErrorBps: Math.max(...states.flatMap(s => s.holdingErrorBps.map(Math.abs))),
      maximumSuccessorTV: Math.max(...states.map(s => s.successorTV)),
      visitedSuccessorTV: visits.reduce((s, i) => s + states[i].successorTV, 0) / visits.length,
      omittedExtremaStates: states.filter(s => s.omittedWorstLow || s.omittedWorstHigh).length,
    };
  });
  const result = { phase, controlsExact: true, scores, pairs, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ ...result, pairs: pairs.map(p => ({ ...p, states: undefined })) }));
}

/** Compare complete volatility-conditioned paths before building policies. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, eventMoveLabel, type EventDistribution, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { retainQuietEventLaw, trainEventVolatilityLaw } from "../packages/bot-algo/src/event-volatility-law.js";
import { trainEventRunDistribution } from "../packages/bot-algo/src/event-run-model.js";
import { compressEventDistribution } from "../packages/bot-algo/src/event-quadrature.js";
import { eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const forecastId = arg("forecast"), outputId = arg("output");
if (!forecastId || !outputId) throw new Error("Specify pooled-head forecast and new output");
const forecast = path.resolve(root, "data/benchmarks", forecastId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fc = read(forecast, "config.json"), source = fc.source, config = read(source, "config.json"), sc = read(config.source, "config.json");
if (fc.contract !== "event-pooling-forecast-v1" || config.contract !== "rolling-event-refit-v1") throw new Error("Requires causal pooled-head origins");
const window = fc.window, phases = [...eventRefitOrigins(window.startTime, config.foldCount, config.foldDays), { ...window, id: "final" }];
const setting = config.settings.find((s: any) => s.window.id === window.id);
const choices = ["base", "head", "local", "pooled", "hybrid", "quadrature"];
const saved = phases.map(p => read(source, `${window.id}-${p.id}-model.json`)), headScores = read(forecast, "summary.json");
const fingerprint = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const p of phases) fingerprint.update(fs.readFileSync(path.join(source, `${window.id}-${p.id}-model.json`)));
if (fingerprint.digest("hex") !== fc.sourceHash) throw new Error("Rolling origin source changed");
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-volatility-law-screen-v1", source, forecast, sourceHash: fc.sourceHash, window,
  choices, prior: 100, quantile: 0.9, globalPriorShare: 0.1,
  selection: "Equal-origin mean joint NLL across three prior origins; original base then recent head eligible; selection saved before test",
  model: "Frozen canonical run partition crossed with physical direction and two observed RV5 bands. Quiet paths use recent data; pooled variant adds older high-volatility paths. Joint successors include next RV5 band.",
  priorApproximation: "100 equivalent observations: 90% same volatility band, 10% global paths; at most three complete observed paths per joint class/successor stratum retain its mass and worst low/high excursions",
  quadrature: "Additional hybrid candidate starts from complete run and volatility priors, then reduces positive support within actual/reciprocal class and successor strata. Preserve seven moments and adverse excursion/duration frontiers; no changed fit population or shrinkage.",
  caveat: "New conditional law compared on repeatedly examined research windows; forecast gains are not trading utility or Bellman convergence." }, null, 2));
const files = ["scripts/screen-event-volatility-law.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-volatility-law.ts",
  "packages/bot-algo/src/event-run-model.ts", "packages/bot-algo/src/event-quadrature.ts", "scripts/research-event-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const started = performance.now(), c = loadEventCandles(fc.poolStart - DAY, window.endTime + DAY);
const history = makeSamples(c, sc.clock, fc.poolStart, window.startTime, [window], sc.stride, "chain", sc.featureNames);
const at = (r: MoveSample) => c[r.start].openTime + 60_000, available = (r: MoveSample) => c[r.end].openTime + 60_000;
const fastCache = new Map<number, number>();
const fast = (index: number) => { if (!fastCache.has(index)) fastCache.set(index, eventFastVolatilityFeatures(c, index)[1]); return fastCache.get(index)!; };
const augment = (rows: MoveSample[]) => rows.map(r => ({ ...r, features: [...r.features, fast(r.start)], nextFeatures: [...r.nextFeatures, fast(r.end)] }));
const olderInputs = augment(history), results: any[] = [];
function evaluate(rows: MoveSample[], model: EventDistribution, cut: number, trace: boolean) {
  const moments = model.kernels.map(kernel => ({ classes: Array.from({ length: 15 }, (_, label) => kernel.reduce((s, a) => s + a.probability * Number(eventMoveLabel(a.return, a.duration, model.clock) === label), 0)),
    mean: kernel.reduce((s, a) => s + a.probability * a.return, 0), duration: kernel.reduce((s, a) => s + a.probability * Math.log1p(a.duration), 0) }));
  const total = { count: 0, nll: 0, mse: 0, logDurationMse: 0, unsupportedClass: 0 }, high = { ...total }, predictions = [];
  for (const r of rows) {
    const leaf = eventLeaf(model, r.features.slice(0, model.featureNames.length)), m = moments[leaf];
    for (const s of r.features[20] > cut ? [total, high] : [total]) {
      s.count++; s.nll -= Math.log(Math.max(1e-12, m.classes[r.label])); s.mse += (r.return - m.mean) ** 2;
      s.unsupportedClass += Number(m.classes[r.label] === 0);
      s.logDurationMse += (Math.log1p(r.duration) - m.duration) ** 2;
    }
    if (trace) predictions.push({ time: at(r), availableAt: available(r), leaf, realizedReturnBps: r.return * 1e4, expectedReturnBps: m.mean * 1e4, duration: r.duration, expectedLogDuration: m.duration });
  }
  const norm = (s: typeof total) => ({ count: s.count, nll: s.count ? s.nll / s.count : null, mse: s.count ? s.mse / s.count : null, logDurationMse: s.count ? s.logDurationMse / s.count : null, unsupportedClass: s.unsupportedClass });
  return { all: norm(total), high: norm(high), predictions };
}
let selected = "base";
for (const [i, phase] of phases.entries()) {
  if (phase.id === "final") {
    const ranking = choices.map(choice => ({ choice,
      meanNll: results.reduce((s, r) => s + r.scores.find((v: any) => v.choice === choice).all.nll, 0) / results.length })).sort((a, b) => a.meanNll - b.meanNll);
    selected = ranking[0].choice; fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ selected, ranking }, null, 2));
  }
  const fit = saved[i], base: EventDistribution = fit.base.model;
  const recentRows = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [window], sc.stride, "chain", sc.featureNames);
  const recent = augment(recentRows);
  const older = olderInputs.filter(r => available(r) < fit.trainStart);
  if (recent.length !== fit.support.samples || [...recent, ...older].some(r => available(r) >= phase.startTime)) throw new Error("Volatility law training chronology mismatch");
  const pooled = trainEventVolatilityLaw(base, recent, older, { prior: 100, quantile: 0.9 });
  const completeBase = trainEventRunDistribution(recentRows, sc.clock, { maxDepth: setting.treeDepth, minLeaf: sc.minLeaf,
    prior: sc.prior, criterion: sc.criterion, directionPrior: sc.runDirectionPrior, priorLimit: recentRows.length });
  if (JSON.stringify(completeBase.nodes) !== JSON.stringify(base.nodes)) throw new Error("Complete prior changed the frozen partition");
  const completePooled = trainEventVolatilityLaw(base, recent, older, { prior: 100, quantile: 0.9, compressPrior: false });
  const models = { base, local: trainEventVolatilityLaw(base, recent, [], { prior: 100, quantile: 0.9 }), pooled,
    hybrid: retainQuietEventLaw(base, pooled, recent),
    quadrature: compressEventDistribution(retainQuietEventLaw(completeBase, completePooled, recent)) };
  fs.writeFileSync(path.join(output, `${phase.id}-models.json`), JSON.stringify({ phase, models }));
  const evaluation = augment(makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames));
  const scores = Object.entries(models).map(([choice, model]) => {
    const { predictions, ...score } = evaluate(evaluation, model, models.local.runVolatility!.cut, phase.id === "final" && selected === choice);
    if (predictions.length) fs.writeFileSync(path.join(output, "predictions.json"), JSON.stringify(predictions));
    return { choice, ...score };
  });
  const reference = headScores.find((r: any) => r.phase.id === phase.id).scores.find((s: any) => s.choice === "recent");
  if (reference.all.count !== evaluation.length) throw new Error("Reference event population differs");
  scores.splice(1, 0, { choice: "head", all: reference.all, high: reference.high });
  const result = { phase, selected: phase.id === "final" ? selected : undefined, scores,
    localCounts: models.local.counts, pooledCounts: models.pooled.counts, metadata: models.pooled.runVolatility, elapsedSec: (performance.now() - started) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ ...result, localCounts: undefined, pooledCounts: undefined }));
}

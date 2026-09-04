/** Prior-only horizon audit: does a paired feature addition predict persistent direction? */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventFeatures } from "../packages/bot-algo/src/event-distribution.js";
import { eventFastVolatilityFeatures } from "../packages/bot-algo/src/event-size-sign.js";
import { predictEventSign, trainEventSign } from "../packages/bot-algo/src/event-sign.js";
import { eventCandleShapes, eventFuturesFeatures, eventFuturesBasisDeviations, eventFittedFuturesInputs, loadEventFuturesRows } from "./event-futures-basis.js";
import { eventSecondDynamicsAt, loadEventSecondDynamics } from "./event-second-dynamics.js";
import { eventFittedSettingName } from "./event-fitted-settings.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";
import { eventSignHorizonPaths } from "./event-paths.js";

const mean = (v: number[]) => v.reduce((s, x) => s + x, 0) / Math.max(1, v.length);
const quantile = (v: number[], q: number) => [...v].sort((a, b) => a - b)[Math.floor((v.length - 1) * q)];
const loss = (q: number, r: number) => -Math.log(r > 0 ? q : 1 - q);

export function main() {
  const root = path.resolve(__dirname, ".."), DAY = 86400000;
  const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
  if (!arg("source") || !arg("output")) throw new Error("Specify paired feature screen and new output");
  const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
  if (fs.existsSync(output)) throw new Error("Choose a new output directory");
  const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
  const sourceConfig = read(source, "config.json"), sourceSummary = read(source, "summary.json");
  if (sourceConfig.contract !== "event-fitted-value-screen-v1" || sourceConfig.settings?.length !== 2) throw new Error("Requires paired feature source");
  const changed = (["secondDynamics", "candleShape"] as const).filter(k => Boolean(sourceConfig.settings[0][k]) !== Boolean(sourceConfig.settings[1][k]));
  if (changed.length !== 1) throw new Error("Requires exactly one feature addition");
  const feature = changed[0], setting = sourceConfig.settings.find((s: any) => !s[feature]), augmented = sourceConfig.settings.find((s: any) => s[feature]);
  if (eventFittedSettingName({ ...augmented, [feature]: undefined }) !== eventFittedSettingName(setting)
    || (setting.pathHorizon && setting.pathHorizon !== 3)) throw new Error("Requires matched settings and a three-event cohort");
  const jc = read(sourceConfig.source, "config.json"), oc = read(jc.source, "config.json"), sc = read(oc.source, "config.json");
  const phases = sourceConfig.phases, fits = phases.map((p: any) => read(jc.source, `${jc.window.id}-${p.id}-model.json`));
  const start = fits[0].trainStart - DAY, end = phases.at(-1).endTime;
  const c = loadEventCandles(start, end), external = loadEventFuturesRows(start, end);
  const seconds = augmented.secondDynamics ? loadEventSecondDynamics(start, end) : undefined;
  const extras = new Map<number, ReturnType<typeof eventFuturesFeatures>>(), deviations = new Map<number, number[] | null>();
  const extra = (i: number) => { if (!extras.has(i)) extras.set(i, eventFuturesFeatures(c, i, t => external.rows.get(t))); return extras.get(i)!; };
  const deviation = (i: number) => { if (!deviations.has(i)) deviations.set(i, eventFuturesBasisDeviations(c, i, t => external.rows.get(t))); return deviations.get(i)!; };
  const shapes = (i: number) => eventCandleShapes(c, i, t => external.rows.get(t));
  const matched = (i: number) => extra(i) && (!setting.historyMinutes || deviation(i)) && (!augmented.candleShape || shapes(i));
  const featureCache = new Map<number, { base: number[]; augmented: number[] }>();
  const inputs = (i: number) => {
    if (featureCache.has(i)) return featureCache.get(i)!;
    const e = extra(i), d = ["deviation", "centered"].includes(setting.basis) ? deviation(i) : [];
    const shape = augmented.candleShape ? shapes(i) : [];
    if (!e || !d || !shape) throw new Error("Missing matched input");
    const base = [...eventFeatures(c, i, sc.featureNames, sc.clock), ...eventFastVolatilityFeatures(c, i),
      ...eventFittedFuturesInputs(setting.basis, e, d)];
    const secondValues = seconds ? eventSecondDynamicsAt(seconds.rows, c[i]) : [];
    const row = { base: [...base, ...(setting.secondDynamics ? secondValues : []), ...(setting.candleShape ? shape : [])],
      augmented: [...base, ...secondValues, ...shape] };
    featureCache.set(i, row); return row;
  };
  const targets = [{ name: "through-1", step: 0, cumulative: true }, { name: "through-2", step: 1, cumulative: true },
    { name: "through-3", step: 2, cumulative: true }, { name: "event-2-only", step: 1, cumulative: false }, { name: "event-3-only", step: 2, cumulative: false }];
  const objectives = ["ordinary", "return-weighted"] as const, variants = ["base", "augmented"] as const;
  const penalty = 0.1, horizon = 3, replicates = 2000, seed = 20260904;
  fs.mkdirSync(output, { recursive: true });
  const sourceHash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")))
    .update(JSON.stringify(c)).update(external.fingerprint).update(seconds?.fingerprint ?? "").digest("hex");
  fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-sign-horizon-audit-v1", source, sourceHash, phases, setting, targets,
    objectives, feature, penalty, horizon, replicates, seed,
    target: "All five targets share origins having three contiguous completed events. Every forecast uses only inputs at the first origin. Later-event-only targets test persistence without including the first event return.",
    weighted: "Weights are absolute target arithmetic returns, normalized by the existing fitter. This predicts a return-weighted class score, not P(up); its midpoint separates the conditional signed mean when that score is correct.",
    caveat: "Reused prior origins; no final outcomes or policy selection. Overlapping multi-event paths. Whole-day bootstrap is descriptive and cannot make adjacent days independent. Signed gross return ignores costs and is not a portfolio backtest. No weighted score is passed as a probability to a trading policy." }, null, 2));
  const files = ["scripts/audit-event-sign-horizons.ts", "scripts/event-paths.ts", "scripts/research-event-policy.ts", "scripts/event-futures-basis.ts", "scripts/event-second-dynamics.ts",
    "packages/bot-algo/src/event-sign.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-size-sign.ts"];
  fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
  if (seconds) fs.writeFileSync(path.join(output, "second-dynamics-sources.json"), JSON.stringify({ ...seconds, rows: undefined }));
  fs.writeFileSync(path.join(output, "external-sources.json"), JSON.stringify({ ...external, rows: undefined }));
  const results: any[] = [], comparisons: Array<{ phase: string; target: string; objective: string; days: Array<{ day: number; sum: number; weight: number }> }> = [];
  const started = performance.now();
  for (const [phaseIndex, phase] of phases.entries()) {
    const trainSingle = makeSamples(c, sc.clock, fits[phaseIndex].trainStart, fits[phaseIndex].trainEnd, [jc.window], sc.stride, "chain", sc.featureNames)
      .filter(r => matched(r.start) && matched(r.end));
    const testSingle = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames)
      .filter(r => matched(r.start) && matched(r.end));
    const reference = sourceSummary.results[phaseIndex].rows.find((r: any) => eventFittedSettingName(r) === eventFittedSettingName(setting));
    const train = eventSignHorizonPaths(trainSingle, horizon), test = eventSignHorizonPaths(testSingle, horizon);
    if (reference.samples !== (setting.pathHorizon ? train.length : trainSingle.length)
      || reference.validation !== (setting.pathHorizon ? test.length : testSingle.length)) throw new Error("Original reference cohort changed");
    if (train.some(r => c[r.end].openTime + 60000 >= phase.startTime) || test.some(r => c[r.end].openTime + 60000 >= phase.endTime)) throw new Error("Future or censored label");
    const rows = [];
    for (const target of targets) {
      const targetReturn = (p: (typeof train)[number]) => target.cumulative ? p.cumulativeReturns[target.step] : p.steps[target.step].return;
      const activeTrain = train.filter(p => targetReturn(p) !== 0), activeTest = test.filter(p => targetReturn(p) !== 0);
      for (const objective of objectives) {
        const weighted = objective === "return-weighted", weights = activeTrain.map(p => Math.abs(targetReturn(p)));
        const trainWeight = weighted ? weights.reduce((s, v) => s + v, 0) : activeTrain.length;
        const positiveWeight = activeTrain.reduce((s, p, i) => s + (targetReturn(p) > 0 ? (weighted ? weights[i] : 1) : 0), 0);
        const constant = (positiveWeight / trainWeight * activeTrain.length + 0.5) / (activeTrain.length + 1);
        const heads = Object.fromEntries(variants.map(variant => [variant, trainEventSign(activeTrain.map(p => ({
          features: inputs(p.start)[variant], return: targetReturn(p) })), penalty, weighted ? weights : undefined)]));
        const predictions = activeTest.map(p => ({ time: c[p.start].openTime + 60000, availableAt: c[p.steps[target.step].end].openTime + 60000,
          return: targetReturn(p), durationMinutes: p.cumulativeMinutes[target.step],
          base: predictEventSign(heads.base, inputs(p.start).base), augmented: predictEventSign(heads.augmented, inputs(p.start).augmented), constant }));
        const totalWeight = predictions.reduce((s, p) => s + (weighted ? Math.abs(p.return) : 1), 0), dayMap = new Map<number, { day: number; sum: number; weight: number }>();
        for (const p of predictions) {
          const day = Math.floor(p.time / DAY), weight = weighted ? Math.abs(p.return) : 1, row = dayMap.get(day) ?? { day, sum: 0, weight: 0 };
          row.sum += weight * (loss(p.base, p.return) - loss(p.augmented, p.return)); row.weight += weight; dayMap.set(day, row);
        }
        comparisons.push({ phase: phase.id, target: target.name, objective, days: [...dayMap.values()] });
        const models = [...variants, "constant" as const].map(variant => {
          const correct = (p: (typeof predictions)[number]) => (p[variant] >= 0.5) === (p.return > 0);
          const abs = predictions.reduce((s, p) => s + Math.abs(p.return), 0);
          return { variant, loss: predictions.reduce((s, p) => s + (weighted ? Math.abs(p.return) : 1) * loss(p[variant], p.return), 0) / totalWeight,
            accuracy: mean(predictions.map(p => Number(correct(p)))), magnitudeWeightedAccuracy: predictions.reduce((s, p) => s + Math.abs(p.return) * Number(correct(p)), 0) / abs,
            signedGrossBps: mean(predictions.map(p => (p[variant] >= 0.5 ? 1 : -1) * p.return * 10000)) };
        });
        rows.push({ target: target.name, objective, training: activeTrain.length, validation: predictions.length,
          elapsedMinutes: { median: quantile(predictions.map(p => p.durationMinutes), 0.5), p90: quantile(predictions.map(p => p.durationMinutes), 0.9) }, models });
        const prefix = `${phase.id}-${target.name}-${objective}`;
        fs.writeFileSync(path.join(output, `${prefix}-heads.json`), JSON.stringify(heads));
        fs.writeFileSync(path.join(output, `${prefix}-predictions.json`), JSON.stringify(predictions));
      }
    }
    results.push({ phase, referenceCohortExact: true, singleTraining: trainSingle.length, singleValidation: testSingle.length, trainingPaths: train.length, validationPaths: test.length, rows });
    console.log(JSON.stringify({ phase: phase.id, trainingPaths: train.length, validationPaths: test.length,
      lossGains: rows.map(r => ({ target: r.target, objective: r.objective, gain: r.models[0].loss - r.models[1].loss })) }));
  }
  let state = seed >>> 0;
  const random = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; return (state >>> 0) / 4294967296; };
  const paired = targets.flatMap(target => objectives.map(objective => {
    const origins = comparisons.filter(r => r.target === target.name && r.objective === objective);
    const gains = origins.map(r => r.days.reduce((s, d) => s + d.sum, 0) / r.days.reduce((s, d) => s + d.weight, 0));
    const samples = Array.from({ length: replicates }, () => mean(origins.map(r => {
      let sum = 0, weight = 0; for (let i = 0; i < r.days.length; i++) { const d = r.days[Math.floor(random() * r.days.length)]; sum += d.sum; weight += d.weight; }
      return sum / weight;
    })));
    return { target: target.name, objective, originGains: gains, meanGain: mean(gains), interval95: [quantile(samples, 0.025), quantile(samples, 0.975)], positiveFraction: mean(samples.map(v => Number(v > 0))) };
  }));
  fs.writeFileSync(path.join(output, "paired-day-audit.json"), JSON.stringify({ seed, replicates, method: "Resample UTC decision days within each origin; equal origin weights. Positive loss gain favors the added feature group. Descriptive intervals for dependent overlapping paths.", paired }, null, 2));
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, paired, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
  console.log(JSON.stringify({ paired, elapsedSec: (performance.now() - started) / 1000 }));
}
if (process.argv[1] && path.resolve(process.argv[1]) === __filename) main();

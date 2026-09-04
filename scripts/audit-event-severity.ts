/** Descriptive diagnosis of severity inside the existing large/sign groups.
 * Realized groups are used only for reporting conditional errors, never inputs. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, type EventDistribution, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, predictEventSizeSigns, reweightEventSizeSigns, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const sourceId = arg("forecast"), outputId = arg("output");
if (!sourceId || !outputId) throw new Error("Specify forecast and new severity audit output");
const source = path.resolve(root, "data/benchmarks", sourceId), output = path.resolve(root, "data/benchmarks", outputId);
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
const fc = read(source, "config.json"), base: string = fc.source, sc = read(base, "config.json"), summary = read(base, "summary.json");
if (fc.contract !== "event-size-sign-forecast-screen-v1" || sc.sampling !== "chain") throw new Error("Severity audit requires chained size/sign forecasts");
const ids = arg("windows").split(",").filter(Boolean);
if (ids.some(id => !summary.some((s: any) => s.window.id === id))) throw new Error("Unknown window");
const fingerprint = createHash("sha256").update(fs.readFileSync(path.join(base, "config.json")));
for (const row of summary) fingerprint.update(fs.readFileSync(path.join(base, `${row.window.id}-model.json`)));
if (fingerprint.digest("hex") !== fc.sourceHash) throw new Error("Forecast source changed");
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, base, sourceHash: fc.sourceHash,
  diagnostic: "Volatility bins use phase-specific training quantiles .5/.9/.99; realized sign/size groups only stratify completed errors. No head or policy is selected or fitted from this audit." }, null, 2));
const sources = ["scripts/audit-event-severity.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-completed-history.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(sources.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const mean = (v: number[]) => v.length ? v.reduce((s, x) => s + x, 0) / v.length : null;
const correlation = (x: number[], y: number[]) => {
  if (x.length < 3) return null;
  const xm = mean(x)!, ym = mean(y)!;
  const xx = x.reduce((s, v) => s + (v - xm) ** 2, 0), yy = y.reduce((s, v) => s + (v - ym) ** 2, 0);
  return xx && yy ? x.reduce((s, v, i) => s + (v - xm) * (y[i] - ym), 0) / Math.sqrt(xx * yy) : null;
};
const results = [];
for (const entry of summary) {
  const { window } = entry, id = window.id;
  if (ids.length && !ids.includes(id)) continue;
  const start = performance.now(), s = read(base, `${id}-model.json`), h = read(source, `${id}-model.json`);
  if (!h.selected || h.selected.gate || !h.head?.gate || h.trainEnd !== s.trainEnd || h.selectionTrainingEnd !== s.selectionTrainingEnd
    || s.trainEnd > window.startTime || s.calibrationEnd > window.startTime) throw new Error("Invalid severity source or chronology");
  const from = window.startTime - (sc.calibrationDays + sc.trainDays) * DAY;
  const c = loadEventCandles(from - 2 * DAY, window.endTime + DAY);
  const excluded = sc.trainingIsolation === "causal" ? [window] : sc.excludedWindows;
  const samples = (from: number, to: number) => makeSamples(c, sc.clock, from, to, excluded, sc.stride, "chain", sc.featureNames);
  // Test samples must be scored, rather than purged as training samples.
  const testSamples = makeSamples(c, sc.clock, window.startTime, window.endTime, [], sc.stride, "chain", sc.featureNames);
  const evaluate = (rows: MoveSample[], model: EventDistribution, head: EventSizeSignHead, cuts: number[]) => {
    let memory: EventCompletedHistory | undefined, previousEnd = -1;
    const severity = model.kernels.map(kernel => [2, 3].map(group => {
      const atoms = kernel.filter(a => eventSizeSignGroup(a.return, head.thresholdLogBps) === group);
      const mass = atoms.reduce((s, a) => s + a.probability, 0);
      return mass ? atoms.reduce((s, a) => s + a.probability * Math.abs(Math.log1p(a.return)) * 1e4, 0) / mass : null;
    }));
    return rows.map(row => {
      const at = c[row.start].openTime + 60_000, fast = eventFastVolatilityFeatures(c, row.start), rv5 = Math.expm1(fast[1]);
      if (h.selected.eventHistory && row.start !== previousEnd) memory = new EventCompletedHistory(at);
      const features = [...row.features, ...(h.selected.fastVolatility ? fast : []), ...(memory?.features(at) ?? [])];
      const probabilities = predictEventSizeSigns(head, features);
      memory?.observe({ ...row, originTime: at, availableAt: c[row.end].openTime + 60_000 }, c[row.end].openTime + 60_000);
      previousEnd = row.end;
      const leaf = eventLeaf(model, row.features), group = eventSizeSignGroup(row.return, head.thresholdLogBps);
      const kernel = reweightEventSizeSigns(model.kernels[leaf], head.thresholdLogBps, probabilities, h.selected.blend);
      const nextRv5 = Math.expm1(eventFastVolatilityFeatures(c, row.end)[1]);
      return { time: at, leaf, group, rv5, band: cuts.filter(cut => rv5 > cut).length,
        duration: row.duration, nextBand: cuts.filter(cut => nextRv5 > cut).length,
        realizedLogReturnBps: Math.log1p(row.return) * 1e4,
        baseExpectedReturnBps: model.kernels[leaf].reduce((s, a) => s + a.probability * a.return, 0) * 1e4,
        expectedReturnBps: kernel.reduce((s, a) => s + a.probability * a.return, 0) * 1e4,
        predictedDownSeverityBps: severity[leaf][0], predictedUpSeverityBps: severity[leaf][1],
        largeProbability: probabilities[2] + probabilities[3],
        extremeProbability: kernel.reduce((s, a) => s + a.probability * Number(Math.abs(Math.log1p(a.return)) * 1e4 >= 2 * head.thresholdLogBps), 0) };
    });
  };
  const phases = [];
  for (const phase of ["calibration", "test"] as const) {
    const training = phase === "calibration" ? samples(from, s.selectionTrainingEnd) : samples(s.trainStart, s.trainEnd);
    const rows = phase === "calibration" ? samples(s.policyCalibrationStart, s.calibrationEnd) : testSamples;
    const head = (phase === "calibration" ? h.selectionHead : h.head) as EventSizeSignHead;
    const model = (phase === "calibration" ? s.selectionPolicy.model : s.policy.model) as EventDistribution;
    const sorted = training.map(row => Math.expm1(eventFastVolatilityFeatures(c, row.start)[1])).sort((a, b) => a - b);
    const cuts = [0.5, 0.9, 0.99].map(q => sorted[Math.floor(q * sorted.length)]);
    const fitted = evaluate(training, model, head, cuts), tested = evaluate(rows, model, head, cuts);
    const aggregate = (rows: typeof tested) => ({ count: rows.length, bands: Array.from({ length: 4 }, (_, band) => {
      const all = rows.filter(r => r.band === band), tail = all.filter(r => r.group === 2 || r.group === 3);
      return { band, count: all.length, tailCount: tail.length, averageRv5Bps: mean(all.map(r => r.rv5)),
        predictedExtremeProbability: mean(all.map(r => r.extremeProbability)),
        actualExtremeProbability: mean(all.map(r => Number(Math.abs(r.realizedLogReturnBps) >= 2 * head.thresholdLogBps))),
        signs: [2, 3].map(group => {
          const selected = tail.filter(r => r.group === group);
          return { group, count: selected.length,
            predictedSeverityBps: mean(selected.map(r => group === 2 ? r.predictedDownSeverityBps : r.predictedUpSeverityBps).filter((v): v is number => v !== null)),
            actualSeverityBps: mean(selected.map(r => Math.abs(r.realizedLogReturnBps))) };
        }) };
    }), states: [...new Set(rows.map(r => `${r.leaf}:${r.band}`))].sort().map(key => {
      const group = rows.filter(r => `${r.leaf}:${r.band}` === key), realized = group.map(r => Math.expm1(r.realizedLogReturnBps / 1e4) * 1e4);
      const average = mean(realized)!;
      return { key, leaf: group[0].leaf, band: group[0].band, count: group.length,
        expectedReturnBps: mean(group.map(r => r.expectedReturnBps)), baseExpectedReturnBps: mean(group.map(r => r.baseExpectedReturnBps)),
        realizedReturnBps: average, standardErrorBps: group.length > 1 ? Math.sqrt(realized.reduce((s, r) => s + (r - average) ** 2, 0) / (group.length - 1) / group.length) : null,
        positiveFraction: mean(group.map(r => Number(r.realizedLogReturnBps > 0))), averageDuration: mean(group.map(r => r.duration)),
        nextBandProbabilities: [0, 1, 2, 3].map(band => group.filter(r => r.nextBand === band).length / group.length) };
    }), tailRv5Correlation: correlation(rows.filter(r => r.group === 2 || r.group === 3).map(r => r.rv5),
      rows.filter(r => r.group === 2 || r.group === 3).map(r => Math.abs(r.realizedLogReturnBps))) });
    phases.push({ phase, thresholdLogBps: head.thresholdLogBps, cuts, training: aggregate(fitted), evaluation: aggregate(tested),
      largestDownEvents: tested.filter(r => r.realizedLogReturnBps < 0).sort((a, b) => a.realizedLogReturnBps - b.realizedLogReturnBps).slice(0, 8) });
    fs.writeFileSync(path.join(output, `${id}-${phase}.json`), JSON.stringify(tested));
    fs.writeFileSync(path.join(output, `${id}-${phase}-training.json`), JSON.stringify(fitted));
  }
  const result = { window, phases, elapsedSec: (performance.now() - start) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ ...result, phases: result.phases.map(p => ({ phase: p.phase, cuts: p.cuts,
    trainingSamples: p.training.count, evaluationSamples: p.evaluation.count, trainingStates: p.training.states.length, evaluationStates: p.evaluation.states.length })) }));
}

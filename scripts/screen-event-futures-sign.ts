/** One-origin, matched-event forecast test before any futures-policy work. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, eventMoveLabel, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, mixEventSizeSigns, predictEventSizeSigns,
  trainEventSizeSign, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { eventProbabilityFromReturnWeight, trainEventSign, predictEventSign, type EventSignHead } from "../packages/bot-algo/src/event-sign.js";
import { eventFuturesFeatures, loadEventFuturesRows, EVENT_FUTURES_PRICE_INPUTS, EVENT_FUTURES_FLOW_INPUTS } from "./event-futures-basis.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";
const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const sizeRegimes = process.argv.includes("--size-regimes"), quantile = 0.75;
const returnWeighted = process.argv.includes("--return-weighted");
if (returnWeighted && sizeRegimes) throw new Error("Choose one head factorization");
if (!arg("source") || !arg("phase") || !arg("output")) throw new Error("Specify saved joint policy, validation phase and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
const config = read(source, "config.json"), original = read(config.source, "config.json"), sc = read(original.source, "config.json");
if (config.contract !== "event-volatility-law-policy-v1" || arg("phase") === "final") throw new Error("Requires prior validation origin, not final test");
const saved = read(config.source, `${config.window.id}-${arg("phase")}-model.json`), phase = read(source, `${arg("phase")}-scores.json`).phase;
const model = read(source, `${arg("phase")}-policy.json`).model;
if (saved.origin !== phase.startTime || saved.support.lastTarget >= phase.startTime) throw new Error("Invalid training chronology");
const c = loadEventCandles(saved.trainStart - DAY, phase.endTime + DAY);
const { rows: external, references: refs, missing, fingerprint } = loadEventFuturesRows(saved.trainStart - DAY, phase.endTime);
const hash = createHash("sha256").update(fingerprint);
const cache = new Map<number, ReturnType<typeof eventFuturesFeatures>>();
const extras = (i: number) => { if (!cache.has(i)) cache.set(i, eventFuturesFeatures(c, i, t => external.get(t))); return cache.get(i)!; };
const training = makeSamples(c, sc.clock, saved.trainStart, saved.trainEnd, [config.window], sc.stride, "chain", sc.featureNames);
const evaluation = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames);
if (training.length !== saved.support.samples) throw new Error("Original event population changed");
const train = training.filter(r => extras(r.start)), test = evaluation.filter(r => extras(r.start));
if (train.length < 100 || test.length < 100 || train.length / training.length < 0.98 || test.length / evaluation.length < 0.98)
  throw new Error(`Insufficient matched coverage: train ${train.length}/${training.length}, validation ${test.length}/${evaluation.length}`);
const magnitudes = train.filter(r => r.return !== 0).map(r => Math.abs(Math.log1p(r.return)) * 1e4).sort((a, b) => a - b);
const thresholdLogBps = magnitudes[Math.floor(quantile * magnitudes.length)];
const features = (r: MoveSample, basis: string) => {
  const ex = extras(r.start)!;
  return [...r.features, ...eventFastVolatilityFeatures(c, r.start), ...(["price", "all"].includes(basis) ? ex.price : []), ...(["flow", "all"].includes(basis) ? ex.flow : [])];
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: returnWeighted ? "event-futures-weighted-sign-screen-v1" : sizeRegimes ? "event-futures-size-sign-screen-v1" : "event-futures-sign-screen-v1", source, phase, trainStart: saved.trainStart, trainEnd: saved.trainEnd,
  ...(sizeRegimes ? { quantile, thresholdLogBps } : {}),
  inputs: { price: EVENT_FUTURES_PRICE_INPUTS, flow: EVENT_FUTURES_FLOW_INPUTS }, bases: ["spot", "price", "flow", "all"], penalties: [0.01, 0.1, 1], blends: [0.5, 1],
  sourceHash: hash.update(fs.readFileSync(path.join(source, `${arg("phase")}-policy.json`))).digest("hex"), references: refs, missing,
  coverage: { train: train.length, originalTrain: training.length, validation: test.length, originalValidation: evaluation.length },
  caveat: `Prior-origin discovery screen only. All models use the same completed events. ${returnWeighted ? "Sign cross entropy is weighted by absolute return; its output is inverted using the original conditional mean magnitudes and is not itself P(up)." : sizeRegimes ? "Size gate and conditional sign heads use the existing training-only 75th-percentile threshold." : "Sign head ignores outcome magnitude."} Joint reweighting retains conditional paths. No economic selection or final-window score.` }, null, 2));
const files = ["scripts/screen-event-futures-sign.ts", "scripts/event-futures-basis.ts", "packages/bot-algo/src/event-sign.ts", "packages/bot-algo/src/event-size-sign.ts", "scripts/research-event-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const laws = model.kernels.map((kernel: any[]) => [-1, 0, 1].map(sign => {
  const atoms = kernel.filter(a => Math.sign(a.return) === sign);
  return { mass: atoms.reduce((s, a) => s + a.probability, 0), mean: atoms.reduce((s, a) => s + a.probability * a.return, 0),
    classes: Array.from({ length: 15 }, (_, label) => atoms.reduce((s, a) => s + a.probability * Number(eventMoveLabel(a.return, a.duration, model.clock) === label), 0)) };
}));
const sizeLaws = sizeRegimes ? model.kernels.map((kernel: any[]) => Array.from({ length: 5 }, (_, group) => {
  const atoms = kernel.filter(a => eventSizeSignGroup(a.return, thresholdLogBps) === group);
  return { mass: atoms.reduce((s, a) => s + a.probability, 0), mean: atoms.reduce((s, a) => s + a.probability * a.return, 0),
    classes: Array.from({ length: 15 }, (_, label) => atoms.reduce((s, a) => s + a.probability * Number(eventMoveLabel(a.return, a.duration, model.clock) === label), 0)) };
})) : undefined;
const score = (head?: EventSignHead | EventSizeSignHead, basis = "spot", blend = 0) => {
  let loss = 0, correct = 0, nll = 0, mse = 0, positive = 0, activeCount = 0, sizeSignLoss = 0, weightedLoss = 0, totalWeight = 0;
  const predictions = [];
  for (const r of test) {
    const leaf = eventLeaf(model, [...r.features, eventFastVolatilityFeatures(c, r.start)[1]]), law = laws[leaf];
    const active = law[0].mass + law[2].mass, base = active ? law[2].mass / active : 0.5;
    const raw = head && !sizeRegimes ? predictEventSign(head as EventSignHead, features(r, basis)) : undefined;
    const headProbability = raw !== undefined && returnWeighted && law[0].mass && law[2].mass
      ? eventProbabilityFromReturnWeight(raw, law[2].mean / law[2].mass, law[0].mean / law[0].mass) : raw;
    let probability = head && !sizeRegimes && law[0].mass && law[2].mass ? (1 - blend) * base + blend * headProbability! : base;
    let selectedLaw = law, mass = [active * (1 - probability), law[1].mass, active * probability], probabilities: number[] | undefined;
    if (sizeRegimes) {
      selectedLaw = sizeLaws![leaf];
      const baseMass = selectedLaw.map((g: any) => g.mass);
      probabilities = head ? predictEventSizeSigns(head as EventSizeSignHead, features(r, basis)) : undefined;
      mass = probabilities ? mixEventSizeSigns(baseMass, probabilities, blend) : baseMass;
      const nonzero = mass.slice(0, 4).reduce((s, p) => s + p, 0);
      probability = nonzero ? (mass[1] + mass[3]) / nonzero : 0.5;
      if (r.return !== 0) sizeSignLoss -= Math.log(Math.max(1e-12, mass[eventSizeSignGroup(r.return, thresholdLogBps)] / nonzero));
    }
    const ratio = mass.map((m, i) => selectedLaw[i].mass ? m / selectedLaw[i].mass : 1), mean = selectedLaw.reduce((s: number, g: any, i: number) => s + ratio[i] * g.mean, 0);
    const y = Number(r.return > 0);
    if (returnWeighted) {
      const gains = law[2].mass ? probability * law[2].mean / law[2].mass : 0;
      const losses = law[0].mass ? -(1 - probability) * law[0].mean / law[0].mass : 0;
      const weightedProbability = gains + losses ? gains / (gains + losses) : 0.5;
      weightedLoss -= Math.abs(r.return) * Math.log(Math.max(1e-12, y ? weightedProbability : 1 - weightedProbability));
      totalWeight += Math.abs(r.return);
    }
    if (r.return !== 0) {
      activeCount++; positive += y;
      loss -= Math.log(Math.max(1e-12, y ? probability : 1 - probability)); correct += Number((probability >= 0.5) === !!y);
    }
    nll -= Math.log(Math.max(1e-12, selectedLaw.reduce((s: number, g: any, i: number) => s + g.classes[r.label] * ratio[i], 0)));
    mse += (r.return - mean) ** 2;
    predictions.push({ time: c[r.start].openTime + 60_000, availableAt: c[r.end].openTime + 60_000, probability, mean, return: r.return, leaf,
      ...(sizeRegimes ? { probabilities, mass } : {}), ...(returnWeighted ? { weightProbability: raw } : {}) });
  }
  return { count: test.length, activeCount, signLoss: loss / activeCount, accuracy: correct / activeCount, positiveRate: positive / activeCount,
    nll: nll / test.length, mse: mse / test.length, ...(sizeRegimes ? { sizeSignLoss: sizeSignLoss / activeCount } : {}),
    ...(returnWeighted ? { returnWeightedLoss: weightedLoss / totalWeight } : {}), predictions };
};
const started = performance.now(), models: any[] = [], results: any[] = [];
const { predictions, ...base } = score(); results.push({ basis: "joint-law", ...base });
for (const basis of ["spot", "price", "flow", "all"]) for (const penalty of [0.01, 0.1, 1]) {
  const rows = train.map(r => ({ features: features(r, basis), return: r.return }));
  const head = sizeRegimes ? trainEventSizeSign(rows, penalty, quantile) : trainEventSign(rows, penalty, returnWeighted ? rows.map(r => Math.abs(r.return)) : undefined);
  if (sizeRegimes && (head as EventSizeSignHead).thresholdLogBps !== thresholdLogBps) throw new Error("Size-regime threshold changed");
  models.push({ basis, penalty, head });
  for (const blend of [0.5, 1]) {
    const { predictions, ...metrics } = score(head, basis, blend);
    results.push({ basis, penalty, blend, ...metrics });
    fs.writeFileSync(path.join(output, `${basis}-${penalty}-${blend}-predictions.json`), JSON.stringify(predictions));
  }
}
fs.writeFileSync(path.join(output, "heads.json"), JSON.stringify(models));
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ phase, results, elapsedSec: (performance.now() - started) / 1000 }, null, 2));
for (const basis of ["joint-law", "spot", "price", "flow", "all"]) console.log(JSON.stringify(results.filter(r => r.basis === basis)
  .sort((a, b) => returnWeighted ? a.returnWeightedLoss - b.returnWeightedLoss : sizeRegimes ? a.sizeSignLoss - b.sizeSignLoss : a.signLoss - b.signLoss)[0]));

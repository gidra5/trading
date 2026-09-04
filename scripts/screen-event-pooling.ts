/** Test whether older completed episodes improve the existing sign/size heads.
 * Reuse strictly prior rolling-origin state maps; never fit on evaluated labels. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, eventMoveLabel, type MoveSample, type EventDistribution } from "../packages/bot-algo/src/event-distribution.js";
import { EventCompletedHistory } from "../packages/bot-algo/src/event-completed-history.js";
import { trainEventSign } from "../packages/bot-algo/src/event-sign.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, mixEventSizeSigns, predictEventSizeSigns, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { eventRefitOrigins } from "./research-event-refits.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const sourceId = arg("source"), outputId = arg("output"), windowId = arg("window");
if (!sourceId || !outputId || !windowId) throw new Error("Specify rolling-refit source, window and new output");
const source = path.resolve(root, "data/benchmarks", sourceId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), sc = read(config.source, "config.json");
if (config.contract !== "rolling-event-refit-v1" || sc.trainingIsolation !== "causal" || sc.sampling !== "chain") throw new Error("Requires causal rolling refits");
const setting = config.settings.find((s: any) => s.window.id === windowId);
if (!setting || windowId.startsWith("fit-")) throw new Error("Unknown or excluded fit window");
const { window } = setting, origins = eventRefitOrigins(window.startTime, config.foldCount, config.foldDays);
const phases = [...origins, { ...window, id: "final" }];
const saved = phases.map(p => read(source, `${windowId}-${p.id}-model.json`));
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
for (const p of phases) hash.update(fs.readFileSync(path.join(source, `${windowId}-${p.id}-model.json`)));
const shardDirectory = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1m");
const firstShard = fs.readdirSync(shardDirectory).filter(f => /^\d{4}-\d{2}-\d{2}\.json$/.test(f)).sort()[0];
if (!firstShard) throw new Error("No chronological candle shards");
const earliest = Date.parse(firstShard.slice(0, 10)), poolStart = earliest + DAY;
const choices = ["recent", "year-large", "all-large", "year-all", "all-all", "year-gate", "all-gate"] as const;
type Choice = typeof choices[number];
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-pooling-forecast-v1", source, sourceHash: hash.digest("hex"), window, choices,
  selection: "Minimum mean joint 15-class NLL over all three earlier origins, equally weighted; unchanged wins ties",
  training: "Keep each origin's recent threshold, component penalties and feature prefixes. Add completed older events from the preceding year or all cached earlier history; current 120-day episode retains its exact event chain. No evaluated target enters its fit.",
  poolStart, approximation: "Reweight current joint paths using pooled size/sign probabilities; magnitude and duration laws within each group remain the recent incumbent law",
  caveat: "Forecast selection on repeatedly examined research windows. Existing tree/head hyperparameters were previously selected; this is not a sealed nested outer holdout or evidence of trading utility." }, null, 2));
const sources = ["scripts/screen-event-pooling.ts", "scripts/research-event-policy.ts", "scripts/research-event-refits.ts", "packages/bot-algo/src/event-sign.ts", "packages/bot-algo/src/event-size-sign.ts", "packages/bot-algo/src/event-completed-history.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(sources.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const started = performance.now(), c = loadEventCandles(earliest, window.endTime + DAY);
const history = makeSamples(c, sc.clock, poolStart, window.startTime, [window], sc.stride, "chain", sc.featureNames);
const at = (r: MoveSample) => c[r.start].openTime + 60_000;
const available = (r: MoveSample) => c[r.end].openTime + 60_000;
console.log(JSON.stringify({ event: "pooled-history", candles: c.length, samples: history.length, seconds: (performance.now() - started) / 1000 }));

function augment(rows: MoveSample[]) {
  let previousEnd = -1, memory: EventCompletedHistory | undefined;
  return rows.map(row => {
    if (setting.head.eventHistory && row.start !== previousEnd) memory = new EventCompletedHistory(at(row));
    const fast = eventFastVolatilityFeatures(c, row.start), features = [...row.features, ...(setting.head.fastVolatility ? fast : []), ...(memory?.features(at(row)) ?? [])];
    memory?.observe({ ...row, originTime: at(row), availableAt: available(row) }, available(row));
    previousEnd = row.end;
    return { row, features, rv5: Math.expm1(fast[1]) };
  });
}
type Input = ReturnType<typeof augment>[number];
function train(rows: Input[], base: EventSizeSignHead): EventSizeSignHead {
  const active = rows.filter(r => r.row.return !== 0), large = (r: Input) => eventSizeSignGroup(r.row.return, base.thresholdLogBps) >= 2;
  const ordinary = active.filter(r => !large(r)), tails = active.filter(large);
  const fit = (rows: Input[], component: EventSizeSignHead["gate"], gate = false) => trainEventSign(rows.map(r => ({
    features: r.features.slice(0, component.means.length), return: gate ? (large(r) ? 1 : -1) : r.row.return,
  })), component.penalty);
  return { ...base, samples: active.length, gate: fit(active, base.gate, true), ordinarySign: fit(ordinary, base.ordinarySign), largeSign: fit(tails, base.largeSign) };
}
function census(rows: Input[], model: EventDistribution, threshold: number, cut: number) {
  const high = rows.filter(r => r.rv5 > cut), tails = rows.filter(r => Math.abs(Math.log1p(r.row.return)) * 1e4 >= threshold);
  const dates = [...new Set(high.map(r => new Date(at(r.row)).toISOString().slice(0, 10)))];
  return { samples: rows.length, large: tails.length, highVolatility: high.length, highVolatilityDays: dates.length,
    firstOrigin: at(rows[0].row), lastTarget: available(rows.at(-1)!.row),
    states: model.kernels.map((_, leaf) => ({ leaf, count: high.filter(r => eventLeaf(model, r.row.features) === leaf).length })) };
}
function evaluate(rows: Input[], model: EventDistribution, head: EventSizeSignHead, cut: number, trace = false) {
  const laws = model.kernels.map(kernel => {
    const groups = Array.from({ length: 5 }, () => ({ mass: 0, mean: 0, logDuration: 0, classes: new Array<number>(15).fill(0) }));
    for (const a of kernel) {
      const g = groups[eventSizeSignGroup(a.return, head.thresholdLogBps)];
      g.mass += a.probability; g.mean += a.probability * a.return; g.logDuration += a.probability * Math.log1p(a.duration);
      g.classes[eventMoveLabel(a.return, a.duration, model.clock)] += a.probability;
    }
    return groups;
  });
  const counters = () => ({ count: 0, active: 0, nll: 0, mse: 0, signLoss: 0, logDurationMse: 0, largeSignLoss: 0, large: 0, gateLoss: 0, ordinarySignLoss: 0 });
  const all = counters(), high = counters(), predictions = [];
  for (const input of rows) {
    const { row } = input, leaf = eventLeaf(model, row.features), groups = laws[leaf];
    const probabilities = predictEventSizeSigns(head, input.features), baseMass = groups.map(g => g.mass), mass = mixEventSizeSigns(baseMass, probabilities, setting.head.blend);
    const ratios = mass.map((v, g) => baseMass[g] ? v / baseMass[g] : 1);
    const mean = groups.reduce((s, g, i) => s + g.mean * ratios[i], 0), duration = groups.reduce((s, g, i) => s + g.logDuration * ratios[i], 0);
    const probability = groups.reduce((s, g, i) => s + g.classes[row.label] * ratios[i], 0), active = mass.slice(0, 4).reduce((s, p) => s + p, 0);
    const sign = active ? (mass[1] + mass[3]) / active : 0.5, largeUp = mass[2] + mass[3] ? mass[3] / (mass[2] + mass[3]) : 0.5;
    const isLarge = Math.abs(Math.log1p(row.return)) * 1e4 >= head.thresholdLogBps;
    for (const stats of input.rv5 > cut ? [all, high] : [all]) {
      stats.count++; stats.nll -= Math.log(Math.max(1e-12, probability)); stats.mse += (row.return - mean) ** 2;
      if (row.return !== 0) {
        stats.active++; stats.signLoss -= Math.log(Math.max(1e-12, row.return > 0 ? sign : 1 - sign));
        const gate = (mass[2] + mass[3]) / active, ordinaryUp = mass[1] / (mass[0] + mass[1]);
        stats.gateLoss -= Math.log(Math.max(1e-12, isLarge ? gate : 1 - gate));
        if (!isLarge) stats.ordinarySignLoss -= Math.log(Math.max(1e-12, row.return > 0 ? ordinaryUp : 1 - ordinaryUp));
      }
      stats.logDurationMse += (Math.log1p(row.duration) - duration) ** 2;
      if (isLarge) { stats.large++; stats.largeSignLoss -= Math.log(Math.max(1e-12, row.return > 0 ? largeUp : 1 - largeUp)); }
    }
    if (trace) predictions.push({ time: at(row), availableAt: available(row), leaf, rv5: input.rv5, realizedReturnBps: row.return * 1e4,
      expectedReturnBps: mean * 1e4, duration: row.duration, expectedLogDuration: duration, sizeSignProbabilities: probabilities });
  }
  const normalize = (s: ReturnType<typeof counters>) => ({ count: s.count, nll: s.count ? s.nll / s.count : null, mse: s.count ? s.mse / s.count : null,
    active: s.active, signLoss: s.active ? s.signLoss / s.active : null, logDurationMse: s.count ? s.logDurationMse / s.count : null,
    gateLoss: s.active ? s.gateLoss / s.active : null, ordinarySignLoss: s.active > s.large ? s.ordinarySignLoss / (s.active - s.large) : null,
    large: s.large, largeSignLoss: s.large ? s.largeSignLoss / s.large : null });
  return { all: normalize(all), high: normalize(high), predictions };
}

const results: any[] = [];
let selected: Choice = "recent";
for (const [i, phase] of phases.entries()) {
  const start = performance.now(), fit = saved[i], model: EventDistribution = fit.base.model;
  if (fit.origin !== phase.startTime || fit.trainEnd > phase.startTime || fit.support.lastTarget >= phase.startTime) throw new Error("Invalid rolling origin chronology");
  if (phase.id === "final") {
    const scores = choices.map(choice => ({ choice, meanNll: results.reduce((s, r) => s + r.scores.find((s: any) => s.choice === choice).all.nll, 0) / results.length }));
    scores.sort((a, b) => a.meanNll - b.meanNll); selected = scores[0].choice;
    fs.writeFileSync(path.join(output, "selection.json"), JSON.stringify({ selected, scores, origins: origins.map(o => o.id) }, null, 2));
  }
  const recentRows = makeSamples(c, sc.clock, fit.trainStart, fit.trainEnd, [window], sc.stride, "chain", sc.featureNames);
  if (recentRows.length !== fit.support.samples) throw new Error("Recent fit population differs from saved origin");
  const recent = augment(recentRows), sortedVol = recent.map(r => r.rv5).sort((a, b) => a - b), cut = sortedVol[Math.floor(sortedVol.length * 0.9)];
  const pools = { year: augment([...history.filter(r => at(r) >= phase.startTime - 365 * DAY && available(r) < fit.trainStart), ...recentRows]),
    all: augment([...history.filter(r => available(r) < fit.trainStart), ...recentRows]) };
  if (Object.values(pools).some(rows => rows.some(r => available(r.row) >= phase.startTime))) throw new Error("Future target in pooled fit");
  const heads: Record<Choice, EventSizeSignHead> = { recent: fit.head,
    "year-all": train(pools.year, fit.head), "all-all": train(pools.all, fit.head), "year-large": fit.head, "all-large": fit.head, "year-gate": fit.head, "all-gate": fit.head };
  heads["year-large"] = { ...fit.head, largeSign: heads["year-all"].largeSign };
  heads["all-large"] = { ...fit.head, largeSign: heads["all-all"].largeSign };
  heads["year-gate"] = { ...fit.head, gate: heads["year-all"].gate };
  heads["all-gate"] = { ...fit.head, gate: heads["all-all"].gate };
  fs.writeFileSync(path.join(output, `${phase.id}-heads.json`), JSON.stringify({ phase, headSetting: setting.head, selected, heads,
    recentSupport: fit.support, pooledSupport: { year: census(pools.year, model, fit.head.thresholdLogBps, cut), all: census(pools.all, model, fit.head.thresholdLogBps, cut) } }));
  const evaluationRows = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames), inputs = augment(evaluationRows);
  const scores = choices.map(choice => {
    const { predictions, ...score } = evaluate(inputs, model, heads[choice], cut, phase.id === "final" && choice === selected);
    if (predictions.length) fs.writeFileSync(path.join(output, "predictions.json"), JSON.stringify(predictions));
    return { choice, ...score };
  });
  const result = { phase, selected: phase.id === "final" ? selected : undefined, scores,
    census: { recent: census(recent, model, fit.head.thresholdLogBps, cut), year: census(pools.year, model, fit.head.thresholdLogBps, cut), all: census(pools.all, model, fit.head.thresholdLogBps, cut) },
    elapsedSec: (performance.now() - start) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ event: "pooling-origin", phase: phase.id, selected: result.selected,
    scores: scores.map(s => ({ choice: s.choice, nll: s.all.nll, mse: s.all.mse, gateLoss: s.all.gateLoss, durationMse: s.all.logDurationMse, highMse: s.high.mse })),
    samples: Object.fromEntries(Object.entries(result.census).map(([k, v]) => [k, v.samples])), elapsedSec: result.elapsedSec }));
}

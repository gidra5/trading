/** Compare modeled multi-event returns with completed earlier-origin paths.
 * This diagnostic never selects a policy or reads final-window outcomes. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, type EventDistribution } from "../packages/bot-algo/src/event-distribution.js";
import { eventFastVolatilityFeatures, eventSizeSignGroup, mixEventSizeSigns, predictEventSizeSigns, type EventSizeSignHead } from "../packages/bot-algo/src/event-size-sign.js";
import { eventFuturesFeatures, loadEventFuturesRows } from "./event-futures-basis.js";
import { loadEventCandles, makeSamples } from "./research-event-policy.js";
import { eventRefitOrigins } from "./research-event-refits.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify projected policy comparison and new output");
const source = path.resolve(root, "data/benchmarks", arg("source")), output = path.resolve(root, "data/benchmarks", arg("output"));
const read = (dir: string, file: string) => JSON.parse(fs.readFileSync(path.join(dir, file), "utf8"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = read(source, "config.json"), jointConfig = read(config.source, "config.json"), originConfig = read(jointConfig.source, "config.json");
const sc = read(originConfig.source, "config.json");
if (!config.projectContinuation || !config.sizeRegimes) throw new Error("Requires projected size/sign policy");
const phases = eventRefitOrigins(config.window.startTime, originConfig.foldCount, originConfig.foldDays), horizons = [1, 2, 4, 8];
const candidates = ["original", "projected"] as const;
fs.mkdirSync(output, { recursive: true });
const fingerprint = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ contract: "event-continuation-audit-v1", source, phases, horizons,
  target: "Sum of log returns over contiguous completed event paths. Current forecast identical; only later state law changes.",
  caveat: "Overlapping path diagnostics, not independent samples or realized policy utility; prior calibration origins only, no final outcomes or policy selection." }, null, 2));
const files = ["scripts/audit-event-continuation.ts", "scripts/event-futures-basis.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-size-sign.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));

const c = loadEventCandles(phases[0].startTime - DAY, phases.at(-1)!.endTime + DAY);
const external = loadEventFuturesRows(phases[0].startTime - DAY, phases.at(-1)!.endTime);
const summarize = (rows: any[]) => Object.fromEntries(candidates.map(name => [name, {
  count: rows.length,
  meanPredictedBps: rows.reduce((s, r) => s + r[name], 0) / rows.length,
  meanRealizedBps: rows.reduce((s, r) => s + r.actual, 0) / rows.length,
  mseBps: rows.reduce((s, r) => s + (r[name] - r.actual) ** 2, 0) / rows.length,
  zeroForecastMseBps: rows.reduce((s, r) => s + r.actual ** 2, 0) / rows.length,
  directionAccuracy: rows.reduce((s, r) => s + Number(Math.sign(r[name]) === Math.sign(r.actual)), 0) / rows.length,
}]));
const results = [], started = performance.now();
for (const phase of phases) {
  const selectedName = "futures-size-sign-projected", setting = config.settings[selectedName];
  const original: EventDistribution = read(config.source, `${phase.id}-policy.json`).model;
  const projected: EventDistribution = read(source, `${phase.id}-${selectedName}-policy.json`).model;
  const head: EventSizeSignHead = read(source, `${phase.id}-heads.json`).heads[selectedName];
  for (const file of [`${phase.id}-${selectedName}-policy.json`, `${phase.id}-heads.json`]) fingerprint.update(fs.readFileSync(path.join(source, file)));
  const moments = (model: EventDistribution) => model.kernels.map(kernel => Array.from({ length: 5 }, (_, group) => {
    const atoms = kernel.filter(a => eventSizeSignGroup(a.return, head.thresholdLogBps) === group);
    const mass = atoms.reduce((s, a) => s + a.probability, 0), next = new Array<number>(model.kernels.length).fill(0);
    for (const a of atoms) next[a.next] += a.probability;
    return { mass, logReturn: atoms.reduce((s, a) => s + a.probability * Math.log1p(a.return), 0), next };
  }));
  const originalMoments = moments(original), projectedMoments = moments(projected);
  const future = (groups: ReturnType<typeof moments>) => {
    const values = [new Array<number>(groups.length).fill(0)];
    for (let h = 1; h < Math.max(...horizons); h++) values.push(groups.map(gs => gs.reduce((sum, g) => sum + g.logReturn
      + g.next.reduce((s, p, next) => s + p * values[h - 1][next], 0), 0)));
    return values;
  };
  const continuation = { original: future(originalMoments), projected: future(projectedMoments) };
  const evaluation = makeSamples(c, sc.clock, phase.startTime, phase.endTime, [], sc.stride, "chain", sc.featureNames);
  const predictions = evaluation.map(r => {
    const ex = eventFuturesFeatures(c, r.start, t => external.rows.get(t)); if (!ex) throw new Error("Missing completed external input");
    const inputs = [...r.features, ...eventFastVolatilityFeatures(c, r.start), ...(["price", "all"].includes(setting.basis) ? ex.price : []), ...(["flow", "all"].includes(setting.basis) ? ex.flow : [])];
    const leaf = eventLeaf(original, [...r.features, eventFastVolatilityFeatures(c, r.start)[1]]);
    const probabilities = predictEventSizeSigns(head, inputs), groups = originalMoments[leaf];
    const mass = mixEventSizeSigns(groups.map(g => g.mass), probabilities, setting.blend);
    const conditional = groups.map((g, i) => ({ logReturn: g.mass ? g.logReturn / g.mass : 0, next: g.next.map(p => g.mass ? p / g.mass : 0), mass: mass[i] }));
    return { leaf, predictions: Object.fromEntries(horizons.map(h => [h, Object.fromEntries(candidates.map(name => [name,
      conditional.reduce((s, g) => s + g.mass * (g.logReturn + g.next.reduce((v, p, next) => v + p * continuation[name][h - 1][next], 0)), 0) * 1e4]))])) };
  });
  const paths = horizons.flatMap(horizon => evaluation.flatMap((r, i) => {
    const moves = evaluation.slice(i, i + horizon);
    if (moves.length !== horizon || moves.some((m, j) => j && m.start !== moves[j - 1].end)) return [];
    return [{ time: c[r.start].openTime + 60_000, availableAt: c[moves.at(-1)!.end].openTime + 60_000,
      leaf: predictions[i].leaf, horizon, actual: moves.reduce((s, m) => s + Math.log1p(m.return), 0) * 1e4, ...predictions[i].predictions[horizon] }];
  }));
  if (paths.filter(r => r.horizon === 1).some(r => r.original !== r.projected)) throw new Error("Current law changed");
  fs.writeFileSync(path.join(output, `${phase.id}-paths.json`), JSON.stringify(paths));
  const result = { phase, horizons: horizons.map(horizon => ({ horizon, ...summarize(paths.filter(r => r.horizon === horizon)) })),
    statesAtEight: original.kernels.map((_, leaf) => ({ leaf, ...summarize(paths.filter(r => r.horizon === 8 && r.leaf === leaf)) })) };
  results.push(result); console.log(JSON.stringify({ phase: phase.id, horizons: result.horizons }));
}
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ results, sourceHash: fingerprint.digest("hex"), externalFingerprint: external.fingerprint, elapsedSec: (performance.now() - started) / 1000 }, null, 2));

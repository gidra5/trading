/** HINDSIGHT diagnostic only. Never use its output as evidence of tradable alpha. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventLeaf, type EventDistribution, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { buildEventPolicy, type SerializedEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { loadEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";
import { EventSecondBasis } from "./event-second-basis.js";

const root = path.resolve(__dirname, "..");
const argument = (key: string, fallback = "") => {
  const at = process.argv.indexOf(`--${key}`); return at < 0 ? fallback : process.argv[at + 1];
};
const sourceId = argument("source"), outputId = argument("output");
if (!sourceId || !outputId) throw new Error("Specify --source and a new --output for this hindsight diagnostic");
const source = path.resolve(root, "data/benchmarks", sourceId), output = path.resolve(root, "data/benchmarks", outputId);
if (fs.existsSync(output)) throw new Error("Choose a new diagnostic output directory");
const config = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
const summary = JSON.parse(fs.readFileSync(path.join(source, "summary.json"), "utf8")) as Array<{
  window: { id: string; startTime: number; endTime: number }; chosenDepth: number; test: { returnPct: number; maxDrawdownPct: number };
}>;
const requested = argument("windows", "all"), ids = requested.split(",");
if (requested !== "all" && ids.some(id => !summary.some(r => r.window.id === id))) throw new Error("Unknown source window");
if (requested === "all" && config.windows.some((id: string) => !summary.some(r => r.window.id === id))) throw new Error("Source run is incomplete");
const rows = summary.filter(r => requested === "all" || ids.includes(r.window.id));
const second = config.secondBasisFingerprint ? new EventSecondBasis(path.join(root, "data/runtime-cache/global-feature-basis")) : undefined;
if (second && second.fingerprint !== config.secondBasisFingerprint) throw new Error("Feature source changed");
const hash = createHash("sha256").update(fs.readFileSync(path.join(source, "config.json")));
const snapshots = rows.map(row => {
  const bytes = fs.readFileSync(path.join(source, `${row.window.id}-model.json`)); hash.update(bytes);
  return JSON.parse(bytes.toString()) as { policy: SerializedEventPolicy };
});
fs.mkdirSync(output, { recursive: true });
const caveat = "HINDSIGHT DIAGNOSTIC: scored-window outcomes estimate the laws and select depth. Not an out-of-sample backtest or deployable strategy.";
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, sourceHash: hash.digest("hex"), caveat, windows: rows.map(r => r.window.id) }, null, 2));
const files = ["scripts/audit-event-policy.ts", "scripts/research-event-policy.ts", "scripts/event-second-basis.ts",
  "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-run-model.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [];
for (let i = 0; i < rows.length; i++) {
  const row = rows[i], saved = snapshots[i].policy, window = row.window;
  if (window.id.startsWith("fit-") || saved.model.hidden || saved.model.runSymmetry || saved.model.runConditioned)
    throw new Error("Audit requires non-fit windows and an unrestricted frozen snapshot-state model");
  const candles = loadEventCandles(window.startTime - 2 * 86_400_000, window.endTime + 86_400_000);
  second?.attach(candles);
  const samples = makeSamples(candles, saved.model.clock, window.startTime, window.endTime, [], 1, "chain", saved.model.featureNames);
  if (!samples.length) throw new Error("No complete scored-window events");
  const groups: MoveSample[][] = saved.model.kernels.map(() => []);
  for (const s of samples) groups[eventLeaf(saved.model, s.features)].push(s);
  const refitLaw = (conditional: boolean): EventDistribution => ({ ...saved.model, meanCalibration: undefined,
    trainingSamples: samples.length, counts: groups.map(g => g.length),
    priorClasses: Array.from({ length: 15 }, (_, label) => samples.filter(s => s.label === label).length / samples.length),
    classProbabilities: groups.map((group, leaf) => {
      const selected = conditional ? group : samples;
      return selected.length ? Array.from({ length: 15 }, (_, label) => selected.filter(s => s.label === label).length / selected.length)
        : saved.model.classProbabilities[leaf];
    }),
    kernels: groups.map((group, leaf) => {
      const selected = conditional ? group : samples;
      if (!selected.length) return saved.model.kernels[leaf];
      return selected.map(s => ({ probability: 1 / selected.length, return: s.return, low: s.low, high: s.high,
        duration: s.duration, next: eventLeaf(saved.model, s.nextFeatures) }));
    }) });
  const estimates = [false, true].map(conditional => {
    const p = buildEventPolicy(refitLaw(conditional), saved.costs, { depths: config.maxDepth, referenceEquity: saved.equities[2],
      referencePrice: saved.prices[1], actionSteps: config.actionSteps });
    const depths = Array.from({ length: config.maxDepth }, (_, d) => {
      const { trace: _trace, ...metrics } = replayEventPolicy(candles, p, window.startTime, window.endTime, d + 1);
      return { depth: d + 1, ...metrics };
    });
    const best = [...depths].sort((a, b) => b.logGrowth - a.logGrowth)[0];
    return { conditional, depths, best, positiveAfterCashOption: Math.max(0, best.returnPct) };
  });
  const stateMeans = groups.map((g, leaf) => ({ leaf, count: g.length,
    historicalMeanBps: saved.model.kernels[leaf].reduce((s, a) => s + a.probability * a.return, 0) * 1e4,
    hindsightMeanBps: g.length ? g.reduce((s, r) => s + r.return, 0) / g.length * 1e4 : null }));
  const result = { caveat, window, samples: samples.length, sourceTest: row.test, stateMeans, estimates };
  results.push(result);
  fs.writeFileSync(path.join(output, `${window.id}.json`), JSON.stringify(result, null, 2));
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ event: "HINDSIGHT-DIAGNOSTIC", window: window.id, actualReturnPct: row.test.returnPct,
    states: groups.length, unconditionalReturnPct: estimates[0].best.returnPct, conditionalReturnPct: estimates[1].best.returnPct,
    conditionalDepth: estimates[1].best.depth }));
}

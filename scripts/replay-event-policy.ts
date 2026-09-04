import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventCalibrationRanges, loadEventCandles, replayEventPolicy } from "./research-event-policy.js";
import { buildEventPolicy, serializeEventPolicy, type SerializedEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { scaleEventMean } from "../packages/bot-algo/src/event-distribution.js";
import { EventSecondBasis } from "./event-second-basis.js";

const root = path.resolve(__dirname, "..");
const arg = (key: string, fallback: string) => {
  const index = process.argv.indexOf(`--${key}`); return index < 0 ? fallback : process.argv[index + 1];
};
const source = path.resolve(root, "data/benchmarks", arg("source", "event-policy-all-mean120-1x-v5"));
const sourceConfig = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
if (sourceConfig.contract !== "causal-event-tree-bellman-v1") throw new Error("Not an event-policy research artifact");
const summary = JSON.parse(fs.readFileSync(path.join(source, "summary.json"), "utf8")) as Array<{
  window: { id: string; startTime: number; endTime: number }; insufficientCalibration?: boolean;
}>;
const selected = arg("windows", "all");
if (selected === "all" && sourceConfig.windows.some((id: string) => !summary.some(r => r.window.id === id)))
  throw new Error("Source run is incomplete; wait for completion or request completed windows explicitly");
if (selected !== "all" && selected.split(",").some(id => !summary.some(r => r.window.id === id))) throw new Error("Unknown source window");
const windows = summary.filter(r => selected === "all" || selected.split(",").includes(r.window.id));
const secondBasis = sourceConfig.secondBasisFingerprint
  ? new EventSecondBasis(path.join(root, "data/runtime-cache/global-feature-basis")) : undefined;
if (secondBasis && secondBasis.fingerprint !== sourceConfig.secondBasisFingerprint) throw new Error("One-second feature source changed");
const output = path.resolve(root, "data/benchmarks", arg("output", "event-policy-replay"));
if (fs.existsSync(path.join(output, "config.json"))) throw new Error("Choose a new output name for a policy replay");
fs.mkdirSync(output, { recursive: true });
const config = { source, sourceHash: createHash("sha256").update(fs.readFileSync(path.join(source, "summary.json"))).digest("hex"),
  leverage: Number(arg("leverage", String(sourceConfig.costs.maxLeverage))),
  feeBps: Number(arg("fee-bps", String(sourceConfig.costs.feeBps))),
  slippageBps: Number(arg("slippage-bps", String(sourceConfig.costs.slippageBps))),
  actionSteps: Number(arg("action-steps", String(sourceConfig.actionSteps))),
  depths: Number(arg("bellman-depth", String(sourceConfig.maxDepth))),
  evaluationDepths: process.argv.includes("--evaluation-depths") ? arg("evaluation-depths", "").split(",").map(Number) : undefined as number[] | undefined,
  riskPenalty: Number(arg("risk-penalty", String(sourceConfig.riskPenalty ?? 0.1))),
  trainingIsolation: sourceConfig.trainingIsolation ?? "suite",
  useSelectionModel: process.argv.includes("--use-selection-model"),
  riskMarking: "worst-order intrabar OHLC drawdown envelope; separate close-only drawdown",
  windows: windows.map(r => r.window.id) };
config.evaluationDepths ??= Array.from({ length: config.depths }, (_, i) => i + 1);
if (!Number.isInteger(config.depths) || config.depths < 1 || !config.evaluationDepths.length
  || config.evaluationDepths.some(d => !Number.isInteger(d) || d < 1 || d > config.depths)
  || new Set(config.evaluationDepths).size !== config.evaluationDepths.length) throw new Error("Invalid Bellman evaluation depths");
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify(config, null, 2));
const sourceFiles = ["scripts/replay-event-policy.ts", "scripts/research-event-policy.ts", "scripts/event-second-basis.ts", "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-hidden.ts", "packages/bot-algo/src/event-run-model.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(sourceFiles.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results: unknown[] = [];
for (const row of windows) {
  const window = row.window, started = performance.now();
  if (window.id.startsWith("fit-")) throw new Error("Fit windows cannot be replayed as tests");
  const saved = JSON.parse(fs.readFileSync(path.join(source, `${window.id}-model.json`), "utf8")) as {
    policy: SerializedEventPolicy; selectionPolicy?: SerializedEventPolicy; calibrationEnd: number; policyCalibrationStart: number;
    selectionExcludedWindows?: Array<{ id: string; startTime: number; endTime: number }>;
  };
  if (saved.calibrationEnd > window.startTime || !saved.policyCalibrationStart) throw new Error("Missing/invalid calibration boundary");
  if (config.useSelectionModel && !saved.selectionPolicy) throw new Error("Source has no separate pre-refit selection model");
  const costs = { ...saved.policy.costs, maxLeverage: config.leverage, feeBps: config.feeBps, slippageBps: config.slippageBps };
  const selectedWith = saved.selectionPolicy ?? saved.policy;
  const p = buildEventPolicy(selectedWith.model, costs, { depths: config.depths,
    referenceEquity: selectedWith.equities[2], referencePrice: selectedWith.prices[1], actionSteps: config.actionSteps });
  const finalPolicy = saved.selectionPolicy && !config.useSelectionModel ? buildEventPolicy(saved.policy.model, costs, { depths: config.depths,
    referenceEquity: saved.policy.equities[2], referencePrice: saved.policy.prices[1], actionSteps: config.actionSteps }) : p;
  const scales = [-2, -1, -0.5, 0, 0.5, 1, 1.5, 2];
  const scaledPolicies = sourceConfig.onlineScale ? scales.map(scale => buildEventPolicy(scaleEventMean(saved.policy.model, scale), costs,
    { depths: config.depths, referenceEquity: saved.policy.equities[2], referencePrice: saved.policy.prices[1], actionSteps: config.actionSteps })) : [];
  const c = loadEventCandles(saved.policyCalibrationStart - 2 * 86_400_000, window.endTime + 86_400_000);
  secondBasis?.attach(c);
  const selectionExcludedWindows = saved.selectionExcludedWindows ?? sourceConfig.excludedWindows;
  const ranges = eventCalibrationRanges(saved.policyCalibrationStart, window.startTime, selectionExcludedWindows);
  const adaptationWindows = sourceConfig.onlineScale ? [8, 32, 128] : [0];
  const candidates = config.evaluationDepths.flatMap(depth => adaptationWindows.map(adaptationWindow => {
    const runs = ranges.map(r => replayEventPolicy(c, p, r.startTime, r.endTime, depth, adaptationWindow
      ? { adaptive: { policies: scaledPolicies, scales, window: adaptationWindow } } : undefined));
    const logGrowth = runs.reduce((s, r) => s + r.logGrowth, 0), drawdown = Math.max(0, ...runs.map(r => r.maxDrawdownPct)) / 100;
    return { depth, adaptationWindow, logGrowth, drawdown, score: logGrowth - config.riskPenalty * drawdown };
  })).sort((a, b) => b.score - a.score);
  const chosen = candidates[0], cash = !!row.insufficientCalibration || !ranges.length || chosen.score <= 0;
  const latest = ranges.at(-1);
  const initialPairs = chosen.adaptationWindow && latest
    ? replayEventPolicy(c, p, latest.startTime, latest.endTime, chosen.depth,
      { adaptive: { policies: scaledPolicies, scales, window: chosen.adaptationWindow } }).adaptationPairs : undefined;
  const adaptive = chosen.adaptationWindow ? { policies: scaledPolicies, scales, window: chosen.adaptationWindow, initialPairs } : undefined;
  // Persist the chosen policy before evaluating the scored interval.
  fs.writeFileSync(path.join(output, `${window.id}-model.json`), JSON.stringify({ policy: serializeEventPolicy(finalPolicy),
    selectionPolicy: saved.selectionPolicy ? serializeEventPolicy(p) : undefined,
    selectionExcludedWindows,
    chosenDepth: chosen.depth, cash, adaptive: chosen.adaptationWindow ? { scales, window: chosen.adaptationWindow, initialPairs } : undefined,
    policyCalibrationStart: saved.policyCalibrationStart, calibrationEnd: window.startTime }));
  const { trace, ...test } = replayEventPolicy(c, finalPolicy, window.startTime, window.endTime, chosen.depth, { cash, trace: true, adaptive });
  fs.writeFileSync(path.join(output, `${window.id}-trades.json`), JSON.stringify(trace));
  const result = { window, chosenDepth: chosen.depth, cash, policyCalibration: candidates, test,
    bellmanConvergence: finalPolicy.tables.map(t => ({ depth: t.depth, ...t.convergence })),
    selectionConvergence: p.tables.map(t => ({ depth: t.depth, ...t.convergence })), elapsedSec: (performance.now() - started) / 1000 };
  results.push(result); fs.writeFileSync(path.join(output, `${window.id}.json`), JSON.stringify(result, null, 2));
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
  console.log(JSON.stringify({ event: "replay", window: window.id, returnPct: test.returnPct, drawdown: test.maxDrawdownPct,
    trades: test.trades, chosenDepth: chosen.depth, cash, elapsedSec: result.elapsedSec }));
}

/** Small paired replay to validate certified planning before whole-window cost. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";
const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify saved one-step screen --source and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const source = directory(arg("source")), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, file: string) => JSON.parse(fs.readFileSync(path.join(d, file), "utf8"));
const config = read(source, "config.json"); assert.equal(config.contract, "event-one-step-screen-v1");
const phase = arg("phase") ? config.phases.find((p: any) => p.id === arg("phase")) : config.phases[0];
assert.ok(phase, "Unknown saved phase");
const previous = read(source, `${phase.id}-marked-trades.json`);
const events = arg("events") === "all" ? previous.length : Number(arg("events") || 16);
const maxEvaluations = Number(arg("budget") || 32), tolerance = 1e-7;
assert.ok(Number.isInteger(events) && events > 0 && Number.isInteger(maxEvaluations) && maxEvaluations >= 2);
assert.ok(events <= previous.length);
const modes = (arg("modes") || "h1,h2").split(",");
assert.ok(modes.length && new Set(modes).size === modes.length && modes.every(m => ["h1", "h2", "countdown2", "countdown1"].includes(m)));
const end = previous[events - 1].endTime, filename = path.join(config.source, `${phase.id}-policy.json`);
const fullOrigin = events === previous.length && end === phase.endTime;
const policy = restoreEventPolicy(read(config.source, `${phase.id}-policy.json`));
const modelHash = createHash("sha256").update(fs.readFileSync(filename)).digest("hex");
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-two-step-prefix-replay-v1", source, forecastSource: config.source, phase, end, events,
  maxEvaluations, tolerance, modelHash, fullOrigin, modes, terminal: "marked",
  method: "Replay the predeclared event prefix or entire saved phase with the same fixed law and execution simulator. Compare selected modes: repeated H1; receding H2; or alternating H2/H1 countdowns starting at depth 2 or 1. Countdown H1 still reacts to the newly observed account and event state; it never commits a future order in advance. End settlement is identical. fullOrigin identifies complete phase coverage. This is a behavior diagnostic, not a stationary-optimality certificate or a new selection score." });
save("sources.json", Object.fromEntries(["scripts/replay-event-two-step.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-two-step.ts",
  "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-one-step-prepared.ts", "packages/bot-algo/src/event-holding-law.ts", "packages/bot-algo/src/event-multi-step-upper.ts", "packages/bot-algo/src/event-log-policy.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const candles = loadEventCandles(phase.startTime - 2 * 86400000, end), results = [], started = performance.now();
for (const mode of modes) {
  const depth = mode === "h1" ? 1 : 2;
  const begin = performance.now();
  const { trace, ...metrics } = replayEventPolicy(candles, policy, phase.startTime, end, depth, { trace: true,
    ...(depth === 1 ? { oneStepTerminal: "marked" as const } : { twoStep: { terminal: "marked" as const, maxEvaluations, tolerance,
      ...(mode.startsWith("countdown") ? { countdown: true as const, initialDepth: mode === "countdown1" ? 1 as const : 2 as const } : {}) } }) });
  assert.deepEqual(trace.map(r => [r.time, r.leaf]), previous.slice(0, events).map((r: any) => [r.time, r.leaf]));
  save(`${mode}-trades.json`, trace);
  const orders = trace.filter(r => r.optimizer === "two-event-bounds").map(r => r.order as { gap: number; converged: boolean });
  const bounds = depth === 2 ? { count: orders.length, converged: orders.filter(o => o.converged).length,
    maxGapBps: Math.max(...orders.map(o => o.gap! * 10000)) } : undefined;
  const row = { mode, depth, metrics, bounds, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(row); save("summary.json", { results, elapsedSec: (performance.now() - started) / 1000 }); console.log(JSON.stringify(row));
}

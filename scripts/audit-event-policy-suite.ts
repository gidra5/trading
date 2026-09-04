/** Fixed-forecast order optimality across the actual non-fit inspector catalog. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { KamaInspector } from "../apps/server/src/kama-inspector.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";
const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify complete saved-law --source and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const source = directory(arg("source")), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const config = read(source, "config.json"), old = read(source, "summary.json");
assert.equal(config.contract, "causal-event-tree-bellman-v1");
assert.ok(!config.onlineScale, "Freeze a complete static joint law, not an adaptive policy");
const catalog = new KamaInspector(path.join(root, "data")).catalog().windows.filter(w => w.id !== "latest" && !w.id.startsWith("fit-"));
assert.deepEqual([...config.windows].sort(), catalog.map(w => w.id).sort());
assert.deepEqual(old.map((r: any) => r.window.id).sort(), catalog.map(w => w.id).sort());
const requested = (arg("windows") || "all").split(","), depth = Number(arg("depth") || 1);
const budget = Number(arg("budget") || 64), tolerance = 1e-7, limitEvents = arg("limit-events") ? Number(arg("limit-events")) : undefined;
const shadowPoints = Number(arg("points") || 129), maxRootEvaluations = Number(arg("root-budget") || 2);
assert.ok([1, 2, 3].includes(depth) && Number.isInteger(budget) && budget >= 2);
assert.ok(limitEvents === undefined || Number.isInteger(limitEvents) && limitEvents > 0);
assert.ok(requested[0] === "all" || requested.every(id => catalog.some(w => w.id === id)));
const windows = catalog.filter(w => requested[0] === "all" || requested.includes(w.id));
const models = windows.map(window => {
  const file = `${window.id}-model.json`, saved = read(source, file), p = restoreEventPolicy(saved.policy);
  assert.ok(Number.isFinite(saved.trainEnd) && saved.trainEnd <= window.startTime && saved.trainStart < saved.trainEnd);
  assert.equal(saved.calibrationEnd, window.startTime);
  assert.deepEqual(old.find((r: any) => r.window.id === window.id).window, window);
  return { window, p, trainStart: saved.trainStart, trainEnd: saved.trainEnd,
    modelHash: createHash("sha256").update(fs.readFileSync(path.join(source, file))).digest("hex") };
});
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-fixed-law-suite-audit-v1", source, depth, budget, tolerance, limitEvents, shadowPoints, maxRootEvaluations,
  catalog, windows: models.map(({ p, ...m }) => ({ ...m, costs: p.costs, clock: p.model.clock, featureNames: p.model.featureNames,
    leaves: p.model.kernels.length, maxAtoms: Math.max(...p.model.kernels.map(k => k.length)) })),
  method: "Validate saved-law coverage against the current inspector catalog excluding fit-* and latest. Reuse each saved static forecast and account constraints; no fitting or calibration selection. H1 uses exact lot optimization. H2 uses a bounded full-lattice search. H3 evaluates ranked feasible candidates and certifies ONLY against its global upper bound; a candidate budget stop retains an unresolved global gap. Known cash-through-horizon lower policies may certify without path expansion. A limit-events run is an explicit prefix profile. Full runs cover the specified windows. Decision-price planning differs from next-open execution; finite-horizon certificates do not establish stationary optimality." });
save("sources.json", Object.fromEntries(["scripts/audit-event-policy-suite.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-positions.ts", "packages/bot-algo/src/event-two-step.ts",
  "packages/bot-algo/src/event-distribution.ts", "packages/bot-algo/src/event-second-features.ts",
  "packages/bot-algo/src/event-three-step.ts", "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-one-step-prepared.ts", "packages/bot-algo/src/event-holding-law.ts", "packages/bot-algo/src/event-multi-step-upper.ts", "packages/bot-algo/src/event-log-policy.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const results = [], started = performance.now();
for (const { window, p, ...metadata } of models) {
  const begin = performance.now(), c = loadEventCandles(window.startTime - 2 * 86400000, window.endTime);
  let end = window.endTime;
  if (limitEvents) {
    const clock = replayEventPolicy(c, p, window.startTime, end, 1, { trace: true, oneStepTerminal: "marked" }).trace;
    end = clock[Math.min(clock.length, limitEvents) - 1].endTime as number;
  }
  let completedDecisions = 0;
  const { trace, ...metrics } = replayEventPolicy(c, p, window.startTime, end, depth, { trace: true,
    ...(depth === 1 ? { oneStepTerminal: "marked" as const } : depth === 2
      ? { twoStep: { terminal: "marked" as const, maxEvaluations: budget, tolerance } }
      : { threeStep: { terminal: "marked" as const, maxEvaluations: budget, tolerance, shadowPoints, maxRootEvaluations },
        onDecision: (row: Record<string, unknown>) => {
          const order = row.order as { quantity: number; gap: number; converged: boolean; rootEvaluations: number };
          const progress = { window: window.id, completedDecisions: ++completedDecisions, time: row.time,
            quantity: order.quantity, gapBps: order.gap * 10000, converged: order.converged, rootEvaluations: order.rootEvaluations,
            elapsedSec: (performance.now() - begin) / 1000 };
          save("progress.json", progress); console.log(JSON.stringify(progress));
        } }) });
  const orders = trace.map(r => r.order as { feasible: boolean; value: number; gap?: number; converged?: boolean });
  const bounds = { count: orders.length, feasible: orders.filter(o => o.feasible).length,
    finiteValue: orders.filter(o => Number.isFinite(o.value)).length, certified: orders.filter(o => depth === 1 ? o.feasible : o.converged && o.feasible).length,
    maximumGapBps: depth === 1 ? null : Math.max(...orders.map(o => o.gap! * 10000)) };
  save(`${window.id}-trades.json`, trace);
  const result = { window, end, fullWindow: end === window.endTime, ...metadata, metrics, bounds, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result); save("summary.json", { results, elapsedSec: (performance.now() - started) / 1000 });
  console.log(JSON.stringify({ window: window.id, end, fullWindow: result.fullWindow, returnPct: metrics.returnPct, drawdown: metrics.maxDrawdownPct,
    trades: metrics.trades, bounds, elapsedSec: result.elapsedSec }));
}

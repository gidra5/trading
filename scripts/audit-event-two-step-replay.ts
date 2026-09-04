/** Diagnose bounded H2 replay without conflating H1 and H2 forecast horizons. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
const arg = (k: string) => { const at = process.argv.indexOf(`--${k}`); return at < 0 ? "" : process.argv[at + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify paired replay --source and new --output");
const root = path.resolve(__dirname, ".."), directory = (s: string) => path.join(root, "data/benchmarks", s);
const source = directory(arg("source")), output = directory(arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const config = read(source, "config.json"); assert.equal(config.contract, "event-two-step-prefix-replay-v1");
const policy = restoreEventPolicy(read(config.forecastSource, `${config.phase.id}-policy.json`));
const trace = read(source, "h2-trades.json"), summary = read(source, "summary.json");
if (config.fullOrigin) assert.deepEqual(summary.results.find((r: any) => r.depth === 1).metrics,
  read(config.source, "summary.json").results.find((r: any) => r.phase.id === config.phase.id).metrics.marked);
const same = (a: number, b: number) => Math.abs(a - b) < policy.costs.quantityStep * 1e-4;
const one = trace.map((r: any) => decideEventOneStep(policy.model.kernels[r.leaf],
  { equity: r.equityBefore, price: r.order.price, exposure: r.exposureBefore }, policy.costs, "marked"));
const rows = trace.map((r: any, i: number) => {
  const next = trace[i + 1], alternative = r.order.initial.candidates.find((v: any) => same(v.quantity, one[i].quantity));
  const continuationMatches = Boolean(next && same(one[i + 1].quantity, next.order.quantity));
  const bothOrdersFilled = Boolean(next && same(r.orderQuantity, r.order.quantity) && same(next.orderQuantity, next.order.quantity));
  const aligned = continuationMatches && bothOrdersFilled;
  const realizedTwoBps = aligned ? Math.log(next.equityAfter / r.equityBefore) * 10000 : null;
  return { time: new Date(r.time).toISOString(), leaf: r.leaf, exposure: r.exposureBefore, h1Quantity: one[i].quantity,
    h2Quantity: r.order.quantity, filledQuantity: r.orderQuantity, h1Exposure: one[i].exposure, h2Exposure: r.order.exposure,
    h1ValueBps: one[i].value * 10000, h2ValueBps: r.order.value * 10000,
    alternativeH2ValueBps: alternative ? alternative.value * 10000 : null,
    h2ImprovementOverH1ActionBps: alternative ? (r.order.value - alternative.value) * 10000 : null,
    gapBps: r.order.gap * 10000, continuationMatches, bothOrdersFilled, aligned, realizedTwoBps,
    optimismBps: aligned ? r.order.value * 10000 - realizedTwoBps! : null };
});
const aligned = rows.filter((r: any) => r.aligned), changed = rows.filter((r: any) => !same(r.h1Quantity, r.h2Quantity));
const result = { count: rows.length, certified: trace.filter((r: any) => r.order.converged).length,
  sameAccountChangedActions: changed.length, continuationMatches: rows.filter((r: any) => r.continuationMatches).length,
  aligned: aligned.length, meanAlignedPredictedBps: aligned.reduce((s: number, r: any) => s + r.h2ValueBps, 0) / aligned.length,
  meanAlignedRealizedBps: aligned.reduce((s: number, r: any) => s + r.realizedTwoBps, 0) / aligned.length,
  mostOptimistic: [...aligned].sort((a: any, b: any) => b.optimismBps - a.optimismBps).slice(0, 8),
  largestActionDifferences: [...changed].sort((a: any, b: any) => Math.abs(b.h2Exposure - b.h1Exposure) - Math.abs(a.h2Exposure - a.h1Exposure)).slice(0, 8) };
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value, null, 2));
save("config.json", { contract: "event-two-step-replay-audit-v1", source,
  method: "Recompute H1 on each H2-visited account under the same law. Score the H1 root action using the exact H1 continuation already recorded by H2. Compare predicted H2 value with realized TWO-event log change only where the next receding-H2 order equals the planned H1 continuation and both orders fill as planned. Intervals overlap and are selected by controller agreement; this is diagnostic, not an independent accuracy estimate. Next-open prices and minute borrowing still differ from the event law." });
save("sources.json", Object.fromEntries(["scripts/audit-event-two-step-replay.ts", "packages/bot-algo/src/event-one-step.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
save("rows.json", rows); save("summary.json", result); console.log(JSON.stringify(result));

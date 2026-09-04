/** Tighten global H3 bounds on saved feasible policies without changing trades. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { prepareEventMultiStepUpper } from "../packages/bot-algo/src/event-multi-step-upper.js";
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("source") || !arg("output")) throw new Error("Specify a completed H3 suite --source and new --output");
const root = path.resolve(__dirname, ".."), source = path.join(root, "data/benchmarks", arg("source"));
const output = path.join(root, "data/benchmarks", arg("output")), points = Number(arg("points") || 257);
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const read = (d: string, f: string) => JSON.parse(fs.readFileSync(path.join(d, f), "utf8"));
const config = read(source, "config.json"), previous = read(source, "summary.json");
assert.equal(config.contract, "event-fixed-law-suite-audit-v1"); assert.equal(config.depth, 3);
assert.equal(previous.results.length, config.windows.length, "Source suite must finish its declared windows");
assert.ok(Number.isInteger(points) && points >= 3 && points <= 513);
fs.mkdirSync(output, { recursive: true });
const save = (f: string, x: unknown) => fs.writeFileSync(path.join(output, f), JSON.stringify(x,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-three-step-bound-refinement-v1", source, points, tolerance: config.tolerance,
  method: "Recompute only global numerical upper bounds at previously unresolved H3 decision states. Verify the original forecast file hash, retain the saved feasible lower policy, root orders and simulation metrics, and intersect old/new upper bounds. No retraining, return-based selection, root reevaluation or new backtest." });
save("sources.json", Object.fromEntries(["scripts/refine-event-three-step-bounds.ts", "packages/bot-algo/src/event-multi-step-upper.ts",
  "packages/bot-algo/src/event-log-policy.ts"].map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const results = [], started = performance.now();
for (const row of previous.results) {
  const file = `${row.window.id}-model.json`, bytes = fs.readFileSync(path.join(config.source, file));
  assert.equal(createHash("sha256").update(bytes).digest("hex"), row.modelHash);
  const policy = restoreEventPolicy(JSON.parse(bytes.toString("utf8")).policy);
  const trace = read(source, `${row.window.id}-trades.json`);
  assert.ok(trace.length === row.bounds.count && trace.every((r: any) => r.optimizer === "three-event-bounds"));
  const unresolved = trace.filter((r: any) => !r.order.converged && r.order.feasible && Number.isFinite(Number(r.order.lowerValue)));
  const begin = performance.now();
  const upper = unresolved.length ? prepareEventMultiStepUpper(policy.model, policy.costs, "marked",
    { depth: 3, shadowPoints: points, method: "marginal" }) : undefined;
  const preparationSec = (performance.now() - begin) / 1000;
  const bounds = trace.map((r: any) => {
    const lowerValue = Number(r.order.lowerValue), oldUpper = Number(r.order.upperValue);
    const account = { equity: r.equityBefore, price: r.order.price, exposure: r.exposureBefore };
    let newUpper = oldUpper;
    if (!r.order.converged && r.order.feasible && Number.isFinite(lowerValue)) {
      const sell = upper!.query(r.leaf, account, 3, -1, "exchange-relaxation"),
        buy = upper!.query(r.leaf, account, 3, 1, "exchange-relaxation");
      newUpper = Math.min(upper!.query(r.leaf, account).upperValue, Math.max(sell.upperValue, buy.upperValue));
      const holdingUpper = Math.min(...r.order.evaluated.filter((v: any) => v.quantity === 0 && v.complete)
        .map((v: any) => Number(v.upperValue)));
      newUpper = Math.min(newUpper, Math.max(holdingUpper, sell.nonzeroUpperValue!, buy.nonzeroUpperValue!));
    }
    assert.ok(newUpper >= lowerValue - 2e-10 || newUpper === lowerValue, "Refined bound below the saved feasible policy");
    const upperValue = Math.max(lowerValue, Math.min(oldUpper, newUpper)), gap = upperValue === lowerValue ? 0 : upperValue - lowerValue;
    return { time: r.time, leaf: r.leaf, quantity: r.order.quantity, lowerValue, oldUpper, newUpper, upperValue, gap,
      previouslyCertified: r.order.converged, certified: r.order.feasible && gap <= config.tolerance };
  });
  save(`${row.window.id}-bounds.json`, bounds);
  const result = { window: row.window, fullWindow: row.fullWindow, modelHash: row.modelHash, metrics: row.metrics,
    count: bounds.length, previouslyCertified: bounds.filter((b: any) => b.previouslyCertified).length,
    certified: bounds.filter((b: any) => b.certified).length, maximumGapBps: Math.max(...bounds.map((b: any) => b.gap * 10000)),
    preparationSec, elapsedSec: (performance.now() - begin) / 1000 };
  results.push(result); save("summary.json", { results, elapsedSec: (performance.now() - started) / 1000 });
  console.log(JSON.stringify({ ...result, metrics: undefined }));
}

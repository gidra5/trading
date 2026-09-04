/** Certify unresolved decisions of an existing replay without changing its path. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventHolding, restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { prepareEventTwoStep } from "../packages/bot-algo/src/event-two-step.js";
import { prepareEventOneStep } from "../packages/bot-algo/src/event-one-step-prepared.js";

const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const directory = (name: string) => path.resolve(__dirname, "../data/benchmarks", name), output = directory(arg("output"));
assert.ok(arg("replays") && arg("output") && !fs.existsSync(output));
const hash = (value: Buffer | string) => createHash("sha256").update(value).digest("hex");
const budget = Number(arg("budget", "8192")), offset = Number(arg("offset", "0"));
assert.ok(Number.isInteger(budget) && budget >= 2 && Number.isInteger(offset) && offset >= 0);
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
const replays = arg("replays").split(",").map(name => {
  const source = directory(name), config = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
  assert.equal(config.contract, "native-event-planning-screen-v1"); assert.equal(config.mode, "replay");
  const modelBytes = fs.readFileSync(path.join(config.source, "model.json")); assert.equal(hash(modelBytes), config.modelHash);
  const traceBytes = fs.readFileSync(path.join(source, "h2-trades.json"));
  return { name, config, policy: restoreEventPolicy(JSON.parse(modelBytes.toString())), traceHash: hash(traceBytes),
    trace: JSON.parse(traceBytes.toString()) as Array<any> };
});
assert.ok(replays.every(r => r.config.modelHash === replays[0].config.modelHash));
const problems = replays.flatMap(r => r.trace.filter(row => !row.order.converged).map(row => ({ replay: r, row })));
const count = Number(arg("count", String(problems.length - offset)));
assert.ok(Number.isInteger(count) && count > 0 && offset + count <= problems.length);
save("config.json", { contract: "native-event-planning-refinement-v1", budget, offset, count,
  replays: replays.map(r => ({ name: r.name, modelHash: r.config.modelHash, traceHash: r.traceHash })),
  method: "Recompute each saved action's complete H2 value with exact H1 continuations, then refine its global upper bound at the identical account and leaf. An original action is certified only when its own verified value lies within the saved tolerance of the new upper bound, even if the refined optimizer proposes another action. No replay paths, forecasts, costs or executed actions are replaced." });
save("sources.json", Object.fromEntries(["scripts/refine-native-event-planning.ts", "packages/bot-algo/src/event-two-step.ts",
  "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-holding-law.ts",
  "packages/bot-algo/src/event-one-step-prepared.ts", "packages/bot-algo/src/event-multi-step-upper.ts",
  "packages/bot-algo/src/event-log-policy.ts"].map(file => [file, fs.readFileSync(path.resolve(__dirname, "..", file), "utf8")])));
const { model, costs } = replays[0].policy, solve = prepareEventTwoStep(model, costs, "marked", { globalUpper: true });
const continuation = model.kernels.map(k => prepareEventOneStep(k, costs, "marked")), results = [];
for (const { replay, row } of problems.slice(offset, offset + count)) {
  const account = { equity: row.equityBefore, price: row.order.price, exposure: row.exposureBefore };
  const started = performance.now(), quantity = row.order.quantity;
  const postEquity = account.equity - Math.abs(quantity) * account.price * (costs.feeBps + costs.slippageBps) / 10000;
  const postExposure = (account.exposure * account.equity + quantity * account.price) / postEquity;
  let originalValue = Math.log(postEquity / account.equity);
  for (const atom of model.kernels[row.leaf]) {
    const h = eventHolding(postExposure, atom, costs); assert.ok(!h.liquidated);
    originalValue += atom.probability * (Math.log(h.factor) + continuation[atom.next].value({
      equity: postEquity * h.factor, price: account.price * (1 + atom.return), exposure: h.exposure }));
  }
  assert.ok(Math.abs(originalValue - row.order.value) <= 1e-10, "Saved H2 action value changed");
  const refined = solve(row.leaf, account, { maxEvaluations: budget, tolerance: replay.config.tolerance, seedQuantities: [quantity] });
  assert.ok(refined.upperValue >= originalValue - 2e-10);
  const originalGap = Math.max(0, refined.upperValue - originalValue);
  const result = { replay: replay.name, time: row.time, leaf: row.leaf, account, quantity, originalValue,
    originalGap, originalCertified: originalGap <= replay.config.tolerance, refined,
    elapsedSeconds: (performance.now() - started) / 1000 };
  results.push(result); save("summary.json", { results });
  console.log(JSON.stringify({ replay: result.replay, time: row.time, originalGapBps: originalGap * 10000,
    originalCertified: result.originalCertified, quantity, refinedQuantity: refined.quantity,
    evaluations: refined.search.evaluations, elapsedSeconds: result.elapsedSeconds }));
}

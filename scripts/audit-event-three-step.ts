/** Validate H3 candidate values against independent exhaustive Bellman paths. */
import fs from "node:fs";
import path from "node:path";
import assert from "node:assert/strict";
import { prepareEventThreeStepActions } from "../packages/bot-algo/src/event-three-step.js";
import { eventBellmanReference } from "../packages/bot-algo/test/event-bellman-reference.js";
const arg = (k: string) => { const i = process.argv.indexOf(`--${k}`); return i < 0 ? "" : process.argv[i + 1]; };
if (!arg("reference") || !arg("output")) throw new Error("Specify existing fixed-law --reference and new --output");
const root = path.resolve(__dirname, ".."), source = path.join(root, "data/benchmarks", arg("reference"));
const output = path.join(root, "data/benchmarks", arg("output"));
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const config = JSON.parse(fs.readFileSync(path.join(source, "config.json"), "utf8"));
assert.equal(config.contract, "event-two-step-bound-audit-v1");
const cases = config.cases.slice(0, Number(arg("cases") || 40));
fs.mkdirSync(output, { recursive: true });
const save = (f: string, x: unknown) => fs.writeFileSync(path.join(output, f), JSON.stringify(x,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "event-three-step-action-audit-v1", source, cases,
  method: "For each fixed small problem and terminal convention, independently enumerate all future feasible lots after a specified root order. Compare H3 candidate upper/lower values for both cold and neighboring-target-seeded H2 searches. This validates action expectations, not global H3 root selection." });
save("sources.json", Object.fromEntries(["scripts/audit-event-three-step.ts", "packages/bot-algo/src/event-three-step.ts",
  "packages/bot-algo/src/event-two-step.ts", "packages/bot-algo/src/event-one-step-prepared.ts", "packages/bot-algo/src/event-holding-law.ts",
  "packages/bot-algo/src/event-one-step-upper.ts", "packages/bot-algo/src/event-multi-step-upper.ts", "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/test/event-bellman-reference.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const rows = [], started = performance.now();
for (const c of cases) for (const terminal of ["marked", "friction"] as const) {
  const reference = eventBellmanReference(c.model, c.costs, { terminal, maxNodes: 1000000 });
  const solve = prepareEventThreeStepActions(c.model, c.costs, terminal);
  for (const q of [0, -2 * c.costs.quantityStep, 2 * c.costs.quantityStep]) {
    const expected = reference.actionValue(c.leaf, c.account, 3, q);
    for (const warmStart of [false, true]) {
      const actual = solve(c.leaf, c.account, q, { maxEvaluations: 512, warmStart });
      const equal = (a: number, b: number) => a === b || Math.abs(a - b) < 1e-10;
      const valid = actual.complete && equal(expected, actual.lowerValue)
        && (actual.upperValue >= expected - 1e-10 || equal(actual.upperValue, expected)) && actual.gap <= 1e-7;
      const row = { name: c.name, terminal, quantity: q, exposure: c.account.exposure, warmStart, expected, actual, valid };
      rows.push(row);
      if (!valid) { save("failure.json", { case: c, row }); throw new Error(`H3 action audit failed: ${c.name} ${terminal}`); }
    }
  }
}
const summary = { cases: cases.length, queries: rows.length, maximumGapBps: Math.max(...rows.map(r => r.actual.gap * 10000)),
  elapsedSec: (performance.now() - started) / 1000, rows };
save("summary.json", summary); console.log(JSON.stringify({ ...summary, rows: undefined }));

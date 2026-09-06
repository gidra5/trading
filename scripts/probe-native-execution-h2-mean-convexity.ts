/** Falsify or support discrete H2 mean-equity convexity inside H1 regions. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { evaluateEventExecutionPath, summarizeEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
import type { EventCandle } from "../packages/bot-algo/src/event-distribution.js";
import { DEFAULT_EVENT_COSTS } from "../packages/bot-algo/src/event-log-policy.js";

const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), output = path.resolve(root, "data/benchmarks", arg("output"));
const samples = Number(arg("samples") || "60");
assert.ok(arg("output") && !fs.existsSync(output)); assert.ok(Number.isInteger(samples) && samples > 0 && samples <= 1000);
fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, value) => typeof value === "number" && !Number.isFinite(value) ? String(value) : value, 2));
save("sources.json", Object.fromEntries([
  "scripts/probe-native-execution-h2-mean-convexity.ts", "packages/bot-algo/src/event-execution-one-step.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-log-policy.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
let seed = 612947;
const random = () => { seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0; return seed / 2 ** 32; };
const started = performance.now(), violations: any[] = [];
let cases = 0, regions = 0, triples = 0, finiteTriples = 0, violatedTriples = 0, minimumSecondDifference = Infinity;
for (let sample = 0; sample < samples; sample++) {
  const costs = { ...DEFAULT_EVENT_COSTS, feeBps: sample % 3 === 0 ? 0 : 11, slippageBps: 0,
    maxLeverage: sample % 2 ? 1 : 3, maintenanceMargin: .04, minQuantity: .25, quantityStep: .25,
    minNotional: sample % 2 ? 5 : 0, maxNotional: [8, 25, 70][sample % 3],
    longBorrowBpsPerDay: 14400, shortBorrowBpsPerDay: 86400 };
  const kernel = [.15, .35, .5].map((probability, j) => {
    const bars: EventCandle[] = [{ openTime: 0, open: 10, high: 10, low: 10, close: 10, volume: 1 }];
    for (let i = 1; i <= 5; i++) {
      const open = bars[i - 1].close * (.8 + .4 * random()), close = open * (.8 + .4 * random());
      bars.push({ openTime: i * 1000, open, close, low: Math.min(open, close) * .97,
        high: Math.max(open, close) * 1.03, volume: 1 });
    }
    if (sample % 4 === 0 && j === 1) bars[1].carriedMark = true;
    return { probability, path: summarizeEventExecutionPath(bars, 0, 5, costs, sample % 5 !== 0) };
  });
  const child = prepareEventExecutionOneStep(kernel, "marked", { objective: "mean" });
  for (const quantity of [-10, -4, -.125, 0, .125, 2, 6, 12]) {
    const account = { equity: 40, price: 10, exposure: quantity / 4 };
    const cover = prepareEventExecutionOneStep(kernel, "marked", { captureRegions: true })(account);
    assert.ok("requestRegions" in cover); cases++;
    const value = (lot: number) => {
      let total = 0;
      for (const atom of kernel) {
        const next = evaluateEventExecutionPath(atom.path, account, lot * costs.quantityStep);
        if (!Number.isFinite(next.logGrowth)) return -Infinity;
        const continuation = child(next);
        if (!Number.isFinite(continuation.value)) return -Infinity;
        total += atom.probability * next.equity * continuation.value / account.equity;
      }
      return total;
    };
    for (const [lo, hi] of cover.requestRegions) {
      if (hi - lo < 2) continue; regions++;
      let previous = value(lo), current = value(lo + 1);
      for (let lot = lo + 2; lot <= hi; lot++) {
        const next = value(lot); triples++;
        if ([previous, current, next].every(Number.isFinite)) {
          finiteTriples++; const second = next - 2 * current + previous;
          minimumSecondDifference = Math.min(minimumSecondDifference, second);
          if (second < -1e-10) {
            violatedTriples++;
            if (violations.length < 100) violations.push({ sample, quantity, region: [lo, hi], lot: lot - 1,
              values: [previous, current, next], secondDifference: second });
          }
        }
        previous = current; current = next;
      }
    }
  }
}
const summary = { contract: "native-execution-h2-mean-convexity-probe-v1", seed: 612947, samples, cases, regions,
  triples, finiteTriples, violatedTriples, minimumSecondDifference, violations,
  conclusion: violatedTriples ? "Discrete risk-neutral H2 convexity is false inside at least one exact H1 guarded root region."
    : "No counterexample was found in this finite probe; this is not a proof of convexity.",
  scope: "Synthetic exhaustive small-lattice diagnostic with the exact marked mean-equity child optimizer, restricted to root actions with finite log wealth. It tests a search theorem, not market performance.",
  elapsedSeconds: (performance.now() - started) / 1000 };
save("summary.json", summary); console.log(JSON.stringify({ ...summary, violations: undefined }));

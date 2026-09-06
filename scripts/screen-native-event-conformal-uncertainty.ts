/** Fit a pre-test split-conformal uncertainty radius to event-mean errors. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { eventConformalResidualRadius } from "../packages/bot-algo/src/event-uncertainty.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";

const arg = (key: string, fallback = "") => {
  const index = process.argv.indexOf(`--${key}`);
  return index < 0 ? fallback : process.argv[index + 1];
};
const root = path.resolve(__dirname, "..");
const directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source"));
const output = directory(arg("output"));
const coverage = Number(arg("coverage", "0.75"));
assert.ok(arg("source") && arg("output") && fs.existsSync(source) && !fs.existsSync(output));
assert.ok(coverage > 0 && coverage < 1);
const read = (name: string) => JSON.parse(fs.readFileSync(path.join(source, name), "utf8"));
const modelBytes = fs.readFileSync(path.join(source, "model.json"));
const policy = restoreEventPolicy(JSON.parse(modelBytes.toString()));
const rows = read("calibration-trades.json") as Array<{
  time: number; endTime: number; expectedReturnBps: number; realizedReturnBps: number;
}>;
assert.ok(rows.length >= 2 && rows.every((row, index) => Number.isFinite(row.expectedReturnBps)
  && Number.isFinite(row.realizedReturnBps) && row.time < row.endTime
  && (!index || row.time >= rows[index - 1]!.endTime)),
"Calibration replay must be an ordered, non-overlapping event chain");
const radiusBps = eventConformalResidualRadius(rows.map(row => ({
  predicted: row.expectedReturnBps,
  realized: row.realizedReturnBps,
})), coverage);
assert.ok(Number.isFinite(radiusBps), "Requested coverage is unsupported by the calibration sample size");
const leaves = policy.model.kernels.map((kernel, leaf) => {
  const meanBps = kernel.reduce((sum, atom) => sum + atom.probability * atom.return, 0) * 10_000;
  const lowBps = meanBps - radiusBps, highBps = meanBps + radiusBps;
  return { leaf, meanBps, lowBps, highBps, ambiguousDirection: lowBps <= 0 && highBps >= 0 };
});
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest("hex");
const result = {
  contract: "native-event-mean-conformal-uncertainty-v1",
  source,
  modelHash: hash(modelBytes),
  coverage,
  calibrationEvents: rows.length,
  radiusBps,
  leaves,
  method: "Use the ordered non-overlapping pre-test calibration event chain. Compute absolute errors of each frozen leaf mean, then use the finite-sample split-conformal ceil((n+1)*coverage) residual order statistic as one shared, stable radius. Downstream maximin H2 evaluates both mean endpoints and forbids opening, enlargement or reversal when their signs disagree. This is marginal uncertainty under an exchangeability approximation, not a regime-conditional guarantee.",
};
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(result, null, 2));
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(new URL(import.meta.url), "utf8"));
console.log(JSON.stringify(result));


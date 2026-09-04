/** Measure account-box continuation bounds on saved, predeclared calibration controls. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBoxUpper } from "../packages/bot-algo/src/event-execution-box.js";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const output = directory(arg("output")), sources = arg("sources").split(",").map(directory);
const linked = process.argv.includes("--linked");
const ordered = process.argv.includes("--ordered"), toleranceBps = Number(arg("tolerance-bps") || ".001");
const coupledWealth = process.argv.includes("--coupled-wealth");
const maxRefinements = Number(arg("refinements") || "10000");
assert.ok(arg("sources") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
fs.mkdirSync(output, { recursive: true });
const save = (name: string, data: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(data,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("sources.json", Object.fromEntries(["scripts/probe-native-execution-box.ts", "packages/bot-algo/src/event-execution-box.ts",
  "packages/bot-algo/src/event-execution-acceptance.ts",
  "packages/bot-algo/src/event-affine-segment.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts"].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), controls = [], results = [];
let complete = true;
for (const source of sources) {
  const config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
  assert.equal(config.contract, "native-execution-bellman-candidates-v1");
  assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
  const law = read(path.join(config.source, "law.json")), kernels = law.kernels.map((k: any[]) => k.map(a => ({
    probability: a.probability, path: law.paths[a.path] })));
  const cases = new Map<string, any>();
  const add = (leaf: number, account: any, value: number) => {
    const bucket = !account.exposure ? "cash" : Math.abs(account.exposure) > law.costs.maxLeverage ? "above-cap" : "invested";
    const key = `${leaf}:${bucket}`;
    if (!cases.has(key)) cases.set(key, { key, leaf, account, exactValue: value });
  };
  add(config.probe.leaf, config.probe.account, config.oneStep.value);
  for (const r of saved.results) for (const b of r.branches) add(b.next, b.account, b.continuation);
  const begin = performance.now(), bounds = kernels.map((k: any[]) => prepareEventExecutionBoxUpper(k, {
    ordered, coupledWealth, valueTolerance: toleranceBps / 10000, maxRefinements }));
  const exact = kernels.map((k: any[]) => prepareEventExecutionOneStep(k));
  controls.push({ source, sourceSummaryHash: hash(path.join(source, "summary.json")), lawHash: config.lawHash,
    modelHash: config.modelHash, cases: cases.size, preparationSeconds: (performance.now() - begin) / 1000 });
  for (const row of cases.values()) {
    const { equity: E, price: P, exposure: x } = row.account, step = law.costs.quantityStep;
    const Q = x * E / P, C = E - Q * P, referenceLogEquity = Math.log(E) + row.exactValue;
    const widths = [];
    for (const lots of [0, 1, 10, 100]) {
      const deltaQ = lots * step, deltaC = deltaQ * P * (1 + (law.costs.feeBps + law.costs.slippageBps) / 10000);
      const box = { price: P, cash: [C - deltaC, C + deltaC] as const,
        quantity: [Q - deltaQ, Q + deltaQ] as const, quantityLattice: true,
        balanceVertices: linked ? [[C - deltaC, Q + deltaQ], [C + deltaC, Q - deltaQ]] as const : undefined };
      const start = performance.now(), result = bounds[row.leaf](box), ms = performance.now() - start;
      if (!ordered) assert.equal(result.complete, true);
      assert.ok(Number.isFinite(result.upperLogEquity));
      assert.ok(result.upperLogEquity >= referenceLogEquity - 1e-10, `${source} ${row.key}, lots=${lots}`);
      // Check the original solver at each box corner in addition to the saved
      // centre optimum. These are bound audits, not performance-selected cases.
      let largestCorner = -Infinity;
      const points = box.balanceVertices ?? box.cash.flatMap(cash => box.quantity.map(quantity => [cash, quantity] as const));
      for (const [cash, quantity] of points) {
        const equity = cash + quantity * P; if (!(equity > 0)) continue;
        const value = exact[row.leaf]({ equity, price: P, exposure: quantity * P / equity }).value + Math.log(equity);
        largestCorner = Math.max(largestCorner, value);
        assert.ok(result.upperLogEquity >= value - 1e-10, `corner ${source} ${row.key}, lots=${lots}`);
      }
      widths.push({ lots, box, ...result, ms, slackBps: (result.upperLogEquity - referenceLogEquity) * 10000,
        largestCornerSlackBps: (result.upperLogEquity - largestCorner) * 10000 });
    }
    results.push({ source, ...row, atoms: kernels[row.leaf].length, widths });
    console.log(JSON.stringify({ source: path.basename(source), key: row.key, widths: widths.map(r => ({
      lots: r.lots, slackBps: r.slackBps, ms: r.ms, evaluatedOrders: r.search.evaluatedOrders,
      refinements: r.search.refinements, complete: r.complete, certified: r.certified, relaxedGap: r.relaxedGap })) }));
    save("summary.json", { contract: "native-execution-box-probe-v1", controls, linked, ordered, coupledWealth, toleranceBps, maxRefinements, complete: false,
      selection: "First saved calibration occurrence of each leaf and cash/invested/above-cap bucket, including the root; widths fixed at 0, 1, 10, 100 lots. No return-based selection.",
      results, elapsedSeconds: (performance.now() - started) / 1000 });
    if ((performance.now() - started) / 1000 > 60) { complete = false; break; }
  }
  if (!complete) break;
}
save("summary.json", { ...read(path.join(output, "summary.json")), complete, elapsedSeconds: (performance.now() - started) / 1000 });
if (!complete) process.exitCode = 1;

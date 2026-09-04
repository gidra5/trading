/** Diagnose acceptance-information slack on every three-account linked-lot probe. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), output = directory(arg("output"));
assert.ok(arg("source") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const saved = read(path.join(source, "summary.json"));
assert.ok(saved.complete && saved.linked); assert.equal(saved.contract, "native-execution-box-probe-v1");
const started = performance.now(), laws = new Map<string, any>(), results = [];
for (const row of saved.results) {
  if (!laws.has(row.source)) {
    const config = read(path.join(row.source, "config.json"));
    assert.equal(config.lawHash, hash(path.join(config.source, "law.json")));
    laws.set(row.source, read(path.join(config.source, "law.json")));
  }
  const law = laws.get(row.source), width = row.widths.find((w: any) => w.lots === 1), b = width.box;
  assert.equal(b.balanceVertices.length, 2);
  const kernel = law.kernels[row.leaf].map((a: any) => ({ probability: a.probability, path: law.paths[a.path] }));
  const solve = prepareEventExecutionOneStep(kernel), step = law.costs.quantityStep, P = b.price;
  const vertices = [...b.balanceVertices].sort((a, z) => a[1] - z[1]);
  const loLot = Math.round(vertices[0][1] / step), hiLot = Math.round(vertices[1][1] / step);
  assert.equal(hiLot - loLot, 2);
  const states = [];
  for (let lot = loLot; lot <= hiLot; lot++) {
    const fraction = (lot - loLot) / (hiLot - loLot), quantity = lot * step;
    const cash = vertices[0][0] + fraction * (vertices[1][0] - vertices[0][0]), equity = cash + quantity * P;
    const account = { equity, price: P, exposure: quantity * P / equity }, optimum = solve(account);
    const transitions = kernel.map((a: any) => evaluateEventExecutionPath(a.path, account, width.optimisticRequest));
    const value = kernel.reduce((s: number, a: any, i: number) => s + a.probability * Math.log(transitions[i].equity / row.account.equity), 0);
    const exact = kernel.reduce((s: number, a: any) => s + a.probability
      * evaluateEventExecutionPath(a.path, account, optimum.quantity).logGrowth, 0);
    assert.equal(optimum.value, exact);
    states.push({ lot, cash, quantity, account, optimum,
      optimalValue: exact + Math.log(account.equity / row.account.equity), relaxedRequestValue: value,
      fillProbability: kernel.reduce((s: number, a: any, i: number) => s + a.probability * Number(transitions[i].filledQuantity !== 0), 0),
      transitions });
  }
  const bestCommonAccountValue = Math.max(...states.map(s => s.optimalValue));
  const bestCommonAccountForRelaxedRequest = Math.max(...states.map(s => s.relaxedRequestValue));
  let revealedAccountValue = 0, uncertainAcceptanceMass = 0;
  for (let i = 0; i < kernel.length; i++) {
    revealedAccountValue += kernel[i].probability * Math.max(...states.map(s => Math.log(s.transitions[i].equity / row.account.equity)));
    if (new Set(states.map(s => s.transitions[i].filledQuantity !== 0)).size > 1) uncertainAcceptanceMass += kernel[i].probability;
  }
  const upperValue = width.upperLogEquity - Math.log(row.account.equity);
  assert.ok(upperValue >= revealedAccountValue - 1e-10 && upperValue >= bestCommonAccountValue - 1e-10);
  results.push({ source: row.source, key: row.key, atoms: kernel.length, relaxedRequest: width.optimisticRequest,
    states: states.map(({ transitions, ...s }) => s), bestCommonAccountValue, bestCommonAccountForRelaxedRequest,
    revealedAccountValue, upperValue, uncertainAcceptanceMass,
    openRatioRangeBps: [Math.min(...kernel.map((a: any) => a.path.openRatio - 1)) * 10000,
      Math.max(...kernel.map((a: any) => a.path.openRatio - 1)) * 10000],
    upperSlackBps: (upperValue - bestCommonAccountValue) * 10000,
    revealedAccountAdvantageBps: (revealedAccountValue - bestCommonAccountValue) * 10000,
    financialRemainderBps: (upperValue - revealedAccountValue) * 10000 });
  console.log(JSON.stringify({ key: row.key, source: path.basename(row.source), ...Object.fromEntries(Object.entries(results.at(-1)!).filter(([k]) =>
    ["upperSlackBps", "revealedAccountAdvantageBps", "financialRemainderBps", "uncertainAcceptanceMass"].includes(k))) }));
}
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify({ contract: "execution-box-acceptance-audit-v1", source,
  sourceSummaryHash: hash(path.join(source, "summary.json")), results, elapsedSeconds: (performance.now() - started) / 1000,
  scope: "For each saved one-lot linked segment, enumerate all three admissible inventories and globally optimize the original H1 policy at each account. Compare to choosing the incoming account separately after observing each future outcome, holding the relaxed request fixed. The latter is an information relaxation, never a real strategy." }, null, 2));
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries([
  "scripts/audit-native-execution-box.ts", "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])), null, 2));

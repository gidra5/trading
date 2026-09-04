/** Audit the complete feasible first-request cover, then bound one predefined H2 region. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { prepareEventExecutionBoxUpper } from "../packages/bot-algo/src/event-execution-box.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const sources = arg("sources").split(",").map(directory), output = directory(arg("output"));
assert.ok(arg("sources") && arg("output") && !fs.existsSync(output));
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
fs.mkdirSync(output, { recursive: true });
save("sources.json", Object.fromEntries(["scripts/probe-native-execution-regions.ts", "packages/bot-algo/src/event-execution-one-step.ts",
  "packages/bot-algo/src/event-execution-box.ts", "packages/bot-algo/src/event-execution-acceptance.ts", "packages/bot-algo/src/event-affine-segment.ts",
  "packages/bot-algo/src/event-execution-path.ts", "packages/bot-algo/src/event-log-policy.ts"]
  .map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
const started = performance.now(), results = [];
for (const source of sources) {
  const config = read(path.join(source, "config.json")), saved = read(path.join(source, "summary.json"));
  assert.equal(config.contract, "native-execution-bellman-candidates-v1");
  assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
  const law = read(path.join(config.source, "law.json")), kernels = law.kernels.map((k: any[]) => k.map(a => ({
    probability: a.probability, next: a.next, path: law.paths[a.path] })));
  const account = config.probe.account, leaf = config.probe.leaf, kernel = kernels[leaf], step = law.costs.quantityStep;
  const begin = performance.now(), cover = prepareEventExecutionOneStep(kernel, "marked", { captureRegions: true })(account);
  assert.ok("requestRegions" in cover && cover.feasible);
  assert.equal(cover.quantity, config.oneStep.quantity); assert.equal(cover.value, config.oneStep.value);
  const regions = cover.requestRegions, audits = { transitions: 0, maxEquityInterpolationError: 0, maxQuantityInterpolationError: 0 };
  for (const [lo, hi] of regions) {
    const middle = Math.floor((lo + hi) / 2), fraction = hi > lo ? (middle - lo) / (hi - lo) : 0;
    for (const atom of kernel) {
      const points = [lo, middle, hi].map(k => evaluateEventExecutionPath(atom.path, account, k * step));
      assert.ok(points.every(p => !p.liquidated && p.canceled === points[0].canceled
        && (p.filledQuantity !== 0) === (points[0].filledQuantity !== 0)));
      const equityError = Math.abs(points[1].equity - (points[0].equity + fraction * (points[2].equity - points[0].equity)));
      const quantityError = Math.abs(points[1].quantity - (points[0].quantity + fraction * (points[2].quantity - points[0].quantity)));
      audits.maxEquityInterpolationError = Math.max(audits.maxEquityInterpolationError, equityError);
      audits.maxQuantityInterpolationError = Math.max(audits.maxQuantityInterpolationError, quantityError);
      assert.ok(equityError < 1e-7 && quantityError < 1e-10); audits.transitions += 3;
    }
  }
  const winningLot = Math.round(cover.quantity / step), distance = ([lo, hi]: readonly number[]) => Math.max(lo - winningLot, winningLot - hi, 0);
  const chosen = [...regions].filter(([lo, hi]) => hi > lo).sort((a, b) => distance(a) - distance(b) || a[0] - b[0])[0];
  assert.ok(chosen);
  const lo = Math.max(chosen[0], Math.min(chosen[1] - 200, winningLot - 100)), hi = Math.min(chosen[1], lo + 200);
  const middle = Math.floor((lo + hi) / 2), incumbent = Math.max(...saved.results.map((r: any) => r.value));
  const exact = kernels.map((k: any[]) => prepareEventExecutionOneStep(k));
  const bounds = kernels.map((k: any[]) => prepareEventExecutionBoxUpper(k, { ordered: true, coupledWealth: true, valueTolerance: 1e-7 }));
  const key = (next: number, a: any) => JSON.stringify([next, a.equity, a.price, a.exposure]);
  const cache = new Map<string, { value: number; quantity: number }>();
  for (const row of saved.results) for (const branch of row.branches) {
    assert.ok(branch.complete); cache.set(key(branch.next, branch.account), { value: branch.continuation, quantity: branch.continuationRequest });
  }
  let upper = 0, lower = 0, mass = 0, boxes = 0, exactSolves = 0, reused = 0, verifiedChildOutcomes = 0;
  const branches = [], regionStart = performance.now();
  for (const [index, atom] of [...kernel.entries()].sort((a: any, b: any) => b[1].probability - a[1].probability)) {
    const first = evaluateEventExecutionPath(atom.path, account, lo * step), last = evaluateEventExecutionPath(atom.path, account, hi * step);
    const mid = evaluateEventExecutionPath(atom.path, account, middle * step);
    const exactKey = key(atom.next, mid);
    let reference = cache.get(exactKey);
    if (!reference) {
      const solved = exact[atom.next](mid); exactSolves++;
      let checked = 0;
      for (const child of kernels[atom.next]) {
        checked += child.probability * evaluateEventExecutionPath(child.path, mid, solved.quantity).logGrowth;
        verifiedChildOutcomes++;
      }
      assert.equal(checked, solved.value);
      reference = { value: solved.value, quantity: solved.quantity }; cache.set(exactKey, reference);
    } else reused++;
    const cashFirst = first.equity - first.quantity * first.price, cashLast = last.equity - last.quantity * last.price;
    let boundValue: number, boundMs = 0;
    if (key(atom.next, first) === key(atom.next, last)) boundValue = Math.log(mid.equity) + reference.value;
    else {
      const before = performance.now();
      const result = bounds[atom.next]({ price: first.price, cash: [Math.min(cashFirst, cashLast), Math.max(cashFirst, cashLast)],
        quantity: [Math.min(first.quantity, last.quantity), Math.max(first.quantity, last.quantity)],
        balanceVertices: [[cashFirst, first.quantity], [cashLast, last.quantity]], quantityLattice: true });
      boundValue = result.upperLogEquity; boundMs = performance.now() - before; boxes++;
    }
    const exactValue = Math.log(mid.equity) + reference.value;
    assert.ok(boundValue >= exactValue - 1e-10);
    upper += atom.probability * boundValue; lower += atom.probability * exactValue; mass += atom.probability;
    branches.push({ index, probability: atom.probability, next: atom.next, boundValue, exactValue, boundMs,
      midpointAccount: { equity: mid.equity, price: mid.price, exposure: mid.exposure }, continuation: reference });
    if (branches.length % 50 === 0) console.log(JSON.stringify({ source: path.basename(source), branches: branches.length, mass, boxes,
      seconds: (performance.now() - regionStart) / 1000 }));
    if ((performance.now() - regionStart) / 1000 > 30) break;
  }
  const complete = branches.length === kernel.length;
  const row = { source, sourceSummaryHash: hash(path.join(source, "summary.json")), lawHash: config.lawHash, modelHash: config.modelHash,
    cover: { ...cover, audits, coveredLots: regions.reduce((s, [lo, hi]) => s + hi - lo + 1, 0),
      singletons: regions.filter(([lo, hi]) => lo === hi).length, nonSingletons: regions.filter(([lo, hi]) => hi > lo).length },
    selectedOriginalRegion: chosen, selectedRegion: [lo, hi], midpointRequest: middle * step, incumbent,
    intervalComplete: complete, boundValue: complete ? upper - Math.log(account.equity) + 1e-9 : null,
    exactMidpointValue: complete ? lower - Math.log(account.equity) : null, completedMass: mass, boxes, exactSolves, reused,
    verifiedChildOutcomes, branches, regionSeconds: (performance.now() - regionStart) / 1000,
    seconds: (performance.now() - begin) / 1000 };
  results.push(row);
  save("summary.json", { contract: "native-execution-root-region-probe-v1", results, elapsedSeconds: (performance.now() - started) / 1000,
    scope: "Complete feasible H1 request cover with constant execution/funding branches. H2 upper for one predeclared nearest non-singleton region, clipped to 201 lots. A 30-second soft budget leaves the full interval value null if any first outcome remains. The midpoint is independently exactly optimized; no global H2 or stationary optimality claim." });
  console.log(JSON.stringify({ ...row, cover: { regions: regions.length, ...audits }, branches: undefined }));
}

/** Join independently audited, frozen-law execution-H1 windows without compounding reset accounts. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
for (const key of ["suite", "additional-audit", "baseline", "screen", "output"]) assert.ok(arg(key), `Missing ${key}`);
const output = directory(arg("output")); assert.ok(!fs.existsSync(output));
const inputs = Object.fromEntries(["suite", "additional-audit", "baseline", "screen"].map(key => {
  const file = path.join(directory(arg(key)), "summary.json"); return [key, { file, sha256: hash(file) }];
}));
const suite = read(inputs.suite.file), additional = read(inputs["additional-audit"].file), baseline = read(inputs.baseline.file);
const screen = read(inputs.screen.file), catalog = read(path.join(directory(arg("screen")), "config.json")).catalog;
assert.equal(suite.processed, suite.requested); assert.ok(suite.results.every((r: any) => r.status === "complete-h1"));
assert.ok(catalog.length && catalog.every((w: any) => !w.id.startsWith("fit-") && w.id !== "latest"));
const entries = [...suite.results.map((r: any) => ({ replay: directory(r.replay), audit: path.join(directory(r.audit), "summary.json") })),
  ...additional.results.filter((r: any) => r.phase === "test").map((r: any) => ({ replay: directory(r.name), audit: inputs["additional-audit"].file }))];
const results: any[] = [], ids = new Set<string>();
for (const entry of entries) {
  const config = read(path.join(entry.replay, "config.json")), summary = read(path.join(entry.replay, "summary.json"));
  const id = config.window.id, window = catalog.find((w: any) => w.id === id);
  assert.ok(window && !ids.has(id)); ids.add(id);
  assert.equal(config.contract, "native-event-execution-one-step-replay-v1"); assert.equal(config.phase, "test");
  assert.equal(config.start, window.startTime); assert.equal(config.end, window.endTime);
  const audit = read(entry.audit).results.find((r: any) => directory(r.name) === entry.replay && r.phase === "test");
  assert.ok(audit); assert.equal(hash(path.join(entry.replay, "execution-trades.json")), audit.traceHash);
  assert.equal(audit.modelHash, config.modelHash); assert.equal(audit.lawHash, config.lawHash);
  assert.equal(hash(path.join(config.baseSource, "model.json")), config.modelHash);
  assert.equal(hash(path.join(config.source, "law.json")), config.lawHash);
  assert.equal(screen.results.find((r: any) => r.window.id === id).modelHash, config.modelHash);
  const prior = baseline.results.find((r: any) => r.id === id); assert.ok(prior);
  assert.equal(prior.modelHash, config.modelHash); assert.equal(prior.start, config.start); assert.equal(prior.end, config.end);
  assert.equal(prior.h1ReturnPct, summary.control.returnPct);
  assert.deepEqual(audit.control, summary.control); assert.deepEqual(audit.execution, summary.execution);
  assert.equal(audit.decisions, summary.completeSearches); assert.equal(audit.decisions, summary.finiteValueDecisions);
  assert.equal(audit.decisions, prior.optimizedDecisions);
  const law = read(path.join(config.source, "law.json")), model = read(path.join(config.baseSource, "model.json"));
  assert.deepEqual(law.costs, model.costs);
  assert.deepEqual(law.kernels.map((k: any[]) => k.map(a => { const p = law.paths[a.path]; return {
    probability: a.probability, return: p.closeRatio - 1, low: Math.min(1, p.lowRatio) - 1,
    high: Math.max(1, p.highRatio) - 1, duration: p.seconds / 60, next: a.next,
  }; })), model.model.kernels);
  const compiled = read(path.join(config.source, "summary.json"));
  assert.ok(compiled.exactSavedModel && compiled.exactWeights && compiled.exactOldJointProjection);
  const compiledConfig = read(path.join(config.source, "config.json"));
  for (const ref of [...compiledConfig.sourceReferences, ...config.replaySourceReferences]) assert.equal(hash(ref.file), ref.sha256);
  if (config.marketAvailability) assert.equal(hash(config.marketAvailability.file), config.marketAvailability.sha256);
  const trace = read(path.join(entry.replay, "execution-trades.json"));
  const waits = summary.execution.marketAvailability?.forcedWaits ?? [];
  assert.equal(waits.length, prior.forcedWaitSegments);
  assert.equal(audit.transitions.filter((r: any) => r.forcedWait).length, waits.length);
  assert.ok(trace.every((r: any) => r.order.complete && r.order.feasible && Number.isFinite(r.order.value)));
  assert.ok(audit.valueRows.every((r: any) => r.advantageOverOldBps >= -1e-6 && r.advantageOverHoldBps >= -1e-6));
  results.push({ id, start: config.start, end: config.end, replay: entry.replay, audit: entry.audit, auditHash: hash(entry.audit),
    modelHash: config.modelHash, lawHash: config.lawHash, traceHash: audit.traceHash,
    decisions: audit.decisions, forcedWaitSegments: waits.length, availability: prior.availability,
    controlReturnPct: summary.control.returnPct, returnPct: summary.execution.returnPct,
    drawdownPct: summary.execution.maxDrawdownPct, orders: summary.execution.trades, cancellations: summary.execution.canceledOrders,
    requests: audit.requests, mostlyRejectedRequests: audit.mostlyRejectedRequests, meanRejectionMass: audit.meanRejectionMass,
    meanGainOverOldBps: audit.meanGainOverOldBps, maxValueError: audit.maxValueError, maxEquityError: audit.maxEquityError,
    controlFees: summary.control.fees, fees: summary.execution.fees, borrow: summary.execution.borrow,
    longPnl: summary.execution.longPnl, shortPnl: summary.execution.shortPnl,
    longMinutes: summary.execution.longMinutes, shortMinutes: summary.execution.shortMinutes,
    evaluatedOrders: summary.evaluatedOrders, executionSeconds: summary.executionSeconds });
}
assert.equal(ids.size, catalog.length); results.sort((a, b) => catalog.findIndex((w: any) => w.id === a.id) - catalog.findIndex((w: any) => w.id === b.id));
const sum = (key: string) => results.reduce((s, r) => s + r[key], 0), count = (f: (r: any) => boolean) => results.filter(f).length;
const decisions = sum("decisions"), requests = sum("requests");
const summary = { contract: "native-execution-h1-coverage-v1", inputs,
  scope: "Complete one-event request searches under the fixed enriched empirical laws and research execution rules. 27 strict observed windows plus the declared March closure scenario. All windows were previously inspected and can overlap; returns of reset accounts are not compounded. This does not establish deeper/stationary Bellman optimality, a position-policy ablation, live-exchange validity, or reliable profitability.",
  windows: results.length, strictObservedWindows: count(r => r.availability === "strict observed seconds"),
  explicitAvailabilityScenarioWindows: count(r => r.availability !== "strict observed seconds"), decisions,
  forcedWaitSegments: sum("forcedWaitSegments"), positiveWindows: count(r => r.returnPct > 1e-10),
  negativeWindows: count(r => r.returnPct < -1e-10), cashWindows: count(r => r.returnPct === 0 && r.orders === 0),
  higherReturn: count(r => r.returnPct > r.controlReturnPct + 1e-10), lowerReturn: count(r => r.returnPct < r.controlReturnPct - 1e-10),
  sameReturn: count(r => Math.abs(r.returnPct - r.controlReturnPct) <= 1e-10),
  worstDrawdownPct: Math.max(...results.map(r => r.drawdownPct)), maxValueError: Math.max(...results.map(r => r.maxValueError)),
  maxEquityError: Math.max(...results.map(r => r.maxEquityError)), requests, orders: sum("orders"), cancellations: sum("cancellations"),
  mostlyRejectedRequests: sum("mostlyRejectedRequests"), meanRejectionMass: results.reduce((s, r) => s + r.meanRejectionMass * r.requests, 0) / requests,
  meanGainOverOldBps: results.reduce((s, r) => s + r.meanGainOverOldBps * r.decisions, 0) / decisions,
  evaluatedOrders: sum("evaluatedOrders"), executionSeconds: sum("executionSeconds"), results };
fs.mkdirSync(output, { recursive: true }); fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename, "utf8"));
fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(summary, null, 2));
fs.writeFileSync(path.join(output, "table.md"), ["| Window | Old H1 return | Execution H1 return | Drawdown | Decisions | Orders | Cancellations |",
  "|---|---:|---:|---:|---:|---:|---:|", ...results.map(r => `| ${r.id} | ${r.controlReturnPct.toFixed(2)}% | ${r.returnPct.toFixed(2)}% | ${r.drawdownPct.toFixed(2)}% | ${r.decisions} | ${r.orders} | ${r.cancellations} |`)].join("\n") + "\n");
console.log(JSON.stringify({ ...summary, results: undefined }));

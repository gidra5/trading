/** Compare calibration-selected family representatives on the already examined
 * window. Diagnostic only: these test returns never select a policy. */
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { buildEventKernelLookahead, buildEventOutcomeLookahead, restoreEventPolicy, type EventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventSizeSignGroup } from "../packages/bot-algo/src/event-size-sign.js";
import { loadEventCandles, replayEventPolicy } from "./research-event-policy.js";

const root = path.resolve(__dirname, ".."), DAY = 86_400_000;
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const sourceId = arg("source"), outputId = arg("output");
if (!sourceId || !outputId) throw new Error("Specify saved severity replay and new output");
const source = path.resolve(root, "data/benchmarks", sourceId), output = path.resolve(root, "data/benchmarks", outputId);
const read = (dir: string, name: string) => JSON.parse(fs.readFileSync(path.join(dir, name), "utf8"));
const config = read(source, "config.json"), summary = read(source, "summary.json");
if (config.contract !== "event-severity-policy-replay-v1" || config.control) throw new Error("Requires saved non-control severity replay");
if (fs.existsSync(output)) throw new Error("Choose a new output directory");
const fc = read(config.forecast, "config.json"), baseSummary = read(config.source, "summary.json");
const hash = createHash("sha256").update(fs.readFileSync(path.join(config.source, "config.json")));
for (const s of baseSummary) hash.update(fs.readFileSync(path.join(config.source, `${s.window.id}-model.json`)));
if (hash.digest("hex") !== fc.sourceHash) throw new Error("Saved base models changed");
fs.mkdirSync(output, { recursive: true });
fs.writeFileSync(path.join(output, "config.json"), JSON.stringify({ source, contract: "event-severity-policy-audit-v1",
  caveat: "Calibration chooses each family depth; already examined test data are diagnostic only. No test ranking or promotion. Zero-tilt controls compare identical heads and continuations." }, null, 2));
const files = ["scripts/audit-event-severity-policy.ts", "scripts/research-event-policy.ts", "packages/bot-algo/src/event-severity.ts", "packages/bot-algo/src/event-log-policy.ts"];
fs.writeFileSync(path.join(output, "sources.json"), JSON.stringify(Object.fromEntries(files.map(f => [f, fs.readFileSync(path.join(root, f), "utf8")]))));
const results = [];
for (const s of summary) {
  const started = performance.now(), { window } = s, id = window.id;
  const model = read(config.source, `${id}-model.json`), heads = read(source, `${id}-severity-model.json`);
  const projected = read(config.incumbent, `${id}-continuation.json`), selected = read(source, `${id}-selection.json`);
  const c = loadEventCandles(window.startTime - 2 * DAY, window.endTime + DAY);
  const policies = [restoreEventPolicy(model.policy), restoreEventPolicy(projected.policy)];
  const options = (p: EventPolicy, severity: boolean, blend = heads.selected.blend) => ({ sizeSign: {
    head: heads.head, blend: heads.headSetting.blend, fastVolatility: heads.headSetting.fastVolatility, eventHistory: heads.headSetting.eventHistory,
    lookahead: buildEventOutcomeLookahead(p, 5, a => eventSizeSignGroup(a.return, heads.head.thresholdLogBps)),
  }, ...(severity ? { severity: { head: heads.severityHead, blend, lookahead: buildEventKernelLookahead(p) } } : {}) });
  const variants = [];
  for (const choice of ["base", "head-frozen", "head", "severity-frozen", "severity"]) {
    const candidate = selected.calibration.find((r: any) => r.choice === choice);
    if (!candidate) continue;
    const p = policies[choice === "head" || choice === "severity" ? 1 : 0];
    const replay = replayEventPolicy(c, p, window.startTime, window.endTime, candidate.depth,
      { ...(choice === "base" ? {} : options(p, choice.startsWith("severity"))), cash: candidate.score <= 0, trace: true });
    const { trace, ...metrics } = replay;
    fs.writeFileSync(path.join(output, `${id}-${choice}-trades.json`), JSON.stringify(trace));
    if (choice === s.chosen.choice && (Math.abs(metrics.returnPct - s.test.returnPct) > 1e-10
      || JSON.stringify(trace) !== JSON.stringify(read(source, `${id}-trades.json`)))) throw new Error("Selected policy does not reproduce");
    let control;
    if (choice.startsWith("severity")) {
      const ordinary = replayEventPolicy(c, p, window.startTime, window.endTime, candidate.depth, { ...options(p, false), trace: true });
      const zero = replayEventPolicy(c, p, window.startTime, window.endTime, candidate.depth, { ...options(p, true, 0), trace: true });
      if (ordinary.trace.length !== zero.trace.length) throw new Error("Control decision counts differ");
      let valueError = 0;
      for (let i = 0; i < ordinary.trace.length; i++) {
        const a = ordinary.trace[i] as any, b = zero.trace[i] as any;
        valueError = Math.max(valueError, Math.abs(a.order.value - b.order.value));
        for (const key of ["time", "endTime", "equityBefore", "equityAfter", "orderQuantity", "previousQuantity", "exposureAfter"]) {
          if (a[key] !== b[key]) throw new Error(`Control differs at ${i}: ${key}`);
        }
      }
      if (valueError > 1e-12 || ordinary.returnPct !== zero.returnPct) throw new Error("Control value/return mismatch");
      control = { decisions: zero.trace.length, maximumValueError: valueError, returnPct: zero.returnPct, identicalActionsAndLedger: true };
    }
    const result = { choice, depth: candidate.depth, calibrationScore: candidate.score, test: metrics, control,
      orders: trace.filter((r: any) => r.orderQuantity !== 0).map((r: any) => ({ time: r.time, expectedReturnBps: r.expectedReturnBps,
        exposureBefore: r.exposureBefore, exposureAfterOrder: r.order.exposure, equityBefore: r.equityBefore, orderQuantity: r.orderQuantity })) };
    variants.push(result);
    console.log(JSON.stringify({ window: id, choice, depth: candidate.depth, returnPct: metrics.returnPct, drawdownPct: metrics.maxDrawdownPct, trades: metrics.trades, control }));
  }
  results.push({ window, selected: selected.chosen, variants, elapsedSec: (performance.now() - started) / 1000 });
  fs.writeFileSync(path.join(output, "summary.json"), JSON.stringify(results, null, 2));
}

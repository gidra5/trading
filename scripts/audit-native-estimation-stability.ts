/** Compare frozen partitions with an unconditional law and inspect day variation. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { distributionMetrics, eventLeaf, reestimateEventTree, trainEventDistribution, type MoveSample } from "../packages/bot-algo/src/event-distribution.js";
import { eventMarginalCrps } from "../packages/bot-algo/src/event-crps.js";
import { restoreEventPolicy, serializeEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { decideEventOneStep } from "../packages/bot-algo/src/event-one-step.js";
import { eventCashHorizon } from "../packages/bot-algo/src/event-cash-horizon.js";
import { eventAverageUniqueness } from "../packages/bot-algo/src/event-sampling.js";
import { loadNativeEventCandles, makeSamples, replayEventPolicy } from "./research-event-policy.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const directory = (name: string) => path.resolve(__dirname, "../data/benchmarks", name), output = directory(arg("output"));
const leaveDayOut = process.argv.includes("--leave-day-out");
const uniqueness = process.argv.includes("--uniqueness");
const chainPhases = arg("chain-phases-seconds") ? arg("chain-phases-seconds").split(",").map(Number) : [];
assert.ok(chainPhases.length <= 8 && new Set(chainPhases).size === chainPhases.length
  && chainPhases.every(v => Number.isInteger(v) && v >= 0));
assert.ok(arg("sources") && arg("output") && !fs.existsSync(output));
const read = (name: string, file: string) => JSON.parse(fs.readFileSync(path.join(directory(name), file), "utf8"));
const hash = (data: Buffer | string) => createHash("sha256").update(data).digest("hex"), DAY = 86400000;
fs.mkdirSync(output, { recursive: true });
const save = (file: string, value: unknown) => fs.writeFileSync(path.join(output, file), JSON.stringify(value, null, 2));
save("method.json", { sources: arg("sources").split(","), leaveDayOut, uniqueness, chainPhasesSeconds: chainPhases,
  uniquenessMethod: "If enabled, compute the average inverse concurrency of every completed estimation label over its actual return intervals (start,end], excluding the anchor return. Fit positive-weight empirical joint laws with the original numeric prior and with the original global prior-to-observation ratio. Prior atoms and class mass use the same observation weights. These are alternative empirical populations, not a guarantee of unbiased forecasts or independent samples. Weights never use calibration or test labels. A unit-weight refit must exactly reproduce the source model.",
  uniquenessReference: "https://mlfinpy.readthedocs.io/en/latest/Sampling.html#sample-uniqueness",
  chainPhaseMethod: "If supplied, start a complete event chain at each declared offset inside each estimation segment. Keep segment ends and all diagnostic targets unchanged. Re-estimate each frozen partition using both the original numeric prior and the original global prior-to-observation ratio. Save complete laws and exact marked H1 replays with next-open execution and fee-paying terminal settlement. Phases are sensitivity cases, not independent evidence or candidates from which to select a winner.",
  method: "Verify saved source references and reconstruct each estimation population. Compare physical CRPS with an unconditional empirical joint law on the same observations, without selecting new forecasts or trading policies. Daily diagnostics retain only labels ending within their origin UTC day. Optional leave-day-out sensitivity removes all labels touching each UTC day, re-estimates the full joint law and its prior with frozen partitions, and solves exact marked H1 from cash at one common account/price. Historical feature inputs are unchanged. This is estimation sensitivity, not independent cross-validation or a confidence interval. Overlapping rows and adjacent event chains are not asserted independent. The test day has been inspected in prior research and is not a new independent holdout." });
fs.writeFileSync(path.join(output, "source.ts"), fs.readFileSync(__filename, "utf8"));
save("dependencies.json", Object.fromEntries(["packages/bot-algo/src/event-sampling.ts", "packages/bot-algo/src/event-distribution.ts",
  "scripts/event-fit-periods.ts",
  "packages/bot-algo/src/event-one-step.ts", "packages/bot-algo/src/event-crps.ts", "packages/bot-algo/src/event-cash-horizon.ts",
  "packages/bot-algo/src/event-log-policy.ts", "scripts/research-event-policy.ts"]
  .map(file => [file, fs.readFileSync(path.resolve(__dirname, "..", file), "utf8")])));
const results = [];
for (const name of arg("sources").split(",")) {
  const config = read(name, "config.json"), summary = read(name, "summary.json"), modelBytes = fs.readFileSync(path.join(directory(name), "model.json"));
  assert.ok(config.frozenPartitionSource && ["chain", "stride"].includes(config.sampling));
  for (const ref of config.sourceReferences) assert.equal(hash(fs.readFileSync(ref.file)), ref.sha256);
  const originalPolicy = restoreEventPolicy(JSON.parse(modelBytes.toString())), { model, costs } = originalPolicy;
  if (chainPhases.length || uniqueness) assert.ok(config.weighting !== "uniqueness", "Weighting sensitivity requires an unweighted baseline");
  if (chainPhases.length) assert.ok(config.sampling === "stride" && originalPolicy.tables.length === 0
    && chainPhases.every(v => v < model.clock.maxCandles), "Phase sensitivity requires a native stride baseline and offsets below its timeout");
  const sourceStart = (config.earlierEstimationStart ?? config.finalEstimationStart) - (config.warmupCandles + 1) * 1000;
  const candles = loadNativeEventCandles([
    ...(config.extraDays ? [{ start: sourceStart, end: config.earlierEstimationEnd }] : []),
    { start: config.finalEstimationStart - (config.warmupCandles + 1) * 1000, end: config.calibrationEnd },
    { start: config.start - (config.warmupCandles + 1) * 1000, end: config.end },
  ]);
  const samples = (start: number, end: number, sampling: "stride" | "chain", excluded = config.excluded) =>
    makeSamples(candles, model.clock, start, end, excluded, sampling === "chain" ? 1 : config.stride, sampling, model.featureNames);
  const estimation = [...(config.extraDays ? samples(config.earlierEstimationStart, config.earlierEstimationEnd, config.sampling) : []),
    ...samples(config.finalEstimationStart, config.fitEnd, config.sampling)];
  assert.equal(estimation.length, summary.estimation);
  const unconditional = trainEventDistribution(estimation, model.clock,
    { maxDepth: 0, minLeaf: 1, prior: 0, featureNames: model.featureNames });
  const targetHash = (rows: MoveSample[]) => hash(JSON.stringify(rows.map(s =>
    [candles[s.start].openTime, candles[s.end].openTime, s.return, s.duration, s.low, s.high])));
  const score = (rows: MoveSample[], forecast = model) => ({ physicalTargetsHash: targetHash(rows),
    conditional: { ...distributionMetrics(forecast, rows), ...eventMarginalCrps(forecast, rows) },
    unconditional: { ...distributionMetrics(unconditional, rows), ...eventMarginalCrps(unconditional, rows) } });
  const day = (s: MoveSample) => Math.floor((candles[s.start].openTime + 1000) / DAY);
  const withinDay = estimation.filter(s => candles[s.end].openTime + 1000 < (day(s) + 1) * DAY);
  const describe = (rows: MoveSample[]) => {
    let end = -Infinity, count = 0;
    for (const s of rows.slice().sort((a, b) => a.end - b.end || a.start - b.start)) if (s.start >= end) { count++; end = s.end; }
    return { count: rows.length, maxNonOverlappingLabels: count,
      meanReturnBps: rows.length ? rows.reduce((sum, s) => sum + s.return * 10000, 0) / rows.length : null };
  };
  const daily = [...new Set(withinDay.map(day))].sort().map(d => {
    const rows = withinDay.filter(s => day(s) === d), global = describe(rows);
    return { date: new Date(d * DAY).toISOString().slice(0, 10), global,
      leaves: model.kernels.map((_, leaf) => ({ leaf, ...describe(rows.filter(s => eventLeaf(model, s.features) === leaf)) })) };
  });
  const calibration = samples(config.calibrationStart, config.calibrationEnd, "stride"), test = samples(config.start, config.end, "chain", []);
  const account = { equity: 10000, price: candles[calibration[0].start].close, exposure: 0 };
  const stateDecisions = (forecast: typeof model) => forecast.kernels.map((kernel, leaf) => {
    const order = decideEventOneStep(kernel, account, costs, "marked");
    assert.ok(order.feasible && Number.isFinite(order.value));
    return { leaf, count: forecast.counts[leaf], meanReturnBps: kernel.reduce((sum, a) => sum + a.probability * a.return * 10000, 0),
      quantity: order.quantity, exposure: order.exposure, expectedLogValue: order.value };
  });
  const evaluate = (forecast: typeof model, filePrefix: string) => {
    const policy = { ...originalPolicy, model: forecast };
    save(`${filePrefix}-model.json`, serializeEventPolicy(policy));
    const replay = (label: "calibration" | "test", start: number, end: number) => {
      const { trace, positions, ...metrics } = replayEventPolicy(candles, policy, start, end, 1, { trace: true, oneStepTerminal: "marked" });
      assert.ok(trace.every(r => (r.order as { feasible: boolean }).feasible));
      assert.ok(Math.abs(10000 + metrics.longPnl + metrics.shortPnl - metrics.fees - metrics.borrow - metrics.finalEquity) < 1e-7);
      assert.ok(Math.abs(metrics.positionSummary.equityChange - (metrics.finalEquity - 10000)) < 1e-7);
      save(`${filePrefix}-${label}-trades.json`, trace); save(`${filePrefix}-${label}-positions.json`, positions);
      return metrics;
    };
    return { modelFile: `${filePrefix}-model.json`, states: stateDecisions(forecast),
      cashBoundDepth: eventCashHorizon(forecast, costs, 100).verifiedDepth,
      calibration: score(calibration, forecast).conditional, test: score(test, forecast).conditional,
      calibrationReplay: replay("calibration", config.calibrationStart, config.calibrationEnd),
      testReplay: replay("test", config.start, config.end) };
  };
  const omissions = leaveDayOut ? [...new Set(estimation.map(day))].sort().map(d => {
    const start = d * DAY, end = start + DAY;
    const retained = estimation.filter(s => candles[s.start].openTime + 1000 >= end || candles[s.end].openTime + 1000 < start);
    assert.ok(retained.length && retained.length < estimation.length);
    const weights = config.weighting === "uniqueness" ? eventAverageUniqueness(retained) : undefined;
    const prior = weights && config.priorMode === "matched-ratio"
      ? config.priorStrength * weights.reduce((sum, w) => sum + w, 0) / retained.length : config.prior;
    const forecast = reestimateEventTree(model, retained, prior, weights);
    assert.deepEqual(forecast.nodes, model.nodes);
    return { omittedDay: new Date(start).toISOString().slice(0, 10), retained: retained.length,
      removed: estimation.length - retained.length, states: stateDecisions(forecast),
      calibration: score(calibration, forecast).conditional, test: score(test, forecast).conditional };
  }) : undefined;
  const phasePopulations = chainPhases.map(phase => {
    const segments = [
      ...(config.extraDays ? [{ start: config.earlierEstimationStart, end: config.earlierEstimationEnd }] : []),
      { start: config.finalEstimationStart, end: config.fitEnd },
    ];
    const rows = segments.flatMap(segment => {
      const chain = samples(segment.start + phase * 1000, segment.end, "chain");
      assert.ok(chain.length && chain.every((row, i) => candles[row.start].openTime + 1000 >= segment.start + phase * 1000
        && candles[row.end].openTime + 1000 < segment.end && (!i || row.start >= chain[i - 1].end)));
      return chain;
    });
    return { phase, rows, keys: new Set(rows.map(s => `${s.start}:${s.end}`)) };
  });
  const phaseResults = phasePopulations.map(({ phase, rows }) => {
    const cases = ["fixed", "matched-ratio"].map(priorMode => {
      const prior = priorMode === "fixed" ? config.prior : config.prior * rows.length / estimation.length;
      const forecast = reestimateEventTree(model, rows, prior);
      assert.deepEqual(forecast.nodes, model.nodes);
      const filePrefix = `${name}-phase-${phase}-${priorMode}`;
      return { priorMode, prior, ...evaluate(forecast, filePrefix) };
    });
    return { phaseSeconds: phase, observations: rows.length, physicalTargetsHash: targetHash(rows),
      support: model.kernels.map((_, leaf) => ({ leaf, ...describe(rows.filter(s => eventLeaf(model, s.features) === leaf)) })), cases };
  });
  const phaseOverlap = phasePopulations.flatMap((a, i) => phasePopulations.slice(i + 1).map(b =>
    ({ firstPhaseSeconds: a.phase, secondPhaseSeconds: b.phase, identicalLabels: [...a.keys].filter(key => b.keys.has(key)).length })));
  let weightedResults;
  if (uniqueness) {
    assert.ok(config.sampling === "stride" && originalPolicy.tables.length === 0);
    const unitControl = reestimateEventTree(model, estimation, config.prior, estimation.map(() => 1));
    assert.deepEqual(unitControl, model, "Unit-weight reconstruction must preserve the complete source model");
    const weights = eventAverageUniqueness(estimation), totalMass = weights.reduce((sum, w) => sum + w, 0);
    save(`${name}-uniqueness-weights.json`, estimation.map((s, i) =>
      ({ start: candles[s.start].openTime + 1000, end: candles[s.end].openTime + 1000, weight: weights[i] })));
    const support = model.kernels.map((_, leaf) => {
      const indices = estimation.flatMap((s, i) => eventLeaf(model, s.features) === leaf ? [i] : []);
      const mass = indices.reduce((sum, i) => sum + weights[i], 0);
      return { leaf, observations: indices.length, mass, meanReturnBps: mass ? indices.reduce((sum, i) => sum + weights[i] * estimation[i].return * 10000, 0) / mass : null };
    });
    weightedResults = { totalMass, support, unitControlExact: true, cases: ["fixed", "matched-ratio"].map(priorMode => {
      const prior = priorMode === "fixed" ? config.prior : config.prior * totalMass / estimation.length;
      const forecast = reestimateEventTree(model, estimation, prior, weights);
      assert.deepEqual(forecast.nodes, model.nodes);
      return { priorMode, prior, ...evaluate(forecast, `${name}-uniqueness-${priorMode}`) };
    }) };
  }
  const result = { name, modelHash: hash(modelBytes), featureNames: model.featureNames, nodes: model.nodes,
    estimation: estimation.length, estimationTargetsHash: targetHash(estimation), daily,
    calibration: score(calibration), test: score(test),
    ...(omissions || phaseResults.length || weightedResults ? { referenceAccount: account, originalStates: stateDecisions(model) } : {}),
    ...(omissions ? { leaveDayOut: omissions } : {}),
    ...(phaseResults.length ? { chainPhases: phaseResults, chainPhaseOverlap: phaseOverlap } : {}),
    ...(weightedResults ? { uniqueness: weightedResults } : {}) };
  results.push(result); save("summary.json", { results });
  console.log(JSON.stringify({ name, calibration: result.calibration, test: result.test, daily }));
  if (phaseResults.length) console.log(JSON.stringify({ name, chainPhases: phaseResults.map(row => ({ phase: row.phaseSeconds,
    observations: row.observations, support: row.support, cases: row.cases.map(c => ({ priorMode: c.priorMode, prior: c.prior,
      states: c.states, cashBoundDepth: c.cashBoundDepth, calibrationReturnPct: c.calibrationReplay.returnPct, testReturnPct: c.testReplay.returnPct })) })),
    chainPhaseOverlap: phaseOverlap }));
  if (weightedResults) console.log(JSON.stringify({ name, uniqueness: { ...weightedResults, cases: weightedResults.cases.map(c =>
    ({ priorMode: c.priorMode, prior: c.prior, states: c.states, cashBoundDepth: c.cashBoundDepth,
      calibrationReturnPct: c.calibrationReplay.returnPct, testReturnPct: c.testReplay.returnPct,
      calibrationCrps: c.calibration.returnCrpsBps, testCrps: c.test.returnCrpsBps })) } }));
}

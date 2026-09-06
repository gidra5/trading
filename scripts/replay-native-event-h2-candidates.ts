/** Replay a compact-law H2 candidate policy with exact next-open accounting. */
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { prepareEventExecutionBackup } from "../packages/bot-algo/src/event-execution-backup.js";
import { prepareEventExecutionOneStep } from "../packages/bot-algo/src/event-execution-one-step.js";
import { evaluateEventExecutionPath } from "../packages/bot-algo/src/event-execution-path.js";
import { EVENT_FEATURES, eventFeatures, eventLeaf } from "../packages/bot-algo/src/event-distribution.js";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventProbabilityFromReturnWeight, eventSignMass, predictEventSign } from "../packages/bot-algo/src/event-sign.js";
import { eventIntervalConfidence, eventRelativeUncertaintyConfidence,
  eventUncertaintyRiskFloor } from "../packages/bot-algo/src/event-uncertainty.js";
import { usesNativeSecondTradeFlowFeatures } from "../packages/bot-algo/src/event-second-features.js";
import { loadEventCandles, loadNativeEventCandles, replayEventPolicy } from "./research-event-policy.js";

const arg = (key: string, fallback = "") => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? fallback : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), directory = (name: string) => path.resolve(root, "data/benchmarks", name);
const source = directory(arg("source")), fullSource = directory(arg("full-source")), output = directory(arg("output"));
const signSource = arg("sign-source") ? directory(arg("sign-source")) : undefined;
const directionEnsembleSource = arg("direction-ensemble-source") ? directory(arg("direction-ensemble-source")) : undefined;
const ensembleSource = arg("ensemble-source") ? directory(arg("ensemble-source")) : undefined;
const blendUncertaintySource = arg("blend-uncertainty-source") ? directory(arg("blend-uncertainty-source")) : undefined;
const meanUncertaintySource = arg("mean-uncertainty-source") ? directory(arg("mean-uncertainty-source")) : undefined;
const phase = arg("phase", "calibration");
const minImprovementBps = Number(arg("min-improvement-bps") || "0");
const initialRiskBps = Number(arg("initial-risk-bps") || "0");
const protectedProfitFraction = Number(arg("protected-profit-fraction") || "0");
const globalRiskRootMaxLots = Number(arg("global-risk-root-max-lots") || "0");
const uncertaintyRiskScaling = process.argv.includes("--uncertainty-risk-scaling");
const softUncertaintySizing = process.argv.includes("--soft-uncertainty-sizing");
const directionEnsembleRole = arg("direction-ensemble-role", "forecast");
assert.ok(arg("source") && arg("full-source") && arg("output") && !fs.existsSync(output));
assert.ok(["calibration", "test"].includes(phase));
assert.ok(["forecast", "risk-only"].includes(directionEnsembleRole) && (directionEnsembleSource || directionEnsembleRole === "forecast"),
  "Direction ensemble role must be forecast or risk-only");
assert.ok(Number.isFinite(minImprovementBps) && minImprovementBps >= 0);
assert.ok(Number.isFinite(initialRiskBps) && initialRiskBps >= 0);
assert.ok(Number.isFinite(protectedProfitFraction) && protectedProfitFraction >= 0 && protectedProfitFraction <= 1);
assert.ok(Number.isSafeInteger(globalRiskRootMaxLots) && globalRiskRootMaxLots >= 0);
assert.ok(!signSource || !directionEnsembleSource, "Use either a sign source or a direction ensemble");
assert.ok(!ensembleSource || signSource, "A magnitude ensemble requires its sign source");
assert.ok(!blendUncertaintySource || signSource, "A blend-uncertainty source requires its sign source");
assert.ok(!blendUncertaintySource || !ensembleSource,
  "Blend uncertainty and the magnitude ensemble describe different forecast laws");
assert.ok(!meanUncertaintySource || !signSource && !directionEnsembleSource && !blendUncertaintySource && !ensembleSource,
  "Mean uncertainty is an alternative forecast law");
assert.ok(!uncertaintyRiskScaling || meanUncertaintySource || blendUncertaintySource || directionEnsembleSource,
  "Uncertainty risk scaling requires a forecast interval");
assert.ok(!softUncertaintySizing || meanUncertaintySource || blendUncertaintySource || directionEnsembleSource,
  "Soft uncertainty sizing requires a forecast interval");
assert.ok(!(softUncertaintySizing && uncertaintyRiskScaling),
  "Soft uncertainty sizing and robust uncertainty withdrawal are alternative policies");
const read = (file: string) => JSON.parse(fs.readFileSync(file, "utf8"));
const hash = (file: string) => createHash("sha256").update(fs.readFileSync(file)).digest("hex");
const executionConfig = read(path.join(source, "config.json")), compact = read(path.join(source, "law.json"));
const compiled = read(path.join(source, "summary.json")), fullConfig = read(path.join(fullSource, "config.json"));
const fullCompiled = read(path.join(fullSource, "summary.json")), full = read(path.join(fullSource, "law.json"));
assert.equal(executionConfig.contract, "native-event-execution-law-v1");
assert.equal(fullConfig.contract, "native-event-execution-law-v1");
assert.equal(hash(path.join(source, "law.json")), compiled.lawHash);
assert.equal(hash(path.join(fullSource, "law.json")), fullCompiled.lawHash);
assert.equal(compact.modelHash, full.modelHash); assert.deepEqual(compact.costs, full.costs);
const baseSource = executionConfig.source, config = read(path.join(baseSource, "config.json"));
const modelFile = path.join(baseSource, "model.json"); assert.equal(hash(modelFile), compact.modelHash);
const policy = restoreEventPolicy(read(modelFile));
const directionEnsembleConfig = directionEnsembleSource ? read(path.join(directionEnsembleSource, "config.json")) : undefined;
const directionEnsemble = directionEnsembleSource ? read(path.join(directionEnsembleSource, "summary.json")) : undefined;
const directionFastSource = directionEnsembleConfig?.fastSource as string | undefined;
const directionSlowSource = directionEnsembleConfig?.slowSource as string | undefined;
const directionSource = signSource ?? directionFastSource;
const signConfig = directionSource ? read(path.join(directionSource, "config.json")) : undefined;
const signHead = directionSource ? read(path.join(directionSource, "head.json")) : undefined;
const signPolicy = signConfig ? restoreEventPolicy(read(path.join(signConfig.source, "model.json"))).model : undefined;
const signFeatureNames = signConfig?.featureNames as readonly string[] | undefined;
const slowConfig = directionSlowSource ? read(path.join(directionSlowSource, "config.json")) : undefined;
const slowModel = directionSlowSource ? read(path.join(directionSlowSource, "slow-model.json")) : undefined;
const ensemble = ensembleSource ? read(path.join(ensembleSource, "summary.json")) : undefined;
const blendUncertainty = blendUncertaintySource ? read(path.join(blendUncertaintySource, "summary.json")) : undefined;
const meanUncertainty = meanUncertaintySource ? read(path.join(meanUncertaintySource, "summary.json")) : undefined;
const ensembleScale = ensemble?.selected?.name === "soft-direction-expected-absolute"
  ? Number(ensemble.selected.parameters.scale) : undefined;
if (signConfig) {
  assert.equal(signConfig.contract, "native-event-sign-screen-v2");
  assert.equal(signConfig.objective, "return-weighted"); assert.equal(signConfig.penalty, .01);
  assert.deepEqual(signConfig.blends, [0, .5, 1]);
}
if (directionEnsembleConfig) {
  assert.equal(directionEnsembleConfig.contract, "native-event-direction-ensemble-screen-v1");
  assert.equal(path.resolve(directionEnsembleConfig.source), baseSource);
  assert.equal(path.resolve(directionFastSource!), directionSource);
  assert.equal(executionConfig.directionEnsembleSource, directionEnsembleSource);
  assert.equal(executionConfig.directionEnsembleConfigHash, hash(path.join(directionEnsembleSource!, "config.json")));
  assert.equal(executionConfig.directionEnsembleSummaryHash, hash(path.join(directionEnsembleSource!, "summary.json")));
  if (executionConfig.directionFastConfigHash)
    assert.equal(executionConfig.directionFastConfigHash, hash(path.join(directionFastSource!, "config.json")));
  if (executionConfig.directionFastHeadHash)
    assert.equal(executionConfig.directionFastHeadHash, hash(path.join(directionFastSource!, "head.json")));
  if (executionConfig.directionSlowConfigHash)
    assert.equal(executionConfig.directionSlowConfigHash, hash(path.join(directionSlowSource!, "config.json")));
  if (executionConfig.directionSlowModelHash)
    assert.equal(executionConfig.directionSlowModelHash, hash(path.join(directionSlowSource!, "slow-model.json")));
  assert.equal(slowConfig.contract, "native-event-slow-direction-screen-v1");
  assert.equal(path.resolve(slowConfig.source), baseSource);
  for (const reference of slowConfig.slowSourceReferences ?? []) assert.equal(hash(reference.file), reference.sha256);
  assert.ok(directionEnsemble.selected?.specification);
}
if (ensembleSource) {
  assert.equal(ensemble.contract, "native-event-sign-magnitude-ensemble-screen-v1");
  assert.equal(path.resolve(ensemble.source), signSource);
  assert.ok(Number.isFinite(ensembleScale) && ensembleScale >= 0);
}
if (blendUncertaintySource) {
  assert.equal(blendUncertainty.contract, "native-event-sign-blend-uncertainty-v1");
  assert.equal(path.resolve(blendUncertainty.source), signSource);
  assert.ok(Number.isFinite(blendUncertainty.centralWeight)
    && blendUncertainty.centralWeight >= 0 && blendUncertainty.centralWeight <= 1);
  assert.equal(blendUncertainty.bootstrap.interval.confidence, .9);
  assert.ok(Number.isFinite(blendUncertainty.bootstrap.interval.low)
    && Number.isFinite(blendUncertainty.bootstrap.interval.high)
    && blendUncertainty.bootstrap.interval.low >= 0
    && blendUncertainty.bootstrap.interval.low <= blendUncertainty.centralWeight
    && blendUncertainty.centralWeight <= blendUncertainty.bootstrap.interval.high
    && blendUncertainty.bootstrap.interval.high <= 1);
}
if (meanUncertaintySource) {
  assert.equal(meanUncertainty.contract, "native-event-mean-conformal-uncertainty-v1");
  assert.equal(meanUncertainty.modelHash, compact.modelHash);
  assert.equal(meanUncertainty.leaves.length, policy.model.kernels.length);
  assert.ok(meanUncertainty.leaves.every((row: any, leaf: number) => row.leaf === leaf
    && Number.isFinite(row.lowBps) && Number.isFinite(row.highBps) && row.lowBps <= row.highBps));
}
for (const ref of executionConfig.sourceReferences) assert.equal(hash(ref.file), ref.sha256);
const phaseStart = phase === "calibration" ? config.calibrationStart : config.window.startTime;
const phaseEnd = phase === "calibration" ? config.calibrationEnd : config.window.endTime;
const start = Number(arg("start") || phaseStart), end = Number(arg("end") || phaseEnd);
assert.ok(Number.isSafeInteger(start) && Number.isSafeInteger(end) && phaseStart <= start && start < end && end <= phaseEnd,
  "Replay range must stay inside its declared phase");
const sourceStart = start - (config.warmupCandles + 1) * 1000;
const references = [];
const includeTradeFlow = usesNativeSecondTradeFlowFeatures(policy.model.featureNames)
  || Boolean(signPolicy && usesNativeSecondTradeFlowFeatures(signPolicy.featureNames))
  || Boolean(signFeatureNames && usesNativeSecondTradeFlowFeatures(signFeatureNames));
for (let day = Math.floor(sourceStart / 86400000) * 86400000; day < end; day += 86400000) {
  const file = path.join(root, "data/market/immutable/refs/candles/spot-btcusdt/btcusdt/1s",
    `${new Date(day).toISOString().slice(0, 10)}.json`);
  references.push({ file, sha256: hash(file) });
  if (includeTradeFlow) {
    const flowFile = path.join(root, "data/market/immutable/refs/trade-flow/spot-btcusdt/btcusdt/1s",
      `${new Date(day).toISOString().slice(0, 10)}.json`);
    references.push({ file: flowFile, sha256: hash(flowFile) });
  }
}
const kernels = compact.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: compact.paths[atom.path],
})));
const fullKernels = full.kernels.map((kernel: any[]) => kernel.map(atom => ({
  probability: atom.probability, next: atom.next, path: full.paths[atom.path],
})));
const h1 = kernels.map((kernel: any[]) => prepareEventExecutionOneStep(kernel));
const h2 = prepareEventExecutionBackup(kernels), step = compact.costs.quantityStep;
const closingCostRate = (compact.costs.feeBps + compact.costs.slippageBps) / 10_000;
const childCache = new Map<string, ReturnType<ReturnType<typeof prepareEventExecutionOneStep>>>();
const signChildSolvers = new Map<string, ReturnType<typeof prepareEventExecutionOneStep>>();
let plannerSeconds = 0, evaluatedCandidates = 0, globallyCoveredDecisions = 0, globallyCertifiedDecisions = 0;
let globallyCoveredLots = 0, maximumGlobalRootLots = 0;
let candles: ReturnType<typeof loadEventCandles>;
const bounded = (probability: number) => Math.max(1e-6, Math.min(1 - 1e-6, probability));
const logit = (probability: number) => Math.log(bounded(probability) / (1 - bounded(probability)));
const sigmoid = (value: number) => 1 / (1 + Math.exp(-Math.max(-40, Math.min(40, value))));
const combineDirectionProbability = (slowProbability: number, fastProbability: number,
  specification: { slowCoefficient: number; fastCoefficient: number }) => bounded(sigmoid(
    specification.slowCoefficient * logit(slowProbability) + specification.fastCoefficient * logit(fastProbability)));
const selectedDirection = directionEnsemble?.selected?.specification as
  { slowCoefficient: number; fastCoefficient: number } | undefined;
const eligibleDirectionSpecifications = directionEnsemble?.candidates?.filter((row: any) =>
  row.selection.returnMse < directionEnsemble.baseline.selection.returnMse
  && row.selection.signNll < directionEnsemble.baseline.selection.signNll
  && row.selectionPositiveMseDays / row.selectionDays.length >= .6
  && row.selectionPositiveNllDays / row.selectionDays.length >= .6)
  .map((row: any) => row.specification) as Array<{ slowCoefficient: number; fastCoefficient: number }> | undefined;
if (directionEnsemble) assert.equal(eligibleDirectionSpecifications!.length, directionEnsemble.eligibleCandidates);
const MINUTE = 60_000;
const minuteCandles = slowConfig ? loadEventCandles(slowConfig.slowTrainStart - 1441 * MINUTE, end) : undefined;
const minuteByClose = minuteCandles ? new Map(minuteCandles.map((row, index) => [row.openTime + MINUTE, index])) : undefined;
const slowLaws = slowModel?.kernels.map((kernel: any[]) => eventSignMass(kernel));
const signStats = (kernel: readonly { probability: number; return: number }[]) => {
  const mass = eventSignMass(kernel), positiveTotal = kernel.reduce((sum, atom) =>
    sum + (atom.return > 0 ? atom.probability * atom.return : 0), 0);
  const negativeTotal = kernel.reduce((sum, atom) =>
    sum + (atom.return < 0 ? atom.probability * atom.return : 0), 0);
  return { ...mass, positiveMean: positiveTotal / (mass.positive || 1), negativeMean: negativeTotal / (mass.negative || 1) };
};
const reweightExecutionMean = (kernel: readonly any[], expectedReturnBps: number) => {
  const stats = signStats(kernel.map(atom => ({ probability: atom.probability, return: atom.path.closeRatio - 1 })));
  const active = stats.positive + stats.negative;
  const target = Math.max(1e-6, Math.min(1 - 1e-6,
    (expectedReturnBps / 10_000 / active - stats.negativeMean) / (stats.positiveMean - stats.negativeMean)));
  return { targetProbability: target, kernel: kernel.map(atom => ({ ...atom,
    probability: atom.path.closeRatio === 1 || !atom.probability ? atom.probability
      : atom.probability * (atom.path.closeRatio > 1 ? active * target / stats.positive : active * (1 - target) / stats.negative),
  })) };
};
const reweightExecutionDirection = (kernel: readonly any[], directionProbability: number) => {
  const stats = signStats(kernel.map(atom => ({ probability: atom.probability, return: atom.path.closeRatio - 1 })));
  const target = Math.max(1e-6, Math.min(1 - 1e-6, directionProbability));
  const active = stats.positive + stats.negative;
  if (!stats.positive || !stats.negative) return { targetProbability: stats.probability,
    kernel: kernel.map(atom => ({ ...atom })) };
  const reweighted = kernel.map(atom => ({ ...atom,
    probability: atom.path.closeRatio === 1 || !atom.probability ? atom.probability
      : atom.probability * (atom.path.closeRatio > 1 ? active * target / stats.positive : active * (1 - target) / stats.negative),
  }));
  const achieved = signStats(reweighted.map(atom => ({ probability: atom.probability,
    return: atom.path.closeRatio - 1 }))).probability;
  assert.ok(Math.abs(achieved - target) < 1e-10, "Execution sign reweighting must preserve ordinary P(up)");
  return { targetProbability: target, kernel: reweighted };
};
const ensembleExpectedReturnBps = (forecast: { leaf: number; adjustedProbability: number; directionProbability: number }) => {
  assert.ok(signPolicy && ensembleScale !== undefined);
  const stats = signStats(signPolicy.kernels[forecast.leaf]);
  const expectedAbsoluteBps = (stats.positive + stats.negative)
    * (forecast.adjustedProbability * stats.positiveMean - (1 - forecast.adjustedProbability) * stats.negativeMean) * 10_000;
  return ensembleScale * (2 * forecast.directionProbability - 1) * expectedAbsoluteBps;
};
const blendDirection = (baseProbability: number, adjustedProbability: number, weight: number) =>
  (1 - weight) * baseProbability + weight * adjustedProbability;
const robustWeights: readonly number[] = blendUncertainty
  ? [blendUncertainty.bootstrap.interval.low, blendUncertainty.bootstrap.interval.high] : [];
const directionEnsembleUncertainty = Boolean(directionEnsemble && (uncertaintyRiskScaling || softUncertaintySizing));
let initialEquity: number | undefined, equityHighWater = -Infinity;
const candidatePolicy = (leaf: number, account: { equity: number; price: number; exposure: number }, time: number) => {
  initialEquity ??= account.equity;
  const liquidatableEquity = account.equity * (1 - Math.abs(account.exposure) * closingCostRate);
  equityHighWater = Math.max(equityHighWater, liquidatableEquity);
  const riskEnabled = initialRiskBps > 0 || protectedProfitFraction > 0
    || uncertaintyRiskScaling || softUncertaintySizing;
  const begin = performance.now(), baseKernel = kernels[leaf];
  let rootKernel = baseKernel, rootProbabilityModels = [baseKernel.map((atom: any) => atom.probability)], sign: any = null;
  let meanInterval: any = null;
  let robustDirection: -1 | 0 | 1 | undefined;
  if (meanUncertainty) {
    meanInterval = meanUncertainty.leaves[leaf];
    rootProbabilityModels = [meanInterval.lowBps, meanInterval.highBps].map((meanBps: number) =>
      reweightExecutionMean(baseKernel, meanBps).kernel.map((atom: any) => atom.probability));
    robustDirection = meanInterval.lowBps > 0 ? 1 : meanInterval.highBps < 0 ? -1 : 0;
  } else if (signPolicy) {
    const index = Math.round((time - 1000 - candles[0].openTime) / 1000);
    assert.equal(candles[index]?.openTime + 1000, time);
    const magnitudeFeatures = eventFeatures(candles, index, signPolicy.featureNames, signPolicy.clock);
    const features = eventFeatures(candles, index, signFeatureNames ?? signPolicy.featureNames, signPolicy.clock);
    const signLeaf = eventLeaf(signPolicy, magnitudeFeatures), sourceStats = signStats(signPolicy.kernels[signLeaf]);
    const raw = predictEventSign(signHead, features);
    const adjusted = eventProbabilityFromReturnWeight(raw, sourceStats.positiveMean, sourceStats.negativeMean);
    let blendWeight = blendUncertainty?.centralWeight ?? .5;
    let combinedProbability = blendDirection(sourceStats.probability, adjusted, blendWeight);
    let robustDirectionInterval = blendUncertainty ? robustWeights.map(weight =>
      blendDirection(sourceStats.probability, adjusted, weight)) : undefined;
    let directionComponents: any = undefined;
    if (directionEnsemble) {
      const minute = minuteByClose!.get(Math.floor(time / MINUTE) * MINUTE);
      assert.notEqual(minute, undefined, "Slow direction history is unavailable at an event decision");
      const slowLeaf = eventLeaf(slowModel, eventFeatures(minuteCandles!, minute!, EVENT_FEATURES, slowModel.clock));
      const slowProbability = slowLaws![slowLeaf].probability;
      combinedProbability = combineDirectionProbability(slowProbability, adjusted, selectedDirection!);
      robustDirectionInterval = directionEnsembleUncertainty ? eligibleDirectionSpecifications!.map(specification =>
        combineDirectionProbability(slowProbability, adjusted, specification)) : undefined;
      blendWeight = NaN;
      directionComponents = { slowLeaf, slowProbability, fastProbability: adjusted,
        specification: selectedDirection, eligibleSpecifications: eligibleDirectionSpecifications };
    }
    const forecast = { leaf: signLeaf, adjustedProbability: adjusted, directionProbability: combinedProbability };
    const ensembleMeanBps = ensembleSource ? ensembleExpectedReturnBps(forecast) : undefined;
    const reweighted = ensembleMeanBps === undefined ? reweightExecutionDirection(baseKernel, combinedProbability)
      : reweightExecutionMean(baseKernel, ensembleMeanBps);
    const riskOnly = Boolean(directionEnsemble && directionEnsembleRole === "risk-only");
    const targetProbability = riskOnly ? signStats(baseKernel.map((atom: any) => ({ probability: atom.probability,
      return: atom.path.closeRatio - 1 }))).probability : reweighted.targetProbability;
    rootKernel = riskOnly ? baseKernel : reweighted.kernel;
    if (robustDirectionInterval && !riskOnly) rootProbabilityModels = robustDirectionInterval.map(direction =>
      reweightExecutionDirection(baseKernel, direction).kernel.map((atom: any) => atom.probability));
    else rootProbabilityModels = [rootKernel.map((atom: any) => atom.probability)];
    if (robustDirectionInterval) robustDirection = Math.min(...robustDirectionInterval) > .5 ? 1
      : Math.max(...robustDirectionInterval) < .5 ? -1 : 0;
    sign = { signLeaf, rawReturnWeight: raw, adjustedProbability: adjusted, baseProbability: sourceStats.probability,
      ...(Number.isFinite(blendWeight) ? { blendWeight } : {}), directionProbability: combinedProbability,
      targetProbability, ensembleMeanBps, directionComponents,
      ...(robustDirectionInterval ? { robustDirectionInterval, robustDirection,
        ...(blendUncertainty ? { robustBlendInterval: robustWeights } : {}) } : {}),
      expectedReturnBps: rootKernel.reduce((sum: number, atom: any) =>
        sum + atom.probability * (atom.path.closeRatio - 1), 0) * 10_000,
      directionEnsembleRole: directionEnsemble ? directionEnsembleRole : undefined };
  }
  if (softUncertaintySizing)
    rootProbabilityModels = [rootKernel.map((atom: any) => atom.probability)];
  const uncertaintyConfidence = !uncertaintyRiskScaling && !softUncertaintySizing ? 1 : meanInterval
    ? (softUncertaintySizing ? eventRelativeUncertaintyConfidence : eventIntervalConfidence)(
      meanInterval.meanBps, meanInterval.lowBps, meanInterval.highBps)
    : (softUncertaintySizing ? eventRelativeUncertaintyConfidence : eventIntervalConfidence)(
      sign.directionProbability, Math.min(...sign.robustDirectionInterval),
      Math.max(...sign.robustDirectionInterval), .5);
  const riskState = uncertaintyRiskScaling || softUncertaintySizing
    ? eventUncertaintyRiskFloor({ initialEquity, liquidatableHighWater: equityHighWater,
      maximumInitialRiskBps: initialRiskBps, minimumProtectedProfitFraction: protectedProfitFraction,
      confidence: uncertaintyConfidence })
    : { floor: Math.max(0, equityHighWater - initialEquity) > 0
      ? initialEquity + protectedProfitFraction * Math.max(0, equityHighWater - initialEquity)
      : initialEquity * (1 - initialRiskBps / 10_000),
      peakProfit: Math.max(0, equityHighWater - initialEquity), confidence: 1,
      effectiveInitialRiskBps: initialRiskBps, protectedProfitFraction };
  const riskFloor = riskState.floor, peakProfit = riskState.peakProfit;
  const rootAlternativeProbabilities = (blendUncertainty || meanUncertainty
    || directionEnsembleUncertainty && directionEnsembleRole === "forecast") && !softUncertaintySizing
    ? rootProbabilityModels : rootProbabilityModels.slice(1);
  const rootOneStep = prepareEventExecutionOneStep(rootKernel, "marked",
    rootAlternativeProbabilities.length ? { alternativeProbabilities: rootAlternativeProbabilities } : {});
  const available = rootKernel.filter((atom: any) => atom.path.openingAvailable);
  const opens = available.map((atom: any) => account.price * atom.path.openRatio);
  const minLots = Math.ceil(Math.max(compact.costs.minQuantity,
    compact.costs.minNotional / Math.min(...opens)) / step - 1e-8);
  const safeMaxLots = Math.floor(compact.costs.maxNotional / Math.max(...opens) / step + 1e-8);
  const currentLots = Math.round(account.exposure * account.equity / account.price / step);
  const leverageLots = Math.floor(compact.costs.maxLeverage * account.equity / account.price / step);
  const clamp = (lots: number) => Math.max(-safeMaxLots, Math.min(safeMaxLots, lots));
  let lots = new Set<number>([0, Math.round(h1[leaf](account).quantity / step),
    Math.round(rootOneStep(account).quantity / step), -currentLots, clamp(-2 * currentLots),
    -safeMaxLots, -Math.floor(safeMaxLots / 2), Math.floor(safeMaxLots / 2), safeMaxLots,
    clamp(-leverageLots - currentLots), clamp(leverageLots - currentLots)]);
  for (const multiple of [1, 2, 3]) for (const sign of [-1, 1]) lots.add(sign * multiple * minLots);
  if (riskEnabled) {
    const constrainedOneStep = rootOneStep(account, { minimumLiquidatableEquity: riskFloor });
    if (constrainedOneStep.feasible) lots.add(Math.round(constrainedOneStep.quantity / step));
  }
  let rootCoverage: any = null;
  if (globalRiskRootMaxLots) {
    assert.ok(riskEnabled, "Global risk-root enumeration requires an enabled risk floor");
    const cover = prepareEventExecutionOneStep(rootKernel, "marked", { captureRegions: true,
      ...(rootAlternativeProbabilities.length ? { alternativeProbabilities: rootAlternativeProbabilities } : {}) })(account,
      { minimumLiquidatableEquity: riskFloor });
    assert.ok("requestRegions" in cover);
    const rootLots = cover.requestRegions.reduce((sum, [lo, hi]) => sum + hi - lo + 1, 0);
    assert.ok(Number.isSafeInteger(rootLots) && rootLots <= globalRiskRootMaxLots,
      `Risk-admissible root cover has ${rootLots} lots, above the ${globalRiskRootMaxLots} computation gate`);
    const transitionClasses = new Map<string, { lot: number; equivalentLots: number }>();
    for (const [lo, hi] of cover.requestRegions) for (let lot = lo; lot <= hi; lot++) {
      const request = lot * step;
      const signature = JSON.stringify(rootKernel.map((atom: any) => {
        const next = evaluateEventExecutionPath(atom.path, account, request);
        return [next.equity, next.price, next.exposure];
      }));
      const saved = transitionClasses.get(signature);
      if (!saved) transitionClasses.set(signature, { lot, equivalentLots: 1 });
      else {
        saved.equivalentLots++;
        if (Math.abs(lot) < Math.abs(saved.lot)) saved.lot = lot;
      }
    }
    lots = new Set([...transitionClasses.values()].map(row => row.lot));
    globallyCoveredDecisions++; globallyCoveredLots += rootLots;
    maximumGlobalRootLots = Math.max(maximumGlobalRootLots, rootLots);
    rootCoverage = { regions: cover.requestRegions, lots: rootLots, maximumLots: cover.search.maximumLots,
      transitionClasses: transitionClasses.size, collapsedEquivalentLots: rootLots - transitionClasses.size,
      complete: true, scope: "every-floor-admissible-root-lot" };
  }
  const candidatesBeforeUncertainty = lots.size;
  if (!softUncertaintySizing && robustDirection === 0) {
    const currentQuantity = account.exposure * account.equity / account.price;
    lots = new Set([...lots].filter(lot => rootKernel.every((atom: any) => {
      const next = evaluateEventExecutionPath(atom.path, account, lot * step);
      return next.quantity * currentQuantity >= -1e-12
        && Math.abs(next.quantity) <= Math.abs(currentQuantity) + step * 1e-7;
    })));
  }
  const uncertaintyFilteredCandidates = candidatesBeforeUncertainty - lots.size;
  assert.ok(lots.size, "The risk floor leaves no admissible root request");
  const solveCandidate = (request: number, minimumFloor?: number) => {
    if (!signPolicy && minimumFloor === undefined) {
      const result = h2(leaf, account, request); assert.ok(result.complete && result.value !== null);
      return { value: result.value, continuationStates: result.continuationStates,
        continuationSolves: result.continuationSolves, modelValues: [result.value] };
    }
    const modelValues = rootProbabilityModels.map(() => 0);
    let value = -Infinity, continuationSolves = 0;
    for (let atomIndex = 0; atomIndex < rootKernel.length; atomIndex++) {
      const atom = rootKernel[atomIndex];
      const next = evaluateEventExecutionPath(atom.path, account, request);
      if (!Number.isFinite(next.logGrowth)) break;
      const nextLiquidatable = next.equity * (1 - Math.abs(next.exposure) * closingCostRate);
      if (minimumFloor !== undefined && nextLiquidatable < minimumFloor - 1e-8) break;
      const childHighWater = Math.max(equityHighWater, nextLiquidatable);
      const nextAccount = { equity: next.equity, price: next.price, exposure: next.exposure };
      let childSolver = h1[atom.next], nextDirection: number | undefined, nextMeanBps: number | undefined;
      let nextModelDirections: number[] | undefined;
      let nextUncertaintyDirections: number[] | undefined;
      let nextMeanBounds: number[] | undefined;
      if (signPolicy) {
        const nextSign = atom.path.nextSign;
        assert.ok(Number.isFinite(nextSign?.directionProbability), "Sign-enriched execution path is missing its causal successor prediction");
        if (blendUncertainty) assert.ok(Number.isFinite(nextSign.baseProbability)
          && Number.isFinite(nextSign.adjustedProbability),
        "Robust sign blend requires causal base and adjusted successor probabilities");
        if (directionEnsemble) {
          assert.ok(Number.isFinite(nextSign.slowProbability) && Number.isFinite(nextSign.fastProbability),
            "Direction-enriched execution path is missing its causal slow/fast successor predictions");
          nextDirection = combineDirectionProbability(nextSign.slowProbability, nextSign.fastProbability,
            selectedDirection!);
          nextUncertaintyDirections = directionEnsembleUncertainty ? eligibleDirectionSpecifications!.map(specification =>
            combineDirectionProbability(nextSign.slowProbability, nextSign.fastProbability, specification)) : [nextDirection];
        } else {
          const nextBlendWeight = blendUncertainty?.centralWeight ?? .5;
          nextDirection = blendUncertainty ? blendDirection(nextSign.baseProbability,
            nextSign.adjustedProbability, nextBlendWeight) : nextSign.directionProbability;
          nextUncertaintyDirections = blendUncertainty ? robustWeights.map(weight =>
            blendDirection(nextSign.baseProbability, nextSign.adjustedProbability, weight)) : [nextDirection];
        }
        nextMeanBps = ensembleSource ? ensembleExpectedReturnBps({ ...nextSign, directionProbability: nextDirection }) : undefined;
        nextModelDirections = softUncertaintySizing ? [nextDirection] : nextUncertaintyDirections;
        if (!directionEnsemble || directionEnsembleRole === "forecast") {
          const solverKey = JSON.stringify([atom.next, nextModelDirections, nextMeanBps]);
          const saved = signChildSolvers.get(solverKey);
          if (saved) childSolver = saved;
          else {
            const reweighted = nextMeanBps === undefined
              ? reweightExecutionDirection(kernels[atom.next], nextModelDirections[0])
              : reweightExecutionMean(kernels[atom.next], nextMeanBps);
            const alternativeProbabilities = nextModelDirections.slice(1).map(direction =>
              reweightExecutionDirection(kernels[atom.next], direction).kernel.map((row: any) => row.probability));
            childSolver = prepareEventExecutionOneStep(reweighted.kernel, "marked",
              alternativeProbabilities.length ? { alternativeProbabilities } : {});
            signChildSolvers.set(solverKey, childSolver);
          }
        }
      } else if (meanUncertainty) {
        const interval = meanUncertainty.leaves[atom.next];
        nextMeanBounds = [interval.lowBps, interval.highBps];
        if (!softUncertaintySizing) {
          const solverKey = JSON.stringify(["mean-uncertainty", atom.next, nextMeanBounds]);
          const saved = signChildSolvers.get(solverKey);
          if (saved) childSolver = saved;
          else {
            const alternativeProbabilities = nextMeanBounds.map(meanBps =>
              reweightExecutionMean(kernels[atom.next], meanBps).kernel.map((row: any) => row.probability));
            childSolver = prepareEventExecutionOneStep(kernels[atom.next], "marked", { alternativeProbabilities });
            signChildSolvers.set(solverKey, childSolver);
          }
        }
      }
      const childConfidence = !uncertaintyRiskScaling && !softUncertaintySizing ? 1 : meanUncertainty
        ? (softUncertaintySizing ? eventRelativeUncertaintyConfidence : eventIntervalConfidence)(
          meanUncertainty.leaves[atom.next].meanBps, nextMeanBounds![0]!, nextMeanBounds![1]!)
        : (softUncertaintySizing ? eventRelativeUncertaintyConfidence : eventIntervalConfidence)(
          nextDirection!, Math.min(...nextUncertaintyDirections!), Math.max(...nextUncertaintyDirections!), .5);
      const childFloor = minimumFloor === undefined ? undefined : uncertaintyRiskScaling || softUncertaintySizing
        ? eventUncertaintyRiskFloor({ initialEquity: initialEquity!, liquidatableHighWater: childHighWater,
          maximumInitialRiskBps: initialRiskBps, minimumProtectedProfitFraction: protectedProfitFraction,
          confidence: childConfidence }).floor
        : Math.max(0, childHighWater - initialEquity!) > 0
          ? initialEquity! + protectedProfitFraction * Math.max(0, childHighWater - initialEquity!)
          : initialEquity! * (1 - initialRiskBps / 10_000);
      const key = JSON.stringify([atom.next, nextModelDirections, nextMeanBps, nextMeanBounds, childFloor,
        nextAccount.equity, nextAccount.price, nextAccount.exposure]);
      let child = childCache.get(key);
      if (!child) {
        child = childSolver(nextAccount, childFloor === undefined ? {} : { minimumLiquidatableEquity: childFloor });
        childCache.set(key, child); continuationSolves++;
      }
      if (!Number.isFinite(child.value)) break;
      const payoff = next.logGrowth + child.value;
      for (let model = 0; model < rootProbabilityModels.length; model++)
        modelValues[model] += rootProbabilityModels[model][atomIndex] * payoff;
      if (atomIndex === rootKernel.length - 1) value = Math.min(...modelValues);
    }
    return { value, continuationStates: rootKernel.length, continuationSolves,
      modelValues: Number.isFinite(value) ? modelValues : modelValues.map(() => -Infinity) };
  };
  const riskCandidates = [...lots].filter(Number.isSafeInteger).map(lot => {
    const request = lot * step; evaluatedCandidates++;
    const unconstrained = globalRiskRootMaxLots ? null : solveCandidate(request);
    // Reserve the cost of flattening at the next decision. Without it, a path
    // can clear the marked-equity floor yet breach it while de-risking.
    const worstEquity = Math.min(...rootKernel.map((atom: any) => {
      const next = evaluateEventExecutionPath(atom.path, account, request);
      return next.equity * (1 - Math.abs(next.exposure) * closingCostRate);
    }));
    const rootRiskEligible = !riskEnabled || worstEquity >= riskFloor - 1e-8;
    const constrained = riskEnabled && rootRiskEligible ? solveCandidate(request, riskFloor) : unconstrained!;
    return { lot, request, value: constrained.value, unconstrainedValue: unconstrained?.value ?? null,
      continuationStates: constrained.continuationStates, continuationSolves: constrained.continuationSolves,
      unconstrainedContinuationSolves: unconstrained?.continuationSolves ?? null, worstEquity, rootRiskEligible,
      modelValues: constrained.modelValues,
      riskEligible: !riskEnabled || rootRiskEligible && Number.isFinite(constrained.value) };
  });
  const unconstrained = globalRiskRootMaxLots ? null : riskCandidates.reduce((best, row) => row.unconstrainedValue! > best.unconstrainedValue!
    || row.unconstrainedValue === best.unconstrainedValue && Math.abs(row.lot) < Math.abs(best.lot) ? row : best);
  const hold = riskCandidates.find(row => row.lot === 0);
  const pairedCandidates = riskCandidates.map(row => ({ ...row,
    robustImprovement: rootProbabilityModels.length > 1 && hold && row.modelValues.length === hold.modelValues.length
      ? Math.min(...row.modelValues.map((value: number, model: number) => value - hold.modelValues[model]))
      : hold ? row.value - hold.value : 0 }));
  const eligible = pairedCandidates.filter(row => row.riskEligible);
  assert.ok(!globalRiskRootMaxLots || eligible.length,
    "No globally enumerated root request has a feasible risk-constrained H2 continuation");
  if (globalRiskRootMaxLots && eligible.length) globallyCertifiedDecisions++;
  if (rootCoverage) rootCoverage.h2FeasibleLots = eligible.length;
  const riskConstrained = (eligible.length ? eligible : pairedCandidates).reduce((best, row) => eligible.length
    ? (row.robustImprovement > best.robustImprovement
      || row.robustImprovement === best.robustImprovement && Math.abs(row.lot) < Math.abs(best.lot) ? row : best)
    : (row.worstEquity > best.worstEquity || row.worstEquity === best.worstEquity
      && (row.unconstrainedValue ?? -Infinity) > (best.unconstrainedValue ?? -Infinity) ? row : best));
  const selected = !hold || riskEnabled && !hold.riskEligible ? riskConstrained
    : riskConstrained.robustImprovement > minImprovementBps / 10_000 ? riskConstrained
      : pairedCandidates.find(row => row.lot === 0)!;
  const selectedValue = Number.isFinite(selected.value) ? selected.value : selected.unconstrainedValue!;
  plannerSeconds += (performance.now() - begin) / 1000;
  return { quantity: selected.request, value: selectedValue, feasible: Number.isFinite(selectedValue), complete: true,
    scope: "compact-execution-h2-candidate-set", selected, unconstrained,
    improvementBps: hold ? ((unconstrained?.unconstrainedValue ?? selected.value) - (hold.unconstrainedValue ?? hold.value)) * 10_000 : null,
    riskImprovementBps: hold ? (selected.value - hold.value) * 10_000 : null,
    robustImprovementBps: hold ? selected.robustImprovement * 10_000 : null, minImprovementBps,
    candidates: pairedCandidates, minLots, safeMaxLots, sign, meanInterval, riskEnabled, riskFloor, peakProfit, liquidatableEquity,
    uncertaintyFilteredCandidates, uncertaintyConfidence, riskState,
    riskEligibleCandidates: eligible.length,
    riskConstraintChangedDecision: unconstrained ? selected.lot !== unconstrained.lot : null,
    rootCoverage };
};

fs.mkdirSync(output, { recursive: true });
const save = (name: string, value: unknown) => fs.writeFileSync(path.join(output, name), JSON.stringify(value,
  (_, v) => typeof v === "number" && !Number.isFinite(v) ? String(v) : v, 2));
save("config.json", { contract: "native-event-execution-h2-candidate-replay-v2", source, fullSource,
  compactLawHash: compiled.lawHash, fullLawHash: fullCompiled.lawHash, modelHash: compact.modelHash,
  minImprovementBps,
  initialRiskBps, protectedProfitFraction, globalRiskRootMaxLots,
  uncertaintyRiskScaling,
  softUncertaintySizing,
  ...(signSource ? { signSource, signConfigHash: hash(path.join(signSource, "config.json")),
    signHeadHash: hash(path.join(signSource, "head.json")), signBlend: blendUncertainty?.centralWeight ?? .5,
    signRole: "current-and-successor-direction" } : {}),
  ...(directionEnsembleSource ? { directionEnsembleSource,
    directionEnsembleConfigHash: hash(path.join(directionEnsembleSource, "config.json")),
    directionEnsembleSummaryHash: hash(path.join(directionEnsembleSource, "summary.json")),
    selectedDirection, eligibleDirectionSpecifications, directionEnsembleRole,
    uncertaintyRole: softUncertaintySizing ? "central-direction-ranking-and-eligible-specification-risk-sizing"
      : "nested-worst-expected-log-across-eligible-direction-specifications" } : {}),
  ...(ensembleSource ? { ensembleSource, ensembleName: ensemble.selected.name, ensembleScale } : {}),
  ...(blendUncertaintySource ? { blendUncertaintySource,
    blendUncertaintyHash: hash(path.join(blendUncertaintySource, "summary.json")),
    robustBlendInterval: robustWeights, robustConfidence: blendUncertainty.bootstrap.interval.confidence,
    uncertaintyRole: softUncertaintySizing ? "central-forecast-ranking-and-relative-interval-risk-sizing"
      : "nested-worst-expected-log-across-blend-interval-endpoints" } : {}),
  ...(meanUncertaintySource ? { meanUncertaintySource,
    meanUncertaintyHash: hash(path.join(meanUncertaintySource, "summary.json")),
    robustMeanCoverage: meanUncertainty.coverage,
    robustMeanRadiusBps: meanUncertainty.radiusBps,
    uncertaintyRole: softUncertaintySizing ? "central-forecast-ranking-and-relative-interval-risk-sizing"
      : "nested-worst-expected-log-across-conformal-mean-endpoints" } : {}),
  phase, start, end, window: config.window, fullWindow: start === phaseStart && end === phaseEnd, costs: policy.costs,
  replaySourceReferences: references,
  method: "At every observed event, score base-quantity requests with exact H2 Bellman continuation under the positive observed-path quadrature law. The default compact candidates include hold, compact H1, the fitted-law one-step optimum, flatten/reverse, safe full/half max-notional requests, leverage targets, and one to three minimum-size lots in both directions. When globalRiskRootMaxLots is positive, the exact one-step region solver first identifies every request whose worst fitted next-event liquidatable equity clears the current floor; the replay enumerates the complete admissible root lattice and aborts rather than approximate if it exceeds the declared computation gate. A supplied sign source or slow-plus-fast direction ensemble can reweight current and causal successor kernels. With directionEnsembleRole=risk-only, the frozen native law still ranks root and child actions while the ensemble contributes only uncertainty confidence to the capital floor. For the direction ensemble, every coefficient pair that passed the first-half calibration eligibility rule forms the forecast interval; no holdout outcome chooses that set. A blend interval, direction specification set, or conformal mean source can either define a nested maximin ambiguity set or, in soft-sizing mode, leave expected-log ranking at the frozen central forecast and scale only the admissible loss budget by |signal|/(|signal|+interval radius). Robust mode forbids opening, enlargement or reversal when the interval spans neutral. The root acts only on the declared positive paired improvement over holding. A supplied validation-selected magnitude ensemble combines soft direction confidence with the sign law's expected absolute return and maps the resulting mean into the execution law. The optional risk constraint requires both the next-event account and every outcome of the globally optimized child action to retain a floor based on initial risk capital and a protected fraction of net-liquidatable high-water profit. A first-event outcome that creates a new peak raises its own child floor before the second Bellman step. If no root candidate qualifies, the policy chooses the request with the safest next-event supported equity. Actual replay includes next-open acceptance, fees, borrowing, maintenance and terminal settlement. The uncompressed execution-H1 policy is the control. Complete risk-root enumeration is a global constrained H2 proof at each replayed state; compact mode is not. Fitted-support risk is not a guarantee against an unbounded market move." });
save("sources.json", Object.fromEntries([
  "scripts/replay-native-event-h2-candidates.ts", "scripts/research-event-policy.ts",
  "packages/bot-algo/src/event-execution-backup.ts", "packages/bot-algo/src/event-execution-box.ts",
  "packages/bot-algo/src/event-execution-one-step.ts", "packages/bot-algo/src/event-execution-path.ts",
  "packages/bot-algo/src/event-log-policy.ts", "packages/bot-algo/src/event-uncertainty.ts",
].map(file => [file, fs.readFileSync(path.join(root, file), "utf8")])));
candles = includeTradeFlow
  ? loadNativeEventCandles([{ start: sourceStart, end }], true)
  : loadEventCandles(sourceStart, end, 1000);
const started = performance.now();
const { trace: controlTrace, positions: controlPositions, ...control } = replayEventPolicy(candles, policy, start, end, 1,
  { trace: true, oneStepTerminal: "marked", executionOneStep: fullKernels });
save("control-trades.json", controlTrace); save("control-positions.json", controlPositions);
const { trace, positions, ...execution } = replayEventPolicy(candles, policy, start, end, 2,
  { trace: true, executionDecision: candidatePolicy,
    onDecision: row => save("progress.json", { time: row.time, end, order: row.order,
      plannerSeconds, evaluatedCandidates, globallyCoveredDecisions, globallyCoveredLots,
      maximumGlobalRootLots, globallyCertifiedDecisions }) });
save("execution-trades.json", trace); save("execution-positions.json", positions);
assert.deepEqual(trace.map(row => [row.time, row.leaf]), controlTrace.map(row => [row.time, row.leaf]));
const summary = { control, execution, decisions: trace.length, plannerSeconds, evaluatedCandidates,
  averageCandidates: evaluatedCandidates / trace.length, globallyCoveredDecisions, globallyCoveredLots,
  maximumGlobalRootLots, globallyCertifiedDecisions,
  globalRiskRootComplete: globalRiskRootMaxLots > 0 && globallyCoveredDecisions === trace.length
    && globallyCertifiedDecisions === trace.length,
  elapsedSeconds: (performance.now() - started) / 1000 };
save("summary.json", summary); save("progress.json", { phase: "finished", ...summary });
console.log(JSON.stringify({ phase, decisions: trace.length, plannerSeconds, evaluatedCandidates,
  controlReturnPct: control.returnPct, executionReturnPct: execution.returnPct,
  controlTrades: control.trades, executionTrades: execution.trades, fees: execution.fees,
  borrowing: execution.borrowing, maxDrawdownPct: execution.maxDrawdownPct }));

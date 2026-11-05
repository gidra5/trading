/** Audit the shared multi-horizon comparison and collect every Lean certificate. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const outputIndex = process.argv.indexOf("--output");
const output = path.resolve(root, outputIndex < 0 ? "data/backtests/oracle-shared-hurdle" : process.argv[outputIndex + 1]);
const json = async (file) => JSON.parse(await readFile(path.resolve(root, file), "utf8"));
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const fileSha = async (file) => sha(await readFile(path.resolve(root, file)));
const lines = (text) => text.split(/\r?\n/).filter((line) => line.trim());
const samePath = (a, b) => assert.equal(path.resolve(root, a), path.resolve(root, b));
const modes = ["none", "zero", "probabilities", "full"];
const names = ["separate-raw", "separate-online", "shared-raw", "shared-online"];
const bounds = [2, 2, Math.log(2), Math.log(2)];
const plan = await json(path.join(output, "replay-plan.json"));
const selectionFile = path.resolve(root, plan.runsRoot, "selection.json");
const frozen = await json(selectionFile);
const selectionSha256 = await fileSha(selectionFile);
assert.equal(frozen.contract, "shared-horizon-hurdle-finetune-v1");
assert.equal(frozen.allHeadSelectionsFrozen, true);
assert.equal(frozen.testLoaded, false);
assert.equal(frozen.testUsedForSelection, false);
assert.equal(frozen.expertBackbones, 3);
assert.deepEqual(frozen.plan.horizonsSeconds, [1, 900, 1800, 7200]);
assert.deepEqual(plan.variants.map((row) => row.name), names);
assert.equal(await fileSha(path.join(plan.runsRoot, "plan.json")), frozen.planSha256);
assert.equal(await fileSha(frozen.sourceSelectionFile), frozen.sourceSelectionSha256);
assert.equal(await fileSha(frozen.onlineSelectionFile), frozen.onlineSelectionSha256);
assert.equal(await fileSha(frozen.initializerSelectionFile), frozen.initializerSelectionSha256);
for (const stage of ["pretrained", "finetuned"]) {
  assert.equal(await fileSha(frozen[stage].checkpoint), frozen[stage].sha256);
  assert.deepEqual(frozen[stage].horizonsSeconds, [1, 900, 1800, 7200]);
}
const epochs = lines(await readFile(path.resolve(root, plan.runsRoot, "training.jsonl"), "utf8"))
  .map((line) => JSON.parse(line)).filter((row) => row.stage === "finetuned" && row.event === "epoch");
assert(epochs.length > 0);
for (const epoch of epochs) {
  assert.deepEqual(epoch.horizonsSeconds, [1, 900, 1800, 7200]);
  assert(Math.abs(epoch.validationScores.reduce((sum, value) => sum + value, 0) / 4 - epoch.meanValidationScore) < 1e-12);
}
const bestEpoch = epochs.reduce((best, row) => row.meanValidationScore < best.meanValidationScore ? row : best);
assert.equal(frozen.finetuned.epoch, bestEpoch.epoch);
assert.equal(frozen.finetuned.validationScore, bestEpoch.meanValidationScore);
assert.deepEqual(frozen.finetuned.validationScoresByHorizon, bestEpoch.validationScores);
const source = await json(frozen.sourceSelectionFile);
const calibration = await json(frozen.onlineSelectionFile);
assert.equal(calibration.contract, "causal-online-hurdle-calibration-v1");
assert.equal(calibration.allCalibrationSelectionsFrozen, true);
assert.equal(calibration.testLoaded, false);
assert.equal(calibration.testUsedForSelection, false);
assert.equal(calibration.sourceSelectionSha256, frozen.sourceSelectionSha256);
assert.equal(await fileSha(calibration.validationCandidatesFile), calibration.validationCandidatesSha256);
const candidates = await json(calibration.validationCandidatesFile);
for (const selection of calibration.horizons) {
  const rows = candidates.horizons.find((row) => row.horizonSeconds === selection.horizonSeconds).candidates;
  for (const mode of modes) {
    const choices = rows.filter((row) => row.config.mode === mode);
    const best = choices.reduce((best, row) => row.validationNll < best.validationNll ? row : best);
    assert.deepEqual(selection.modes[mode], { candidateId: best.candidateId, config: best.config, validationNll: best.validationNll });
  }
  const preferred = modes.reduce((best, mode) => selection.modes[mode].validationNll < selection.modes[best].validationNll ? mode : best);
  assert.equal(selection.validationPreferredMode, preferred);
}
const evaluationPlanFile = path.join(plan.runsRoot, "evaluation-plan.json");
const evaluationPlan = await json(evaluationPlanFile);
const result = await json(path.join(plan.runsRoot, "result.json"));
assert.equal(evaluationPlan.contract, "shared-horizon-hurdle-evaluation-v1");
assert.equal(evaluationPlan.selectionSha256, selectionSha256);
assert.equal(evaluationPlan.trainingPlanSha256, frozen.planSha256);
assert.equal(evaluationPlan.calibrationSelectionSha256, frozen.onlineSelectionSha256);
assert.equal(evaluationPlan.sourceSelectionSha256, frozen.sourceSelectionSha256);
assert.equal(evaluationPlan.replayPlanSha256, await fileSha(path.join(output, "replay-plan.json")));
assert.equal(evaluationPlan.trainingResultSha256, await fileSha(evaluationPlan.trainingResultFile));
assert.equal(result.evaluationPlanSha256, await fileSha(evaluationPlanFile));
assert.equal(result.selectionSha256, selectionSha256);
assert.equal(result.calibrationReselection, false);
assert.equal(result.testUsedForSelection, false);
for (const artifact of evaluationPlan.baselineArtifactMetadata) assert.equal(await fileSha(artifact.file), artifact.sha256);

const horizons = [], certificates = [];
let commonTapeSha256;
for (const horizon of plan.horizonsSeconds) {
  const count = plan.decisionCountsByHorizon[horizon];
  assert.equal(count * horizon, plan.comparisonObservedSeconds);
  const calibrationSelection = calibration.horizons.find((row) => row.horizonSeconds === horizon);
  const onlineConfig = calibrationSelection.modes[calibrationSelection.validationPreferredMode].config;
  assert.deepEqual(evaluationPlan.frozenCalibration[horizon], calibrationSelection.modes[calibrationSelection.validationPreferredMode]);
  const sourceRecord = source.horizons.find((row) => row.horizonSeconds === horizon).finetuned;
  assert.equal(await fileSha(sourceRecord.checkpoint), sourceRecord.sha256);
  const variants = [];
  for (const variant of plan.variants) {
    const isShared = variant.name.startsWith("shared-");
    const isOnline = variant.name.endsWith("online");
    const record = isShared ? frozen.finetuned : sourceRecord;
    const selectedConfig = isOnline ? onlineConfig : {
      horizon_seconds: horizon, mode: "none", half_life_seconds: 14400, ridge_precision: 8, cadence_seconds: 4,
    };
    const forecastFile = path.join(output, `${variant.name}-h${horizon}.jsonl`);
    const manifest = await json(forecastFile.replace(/\.jsonl$/, ".manifest.json"));
    assert.equal(manifest.contract, evaluationPlan.contract);
    assert.equal(manifest.variant, variant.name);
    assert.deepEqual(manifest.combined, record);
    assert.equal(manifest.checkpointSha256, record.sha256);
    assert.deepEqual(manifest.probabilityOrder, ["down", "zero", "up"]);
    assert.equal(manifest.neuralRefit, false);
    assert.equal(manifest.reselection, false);
    assert.equal(manifest.headSelectionsFrozenBeforeTest, true);
    assert.equal(manifest.calibrationSettingsFrozenBeforeTest, true);
    assert.equal(manifest.testUsedForSelection, false);
    assert.equal(manifest.onlineLabelUpdates, selectedConfig.mode !== "none");
    assert.equal(manifest.calibrationColdStart, true);
    assert.equal(manifest.calibrationShapeUpdates, false);
    assert.equal(manifest.sharedAcrossHorizons, isShared);
    assert.equal(manifest.baselineArtifactsReused, !isShared);
    samePath(manifest.selectionFile, selectionFile);
    assert.equal(manifest.selectionSha256, selectionSha256);
    samePath(manifest.sourceSelectionFile, frozen.sourceSelectionFile);
    assert.equal(manifest.sourceSelectionSha256, frozen.sourceSelectionSha256);
    samePath(manifest.calibrationSelectionFile, frozen.onlineSelectionFile);
    assert.equal(manifest.calibrationSelectionSha256, frozen.onlineSelectionSha256);
    assert.deepEqual(manifest.selectedConfig, selectedConfig);
    assert.equal(manifest.sourceCalibrationPreferredMode, calibrationSelection.validationPreferredMode);
    assert.equal(await fileSha(manifest.calibrationTraceFile), manifest.calibrationTraceSha256);
    const cache = manifest.canonicalCache;
    if (isShared) {
      assert.deepEqual(cache.horizonsSeconds, [1, 900, 1800, 7200]);
      assert.equal(cache.selectionSha256, selectionSha256);
      assert.equal(cache.finetuned.checkpointSha256, record.sha256);
      assert.equal(await fileSha(cache.finetuned.cacheFile), cache.finetuned.cacheSha256);
      assert.deepEqual(cache.finetuned.shape, [cache.canonicalUnionOrigins, 4, 6]);
      assert(cache.horizonValidPrefixCounts[horizon] <= cache.canonicalUnionOrigins);
    } else {
      assert.equal(await fileSha(cache.cacheFile), cache.cacheSha256);
      assert.equal(await fileSha(cache.marketFile), cache.marketSha256);
      assert.equal(cache.checkpointSha256, record.sha256);
      assert.equal(cache.planSha256, calibration.planSha256);
      assert.equal(cache.cadenceSeconds, 4);
    }
    assert.equal(cache.firstTimestamp, plan.firstTimestamp);
    assert.equal(await fileSha(manifest.calibratedStream.parametersFile), manifest.calibratedStream.parametersSha256);
    const forecastLines = lines(await readFile(forecastFile, "utf8"));
    const forecastSha256 = sha(forecastLines.join("\n") + "\n");
    samePath(manifest.forecastFile, forecastFile);
    assert.equal(manifest.forecastSha256, forecastSha256);
    assert.equal(manifest.split, "test");
    assert.equal(manifest.horizonSeconds, horizon);
    assert.equal(manifest.rows, count);
    assert.equal(forecastLines.length, count);
    assert.equal(manifest.firstTimestamp, plan.firstTimestamp);
    assert.equal(manifest.lastRealizedTimestamp, plan.terminalTimestamp);
    assert.equal(manifest.replayStateSamples.length, count);
    assert.equal(manifest.replayCanonicalIndices.length, count);
    const tape = [], expectedMeansBps = [];
    for (const [index, line] of forecastLines.entries()) {
      const row = JSON.parse(line);
      assert.equal(row.horizonSeconds, horizon);
      assert.equal(row.timestamp, plan.firstTimestamp + index * horizon * 1000);
      assert.equal(row.realizedNextTimestamp, row.timestamp + horizon * 1000);
      assert.equal(row.realizedPrices.length, horizon);
      assert.equal(row.realizedPrices.at(-1), row.realizedNextPrice);
      const state = manifest.replayStateSamples[index];
      assert.equal(state.timestamp, row.timestamp);
      const canonicalIndex = (row.timestamp - cache.firstTimestamp) / 4000;
      assert.equal(manifest.replayCanonicalIndices[index], canonicalIndex);
      const matured = Math.max(0, Math.min(canonicalIndex, Math.floor((row.timestamp - cache.firstTimestamp - horizon * 1000) / 4000) + 1));
      assert.equal(state.maturedObservations, matured);
      assert.equal(state.latestMaturityUsed, matured === 0 ? null : cache.firstTimestamp + (matured - 1) * 4000 + horizon * 1000);
      const enabled = { none: 0, zero: 1, probabilities: 2, full: 4 }[selectedConfig.mode];
      assert.equal(state.coefficients.length, 4);
      state.coefficients.forEach((value, component) => {
        assert(Number.isFinite(value) && Math.abs(value) <= bounds[component]);
        if (component >= enabled || index === 0) assert.equal(Math.abs(value), 0);
      });
      if (!index) tape.push(row.observedPrice);
      else assert.equal(row.observedPrice, tape.at(-1));
      for (const price of row.realizedPrices) {
        assert(Number.isFinite(price) && price > 0);
        tape.push(price);
      }
      const branches = row.tree.children;
      assert.equal(branches.length, 3);
      assert(branches.every((branch) => Number.isFinite(branch.price) && branch.price > 0 &&
        Number.isFinite(branch.probability) && branch.probability >= 0));
      assert(Math.abs(branches.reduce((sum, branch) => sum + branch.probability, 0) - 1) < 1e-12);
      expectedMeansBps.push(branches.reduce((sum, branch) => sum + branch.probability * (branch.price / row.observedPrice - 1), 0) * 1e4);
    }
    assert.equal(tape.length, plan.comparisonObservedSeconds + 1);
    const tapeSha256 = sha(JSON.stringify(tape));
    if (commonTapeSha256) assert.equal(tapeSha256, commonTapeSha256);
    commonTapeSha256 = tapeSha256;
    const replays = [];
    for (const feeScale of variant.feeScales) {
      const directory = path.join(output, `${variant.name}-h${horizon}-fees${feeScale}`);
      const report = await json(path.join(directory, "report.json"));
      samePath(report.input, forecastFile);
      assert.equal(report.contract, "oracle-long-cash-endpoint-mpc-v1");
      assert.equal(report.forecastSha256, forecastSha256);
      assert.equal(report.gamma, plan.riskAversion);
      assert.equal(report.feeScale, String(feeScale));
      assert.equal(report.firstTimestamp, plan.firstTimestamp);
      assert.equal(report.terminalTimestamp, plan.terminalTimestamp);
      assert.equal(report.horizonSeconds, horizon);
      assert.equal(report.decisionCount, count);
      assert.equal(report.observedSeconds, plan.comparisonObservedSeconds);
      const decisions = lines(await readFile(path.join(directory, "decisions.jsonl"), "utf8"));
      assert.equal(decisions.length, count);
      assert.equal(report.decisionCertificates, count);
      certificates.push(...decisions);
      replays.push(report);
    }
    variants.push({ name: variant.name, manifest, forecastSha256,
      unroundedForecastMeanBps: { min: Math.min(...expectedMeansBps), max: Math.max(...expectedMeansBps),
        average: expectedMeansBps.reduce((sum, value) => sum + value, 0) / count }, replays });
  }
  horizons.push({ horizonSeconds: horizon, calibrationSelection, variants });
}
assert.equal(certificates.length, 2184);
const certificateText = certificates.join("\n") + "\n";
await writeFile(path.join(output, "decisions.jsonl"), certificateText);
await writeFile(path.join(output, "comparison.json"), JSON.stringify({ plan, horizons,
  selectionSha256, sharedCheckpointSha256: frozen.finetuned.sha256, calibrationSelectionSha256: frozen.onlineSelectionSha256,
  commonTapeSha256, artifactChecksPassed: true, certificateRecords: certificates.length, certificatesSha256: sha(certificateText),
  checks: "One global multi-horizon checkpoint chosen by mean validation NLL; unchanged prior validation-only calibration settings; frozen source/checkpoint/cache/trace hashes; exact replay maturity counts, cold identity and coefficient bounds; identical complete one-second price tape; every decision certificate",
  scope: "Artifact and replay consistency. Independent Python audit checks neural horizon mapping, full calibration traces and metrics. Lean verifies supplied finite payoff comparisons; forecast accuracy remains empirical."
}, null, 2) + "\n");
console.log(JSON.stringify({ artifactChecksPassed: true, certificateRecords: certificates.length,
  sharedSelectedEpoch: frozen.finetuned.epoch, horizons: horizons.map(({ horizonSeconds, variants }) => ({ horizonSeconds,
    variants: variants.map(({ name, replays }) => ({ name, replays: replays.map((report) => ({
      feeScale: report.feeScale, returnPercent: report.returnPercent, markets: report.markets })) })) })) }, null, 2));

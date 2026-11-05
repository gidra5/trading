import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Assemble an explicitly planned horizon comparison; do not choose a model
// using test PnL. Replay directories are h{seconds}-fees{scale} beside the plan.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const index = args.indexOf("--plan");
const planPath = path.resolve(root, index < 0
  ? "data/backtests/oracle-horizons-5m15m/comparison-plan.json" : args[index + 1]);
const output = path.dirname(planPath);
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const sha256 = (bytes) => createHash("sha256").update(bytes).digest("hex");
const plan = await json(planPath);
const runs = path.resolve(root, plan.runsRoot);
const horizons = [];
const certificateSources = [];
let firstTimestamp;
for (const horizon of plan.horizonsSeconds) {
  const directory = path.join(runs, `h${horizon}`);
  const selection = await json(path.join(directory, "model-selection.json"));
  const audit = await json(path.join(directory, "validation-economic-gate.json"));
  assert.equal(selection.testUsedForSelection, false);
  assert.equal(audit.testLoadedByThisAudit, false);
  assert.equal(audit.changesModelSelection, false);
  assert.equal(audit.selectedCheckpoint, selection.selectedCheckpoint);
  assert.equal(selection.horizonSeconds, horizon);
  assert.equal(audit.horizonSeconds, horizon);
  assert.equal(audit.decisionStrideSeconds, horizon);
  assert.equal(audit.selectedMeanLossWeight, selection.selectedMeanLossWeight);
  assert(selection.candidates.length > 0);
  assert(selection.candidates.every((entry) => Number.isFinite(entry.validation.selectionScore)));
  const selectedCandidates = selection.candidates.filter((entry) =>
    entry.meanLossWeight === selection.selectedMeanLossWeight);
  assert.equal(selectedCandidates.length, 1);
  const selectedCandidate = selectedCandidates[0];
  assert.equal(selectedCandidate.validation.selectionScore,
    Math.min(...selection.candidates.map((entry) => entry.validation.selectionScore)));
  assert.equal(selectedCandidate.selectedModel, selection.selectedModel);
  assert.equal(path.resolve(root, selectedCandidate.selectedCheckpoint),
    path.resolve(root, selection.selectedCheckpoint));
  const candidate = audit.models[`lambda${selection.selectedMeanLossWeight}`];
  assert(candidate);
  assert.equal(path.resolve(root, candidate.checkpoint), path.resolve(root, selection.selectedCheckpoint));
  assert.deepEqual(candidate.validationScore, selectedCandidate.validation);
  const selectedResult = await json(path.resolve(root, selectedCandidate.result));
  assert.equal(selectedResult.selection.testUsedForSelection, false);
  assert.equal(selectedResult.selection.selectedModel, selection.selectedModel);
  assert.equal(path.resolve(root, selectedResult.selection.selectedCheckpoint),
    path.resolve(root, selection.selectedCheckpoint));
  const modelScores = Object.values(selectedResult.selection.validation).map((entry) => entry.selectionScore);
  assert(modelScores.length > 0 && modelScores.every(Number.isFinite));
  assert.equal(selectedResult.selection.validation[selection.selectedModel].selectionScore, Math.min(...modelScores));
  const forecastManifest = await json(path.join(output, `oracle-crra-horizon-h${horizon}.manifest.json`));
  assert.equal(forecastManifest.split, "test");
  assert.equal(forecastManifest.testUsedForSelection, false);
  assert.equal(forecastManifest.horizonSeconds, horizon);
  assert.equal(forecastManifest.decisionStrideSeconds, horizon);
  assert.equal(forecastManifest.horizon, 1);
  assert.equal(path.resolve(root, forecastManifest.checkpoint), path.resolve(root, selection.selectedCheckpoint));
  assert.equal(forecastManifest.checkpointSha256, selectedResult.checkpointSha256);
  assert.equal(sha256(await readFile(path.resolve(root, selection.selectedCheckpoint))), forecastManifest.checkpointSha256);
  const forecastPath = path.resolve(root, forecastManifest.forecastFile);
  assert.equal(forecastPath, path.join(output, `oracle-crra-horizon-h${horizon}.jsonl`));
  const forecastLines = (await readFile(forecastPath, "utf8")).split(/\r\n|\n|\r/).filter((line) => line.trim());
  assert.equal(forecastLines.length, forecastManifest.rows);
  const consumedRows = plan.decisionCountsByHorizon[String(horizon)];
  assert(Number.isSafeInteger(consumedRows) && consumedRows > 0 && consumedRows <= forecastLines.length);
  assert.equal(consumedRows * horizon, plan.comparisonObservedSeconds);
  // The replay skips blank lines and hashes each consumed original line with
  // one LF, including a final LF. An export can contain unused trailing rows.
  const consumedForecastSha256 = sha256(forecastLines.slice(0, consumedRows).map((line) => line + "\n").join(""));
  const firstForecast = JSON.parse(forecastLines[0]);
  const lastConsumedForecast = JSON.parse(forecastLines[consumedRows - 1]);
  assert.equal(firstForecast.timestamp, forecastManifest.firstTimestamp);
  assert.equal(lastConsumedForecast.realizedNextTimestamp,
    firstForecast.timestamp + plan.comparisonObservedSeconds * 1000);
  const replays = [];
  for (const feeScale of plan.feeScales) {
    const replayDirectory = path.join(output, `h${horizon}-fees${feeScale}`);
    const report = await json(path.join(replayDirectory, "report.json"));
    assert.equal(report.contract, "oracle-long-cash-endpoint-mpc-v1");
    assert.equal(report.gamma, plan.riskAversion);
    assert.equal(report.feeScale, String(feeScale));
    assert.equal(report.horizonSeconds, horizon);
    assert.equal(report.decisionCount, plan.decisionCountsByHorizon[String(horizon)]);
    assert.equal(report.observedSeconds, plan.comparisonObservedSeconds);
    assert.equal(path.resolve(root, report.input), forecastPath);
    assert.equal(report.forecastSha256, consumedForecastSha256);
    assert.equal(report.firstTimestamp, forecastManifest.firstTimestamp);
    firstTimestamp ??= report.firstTimestamp;
    assert.equal(report.firstTimestamp, firstTimestamp);
    assert.equal(report.terminalTimestamp, firstTimestamp + plan.comparisonObservedSeconds * 1000);
    if (replays.length) assert.equal(report.forecastSha256, replays[0].forecastSha256);
    replays.push(report);
    certificateSources.push(path.join(replayDirectory, "decisions.jsonl"));
  }
  horizons.push({ horizonSeconds: horizon, selectedModel: selection.selectedModel,
    selectedMeanLossWeight: selection.selectedMeanLossWeight, selectedCheckpoint: selection.selectedCheckpoint,
    validationDistribution: candidate.validationScore, validationEconomics: candidate.economics,
    economicSwitchingGatePassed: audit.selectedModelGatePassed,
    calibrationEvidence: audit.calibrationEvidence,
    testDistribution: selectedResult.test[selection.selectedModel].distribution,
    testOverlappingLabelEconomics: selectedResult.test[selection.selectedModel].economics,
    forecastManifest, replays });
}
const certificateLines = (await Promise.all(certificateSources.map((file) => readFile(file, "utf8"))))
  .flatMap((content) => content.trim().split(/\r?\n/).filter(Boolean));
await writeFile(path.join(output, "decisions.jsonl"), certificateLines.join("\n") + "\n");
const result = { plan, horizons, certificateSources, certificateRecords: certificateLines.length,
  limitations: [
    "Model selection uses validation; test PnL does not select a checkpoint or horizon.",
    "Returns are from a common held-out interval; they do not guarantee profitability in another period.",
    "Current outputs are synthetic Oracle long/cash endpoint holds, with every actual intermediate mark.",
    "Forecast endpoint laws do not describe intra-horizon liquidation or limit fills in leveraged accounts.",
    "Finite-tree/menu optimality does not certify continuous forecasts, calibration, or unrestricted trading.",
    "Full-close thresholds are diagnostics; actual replay wealth includes any residual after the valid capped close.",
  ] };
await writeFile(path.join(output, "comparison.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify({ output, certificateRecords: certificateLines.length,
  horizons: horizons.map((entry) => ({ horizonSeconds: entry.horizonSeconds, selectedModel: entry.selectedModel,
    gatePassed: entry.economicSwitchingGatePassed, replays: entry.replays.map((report) => ({
      feeScale: report.feeScale, returnPercent: report.returnPercent, markets: report.markets })) })) }, null, 2));

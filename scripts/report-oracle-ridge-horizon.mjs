import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "data/backtests/oracle-horizon-search");
const json = async (file) => JSON.parse(await readFile(path.resolve(root, file), "utf8"));
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const samePath = (a, b) => assert.equal(path.resolve(root, a), path.resolve(root, b));
const lines = (text) => text.split(/\r?\n/).filter((line) => line.trim());
const plan = await json(path.join(output, "replay-plan.json"));
const selection = await json(plan.selectedRidgeSource);
const ridgeRoot = path.dirname(path.resolve(root, plan.selectedRidgeSource));
const result = await json(path.join(ridgeRoot, "result.json"));
assert.deepEqual(result.selection, selection);
assert.equal(selection.selectedHorizonSeconds, plan.selectedHorizonSeconds);
assert.equal(selection.selectedCandidate, plan.selectedCandidate);
assert.equal(result.test.horizonSeconds, plan.selectedHorizonSeconds);
assert.equal(result.test.selectedCandidate, plan.selectedCandidate);
assert.equal(result.test.testUsedForSelection, false);

// Recompute the declared validation choice, including the no-signal controls.
for (const row of selection.horizons) {
  const selectedMse = row.validation[row.selectedCandidate].excessPriceMse;
  assert.equal(selectedMse, Math.min(...Object.values(row.validation).map((score) => score.excessPriceMse)));
}
const firstCrossing = selection.horizons.find((row) =>
  row.validation[row.selectedCandidate].nonoverlapping.chooseLongCount > 0);
assert.equal(firstCrossing.horizonSeconds, plan.selectedHorizonSeconds);

const certificates = [];
const runs = [];
let realizedTapeSha256;
for (const run of plan.runs) {
  const report = await json(path.join(run.output, "report.json"));
  const manifest = await json(run.forecasts.replace(/\.jsonl$/, ".manifest.json"));
  const forecastLines = lines(await readFile(path.resolve(root, run.forecasts), "utf8"));
  const forecastSha256 = sha(forecastLines.join("\n") + "\n");
  const checkpointSha256 = sha(await readFile(path.resolve(root, manifest.checkpoint)));
  samePath(report.input, run.forecasts);
  samePath(manifest.forecastFile, run.forecasts);
  assert.equal(report.forecastSha256, forecastSha256);
  if (manifest.forecastSha256) assert.equal(manifest.forecastSha256, forecastSha256);
  assert.equal(manifest.checkpointSha256, checkpointSha256);
  assert.equal(manifest.split, "test");
  assert.equal(manifest.testUsedForSelection, false);
  assert.equal(manifest.horizonSeconds, plan.selectedHorizonSeconds);
  assert.equal(manifest.decisionStrideSeconds, plan.selectedHorizonSeconds);
  assert.equal(manifest.rows, plan.count);
  assert.equal(forecastLines.length, plan.count);
  assert.equal(manifest.firstTimestamp, plan.firstTimestamp);
  assert.equal(manifest.lastRealizedTimestamp, plan.terminalTimestamp);
  assert.equal(report.gamma, plan.gamma);
  assert.equal(report.feeScale, String(run.feeScale));
  assert.equal(report.decisionCount, plan.count);
  assert.equal(report.observedSeconds, plan.observedSeconds);
  assert.equal(report.firstTimestamp, plan.firstTimestamp);
  assert.equal(report.terminalTimestamp, plan.terminalTimestamp);
  assert.equal(report.horizonSeconds, plan.selectedHorizonSeconds);
  if (run.model === "selectedRidge") {
    samePath(manifest.selectionFile, plan.selectedRidgeSource);
    samePath(manifest.checkpoint, path.join(ridgeRoot, `h${plan.selectedHorizonSeconds}.npz`));
    assert.equal(manifest.selectionSha256, sha(await readFile(path.resolve(root, plan.selectedRidgeSource))));
    assert.equal(manifest.selectedCandidate, plan.selectedCandidate);
    assert.equal(manifest.refit, false);
    assert.equal(manifest.reselection, false);
  } else {
    const neuralSearch = await json("data/training/runs/stochastic-crra-horizon-search-v1/search-results.json");
    const selected = neuralSearch.rows.find((row) => row.horizonSeconds === plan.selectedHorizonSeconds);
    samePath(manifest.checkpoint, selected.selectedCheckpoint);
  }
  const tape = forecastLines.map((line, index) => {
    const row = JSON.parse(line);
    assert.equal(row.timestamp, plan.firstTimestamp + index * plan.selectedHorizonSeconds * 1000);
    assert.equal(row.realizedNextTimestamp, row.timestamp + plan.selectedHorizonSeconds * 1000);
    assert.equal(row.realizedPrices.length, plan.selectedHorizonSeconds);
    assert.equal(row.realizedPrices.at(-1), row.realizedNextPrice);
    return [row.timestamp, row.observedPrice, row.realizedNextTimestamp, row.realizedPrices];
  });
  const tapeSha = sha(JSON.stringify(tape));
  if (realizedTapeSha256) assert.equal(tapeSha, realizedTapeSha256);
  realizedTapeSha256 = tapeSha;
  const decisions = lines(await readFile(path.resolve(root, run.output, "decisions.jsonl"), "utf8"));
  assert.equal(decisions.length, Math.min(plan.certificateCount, plan.count));
  assert.equal(report.decisionCertificates, decisions.length);
  certificates.push(...decisions);
  runs.push({ model: run.model, report, checkpoint: manifest.checkpoint, checkpointSha256,
    forecastSha256, manifestMeanConservationError: manifest.maximumPriceRatioMeanError });
}
const certificateText = certificates.join("\n") + "\n";
await writeFile(path.join(output, "decisions.jsonl"), certificateText);
const comparison = { plan, selectedValidation: firstCrossing.validation[plan.selectedCandidate],
  selectedTest: result.test.scores[plan.selectedCandidate], runs, realizedTapeSha256,
  decisionCertificates: certificates.length, certificatesSha256: sha(certificateText),
  artifactChecksPassed: true,
  checks: "Declared validation choices, actual model/selection/forecast hashes, same full realized tape, split/horizon/interval and complete certificate counts",
  scope: "Provenance checks do not establish model calibration; Lean certificate checking is a separate command" };
await writeFile(path.join(output, "comparison.json"), JSON.stringify(comparison, null, 2) + "\n");
console.log(JSON.stringify({ artifactChecksPassed: true, decisionCertificates: certificates.length,
  runs: runs.map(({ model, report }) => ({ model, feeScale: report.feeScale,
    returnPercent: report.returnPercent, markets: report.markets })) }, null, 2));

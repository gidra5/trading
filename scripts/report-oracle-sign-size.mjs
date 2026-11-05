import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "data/backtests/oracle-sign-size");
const json = async (file) => JSON.parse(await readFile(path.resolve(root, file), "utf8"));
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const samePath = (a, b) => assert.equal(path.resolve(root, a), path.resolve(root, b));
const lines = (text) => text.split(/\r?\n/).filter((line) => line.trim());
const plan = await json(path.join(output, "replay-plan.json"));
const frozen = await json(path.resolve(root, plan.runsRoot, "selection.json"));
assert.equal(frozen.allHeadSelectionsFrozen, true);
assert.equal(frozen.testLoaded, false);
assert.equal(frozen.testUsedForSelection, false);
assert.deepEqual(frozen.plan.horizonsSeconds, plan.horizonsSeconds);
const horizons = [];
const certificates = [];
let commonTapeSha256;

for (const horizon of plan.horizonsSeconds) {
  const count = plan.decisionCountsByHorizon[horizon];
  assert.equal(count * horizon, plan.comparisonObservedSeconds);
  const selectionFile = path.resolve(root, plan.runsRoot, `h${horizon}`, "selection.json");
  const selection = await json(selectionFile);
  assert.equal(selection.horizonSeconds, horizon);
  assert.equal(selection.testUsedForSelection, false);
  assert.equal(selection.testLoaded, false);
  assert.deepEqual(selection, frozen.horizons.find((row) => row.horizonSeconds === horizon));
  for (const head of ["sign", "up", "down"]) {
    const metrics = selection.headValidation[head];
    const key = head === "sign" ? "logLoss" : "mseBpsSquared";
    const expected = metrics.boostedValidation &&
      metrics.boostedValidation[key] < metrics.constantValidation[key] ? "boosted" : "constant";
    assert.equal(selection.selectedHeads[head], expected);
    assert.equal(metrics.selected, expected);
  }
  const test = await json(path.resolve(root, plan.runsRoot, `h${horizon}`, "test.json"));
  assert.equal(test.headSelectionsFrozenBeforeTest, true);
  assert.equal(test.testUsedForSelection, false);
  assert.deepEqual(test.selectedHeads, selection.selectedHeads);
  const selectionSha256 = sha(await readFile(selectionFile));
  const variants = [];
  let sharedCheckpointSha256;
  for (const variant of plan.variants) {
    const forecastFile = path.join(output, `${variant.name}-h${horizon}.jsonl`);
    const manifest = await json(forecastFile.replace(/\.jsonl$/, ".manifest.json"));
    assert.equal(manifest.variant, variant.name === "factorized" ? "full" : "signConstantConditionalSizes");
    assert.deepEqual(manifest.selectedHeads, selection.selectedHeads);
    assert.equal(manifest.refit, false);
    assert.equal(manifest.reselection, false);
    assert.equal(manifest.headSelectionsFrozenBeforeTest, true);
    samePath(manifest.allHorizonSelectionFile, path.resolve(root, plan.runsRoot, "selection.json"));
    assert.equal(manifest.allHorizonSelectionSha256,
      sha(await readFile(path.resolve(root, plan.runsRoot, "selection.json"))));
    const forecastLines = lines(await readFile(forecastFile, "utf8"));
    const forecastSha256 = sha(forecastLines.join("\n") + "\n");
    samePath(manifest.forecastFile, forecastFile);
    samePath(manifest.selectionFile, selectionFile);
    samePath(manifest.checkpoint, path.resolve(root, plan.runsRoot, `h${horizon}`, "model.joblib"));
    const checkpointSha256 = sha(await readFile(path.resolve(root, manifest.checkpoint)));
    samePath(selection.modelFile, manifest.checkpoint);
    assert.equal(selection.modelSha256, checkpointSha256);
    assert.equal(manifest.checkpointSha256, checkpointSha256);
    if (sharedCheckpointSha256) assert.equal(checkpointSha256, sharedCheckpointSha256);
    sharedCheckpointSha256 = checkpointSha256;
    assert.equal(manifest.selectionSha256, selectionSha256);
    assert.equal(manifest.forecastSha256, forecastSha256);
    assert.equal(manifest.split, "test");
    assert.equal(manifest.testUsedForSelection, false);
    assert.equal(manifest.horizonSeconds, horizon);
    assert.equal(manifest.decisionStrideSeconds, horizon);
    assert.equal(manifest.rows, count);
    assert.equal(forecastLines.length, count);
    assert.equal(manifest.firstTimestamp, plan.firstTimestamp);
    assert.equal(manifest.lastRealizedTimestamp, plan.terminalTimestamp);
    const tape = [];
    const expectedMeansBps = [];
    for (const [index, line] of forecastLines.entries()) {
      const row = JSON.parse(line);
      assert.equal(row.timestamp, plan.firstTimestamp + index * horizon * 1000);
      assert.equal(row.realizedNextTimestamp, row.timestamp + horizon * 1000);
      assert.equal(row.realizedPrices.length, horizon);
      assert.equal(row.realizedPrices.at(-1), row.realizedNextPrice);
      if (!index) tape.push(row.observedPrice);
      else assert.equal(row.observedPrice, tape.at(-1));
      for (const price of row.realizedPrices) {
        assert(Number.isFinite(price) && price > 0);
        tape.push(price);
      }
      assert.equal(row.tree.children.length, 3);
      const branches = row.tree.children;
      assert(branches.every((branch) => Number.isFinite(branch.price) && branch.price > 0 &&
        Number.isFinite(branch.probability) && branch.probability >= 0));
      assert(Math.abs(branches.reduce((sum, branch) => sum + branch.probability, 0) - 1) < 1e-12);
      expectedMeansBps.push(branches.reduce((sum, branch) =>
        sum + branch.probability * (branch.price / row.observedPrice - 1), 0) * 1e4);
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
    variants.push({ name: variant.name, manifest, forecastSha256, checkpointSha256,
      unroundedForecastMeanBps: { min: Math.min(...expectedMeansBps), max: Math.max(...expectedMeansBps),
        average: expectedMeansBps.reduce((sum, value) => sum + value, 0) / count }, replays });
  }
  horizons.push({ horizonSeconds: horizon, selection, test, variants });
}
const certificateText = certificates.join("\n") + "\n";
await writeFile(path.join(output, "decisions.jsonl"), certificateText);
await writeFile(path.join(output, "comparison.json"), JSON.stringify({ plan, horizons,
  commonTapeSha256, artifactChecksPassed: true, certificateRecords: certificates.length,
  certificatesSha256: sha(certificateText),
  checks: "Actual checkpoint/selection/forecast hashes, same complete one-second price tape across all horizons and ablations, physical scenario support, probabilities, replay intervals and every decision certificate",
  scope: "Provenance and replay consistency; Lean payoff comparisons and forecast calibration are separate checks"
}, null, 2) + "\n");
console.log(JSON.stringify({ artifactChecksPassed: true, certificateRecords: certificates.length,
  horizons: horizons.map(({ horizonSeconds, variants }) => ({ horizonSeconds,
    variants: variants.map(({ name, replays }) => ({ name, replays: replays.map((report) => ({
      feeScale: report.feeScale, returnPercent: report.returnPercent, markets: report.markets })) })) })) }, null, 2));

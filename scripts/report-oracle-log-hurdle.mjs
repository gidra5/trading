/** Check frozen log-law replay artifacts and collect every Lean certificate. */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const oi = process.argv.indexOf("--output");
const output = path.resolve(root, oi < 0 ? "data/backtests/oracle-log-hurdle" : process.argv[oi + 1]);
const json = async (file) => JSON.parse(await readFile(path.resolve(root, file), "utf8"));
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");
const fileSha = async (file) => sha(await readFile(path.resolve(root, file)));
const lines = (text) => text.split(/\r?\n/).filter((line) => line.trim());
const samePath = (a, b) => assert.equal(path.resolve(root, a), path.resolve(root, b));
const plan = await json(path.join(output, "replay-plan.json"));
const selectionFile = path.resolve(root, plan.runsRoot, "selection.json");
const selected = await json(selectionFile);
const selectionSha = await fileSha(selectionFile);
const perHorizon = selected.contract === "per-horizon-log-hurdle-training-v1";
assert.equal(plan.contract, perHorizon ? "oracle-per-horizon-log-comparison-v1" : "oracle-log-hurdle-comparison-v1");
assert.equal(selected.contract, perHorizon ? "per-horizon-log-hurdle-training-v1" : "shared-log-hurdle-training-v1");
const checkpointFor = (h, stage) => perHorizon ? selected.horizons.find((row) => row.horizonSeconds === h)[stage] : selected[stage];
assert.equal(selected.allHeadSelectionsFrozen, true);
assert.equal(selected.testLoaded, false);
assert.equal(selected.testUsedForSelection, false);
assert.equal(selected.expertBackbones, perHorizon ? 12 : 3);
assert.equal(plan.riskAversion, 0);
assert.deepEqual(plan.horizonsSeconds, [900, 1800, 7200]);
assert.equal(await fileSha(path.join(plan.runsRoot, "plan.json")), selected.planSha256);
assert.equal(await fileSha(selected.sourceSelectionFile), selected.sourceSelectionSha256);
assert.equal(await fileSha(selected.onlineSelectionFile), selected.onlineSelectionSha256);
for (const h of perHorizon ? [1, 900, 1800, 7200] : [1]) {
  for (const stage of ["pretrained", "finetuned"]) {
    const record = checkpointFor(h, stage);
    assert.equal(await fileSha(record.checkpoint), record.sha256);
  }
}
const calibration = await json(selected.onlineSelectionFile);
const evaluation = await json(path.join(output, "evaluation.json"));
assert.equal(evaluation.selectionSha256, selectionSha);
assert.equal(evaluation.newHyperparameterSelection, false);
assert.equal(evaluation.testUsedForSelection, false);
assert.equal(await fileSha(evaluation.evaluationPlanFile), evaluation.evaluationPlanSha256);
for (const [file, digest] of Object.entries(evaluation.artifactHashes)) assert.equal(await fileSha(file), digest);

let commonTapeSha256;
const horizons = [], certificates = [];
const bounds = [2, 2, Math.log(2), Math.log(2)];
for (const h of plan.horizonsSeconds) {
  const count = plan.decisionCountsByHorizon[h];
  assert.equal(count * h, plan.comparisonObservedSeconds);
  const preferred = calibration.horizons.find((entry) => entry.horizonSeconds === h);
  const config = preferred.modes[preferred.validationPreferredMode].config;
  const variants = [];
  for (const variant of plan.variants) {
    const stage = variant.name.startsWith("pretrained-") ? "pretrained" : "finetuned";
    const calibrated = variant.name.endsWith("-calibrated");
    const forecastFile = path.join(output, `${variant.name}-h${h}.jsonl`);
    const manifest = await json(forecastFile.replace(/\.jsonl$/, ".manifest.json"));
    assert.equal(manifest.contract, "matched-log-hurdle-evaluation-v1");
    assert.equal(manifest.variant, variant.name);
    assert.deepEqual(manifest.combined, checkpointFor(h, stage));
    assert.equal(manifest.checkpointSha256, checkpointFor(h, stage).sha256);
    assert.equal(manifest.selectionSha256, selectionSha);
    samePath(manifest.selectionFile, selectionFile);
    assert.equal(manifest.testUsedForSelection, false);
    assert.equal(manifest.refit, false);
    assert.equal(manifest.reselection, false);
    assert.equal(manifest.headSelectionsFrozenBeforeTest, true);
    assert.equal(manifest.calibrationSettingsFrozenBeforeTest, true);
    assert.equal(manifest.onlineLabelUpdates, calibrated);
    assert.equal(manifest.calibrationShapeUpdates, false);
    if (calibrated) {
      assert.deepEqual(manifest.frozenCalibration.config, config);
      assert.equal(manifest.calibrationColdStart, true);
      assert.equal(manifest.replayStateSamples.length, count);
      assert.equal(await fileSha(manifest.savedStream.traceFile), manifest.savedStream.traceSha256);
    } else assert.deepEqual(manifest.replayStateSamples, []);
    assert.equal(await fileSha(manifest.savedStream.parametersFile), manifest.savedStream.parametersSha256);
    const forecastText = await readFile(forecastFile, "utf8");
    const forecastLines = lines(forecastText), forecastSha = sha(forecastLines.join("\n") + "\n");
    assert.equal(manifest.forecastSha256, forecastSha);
    assert.equal(manifest.rows, count);
    assert.equal(forecastLines.length, count);
    assert.equal(manifest.firstTimestamp, plan.firstTimestamp);
    assert.equal(manifest.lastRealizedTimestamp, plan.terminalTimestamp);
    assert.deepEqual(manifest.probabilityOrder, ["down", "zero", "up"]);
    const tape = [], means = [];
    for (const [i, line] of forecastLines.entries()) {
      const row = JSON.parse(line);
      assert.equal(row.horizonSeconds, h);
      assert.equal(row.timestamp, plan.firstTimestamp + i * h * 1000);
      assert.equal(row.realizedNextTimestamp, row.timestamp + h * 1000);
      assert.equal(row.realizedPrices.length, h);
      assert.equal(row.realizedNextPrice, row.realizedPrices.at(-1));
      assert.equal(manifest.replayCanonicalIndices[i], i * h / 4);
      if (calibrated) {
        const state = manifest.replayStateSamples[i];
        const matured = Math.max(0, Math.floor((i * h - h) / 4) + 1);
        assert.equal(state.timestamp, row.timestamp);
        assert.equal(state.maturedObservations, matured);
        assert.equal(state.latestMaturityUsed, matured ? plan.firstTimestamp + (matured - 1) * 4000 + h * 1000 : null);
        const enabled = {none: 0, zero: 1, probabilities: 2, full: 4}[config.mode];
        state.coefficients.forEach((v, k) => {
          assert(Number.isFinite(v) && Math.abs(v) <= bounds[k]);
          if (k >= enabled || i === 0) assert.equal(Math.abs(v), 0);
        });
      }
      if (i === 0) tape.push(row.observedPrice);
      else assert.equal(row.observedPrice, tape.at(-1));
      for (const price of row.realizedPrices) {
        assert(Number.isFinite(price) && price > 0);
        tape.push(price);
      }
      const children = row.tree.children;
      assert.equal(children.length, 3);
      assert(children.every((v) => Number.isFinite(v.price) && v.price > 0 && Number.isFinite(v.probability) && v.probability >= 0));
      assert(Math.abs(children.reduce((s, v) => s + v.probability, 0) - 1) < 1e-12);
      assert(children[0].price <= row.observedPrice && children[1].price === row.observedPrice && children[2].price >= row.observedPrice);
      means.push(children.reduce((s, v) => s + v.probability * (v.price / row.observedPrice - 1), 0) * 1e4);
    }
    assert.equal(tape.length, plan.comparisonObservedSeconds + 1);
    const tapeSha = sha(JSON.stringify(tape));
    if (commonTapeSha256) assert.equal(tapeSha, commonTapeSha256);
    commonTapeSha256 = tapeSha;
    const replays = [];
    for (const feeScale of variant.feeScales) {
      const directory = path.join(output, `${variant.name}-h${h}-fees${feeScale}`);
      const report = await json(path.join(directory, "report.json"));
      samePath(report.input, forecastFile);
      assert.equal(report.contract, "oracle-long-cash-endpoint-mpc-v1");
      assert.equal(report.forecastSha256, forecastSha);
      assert.equal(report.gamma, 0);
      assert.equal(report.feeScale, String(feeScale));
      assert.equal(report.firstTimestamp, plan.firstTimestamp);
      assert.equal(report.terminalTimestamp, plan.terminalTimestamp);
      assert.equal(report.horizonSeconds, h);
      assert.equal(report.decisionCount, count);
      assert.equal(report.observedSeconds, plan.comparisonObservedSeconds);
      const records = lines(await readFile(path.join(directory, "decisions.jsonl"), "utf8"));
      assert.equal(records.length, count);
      assert.equal(report.decisionCertificates, count);
      certificates.push(...records);
      replays.push(report);
    }
    variants.push({name: variant.name, forecastSha256: forecastSha, manifest,
      meanForecastBps: means.reduce((s, v) => s + v, 0) / count, replays});
  }
  horizons.push({horizonSeconds: h, variants});
}
assert.equal(certificates.length, 2184);
const certificateText = certificates.join("\n") + "\n";
await writeFile(path.join(output, "decisions.jsonl"), certificateText);
await writeFile(path.join(output, "comparison.json"), JSON.stringify({plan, horizons,
  selectionSha256: selectionSha, selectedCheckpointHashes: Object.fromEntries([1, 900, 1800, 7200].map((h) => [h, checkpointFor(h, "finetuned").sha256])), commonTapeSha256,
  artifactChecksPassed: true, certificateRecords: certificates.length, certificatesSha256: sha(certificateText),
  scope: "Frozen model and calibration artifacts, identical one-second tape and complete certificate coverage. Lean verifies supplied finite-menu decisions, not forecast accuracy or unrestricted stochastic-policy optimality."
}, null, 2) + "\n");
console.log(JSON.stringify({artifactChecksPassed: true, certificateRecords: certificates.length}));

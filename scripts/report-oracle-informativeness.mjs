import assert from "node:assert/strict";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

// Consolidate completed, prespecified comparisons. This never selects a model
// from test returns or changes a checkpoint after an economic gate fails.
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const backtests = path.join(root, "data/backtests/stochastic-crra");
const runs = path.join(root, "data/training/runs/stochastic-crra-direct-horizons-gaussian-v2");
const output = path.join(backtests, "informativeness-comparison");
const json = async (file) => JSON.parse(await readFile(file, "utf8"));
const certificateSources = [];
async function replay(name) {
  const directory = path.join(backtests, name);
  const report = await json(path.join(directory, "report.json"));
  assert.equal(report.gamma, 0, `${name}: expected gamma zero`);
  certificateSources.push(path.join(directory, "decisions.jsonl"));
  return { name, ...report };
}
const fees = await Promise.all(["0", "001", "01", "1"].map((fee) => replay(`ablation-h1-fees${fee}`)));
for (const report of fees) {
  assert.equal(report.rows, 600);
  assert.equal(report.horizon, 1);
  assert.equal(report.evaluatedForecastPrefixSha256, fees[0].evaluatedForecastPrefixSha256,
    "Fee comparison must use identical forecast rows");
  assert.deepEqual(report.menu, fees[0].menu);
}
const twoStep = await Promise.all(["0", "1"].map((fee) => replay(`ablation-h2-fees${fee}`)));
assert.equal(twoStep[0].evaluatedForecastPrefixSha256, twoStep[1].evaluatedForecastPrefixSha256);
const horizons = [];
for (const horizon of [1, 30, 120]) {
  const [selection, gate, free, paid, fullPaid] = await Promise.all([
    json(path.join(runs, `h${horizon}/model-selection.json`)),
    json(path.join(runs, `h${horizon}/validation-economic-gate.json`)),
    replay(`endpoint-h${horizon}-fees0`), replay(`endpoint-h${horizon}-fees1`),
    replay(`endpoint-h${horizon}-full-fees1`),
  ]);
  assert.equal(gate.split, "validation");
  assert.equal(gate.testLoadedByThisAudit, false);
  assert.equal(gate.changesModelSelection, false);
  assert.equal(selection.selectedCheckpoint, gate.selectedCheckpoint);
  assert.equal(free.forecastSha256, paid.forecastSha256);
  for (const report of [free, paid]) {
    assert.equal(report.observedSeconds, 3600);
    assert.equal(report.horizonSeconds, horizon);
    assert.equal(report.firstTimestamp, fees[0].firstTimestamp);
  }
  const selected = gate.models[`lambda${selection.selectedMeanLossWeight}`];
  horizons.push({ horizonSeconds: horizon, selectedCheckpoint: selection.selectedCheckpoint,
    selectedMeanLossWeight: selection.selectedMeanLossWeight,
    validation: selected.validationScore, validationEconomics: selected.economics,
    economicGatePassed: gate.selectedModelGatePassed, calibrationEvidence: gate.calibrationEvidence,
    firstHour: { zeroFees: free, paid }, fullHeldOutPaid: fullPaid });
}
const certificates = (await Promise.all(certificateSources.sort().map((file) => readFile(file, "utf8"))))
  .flatMap((content) => content.trim().split(/\r?\n/).filter(Boolean));
await mkdir(output, { recursive: true });
await writeFile(path.join(output, "decisions.jsonl"), certificates.join("\n") + "\n");
const result = { contract: "oracle-informativeness-comparison-v1", gamma: 0,
  feeAblationFirstTenMinutes: fees, twoStepFirstMinute: twoStep, directHorizons: horizons,
  certificateSources, decisionCertificateRecords: certificates.length,
  limitations: [
    "Synthetic Oracle fees and unit funding on a BTCUSDT spot tape; no USD-M funding or live fills.",
    "Fee ablation retains the original finite trees and leveraged/limit menu.",
    "Horizon comparison uses the same Gaussian-CDF model family and long/cash endpoint menu.",
    "Endpoint holds preserve every actual mark, but allow no intermediate decisions or pending limits.",
    "Certificates prove finite quantized scenario/menu comparisons, not continuous-law or true-market optimality.",
    "Economic validation gates test predicted strict switching; they do not certify calibration or a uniform error bound.",
    "Reported terminal wealth includes any residual position after the valid capped close.",
    "Test returns were not used for model selection; short fee-free profits do not establish paid profitability.",
  ] };
await writeFile(path.join(output, "report.json"), JSON.stringify(result, null, 2) + "\n");
console.log(JSON.stringify({ output, records: certificates.length,
  gates: horizons.map(({ horizonSeconds, economicGatePassed }) => ({ horizonSeconds, economicGatePassed })) }, null, 2));

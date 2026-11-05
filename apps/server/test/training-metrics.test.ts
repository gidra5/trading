import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { test } from "node:test";
import { TrainingMetricsReader } from "../src/training-metrics.js";

test("live log cursors defer incomplete records and comparison uses the trainer's selection scope", async (t) => {
  const root = await fs.mkdtemp(path.join(os.tmpdir(), "trading-metrics-"));
  t.after(() => fs.rm(root, { recursive: true, force: true }));
  const runDir = "data/training/runs/fixture";
  const state = path.join(root, runDir, "state");
  const logs = path.join(root, runDir, "logs");
  await fs.mkdir(state, { recursive: true });
  await fs.mkdir(logs);
  await fs.writeFile(path.join(state, "plan.json"), JSON.stringify({ plan: {
    id: "fixture", label: "Fixture", runDir, training: { epochs: 512 },
  } }));
  await fs.writeFile(path.join(state, "status.json"), JSON.stringify({ stage: "training" }));
  const start = { event: "training-start", startEpoch: 0, parameters: 123 };
  const epoch = (index: number, featureMse: number, returnMse: number) => ({
    event: "minute-return-epoch", epoch: index,
    evaluationScope: "next-completed-1s-signed-log-return",
    checkpointSelectionScope: "complete-next-feature-state",
    featureState: { validation: { normalizedMse: featureMse, correlation: 1 - featureMse } },
    train: { normalizedMse: returnMse },
    validation: { normalizedMse: returnMse, correlation: 0.2 },
    validationDistribution: { perLeadExpectation: [{ correlation: 0.2 }, { correlation: 0.1 }] },
  });
  const logFile = path.join(logs, "training.jsonl");
  const initial = JSON.stringify(start) + "\n" + JSON.stringify(epoch(0, 0.4, 0.9)) + "\n";
  const pending = JSON.stringify(epoch(1, 0.3, 1.0));
  await fs.writeFile(logFile, initial + pending.slice(0, 15));
  const reader = new TrainingMetricsReader(root);
  const first = await reader.read(0, "fixture");
  assert.equal(first.events.length, 2);
  assert.equal(first.events[1]?.event, "epoch");
  assert.equal(first.cursor, Buffer.byteLength(initial));
  await fs.appendFile(logFile, pending.slice(15) + "\n");
  const second = await reader.read(first.cursor, "fixture");
  assert.equal(second.events.length, 1);
  assert.equal(second.events[0]?.epoch, 1);
  const comparison = (await reader.compare(["fixture"])).runs[0]!;
  assert.equal(comparison.parameterCount, 123);
  assert.equal(comparison.evaluationHorizonSteps, 1);
  assert.equal(comparison.fit[1]?.validationNormalizedMse, 1.0);
  // Choose by feature MSE, but display return-only MSE from that epoch.
  assert.equal(comparison.checkpointSelections["validation-mse"]?.epoch, 1);
  assert.deepEqual(comparison.checkpointSelections["validation-mse"]?.validation, {
    normalizedMse: 1.0, correlation: 0.2,
  });
  assert.equal(comparison.checkpointSelections["validation-nll"], undefined);
  assert.deepEqual(comparison.checkpointSelections["validation-mse"]?.distribution, {
    train: undefined,
    validation: { perLeadExpectation: [{ correlation: 0.2 }, { correlation: 0.1 }] },
    test: undefined,
  });

  const final = { epoch: 1, validation: { normalizedMse: 0.8, correlation: 0.25 } };
  await fs.writeFile(path.join(state, "stopped-evaluation.json"), JSON.stringify({
    policies: { "best-validation-mse": final, last: final },
  }));
  const evaluated = (await reader.compare(["fixture"])).runs[0]!;
  assert.deepEqual(evaluated.checkpointSelections["validation-mse"], final);
  assert.deepEqual(evaluated.checkpointSelections["stopped-last"], final);
});

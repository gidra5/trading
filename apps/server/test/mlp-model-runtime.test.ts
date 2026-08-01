import assert from "node:assert/strict";
import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  MLP_FEATURE_SCHEMA_VERSION,
  MLP_INPUT_FEATURE_COUNT,
  MLP_OUTPUT_ACTION_COUNT,
} from "@trading/bot-algo";
import { mlpInferenceTimes, mlpModelSummaries } from "../src/mlp-model-runtime.js";

test("model-target alignment evaluates each oracle time with inputs one policy delay later", () => {
  const oracleTimes = [1_000, 2_000, 3_000];
  assert.deepEqual(mlpInferenceTimes(oracleTimes, 60_000, true), [61_000, 62_000, 63_000]);
  assert.strictEqual(mlpInferenceTimes(oracleTimes, 60_000, false), oracleTimes);
  assert.strictEqual(mlpInferenceTimes(oracleTimes, 0, true), oracleTimes);
});

test("joint and dynamic-study artifacts remain recursively discoverable by the UI catalog", async (context) => {
  const dataDir = await fs.mkdtemp(path.join(os.tmpdir(), "trading-mlp-models-"));
  context.after(() => fs.rm(dataDir, { recursive: true, force: true }));
  const artifact = path.join(
    dataDir,
    "training",
    "runs",
    "studies",
    "joint",
    "study",
    "delay-60s",
    "factorial-00-e6-p2-fingerprint",
  );
  await fs.mkdir(artifact, { recursive: true });
  await fs.writeFile(path.join(artifact, "model.onnx"), "fixture");
  await fs.writeFile(path.join(artifact, "manifest.json"), JSON.stringify({
    id: "study-delay-60s-factorial-00",
    label: "Study delay 1m · weights factorial-00",
    createdAt: "2026-07-22T00:00:00.000Z",
    featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
    inputFeatureCount: MLP_INPUT_FEATURE_COUNT,
    outputRepresentation: "base-action-logits",
    outputActionCount: MLP_OUTPUT_ACTION_COUNT,
    actionGrid: Array.from(
      { length: MLP_OUTPUT_ACTION_COUNT },
      (_, index) => -250 + index * 500 / (MLP_OUTPUT_ACTION_COUNT - 1),
    ),
    hiddenLayerCount: 16,
    hiddenWidth: 1_024,
    predictionDelayMs: 60_000,
    modelFile: "model.onnx",
  }));
  const dynamicArtifact = path.join(
    dataDir,
    "training",
    "runs",
    "studies",
    "dynamic",
    "study",
    "branches",
    "selected-curriculum",
  );
  await fs.mkdir(dynamicArtifact, { recursive: true });
  await fs.writeFile(path.join(dynamicArtifact, "model.onnx"), "fixture");
  await fs.writeFile(path.join(dynamicArtifact, "manifest.json"), JSON.stringify({
    id: "dynamic-selected-curriculum",
    label: "Dynamic selected curriculum",
    createdAt: "2026-07-23T00:00:00.000Z",
    featureSchemaVersion: MLP_FEATURE_SCHEMA_VERSION,
    inputFeatureCount: MLP_INPUT_FEATURE_COUNT,
    outputRepresentation: "base-action-logits",
    outputActionCount: MLP_OUTPUT_ACTION_COUNT,
    actionGrid: Array.from(
      { length: MLP_OUTPUT_ACTION_COUNT },
      (_, index) => -250 + index * 500 / (MLP_OUTPUT_ACTION_COUNT - 1),
    ),
    hiddenLayerCount: 16,
    hiddenWidth: 1_024,
    predictionDelayMs: 0,
    modelFile: "model.onnx",
    training: {
      trainExamples: 1,
      validationExamples: 1,
      testExamples: 1,
      bestEpoch: 0,
      bestValidationLoss: 0.5,
      testLoss: 0.6,
      seed: 1,
      device: "cuda",
      curriculum: {
        version: 1,
        stage: 6,
        delayMs: 0,
        parentKey: "stage-5-parent",
        weightProfile: "factorial-03",
        lineage: [],
      },
    },
  }));

  assert.deepEqual(mlpModelSummaries(dataDir).map((model) => model.id), [
    "dynamic-selected-curriculum",
    "study-delay-60s-factorial-00",
  ]);
  assert.equal(
    mlpModelSummaries(dataDir)[0]?.training?.curriculum?.parentKey,
    "stage-5-parent",
  );
});

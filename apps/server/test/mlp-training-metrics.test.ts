import assert from "node:assert/strict";
import { appendFile, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { MlpTrainingMetricsReader } from "../src/mlp-training-metrics.js";

test("MLP training metrics stream only complete new metric events", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "mlp-training-metrics-"));
  const planFile = path.join(root, "ml", "training-plan.json");
  const runDir = path.join(root, "data", "runs", "test-model");
  const datasetDir = path.join(root, "data", "datasets", "test-model");
  await Promise.all([
    mkdir(path.dirname(planFile), { recursive: true }),
    mkdir(runDir, { recursive: true }),
    mkdir(datasetDir, { recursive: true }),
  ]);
  await writeFile(planFile, JSON.stringify({
    id: "test-model",
    label: "Test model",
    runDir: "data/runs/test-model",
    datasetDir: "data/datasets/test-model",
    training: { epochs: 12, lossWeights: { crossEntropy: 1 } },
  }));
  await Promise.all([
    writeFile(path.join(runDir, "status.json"), JSON.stringify({
      pid: process.pid,
      stage: "dataset-refinement",
      updatedAt: new Date().toISOString(),
    })),
    writeFile(path.join(datasetDir, "progress.json"), JSON.stringify({
      shards: [{ refinementPass: 1 }, { refinementPass: 0 }],
    })),
    writeFile(path.join(datasetDir, "teacher-refinement-queue.json"), JSON.stringify({
      cases: [{}, {}],
    })),
    writeFile(path.join(runDir, "training.log"), [
      "human-readable line",
      JSON.stringify({ event: "dataset-progress", day: 1, examplesPerSecond: 120 }),
      JSON.stringify({ event: "unrelated-event", value: 4 }),
      "",
    ].join("\n")),
  ]);

  try {
    const reader = new MlpTrainingMetricsReader(planFile);
    const first = await reader.read(0);
    assert.equal(first.running, true);
    assert.equal(first.plan.epochs, 12);
    assert.equal(first.progress.refinedShards, 1);
    assert.equal(first.progress.totalShards, 2);
    assert.equal(first.progress.remainingTeacherFits, 2);
    assert.deepEqual(first.events.map((event) => event.event), ["dataset-progress"]);

    await appendFile(path.join(runDir, "training.log"), [
      JSON.stringify({ event: "train-step", globalStep: 25, latest: { loss: 1.2 } }),
      "{\"event\":\"epoch\"",
    ].join("\n"));
    const second = await reader.read(first.cursor);
    assert.equal(second.reset, false);
    assert.deepEqual(second.events.map((event) => event.event), ["train-step"]);
    assert.ok(second.cursor > first.cursor);

    const reset = await reader.read(Number.MAX_SAFE_INTEGER);
    assert.equal(reset.reset, true);
    assert.ok(reset.events.some((event) => event.event === "dataset-progress"));

    const nextRunDir = path.join(root, "data", "runs", "next-model");
    await mkdir(nextRunDir, { recursive: true });
    await writeFile(planFile, JSON.stringify({
      id: "next-model",
      label: "Next model",
      runDir: "data/runs/next-model",
      datasetDir: "data/datasets/test-model",
      training: { epochs: 15, lossWeights: { stateMutualInformation: 0.1 } },
    }));
    await writeFile(path.join(nextRunDir, "training.log"), [
      JSON.stringify({ event: "training-start", epochs: 15 }),
      "",
    ].join("\n"));
    const switched = await reader.read(second.cursor);
    assert.equal(switched.plan.id, "next-model");
    assert.equal(switched.plan.epochs, 15);
    assert.equal(switched.reset, true);
    assert.deepEqual(switched.events.map((event) => event.event), ["training-start"]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

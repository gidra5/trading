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
    samplingIntervalMs: 1_000,
    training: {
      epochs: 12,
      lossWeights: { crossEntropy: 1 },
      timeWeighting: {
        mode: "distanceImbalance",
        distanceEpsilon: 1e-6,
        minimumWeight: 1e-6,
        stateAggregation: "globalDistanceRatio",
        minimumAdviceMagnitude: 0.25,
        memoryHalfLifeSteps: 15,
        growthPerPriorAdvice: 0.25,
        maximumMultiplier: 4,
        resetAfterGapSteps: 60,
      },
    },
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
    assert.equal(first.selectedRunKey, "ml/training-plan.json");
    assert.deepEqual(first.runs.map((run) => run.key), ["ml/training-plan.json"]);
    assert.equal(first.running, true);
    assert.equal(first.plan.epochs, 12);
    assert.equal(first.plan.samplingIntervalMs, 1_000);
    assert.deepEqual(first.plan.timeWeighting, {
      mode: "distanceImbalance",
      distanceEpsilon: 1e-6,
      minimumWeight: 1e-6,
      stateAggregation: "globalDistanceRatio",
      minimumAdviceMagnitude: 0.25,
      memoryHalfLifeSteps: 15,
      growthPerPriorAdvice: 0.25,
      maximumMultiplier: 4,
      resetAfterGapSteps: 60,
    });
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
      training: { epochs: 15, lossWeights: { temporalMutualInformation: 0.1 } },
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

test("MLP training metrics discovers plans and defaults to the live run", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "mlp-training-catalog-"));
  const defaultPlanFile = path.join(root, "ml", "training-plan.json");
  const livePlanFile = path.join(root, "ml", "training-plans", "live.json");
  const defaultRunDir = path.join(root, "data", "runs", "default");
  const liveRunDir = path.join(root, "data", "runs", "live");
  const datasetDir = path.join(root, "data", "datasets", "shared");
  await Promise.all([
    mkdir(path.dirname(defaultPlanFile), { recursive: true }),
    mkdir(path.dirname(livePlanFile), { recursive: true }),
    mkdir(defaultRunDir, { recursive: true }),
    mkdir(liveRunDir, { recursive: true }),
    mkdir(datasetDir, { recursive: true }),
  ]);
  await Promise.all([
    writeFile(defaultPlanFile, JSON.stringify({
      id: "default",
      label: "Old production training",
      runDir: "data/runs/default",
      datasetDir: "data/datasets/shared",
      training: { epochs: 256, patience: 64 },
    })),
    writeFile(livePlanFile, JSON.stringify({
      id: "live",
      label: "Fresh Oracle MI training",
      runDir: "data/runs/live",
      datasetDir: "data/datasets/shared",
      training: { epochs: 64, patience: 8 },
    })),
    writeFile(path.join(defaultRunDir, "status.json"), JSON.stringify({
      stage: "complete",
      completedAt: "2026-07-20T00:00:00.000Z",
    })),
    writeFile(path.join(liveRunDir, "status.json"), JSON.stringify({
      pid: process.pid,
      stage: "training",
      updatedAt: "2026-07-25T00:00:00.000Z",
      predictionDelayMs: 59 * 60_000,
    })),
    writeFile(path.join(defaultRunDir, "training.log"), `${JSON.stringify({
      event: "training-complete",
    })}\n`),
    writeFile(path.join(liveRunDir, "training.log"), `${JSON.stringify({
      event: "train-step",
      globalStep: 10,
    })}\n`),
  ]);

  try {
    const reader = new MlpTrainingMetricsReader(defaultPlanFile, root);
    const active = await reader.read(0);
    assert.equal(active.selectedRunKey, "ml/training-plans/live.json");
    assert.equal(active.plan.id, "live");
    assert.equal(active.plan.epochs, 64);
    assert.equal(active.plan.patience, 8);
    assert.equal(active.plan.predictionDelayMs, 59 * 60_000);
    assert.equal(active.running, true);
    assert.deepEqual(active.runs.map((run) => run.key), [
      "ml/training-plans/live.json",
      "ml/training-plan.json",
    ]);
    assert.deepEqual(active.events.map((event) => event.event), ["train-step"]);

    const production = await reader.read(0, "ml/training-plan.json");
    assert.equal(production.selectedRunKey, "ml/training-plan.json");
    assert.equal(production.plan.id, "default");
    assert.equal(production.running, false);
    assert.deepEqual(production.events.map((event) => event.event), ["training-complete"]);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

import assert from "node:assert/strict";
import { appendFile, mkdir, mkdtemp, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import { MlpTrainingMetricsReader } from "../src/mlp-training-metrics.js";

test("MLP training metrics stream only complete new metric events", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "mlp-training-metrics-"));
  const planFile = path.join(root, "ml", "training-plan.json");
  const runDir = path.join(root, "data", "training", "runs", "test-model");
  const datasetDir = path.join(root, "data", "training", "datasets", "test-model");
  await Promise.all([
    mkdir(path.dirname(planFile), { recursive: true }),
    mkdir(runDir, { recursive: true }),
    mkdir(datasetDir, { recursive: true }),
    mkdir(path.join(runDir, "state"), { recursive: true }),
    mkdir(path.join(runDir, "logs"), { recursive: true }),
    mkdir(path.join(datasetDir, "state"), { recursive: true }),
  ]);
  await writeFile(planFile, JSON.stringify({
    id: "test-model",
    label: "Test model",
    runDir: "data/training/runs/test-model",
    datasetDir: "data/training/datasets/test-model",
    samplingIntervalMs: 1_000,
    training: {
      epochs: 12,
      lossWeights: { crossEntropy: 1 },
      reverseKl: {
        predictionMixtureWeight: 1e-2,
      },
      outputRegularizer: {
        applicationProbabilities: {
          reverseKl: 0.01,
          entropySharpness: 0.1,
        },
        samplingUnit: "optimizer-update",
        independentGates: true,
        inverseProbabilityScaling: true,
      },
      softWeightBound: {
        desiredMagnitude: 1,
        sharpness: 10,
        absoluteEpsilon: 1e-8,
      },
      branchNormalization: {
        learnableCentering: false,
      },
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
    writeFile(path.join(runDir, "state", "status.json"), JSON.stringify({
      pid: process.pid,
      stage: "dataset-refinement",
      updatedAt: new Date().toISOString(),
    })),
    writeFile(path.join(datasetDir, "state", "progress.json"), JSON.stringify({
      shards: [{ refinementPass: 1 }, { refinementPass: 0 }],
    })),
    writeFile(path.join(datasetDir, "state", "teacher-refinement-queue.json"), JSON.stringify({
      cases: [{}, {}],
    })),
    writeFile(path.join(runDir, "logs", "training.history.jsonl"), [
      JSON.stringify({ event: "epoch", epoch: 1, globalStep: 24 }),
      "",
    ].join("\n")),
    writeFile(path.join(runDir, "logs", "training.jsonl"), [
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
    assert.deepEqual(first.plan.reverseKl, {
      predictionMixtureWeight: 1e-2,
    });
    assert.deepEqual(first.plan.outputRegularizer, {
      applicationProbabilities: {
        reverseKl: 0.01,
        entropySharpness: 0.1,
      },
      samplingUnit: "optimizer-update",
      independentGates: true,
      inverseProbabilityScaling: true,
    });
    assert.deepEqual(first.plan.softWeightBound, {
      desiredMagnitude: 1,
      sharpness: 10,
      absoluteEpsilon: 1e-8,
    });
    assert.deepEqual(first.plan.branchNormalization, {
      learnableCentering: false,
    });
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
    assert.deepEqual(first.events.map((event) => event.event), ["epoch", "dataset-progress"]);

    const reattachedHistory = await reader.read(1);
    assert.equal(reattachedHistory.reset, true);
    assert.deepEqual(
      reattachedHistory.events.map((event) => event.event),
      ["epoch", "dataset-progress"],
    );

    await appendFile(path.join(runDir, "logs", "training.jsonl"), [
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

    const nextRunDir = path.join(root, "data", "training", "runs", "next-model");
    await mkdir(path.join(nextRunDir, "logs"), { recursive: true });
    await writeFile(planFile, JSON.stringify({
      id: "next-model",
      label: "Next model",
      runDir: "data/training/runs/next-model",
      datasetDir: "data/training/datasets/test-model",
      training: { epochs: 15, lossWeights: { oracleMutualInformation: 0.1 } },
    }));
    await writeFile(path.join(nextRunDir, "logs", "training.jsonl"), [
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
  const defaultRunDir = path.join(root, "data", "training", "runs", "default");
  const liveRunDir = path.join(root, "data", "training", "runs", "live");
  const datasetDir = path.join(root, "data", "training", "datasets", "shared");
  await Promise.all([
    mkdir(path.dirname(defaultPlanFile), { recursive: true }),
    mkdir(path.dirname(livePlanFile), { recursive: true }),
    mkdir(defaultRunDir, { recursive: true }),
    mkdir(liveRunDir, { recursive: true }),
    mkdir(datasetDir, { recursive: true }),
    mkdir(path.join(defaultRunDir, "state"), { recursive: true }),
    mkdir(path.join(defaultRunDir, "logs"), { recursive: true }),
    mkdir(path.join(liveRunDir, "state"), { recursive: true }),
    mkdir(path.join(liveRunDir, "logs"), { recursive: true }),
  ]);
  await Promise.all([
    writeFile(defaultPlanFile, JSON.stringify({
      id: "default",
      label: "Old production training",
      runDir: "data/training/runs/default",
      datasetDir: "data/training/datasets/shared",
      training: { epochs: 256, patience: 64 },
    })),
    writeFile(livePlanFile, JSON.stringify({
      id: "live",
      label: "Fresh Oracle MI training",
      runDir: "data/training/runs/live",
      datasetDir: "data/training/datasets/shared",
      training: { epochs: 64, patience: 8 },
    })),
    writeFile(path.join(defaultRunDir, "state", "status.json"), JSON.stringify({
      stage: "complete",
      completedAt: "2026-07-20T00:00:00.000Z",
    })),
    writeFile(path.join(liveRunDir, "state", "status.json"), JSON.stringify({
      pid: process.pid,
      stage: "training",
      updatedAt: "2026-07-25T00:00:00.000Z",
      predictionDelayMs: 59 * 60_000,
    })),
    writeFile(path.join(defaultRunDir, "logs", "training.jsonl"), `${JSON.stringify({
      event: "training-complete",
    })}\n`),
    writeFile(path.join(liveRunDir, "logs", "training.jsonl"), `${JSON.stringify({
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

test("MLP training metrics discovers decoder plans with nested datasets", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "mlp-training-decoder-"));
  const defaultPlanFile = path.join(root, "ml", "training-plan.json");
  const decoderPlanFile = path.join(root, "ml", "training-plans", "decoder.json");
  const datasetDir = path.join(root, "data", "training", "datasets", "decoder");
  const runDir = path.join(root, "data", "training", "runs", "decoder");
  await Promise.all([
    mkdir(path.dirname(defaultPlanFile), { recursive: true }),
    mkdir(path.dirname(decoderPlanFile), { recursive: true }),
    mkdir(datasetDir, { recursive: true }),
    mkdir(path.join(runDir, "logs"), { recursive: true }),
    mkdir(path.join(runDir, "state"), { recursive: true }),
  ]);
  await Promise.all([
    writeFile(defaultPlanFile, JSON.stringify({
      id: "default",
      label: "Default",
      runDir: "data/training/runs/default",
      datasetDir: "data/training/datasets/decoder",
    })),
    writeFile(decoderPlanFile, JSON.stringify({
      id: "decoder",
      label: "Decoder curriculum",
      runDir: "data/training/runs/decoder",
      dataset: { datasetDir: "data/training/datasets/decoder" },
      architecture: { dropout: 0.05, dropoutRate: 0.5 },
      training: { epochs: 200 },
    })),
    writeFile(path.join(runDir, "state", "status.json"), JSON.stringify({
      pid: process.pid,
      stage: "training",
      updatedAt: "2026-08-02T06:00:00.000Z",
    })),
    writeFile(path.join(runDir, "logs", "training.log"), `${JSON.stringify({
      event: "epoch",
      epoch: 0,
      trainingTargetTemperature: 0.5,
      curriculumValidationKl: 0.08,
    })}\n`),
  ]);

  try {
    const reader = new MlpTrainingMetricsReader(defaultPlanFile, root);
    const decoder = await reader.read(0);
    assert.equal(decoder.plan.id, "decoder");
    assert.equal(decoder.plan.epochs, 200);
    assert.equal(decoder.plan.dropout, 0.05);
    assert.equal(decoder.plan.dropoutRate, 0.5);
    assert.equal(decoder.events[0]?.trainingTargetTemperature, 0.5);
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("MLP training metrics exposes catalogued archived training.log runs", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "mlp-training-archive-"));
  const planFile = path.join(root, "ml", "training-plan.json");
  const catalogFile = path.join(root, "ml", "training-run-catalog.json");
  const liveRunDir = path.join(root, "data", "training", "runs", "live");
  const archivedRunDir = path.join(root, "data", "training", "runs", "archive");
  const datasetDir = path.join(root, "data", "training", "datasets", "shared");
  await Promise.all([
    mkdir(path.dirname(planFile), { recursive: true }),
    mkdir(path.join(liveRunDir, "logs"), { recursive: true }),
    mkdir(path.join(archivedRunDir, "logs"), { recursive: true }),
    mkdir(path.join(archivedRunDir, "state"), { recursive: true }),
    mkdir(datasetDir, { recursive: true }),
  ]);
  await Promise.all([
    writeFile(planFile, JSON.stringify({
      id: "live",
      label: "Live run",
      runDir: "data/training/runs/live",
      datasetDir: "data/training/datasets/shared",
    })),
    writeFile(catalogFile, JSON.stringify({
      runs: [{
        id: "historical-best",
        label: "Historical best",
        runDir: "data/training/runs/archive",
        datasetDir: "data/training/datasets/shared",
        archived: true,
        archivedAt: "2026-07-29T20:56:15.000Z",
        bestValidationKl: 0.09278,
        bestValidationKlEpoch: 133,
      }],
    })),
    writeFile(path.join(liveRunDir, "logs", "training.jsonl"), ""),
    writeFile(path.join(archivedRunDir, "state", "status.json"), JSON.stringify({
      pid: process.pid,
      stage: "training",
      updatedAt: "2026-07-31T20:28:31.000Z",
    })),
    writeFile(path.join(archivedRunDir, "logs", "training.log"), `${JSON.stringify({
      event: "epoch",
      epoch: 133,
      globalStep: 24_388,
      validation: { baseKlDivergence: 0.09278 },
    })}\n`),
  ]);

  try {
    const reader = new MlpTrainingMetricsReader(planFile, root);
    const archived = await reader.read(0, "archive/historical-best");
    assert.equal(archived.selectedRunKey, "archive/historical-best");
    assert.equal(archived.plan.archived, true);
    assert.equal(archived.plan.bestValidationKl, 0.09278);
    assert.equal(archived.plan.bestValidationKlEpoch, 133);
    assert.equal(archived.status?.stage, "training");
    assert.equal(archived.status?.updatedAt, "2026-07-31T20:28:31.000Z");
    assert.equal(archived.running, true);
    assert.deepEqual(archived.events, [{
      event: "epoch",
      epoch: 133,
      globalStep: 24_388,
      validation: { baseKlDivergence: 0.09278 },
    }]);
    const summary = archived.runs.find((run) => run.key === "archive/historical-best");
    assert.equal(summary?.archived, true);
    assert.equal(summary?.running, true);
    assert.equal(summary?.stage, "training");
    assert.equal(summary?.bestValidationKl, 0.09278);
    assert.equal(summary?.bestValidationKlEpoch, 133);
    const defaultRun = await reader.read(0);
    assert.equal(defaultRun.selectedRunKey, "archive/historical-best");
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

test("MLP training metrics canonicalizes minute-return events and patience", async () => {
  const root = await mkdtemp(path.join(tmpdir(), "mlp-training-minute-return-"));
  const planFile = path.join(root, "ml", "training-plan.json");
  const runDir = path.join(root, "data", "training", "runs", "next-second");
  const datasetDir = path.join(root, "data", "training", "datasets", "next-second");
  await Promise.all([
    mkdir(path.dirname(planFile), { recursive: true }),
    mkdir(path.join(runDir, "logs"), { recursive: true }),
    mkdir(path.join(runDir, "state"), { recursive: true }),
    mkdir(datasetDir, { recursive: true }),
  ]);
  await writeFile(planFile, JSON.stringify({
    id: "next-second",
    label: "Next second",
    runDir: "data/training/runs/next-second",
    datasetDir: "data/training/datasets/next-second",
    training: { epochs: 256, earlyStoppingPatience: 32 },
  }));
  await writeFile(path.join(runDir, "state", "status.json"), JSON.stringify({
    pid: process.pid,
    stage: "training",
    updatedAt: new Date().toISOString(),
  }));
  await writeFile(path.join(runDir, "logs", "training.jsonl"), [
    JSON.stringify({ event: "minute-return-dataset-selected", counts: { train: 10 } }),
    JSON.stringify({
      event: "minute-return-epoch",
      epoch: 3,
      validation: { normalizedMse: 0.98, mseSkillVsZero: 0.02 },
    }),
    JSON.stringify({ event: "minute-return-complete", bestEpoch: 3 }),
    "",
  ].join("\n"));

  try {
    const reader = new MlpTrainingMetricsReader(planFile, root);
    const result = await reader.read(0);
    assert.equal(result.plan.patience, 32);
    assert.equal(result.runs[0]?.patience, 32);
    assert.deepEqual(result.events.map((event) => event.event), [
      "dataset-complete",
      "epoch",
      "training-complete",
    ]);
    assert.equal(result.events[1]?.sourceEvent, "minute-return-epoch");
    assert.deepEqual(result.events[1]?.validation, {
      normalizedMse: 0.98,
      mseSkillVsZero: 0.02,
    });
  } finally {
    await rm(root, { recursive: true, force: true });
  }
});

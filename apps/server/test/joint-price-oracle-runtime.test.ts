import assert from "node:assert/strict";
import crypto from "node:crypto";
import { mkdtemp, mkdir, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import test from "node:test";
import {
  discoverJointPriceOracleArtifacts,
  JOINT_PRICE_ORACLE_ACTION_CALIBRATION_OBJECTIVE,
  JOINT_PRICE_ORACLE_ACTION_CALIBRATION_TIE_BREAKERS,
  JOINT_PRICE_ORACLE_DATA_CONTRACT,
  JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_OBJECTIVE,
  JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_TIE_BREAKERS,
  type JointPriceOracleManifest,
  validateJointPriceOracleManifest,
} from "../src/joint-price-oracle-runtime.js";

test("action calibration requires a version-3 validation provenance contract", () => {
  const manifest = actionManifest("0".repeat(64));
  validateJointPriceOracleManifest(manifest);
  assert.throws(
    () => validateJointPriceOracleManifest({ ...manifest, version: 2 }),
    /1h\/1m\/1m contract/,
  );
  assert.throws(
    () => validateJointPriceOracleManifest({
      ...manifest,
      calibration: {
        ...manifest.calibration,
        objective: "raw.klDivergence" as typeof JOINT_PRICE_ORACLE_ACTION_CALIBRATION_OBJECTIVE,
      },
    }),
    /1h\/1m\/1m contract/,
  );
  validateJointPriceOracleManifest({
    ...manifest,
    calibration: {
      ...manifest.calibration,
      objective: JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_OBJECTIVE,
      tieBreakers: [
        ...JOINT_PRICE_ORACLE_EXACT_STATE_CALIBRATION_TIE_BREAKERS,
      ],
    },
  });
});

test("artifact discovery verifies the complete validation calibration report", async () => {
  const dataDir = await mkdtemp(path.join(tmpdir(), "joint-oracle-calibration-"));
  const artifactDir = path.join(dataDir, "models", "joint-price-oracle", "candidate");
  try {
    await mkdir(artifactDir, { recursive: true });
    const report = calibrationReport();
    const reportText = `${JSON.stringify(report, null, 2)}\n`;
    const reportHash = crypto.createHash("sha256").update(reportText).digest("hex");
    await writeFile(path.join(artifactDir, "calibration.json"), reportText);
    await writeFile(
      path.join(artifactDir, "manifest.json"),
      `${JSON.stringify(actionManifest(reportHash), null, 2)}\n`,
    );
    assert.equal(discoverJointPriceOracleArtifacts(dataDir).length, 1);

    await writeFile(path.join(artifactDir, "calibration.json"), `${reportText} `);
    assert.throws(
      () => discoverJointPriceOracleArtifacts(dataDir),
      /calibration report hash/,
    );
  } finally {
    await rm(dataDir, { recursive: true, force: true });
  }
});

function actionManifest(reportHash: string): JointPriceOracleManifest {
  const metrics = calibrationMetrics();
  return {
    version: 3,
    kind: "joint-price-oracle",
    id: "candidate",
    label: "candidate",
    createdAt: "2026-08-01T00:00:00Z",
    modelFile: "model.onnx",
    modelSha256: "1".repeat(64),
    architectureContract: "architecture",
    dataContract: JOINT_PRICE_ORACLE_DATA_CONTRACT,
    input: {
      name: "closes",
      dtype: "float32",
      intervalMs: 1_000,
      contextLength: 3_600,
      variableCount: 1,
    },
    output: {
      name: "action_logits",
      dtype: "float32",
      actionCount: 101,
      actionGrid: Array.from(
        { length: 101 },
        (_unused, index) => (index - 50) * 500 / 254,
      ),
    },
    oracle: {
      intervalMs: 1_000,
      decisionIntervalMs: 60_000,
      decisionPhaseMs: 999,
      options: {
        holdingPeriodSteps: 60,
        decisionDelaySteps: 60,
        valueHorizonSteps: 3_600,
        friction: 0.00175,
        temperature: 0.01,
      },
    },
    training: {
      bestEpoch: 7,
      globalStep: 123,
      validation: {},
      datasetFingerprint: "dataset-sha",
    },
    calibration: {
      method: "validation-action-temperature-scaling",
      logitTemperature: 0.5,
      objective: JOINT_PRICE_ORACLE_ACTION_CALIBRATION_OBJECTIVE,
      tieBreakers: [...JOINT_PRICE_ORACLE_ACTION_CALIBRATION_TIE_BREAKERS],
      calibratedAt: "2026-08-01T00:01:00Z",
      validationReportFile: "calibration.json",
      validationReportSha256: reportHash,
      validationReportVersion: 3,
      validationExamples: 100,
      candidateCount: 2,
      checkpoint: { kind: "best", epoch: 7, globalStep: 123 },
      datasetFingerprint: "dataset-sha",
      executionPolicy: executionPolicy(),
      rolloutScoreVersion: 2,
      rolloutResetPolicy: "reset-only-at-true-timeline-gaps",
      selectedMetrics: metrics,
      identityMetrics: metrics,
    },
  };
}

function calibrationReport(): Record<string, unknown> {
  const metrics = calibrationMetrics();
  return {
    version: 3,
    plan: { id: "candidate" },
    checkpoint: { kind: "best", epoch: 7, globalStep: 123 },
    split: { name: "validation", examples: 100 },
    datasetFingerprint: "dataset-sha",
    actionEvaluation: {
      executionPolicy: executionPolicy(),
      rolloutScoreVersion: 2,
      rolloutResetPolicy: "reset-only-at-true-timeline-gaps",
    },
    actionTemperatureCalibration: {
      method: "validation-action-temperature-scaling-v1",
      objective: {
        metric: JOINT_PRICE_ORACLE_ACTION_CALIBRATION_OBJECTIVE,
        direction: "maximize",
      },
      tieBreakers: [...JOINT_PRICE_ORACLE_ACTION_CALIBRATION_TIE_BREAKERS],
      candidateCount: 2,
      selectedLogitTemperature: 0.5,
      selectedMetrics: metrics,
      identityMetrics: metrics,
    },
  };
}

function calibrationMetrics() {
  return {
    raw: { klDivergence: 0.5 },
    actions: { signedTransitionF1: 0.4 },
  };
}

function executionPolicy() {
  return {
    version: 2 as const,
    maximumLeverage: 100,
    minimumConfidence: 0.05,
    confidenceExposurePower: 0,
    confidenceLeverageFloor: 0.75,
  };
}

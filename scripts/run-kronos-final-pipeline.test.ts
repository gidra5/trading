import assert from "node:assert/strict";
import test from "node:test";

import {
  assertPolicyGate,
  assertValidationReport,
  type PolicyIdentity,
  type ValidationIdentity,
} from "./run-kronos-final-pipeline.js";

const RUN_SIGNATURE = "a".repeat(64);
const FORECAST_SHA256 = "b".repeat(64);
const POLICY_SHA256 = "c".repeat(64);
const POLICY_ID = "policy-1";

function validPolicy(): PolicyIdentity {
  const aggregate = {
    selectionEligible: true,
    geometricMeanReturnPct: 1,
    tradeCount: 2,
    liquidatedPositionCount: 0,
  };
  return {
    contract: "kronos-bot-policy-v3",
    selectionContract:
      "pretraining-ranking-with-untouched-post-training-confirmation-v2",
    forecastRunSignature: RUN_SIGNATURE,
    selectedPolicyId: POLICY_ID,
    selection: { ...aggregate },
    postTrainingConfirmation: { ...aggregate },
  };
}

function validValidation(): ValidationIdentity {
  const episodes = Array.from({ length: 6 }, (_, index) => ({
    windowId: `validation-episode-${index + 1}`,
    startTime: index * 100_000,
    endTime: index * 100_000 + 90_000,
    forecastRows: 10,
    forecastRowsConsumed: 10,
    summary: {
      netPnl: 10,
      returnPct: 0.1,
      tradeCount: 1,
      feesPaid: 1,
      maintenancePaid: 2,
      liquidatedPositionCount: 0,
    },
    fills: [{
      side: "buy" as const,
      price: 100,
      quantity: 1,
      quoteQuantity: 100,
      feeQuote: 1,
      realizedPnl: 0,
      filledAt: index * 100_000 + 10_000,
      reason: "learned-oracle-1m",
    }],
  }));
  return {
    version: 3,
    contract: "kronos-bot-backtest-v3",
    phase: "validation",
    forecast: { runSignature: RUN_SIGNATURE, sha256: FORECAST_SHA256 },
    policyArtifact: { sha256: POLICY_SHA256 },
    policyId: POLICY_ID,
    split: { windowIds: Array.from({ length: 8 }, (_, index) => `window-${index}`) },
    aggregate: {
      windows: 6,
      netPnl: 60,
      geometricMeanReturnPct: 0.1,
      tradeCount: 6,
      feesPaid: 6,
      maintenancePaid: 12,
      liquidatedPositionCount: 0,
      forecastRows: 60,
      forecastRowsConsumed: 60,
    },
    gates: {
      allForecastsConsumed: true,
      actualFills: true,
      noLiquidations: true,
      positiveNetPnl: true,
    },
    episodes,
    controls: [
      { id: "constant-long-1x" },
      { id: "constant-short-1x" },
      { id: "constant-long-5x" },
      { id: "constant-short-5x" },
    ],
  };
}

test("final validation audit proves fills, coverage, costs, and risk", () => {
  assert.doesNotThrow(() => assertValidationReport(
    validValidation(), RUN_SIGNATURE, POLICY_ID, FORECAST_SHA256, POLICY_SHA256,
  ));
});

test("final policy gate rejects stale confirmation semantics", () => {
  assert.doesNotThrow(() => assertPolicyGate(validPolicy(), RUN_SIGNATURE));
  assert.throws(
    () => assertPolicyGate({
      ...validPolicy(),
      selectionContract: "broad-episodes-with-post-training-confirmation-v1",
    }, RUN_SIGNATURE),
    /Frozen policy does not match/,
  );
});

test("final validation audit binds policy and forecast bytes", () => {
  assert.throws(
    () => assertValidationReport(
      validValidation(), RUN_SIGNATURE, "different-policy", FORECAST_SHA256,
    ),
    /identity check/,
  );
  assert.throws(
    () => assertValidationReport(
      validValidation(), RUN_SIGNATURE, POLICY_ID, FORECAST_SHA256, "d".repeat(64),
    ),
    /identity check/,
  );
});

test("final validation audit rejects missing execution evidence", () => {
  const validation = validValidation();
  validation.episodes[0]!.fills = [];

  assert.throws(
    () => assertValidationReport(validation, RUN_SIGNATURE),
    /inconsistent execution evidence/,
  );
});

test("final validation audit keeps held-out losses truthful", () => {
  const validation = validValidation();
  validation.aggregate.netPnl = -10;
  validation.gates.positiveNetPnl = false;

  assert.doesNotThrow(() => assertValidationReport(validation, RUN_SIGNATURE));
});

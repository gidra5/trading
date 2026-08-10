import assert from "node:assert/strict";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import {
  atomicJson,
  aggregateReports,
  calibrationPolicies,
  confirmedPolicySelection,
  distributionForPolicy,
  kronosInspectorWindows,
  mergeOverlappingWindows,
  parseKronosForecastArtifact,
  policyCalibrationEpisodeSplit,
  policyId,
  probabilityDistribution,
  validateDenseForecastCoverage,
  validateForecastConsumption,
  validatePolicy,
  type KronosBotPolicy,
  type PolicyFilterState,
  type WindowReport,
} from "./backtest-kronos.js";

const GRID = [-2, 0, 2];

test("fine-tune exclusion plan exactly matches pre-validation inspector episodes", () => {
  const planFile = path.resolve(
    import.meta.dirname,
    "../ml/training-plans/kronos-btcusdt-1m-policy-holdout-v1.json",
  );
  const plan = JSON.parse(fs.readFileSync(planFile, "utf8")) as {
    version: number;
    contract: string;
    validationCutoff: string;
    ranges: Array<{
      id: string;
      start: string;
      end: string;
      sourceWindowIds: string[];
    }>;
  };
  const cutoff = Date.parse(plan.validationCutoff);
  const episodes = mergeOverlappingWindows(
    kronosInspectorWindows().filter((window) => window.startTime < cutoff),
    "calibration",
  );
  assert.equal(plan.version, 1);
  assert.equal(
    plan.contract,
    "kronos-btcusdt-1m-finetune-exclusion-plan-v1",
  );
  assert.equal(cutoff, Date.parse("2024-07-01T00:00:00.000Z"));
  assert.deepEqual(
    plan.ranges,
    episodes.map((episode) => ({
      id: episode.id,
      start: new Date(episode.startTime).toISOString(),
      end: new Date(episode.endTime).toISOString(),
      sourceWindowIds: episode.sourceWindowIds,
    })),
  );
});

test("policy ranking leaves post-training confirmation episodes untouched", () => {
  const cutoff = Date.parse("2024-07-01T00:00:00.000Z");
  const trainingEnd = Date.parse("2024-01-01T00:00:00.000Z");
  const episodes = mergeOverlappingWindows(
    kronosInspectorWindows().filter((window) => window.startTime < cutoff),
    "calibration",
  );
  const split = policyCalibrationEpisodeSplit(episodes);

  assert.equal(split.selectionEpisodes.length, 11);
  assert.equal(split.postTrainingConfirmationEpisodes.length, 2);
  assert.ok(split.selectionEpisodes.every((episode) => episode.endTime <= trainingEnd));
  assert.ok(split.postTrainingConfirmationEpisodes.every((episode) =>
    episode.startTime >= trainingEnd));
  assert.deepEqual(
    new Set([
      ...split.selectionEpisodes,
      ...split.postTrainingConfirmationEpisodes,
    ].map((episode) => episode.id)),
    new Set(episodes.map((episode) => episode.id)),
  );
});

test("atomic JSON publication retries transient destination locks", () => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "kronos-atomic-json-"));
  const output = path.join(directory, "progress.json");
  const delays: number[] = [];
  let attempts = 0;
  try {
    atomicJson(
      output,
      { completedCandidates: 5, invalid: Number.POSITIVE_INFINITY },
      (temporary, destination) => {
        attempts += 1;
        if (attempts < 3) {
          const error = new Error("destination is temporarily locked") as NodeJS.ErrnoException;
          error.code = "EPERM";
          throw error;
        }
        fs.renameSync(temporary, destination);
      },
      (milliseconds) => delays.push(milliseconds),
    );
    assert.equal(attempts, 3);
    assert.deepEqual(delays, [50, 100]);
    assert.deepEqual(JSON.parse(fs.readFileSync(output, "utf8")), {
      completedCandidates: 5,
      invalid: null,
    });
    assert.deepEqual(
      fs.readdirSync(directory).filter((name) => name.endsWith(".tmp")),
      [],
    );
  } finally {
    fs.rmSync(directory, { recursive: true, force: true });
  }
});

function artifact() {
  return {
    version: 2,
    contract: "kronos-causal-15x1m-forecast-v2",
    generatedAt: "2026-08-06T00:00:00.000Z",
    runSignature: "fixture",
    modelId: "base-fixture",
    market: "Binance spot BTCUSDT",
    intervalMs: 60_000,
    lookbackCandles: 512,
    horizonCandles: 15,
    temperature: 0.8,
    topP: 0.9,
    sampleCount: 20,
    originStrideCandles: 15,
    oracleGrid: GRID,
    executionOracleGrid: [-1, 0, 1],
    executionOracle: {
      holding_period_steps: 15,
      decision_delay_steps: 15,
      value_horizon_steps: 15,
      friction: 0.00175,
      grid_size: 3,
      temperature: 0.01,
      min_exposure: -1,
      max_exposure: 1,
      max_effective_exposure: 2.5,
      quote_borrow_rate: 0.000016658479699709332,
      asset_borrow_rate: 0.000016658479699709332,
    },
    rows: [row(0), row(900_000)],
  };
}

function row(targetStartTime: number) {
  return {
    decisionTime: targetStartTime - 1,
    targetStartTime,
    windowIds: ["fixture"],
    anchorPrice: 100,
    horizonLogReturnMean: 0.001,
    horizonLogReturnMedian: 0.001,
    horizonLogReturnStd: 0.002,
    horizonUpProbability: 0.65,
    horizonLogReturnP10: -0.001,
    horizonLogReturnP90: 0.003,
    meanCloseLogPath: Array.from({ length: 15 }, (_, index) => index / 10_000),
    medianCloseLogPath: Array.from({ length: 15 }, (_, index) => index / 10_000),
    oracleProbabilities: [0.1, 0.2, 0.7],
    executionOracleProbabilities: [0.2, 0.3, 0.5],
    executionUtilityProbabilities: [0.1, 0.2, 0.7],
  };
}

test("forecast parser enforces the strictly causal, target-free row contract", () => {
  const parsed = parseKronosForecastArtifact(artifact());
  assert.equal(parsed.rows[0]!.decisionTime, -1);
  assert.throws(
    () => parseKronosForecastArtifact({
      ...artifact(),
      rows: [{ ...row(0), realizedClose: 101 }],
    }),
    /forbidden or unknown field realizedClose/,
  );
  assert.throws(
    () => parseKronosForecastArtifact({
      ...artifact(),
      rows: [{ ...row(0), decisionTime: 0 }],
    }),
    /causal decision-time contract/,
  );
  assert.throws(
    () => parseKronosForecastArtifact({
      ...artifact(),
      executionOracle: {
        ...artifact().executionOracle,
        holding_period_steps: 1,
        decision_delay_steps: 1,
        value_horizon_steps: 15,
      },
    }),
    /complete 15-minute bot decision interval/,
  );
  assert.throws(
    () => parseKronosForecastArtifact({
      ...artifact(),
      executionOracle: { ...artifact().executionOracle, friction: 0.001 },
    }),
    /must match the bot simulator/,
  );
  assert.throws(
    () => parseKronosForecastArtifact({
      ...artifact(),
      rows: artifact().rows.map(({ executionUtilityProbabilities: _, ...value }) => value),
    }),
    /both execution-oracle aggregations/,
  );
});

test("forecast parser accepts the foundation-model causal contract without Kronos sampling fields", () => {
  const source = artifact();
  const parsed = parseKronosForecastArtifact({
    ...source,
    version: 1,
    contract: "foundation-forecast-causal-15x1m-v1",
    temperature: undefined,
    topP: undefined,
    sampleCount: undefined,
    originStrideCandles: undefined,
    nativeQuantileLevels: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  });
  assert.equal(parsed.version, 1);
  assert.equal(parsed.contract, "foundation-forecast-causal-15x1m-v1");
  assert.equal(parsed.temperature, 1);
  assert.equal(parsed.topP, 1);
  assert.equal(parsed.sampleCount, 9);
  assert.equal(parsed.rows.length, 2);
});

test("dense coverage requires every scheduled 15-minute origin", () => {
  const parsed = parseKronosForecastArtifact(artifact());
  const window = {
    id: "fixture",
    label: "Fixture",
    group: "test",
    startTime: 0,
    endTime: 1_800_000,
    sourceIntervalMs: 60_000,
  };
  validateDenseForecastCoverage(parsed, [window]);
  const sparse = parseKronosForecastArtifact({ ...artifact(), rows: [row(0)] });
  assert.throws(
    () => validateDenseForecastCoverage(sparse, [window]),
    /missing dense forecast origin/,
  );
});

test("only a recorded liquidation may terminate forecast consumption early", () => {
  validateForecastConsumption(10, 10, 0, "complete");
  validateForecastConsumption(7, 10, 1, "liquidated");
  assert.throws(
    () => validateForecastConsumption(7, 10, 0, "truncated"),
    /truncated consumed 7\/10 forecasts/,
  );
});

test("overlapping inspector windows collapse into independent chronological episodes", () => {
  const episodes = mergeOverlappingWindows([
    {
      id: "later-overlap",
      label: "Later overlap",
      group: "test",
      startTime: 1_800_000,
      endTime: 3_600_000,
      sourceIntervalMs: 1_000,
    },
    {
      id: "first",
      label: "First",
      group: "test",
      startTime: 0,
      endTime: 2_700_000,
      sourceIntervalMs: 1_000,
    },
    {
      id: "independent",
      label: "Independent",
      group: "test",
      startTime: 5_400_000,
      endTime: 6_300_000,
      sourceIntervalMs: 1_000,
    },
  ], "calibration");

  assert.deepEqual(episodes.map((episode) => ({
    id: episode.id,
    startTime: episode.startTime,
    endTime: episode.endTime,
    sourceWindowIds: episode.sourceWindowIds,
  })), [
    {
      id: "calibration-episode-01",
      startTime: 0,
      endTime: 3_600_000,
      sourceWindowIds: ["first", "later-overlap"],
    },
    {
      id: "calibration-episode-02",
      startTime: 5_400_000,
      endTime: 6_300_000,
      sourceWindowIds: ["independent"],
    },
  ]);
});

test("probability conversion normalizes moments and modal exposure", () => {
  const distribution = probabilityDistribution([1, 2, 7], GRID);
  assert.ok(Array.from(distribution.probabilities).every(
    (value, index) => Math.abs(value - [0.1, 0.2, 0.7][index]!) < 1e-6,
  ));
  assert.ok(Math.abs(distribution.mean - 1.2) < 1e-6);
  assert.ok(Math.abs(distribution.secondMoment - 3.2) < 1e-6);
  assert.equal(distribution.modalExposure, 2);
  assert.equal(distribution.feasibleActionCount, 3);
});

test("policy identity is deterministic and invalid risk values are rejected", () => {
  const policy: KronosBotPolicy = {
    signalSource: "execution-oracle",
    filteredAction: "flat",
    minimumConsecutiveDirections: 1,
    maximumLeverage: 1,
    staticConfidenceScale: 0.75,
    confidenceExposurePower: 1,
    confidenceLeverageFloor: 0.75,
    expansionConfirmationMass: 0.5,
    expansionDeltaCapFraction: 1,
    minimumAbsoluteMeanReturnBps: 10,
    minimumDirectionalConfidence: 0.2,
    requireOracleReturnAgreement: true,
  };
  validatePolicy(policy);
  assert.equal(policyId(policy), policyId({ ...policy }));
  assert.throws(
    () => validatePolicy({ ...policy, staticConfidenceScale: 1.1 }),
    /staticConfidenceScale must be in \[0, 1\]/,
  );
  assert.throws(
    () => validatePolicy({ ...policy, filteredAction: "wait" as "hold" }),
    /filteredAction must be flat or hold/,
  );
  assert.throws(
    () => validatePolicy({ ...policy, minimumConsecutiveDirections: 0 }),
    /minimumConsecutiveDirections must be a positive integer/,
  );
});

test("consecutive-direction filtering holds until the signal is confirmed", () => {
  const policy: KronosBotPolicy = {
    signalSource: "horizon-median",
    filteredAction: "hold",
    minimumConsecutiveDirections: 2,
    maximumLeverage: 1,
    staticConfidenceScale: 1,
    confidenceExposurePower: 0,
    confidenceLeverageFloor: 1,
    expansionConfirmationMass: 0,
    expansionDeltaCapFraction: 1,
    minimumAbsoluteMeanReturnBps: 0,
    minimumDirectionalConfidence: 0,
    requireOracleReturnAgreement: false,
  };
  const neutral = probabilityDistribution([0, 1, 0], GRID);
  const short = probabilityDistribution([1, 0, 0], GRID);
  const long = probabilityDistribution([0, 0, 1], GRID);
  const state: PolicyFilterState = { direction: 0, consecutiveDirections: 0 };
  const positive = {
    row: parseKronosForecastArtifact(artifact()).rows[0]!,
    distribution: probabilityDistribution([0.2, 0.3, 0.5], GRID),
    utilityDistribution: probabilityDistribution([0.1, 0.2, 0.7], GRID),
    calibratedReturns: new Map<string, number>(),
  };

  const first = distributionForPolicy(positive, policy, state, neutral, short, long);
  assert.equal(first.filtered, true);
  assert.equal(first.distribution, null);
  assert.deepEqual(state, { direction: 1, consecutiveDirections: 1 });

  const second = distributionForPolicy(positive, policy, state, neutral, short, long);
  assert.equal(second.filtered, false);
  assert.equal(second.distribution?.modalExposure, 2);
  assert.deepEqual(state, { direction: 1, consecutiveDirections: 2 });

  const negative = {
    ...positive,
    row: { ...positive.row, horizonLogReturnMedian: -0.001 },
  };
  const reversal = distributionForPolicy(negative, policy, state, neutral, short, long);
  assert.equal(reversal.filtered, true);
  assert.equal(reversal.distribution, null);
  assert.deepEqual(state, { direction: -1, consecutiveDirections: 1 });
});

test("execution mean and sign estimators follow the oracle-distribution mean", () => {
  const basePolicy: KronosBotPolicy = {
    signalSource: "execution-sign",
    filteredAction: "hold",
    minimumConsecutiveDirections: 1,
    maximumLeverage: 2,
    staticConfidenceScale: 1,
    confidenceExposurePower: 0,
    confidenceLeverageFloor: 1,
    expansionConfirmationMass: 0,
    expansionDeltaCapFraction: 1,
    minimumAbsoluteMeanReturnBps: 0,
    minimumDirectionalConfidence: 0,
    requireOracleReturnAgreement: false,
  };
  const neutral = probabilityDistribution([0, 1, 0], GRID);
  const short = probabilityDistribution([1, 0, 0], GRID);
  const long = probabilityDistribution([0, 0, 1], GRID);
  const forecast = {
    row: parseKronosForecastArtifact(artifact()).rows[0]!,
    // The distribution mean is negative even though the row's horizon return is positive.
    distribution: probabilityDistribution([0.8, 0.1, 0.1], GRID),
    utilityDistribution: probabilityDistribution([0.1, 0.1, 0.8], GRID),
    calibratedReturns: new Map<string, number>(),
  };

  const signed = distributionForPolicy(
    forecast,
    basePolicy,
    { direction: 0, consecutiveDirections: 0 },
    neutral,
    short,
    long,
  );
  assert.equal(signed.distribution?.modalExposure, -2);

  const mean = distributionForPolicy(
    forecast,
    { ...basePolicy, signalSource: "execution-mean" },
    { direction: 0, consecutiveDirections: 0 },
    neutral,
    short,
    long,
  );
  assert.equal(mean.distribution?.modalExposure, -2);

  const utility = distributionForPolicy(
    forecast,
    { ...basePolicy, signalSource: "execution-utility" },
    { direction: 0, consecutiveDirections: 0 },
    neutral,
    short,
    long,
  );
  assert.equal(utility.distribution?.modalExposure, 2);

  const utilityMean = distributionForPolicy(
    forecast,
    { ...basePolicy, signalSource: "execution-utility-mean" },
    { direction: 0, consecutiveDirections: 0 },
    neutral,
    short,
    long,
  );
  assert.equal(utilityMean.distribution?.modalExposure, 2);

  const utilitySign = distributionForPolicy(
    forecast,
    { ...basePolicy, signalSource: "execution-utility-sign" },
    { direction: 0, consecutiveDirections: 0 },
    neutral,
    short,
    long,
  );
  assert.equal(utilitySign.distribution?.modalExposure, 2);

  const neutralOracle = distributionForPolicy(
    {
      ...forecast,
      distribution: probabilityDistribution([0.5, 0, 0.5], GRID),
    },
    {
      ...basePolicy,
      signalSource: "horizon-mean",
      requireOracleReturnAgreement: true,
    },
    { direction: 0, consecutiveDirections: 0 },
    neutral,
    short,
    long,
  );
  assert.equal(neutralOracle.filtered, true);
  assert.equal(neutralOracle.distribution, null);

  const calibrated = distributionForPolicy(
    {
      ...forecast,
      calibratedReturns: new Map([["ridge-1e0", -0.002]]),
    },
    {
      ...basePolicy,
      signalSource: "calibrated-return",
      returnCalibrationId: "ridge-1e0",
      minimumAbsoluteMeanReturnBps: 10,
    },
    { direction: 0, consecutiveDirections: 0 },
    neutral,
    short,
    long,
  );
  assert.equal(calibrated.filtered, false);
  assert.equal(calibrated.distribution?.modalExposure, -2);
  assert.throws(
    () => validatePolicy({
      ...basePolicy,
      signalSource: "calibrated-return",
    }),
    /require returnCalibrationId/,
  );
});

test("calibration policy grid is bounded and contains no duplicate policies", () => {
  const policies = calibrationPolicies(5);
  assert.equal(policies.length, 630);
  assert.equal(new Set(policies.map(policyId)).size, policies.length);
  assert.ok(policies.some((policy) => policy.minimumConsecutiveDirections === 12));
  assert.ok(policies.some((policy) => policy.signalSource === "execution-mean"));
  assert.ok(policies.some((policy) => policy.signalSource === "execution-sign"));
  assert.ok(policies.some((policy) => policy.signalSource === "execution-utility"));
  assert.ok(policies.some((policy) => policy.signalSource === "execution-utility-mean"));
  assert.ok(policies.some((policy) => policy.signalSource === "execution-utility-sign"));
  assert.ok(policies.some((policy) => policy.expansionDeltaCapFraction === 0.75));
  assert.ok(policies.filter((policy) => policy.expansionDeltaCapFraction === 0.75)
    .every((policy) => policy.staticConfidenceScale === 1));

  const withReturnCalibration = calibrationPolicies(5, ["ridge-1e-2"]);
  assert.equal(withReturnCalibration.length, 690);
  assert.equal(new Set(withReturnCalibration.map(policyId)).size, 690);
  assert.ok(withReturnCalibration.some((policy) =>
    policy.signalSource === "calibrated-return"
    && policy.returnCalibrationId === "ridge-1e-2"));

  const completeGrid = calibrationPolicies(5, ["a", "b", "c", "d", "e", "f"]);
  assert.equal(completeGrid.length, 990);
  assert.equal(new Set(completeGrid.map(policyId)).size, 990);
});

test("policy aggregation enforces profitability and chronological robustness", () => {
  const returns = [2, 2, -1, -1];
  const reports = returns.map((returnPct, index): WindowReport => ({
    windowId: `window-${index}`,
    label: `Window ${index}`,
    startTime: index * 1_000,
    endTime: (index + 1) * 1_000,
    forecastRows: 1,
    forecastRowsConsumed: 1,
    oracleDecisions: 1,
    emittedSignals: 1,
    filteredForecasts: 0,
    heldForecasts: 0,
    flattenedForecasts: 0,
    summary: {
      finalEquity: 10_000 * (1 + returnPct / 100),
      netPnl: 100 * returnPct,
      returnPct,
      maxInitialBalanceDrawdownPct: Math.max(0, -returnPct),
      maxDrawdownPct: Math.max(0, -returnPct),
      maxEffectiveLeverage: 1,
      perfectMarginNetPnl: 1_000,
      perfectMarginReturnPct: 10,
      perfectMarginCapturePct: returnPct,
      tradeCount: 1,
      feesPaid: 1,
      maintenancePaid: 0,
      winRate: returnPct > 0 ? 1 : 0,
      closedPositionCount: 1,
      profitableClosedPositionCount: returnPct > 0 ? 1 : 0,
      liquidatedPositionCount: 0,
    },
  }));
  const aggregate = aggregateReports(reports);
  assert.equal(aggregate.selectionEligible, true);
  assert.equal(aggregate.profitableWindows, 2);
  assert.equal(aggregate.chronologicalFoldCount, 4);
  assert.ok(Math.abs(aggregate.worstChronologicalFoldReturnPct + 1) < 1e-9);

  const allLosing = aggregateReports(reports.map((report) => ({
    ...report,
    summary: {
      ...report.summary,
      finalEquity: 9_900,
      netPnl: -100,
      returnPct: -1,
    },
  })));
  assert.equal(allLosing.selectionEligible, false);

  const oneOutlierWinner = aggregateReports(reports.map((report, index) => {
    const returnPct = index === 0 ? 20 : -1;
    return {
      ...report,
      summary: {
        ...report.summary,
        finalEquity: 10_000 * (1 + returnPct / 100),
        netPnl: 100 * returnPct,
        returnPct,
      },
    };
  }));
  assert.equal(oneOutlierWinner.geometricMeanReturnPct > 0, true);
  assert.equal(oneOutlierWinner.selectionEligible, false);

  const liquidatedWinner = aggregateReports(reports.map((report, index) => ({
    ...report,
    summary: {
      ...report.summary,
      liquidatedPositionCount: index === 0 ? 1 : 0,
    },
  })));
  assert.equal(liquidatedWinner.geometricMeanReturnPct > 0, true);
  assert.equal(liquidatedWinner.selectionEligible, false);

  assert.deepEqual(confirmedPolicySelection(aggregate, aggregate), {
    eligible: true,
    score: aggregate.selectionScore,
  });
  assert.deepEqual(confirmedPolicySelection(aggregate, allLosing), {
    eligible: false,
    score: -1e12,
  });
});

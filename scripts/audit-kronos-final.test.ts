import assert from "node:assert/strict";
import test from "node:test";

import { auditBenchmarkReport, MODEL_SPECS } from "./audit-kronos-final.js";

const WINDOW_IDS = Array.from({ length: 28 }, (_, index) => `window-${index}`);

function benchmarkFixture(): Record<string, unknown> {
  return {
    version: 2,
    contract: "kronos-probabilistic-nonfit-inspector-windows-15x1m-v2",
    runSignature: "a".repeat(64),
    kronosSource: { commit: "67b630e67f6a18c9e9be918d9b4337c960db1e9a" },
    data: {
      windows: 28,
      excludedWindows: ["fit-full", "fit-1", "fit-2", "fit-3", "fit-4", "latest-3m"],
      horizonCandles: 15,
      lookbackCandles: 512,
      originStrideCandles: 15,
      uniqueOrigins: 106,
      windowOriginMemberships: 112,
    },
    sampling: {
      temperature: 0.8,
      topP: 0.9,
      sampleCount: 20,
      retainedPaths: true,
      constraintRepair: "KQSP",
    },
    models: MODEL_SPECS.map((spec) => ({
      id: spec.id,
      model: {
        repoId: spec.repoId,
        revision: spec.revision,
        tokenizerRepoId: spec.tokenizerRepoId,
        tokenizerRevision: spec.tokenizerRevision,
        publishedMaxContext: spec.publishedMaxContext,
      },
      primaryEstimator: "ensembleMean",
      uniqueOrigins: {
        examples: 106,
        candle: { mseSkillVsPersistence: 0.1, anchoredLogCorrelation: 0.2 },
        closeReturn: { correlation: 0.3 },
        paperAligned: { horizonReturnIc: 0.4, priceSeriesIc: 0.5 },
        oracle: { forwardKl: 1.2 },
      },
      windows: WINDOW_IDS.map((id) => ({ id })),
      probabilistic: {
        uniqueOrigins: {
          samplePathCrpsAnchoredLog: 0.001,
          repairedQuantileValidOhlcFraction: 1,
        },
      },
    })),
  };
}

const OPTIONS = {
  expectedModels: MODEL_SPECS,
  expectedWindowIds: WINDOW_IDS,
  expectedUniqueOrigins: 106,
  expectedMemberships: 112,
  expectedStride: 15,
} as const;

test("Kronos completion audit accepts exact all-model metric evidence", () => {
  const result = auditBenchmarkReport(benchmarkFixture(), OPTIONS);
  assert.equal(result.models.length, 3);
  assert.deepEqual(result.models.map((model) => model.id), ["mini", "small", "base"]);
});

test("Kronos completion audit rejects incomplete KQSP repair", () => {
  const fixture = benchmarkFixture();
  const models = fixture.models as Array<Record<string, unknown>>;
  const probabilistic = models[0]!.probabilistic as Record<string, unknown>;
  const unique = probabilistic.uniqueOrigins as Record<string, unknown>;
  unique.repairedQuantileValidOhlcFraction = 0.999;

  assert.throws(
    () => auditBenchmarkReport(fixture, OPTIONS),
    /KQSP repaired validity/,
  );
});

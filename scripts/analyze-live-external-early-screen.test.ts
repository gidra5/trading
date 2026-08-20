import assert from "node:assert/strict";
import test from "node:test";
import {
  buildReadinessTable,
  classifyEarlyScore,
  effectiveOutcomeCount,
  observedCoverage,
} from "./analyze-live-external-early-screen.ts";

test("readiness counts non-overlapping target outcomes", () => {
  const rows = buildReadinessTable();
  assert.equal(rows.find((row) => row.horizonSeconds === 1)!.independentOutcomesAt1Day, 86_400);
  assert.equal(rows.find((row) => row.horizonSeconds === 3_600)!.independentOutcomesAt1Day, 24);
  assert.equal(effectiveOutcomeCount(3_600, 60, 1_000), 60);
});

test("coverage counts observations rather than wall-clock downtime", () => {
  const rows = [0, 1, 2, 100, 101].map((second) => ({ recordedAt: second * 1_000 }));
  assert.deepEqual(observedCoverage(rows), {
    seconds: 9,
    firstObservedAt: 0,
    lastObservedAt: 101_000,
    wallSpanSeconds: 102,
  });
});

test("early decisions require positive gain in most chronological blocks", () => {
  assert.deepEqual(classifyEarlyScore(24, 0.01, [0.01, 0.02, 0.03, -0.01]), {
    evidence: "early",
    decision: "promising",
  });
  assert.deepEqual(classifyEarlyScore(1, -0.01, [-0.01, -0.02, -0.03, 0.01]), {
    evidence: "smoke-only",
    decision: "weak-in-this-window",
  });
});

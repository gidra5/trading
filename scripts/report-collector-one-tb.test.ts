import assert from "node:assert/strict";
import test from "node:test";
import { capacityRequirement, component, projectMeasuredTransfer } from "./report-collector-one-tb.ts";

test("one-TB archive math reserves physical working space before applying activity stress", () => {
  const result = capacityRequirement([{ source: "spot:BTCUSDT/trade", records: 10e9, compressedBytes: 2e12 }]);
  assert.equal(result.archiveBudget, 800e9);
  assert.equal(result.stressBytes, 6e12);
  assert.equal(result.requiredWholeArchiveRatio, 800e9 / 6e12);
  assert.equal(result.availableBytesPerModeledRecord, 800e9 / 30e9);
});

test("kind transfers use worst verified measurement and give unknown sources no credit", () => {
  const rows = [{ source: "spot:BTCUSDT/trade", records: 1, compressedBytes: 100 },
    { source: "usdm-futures:BTCUSDT/local-checkpoint", records: 1, compressedBytes: 100 }];
  const basis = { candidate: "fixture", note: "fixture", measurements: [
    { component: "spot/trade", originalGzipBytes: 100, encodedBytes: 10, roundTripByteExact: true, scope: "busy", source: "fixture" },
    { component: "spot/trade", originalGzipBytes: 100, encodedBytes: 50, roundTripByteExact: true, scope: "quiet", source: "fixture" },
  ] };
  const result = projectMeasuredTransfer(rows, basis);
  assert.equal(result.baseBytes, 150);
  assert.equal(result.unmeasuredOriginalBytes, 100);
  assert.equal(component(rows[1]!.source), "usdm-futures/local-checkpoint");
  assert.throws(() => projectMeasuredTransfer(rows, { ...basis, measurements: [{ ...basis.measurements[0]!, roundTripByteExact: false }] }));
});

test("shared socket metadata correction does not reduce independent checkpoint bytes", () => {
  const rows = [{ source: "usdm-futures:BTCUSDT/BBO", records: 10, compressedBytes: 100 },
    { source: "usdm-futures:BTCUSDT/local-checkpoint", records: 1, compressedBytes: 100 }];
  const result = projectMeasuredTransfer(rows, { candidate: "fixture", note: "fixture", measurements: [],
    baselineCorrection: { components: ["usdm-futures/BBO"], factor: .5, evidence: "matched shard fixture" } });
  assert.equal(result.correctedBaselineBytes, 150);
  assert.equal(result.rows[1]!.baselineCorrection, 1);
});

import assert from "node:assert/strict";
import test from "node:test";
import { buildExternalFeatureAudit } from "./analyze-external-feature-basis.ts";

test("external feature audit covers every requested family and records lookbacks", () => {
  const audit = buildExternalFeatureAudit();
  assert.equal(audit.families.length, 16);
  assert.ok(audit.families.every((family) => family.candidateLookbacks.length > 0));
  assert.ok(audit.families.some((family) => family.id === "liquidations"));
  assert.ok(audit.families.some((family) => family.id === "options-skew"));
  assert.ok(audit.families.some((family) => family.id === "news-gdelt"));
  assert.ok(audit.families.some((family) => family.id === "mempool"));
});

test("selected basis is evidence backed and unresolved sources are classified", () => {
  const audit = buildExternalFeatureAudit();
  assert.deepEqual(
    audit.selectedBasis.map((row) => row.id),
    ["spot-book-imbalance", "spot-trade-flow", "cross-market-returns"],
  );
  assert.ok(audit.provisionalBasis.some((row) => row.id === "spot-order-events"));
  assert.ok(audit.provisionalBasis.some((row) => row.id === "cross-exchange-books"));
  assert.ok(audit.provisionalBasis.some((row) => row.id === "liquidations"));
  assert.ok(audit.credentialRequired.some((row) => row.id === "macro-surprises"));
  assert.ok(audit.proxyOnly.some((row) => row.id === "onchain-flows"));
  assert.ok(audit.existingMatched15m.relativeChange > 0);
});

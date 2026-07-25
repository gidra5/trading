import assert from "node:assert/strict";
import { test } from "node:test";
import {
  pearsonCorrelation,
  rankValues,
  selectPortfolioBasis,
  type AssetReturnSeries,
} from "../src/portfolio-basis.js";

test("pivoted basis selects orthogonal assets before correlated duplicates", () => {
  const a = [1, -1, 1, -1, 1, -1, 1, -1];
  const b = [1, 1, -1, -1, 1, 1, -1, -1];
  const series: AssetReturnSeries[] = [
    asset("A", a),
    asset("A_COPY", a.map((value) => value * 3)),
    asset("A_INVERSE", a.map((value) => -value)),
    asset("B", b),
    asset("A_PLUS_B", a.map((value, index) => value + b[index])),
  ];

  const result = selectPortfolioBasis(series, {
    size: 2,
    anchorSymbol: "AUSDT",
  });

  assert.deepEqual(
    result.entries.map((entry) => entry.baseAsset),
    ["A", "B"],
  );
  assert.ok(result.entries[1].residualRatio > 0.999999);
  assert.ok(result.pairwiseMaxAbsCorrelation < 1e-12);
  for (const coverage of result.coverage) {
    assert.ok(coverage.rSquared > 0.999999, `${coverage.symbol} was not covered`);
  }
});

test("basis residual measures information beyond the entire selected span", () => {
  const a = [1, -1, 1, -1, 1, -1, 1, -1];
  const b = [1, 1, -1, -1, 1, 1, -1, -1];
  const c = [1, 1, 1, 1, -1, -1, -1, -1];
  const series = [
    asset("A", a),
    asset("MIX", a.map((value, index) => value + b[index])),
    asset("C", c),
    asset("B", b),
  ];

  const result = selectPortfolioBasis(series, {
    size: 3,
    anchorSymbol: "A",
  });

  assert.deepEqual(
    result.entries.map((entry) => entry.baseAsset),
    ["A", "C", "B"],
  );
  assert.ok(result.marketMinRSquared > 0.999999);
});

test("coverage sizing grows the basis until median and tail targets are reached", () => {
  const a = [1, -1, 1, -1, 1, -1, 1, -1];
  const b = [1, 1, -1, -1, 1, 1, -1, -1];
  const result = selectPortfolioBasis(
    [
      asset("A", a),
      asset("A_COPY", a.map((value) => value * 2)),
      asset("A_INVERSE", a.map((value) => -value)),
      asset("B", b),
      asset("MIX", a.map((value, index) => value + b[index])),
    ],
    {
      anchorSymbol: "A",
      minSize: 1,
      maxSize: 5,
      targetMedianRSquared: 0.99,
      targetP10RSquared: 0.99,
    },
  );

  assert.equal(result.sizingMode, "coverage");
  assert.equal(result.entries.length, 2);
  assert.equal(result.targetReached, true);
  assert.equal(result.coverageCurve.length, 2);
  assert.ok(result.marketP10RSquared > 0.999999);
});

test("pivot priority breaks only near-equivalent orthogonality choices", () => {
  const a = [1, -1, 1, -1, 1, -1, 1, -1];
  const b = [1, 1, -1, -1, 1, 1, -1, -1];
  const series = [
    asset("A", a),
    asset("B_LOW_RETURN", b),
    asset("B_HIGH_RETURN", b.map((value) => value * 4)),
    asset("CORRELATED_HIGH_RETURN", a.map((value, index) => value + b[index])),
  ];

  const result = selectPortfolioBasis(series, {
    size: 2,
    anchorSymbol: "A",
    pivotPriorityScores: [0, 1, 4, 100],
    residualEquivalenceBand: 0.01,
  });

  assert.deepEqual(
    result.entries.map((entry) => entry.baseAsset),
    ["A", "B_HIGH_RETURN"],
  );
  assert.equal(result.pivotPriorityMode, "score-within-residual-band");
  assert.equal(result.residualEquivalenceBand, 0.01);
});

test("rank transform averages tied ranks and supports Spearman selection", () => {
  assert.deepEqual(rankValues([30, 10, 10, 20]), [4, 1.5, 1.5, 3]);

  const result = selectPortfolioBasis(
    [
      asset("A", [1, 2, 3, 4, 5, 6]),
      asset("MONOTONIC", [1, 4, 9, 16, 25, 36]),
      asset("OTHER", [1, -1, 1, -1, 1, -1]),
    ],
    {
      size: 2,
      anchorSymbol: "A",
      correlationMethod: "spearman",
    },
  );

  assert.deepEqual(
    result.entries.map((entry) => entry.baseAsset),
    ["A", "OTHER"],
  );
  assert.ok(Math.abs(result.correlationMatrix[0][1] - 1) < 1e-12);
});

test("correlation rejects mismatched and zero-variance inputs", () => {
  assert.throws(() => pearsonCorrelation([1, 2], [1]), /equal lengths/);
  assert.throws(
    () =>
      selectPortfolioBasis(
        [asset("FLAT", [1, 1, 1]), asset("MOVE", [1, 2, 3])],
        { size: 1 },
      ),
    /zero return variance/,
  );
});

function asset(baseAsset: string, returns: number[]): AssetReturnSeries {
  return {
    symbol: `${baseAsset}USDT`,
    baseAsset,
    quoteAsset: "USDT",
    returns,
  };
}

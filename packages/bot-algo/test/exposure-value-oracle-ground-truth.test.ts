import assert from "node:assert/strict";
import test from "node:test";
import {
  prepareExposureValueOracle,
  type ExposureValueOracle,
  type ExposureValueOracleOptions,
} from "../src/exposure-value-distillation.js";
import {
  prepareExposureValueOracleCuda,
  vwKamaCudaStatus,
} from "../src/vw-kama-cuda.js";
import { prepareBruteForceExposureValueOracle } from "./exposure-value-oracle-reference.js";

interface OracleFixture {
  name: string;
  prices: readonly number[];
  options: ExposureValueOracleOptions;
}

const FIXTURES: readonly OracleFixture[] = [
  {
    name: "off-grid drift and frequent successor switches",
    prices: [100, 107, 94, 112, 89, 118, 103, 121],
    options: {
      scoreStartIndex: 0,
      holdingPeriodSteps: 1,
      valueHorizonSteps: 6,
      friction: 0.0075,
      gridSize: 9,
      minExposure: -1.5,
      maxExposure: 1.5,
      maxEffectiveExposure: 4,
      temperature: 0.025,
      quoteBorrowRate: 0.0003,
      assetBorrowRate: 0.0004,
    },
  },
  {
    name: "partial final holding block and nonzero score start",
    prices: [91, 96, 93, 104, 99, 111, 102, 115, 108, 119],
    options: {
      scoreStartIndex: 2,
      holdingPeriodSteps: 2,
      valueHorizonSteps: 5,
      friction: 0.0025,
      gridSize: 7,
      minExposure: -2,
      maxExposure: 2,
      maxEffectiveExposure: 4,
      temperature: 0.01,
      quoteBorrowRate: 0.0002,
      assetBorrowRate: 0.00025,
    },
  },
  {
    name: "leveraged actions crossing liquidation boundaries",
    prices: [100, 58, 143, 72, 160, 80, 155],
    options: {
      scoreStartIndex: 0,
      holdingPeriodSteps: 2,
      valueHorizonSteps: 5,
      friction: 0.004,
      gridSize: 9,
      minExposure: -4,
      maxExposure: 4,
      maxEffectiveExposure: 4,
      temperature: 0.04,
      quoteBorrowRate: 0.0002,
      assetBorrowRate: 0.0003,
    },
  },
];

test("optimized CPU oracle values and distributions match brute-force ground truth", () => {
  for (const fixture of FIXTURES) {
    const expected = prepareBruteForceExposureValueOracle(
      fixture.prices,
      fixture.options,
    );
    const actual = prepareExposureValueOracle(fixture.prices, {
      ...fixture.options,
      includeActionValues: true,
      includeProbabilities: true,
      includePath: false,
      distributionOnly: true,
    });
    assertOracleMatchesReference(actual, expected, fixture, 1e-12, 6e-8);
  }
});

test("finite negative returns remain valid while infeasible actions have zero mass", () => {
  const options: ExposureValueOracleOptions = {
    scoreStartIndex: 0,
    holdingPeriodSteps: 1,
    valueHorizonSteps: 1,
    friction: 0.001,
    gridSize: 5,
    minExposure: -2,
    maxExposure: 2,
    maxEffectiveExposure: 4,
    temperature: 0.02,
  };
  const oracle = prepareExposureValueOracle([100, 60], {
    ...options,
    includeActionValues: true,
    includeProbabilities: true,
    includePath: false,
    distributionOnly: true,
  });
  const longIndex = 3;
  const liquidatedLongIndex = 4;
  assert.ok(Number.isFinite(oracle.actionValues![longIndex]));
  assert.ok(oracle.actionValues![longIndex]! < 0);
  assert.ok(oracle.probabilities![longIndex]! > 0);
  assert.equal(
    oracle.actionValues![liquidatedLongIndex],
    Number.NEGATIVE_INFINITY,
  );
  assert.equal(oracle.probabilities![liquidatedLongIndex], 0);
});

test("CUDA oracle distributions match brute-force ground truth", async (context) => {
  const status = await vwKamaCudaStatus();
  if (!status.available) {
    context.skip(status.reason);
    return;
  }
  for (const fixture of FIXTURES) {
    const expected = prepareBruteForceExposureValueOracle(
      fixture.prices,
      fixture.options,
    );
    const { oracle: actual } = await prepareExposureValueOracleCuda(fixture.prices, {
      ...fixture.options,
      includeProbabilities: true,
      includePath: false,
      distributionOnly: true,
    });
    assertOracleProbabilitiesMatchReference(actual, expected, fixture, 1e-6);
  }
});

function assertOracleMatchesReference(
  actual: ExposureValueOracle,
  expected: ReturnType<typeof prepareBruteForceExposureValueOracle>,
  fixture: OracleFixture,
  actionValueTolerance: number,
  probabilityTolerance: number,
): void {
  assert.deepEqual(Array.from(actual.grid), Array.from(expected.grid), fixture.name);
  assert.ok(actual.actionValues, `${fixture.name}: action values were not retained`);
  assert.ok(actual.probabilities, `${fixture.name}: probabilities were not retained`);
  const firstComparedCell = fixture.options.scoreStartIndex * expected.grid.length;
  for (let cell = firstComparedCell; cell < expected.actionValues.length; cell += 1) {
    const expectedValue = expected.actionValues[cell]!;
    const actualValue = actual.actionValues[cell]!;
    if (!Number.isFinite(expectedValue) || !Number.isFinite(actualValue)) {
      assert.equal(actualValue, expectedValue, `${fixture.name}: action value cell ${cell}`);
    } else {
      assert.ok(
        Math.abs(actualValue - expectedValue) <= actionValueTolerance,
        `${fixture.name}: action value cell ${cell} differs: ${actualValue} vs ${expectedValue}`,
      );
    }
    assert.ok(
      Math.abs(actual.probabilities[cell]! - expected.probabilities[cell]!)
        <= probabilityTolerance,
      `${fixture.name}: probability cell ${cell} differs: `
        + `${actual.probabilities[cell]} vs ${expected.probabilities[cell]}`,
    );
    if (expectedValue === Number.NEGATIVE_INFINITY) {
      assert.equal(
        expected.probabilities[cell],
        0,
        `${fixture.name}: reference infeasible action cell ${cell}`,
      );
      assert.equal(
        actual.probabilities[cell],
        0,
        `${fixture.name}: CPU infeasible action cell ${cell}`,
      );
    }
  }
}

function assertOracleProbabilitiesMatchReference(
  actual: ExposureValueOracle,
  expected: ReturnType<typeof prepareBruteForceExposureValueOracle>,
  fixture: OracleFixture,
  probabilityTolerance: number,
): void {
  assert.deepEqual(Array.from(actual.grid), Array.from(expected.grid), fixture.name);
  assert.ok(actual.probabilities, `${fixture.name}: probabilities were not retained`);
  const firstComparedCell = fixture.options.scoreStartIndex * expected.grid.length;
  for (let cell = firstComparedCell; cell < expected.probabilities.length; cell += 1) {
    assert.ok(
      Math.abs(actual.probabilities[cell]! - expected.probabilities[cell]!)
        <= probabilityTolerance,
      `${fixture.name}: probability cell ${cell} differs: `
        + `${actual.probabilities[cell]} vs ${expected.probabilities[cell]}`,
    );
    if (expected.actionValues[cell] === Number.NEGATIVE_INFINITY) {
      assert.equal(
        actual.probabilities[cell],
        0,
        `${fixture.name}: CUDA infeasible action cell ${cell}`,
      );
    }
  }
}

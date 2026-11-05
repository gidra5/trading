import assert from "node:assert/strict";
import test from "node:test";
import { estimateBookCost, type Book } from "../src/clients/binance-common.js";

const book: Book = {
  bids: [
    [99, 1],
    [98, 1],
  ],
  asks: [
    [101, 1],
    [103, 1],
  ],
};

test("explicit sizes bypass averaging and walk partial book levels", () => {
  const rules = { min: 0.5, max: 3, step: 0.5 };
  assert.ok(Math.abs(estimateBookCost(book, rules, 1) - 0.01) < 1e-12);
  assert.ok(
    Math.abs(estimateBookCost(book, rules, 1.5) - (0.03 - 0.02 / 1.5)) < 1e-12,
  );
  assert.ok(Math.abs(estimateBookCost(book, rules, 2) - 0.02) < 1e-12);
  assert.throws(() => estimateBookCost(book, rules, 2.5), /Insufficient/);
  for (const size of [0, -1, NaN, Infinity]) {
    assert.throws(() => estimateBookCost(book, rules, size), /positive/);
  }
  for (const size of [0.1, 1.1, 4]) {
    assert.throws(
      () => estimateBookCost(book, rules, size),
      /quantity limits and step/,
    );
  }
});

test("small lot grids average every valid size with exponential weights", () => {
  const expected = (0.01 + 0.02 * Math.exp(-5)) / (1 + Math.exp(-5));
  const cost = estimateBookCost(book, { min: 1, max: 2, step: 1 });
  assert.ok(Math.abs(cost - expected) < 1e-12);
  assert.ok(
    cost < 0.015,
    "smaller sizes get more weight than in a uniform average",
  );
});

test("configurable decay supports uniform and increasingly small-size-weighted averages", () => {
  const rules = { min: 1, max: 2, step: 1 };
  for (const rate of [0, 1, 5, 20]) {
    const expected = (0.01 + 0.02 * Math.exp(-rate)) / (1 + Math.exp(-rate));
    assert.ok(
      Math.abs(estimateBookCost(book, rules, undefined, rate) - expected) <
        1e-12,
    );
    assert.ok(Math.abs(estimateBookCost(book, rules, 2, rate) - 0.02) < 1e-12);
  }
  const continuousUniform = 0.03 - 0.02 * Math.log(2);
  for (const rate of [0, Number.MIN_VALUE]) {
    assert.ok(
      Math.abs(
        estimateBookCost(book, { ...rules, step: null }, undefined, rate) -
          continuousUniform,
      ) < 1e-7,
    );
    assert.ok(
      Math.abs(
        estimateBookCost(book, { ...rules, step: 0.001 }, undefined, rate) -
          continuousUniform,
      ) < 1e-5,
    );
  }
  for (const rate of [-1, NaN, Infinity]) {
    assert.throws(
      () => estimateBookCost(book, rules, undefined, rate),
      /frictionDecayRate/,
    );
  }
});

test("continuous quantiles agree with integration of the exponential density", () => {
  // On [1,2], ask VWAP = 103 - 2/size and mid = 100.
  const density = (size: number) => Math.exp(-5 * (size - 1));
  const cost = (size: number) => 0.03 - 0.02 / size;
  const steps = 20000;
  let weighted = 0,
    mass = 0;
  for (let index = 0; index <= steps; index++) {
    const size = 1 + index / steps;
    const weight = density(size) * (index === 0 || index === steps ? 0.5 : 1);
    weighted += weight * cost(size);
    mass += weight;
  }
  const actual = estimateBookCost(book, { min: 1, max: 2, step: null });
  assert.ok(Math.abs(actual - weighted / mass) < 1e-6);
});

test("large lot grids agree with an exhaustive weighted sum", () => {
  let weighted = 0,
    mass = 0;
  for (let index = 0; index <= 1000; index++) {
    const size = 1 + index / 1000;
    const weight = Math.exp((-5 * index) / 1000);
    weighted += weight * (0.03 - 0.02 / size);
    mass += weight;
  }
  const rules = { min: 1, max: 2, step: 0.001 };
  const actual = estimateBookCost(book, rules);
  assert.ok(Math.abs(actual - weighted / mass) < 1e-6);
  assert.equal(estimateBookCost(book, rules), actual);
});

test("range respects visible depth on both sides and the quantity grid", () => {
  const asymmetric: Book = {
    bids: book.bids,
    asks: [
      [101, 1],
      [103, 9],
    ],
  };
  const expected = estimateBookCost(book, { min: 1, max: 2, step: 1 });
  assert.equal(
    estimateBookCost(asymmetric, { min: 0.1, max: 100, step: 1 }),
    expected,
  );
  assert.equal(
    estimateBookCost(asymmetric, { min: 1, max: null, step: 1 }),
    expected,
  );
  assert.ok(
    Math.abs(estimateBookCost(book, { min: 1.1, max: 2, step: 1 }) - 0.02) <
      1e-12,
  );
  assert.ok(
    Math.abs(estimateBookCost(book, { min: 1, max: 1.9, step: 1 }) - 0.01) <
      1e-12,
  );
});

test("single-level books retain constant spread at every size", () => {
  assert.ok(
    Math.abs(
      estimateBookCost(
        { bids: [[99, 10]], asks: [[101, 10]] },
        { min: null, max: null, step: null },
      ) - 0.01,
    ) < 1e-12,
  );
});

test("invalid books and unobservable size ranges fail explicitly", () => {
  const rules = { min: 1, max: 2, step: 1 };
  assert.throws(
    () => estimateBookCost({ bids: [], asks: book.asks }, rules),
    /empty/,
  );
  assert.throws(
    () => estimateBookCost({ bids: [[102, 1]], asks: [[101, 1]] }, rules),
    /Crossed/,
  );
  assert.throws(
    () =>
      estimateBookCost(
        {
          bids: [
            [99, 1],
            [100, 1],
          ],
          asks: book.asks,
        },
        rules,
      ),
    /Unsorted/,
  );
  assert.throws(
    () => estimateBookCost(book, { min: 3, max: 10, step: 1 }),
    /No valid order sizes/,
  );
  assert.throws(
    () => estimateBookCost(book, { min: 1, max: 2, step: 0 }),
    /quantity step/,
  );
});

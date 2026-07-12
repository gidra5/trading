import assert from "node:assert/strict";
import test from "node:test";
import { KamaRateNoise } from "../src/index.js";

test("KAMA rate noise stays low for a smooth rate and reacts to reversals", () => {
  const noise = new KamaRateNoise(3);
  noise.update(10);
  assert.equal(noise.update(10), 0);
  assert.equal(noise.update(-10), 10);
  assert.equal(noise.update(-10), 5);
});

test("KAMA rate noise restores exactly", () => {
  const source = new KamaRateNoise(5);
  source.update(2);
  source.update(-4);
  const restored = new KamaRateNoise(5);
  restored.restore(source.snapshot());
  assert.equal(restored.update(3), source.update(3));
});

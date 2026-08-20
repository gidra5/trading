import assert from "node:assert/strict";
import test from "node:test";
import { conditionalGain } from "./analyze-public-external-feature-information.ts";

test("conditional gain is positive for a genuinely informative candidate", () => {
  const count = 4_000;
  const base = new Uint8Array(count);
  const feature = new Uint8Array(count);
  const targets = new Uint8Array(count);
  const training: number[] = [];
  const evaluation: number[] = [];
  for (let index = 0; index < count; index += 1) {
    feature[index] = index % 4;
    targets[index] = feature[index]! < 2 ? 0 : 1;
    (index < count / 2 ? training : evaluation).push(index);
  }
  const gain = conditionalGain({ base, baseContexts: 1, featureBins: [feature] }, training, evaluation, [], 0, targets, 2);
  assert.ok(gain > 0.9);
});

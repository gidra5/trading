const WEIGHT_TOLERANCE = 1e-12;

/**
 * Convert non-negative constituent size measures into proportional weights,
 * iteratively redistributing excess from constituents that breach the cap.
 *
 * This is the single-constituent capping pattern used by capped
 * capitalization-weighted indices. All-zero inputs fall back to equal weight.
 */
export function cappedProportionalWeights(
  sizes: readonly number[],
  maximumWeight = 0.05,
): number[] {
  if (sizes.length === 0) {
    throw new Error("Index weighting requires at least one constituent.");
  }
  if (
    !Number.isFinite(maximumWeight) ||
    maximumWeight <= 0 ||
    maximumWeight > 1
  ) {
    throw new Error("Maximum constituent weight must be in (0, 1].");
  }
  if (maximumWeight * sizes.length < 1 - WEIGHT_TOLERANCE) {
    throw new Error(
      `A ${(maximumWeight * 100).toFixed(2)}% cap is infeasible for ` +
        `${sizes.length} constituents.`,
    );
  }
  for (const size of sizes) {
    if (!Number.isFinite(size) || size < 0) {
      throw new Error(
        "Index constituent sizes must be finite non-negative numbers.",
      );
    }
  }

  const weights = new Array<number>(sizes.length).fill(0);
  let active = sizes.map((_, index) => index);
  let remainingWeight = 1;

  while (active.length > 0) {
    const activeSize = active.reduce(
      (total, index) => total + sizes[index],
      0,
    );
    const proposedWeight = (index: number): number =>
      activeSize > 0
        ? remainingWeight * (sizes[index] / activeSize)
        : remainingWeight / active.length;
    const breaches = active.filter(
      (index) => proposedWeight(index) > maximumWeight + WEIGHT_TOLERANCE,
    );

    if (breaches.length === 0) {
      for (const index of active) {
        weights[index] = proposedWeight(index);
      }
      break;
    }

    const breached = new Set(breaches);
    for (const index of breaches) {
      weights[index] = maximumWeight;
      remainingWeight -= maximumWeight;
    }
    active = active.filter((index) => !breached.has(index));
  }

  const difference =
    1 - weights.reduce((total, weight) => total + weight, 0);
  if (Math.abs(difference) > WEIGHT_TOLERANCE) {
    const adjustable = weights.findIndex(
      (weight) =>
        weight + difference >= -WEIGHT_TOLERANCE &&
        weight + difference <= maximumWeight + WEIGHT_TOLERANCE,
    );
    if (adjustable < 0) {
      throw new Error("Unable to normalize capped constituent weights.");
    }
    weights[adjustable] += difference;
  }
  return weights;
}

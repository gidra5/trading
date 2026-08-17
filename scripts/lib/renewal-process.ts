export function stationaryRenewalCountDistribution(
  interarrival: readonly number[],
  horizon: number,
): number[] {
  if (!Number.isInteger(horizon) || horizon < 1) throw new Error("Horizon must be positive.");
  assertProbabilityMass(interarrival, "interarrival");
  const mean = interarrival.reduce(
    (sum, probability, delay) => sum + probability * delay,
    0,
  );
  if (!(mean > 0)) throw new Error("Interarrival mean must be positive.");
  const survival = Array.from({ length: Math.max(interarrival.length, horizon + 2) }, () => 0);
  let tail = 0;
  for (let delay = interarrival.length - 1; delay >= 1; delay -= 1) {
    tail += interarrival[delay] ?? 0;
    survival[delay] = tail;
  }
  const afterEvent = Array.from(
    { length: horizon + 1 },
    () => Array.from({ length: horizon + 1 }, () => 0),
  );
  afterEvent[0]![0] = 1;
  for (let seconds = 1; seconds <= horizon; seconds += 1) {
    afterEvent[seconds]![0] = survival[seconds + 1] ?? 0;
    for (let delay = 1; delay <= seconds; delay += 1) {
      const probability = interarrival[delay] ?? 0;
      if (probability === 0) continue;
      for (let count = 1; count <= seconds; count += 1) {
        afterEvent[seconds]![count]! += probability * afterEvent[seconds - delay]![count - 1]!;
      }
    }
  }
  const result = Array.from({ length: horizon + 1 }, () => 0);
  for (let firstDelay = 1; firstDelay <= horizon; firstDelay += 1) {
    const firstProbability = (survival[firstDelay] ?? 0) / mean;
    for (let remainingCount = 0; remainingCount < horizon; remainingCount += 1) {
      result[remainingCount + 1]! += firstProbability
        * afterEvent[horizon - firstDelay]![remainingCount]!;
    }
  }
  for (let firstDelay = horizon + 1; firstDelay < survival.length; firstDelay += 1) {
    result[0]! += (survival[firstDelay] ?? 0) / mean;
  }
  assertProbabilityMass(result, "stationary renewal count");
  return result;
}

function assertProbabilityMass(probabilities: ArrayLike<number>, label: string): void {
  let mass = 0;
  for (let index = 0; index < probabilities.length; index += 1) mass += probabilities[index]!;
  if (Math.abs(mass - 1) > 1e-8) throw new Error(`${label} has probability mass ${mass}.`);
}

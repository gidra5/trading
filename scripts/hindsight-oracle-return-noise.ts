export interface OracleReturnNoiseConfig {
  /** Exact Pearson correlation of rolling log returns at the configured horizon. */
  correlation: number;
  /** Deterministic sample seed. */
  seed: number;
  /** Number of source returns in each rolling correlation horizon. */
  rollingHorizonSteps: number;
}

interface CloseCandle {
  close: number;
  closeTime: number;
}

export function fillOracleClosePrices(
  target: Float64Array,
  candles: readonly CloseCandle[],
  future: readonly CloseCandle[],
  noise?: OracleReturnNoiseConfig,
): void {
  if (target.length !== candles.length + future.length) {
    throw new Error("Oracle price target length does not match its candle sources.");
  }
  if (target.length === 0) return;
  if (noise) validateOracleReturnNoise(noise);

  const candleAt = (index: number): CloseCandle => index < candles.length
    ? candles[index]!
    : future[index - candles.length]!;
  const first = candleAt(0);
  validateClose(first.close);
  target[0] = first.close;

  if (!noise || noise.correlation === 1) {
    for (let index = 1; index < target.length; index += 1) {
      const candle = candleAt(index);
      validateClose(candle.close);
      target[index] = candle.close;
    }
    return;
  }

  const returns = new Float64Array(target.length - 1);
  let meanReturn = 0;
  for (let index = 1; index < target.length; index += 1) {
    const previous = candleAt(index - 1);
    const candle = candleAt(index);
    validateClose(previous.close);
    validateClose(candle.close);
    const value = Math.log(candle.close / previous.close);
    returns[index - 1] = value;
    meanReturn += value;
  }
  meanReturn /= Math.max(1, returns.length);

  const centeredReturns = Float64Array.from(
    returns,
    (value) => value - meanReturn,
  );
  // Start from a time-shuffled version of the observed returns, then remove its
  // projection onto the real rolling-horizon returns. After matching rolling
  // variances, the two components are orthogonal and the requested mixture
  // coefficient is therefore the exact rolling-horizon Pearson correlation.
  const shuffledReturns = centeredReturns.slice();
  shuffleReturns(
    shuffledReturns,
    noise.seed,
    candleAt(0).closeTime,
    candleAt(target.length - 1).closeTime,
  );
  const calibration = rollingNoiseCalibration(
    centeredReturns,
    shuffledReturns,
    noise.rollingHorizonSteps,
  );
  const randomWeight = Math.sqrt(1 - noise.correlation ** 2);
  for (let index = 1; index < target.length; index += 1) {
    const candle = candleAt(index);
    const realReturn = centeredReturns[index - 1]!;
    const orthogonalNoise = calibration.scale * (
      shuffledReturns[index - 1]! - calibration.projection * realReturn
    );
    const perturbedReturn = meanReturn
      + noise.correlation * realReturn
      + randomWeight * orthogonalNoise;
    const price = target[index - 1]! * Math.exp(perturbedReturn);
    if (!(price > 0) || !Number.isFinite(price)) {
      throw new Error(`Perturbed oracle price is invalid at ${candle.closeTime}.`);
    }
    target[index] = price;
  }
}

export function validateOracleReturnNoise(noise: OracleReturnNoiseConfig): void {
  if (!(noise.correlation >= 0 && noise.correlation <= 1)) {
    throw new Error("Oracle return correlation must be in [0, 1].");
  }
  if (!Number.isSafeInteger(noise.seed) || noise.seed < 0) {
    throw new Error("Oracle return-noise seed must be a non-negative safe integer.");
  }
  if (!Number.isSafeInteger(noise.rollingHorizonSteps) || noise.rollingHorizonSteps < 1) {
    throw new Error("Oracle return-noise rolling horizon must be a positive safe integer.");
  }
}

function rollingNoiseCalibration(
  real: Float64Array,
  shuffled: Float64Array,
  horizon: number,
): { projection: number; scale: number } {
  if (horizon > real.length) {
    throw new Error(
      `Oracle return-noise rolling horizon ${horizon} exceeds ${real.length} returns.`,
    );
  }
  const realRolling = rollingSums(real, horizon);
  const shuffledRolling = rollingSums(shuffled, horizon);
  const realMean = mean(realRolling);
  const shuffledMean = mean(shuffledRolling);
  let realVariance = 0;
  let shuffledVariance = 0;
  let covariance = 0;
  for (let index = 0; index < realRolling.length; index += 1) {
    const realDelta = realRolling[index]! - realMean;
    const shuffledDelta = shuffledRolling[index]! - shuffledMean;
    realVariance += realDelta ** 2;
    shuffledVariance += shuffledDelta ** 2;
    covariance += realDelta * shuffledDelta;
  }
  if (!(realVariance > 0)) {
    throw new Error("Real rolling oracle returns have no variance for noise calibration.");
  }
  const projection = covariance / realVariance;
  const orthogonalVariance = shuffledVariance - covariance ** 2 / realVariance;
  if (!(orthogonalVariance > Number.EPSILON * shuffledVariance)) {
    throw new Error("Shuffled rolling oracle returns are degenerate after orthogonalization.");
  }
  return {
    projection,
    scale: Math.sqrt(realVariance / orthogonalVariance),
  };
}

function rollingSums(values: Float64Array, horizon: number): Float64Array {
  const output = new Float64Array(values.length - horizon + 1);
  let rolling = 0;
  for (let index = 0; index < horizon; index += 1) rolling += values[index]!;
  output[0] = rolling;
  for (let index = 1; index < output.length; index += 1) {
    rolling += values[index + horizon - 1]! - values[index - 1]!;
    output[index] = rolling;
  }
  return output;
}

function mean(values: Float64Array): number {
  return values.reduce((total, value) => total + value, 0) / values.length;
}

function shuffleReturns(
  returns: Float64Array,
  seed: number,
  firstTimestamp: number,
  lastTimestamp: number,
): void {
  let state = hash(seed, firstTimestamp);
  state = hash(state, lastTimestamp);
  for (let index = returns.length - 1; index > 0; index -= 1) {
    state = nextRandom(state);
    const swapIndex = state % (index + 1);
    const value = returns[index]!;
    returns[index] = returns[swapIndex]!;
    returns[swapIndex] = value;
  }
}

function hash(seed: number, timestamp: number): number {
  const low = timestamp >>> 0;
  const high = Math.floor(timestamp / 0x1_0000_0000) >>> 0;
  let value = (seed ^ low ^ Math.imul(high, 0x9e37_79b1)) >>> 0;
  value = Math.imul(value ^ value >>> 16, 0x85eb_ca6b) >>> 0;
  value = Math.imul(value ^ value >>> 13, 0xc2b2_ae35) >>> 0;
  value ^= value >>> 16;
  return value >>> 0 || 0x9e37_79b9;
}

function nextRandom(value: number): number {
  value ^= value << 13;
  value ^= value >>> 17;
  value ^= value << 5;
  return value >>> 0 || 0x9e37_79b9;
}

function validateClose(close: number): void {
  if (!(close > 0) || !Number.isFinite(close)) {
    throw new Error("Oracle return noise requires finite positive closes.");
  }
}

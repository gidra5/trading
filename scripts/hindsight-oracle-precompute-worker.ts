import { parentPort, workerData } from "node:worker_threads";
import {
  exposureValueOracleActionDistribution,
  type ExposureValueOracleOptions,
} from "@trading/bot-algo";

interface OracleWorkerData {
  prices: SharedArrayBuffer;
  priceCount: number;
  scoredLength: number;
  probabilities: SharedArrayBuffer;
  feasibleActionCounts: SharedArrayBuffer;
  loaded: SharedArrayBuffer;
  cursor: SharedArrayBuffer;
  completed: SharedArrayBuffer;
  decisionCount: number;
  holdingPeriodSteps: number;
  valueHorizonSteps: number;
  usableIndexes: number[];
  outputColumns: number;
  options: ExposureValueOracleOptions;
}

const data = workerData as OracleWorkerData;
const prices = new Float64Array(data.prices, 0, data.priceCount);
const probabilities = new Float32Array(data.probabilities);
const feasibleActionCounts = new Uint16Array(data.feasibleActionCounts);
const loaded = new Uint8Array(data.loaded);
const cursor = new Int32Array(data.cursor);
const completed = new Int32Array(data.completed);

while (true) {
  const row = Atomics.add(cursor, 0, 1);
  if (row >= data.decisionCount) break;
  if (loaded[row]) continue;
  const candleIndex = row * data.holdingPeriodSteps;
  const terminalIndex = Math.min(prices.length - 1, candleIndex + data.valueHorizonSteps);
  const horizonPrices = prices.subarray(candleIndex, terminalIndex + 1);
  const distribution = exposureValueOracleActionDistribution(horizonPrices, {
    ...data.options,
    holdingPeriodSteps: Math.min(data.holdingPeriodSteps, horizonPrices.length - 1),
    terminalIndex: horizonPrices.length - 1,
  });
  const outputOffset = row * data.outputColumns;
  let total = 0;
  for (let column = 0; column < data.usableIndexes.length; column += 1) {
    const probability = distribution.probabilities[data.usableIndexes[column]!]!;
    probabilities[outputOffset + column] = probability;
    total += probability;
  }
  if (!(total > 0)) throw new Error(`Computed oracle row ${row} has no usable probability mass.`);
  let feasibleActionCount = 0;
  for (let column = 0; column < data.outputColumns; column += 1) {
    const cell = outputOffset + column;
    probabilities[cell] /= total;
    if (probabilities[cell]! > 0) feasibleActionCount += 1;
  }
  feasibleActionCounts[row] = feasibleActionCount;
  loaded[row] = 1;
  Atomics.add(completed, 0, 1);
}

parentPort?.close();

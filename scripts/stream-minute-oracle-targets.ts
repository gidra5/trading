import fs from "node:fs/promises";
import path from "node:path";
import readline from "node:readline";
import { fileURLToPath } from "node:url";
import { prepareExposureValueOracle } from "@trading/bot-algo";
import { MlpFeatureStore } from "../apps/server/src/mlp-feature-store.js";

const MINUTE_MS = 60_000;
const DAY_MS = 86_400_000;

interface Plan {
  dataDir: string;
  execution: {
    feeBps: number;
    minimumEffectiveExposure: number;
    maximumEffectiveExposure: number;
    maintenanceBpsHour: {
      quoteLend: number;
      quoteBorrow: number;
      assetBorrow: number;
    };
    gridSize: number;
    temperature: number;
  };
}

void main();

async function main(): Promise<void> {
  const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const planIndex = process.argv.indexOf("--plan");
  if (planIndex < 0 || !process.argv[planIndex + 1]) {
    throw new Error("Usage: stream-minute-oracle-targets.ts --plan <training-plan.json>");
  }
  const planFile = path.resolve(process.argv[planIndex + 1]!);
  const plan = JSON.parse(await fs.readFile(planFile, "utf8")) as Plan;
  const featureStore = new MlpFeatureStore(path.resolve(repoRoot, plan.dataDir));
  const input = readline.createInterface({ input: process.stdin });

  for await (const line of input) {
    if (!line) continue;
    const day = Date.parse(line);
    if (!Number.isFinite(day) || day % DAY_MS !== 0) {
      throw new Error(`Invalid UTC target date '${line}'.`);
    }
    const minutes = await featureStore.loadCompletedMinuteRange(
      day - MINUTE_MS,
      day + DAY_MS + 60 * MINUTE_MS,
    );
    const expectedMinutes = 1 + 1_440 + 60;
    if (minutes.length !== expectedMinutes
      || minutes.some((candle, index) =>
        candle.openTime !== day - MINUTE_MS + index * MINUTE_MS)) {
      throw new Error(
        `${line} one-minute oracle source is incomplete: `
        + `${minutes.length}/${expectedMinutes} contiguous minutes.`,
      );
    }
    const oracle = prepareExposureValueOracle(
      Float64Array.from(minutes, (candle) => candle.close),
      {
        scoreStartIndex: 0,
        holdingPeriodSteps: 1,
        valueHorizonSteps: 60,
        friction: plan.execution.feeBps / 10_000,
        gridSize: plan.execution.gridSize,
        minExposure: plan.execution.minimumEffectiveExposure,
        maxExposure: plan.execution.maximumEffectiveExposure,
        maxEffectiveExposure: Math.max(
          Math.abs(plan.execution.minimumEffectiveExposure),
          Math.abs(plan.execution.maximumEffectiveExposure),
        ),
        terminalIndex: 1_440,
        temperature: plan.execution.temperature,
        opportunityEpsilon: 0,
        quoteLendRate: bpsHourToPerMinute(
          plan.execution.maintenanceBpsHour.quoteLend,
        ),
        quoteBorrowRate: bpsHourToPerMinute(
          plan.execution.maintenanceBpsHour.quoteBorrow,
        ),
        assetBorrowRate: bpsHourToPerMinute(
          plan.execution.maintenanceBpsHour.assetBorrow,
        ),
        includeActionValues: false,
        includeProbabilities: true,
      },
    );
    const probabilities = oracle.probabilities;
    if (!probabilities) {
      throw new Error(`One-minute oracle probabilities are missing for ${line}.`);
    }
    const rows = probabilities.length / oracle.grid.length;
    const header = Buffer.allocUnsafe(8);
    header.writeUInt32LE(rows, 0);
    header.writeUInt32LE(oracle.grid.length, 4);
    process.stdout.write(header);
    process.stdout.write(Buffer.from(
      probabilities.buffer,
      probabilities.byteOffset,
      probabilities.byteLength,
    ));
  }
}

function bpsHourToPerMinute(value: number): number {
  return Math.expm1(Math.log1p(value / 10_000) / 60);
}

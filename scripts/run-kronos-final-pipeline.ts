import fs from "node:fs";
import path from "node:path";
import crypto from "node:crypto";
import { spawn } from "node:child_process";

const REPO_ROOT = path.resolve(import.meta.dirname, "..");

interface ForecastIdentity {
  contract: string;
  runSignature: string;
  modelId: string;
  rows: unknown[];
}

export interface PolicyIdentity {
  contract: string;
  selectionContract: string;
  forecastRunSignature: string;
  selectedPolicyId: string;
  selection: {
    selectionEligible: boolean;
    geometricMeanReturnPct: number;
    tradeCount: number;
    liquidatedPositionCount: number;
  };
  postTrainingConfirmation: {
    selectionEligible: boolean;
    geometricMeanReturnPct: number;
    tradeCount: number;
    liquidatedPositionCount: number;
  };
}

export interface ValidationEpisodeIdentity {
  windowId: string;
  startTime: number;
  endTime: number;
  forecastRows: number;
  forecastRowsConsumed: number;
  summary: {
    netPnl: number;
    returnPct: number;
    tradeCount: number;
    feesPaid: number;
    maintenancePaid: number;
    liquidatedPositionCount: number;
  };
  fills: Array<{
    side: "buy" | "sell";
    price: number;
    quantity: number;
    quoteQuantity: number;
    feeQuote: number;
    realizedPnl: number;
    filledAt: number;
    reason: string;
    liquidation?: boolean;
  }>;
}

export interface ValidationIdentity {
  version: number;
  contract: string;
  phase: string;
  forecast: { runSignature: string; sha256: string };
  policyArtifact: { sha256: string };
  policyId: string;
  split: { windowIds: string[] };
  aggregate: {
    windows: number;
    netPnl: number;
    geometricMeanReturnPct: number;
    tradeCount: number;
    feesPaid: number;
    maintenancePaid: number;
    liquidatedPositionCount: number;
    forecastRows: number;
    forecastRowsConsumed: number;
  };
  gates: {
    allForecastsConsumed: boolean;
    actualFills: boolean;
    noLiquidations: boolean;
    positiveNetPnl: boolean;
  };
  episodes: ValidationEpisodeIdentity[];
  controls: Array<{ id: string }>;
}

function argument(name: string, fallback: string): string {
  const prefix = `--${name}=`;
  const inline = process.argv.slice(2).find((value) => value.startsWith(prefix));
  if (inline) return inline.slice(prefix.length);
  const index = process.argv.indexOf(`--${name}`);
  return index >= 0 && process.argv[index + 1] ? process.argv[index + 1]! : fallback;
}

function resolveArgument(name: string, fallback: string): string {
  return path.resolve(REPO_ROOT, argument(name, fallback));
}

function loadJson<T>(file: string): T {
  return JSON.parse(fs.readFileSync(file, "utf8")) as T;
}

async function runNpm(script: string, args: readonly string[]): Promise<void> {
  console.log(`PIPELINE START npm run ${script} -- ${args.join(" ")}`);
  const npmEntryPoint = process.env.npm_execpath;
  const useNodeEntryPoint = process.platform === "win32"
    && npmEntryPoint !== undefined
    && fs.existsSync(npmEntryPoint);
  const command = useNodeEntryPoint ? process.execPath : "npm";
  const commandArgs = [
    ...(useNodeEntryPoint ? [npmEntryPoint] : []),
    "run", script, "--", ...args,
  ];
  const exitCode = await new Promise<number>((resolve, reject) => {
    const child = spawn(command, commandArgs, {
      cwd: REPO_ROOT,
      stdio: "inherit",
      windowsHide: true,
    });
    child.once("error", reject);
    child.once("exit", (code, signal) => {
      if (signal) reject(new Error(`${script} terminated by ${signal}.`));
      else resolve(code ?? 1);
    });
  });
  if (exitCode !== 0) throw new Error(`${script} exited with code ${exitCode}.`);
  console.log(`PIPELINE COMPLETE ${script}`);
}

export function assertPolicyGate(policy: PolicyIdentity, runSignature: string): void {
  if (policy.contract !== "kronos-bot-policy-v3"
    || policy.selectionContract
      !== "pretraining-ranking-with-untouched-post-training-confirmation-v2"
    || policy.forecastRunSignature !== runSignature) {
    throw new Error("Frozen policy does not match the completed forecast run.");
  }
  for (const [label, aggregate] of [
    ["broad calibration", policy.selection],
    ["post-training confirmation", policy.postTrainingConfirmation],
  ] as const) {
    if (!aggregate?.selectionEligible
      || !(aggregate.geometricMeanReturnPct > 0)
      || !(aggregate.tradeCount > 0)
      || aggregate.liquidatedPositionCount !== 0) {
      throw new Error(`Frozen policy failed its ${label} gate.`);
    }
  }
}

function closeEnough(left: number, right: number): boolean {
  return Math.abs(left - right) <= 1e-8 * Math.max(1, Math.abs(left), Math.abs(right));
}

export function assertValidationReport(
  validation: ValidationIdentity,
  runSignature: string,
  expectedPolicyId?: string,
  expectedForecastSha256?: string,
  expectedPolicyArtifactSha256?: string,
): void {
  if (validation.version !== 3
    || validation.contract !== "kronos-bot-backtest-v3"
    || validation.phase !== "validation"
    || validation.forecast.runSignature !== runSignature
    || (expectedPolicyId !== undefined && validation.policyId !== expectedPolicyId)
    || (expectedForecastSha256 !== undefined
      && validation.forecast.sha256 !== expectedForecastSha256)
    || (expectedPolicyArtifactSha256 !== undefined
      && validation.policyArtifact?.sha256 !== expectedPolicyArtifactSha256)) {
    throw new Error("Canonical validation report failed its identity check.");
  }
  if (validation.episodes.length !== 6
    || validation.aggregate.windows !== 6
    || validation.split.windowIds.length !== 8) {
    throw new Error("Validation report does not cover the expected 8 windows / 6 episodes.");
  }
  const fillCount = validation.episodes.reduce(
    (total, episode) => total + episode.fills.length,
    0,
  );
  const summaryTradeCount = validation.episodes.reduce(
    (total, episode) => total + episode.summary.tradeCount,
    0,
  );
  const feesPaid = validation.episodes.reduce(
    (total, episode) => total + episode.summary.feesPaid,
    0,
  );
  const maintenancePaid = validation.episodes.reduce(
    (total, episode) => total + episode.summary.maintenancePaid,
    0,
  );
  for (const episode of validation.episodes) {
    if (episode.fills.length !== episode.summary.tradeCount
      || episode.forecastRowsConsumed !== episode.forecastRows) {
      throw new Error(`Validation episode ${episode.windowId} has inconsistent execution evidence.`);
    }
    for (const fill of episode.fills) {
      if (!Number.isFinite(fill.price) || !(fill.price > 0)
        || !Number.isFinite(fill.quantity) || !(fill.quantity > 0)
        || !Number.isFinite(fill.quoteQuantity) || !(fill.quoteQuantity > 0)
        || !Number.isFinite(fill.feeQuote) || fill.feeQuote < 0
        || !Number.isFinite(fill.realizedPnl)
        || !Number.isInteger(fill.filledAt)
        || fill.filledAt < episode.startTime || fill.filledAt >= episode.endTime
        || !fill.reason) {
        throw new Error(`Validation episode ${episode.windowId} contains an invalid fill.`);
      }
    }
  }
  const controlIds = new Set(validation.controls.map((control) => control.id));
  if (controlIds.size !== 4
    || !["constant-long-1x", "constant-short-1x", "constant-long-5x", "constant-short-5x"]
      .every((id) => controlIds.has(id))) {
    throw new Error("Validation report is missing a directional control.");
  }
  if (fillCount === 0
    || fillCount !== summaryTradeCount
    || fillCount !== validation.aggregate.tradeCount
    || !closeEnough(feesPaid, validation.aggregate.feesPaid)
    || !closeEnough(maintenancePaid, validation.aggregate.maintenancePaid)
    || validation.aggregate.forecastRowsConsumed !== validation.aggregate.forecastRows
    || validation.aggregate.liquidatedPositionCount !== 0
    || !validation.gates.allForecastsConsumed
    || !validation.gates.actualFills
    || !validation.gates.noLiquidations
    || validation.gates.positiveNetPnl !== (validation.aggregate.netPnl > 0)) {
    throw new Error("Validation report failed its fill, cost, consumption, or risk audit.");
  }
}

async function main(): Promise<void> {
  const checkpoint = resolveArgument(
    "checkpoint",
    ".tools/Kronos-finetuned/btcusdt-1m-base-policy-holdout-v2/best_model",
  );
  const forecastFile = resolveArgument(
    "forecasts",
    "data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-forecasts.json",
  );
  const metricsFile = resolveArgument(
    "metrics",
    "data/benchmarks/kronos-base-policy-holdout-ensemble-dense-execution-metrics.json",
  );
  const modelLabel = argument(
    "model-label",
    "base-policy-holdout-pretrained-ensemble",
  );
  if (!fs.existsSync(path.join(checkpoint, "model.safetensors"))) {
    throw new Error(`Missing fine-tuned predictor checkpoint: ${checkpoint}`);
  }
  const forecastExists = fs.existsSync(forecastFile);
  const metricsExists = fs.existsSync(metricsFile);
  if (forecastExists !== metricsExists) {
    throw new Error("Dense forecast and metrics artifacts must appear atomically as a pair.");
  }
  if (!forecastExists) {
    await runNpm("kronos:benchmark", [
      "--models", "base",
      "--model-checkpoint", checkpoint,
      "--ensemble-predictor-checkpoint", "pretrained",
      "--ensemble-sample-count", "10",
      "--model-label", modelLabel,
      "--temperature", "0.8",
      "--sample-count", "20",
      "--batch-size", "2",
      "--progress-checkpoint-seconds", "120",
      "--forecast-output", forecastFile,
      "--output", metricsFile,
    ]);
  }
  const forecast = loadJson<ForecastIdentity>(forecastFile);
  const metrics = loadJson<{ runSignature: string }>(metricsFile);
  if (forecast.contract !== "kronos-causal-15x1m-forecast-v2"
    || !/^[a-f0-9]{64}$/.test(forecast.runSignature)
    || forecast.runSignature !== metrics.runSignature
    || forecast.rows.length !== 11_232) {
    throw new Error("Dense forecast/metrics artifacts failed final identity checks.");
  }
  const prefix = forecast.runSignature.slice(0, 12);
  const policyFile = path.resolve(
    REPO_ROOT,
    `data/benchmarks/kronos-bot-policy-${prefix}.json`,
  );
  const validationFile = path.resolve(
    REPO_ROOT,
    `data/benchmarks/kronos-bot-validation-${prefix}.json`,
  );
  if (!fs.existsSync(policyFile)) {
    await runNpm("kronos:backtest", [
      "--phase=calibrate",
      `--forecasts=${forecastFile}`,
    ]);
  }
  const policy = loadJson<PolicyIdentity>(policyFile);
  assertPolicyGate(policy, forecast.runSignature);
  if (!fs.existsSync(validationFile)) {
    await runNpm("kronos:backtest", [
      "--phase=validate",
      `--forecasts=${forecastFile}`,
      `--policy=${policyFile}`,
    ]);
  }
  const validation = loadJson<ValidationIdentity>(validationFile);
  const forecastSha256 = crypto.createHash("sha256")
    .update(fs.readFileSync(forecastFile)).digest("hex");
  const policyArtifactSha256 = crypto.createHash("sha256")
    .update(fs.readFileSync(policyFile)).digest("hex");
  assertValidationReport(
    validation,
    forecast.runSignature,
    policy.selectedPolicyId,
    forecastSha256,
    policyArtifactSha256,
  );
  await runNpm("kronos:audit", [
    "--forecasts", forecastFile,
    "--metrics", metricsFile,
    "--policy", policyFile,
    "--validation", validationFile,
  ]);
  console.log(JSON.stringify({
    forecastFile,
    metricsFile,
    policyFile,
    validationFile,
    selectedPolicyId: policy.selectedPolicyId,
    aggregate: validation.aggregate,
    gates: validation.gates,
  }, null, 2));
}

if (path.resolve(process.argv[1] ?? "") === import.meta.filename) {
  void main().catch((error: unknown) => {
    console.error(error);
    process.exitCode = 1;
  });
}

import { spawn } from "node:child_process";
import { createWriteStream } from "node:fs";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface LoopArgs {
  agentCommand: string;
  benchmarkCommand: string;
  verifyCommand: string;
  outputDir: string;
  maxIterations: number;
  sleepSec: number;
  startSeed: number;
  agentTimeoutMin: number;
  dryRun: boolean;
}

interface CommandResult {
  command: string;
  exitCode: number | null;
  signal: NodeJS.Signals | null;
  elapsedMs: number;
  stdoutTail: string;
  stderrTail: string;
  timedOut: boolean;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_AGENT_COMMAND =
  process.env.TRADING_EXPERIMENT_AGENT_COMMAND ??
  "codex exec -C . --sandbox workspace-write -";
const DEFAULT_BENCHMARK_COMMAND =
  process.env.TRADING_EXPERIMENT_BENCHMARK_COMMAND ??
  [
    "npm run benchmark:strategies --",
    "--mode random-lengths",
    "--lookback-days 1825",
    "--min-window-days 7",
    "--max-window-days 120",
    "--samples 48",
    "--seed {seed}",
  ].join(" ");
const DEFAULT_VERIFY_COMMAND =
  process.env.TRADING_EXPERIMENT_VERIFY_COMMAND ?? "npm run typecheck";
const OUTPUT_TAIL_CHARS = 160_000;

let stopRequested = false;
process.once("SIGINT", () => {
  stopRequested = true;
  console.error("Stop requested; finishing current command before exiting.");
});
process.once("SIGTERM", () => {
  stopRequested = true;
  console.error("Stop requested; finishing current command before exiting.");
});

main().catch((error: unknown) => {
  console.error(error instanceof Error ? error.stack ?? error.message : error);
  process.exitCode = 1;
});

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));
  await runLoop(args);
}

async function runLoop(options: LoopArgs): Promise<void> {
  await fs.mkdir(options.outputDir, { recursive: true });

  for (let offset = 0; !stopRequested; offset += 1) {
    if (options.maxIterations > 0 && offset >= options.maxIterations) {
      break;
    }

    const iteration = offset + 1;
    const seed = options.startSeed + offset;
    const runDir = path.join(
      options.outputDir,
      `${String(iteration).padStart(4, "0")}-${timestampForPath()}`,
    );
    await fs.mkdir(runDir, { recursive: true });

    console.error(`\nExperiment loop iteration ${iteration}, seed ${seed}`);
    const benchmarkCommand = applyTemplate(options.benchmarkCommand, {
      iteration,
      seed,
      runDir,
    });
    const benchmark = await runShellCommand(benchmarkCommand, {
      cwd: repoRoot,
      logPath: path.join(runDir, "benchmark.log"),
    });
    await fs.writeFile(path.join(runDir, "benchmark.stdout.md"), benchmark.stdoutTail);
    await fs.writeFile(path.join(runDir, "benchmark.stderr.log"), benchmark.stderrTail);

    const prompt = buildAgentPrompt({
      iteration,
      seed,
      runDir,
      benchmark,
      benchmarkCommand,
      verifyCommand: options.verifyCommand,
    });
    await fs.writeFile(path.join(runDir, "agent-prompt.md"), prompt);

    if (options.dryRun) {
      console.error(`Dry run enabled; wrote prompt to ${path.relative(repoRoot, runDir)}.`);
      await writeIterationSummary(runDir, { benchmark });
      continue;
    }

    const agent = await runShellCommand(options.agentCommand, {
      cwd: repoRoot,
      input: prompt,
      logPath: path.join(runDir, "agent.log"),
      timeoutMs: options.agentTimeoutMin > 0 ? options.agentTimeoutMin * 60_000 : undefined,
    });
    await fs.writeFile(path.join(runDir, "agent.stdout.log"), agent.stdoutTail);
    await fs.writeFile(path.join(runDir, "agent.stderr.log"), agent.stderrTail);

    const verify =
      options.verifyCommand.trim().length > 0
        ? await runShellCommand(options.verifyCommand, {
            cwd: repoRoot,
            logPath: path.join(runDir, "verify.log"),
          })
        : undefined;

    await writeIterationSummary(runDir, { benchmark, agent, verify });

    if (stopRequested) {
      break;
    }
    if (options.sleepSec > 0) {
      await sleep(options.sleepSec * 1000);
    }
  }
}

function buildAgentPrompt(input: {
  iteration: number;
  seed: number;
  runDir: string;
  benchmark: CommandResult;
  benchmarkCommand: string;
  verifyCommand: string;
}): string {
  const relativeRunDir = path.relative(repoRoot, input.runDir);
  return `You are an autonomous trading-strategy experiment agent working in this repo.

Goal: improve the legacy valley/peak algorithm so it captures more of the perfect-margin-trader oracle profit without overfitting to a recent 30-day trend.

Mandatory validation frame:
- Treat full BTC-cycle random-length testing as the primary evidence source.
- Do not promote a change based only on a contiguous 30-day backtest.
- Prefer random windows sampled across the whole available BTC cycle, currently requested as 1825 lookback days.
- Compare every legacy valley/peak change against the current legacy default and the perfect-margin oracle metrics.
- Optimize for oracle capture, net return, risk-adjusted return, drawdown, and robustness across sampled regimes, not one lucky window.

Current iteration:
- Iteration: ${input.iteration}
- Seed: ${input.seed}
- Evidence directory: ${relativeRunDir}
- Benchmark command: ${input.benchmarkCommand}
- Benchmark exit code: ${input.benchmark.exitCode}

Benchmark stdout:
\`\`\`md
${input.benchmark.stdoutTail.trim()}
\`\`\`

Benchmark stderr:
\`\`\`text
${input.benchmark.stderrTail.trim()}
\`\`\`

Your task:
1. Review the benchmark and identify what the legacy valley/peak variants reveal about market regimes, costs, and failure modes.
2. Run additional experiments only if they use cycle-wide random-length windows, full-cycle folds, or a broader regime sample. Avoid optimizing against a single recent 30-day window.
3. Improve the legacy valley/peak algorithm, its default config, grid candidates, or experiment harness when the evidence justifies it.
4. Record the important observation in docs/experiment-plan.md or docs/strategy-research.md.
5. Run ${input.verifyCommand || "the relevant typecheck/build command"} before finishing.

Constraints:
- Do not start another indefinite experiment loop.
- Do not add non-legacy strategy controls.
- Keep changes inspectable and focused.
- If no code change is justified, write that conclusion to the docs and leave the implementation alone.
`;
}

async function runShellCommand(
  command: string,
  options: {
    cwd: string;
    input?: string;
    logPath: string;
    timeoutMs?: number;
  },
): Promise<CommandResult> {
  const startedAt = Date.now();
  await fs.mkdir(path.dirname(options.logPath), { recursive: true });
  const log = createWriteStream(options.logPath, { flags: "a" });
  const child = spawn(command, {
    cwd: options.cwd,
    env: process.env,
    shell: true,
    stdio: ["pipe", "pipe", "pipe"],
  });

  let stdoutTail = "";
  let stderrTail = "";
  let timedOut = false;
  let timeout: NodeJS.Timeout | undefined;

  if (options.timeoutMs && options.timeoutMs > 0) {
    timeout = setTimeout(() => {
      timedOut = true;
      child.kill("SIGTERM");
    }, options.timeoutMs);
  }

  child.stdout.on("data", (chunk: Buffer) => {
    const text = chunk.toString("utf8");
    process.stdout.write(text);
    log.write(text);
    stdoutTail = appendTail(stdoutTail, text);
  });
  child.stderr.on("data", (chunk: Buffer) => {
    const text = chunk.toString("utf8");
    process.stderr.write(text);
    log.write(text);
    stderrTail = appendTail(stderrTail, text);
  });
  child.on("error", (error) => {
    const text = `${error.stack ?? error.message}\n`;
    process.stderr.write(text);
    log.write(text);
    stderrTail = appendTail(stderrTail, text);
  });

  if (options.input !== undefined) {
    child.stdin.write(options.input);
  }
  child.stdin.end();

  const { exitCode, signal } = await new Promise<{
    exitCode: number | null;
    signal: NodeJS.Signals | null;
  }>((resolve) => {
    child.on("close", (exitCode, signal) => resolve({ exitCode, signal }));
  });

  if (timeout) {
    clearTimeout(timeout);
  }
  await new Promise<void>((resolve) => log.end(resolve));

  return {
    command,
    exitCode,
    signal,
    elapsedMs: Date.now() - startedAt,
    stdoutTail,
    stderrTail,
    timedOut,
  };
}

async function writeIterationSummary(
  runDir: string,
  summary: {
    benchmark: CommandResult;
    agent?: CommandResult;
    verify?: CommandResult;
  },
): Promise<void> {
  await fs.writeFile(
    path.join(runDir, "summary.json"),
    `${JSON.stringify(
      {
        createdAt: new Date().toISOString(),
        ...summary,
      },
      null,
      2,
    )}\n`,
  );
}

function parseArgs(argv: string[]): LoopArgs {
  const values = new Map<string, string>();
  for (let index = 0; index < argv.length; index += 1) {
    const key = argv[index];
    if (!key.startsWith("--")) {
      continue;
    }
    const next = argv[index + 1];
    if (next && !next.startsWith("--")) {
      values.set(key.slice(2), next);
      index += 1;
    } else {
      values.set(key.slice(2), "true");
    }
  }

  return {
    agentCommand: values.get("agent-command") ?? DEFAULT_AGENT_COMMAND,
    benchmarkCommand: values.get("benchmark-command") ?? DEFAULT_BENCHMARK_COMMAND,
    verifyCommand:
      values.get("verify-command") === "none"
        ? ""
        : values.get("verify-command") ?? DEFAULT_VERIFY_COMMAND,
    outputDir: path.resolve(
      repoRoot,
      values.get("output-dir") ?? path.join("data", "experiments", "agent-loop"),
    ),
    maxIterations: parseNonNegativeInt(values.get("max-iterations"), 0),
    sleepSec: parseNonNegativeNumber(values.get("sleep-sec"), 60),
    startSeed: parseNonNegativeInt(values.get("start-seed"), 7331),
    agentTimeoutMin: parseNonNegativeNumber(values.get("agent-timeout-min"), 180),
    dryRun: values.has("dry-run"),
  };
}

function applyTemplate(
  value: string,
  replacements: { iteration: number; seed: number; runDir: string },
): string {
  return value
    .replaceAll("{iteration}", String(replacements.iteration))
    .replaceAll("{seed}", String(replacements.seed))
    .replaceAll("{runDir}", shellQuote(replacements.runDir));
}

function appendTail(current: string, next: string): string {
  const value = current + next;
  return value.length <= OUTPUT_TAIL_CHARS ? value : value.slice(-OUTPUT_TAIL_CHARS);
}

function parseNonNegativeInt(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? Math.round(parsed) : fallback;
}

function parseNonNegativeNumber(value: string | undefined, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) && parsed >= 0 ? parsed : fallback;
}

function timestampForPath(): string {
  return new Date().toISOString().replace(/[:.]/g, "-");
}

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

import { spawn, type ChildProcessWithoutNullStreams, type SpawnOptionsWithoutStdio } from "node:child_process";
import path from "node:path";
import type { ForecastTree } from "@trading/bot-algo/oracle-crra";

export interface OracleForecastInput {
  readonly observedPrice: number;
  readonly features: readonly number[];
}
export interface OracleForecastProviderConfig {
  readonly pythonExecutable: string;
  readonly checkpoint: string;
  readonly cwd: string;
  readonly workerPath?: string;
  readonly bins?: number;
  readonly horizon?: number;
  readonly device?: "auto" | "cpu" | "cuda";
  readonly timeoutMs?: number;
}
export interface OracleForecastProviderDependencies {
  readonly spawn?: (
    executable: string, args: readonly string[], options: SpawnOptionsWithoutStdio,
  ) => ChildProcessWithoutNullStreams;
}
interface PendingRequest {
  readonly resolve: (tree: ForecastTree) => void;
  readonly reject: (error: Error) => void;
  readonly timer: ReturnType<typeof setTimeout>;
}

const FEATURE_COUNT = 178;
const MAX_LINE_BYTES = 4 * 1024 * 1024;
const STDERR_CHARACTERS = 4096;

/** Persistent local inference only. It never submits exchange orders. */
export class OracleForecastProvider {
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly pending = new Map<string, PendingRequest>();
  private readonly timeoutMs: number;
  private readonly exited: Promise<void>;
  private phase: "open" | "closing" | "closed" | "failed" = "open";
  private failure: Error | undefined;
  private nextId = 1;
  private stdoutBuffer = "";
  private stderrTail = "";
  private killTimer: ReturnType<typeof setTimeout> | undefined;
  private hardKillTimer: ReturnType<typeof setTimeout> | undefined;

  constructor(readonly config: OracleForecastProviderConfig, dependencies: OracleForecastProviderDependencies = {}) {
    this.timeoutMs = config.timeoutMs ?? 30_000;
    if (!config.pythonExecutable || !config.checkpoint || !config.cwd)
      throw new Error("Forecast worker requires Python executable, checkpoint and cwd.");
    if (!Number.isSafeInteger(this.timeoutMs) || this.timeoutMs <= 0)
      throw new Error("Forecast timeout must be a positive integer.");
    const bins = config.bins ?? 3;
    const horizon = config.horizon ?? 2;
    if (bins !== 3 || horizon !== 2)
      throw new Error("This forecast checkpoint worker supports exactly 3 continuous bins plus zero and horizon 2.");
    const launch = dependencies.spawn ?? ((executable, args, options) => spawn(executable, args, options));
    this.child = launch(config.pythonExecutable, [
      "-u", config.workerPath ?? path.join(config.cwd, "ml", "serve_stochastic_price_forecast.py"),
      "--checkpoint", config.checkpoint, "--device", config.device ?? "auto",
    ], { cwd: config.cwd, windowsHide: true, stdio: "pipe" });
    this.child.stdout.setEncoding("utf8");
    this.child.stderr.setEncoding("utf8");
    this.child.stdout.on("data", (chunk: string) => this.receive(chunk));
    this.child.stderr.on("data", (chunk: string) => {
      this.stderrTail = (this.stderrTail + chunk).slice(-STDERR_CHARACTERS);
    });
    this.child.stdin.on("error", (error) => {
      if (this.phase === "open") this.fail(new Error(`Forecast worker input failed: ${error.message}`));
    });
    this.child.on("error", (error) => this.fail(new Error(`Forecast worker could not run: ${error.message}`)));
    this.exited = new Promise<void>((resolve) => {
      this.child.once("close", (code, signal) => {
        if (this.phase === "open") this.fail(new Error(
          `Forecast worker exited (${signal ?? code ?? "unknown"}).${this.stderrTail ? ` ${this.stderrTail.trim()}` : ""}`,
        ));
        clearTimeout(this.killTimer);
        clearTimeout(this.hardKillTimer);
        if (this.phase !== "failed") this.phase = "closed";
        resolve();
      });
    });
  }

  predict(input: OracleForecastInput): Promise<ForecastTree> {
    if (this.phase !== "open") return Promise.reject(this.failure ?? new Error("Forecast provider is closed."));
    if (!(input.observedPrice > 0) || !Number.isFinite(input.observedPrice) ||
        input.features.length !== FEATURE_COUNT || input.features.some((value) => !Number.isFinite(value)))
      return Promise.reject(new Error(`Forecast input requires positive finite observedPrice and ${FEATURE_COUNT} finite features.`));
    const id = String(this.nextId++);
    return new Promise<ForecastTree>((resolve, reject) => {
      const timer = setTimeout(() => this.fail(new Error(`Forecast request ${id} timed out after ${this.timeoutMs} ms.`)), this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      this.child.stdin.write(JSON.stringify({ id, observedPrice: input.observedPrice, features: input.features }) + "\n", (error) => {
        if (error && this.phase === "open") this.fail(new Error(`Forecast worker input failed: ${error.message}`));
      });
    });
  }

  async close(): Promise<void> {
    if (this.phase === "closed") return;
    if (this.phase === "open") {
      this.phase = "closing";
      this.rejectAll(new Error("Forecast provider closed before inference completed."));
      this.child.stdin.end();
      this.scheduleTermination();
    }
    await this.exited;
  }

  private rejectAll(error: Error): void {
    for (const request of this.pending.values()) {
      clearTimeout(request.timer);
      request.reject(error);
    }
    this.pending.clear();
  }

  private scheduleTermination(): void {
    if (this.child.exitCode !== null || this.child.signalCode !== null || this.killTimer) return;
    this.killTimer = setTimeout(() => this.child.kill(), 250);
    this.hardKillTimer = setTimeout(() => this.child.kill("SIGKILL"), 1000);
  }

  private fail(error: Error): void {
    if (this.phase !== "open") return;
    this.phase = "failed";
    this.failure = error;
    this.rejectAll(error);
    this.child.stdin.destroy();
    this.child.kill();
    this.scheduleTermination();
  }

  private receive(chunk: string): void {
    if (this.phase !== "open") return;
    this.stdoutBuffer += chunk;
    let newline: number;
    while ((newline = this.stdoutBuffer.indexOf("\n")) >= 0) {
      const line = this.stdoutBuffer.slice(0, newline).trim();
      this.stdoutBuffer = this.stdoutBuffer.slice(newline + 1);
      if (line.length > MAX_LINE_BYTES) { this.fail(new Error("Forecast worker response exceeds the size limit.")); return; }
      if (!line) continue;
      let response: unknown;
      try { response = JSON.parse(line); }
      catch { this.fail(new Error("Forecast worker returned invalid JSON.")); return; }
      if (!response || typeof response !== "object" || !("id" in response) || typeof response.id !== "string") {
        this.fail(new Error("Forecast worker response lacks its request ID.")); return;
      }
      const request = this.pending.get(response.id);
      if (!request) { this.fail(new Error("Forecast worker returned an unknown or duplicate request ID.")); return; }
      if ("error" in response) {
        const error = response.error;
        if (!error || typeof error !== "object" || !("code" in error) || typeof error.code !== "string" ||
            !("message" in error) || typeof error.message !== "string") {
          this.fail(new Error("Forecast worker error must contain a code and message.")); return;
        }
        clearTimeout(request.timer);
        this.pending.delete(response.id);
        request.reject(new Error(`Forecast inference failed (${error.code.slice(0, 80)}): ${error.message.slice(0, STDERR_CHARACTERS)}`));
      } else if ("tree" in response && validTree(response.tree, this.config.horizon ?? 2)) {
        clearTimeout(request.timer);
        this.pending.delete(response.id);
        request.resolve(response.tree);
      } else { this.fail(new Error("Forecast worker returned an invalid price tree.")); return; }
    }
    if (this.stdoutBuffer.length > MAX_LINE_BYTES) this.fail(new Error("Forecast worker response exceeds the size limit."));
  }
}

/** Runtime validation precedes rational normalization and observed-price merging. */
function validTree(value: unknown, remaining: number): value is ForecastTree {
  if (!value || typeof value !== "object" || !("children" in value) || !Array.isArray(value.children)) return false;
  if (!remaining) return value.children.length === 0;
  if (!value.children.length || value.children.length > 1024) return false;
  let mass = 0;
  for (const branch of value.children) {
    if (!branch || typeof branch !== "object" || typeof branch.price !== "number" || !Number.isFinite(branch.price) || branch.price <= 0 ||
        typeof branch.probability !== "number" || !Number.isFinite(branch.probability) || branch.probability < 0) return false;
    mass += branch.probability;
    if (!validTree({ children: branch.children ?? [] }, remaining - 1)) return false;
  }
  return Number.isFinite(mass) && mass > 0;
}

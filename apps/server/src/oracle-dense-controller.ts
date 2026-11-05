import { spawn, type ChildProcessWithoutNullStreams, type SpawnOptionsWithoutStdio } from "node:child_process";
import path from "node:path";

export interface DenseObservation {
  readonly features: readonly [readonly number[], readonly number[]];
  readonly prices: readonly [number, number];
  /** Unix seconds, including the preceding observed second. */
  readonly timestamps: readonly [number, number];
}
export interface DenseOrder {
  readonly side: "buy" | "sell";
  readonly price: number;
  readonly notional: number;
}
export interface DenseAccount {
  readonly quote: number;
  readonly asset: number;
  readonly price: number;
  readonly previousPrice: number;
  readonly orders: readonly DenseOrder[];
}
export interface DenseInitialization {
  readonly observation: DenseObservation;
  /** Already-arrived account: funding and committed fills have been applied. */
  readonly account: Omit<DenseAccount, "price">;
  readonly terminalSeconds: number;
  readonly consumedQuote: number;
  readonly filled: boolean;
}
export interface DenseDecision {
  readonly phase: "active" | "complete";
  readonly update: number;
  readonly timestamp: number;
  /** This account has already executed the selected action in Oracle paper execution. */
  readonly account: DenseAccount;
  readonly terminalWealth: number | null;
  readonly diagnostics: Readonly<Record<string, unknown>>;
}
export interface DenseWorkerConfig {
  readonly pythonExecutable: string;
  readonly configPath: string;
  readonly cwd: string;
  readonly stateDirectory?: string;
  readonly initializationTimeoutMs?: number;
  readonly tickTimeoutMs?: number;
}
export interface DenseWorkerDependencies {
  readonly spawn?: (executable: string, args: readonly string[], options: SpawnOptionsWithoutStdio) => ChildProcessWithoutNullStreams;
}

const finite = (x: unknown): x is number => typeof x === "number" && Number.isFinite(x);
const record = (x: unknown): x is Record<string, unknown> => x !== null && typeof x === "object" && !Array.isArray(x);

function validObservation(x: DenseObservation): boolean {
  return x.features.length === 2 && x.features.every(row => row.length === 59 && row.every(finite)) &&
    x.prices.length === 2 && x.prices.every(p => finite(p) && p > 0) &&
    x.timestamps.length === 2 && x.timestamps.every(Number.isSafeInteger) && x.timestamps[1] - x.timestamps[0] === 1;
}
function validAccount(x: unknown): x is DenseAccount {
  return record(x) && finite(x.quote) && finite(x.asset) && finite(x.price) && x.price > 0 &&
    finite(x.previousPrice) && x.previousPrice > 0 && Array.isArray(x.orders) && x.orders.length <= 3 &&
    x.orders.every(o => record(o) && (o.side === "buy" || o.side === "sell") &&
      finite(o.price) && o.price > 0 && finite(o.notional) && o.notional > 0);
}

/** Persistent finite-grid log-utility paper bot. Does not submit exchange orders. */
export class DenseOracleController {
  private readonly child: ChildProcessWithoutNullStreams;
  private readonly exited: Promise<void>;
  private phase: "new" | "active" | "complete" | "failed" | "closing" | "closed" = "new";
  private pending: {
    readonly id: string;
    readonly resolve: (reply: Record<string, unknown>) => void;
    readonly reject: (error: Error) => void;
    readonly timer: ReturnType<typeof setTimeout>;
  } | undefined;
  private busy = false;
  private buffer = "";
  private stderr = "";
  private nextId = 1;
  private failure: Error | undefined;
  private lastObservation: DenseObservation | undefined;
  private terminal = 0;
  private update = 0;
  private killTimer: ReturnType<typeof setTimeout> | undefined;
  private hardKillTimer: ReturnType<typeof setTimeout> | undefined;

  get failureReason(): string | undefined { return this.failure?.message; }

  constructor(readonly config: DenseWorkerConfig, dependencies: DenseWorkerDependencies = {}) {
    if (!config.pythonExecutable || !config.configPath || !config.cwd)
      throw new Error("Dense worker requires Python, configPath and cwd");
    for (const timeout of [config.initializationTimeoutMs ?? 1_200_000, config.tickTimeoutMs ?? 180_000])
      if (!Number.isSafeInteger(timeout) || timeout <= 0) throw new Error("Invalid worker timeout");
    const launch = dependencies.spawn ?? ((exe, args, options) => spawn(exe, args, options));
    this.child = launch(config.pythonExecutable, ["-u", path.join(config.cwd, "ml", "serve_oracle_dense.py"),
      "--config", config.configPath, ...(config.stateDirectory ? ["--state-directory", config.stateDirectory] : [])],
      { cwd: config.cwd, windowsHide: true, stdio: "pipe" });
    this.child.stdout.setEncoding("utf8");
    this.child.stderr.setEncoding("utf8");
    this.child.stdout.on("data", (chunk: string) => this.receive(chunk));
    this.child.stderr.on("data", (chunk: string) => { this.stderr = (this.stderr + chunk).slice(-4096); });
    this.child.stdin.on("error", error => this.fail(new Error(`Dense worker input failed: ${error.message}`)));
    this.child.on("error", error => this.fail(new Error(`Dense worker could not run: ${error.message}`)));
    this.exited = new Promise(resolve => this.child.once("close", (code, signal) => {
      if (this.phase !== "closing" && this.phase !== "closed" && this.phase !== "failed")
        this.fail(new Error(`Dense worker exited (${signal ?? code}). ${this.stderr}`));
      clearTimeout(this.killTimer);
      clearTimeout(this.hardKillTimer);
      if (this.phase === "closing") this.phase = "closed";
      resolve();
    }));
  }

  async initialize(input: DenseInitialization): Promise<DenseDecision> {
    if (this.phase !== "new" || this.busy) throw this.failure ?? new Error("Controller already initialized or busy");
    if (!validObservation(input.observation) || !Number.isSafeInteger(input.terminalSeconds) ||
        input.terminalSeconds <= input.observation.timestamps[1] || !finite(input.consumedQuote) ||
        input.consumedQuote < 0 || typeof input.filled !== "boolean" || (input.consumedQuote > 0 && !input.filled) ||
        !validAccount({ ...input.account, price: input.observation.prices[1] }))
      throw new Error("Invalid dense initialization");
    this.terminal = input.terminalSeconds;
    return this.request({ kind: "initialize", ...input }, input.observation, 0,
      this.config.initializationTimeoutMs ?? 1_200_000);
  }

  async onTick(observation: DenseObservation): Promise<DenseDecision> {
    if (this.phase !== "active" || this.busy) throw this.failure ?? new Error("Controller is not ready for a tick");
    const previous = this.lastObservation!;
    if (!validObservation(observation) || observation.timestamps[0] !== previous.timestamps[1] ||
        observation.prices[0] !== previous.prices[1] ||
        observation.features[0].some((x, i) => Math.fround(x) !== Math.fround(previous.features[1][i])))
      throw new Error("Tick must extend the accepted history by exactly one second");
    return this.request({ kind: "tick", observation }, observation, this.update + 1,
      this.config.tickTimeoutMs ?? 180_000);
  }

  async checkpoint(): Promise<DenseDecision> {
    if (this.phase !== "active" || this.busy || !this.config.stateDirectory)
      throw this.failure ?? new Error("Checkpoint requires an active idle controller and state directory");
    return this.request({ kind: "checkpoint" }, this.lastObservation!, this.update,
      this.config.initializationTimeoutMs ?? 1_200_000);
  }

  async restore(): Promise<DenseDecision> {
    if (this.phase !== "new" || this.busy || !this.config.stateDirectory)
      throw this.failure ?? new Error("Restore requires a fresh controller and state directory");
    this.busy = true;
    try {
      const response = await this.exchange({ kind: "restore" }, this.config.initializationTimeoutMs ?? 1_200_000);
      const observation = response.observation as DenseObservation;
      if (!record(observation) || !validObservation(observation) || !Number.isSafeInteger(response.update) ||
          (response.update as number) < 0 || !Number.isSafeInteger(response.terminalSeconds) ||
          (response.terminalSeconds as number) <= observation.timestamps[1])
        throw new Error("Invalid checkpoint boundary from dense worker");
      this.terminal = response.terminalSeconds as number;
      return this.accept(response, observation, response.update as number);
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
      throw error;
    } finally { this.busy = false; }
  }

  private exchange(body: Record<string, unknown>, timeout: number): Promise<Record<string, unknown>> {
    const id = String(this.nextId++);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => this.fail(new Error(`Dense request timed out after ${timeout} ms`)), timeout);
      this.pending = { id, resolve, reject, timer };
      this.child.stdin.write(JSON.stringify({ id, ...body }) + "\n", error => {
        if (error) this.fail(new Error(`Dense worker input failed: ${error.message}`));
      });
    });
  }

  private async request(body: Record<string, unknown>, observation: DenseObservation, update: number, timeout: number): Promise<DenseDecision> {
    this.busy = true;
    try {
      const response = await this.exchange(body, timeout);
      return this.accept(response, observation, update);
    } catch (error) {
      this.fail(error instanceof Error ? error : new Error(String(error)));
      throw error;
    } finally { this.busy = false; }
  }

  private accept(response: Record<string, unknown>, observation: DenseObservation, update: number): DenseDecision {
      if (this.failure) throw this.failure;
      const complete = observation.timestamps[1] === this.terminal;
      if (response.update !== update || response.timestamp !== observation.timestamps[1] ||
          response.phase !== (complete ? "complete" : "active") || !validAccount(response.account) ||
          response.account.price !== observation.prices[1] || !record(response.diagnostics) ||
          (complete ? (!finite(response.terminalWealth) || response.terminalWealth <= 0 ||
            response.terminalWealth !== response.account.quote || response.account.asset !== 0 || response.account.orders.length !== 0)
            : response.terminalWealth !== null)) throw new Error("Invalid dense worker decision or terminal account");
      this.phase = complete ? "complete" : "active";
      this.lastObservation = observation;
      this.update = update;
      return response as unknown as DenseDecision;
  }

  private receive(chunk: string): void {
    if (this.phase === "failed" || this.phase === "closing" || this.phase === "closed") return;
    this.buffer += chunk;
    let newline: number;
    while ((newline = this.buffer.indexOf("\n")) >= 0) {
      const line = this.buffer.slice(0, newline).trim();
      this.buffer = this.buffer.slice(newline + 1);
      if (!line) continue;
      try {
        if (line.length > 262144) throw new Error("Dense response exceeds size limit");
        const response: unknown = JSON.parse(line);
        if (!record(response) || !this.pending || response.id !== this.pending.id)
          throw new Error("Unknown or duplicate dense request ID");
        if ("error" in response)
          throw new Error(`Dense worker failed: ${record(response.error) ? String(response.error.message).slice(0, 4096) : "invalid error"}`);
        const pending = this.pending;
        this.pending = undefined;
        clearTimeout(pending.timer);
        pending.resolve(response);
      } catch (error) { this.fail(error instanceof Error ? error : new Error(String(error))); return; }
    }
    if (this.buffer.length > 262144) this.fail(new Error("Dense response exceeds size limit"));
  }

  private terminate(): void {
    if (this.child.exitCode !== null || this.child.signalCode !== null || this.killTimer) return;
    this.killTimer = setTimeout(() => this.child.kill(), 250);
    this.hardKillTimer = setTimeout(() => this.child.kill("SIGKILL"), 1000);
  }

  private fail(error: Error): void {
    if (this.phase === "failed" || this.phase === "closed" || this.phase === "closing") return;
    this.failure = error;
    this.phase = "failed";
    if (this.pending) {
      clearTimeout(this.pending.timer);
      this.pending.reject(error);
      this.pending = undefined;
    }
    this.child.stdin.destroy();
    this.child.kill();
    this.terminate();
  }

  async close(): Promise<void> {
    if (this.phase === "closed") return;
    if (this.phase !== "failed") {
      this.phase = "closing";
      if (this.pending) {
        clearTimeout(this.pending.timer);
        this.pending.reject(new Error("Dense controller closed during evaluation"));
        this.pending = undefined;
      }
      this.child.stdin.end();
      this.terminate();
    }
    await this.exited;
  }
}

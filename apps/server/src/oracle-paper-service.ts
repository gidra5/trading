import { existsSync, mkdirSync, renameSync, rmSync, writeFileSync } from "node:fs";
import path from "node:path";
import { DenseOracleController, type DenseDecision, type DenseInitialization, type DenseObservation,
  type DenseWorkerConfig } from "./oracle-dense-controller.js";
import { SymbolSchema } from "./schemas.js";

export interface OraclePaperOptions {
  readonly symbol: string;
  readonly worker: DenseWorkerConfig & { readonly stateDirectory: string };
  readonly checkpointEvery?: number;
}
type Controller = Pick<DenseOracleController, "initialize" | "onTick" | "checkpoint" | "restore" | "close" | "failureReason">;
class PaperConflict extends Error { readonly statusCode = 409; }

/** A single explicitly configured paper account; no exchange credentials or APIs. */
export class OraclePaperService {
  private controller: Controller | undefined;
  private last: DenseDecision | undefined;
  private busy = false;
  private failure: string | undefined;
  private readonly completedPath: string;
  private readonly pointerPath: string;
  private readonly checkpointEvery: number;

  constructor(readonly options: OraclePaperOptions,
    private readonly factory: (config: DenseWorkerConfig) => Controller = config => new DenseOracleController(config)) {
    SymbolSchema.parse(options.symbol);
    if (!options.worker.stateDirectory) throw new Error("Paper worker state directory is required");
    this.checkpointEvery = options.checkpointEvery ?? 60;
    if (!Number.isSafeInteger(this.checkpointEvery) || this.checkpointEvery < 1)
      throw new Error("Invalid paper checkpoint interval");
    this.completedPath = path.join(options.worker.stateDirectory, "completed.json");
    this.pointerPath = path.join(options.worker.stateDirectory, "current.json");
  }

  status() {
    const failure = this.controller?.failureReason ?? this.failure;
    return { symbol: this.options.symbol, mode: "paper" as const, busy: this.busy,
      running: this.controller !== undefined && failure === undefined, failure: failure ?? null,
      checkpointAvailable: existsSync(this.pointerPath),
      completed: existsSync(this.completedPath), decision: this.last ?? null };
  }

  private async releaseFailed(error: unknown): Promise<void> {
    this.failure = (error instanceof Error ? error.message : String(error)).slice(0, 4096);
    const controller = this.controller;
    this.controller = undefined;
    await controller?.close();
  }

  private async exclusive<T>(run: () => Promise<T>): Promise<T> {
    if (this.busy) throw new PaperConflict("Paper account is processing another request");
    this.busy = true;
    try { return await run(); } finally { this.busy = false; }
  }

  start(initial: DenseInitialization): Promise<DenseDecision> {
    return this.exclusive(async () => {
      if (this.controller || existsSync(this.pointerPath) || existsSync(this.completedPath))
        throw new PaperConflict("Paper account exists; resume it or explicitly reset before starting a new episode");
      const controller = this.factory(this.options.worker);
      try {
        const decision = await controller.initialize(initial);
        await controller.checkpoint();
        this.controller = controller;
        this.last = decision;
        this.failure = undefined;
        return decision;
      } catch (error) {
        this.failure = (error instanceof Error ? error.message : String(error)).slice(0, 4096);
        await controller.close();
        throw error;
      }
    });
  }

  resume(): Promise<DenseDecision> {
    return this.exclusive(async () => {
      if (this.controller?.failureReason) await this.releaseFailed(this.controller.failureReason);
      if (this.controller || existsSync(this.completedPath))
        throw new PaperConflict("Paper account is already running or completed");
      const controller = this.factory(this.options.worker);
      try {
        const decision = await controller.restore();
        this.controller = controller;
        this.last = decision;
        this.failure = undefined;
        return decision;
      } catch (error) {
        this.failure = (error instanceof Error ? error.message : String(error)).slice(0, 4096);
        await controller.close();
        throw error;
      }
    });
  }

  tick(observation: DenseObservation): Promise<DenseDecision> {
    return this.exclusive(async () => {
      if (!this.controller) throw new PaperConflict("Start or resume the paper account before sending observations");
      let advanced = false;
      try {
        const decision = await this.controller.onTick(observation);
        advanced = true;
        if (decision.phase === "complete") {
          mkdirSync(this.options.worker.stateDirectory, { recursive: true });
          const temporary = this.completedPath + ".tmp";
          writeFileSync(temporary, JSON.stringify(decision), { flush: true });
          renameSync(temporary, this.completedPath);
          await this.controller.close();
          this.controller = undefined;
        } else if (decision.update % this.checkpointEvery === 0) {
          await this.controller.checkpoint();
        }
        this.last = decision;
        return decision;
      } catch (error) {
        // Local rejected input leaves the worker untouched. A failed worker or
        // an uncommitted advanced tick must recover from the saved boundary.
        if (advanced || this.controller?.failureReason) await this.releaseFailed(error);
        throw error;
      }
    });
  }

  checkpoint(): Promise<DenseDecision> {
    return this.exclusive(async () => {
      if (!this.controller || this.last?.phase !== "active") throw new PaperConflict("No active paper account");
      try { return await this.controller.checkpoint(); }
      catch (error) {
        if (this.controller.failureReason) await this.releaseFailed(error);
        throw error;
      }
    });
  }

  stop(): Promise<{ readonly status: "stopped" }> {
    return this.exclusive(async () => {
      if (this.controller) {
        const controller = this.controller;
        try {
          if (controller.failureReason) this.failure = controller.failureReason;
          else if (this.last?.phase === "active") await controller.checkpoint();
        } catch (error) {
          this.failure = (error instanceof Error ? error.message : String(error)).slice(0, 4096);
          throw error;
        }
        finally { await controller.close(); this.controller = undefined; }
      }
      return { status: "stopped" };
    });
  }

  reset(): Promise<{ readonly status: "reset" }> {
    return this.exclusive(async () => {
      if (this.controller) throw new PaperConflict("Stop the paper account before resetting it");
      // Remove only this service's small pointers. Immutable historical snapshots remain.
      rmSync(this.pointerPath, { force: true });
      rmSync(this.completedPath, { force: true });
      this.last = undefined;
      this.failure = undefined;
      return { status: "reset" };
    });
  }

  async close(): Promise<void> { await this.stop(); }
}

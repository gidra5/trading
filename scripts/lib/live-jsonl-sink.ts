import fs from "node:fs";
import fsp from "node:fs/promises";
import path from "node:path";
import { gzip } from "node:zlib";
import { promisify } from "node:util";

const compress = promisify(gzip);
interface PendingFile {
  readonly file: string;
  readonly source: string;
  readonly date: string;
  chunks: string[];
  bytes: number;
  completed: Promise<void>;
}
export interface LiveSinkOptions {
  readonly compression?: "gzip" | "none";
  readonly fsync?: boolean;
  readonly gzipLevel?: number;
  readonly sessionId?: string;
  readonly sessionFiles?: boolean;
  readonly maxPendingBytes?: number;
  readonly onError?: (error: Error) => void;
}

/** A failed/overloaded sink stops accepting data. Disk failures must not look like feed coverage. */
export class RotatingJsonlSink {
  private readonly files = new Map<string, PendingFile>();
  private readonly pending = new Set<Promise<void>>();
  private readonly written = new Map<string, number>();
  private pendingBytes = 0;
  private recordSequence = 0;
  private failure: Error | undefined;
  private closed = false;
  private readonly timer: NodeJS.Timeout;
  private lastCommittedAt: number | null = null;

  constructor(private readonly root: string, private readonly options: LiveSinkOptions = {}) {
    if (!Number.isInteger(options.gzipLevel ?? 6) || (options.gzipLevel ?? 6) < 0 || (options.gzipLevel ?? 6) > 9) throw new Error("gzipLevel must be 0..9");
    this.timer = setInterval(() => { void this.flush().catch(() => {}); }, 1_000);
    this.timer.unref();
  }

  write(source: string, payload: unknown, recordedAt = Date.now()) {
    if (this.failure) throw this.failure;
    if (this.closed) throw new Error("Live sink is closed");
    if (!/^[a-zA-Z0-9_./-]+$/.test(source) || source.split(/[\\/]/).includes("..") || path.isAbsolute(source)) {
      throw new Error(`Invalid live source: ${source}`);
    }
    const date = new Date(recordedAt).toISOString().slice(0, 10);
    const suffix = this.options.sessionFiles ? `.${this.options.sessionId}` : "";
    const file = path.join(this.root, source, `${date}${suffix}.jsonl${this.options.compression === "none" ? "" : ".gz"}`);
    let state = this.files.get(file);
    if (!state) {
      fs.mkdirSync(path.dirname(file), { recursive: true });
      state = { file, source, date, chunks: [], bytes: 0, completed: Promise.resolve() };
      this.files.set(file, state);
    }
    const line = `${JSON.stringify({ recordedAt, sessionId: this.options.sessionId, recordSequence: ++this.recordSequence, payload })}\n`;
    const bytes = Buffer.byteLength(line);
    if (this.pendingBytes + bytes > (this.options.maxPendingBytes ?? 256 * 1_024 * 1_024)) {
      throw this.fail(new Error("Live sink backlog limit exceeded; collection stopped without dropping records silently"));
    }
    state.chunks.push(line);
    state.bytes += bytes;
    this.pendingBytes += bytes;
    if (state.bytes >= 256 * 1_024) void this.flushFile(state, source).catch(() => {});
  }

  async flush() {
    await Promise.all([...this.files.values()].map((state) => this.flushFile(state, state.source)));
    const today = new Date().toISOString().slice(0, 10);
    for (const [file, state] of this.files) if (state.date < today && state.chunks.length === 0 && !this.pending.has(state.completed)) this.files.delete(file);
    if (this.failure) throw this.failure;
  }

  async close() {
    clearInterval(this.timer);
    this.closed = true;
    await this.flush();
    await Promise.all(this.pending);
    this.files.clear();
  }

  bytesWritten() { return Object.fromEntries(this.written); }
  health() {
    return { pendingBytes: this.pendingBytes, lastCommittedAt: this.lastCommittedAt, fsync: this.options.fsync ?? false,
      compression: this.options.compression ?? "gzip", gzipLevel: this.options.gzipLevel ?? 6, failure: this.failure?.message ?? null };
  }

  private flushFile(state: PendingFile, source: string): Promise<void> {
    if (state.chunks.length === 0) return state.completed;
    const contents = state.chunks.join("");
    const bytes = state.bytes;
    state.chunks = [];
    state.bytes = 0;
    const write = state.completed.then(async () => {
      if (this.failure) throw this.failure;
      const output = this.options.compression === "none" ? Buffer.from(contents) : await compress(contents, { level: this.options.gzipLevel ?? 6 });
      const handle = await fsp.open(state.file, "a");
      try {
        await handle.writeFile(output);
        if (this.options.fsync) await handle.sync();
      } finally { await handle.close(); }
      this.written.set(source, (this.written.get(source) ?? 0) + output.length);
      this.lastCommittedAt = Date.now();
    }).catch((error: unknown) => { throw this.fail(error); }).finally(() => { this.pendingBytes -= bytes; });
    state.completed = write;
    this.pending.add(write);
    void write.then(() => this.pending.delete(write), () => this.pending.delete(write));
    return write;
  }

  private fail(error: unknown) {
    if (!this.failure) {
      this.failure = error instanceof Error ? error : new Error(String(error));
      this.options.onError?.(this.failure);
    }
    return this.failure;
  }
}

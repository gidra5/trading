import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import {
  deduplicateFilesWithHardLinks,
  SequentialShardStore,
  TradingStorageLayout,
} from "@trading/storage";

const args = process.argv.slice(2);
const command = args[0] ?? "audit";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const dataDir = path.resolve(argument("data-dir") ?? path.join(repoRoot, "data"));
const layout = new TradingStorageLayout(dataDir);
const marketStore = new SequentialShardStore(layout.marketStore);
const trainingStore = new SequentialShardStore(layout.trainingStore);
const trainingReferenceRoots = [layout.trainingDatasets, layout.trainingRuns];

void main();

async function main(): Promise<void> {
  if (command === "audit") {
    print({
      market: await marketStore.audit(),
      training: await trainingStore.audit({
        externalReferenceRoots: trainingReferenceRoots,
      }),
      workspaces: {
        market: await directoryStats(layout.marketMutable),
        datasets: await directoryStats(layout.trainingDatasets),
        runs: await directoryStats(layout.trainingRuns),
        cache: await directoryStats(layout.trainingCache),
        analysis: await directoryStats(layout.trainingAnalysis),
        runtime: await directoryStats(layout.runtimeRoot),
        servingModels: await directoryStats(path.join(layout.dataRoot, "models")),
      },
    });
    return;
  }
  if (command === "gc") {
    const apply = args.includes("--apply");
    const hours = Number(argument("minimum-age-hours") ?? 24);
    if (!Number.isFinite(hours) || hours < 1) {
      throw new Error("--minimum-age-hours must be at least 1");
    }
    const [market, training] = await Promise.all([
      marketStore.collectGarbage({
        apply,
        minimumAgeMs: hours * 60 * 60 * 1_000,
      }),
      trainingStore.collectGarbage({
        apply,
        minimumAgeMs: hours * 60 * 60 * 1_000,
        externalReferenceRoots: trainingReferenceRoots,
      }),
    ]);
    print({ market, training, mode: apply ? "applied" : "dry-run" });
    if (!apply && [...market.orphanObjects, ...training.orphanObjects].some(
      (item) => item.modifiedAtMs <= Date.now() - hours * 60 * 60 * 1_000,
    )) {
      process.stdout.write(
        "Dry run only. Re-run with --apply after confirming no active writer is between object and reference commits.\n",
      );
    }
    return;
  }
  if (command === "dedupe") {
    const root = path.resolve(argument("root") ?? layout.trainingDatasets);
    if (!isInsideOrEqual(root, dataDir)) {
      throw new Error(`Deduplication root must stay inside ${dataDir}.`);
    }
    const apply = args.includes("--apply");
    if (apply && !args.includes("--confirm-no-writers")) {
      throw new Error(
        "Applying hard-link deduplication requires --confirm-no-writers after confirming "
        + "no process is creating or updating files beneath the selected root.",
      );
    }
    const minimumMiB = Number(argument("minimum-mib") ?? 0.0625);
    const hashConcurrency = Number(argument("hash-concurrency") ?? 2);
    if (!Number.isFinite(minimumMiB) || minimumMiB < 0) {
      throw new Error("--minimum-mib must be non-negative.");
    }
    let reported = 0;
    const report = await deduplicateFilesWithHardLinks({
      root,
      apply,
      minimumBytes: Math.round(minimumMiB * 1_024 * 1_024),
      hashConcurrency,
      onProgress: ({ hashedObjects, candidateObjects }) => {
        if (hashedObjects - reported >= 100 || hashedObjects === candidateObjects) {
          process.stderr.write(`Hashed ${hashedObjects}/${candidateObjects} candidate objects.\n`);
          reported = hashedObjects;
        }
      },
    });
    print(report);
    return;
  }
  throw new Error(
    "Usage: storage-maintenance.ts audit|gc|dedupe "
    + "[--data-dir data] [--root directory] [--minimum-age-hours 24] "
    + "[--minimum-mib 0.0625] [--hash-concurrency 2] [--apply]",
  );
}

async function directoryStats(root: string): Promise<{ files: number; bytes: number }> {
  let files = 0;
  let bytes = 0;
  async function walk(directory: string): Promise<void> {
    let entries: Array<import("node:fs").Dirent>;
    try {
      entries = await fs.readdir(directory, { withFileTypes: true });
    } catch (error) {
      if ((error as NodeJS.ErrnoException).code === "ENOENT") return;
      throw error;
    }
    for (const entry of entries) {
      const file = path.join(directory, entry.name);
      if (entry.isDirectory()) await walk(file);
      else if (entry.isFile()) {
        files += 1;
        bytes += (await fs.stat(file)).size;
      }
    }
  }
  await walk(root);
  return { files, bytes };
}

function argument(name: string): string | undefined {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : undefined;
}

function print(value: unknown): void {
  process.stdout.write(`${JSON.stringify(value, null, 2)}\n`);
}

function isInsideOrEqual(candidate: string, root: string): boolean {
  const relative = path.relative(path.resolve(root), path.resolve(candidate));
  return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

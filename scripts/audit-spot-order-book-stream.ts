import fs from "node:fs";
import readline from "node:readline";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const DEFAULT_INPUT = "data/market/mutable/streams/spot-btcusdt/btcusdt-orderbook.jsonl";
const DEFAULT_OUTPUT = "data/benchmarks/spot-order-book-stream-audit.json";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

interface Snapshot {
  symbol: string;
  eventTime: number;
  bids: Array<{ price: number; quantity: number }>;
  asks: Array<{ price: number; quantity: number }>;
}

async function main(): Promise<void> {
  const input = path.resolve(repoRoot, process.argv[2] ?? DEFAULT_INPUT);
  const output = path.resolve(repoRoot, process.argv[3] ?? DEFAULT_OUTPUT);
  const days = new Map<string, {
    snapshots: number;
    firstTime: number;
    lastTime: number;
    gapsOver2s: number;
    gapsOver120s: number;
    maximumGapMs: number;
  }>();
  const segments: Array<{ startTime: number; endTime: number; snapshots: number }> = [];
  let segment: { startTime: number; endTime: number; snapshots: number } | undefined;
  let lines = 0;
  let valid = 0;
  let invalid = 0;
  let previousTime = Number.NaN;
  let nonIncreasing = 0;
  let maximumGapMs = 0;
  const reader = readline.createInterface({
    input: fs.createReadStream(input, { encoding: "utf8" }),
    crlfDelay: Infinity,
  });
  for await (const line of reader) {
    lines += 1;
    let snapshot: Snapshot;
    try {
      snapshot = JSON.parse(line) as Snapshot;
    } catch {
      invalid += 1;
      continue;
    }
    if (snapshot.symbol !== "BTCUSDT"
      || !Number.isSafeInteger(snapshot.eventTime)
      || snapshot.bids.length !== 10
      || snapshot.asks.length !== 10
      || snapshot.bids.some((row) => !Number.isFinite(row.price) || !Number.isFinite(row.quantity))
      || snapshot.asks.some((row) => !Number.isFinite(row.price) || !Number.isFinite(row.quantity))) {
      invalid += 1;
      continue;
    }
    valid += 1;
    const gap = Number.isFinite(previousTime) ? snapshot.eventTime - previousTime : 0;
    if (gap <= 0 && Number.isFinite(previousTime)) nonIncreasing += 1;
    maximumGapMs = Math.max(maximumGapMs, gap);
    if (!segment || gap > 120_000) {
      if (segment) segments.push(segment);
      segment = {
        startTime: snapshot.eventTime,
        endTime: snapshot.eventTime,
        snapshots: 1,
      };
    } else {
      segment.endTime = snapshot.eventTime;
      segment.snapshots += 1;
    }
    const day = new Date(snapshot.eventTime).toISOString().slice(0, 10);
    const row = days.get(day) ?? {
      snapshots: 0,
      firstTime: snapshot.eventTime,
      lastTime: snapshot.eventTime,
      gapsOver2s: 0,
      gapsOver120s: 0,
      maximumGapMs: 0,
    };
    row.snapshots += 1;
    row.firstTime = Math.min(row.firstTime, snapshot.eventTime);
    row.lastTime = Math.max(row.lastTime, snapshot.eventTime);
    if (gap > 2_500) row.gapsOver2s += 1;
    if (gap > 120_000) row.gapsOver120s += 1;
    row.maximumGapMs = Math.max(row.maximumGapMs, gap);
    days.set(day, row);
    previousTime = snapshot.eventTime;
    if (lines % 250_000 === 0) console.error(`Audited ${lines.toLocaleString()} lines...`);
  }
  if (segment) segments.push(segment);
  const normalizedSegments = segments.map((row) => ({
    ...row,
    startTime: new Date(row.startTime).toISOString(),
    endTime: new Date(row.endTime).toISOString(),
    durationHours: (row.endTime - row.startTime) / 3_600_000,
    meanCadenceSeconds: row.snapshots > 1
      ? (row.endTime - row.startTime) / 1_000 / (row.snapshots - 1)
      : 0,
  })).sort((left, right) => right.durationHours - left.durationHours);
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    input: path.relative(repoRoot, input),
    fileBytes: fs.statSync(input).size,
    lines,
    validSnapshots: valid,
    invalidLines: invalid,
    nonIncreasingTimestamps: nonIncreasing,
    firstTime: normalizedSegments.length > 0
      ? [...normalizedSegments].sort((a, b) => a.startTime.localeCompare(b.startTime))[0]!.startTime
      : null,
    lastTime: normalizedSegments.length > 0
      ? [...normalizedSegments].sort((a, b) => b.endTime.localeCompare(a.endTime))[0]!.endTime
      : null,
    maximumGapSeconds: maximumGapMs / 1_000,
    continuousSegmentsAt120s: normalizedSegments,
    days: [...days.entries()].sort(([left], [right]) => left.localeCompare(right)).map(([day, row]) => ({
      day,
      ...row,
      firstTime: new Date(row.firstTime).toISOString(),
      lastTime: new Date(row.lastTime).toISOString(),
      maximumGapSeconds: row.maximumGapMs / 1_000,
    })),
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.relative(repoRoot, output)}`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  main().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  });
}

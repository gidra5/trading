import fs from "node:fs";
import type { IncomingMessage } from "node:http";
import https from "node:https";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const API = "https://api.gdeltproject.org/api/v2/doc/doc";
const STEP_MS = 15 * 60_000;
const CHUNK_MS = 71 * 3_600_000 + 45 * 60_000;
const DEFAULT_QUERY = "(bitcoin OR cryptocurrency OR crypto)";
const DEFAULT_START = "2026-07-18";
const DEFAULT_END = "2026-08-16";
const DEFAULT_OUTPUT = "data/market/immutable/external/gdelt-doc-crypto-15m/2026-07-18_2026-08-16.json";

interface Point { date: string; value: number; norm?: number }

export async function run(args = process.argv.slice(2)): Promise<void> {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const startDay = value("--start") ?? DEFAULT_START;
  const endDay = value("--end") ?? DEFAULT_END;
  const start = parseDay(startDay);
  const endExclusive = parseDay(endDay) + 86_400_000;
  const query = value("--query") ?? DEFAULT_QUERY;
  const output = path.resolve(root, value("--output") ?? DEFAULT_OUTPUT);
  const partialFile = path.resolve(
    root,
    value("--checkpoint") ?? "data/runtime-cache/gdelt-doc-history-2026-07-18_2026-08-16.partial.json",
  );
  const volume = new Map<number, { count: number; norm: number }>();
  const tone = new Map<number, number>();
  let resumeAt = start;
  if (fs.existsSync(partialFile)) {
    const partial = JSON.parse(fs.readFileSync(partialFile, "utf8")) as {
      query: string;
      start: number;
      endExclusive: number;
      resumeAt: number;
      volume: Array<[number, { count: number; norm: number }]>;
      tone: Array<[number, number]>;
    };
    if (partial.query !== query || partial.start !== start || partial.endExclusive !== endExclusive) {
      throw new Error(`GDELT partial checkpoint does not match the requested range/query: ${partialFile}`);
    }
    for (const item of partial.volume) volume.set(item[0], item[1]);
    for (const item of partial.tone) tone.set(item[0], item[1]);
    resumeAt = partial.resumeAt;
    console.error(`Resuming GDELT at ${new Date(resumeAt).toISOString()} from ${volume.size} checkpointed points.`);
  }
  let requests = 0;
  for (let chunkStart = resumeAt; chunkStart < endExclusive; chunkStart += CHUNK_MS + STEP_MS) {
    const chunkEnd = Math.min(endExclusive - 1_000, chunkStart + CHUNK_MS);
    const volumeResult = await fetchTimeline(query, "timelinevolraw", chunkStart, chunkEnd);
    requests += 1;
    requireResolution(volumeResult, "timelinevolraw");
    for (const point of timelinePoints(volumeResult)) {
      const time = pointTime(point.date);
      if (time < start || time >= endExclusive || !Number.isFinite(point.value)
        || !Number.isFinite(point.norm) || point.value < 0 || point.norm! <= 0) continue;
      volume.set(time, { count: point.value, norm: point.norm! });
    }
    await delay(10_250);
    const toneResult = await fetchTimeline(query, "timelinetone", chunkStart, chunkEnd);
    requests += 1;
    requireResolution(toneResult, "timelinetone");
    for (const point of timelinePoints(toneResult)) {
      const time = pointTime(point.date);
      if (time >= start && time < endExclusive && Number.isFinite(point.value)) tone.set(time, point.value);
    }
    console.error(`${new Date(chunkStart).toISOString()}..${new Date(chunkEnd).toISOString()}: ${volume.size} volume / ${tone.size} tone points`);
    const nextStart = chunkStart + CHUNK_MS + STEP_MS;
    fs.mkdirSync(path.dirname(output), { recursive: true });
    fs.writeFileSync(partialFile, `${JSON.stringify({
      query,
      start,
      endExclusive,
      resumeAt: nextStart,
      volume: [...volume],
      tone: [...tone],
    })}\n`, "utf8");
    if (chunkEnd < endExclusive - 1_000) await delay(10_250);
  }
  const rows = [];
  for (let time = start; time < endExclusive; time += STEP_MS) {
    const activity = volume.get(time);
    const sentiment = tone.get(time);
    rows.push({
      time,
      articleCount: activity?.count ?? null,
      monitoredArticleCount: activity?.norm ?? null,
      articleIntensity: activity ? activity.count / activity.norm : null,
      averageTone: sentiment ?? null,
      observed: activity !== undefined || sentiment !== undefined,
    });
  }
  if (rows.every((row) => !row.observed)) throw new Error("GDELT returned no valid observations.");
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    source: "GDELT DOC 2.0 API",
    sourceUrl: API,
    query,
    start: new Date(start).toISOString(),
    endExclusive: new Date(endExclusive).toISOString(),
    stepMs: STEP_MS,
    requests,
    timing: "retrospective article publication timeline; not equivalent to live first-observed receive time",
    missingPolicy: "missing API intervals remain null and are never converted to zero",
    observedVolumeRows: volume.size,
    observedToneRows: tone.size,
    rows,
  };
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.rmSync(partialFile, { force: true });
  console.log(`Wrote ${path.relative(root, output)}: ${rows.length} grid rows, ${volume.size} volume, ${tone.size} tone.`);
}

async function fetchTimeline(query: string, mode: string, start: number, end: number): Promise<any> {
  const url = new URL(API);
  url.search = new URLSearchParams({
    query,
    mode,
    format: "json",
    startdatetime: gdeltTime(start),
    enddatetime: gdeltTime(end),
    timelinesmooth: "0",
  }).toString();
  let failure: unknown;
  for (let attempt = 1; attempt <= 4; attempt += 1) {
    try {
      const response = await requestText(url, 2_000_000);
      if (response.status < 200 || response.status >= 300 || !response.text.trimStart().startsWith("{")) {
        throw new Error(`${mode}: HTTP ${response.status}: ${response.text.slice(0, 240)}`);
      }
      return JSON.parse(response.text);
    } catch (error) {
      failure = error;
      if (attempt < 4) await delay(30_000 * attempt);
    }
  }
  throw failure;
}

async function requestText(
  url: URL,
  maximumBytes: number,
  redirects = 4,
): Promise<{ status: number; text: string }> {
  const response = await responseFor(url, redirects);
  const chunks: Buffer[] = [];
  let bytes = 0;
  const hardTimeout = setTimeout(() => response.destroy(new Error(`${url}: response body timed out.`)), 65_000);
  try {
    for await (const chunk of response) {
      bytes += (chunk as Buffer).byteLength;
      if (bytes > maximumBytes) throw new Error(`${url}: response exceeds ${maximumBytes} bytes.`);
      chunks.push(chunk as Buffer);
    }
  } finally {
    clearTimeout(hardTimeout);
  }
  return { status: response.statusCode ?? 0, text: Buffer.concat(chunks).toString("utf8") };
}

function responseFor(url: URL, redirects: number): Promise<IncomingMessage> {
  return new Promise((resolve, reject) => {
    const request = https.get(url, {
      family: 4,
      headers: { "user-agent": "trading-gdelt-history/1.0" },
    }, (response) => {
      const status = response.statusCode ?? 0;
      if (status >= 300 && status < 400 && response.headers.location) {
        response.resume();
        if (redirects <= 0) {
          reject(new Error(`${url}: too many redirects.`));
          return;
        }
        responseFor(new URL(response.headers.location, url), redirects - 1).then(resolve, reject);
        return;
      }
      resolve(response);
    });
    request.setTimeout(60_000, () => request.destroy(new Error(`${url}: timed out.`)));
    request.once("error", reject);
  });
}

function requireResolution(result: any, mode: string): void {
  if (result?.query_details?.date_resolution !== "15m" || !Array.isArray(result?.timeline)) {
    throw new Error(`${mode}: GDELT did not return a 15-minute timeline.`);
  }
}

function timelinePoints(result: any): Point[] {
  const points = result.timeline.flatMap((series: any) => Array.isArray(series?.data) ? series.data : []);
  return points as Point[];
}

function pointTime(raw: string): number {
  if (!/^\d{8}T\d{6}Z$/.test(raw)) throw new Error(`Invalid GDELT timestamp: ${raw}`);
  const iso = `${raw.slice(0, 4)}-${raw.slice(4, 6)}-${raw.slice(6, 8)}T${raw.slice(9, 11)}:${raw.slice(11, 13)}:${raw.slice(13, 15)}.000Z`;
  const time = Date.parse(iso);
  if (!Number.isFinite(time) || time % STEP_MS !== 0) throw new Error(`Off-grid GDELT timestamp: ${raw}`);
  return time;
}

function gdeltTime(time: number): string {
  return new Date(time).toISOString().replace(/[-:T.Z]/g, "").slice(0, 14);
}

function parseDay(day: string): number {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(day)) throw new Error(`Invalid day: ${day}`);
  const time = Date.parse(`${day}T00:00:00.000Z`);
  if (!Number.isFinite(time) || new Date(time).toISOString().slice(0, 10) !== day) throw new Error(`Invalid day: ${day}`);
  return time;
}

function delay(milliseconds: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, milliseconds));
}

const invoked = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (invoked === path.resolve(import.meta.filename)) {
  run().catch((error: unknown) => {
    console.error(error instanceof Error ? error.stack ?? error.message : String(error));
    process.exitCode = 1;
  });
}

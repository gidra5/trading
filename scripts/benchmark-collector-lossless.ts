import fs from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createGunzip, gzipSync, gunzipSync, zstdCompressSync, zstdDecompressSync, constants } from "node:zlib";
import { createHash } from "node:crypto";
import { performance } from "node:perf_hooks";
import { encodeLosslessJsonlBlock, decodeLosslessJsonlBlock } from "./lib/lossless-jsonl-block.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
type Row = Readonly<{ line: string; at: number; kind: string }>;
type Candidate = Readonly<{ name: string; groupSeconds: number; compression: string; encodedBytes: number;
  originalGzipBytes: number; ratioToGzip1s: number; encodeMs: number; decodeMs: number; roundTripByteExact: true }>;
interface StreamResult { readonly kind: string; readonly rows: number; readonly baselineGzip1sBytes: number; readonly candidates: readonly Candidate[]; }
interface DatasetResult { readonly id: string; readonly file: string; readonly role: string; readonly rows: number; readonly spanSeconds: number;
  readonly selectedBytes: number; readonly originalSha256: string; readonly streams: readonly StreamResult[]; }
const option = (args: readonly string[], name: string, fallback: string) => { const i = args.indexOf(name); return i < 0 ? fallback : args[i + 1] ?? fallback; };
const relative = (file: string) => path.relative(ROOT, file).replaceAll("\\", "/");
const sha = (buffer: Buffer) => createHash("sha256").update(buffer).digest("hex");
function kindFor(file: string, row: any): string | null {
  const payload = row.payload;
  if (file.includes("binance-event-tape")) {
    if (payload?.kind !== "wire") return null;
    const stream = String(payload.stream).split("@").slice(1).join("@");
    const kind = stream.startsWith("depth") ? "depth-diff" : stream === "bookTicker" ? "BBO" : stream === "markPrice@1s" ? "mark-price-funding" : stream;
    return `${payload.venue}/${kind}`;
  }
  const source = file.replaceAll("\\", "/");
  if (source.includes("binance-spot-depth-diff/")) return "spot/depth-diff";
  if (source.includes("binance-spot-aggtrade/")) return "spot/trade";
  if (source.includes("binance-usdm-book-ticker/")) return "usdm-futures/BBO";
  if (source.includes("binance-usdm-mark-price/")) return "usdm-futures/mark-price-funding";
  return null;
}
async function filesWithin(directory: string): Promise<readonly string[]> {
  const rows = await fs.readdir(directory, { withFileTypes: true });
  const output: string[] = [];
  for (const row of rows) {
    const file = path.join(directory, row.name);
    if (row.isDirectory()) output.push(...await filesWithin(file));
    else if (row.name.endsWith(".jsonl.gz")) output.push(file);
  }
  return output;
}
async function window(file: string, maxRows: number, maxBytes: number, maxSeconds: number): Promise<readonly Row[]> {
  const source = createReadStream(file); const unzip = createGunzip(); source.pipe(unzip); unzip.setEncoding("utf8");
  const rows: Row[] = []; let text = "", bytes = 0, firstAt: number | undefined;
  const consume = (line: string): boolean => {
    const row = JSON.parse(line); const kind = kindFor(file, row); if (!kind) return true;
    const at = Number(row.recordedAt); if (!Number.isFinite(at)) throw new Error(`Invalid recordedAt in ${file}`);
    firstAt ??= at;
    if (rows.length >= maxRows || bytes + Buffer.byteLength(line) > maxBytes || at - firstAt > maxSeconds * 1000) return false;
    rows.push({ line, at, kind }); bytes += Buffer.byteLength(line); return true;
  };
  try {
    for await (const chunk of unzip) {
      text += chunk;
      let index: number;
      while ((index = text.indexOf("\n")) >= 0) {
        const line = text.slice(0, index + 1); text = text.slice(index + 1);
        if (line.trim() && !consume(line)) return rows;
      }
    }
    if (text.trim()) consume(text);
    return rows;
  } finally { source.destroy(); unzip.destroy(); }
}
function groups(rows: readonly Row[], seconds: number): readonly (readonly Row[])[] {
  const output: Row[][] = [];
  let previous: number | undefined;
  for (const row of rows) {
    const key = Math.floor(row.at / (seconds * 1000));
    if (key !== previous) { output.push([]); previous = key; }
    output.at(-1)!.push(row);
  }
  return output;
}
function benchmark(rows: readonly Row[], seconds: number, mode: "gzip6" | "zstd12" | "lexical-zstd12" | "lexical-zstd19", baseline: number): Candidate {
  let encodedBytes = 0, encodeMs = 0, decodeMs = 0;
  for (const group of groups(rows, seconds)) {
    const lines = group.map((row) => row.line); const original = Buffer.from(lines.join(""));
    let started = performance.now();
    const lexical = mode.startsWith("lexical-");
    const prepared = lexical ? encodeLosslessJsonlBlock(lines) : original;
    const encoded = mode === "gzip6" ? gzipSync(prepared, { level: 6 }) : zstdCompressSync(prepared, {
      params: { [constants.ZSTD_c_compressionLevel]: mode === "lexical-zstd19" ? 19 : 12 },
    });
    encodeMs += performance.now() - started; encodedBytes += encoded.length;
    started = performance.now();
    const decoded = mode === "gzip6" ? gunzipSync(encoded) : zstdDecompressSync(encoded);
    const restored = lexical ? Buffer.from(decodeLosslessJsonlBlock(decoded).join("")) : decoded;
    if (!original.equals(restored)) throw new Error(`Non-lossless ${mode} block: ${sha(original)} != ${sha(restored)}`);
    decodeMs += performance.now() - started;
  }
  return { name: `${mode}-${seconds}s`, groupSeconds: seconds, compression: mode, encodedBytes,
    originalGzipBytes: baseline, ratioToGzip1s: encodedBytes / baseline, encodeMs, decodeMs, roundTripByteExact: true };
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Benchmark exact JSONL recompression and experimental lexical dictionary/delta blocks.
node --import tsx scripts/benchmark-collector-lossless.ts [options]
  --output FILE       Default docs/experiments/collector-lossless-codec-2026-11-02.json
  --max-rows N        Per-source cap (default40000)
  --max-mib N         Selected uncompressed source window cap (default32)
  --max-seconds N     Per-source elapsed window cap (default3600)
  --level19           Also measure lexical zstd19 hourly blocks
  --only-full         Skip long legacy archives
  --summarize         Regenerate summary/mappings from existing measurements only
All original selected record bytes roundtrip, including raw payloads. No sampling
or lossy reduction within selected windows. Unknown kinds retain a1.0 transfer
ratio. The experimental format is not a production archive or untrusted decoder.`); return;
  }
  const output = path.resolve(ROOT, option(args, "--output", "docs/experiments/collector-lossless-codec-2026-11-02.json"));
  const maxRows = Number(option(args, "--max-rows", "40000"));
  const maxBytes = Number(option(args, "--max-mib", "32")) * 1024 * 1024;
  const maxSeconds = Number(option(args, "--max-seconds", "3600"));
  if (![maxRows, maxBytes, maxSeconds].every((value) => Number.isFinite(value) && value > 0)) throw new Error("Window caps must be positive");
  const datasets: DatasetResult[] = [];
  if (args.includes("--summarize")) {
    const previous = JSON.parse(await fs.readFile(output, "utf8"));
    datasets.push(...previous.datasets);
    await save(); return;
  }
  const inputs: Array<{ file: string; id: string; role: string }> = [];
  for (const directory of ["full-btc-smoke", "multiasset-lossless-sample"]) {
    const root = path.join(ROOT, "data/runtime-cache/collector-capacity", directory);
    for (const file of await filesWithin(root)) if (/binance-event-tape[\\/](?:spot|usdm-futures)-(?:public|market)-/.test(file)) {
      inputs.push({ file, id: `${directory}/${path.basename(path.dirname(file))}`, role: "actual-full" });
    }
  }
  if (!args.includes("--only-full")) {
    const saved = JSON.parse(await fs.readFile(path.join(ROOT, "docs/experiments/collector-storage-capacity-2026-11-01-inputs.json"), "utf8"));
    for (const item of saved.measurements as readonly Readonly<{ source: string; sampleFile: string }>[]) {
      if (!["binance-spot-aggtrade", "binance-spot-depth-diff", "binance-usdm-book-ticker", "binance-usdm-mark-price"].includes(item.source)) continue;
      inputs.push({ file: path.resolve(ROOT, item.sampleFile), id: `legacy/${item.source}`, role: "actual-legacy" });
    }
  }
  for (const input of inputs) {
    const rows = await window(input.file, maxRows, maxBytes, maxSeconds); if (!rows.length) continue;
    const streamGroups: Array<{ kind: string; rows: readonly Row[] }> = [...new Set(rows.map((row) => row.kind))].map((kind) => ({ kind, rows: rows.filter((row) => row.kind === kind) }));
    if (input.role === "actual-full") streamGroups.push({ kind: "whole-actual-shard", rows });
    const streams: StreamResult[] = [];
    for (const group of streamGroups) {
      const baselineGzip1sBytes = groups(group.rows, 1).reduce((sum, block) => sum + gzipSync(block.map((row) => row.line).join(""), { level: 6 }).length, 0);
      const candidates: Candidate[] = [];
      for (const seconds of [60, 3600]) for (const mode of ["gzip6", "zstd12", "lexical-zstd12"] as const) candidates.push(benchmark(group.rows, seconds, mode, baselineGzip1sBytes));
      if (args.includes("--level19")) candidates.push(benchmark(group.rows, 3600, "lexical-zstd19", baselineGzip1sBytes));
      streams.push({ kind: group.kind, rows: group.rows.length, baselineGzip1sBytes, candidates });
      console.log(JSON.stringify({ dataset: input.id, kind: group.kind, rows: group.rows.length,
        baselineGzip1sBytes, lexicalHourRatio: candidates.find((row) => row.name === "lexical-zstd12-3600s")!.ratioToGzip1s }));
    }
    const original = Buffer.from(rows.map((row) => row.line).join(""));
    datasets.push({ id: input.id, file: relative(input.file), role: input.role, rows: rows.length,
      spanSeconds: (rows.at(-1)!.at - rows[0]!.at) / 1000, selectedBytes: original.length, originalSha256: sha(original), streams });
    await save();
  }
  await save();
  async function save() {
    const chosenCandidate = datasets.every((dataset) => dataset.streams.every((stream) => stream.candidates.some((candidate) => candidate.name === "lexical-zstd19-3600s")))
      ? "lexical-zstd19-3600s" : "lexical-zstd12-3600s";
    const components = [...new Set(datasets.flatMap((dataset) => dataset.streams.map((stream) => stream.kind)))].filter((kind) => kind !== "whole-actual-shard");
    const componentRatios = components.map((component) => {
      const evidence = datasets.flatMap((dataset) => dataset.streams.filter((stream) => stream.kind === component).map((stream) => {
        const candidate = stream.candidates.find((row) => row.name === chosenCandidate)!;
        return { dataset: dataset.id, role: dataset.role, rows: stream.rows, spanSeconds: dataset.spanSeconds,
          originalGzipBytes: stream.baselineGzip1sBytes, encodedBytes: candidate.encodedBytes, ratio: candidate.ratioToGzip1s, roundTripByteExact: true };
      }));
      const worst = evidence.reduce((a, b) => a.ratio > b.ratio ? a : b);
      return { component, candidate: chosenCandidate, originalGzipBytes: worst.originalGzipBytes, encodedBytes: worst.encodedBytes,
        ratio: worst.ratio, roundTripByteExact: true, policy: "Worst observed candidate/baseline ratio across all corresponding actual windows; unmeasured components remain1.0", evidence };
    });
    const shardCalibration = datasets.filter((dataset) => dataset.role === "actual-full").map((dataset) => {
      const whole = dataset.streams.find((stream) => stream.kind === "whole-actual-shard")!;
      const separated = dataset.streams.filter((stream) => stream.kind !== "whole-actual-shard").reduce((sum, stream) => sum + stream.baselineGzip1sBytes, 0);
      return { dataset: dataset.id, rows: whole.rows, separatedKindsGzip1sBytes: separated,
        wholeShardGzip1sBytes: whole.baselineGzip1sBytes, wholeToSeparatedRatio: whole.baselineGzip1sBytes / separated,
        encodedBytes: whole.candidates.find((row) => row.name === chosenCandidate)!.encodedBytes,
        note: "Actual same-shard interleaving is retained. This comparison must not be multiplied independently into the component codec ratios." };
    });
    const report = { version: 1, generatedAt: new Date().toISOString(), runtime: process.version, parameters: { maxRows, maxBytes, maxSeconds },
      preservation: "Every selected original JSONL byte is reconstructed, including raw wire text, unknown fields, whitespace, ordering, decimal spelling and unsafe integer lexemes. No event is dropped within a selected window.",
      limitations: ["Legacy spot aggTrade differs from individual trade granularity; its ratio is only a conservative transfer comparison.",
        "An hourly bucket is shorter than an hour when the source window/cap is shorter; spanSeconds is reported.35s/90s smoke is not an observed full hour.",
        "Per-component1sgzip matches the previous conservative capacity calibration. Whole-actual-shard rows separately show genuine cross-stream/market amortization.",
        "Short Sunday TradFi observations do not establish weekday activity, monthly entropy or spike behavior.",
        "The lexical codec is experimental. Production archival needs bounded decode, checksums, source hashing, crash recovery, indexes and retention design; no files are deleted here."],
      chosenCandidate, componentRatios, shardCalibration, unmappedComponentRatio: 1, datasets };
    await fs.mkdir(path.dirname(output), { recursive: true }); await fs.writeFile(output, JSON.stringify(report, null, 2) + "\n");
    const percent = (value: number) => `${(value * 100).toFixed(2)}%`;
    const markdown = ["# Lossless collector compression benchmark", `Generated ${report.generatedAt}, Node ${process.version}. Experimental measurement only; no production archival integration.`,
      "Every selected original JSONL byte roundtrips: raw exchange strings, decimal spelling, unknown fields, unsafe integer lexemes, whitespace, key ordering and row ordering. No events or fields are removed within the measured windows. The lexical codec uses repeated structural templates and field dictionaries, BigInt integer/decimal deltas, and Zstandard. Decimal text is reconstructed exactly, rather than rounded or converted to floating point.",
      `Selected candidate: **${chosenCandidate}**. The table uses the **worst observed ratio per component** across the measured full-recorder and legacy windows. Baseline is the same records grouped into 1-second gzip level-6 members, matching the previous conservative per-component capacity calibration.`,
      "| Component | Remaining bytes vs baseline | Compression improvement | Evidence windows |\n| --- | ---: | ---: | ---: |\n" + componentRatios.map((row) => `| ${row.component} | ${percent(row.ratio)} | ${(1 / row.ratio).toFixed(2)}× | ${row.evidence.length} |`).join("\n"),
      "The 3× activity estimate of 9.93 TB requires remaining storage below approximately 10.07% to fit 1 TB, or 8.06% with 20% capacity reserved. These measured component ratios do not demonstrate that reduction. In particular, depth remains above half its previous gzip size in the conservative transfer, which prevents the measured design from meeting the 1 TB stressed-month target. This is a result for the measured design, not an information-theoretic impossibility proof for every lossless codec.",
      "| Actual dataset | Rows | Observed seconds | Selected uncompressed MiB |\n| --- | ---: | ---: | ---: |\n" + datasets.map((dataset) => `| ${dataset.id} | ${dataset.rows} | ${dataset.spanSeconds.toFixed(3)} | ${(dataset.selectedBytes / 2 ** 20).toFixed(2)} |`).join("\n"),
      "The 20-market capture ran for 90 seconds. Its busy public shard reaches the selected 32 MiB window cap after about 21 seconds; that shorter observed duration is explicit above. The market shard spans about 89 seconds. Legacy depth/BBO/mark windows are approximately one hour, while the trade window reaches the 40,000-row cap after about 20 minutes. Legacy spot aggregate trades have different granularity from the newer individual trade feed.",
      "| Actual shard | Whole 1s gzip / separated kinds | Chosen encoded bytes |\n| --- | ---: | ---: |\n" + shardCalibration.map((row) => `| ${row.dataset} | ${percent(row.wholeToSeparatedRatio)} | ${row.encodedBytes} |`).join("\n"),
      "These shard rows directly test cross-stream/market envelope amortization. The mixed market shard saves about 9% versus separated component gzip members; the mixed public shard is slightly larger. They do not support an additional order-of-magnitude baseline discount. Do not multiply these shard ratios into the component codec ratios as independent savings.",
      "| Dataset / component | 60s gzip ratio | Hour-bucket raw Zstd12 ratio | Hour-bucket lexical Zstd19 ratio | Chosen encode / decode ms |\n| --- | ---: | ---: | ---: | ---: |\n" + datasets.flatMap((dataset) => dataset.streams.map((stream) => {
        const chosen = stream.candidates.find((candidate) => candidate.name === chosenCandidate)!;
        const gzip = stream.candidates.find((candidate) => candidate.name === "gzip6-60s")!;
        const raw = stream.candidates.find((candidate) => candidate.name === "zstd12-3600s")!;
        return `| ${dataset.id} / ${stream.kind} | ${percent(gzip.ratioToGzip1s)} | ${percent(raw.ratioToGzip1s)} | ${percent(chosen.ratioToGzip1s)} | ${chosen.encodeMs.toFixed(1)} / ${chosen.decodeMs.toFixed(1)} |`;
      })).join("\n"),
      "An hour bucket is not an observed full hour when the source is shorter or capped. The complete JSON also records 60-second lexical/Zstd candidates, baseline and encoded byte counts, timings, source paths and SHA-256 hashes of every selected original record window. The measured codec header/payload overhead is included. Compression timings are single-run local observations, not production throughput guarantees.",
      "Unmeasured checkpoints, REST seeds, open interest, coverage/control and peripheral feed kinds retain a 1.0 transfer ratio in capacity calculations. A short Sunday mixed TradFi sample does not establish weekday entropy, month-long activity or 3× spike compression behavior. Activity extrapolation remains a separate capacity scenario.",
      "The prototype decoder is for bounded trusted benchmark inputs. It has no production integrity checksum or decoded-size/structure budget in its on-disk header; the benchmark independently compares every decoded block against the original bytes and records source SHA-256. Production packing would additionally require verified archival integrity, bounded decoding, crash recovery, indexes and retention design. No source files are deleted or replaced.",
      "Reproduce measurements:\n\n```powershell\nnode --import tsx scripts/benchmark-collector-lossless.ts --level19\nnode --import tsx --test scripts/lib/lossless-jsonl-block.test.ts\n```\n\nRegenerate summary without rerunning compression:\n\n```powershell\nnode --import tsx scripts/benchmark-collector-lossless.ts --summarize\n```",
    ].join("\n\n") + "\n";
    await fs.writeFile(output.replace(/\.json$/i, "") + ".md", markdown);
  }
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) run().catch((error) => { console.error(error); process.exitCode = 1; });

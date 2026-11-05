import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { gunzipSync, gzipSync } from "node:zlib";
import { budgetStreams, formatBytes, MONTH_SECONDS, rateClass, type RateClass, type StreamBudget } from "./lib/collector-storage.ts";
import { compressBatches } from "./measure-collector-storage.ts";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const STEM = "docs/experiments/collector-storage-capacity-2026-11-01";
type Ticker = Readonly<{ symbol: string; count: number; openTime: number; closeTime: number; quoteVolume: number }>;
type Sample = Readonly<{ venue: string; symbol: string; stream: string; rateClass: RateClass; messages: number;
  observedSeconds: number; rawBytes: number; gzipOneSecondBytes: number; gzipSixtySecondBytes: number; individualFills: number; failure: string | null }>;
type Snapshot = Readonly<{ venue: string; symbol: string; levels: number; gzipBytes: number; uncompressedBytes: number; error?: string }>;
type Archive = Readonly<{ source: string; sampleFile: string; rows: number; observedSeconds: number; calendarSpanSeconds: number;
  compressedBytes: number; uncompressedBytes: number; compressedBytesPerRow: number | null; uncompressedBytesPerRow: number | null;
  compressedBytesPerObservedSecond: number | null; rowsPerObservedSecond: number | null; coverageMethod: string }>;
type Inputs = Readonly<{ generatedAt: string; measurements: readonly Archive[]; samples: readonly Sample[]; snapshots: readonly Snapshot[];
  tickers: Readonly<Record<string, readonly Ticker[]>>; measuredAt: Readonly<Record<string, string>>; fullCapture?: FullCapture | null;
  dex?: Awaited<ReturnType<typeof dexBudget>> }>;
type FullCapture = Readonly<{ directory: string; wireMessages: number; actualGzipBytesIncludingBootstrapAndPeripherals: number;
  groups: readonly Readonly<{ venue: string; kind: string; rows: number; gzip1s: number; rawBytes: number;
    legacyGzip1s: number | null; legacyRawBytes: number | null; compressedRatio: number | null; rawRatio: number | null }>[] }>;
type Entry = Readonly<{ baseAsset: string; assetId?: string; uniqueAssetId?: string; assetClass?: string; sectors?: readonly string[];
  productSymbols: Readonly<Record<string, readonly string[]>> }>;
const median = (values: readonly number[], fallback: number): number => { const sorted = [...values].filter((x) => x > 0 && Number.isFinite(x)).sort((a, b) => a - b); return sorted.length ? sorted[Math.floor(sorted.length / 2)]! : fallback; };
const option = (args: readonly string[], name: string, fallback: string) => { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1] ?? fallback; };
const readJson = async (file: string) => JSON.parse(await fs.readFile(path.resolve(ROOT, file), "utf8"));
export type StorageParameters = Readonly<{ seconds: number; checkpointSeconds: number; checkpointGrowth: number; rawEnvelopeMultiplier: number;
  depthDerivedCompressedBytes: number; tradeDerivedCompressedBytes: number; compactCompressedBytes: number;
  tradfiFillsFloor: number; tradfiDepthFloor: number; tradfiBboFloor: number }>;
export const DEFAULT_STORAGE_PARAMETERS: StorageParameters = Object.freeze({
  seconds: MONTH_SECONDS, checkpointSeconds: 300, checkpointGrowth: 2, rawEnvelopeMultiplier: 1.5,
  depthDerivedCompressedBytes: 250, tradeDerivedCompressedBytes: 220, compactCompressedBytes: 250,
  tradfiFillsFloor: .5, tradfiDepthFloor: 5, tradfiBboFloor: 10,
});
type ProfileName = "current-btc-compact" | "all-routes-feature-only-lossy" | "all-routes-full-public-replay" | "all-routes-full-plus-derived";
type StorageProfile = Readonly<{ base: ReturnType<typeof budgetStreams>; stress2x: ReturnType<typeof budgetStreams>; stress3x: ReturnType<typeof budgetStreams> }>;

async function refreshInputs(directory: string): Promise<Inputs> {
  const archive = await readJson(path.join(directory, "archive-measurements-2026-11-01.json"));
  const live = await readJson(path.join(directory, "live-measurements-2026-11-01.json"));
  let sector: { samples: Sample[]; measuredAt: string } = { samples: [], measuredAt: "unmeasured" };
  try { sector = await readJson(path.join(directory, "sector-live-measurements-2026-11-01.json")); } catch { /* optional second pilot */ }
  const rest = await readJson(path.join(directory, "rest-measurements-2026-11-01.json"));
  return { generatedAt: new Date().toISOString(), measurements: archive.measurements, samples: [...live.samples, ...sector.samples], snapshots: rest.snapshots,
    tickers: Object.fromEntries(Object.entries(rest.tickers).map(([venue, rows]) => [venue, (rows as Ticker[]).map((row) => ({ symbol: row.symbol,
      count: Number(row.count), openTime: row.openTime, closeTime: row.closeTime, quoteVolume: Number(row.quoteVolume) }))])),
    measuredAt: { archives: archive.measuredAt, live: live.measuredAt, sectors: sector.measuredAt, rest: rest.measuredAt },
    fullCapture: await measureFullSmoke("data/runtime-cache/collector-capacity/full-btc-smoke"), dex: await dexBudget() };
}

async function measureFullSmoke(directory: string): Promise<FullCapture | null> {
  try {
    const audit = await readJson(path.join(directory, "audit.json"));
    const groups = new Map<string, { venue: string; kind: string; full: string[]; legacy: string[] }>();
    async function scan(relative: string): Promise<void> {
      for (const item of await fs.readdir(path.resolve(ROOT, relative), { withFileTypes: true })) {
        const name = path.join(relative, item.name);
        if (item.isDirectory()) { await scan(name); continue; }
        if (!item.name.endsWith(".gz")) continue;
        for (const line of gunzipSync(await fs.readFile(path.resolve(ROOT, name))).toString().split("\n").filter(Boolean)) {
          const record = JSON.parse(line), payload = record.payload;
          if (!payload?.venue || !["wire", "depth_bbo", "trade_book_alignment"].includes(payload.kind)) continue;
          const kind = payload.kind === "wire" ? String(payload.stream).split("@").slice(1).join("@") : payload.kind;
          const key = `${payload.venue}:${kind}`;
          const group: { venue: string; kind: string; full: string[]; legacy: string[] } = groups.get(key) ?? { venue: payload.venue, kind, full: [], legacy: [] };
          group.full.push(line + "\n");
          if (payload.kind === "wire") {
            const decoded = JSON.parse(payload.raw);
            group.legacy.push(JSON.stringify({ recordedAt: record.recordedAt,
              payload: { receivedAt: payload.receivedAt, monotonicNs: payload.monotonicNs, message: decoded.data ?? decoded } }) + "\n");
          }
          groups.set(key, group);
        }
      }
    }
    await scan(path.join(directory, "binance-event-tape"));
    return { directory, wireMessages: audit.kindCounts.wire, actualGzipBytesIncludingBootstrapAndPeripherals: audit.compressedBytes,
      groups: [...groups.values()].map((group) => {
        const gzip1s = compressBatches(group.full, 1000), rawBytes = group.full.reduce((sum, line) => sum + Buffer.byteLength(line), 0);
        const legacyGzip1s = group.legacy.length ? compressBatches(group.legacy, 1000) : null,
          legacyRawBytes = group.legacy.length ? group.legacy.reduce((sum, line) => sum + Buffer.byteLength(line), 0) : null;
        return { venue: group.venue, kind: group.kind, rows: group.full.length, gzip1s, rawBytes, legacyGzip1s, legacyRawBytes,
          compressedRatio: legacyGzip1s ? gzip1s / legacyGzip1s : null, rawRatio: legacyRawBytes ? rawBytes / legacyRawBytes : null };
      }) };
  } catch { return null; }
}

export function buildEstimate(entries: readonly Entry[], inputs: Inputs, parameters: StorageParameters = DEFAULT_STORAGE_PARAMETERS) {
  for (const [name, value] of Object.entries(parameters)) if (!Number.isFinite(value) || value <= 0) throw new Error(`Invalid positive parameter ${name}`);
  const full: StreamBudget[] = [], derived: StreamBudget[] = [], featureOnly: StreamBudget[] = [], routeRows: Array<Record<string, unknown>> = [];
  const seen = new Set<string>();
  const archive = (source: string) => inputs.measurements.find((row) => row.source === source);
  const add = (target: StreamBudget[], source: string, recordsPerSecond: number, raw: number, compressed: number, evidence: string) =>
    target.push({ source, recordsPerSecond, uncompressedBytesPerRecord: raw, compressedBytesPerRecord: compressed, evidence });
  for (const entry of entries) for (const [venue, symbols] of Object.entries(entry.productSymbols)) for (const symbol of symbols) {
    if (venue !== "spot" && venue !== "usdm-futures") continue;
    const key = `${venue}:${symbol}`; if (seen.has(key)) continue; seen.add(key);
    const firstFullRow = full.length;
    const ticker = inputs.tickers[venue]?.find((row) => row.symbol === symbol);
    const seconds = ticker ? (ticker.closeTime - ticker.openTime) / 1000 : 0;
    const observedFills = ticker && seconds > 0 ? ticker.count / seconds : null;
    const identity = entry.assetId ?? entry.uniqueAssetId ?? `${entry.assetClass ?? "crypto"}:${entry.baseAsset}`;
    const tradfi = /^tradfi:/.test(identity) || entry.assetClass === "tradfi" || (entry.sectors?.some((x) => /equity|commodity|forex|stock|index|premarket/i.test(x)) ?? false);
    const fills = Math.max(observedFills ?? .1, tradfi ? parameters.tradfiFillsFloor : .001);
    const tier = rateClass(fills);
    const choose = (kind: string) => {
      const candidates = inputs.samples.filter((row) => row.venue === venue && row.stream.includes(`@${kind}`) && !row.failure && row.messages > 0);
      const exact = candidates.filter((row) => row.symbol === symbol);
      // Prefer this market's bounded measurements; otherwise a matched activity class.
      const matched = candidates.filter((row) => row.rateClass === tier);
      return exact.length ? exact : matched.length ? matched : candidates;
    };
    const depthSamples = choose("depth"), bboSamples = choose("bookTicker"), tradeSamples = choose(venue === "spot" ? "trade" : "aggTrade");
    const fallbackDepth = tier === "quiet" ? .5 : tier === "medium" ? 3 : 8;
    const depthRate = Math.max(median(depthSamples.map((row) => row.messages / row.observedSeconds), fallbackDepth), tradfi ? parameters.tradfiDepthFloor : .1);
    const bboRate = Math.max(median(bboSamples.map((row) => row.messages / row.observedSeconds), fills * 3), tradfi ? parameters.tradfiBboFloor : .1);
    const aggregation = venue === "spot" ? 1 : Math.max(1, median(tradeSamples.map((row) => row.individualFills / row.messages), 1));
    const tradeRate = fills / aggregation;
    const shape = (samples: readonly Sample[], fallbackRaw: number, fallbackGzip: number) => ({ raw: median(samples.map((row) => row.rawBytes / row.messages), fallbackRaw),
      gzip: median(samples.map((row) => row.gzipOneSecondBytes / row.messages), fallbackGzip) });
    const depth = shape(depthSamples, 700, 200), bbo = shape(bboSamples, 280, 100), trade = shape(tradeSamples, 280, 150);
    const evidence = `24h fills + ${tier} activity class; byte shape from bounded samples, full-envelope ratio calibrated from actual 35s tape where available`;
    for (const [kind, rate, size] of [["depth-diff", depthRate, depth], ["BBO", bboRate, bbo], [venue === "spot" ? "trade" : "aggTrade", tradeRate, trade]] as const) {
      const measured = inputs.fullCapture?.groups.find((row) => row.venue === venue && row.kind.startsWith(kind === "depth-diff" ? "depth@" : kind === "BBO" ? "bookTicker" : kind));
      const calibrationScale = parameters.rawEnvelopeMultiplier / DEFAULT_STORAGE_PARAMETERS.rawEnvelopeMultiplier;
      add(full, `${key}/${kind}`, rate,
        measured?.rawRatio ? size.raw * measured.rawRatio * calibrationScale : size.raw * parameters.rawEnvelopeMultiplier + 300,
        size.gzip * (measured?.compressedRatio ? measured.compressedRatio * calibrationScale : parameters.rawEnvelopeMultiplier), evidence);
    }
    // The implementation persists these derived rows in addition to the exact raw stream.
    const measuredDepth = inputs.fullCapture?.groups.find((row) => row.venue === venue && row.kind === "depth_bbo"),
      measuredTrade = inputs.fullCapture?.groups.find((row) => row.venue === venue && row.kind === "trade_book_alignment");
    add(derived, `${key}/depth-bbo-derived`, depthRate, measuredDepth ? measuredDepth.rawBytes / measuredDepth.rows : 1200,
      measuredDepth ? measuredDepth.gzip1s / measuredDepth.rows * parameters.depthDerivedCompressedBytes / DEFAULT_STORAGE_PARAMETERS.depthDerivedCompressedBytes : parameters.depthDerivedCompressedBytes,
      "optional derived rows, default off; actual 35s tape 1s gzip calibration where available");
    add(derived, `${key}/trade-book-alignment`, tradeRate, measuredTrade ? measuredTrade.rawBytes / measuredTrade.rows : 1100,
      measuredTrade ? measuredTrade.gzip1s / measuredTrade.rows * parameters.tradeDerivedCompressedBytes / DEFAULT_STORAGE_PARAMETERS.tradeDerivedCompressedBytes : parameters.tradeDerivedCompressedBytes,
      "optional derived rows, default off; actual 35s tape 1s gzip calibration where available");
    const snapshots = inputs.snapshots.filter((row) => row.venue === venue && !row.error && row.gzipBytes > 0);
    const own = snapshots.find((row) => row.symbol === symbol);
    const classSnapshots = snapshots.filter((row) => {
      const ticker = inputs.tickers[venue]?.find((ticker) => ticker.symbol === row.symbol);
      return ticker && rateClass(ticker.count / Math.max(1, (ticker.closeTime - ticker.openTime) / 1000)) === tier;
    });
    const candidates = classSnapshots.length ? classSnapshots : snapshots;
    const seedGzip = own?.gzipBytes ?? median(candidates.map((row) => row.gzipBytes), venue === "spot" ? 30_000 : 10_000);
    const seedRaw = own?.uncompressedBytes ?? median(candidates.map((row) => row.uncompressedBytes), venue === "spot" ? 200_000 : 45_000);
    add(full, `${key}/local-checkpoint`, 1 / parameters.checkpointSeconds, seedRaw * parameters.checkpointGrowth, seedGzip * parameters.checkpointGrowth,
      "measured depth seed sizes; 2x growth reserve for bounded local known-level checkpoint, not a fresh exchange-wide snapshot");
    add(full, `${key}/REST-seed-and-resync`, 2 / 86_400, seedRaw, seedGzip, "planning: initial/reconnect/reseed equivalent 2/day; checkpoints are local and do not spend REST weights");
    add(full, `${key}/session-gap-health`, 1 / 60, 700, 250, "planning allowance 1 metadata row/min/route");
    add(featureOnly, `${key}/one-second-book-and-flow-summary`, 1, 1200, parameters.compactCompressedBytes,
      "hypothetical lossy feature-only profile; 250B/s/route allowance informed by 5-venue compact archive; not implemented full collector");
    if (venue === "usdm-futures") for (const target of [full, featureOnly]) {
      const mark = archive("binance-usdm-mark-price");
      const measuredMark = inputs.fullCapture?.groups.find((row) => row.venue === venue && row.kind === "markPrice@1s");
      add(target, `${key}/mark-price-funding`, 1, measuredMark ? measuredMark.rawBytes / measuredMark.rows : (mark?.uncompressedBytesPerRow ?? 400) * parameters.rawEnvelopeMultiplier,
        measuredMark ? measuredMark.gzip1s / measuredMark.rows : (mark?.compressedBytesPerRow ?? 205) * parameters.rawEnvelopeMultiplier,
        "archive + actual 35s full-tape 1s gzip calibration where available");
      add(target, `${key}/open-interest`, 1 / 300, 600, 250, "unmeasured conservative public REST rotating 300s allowance");
    }
    const routeBudget = budgetStreams(full.slice(firstFullRow), parameters.seconds);
    routeRows.push({ assetId: identity, baseAsset: entry.baseAsset, venue, symbol, tradfi, tickerFillsPerSecond: observedFills, planningFillsPerSecond: fills,
      tier, depthRate, bboRate, tradeMessageRate: tradeRate, fillsPerAggregate: aggregation, seedGzipBytes: seedGzip,
      sampledDepth: depthSamples.some((row) => row.symbol === symbol), sampledBbo: bboSamples.some((row) => row.symbol === symbol),
      monthlyGzipBytes: routeBudget.compressedBytes, monthlyJsonlBytes: routeBudget.uncompressedBytes });
  }
  const external: StreamBudget[] = [];
  const externalNames = ["binance-usdm-liquidations", "coinbase-btcusd-level2", "kraken-btcusd-book", "deribit-btc-perpetual-book",
    "deribit-btc-perpetual-trades", "deribit-btc-option-trades", "cross-exchange-book-1s", "deribit-btc-option-summary", "mempool-live", "gdelt-crypto-news", "binance-usdm-premium-index"];
  for (const source of externalNames) {
    const row = archive(source); if (row?.rowsPerObservedSecond && row.compressedBytesPerRow && row.uncompressedBytesPerRow) {
      add(external, `BTC-peripherals/${source}`, row.rowsPerObservedSecond, row.uncompressedBytesPerRow, row.compressedBytesPerRow, `archive measured ${row.sampleFile}, gap-clipped source support; no scaling to all assets`);
    }
  }
  // Complete option surface is large but slow. Explicit planning reserve because the archived raw files are nested snapshot objects.
  add(external, "BTC-peripherals/option-surface-raw", 1 / 900, 500_000, 75_000, "unmeasured 15-minute full-surface allowance; matches current collector cadence");
  const fullWithExternal = [...full, ...external];
  const featureWithExternal = [...featureOnly, ...external.filter((row) => !/level2|kraken.*book|perpetual-book/.test(row.source))];
  const currentCompact = inputs.measurements.filter((row) => !/depth-diff|depth-snapshot|book-ticker|level2|kraken.*book|perpetual-book/.test(row.source))
    .filter((row) => row.rowsPerObservedSecond && row.compressedBytesPerRow && row.uncompressedBytesPerRow)
    .map((row): StreamBudget => ({ source: `current-BTC/${row.source}`, recordsPerSecond: row.rowsPerObservedSecond!,
      uncompressedBytesPerRecord: row.uncompressedBytesPerRow!, compressedBytesPerRecord: row.compressedBytesPerRow!, evidence: row.sampleFile }));
  add(currentCompact, "current-BTC/option-surface-raw", 1 / 900, 500_000, 75_000, "15-minute raw-surface planning allowance, not included in JSONL archive scan");
  const profileStreams: Readonly<Record<ProfileName, readonly StreamBudget[]>> = {
    "current-btc-compact": currentCompact, "all-routes-feature-only-lossy": featureWithExternal,
    "all-routes-full-public-replay": fullWithExternal, "all-routes-full-plus-derived": [...fullWithExternal, ...derived],
  };
  const profiles = Object.fromEntries(Object.entries(profileStreams).map(([name, source]) => {
    return [name, { base: budgetStreams(source, parameters.seconds), stress2x: budgetStreams(source, parameters.seconds, 2), stress3x: budgetStreams(source, parameters.seconds, 3) }];
  })) as Readonly<Record<ProfileName, StorageProfile>>;
  return { parameters, routeCount: routeRows.length, assetCount: entries.length, routes: routeRows,
    routeCounts: { spot: routeRows.filter((row) => row.venue === "spot").length, futures: routeRows.filter((row) => row.venue === "usdm-futures").length,
      tradfi: routeRows.filter((row) => row.tradfi).length, tickerMissing: routeRows.filter((row) => row.tickerFillsPerSecond === null).length },
    classCounts: Object.fromEntries(["spot", "usdm-futures"].map((venue) => [venue, Object.fromEntries(["quiet", "medium", "busy", "very-busy"].map((tier) => [tier, routeRows.filter((row) => row.venue === venue && row.tier === tier).length]))])), profiles };
}

async function dexBudget() {
  const file = path.resolve(ROOT, "data/runtime-cache/collector-capacity/hyperliquid-btc-smoke.jsonl.gz");
  try {
    const meta = await readJson("data/runtime-cache/collector-capacity/hyperliquid-btc-smoke.json");
    const rows = gunzipSync(await fs.readFile(file)).toString().split("\n").filter(Boolean).map((line) => ({ line: line + "\n", row: JSON.parse(line) }));
    const streams: StreamBudget[] = [], measurements = [];
    for (const channel of ["trades", "l2Book", "bbo"]) {
      let matching = rows.filter(({ row }) => row.payload.message.channel === channel);
      if (channel === "trades") matching = matching.slice(1); // Subscription sends historical recent trades first.
      const lines = matching.map(({ line }) => line), raw = lines.reduce((sum, line) => sum + Buffer.byteLength(line), 0);
      const gzip1s = compressBatches(lines, 1000), gzipWhole = lines.length ? gzipSync(lines.join(""), { level: 6 }).length : 0;
      measurements.push({ channel, messages: matching.length, seconds: meta.seconds, rawBytes: raw, gzip1s, gzipWhole, initializationTradeReplayExcluded: channel === "trades" });
      if (matching.length) streams.push({ source: `Hyperliquid-BTC/${channel}`, recordsPerSecond: matching.length / meta.seconds,
        uncompressedBytesPerRecord: raw / matching.length, compressedBytesPerRecord: gzip1s / matching.length,
        evidence: "20-second illustrative Sunday pilot; L2 is periodic limited-depth snapshot, not full order lifecycle" });
    }
    const publicTape = budgetStreams(streams);
    const accounts = [0, 1, 10].map((users) => ({ users,
      monthly: budgetStreams([{ source: "watched-participant-events", recordsPerSecond: users * 2, uncompressedBytesPerRecord: 1500,
        compressedBytesPerRecord: 400, evidence: "unmeasured allowance 2 messages/s/address across order/fill/state channels; real active accounts may be much larger" }]) }));
    return { measurements, publicTape, publicTapeStress3x: budgetStreams(streams, MONTH_SECONDS, 3), accounts,
      limitation: "Optional public BTC pilot plus bounded watched addresses only. Address is not economic identity; strategy and off-venue state remain latent. No population-wide account-state collection or exact L3 archive is assumed." };
  } catch (error) { return { unavailable: String(error) }; }
}

export async function run(args = process.argv.slice(2)) {
  if (args.includes("--help")) {
    console.log(`Offline storage estimate from saved public measurements. Defaults to 30 days / 300s local checkpoints.
  --universe FILE --inputs FILE --output-stem FILE
  --refresh-inputs --measurements-dir DIR
  --days N --checkpoint-seconds N --checkpoint-growth N
  --raw-envelope-multiplier N --depth-derived-bytes N --trade-derived-bytes N
  --tradfi-fills-floor N --tradfi-depth-floor N --tradfi-bbo-floor N
Refresh acquisition separately with scripts/measure-collector-storage.ts; this estimator opens no sockets.`);
    return;
  }
  const stem = option(args, "--output-stem", STEM), inputFile = option(args, "--inputs", `${STEM}-inputs.json`);
  const inputs: Inputs = args.includes("--refresh-inputs") ? await refreshInputs(option(args, "--measurements-dir", "data/research/collector-storage")) : await readJson(inputFile);
  if (args.includes("--refresh-inputs")) await fs.writeFile(path.resolve(ROOT, inputFile), JSON.stringify(inputs, null, 2) + "\n");
  const universeFile = option(args, "--universe", "docs/portfolio/restored-market-universe.json"), universeText = await fs.readFile(path.resolve(ROOT, universeFile), "utf8");
  const number = (flag: string, fallback: number) => Number(option(args, flag, String(fallback)));
  const parameters: StorageParameters = { ...DEFAULT_STORAGE_PARAMETERS, seconds: number("--days", 30) * 86_400,
    checkpointSeconds: number("--checkpoint-seconds", 300), checkpointGrowth: number("--checkpoint-growth", 2),
    rawEnvelopeMultiplier: number("--raw-envelope-multiplier", 1.5), depthDerivedCompressedBytes: number("--depth-derived-bytes", 250),
    tradeDerivedCompressedBytes: number("--trade-derived-bytes", 220), tradfiFillsFloor: number("--tradfi-fills-floor", .5),
    tradfiDepthFloor: number("--tradfi-depth-floor", 5), tradfiBboFloor: number("--tradfi-bbo-floor", 10) };
  const universe = JSON.parse(universeText), estimate = buildEstimate(universe.entries, inputs, parameters);
  const dex = inputs.dex ?? await dexBudget();
  const report = { version: 1, generatedAt: new Date().toISOString(), universe: { file: universeFile, sha256: createHash("sha256").update(universeText).digest("hex"), generatedAt: universe.generatedAt },
    inputFile, measuredAt: inputs.measuredAt, ...estimate, dex,
    caveats: ["Capacity scenarios, not guaranteed event-rate forecasts. 60s crypto and 20s sector samples are short, Sunday-biased observations.",
      "TradFi floors budget 24/7 operation even when underlying cash markets close; no zero-activity month is extrapolated from Sunday.",
      "Full public replay means exact published L2 deltas, trades or exchange aggregates, BBO, bounded checkpoints and explicit gaps. Public L2 cannot identify cancellations separately from fills or expose participant IDs.",
      "Rows with unavailable credentials/history are not invented as stored feeds. Existing peripheral collection is BTC-specific, not every asset.",
      "Compressed baseline uses per-source one-second gzip members. Rich envelope overhead and optional derived-row sizes are calibrated from the 35s full-recorder smoke; broad market traffic and quiet-feed transfer remain uncertain.",
      "Archive observed time is a receive-time gap-clipped support heuristic, not calendar span, connectivity certification or a guarantee of complete delivery.",
      "Stress multipliers cover changing activity, changed levels per depth message and size growth, not merely message cadence; maintain separate free-space and backup capacity.",
      "Binary/Parquet/Zstd alternatives have not been measured; do not apply an assumed 5x or 10x saving to provision storage."] };
  await fs.writeFile(path.resolve(ROOT, `${stem}.json`), JSON.stringify(report, null, 2) + "\n");
  const full = estimate.profiles["all-routes-full-public-replay"]!, fullDerived = estimate.profiles["all-routes-full-plus-derived"]!, feature = estimate.profiles["all-routes-feature-only-lossy"]!, compact = estimate.profiles["current-btc-compact"]!;
  const total = (profile: typeof full) => `| ${formatBytes(profile.base.compressedBytes)} | ${formatBytes(profile.stress2x.compressedBytes)} | ${formatBytes(profile.stress3x.compressedBytes)} |`;
  const lines = ["# Collector storage capacity — 2026-11-01", "", `For **${estimate.assetCount} manifest entries / ${estimate.routeCount} distinct Binance venue+symbol routes** (${estimate.routeCounts.spot} spot, ${estimate.routeCounts.futures} USD-M), budget ${parameters.seconds / 86_400} days of 24/7 recording: **${parameters.seconds.toLocaleString("en-US")} seconds**. The main raw-public-replay estimate is **${formatBytes(full.base.compressedBytes)}**; the 2–3x stress range is **${formatBytes(full.stress2x.compressedBytes)}–${formatBytes(full.stress3x.compressedBytes)}**. These figures include public peripheral feeds and explicit metadata/checkpoint allowances. Optional derived rows are shown separately. Source snapshots were taken November 1; later report regeneration does not refresh those observations.`, "",
    "| Profile | Base gzip | 2x volume | 3x volume |", "|---|---:|---:|---:|", `| Current BTC compact collector, with raw trade feeds | ${total(compact).slice(2)}`,
    `| All routes, hypothetical 1s features only (lossy) | ${total(feature).slice(2)}`, `| All routes, full public replay + current BTC peripherals (default) | ${total(full).slice(2)}`,
    `| Full replay plus optional per-event derived rows | ${total(fullDerived).slice(2)}`, "",
    `Full profile uncompressed JSONL: **${formatBytes(full.base.uncompressedBytes)}** base. Add at least 25% free-space/rotation/index allowance: **${formatBytes(full.stress3x.compressedBytes * 1.25)}** usable capacity for the 3x case. An independent backup adds another copy. This is a capacity envelope; the short sample cannot establish a month-long stochastic traffic distribution.`, "",
    "## What is retained", "", "The default full profile retains spot individual trades, USD-M aggregate trades at the exchange's available granularity, every published 100ms L2 diff, unthrottled BBO, local 5-minute checkpoints, session/sequence/gap markers, 1s futures mark/funding and rotating 300s open interest. REST initial/resync seed allowance is two per route per day. Checkpoints are local reconstructed known levels, not repeated REST downloads. Seed sizes are measured at 5,000 spot / 1,000 futures levels per side; the model reserves 2x their size for checkpoint growth. Optional per-depth BBO and per-trade causal book-alignment rows duplicate reconstructible information and are off by default; the fourth profile includes their explicit storage allowance. Turning those derived rows off preserves public replay quality.", "",
    "Public L2 does not distinguish cancellations from executions, reveal trader IDs, or provide complete queue/L3 state. Sparse local checkpoints cannot restore unknown depth beyond the initial REST range. Sequence gaps remain gaps until resynchronized. The full profile is replay of what the public feed publishes, not all matching-engine events.", "",
    "The feature-only row is a hypothetical lower-volume design. It does not preserve within-second order, cancellation/depletion paths or event timing, and is not selectable as the implemented multi-asset full collector. The existing BTC compact collector also keeps raw aggregate/perpetual/option trades, so it is not a pure feature-only profile.", "",
    "## Empirical basis and assumptions", "", `Inputs: [saved measurement basis](${path.basename(inputFile)}); machine-readable [estimate](${path.basename(stem)}.json) and [per-route CSV](${path.basename(stem)}.csv). Universe SHA-256: \`${report.universe.sha256}\`.`, "",
    `Activity classes per venue: \`${JSON.stringify(estimate.classCounts)}\`. TradFi routes: **${estimate.routeCounts.tradfi}**; missing ticker rows: **${estimate.routeCounts.tickerMissing}**. Spot message counts use 24-hour individual fill counts. Futures aggregate rates divide fill counts by the empirically sampled fills-per-message ratio. Depth/BBO rates and bytes use measured own-market samples where available, then matched venue/activity classes. Crypto tiers are <0.1, <1, <10 and ≥10 fills/s. Markets are not all assigned BTC rates.`, "",
    `TradFi receives positive planning floors of ${estimate.parameters.tradfiFillsFloor} fills/s, ${estimate.parameters.tradfiDepthFloor} L2 messages/s and ${estimate.parameters.tradfiBboFloor} BBO messages/s, running 24/7, because the November 1 observation was Sunday. Those are conservative assumptions, not measured weekday averages or claims about venue trading hours.`, "",
    inputs.fullCapture ? `Actual full-recorder calibration uses **${inputs.fullCapture.wireMessages.toLocaleString("en-US")} wire messages** from the 35s BTC spot/futures smoke in \`${inputs.fullCapture.directory}\`. For each venue/stream, the same messages are encoded using the actual full envelope and a reconstructed legacy envelope, then regrouped into 1s gzip level-6 members. The measured byte ratio scales each activity class's legacy-shaped samples. Initial snapshots/control records are excluded from that ratio and budgeted separately; the complete smoke's ${inputs.fullCapture.actualGzipBytesIncludingBootstrapAndPeripherals.toLocaleString("en-US")} gzip bytes are not divided by 35s and extrapolated as recurring traffic.`
      : `No full-recorder calibration is saved; raw envelopes use the ${parameters.rawEnvelopeMultiplier}x planning allowance and optional derived rows use ${parameters.depthDerivedCompressedBytes}/${parameters.tradeDerivedCompressedBytes} compressed bytes per row.`, "",
    ...(inputs.fullCapture ? ["| Full-recorder stream | Rows | gzip B/row, 1s groups | Full / legacy gzip |", "|---|---:|---:|---:|",
      ...inputs.fullCapture.groups.map((row) => `| ${row.venue}:${row.kind} | ${row.rows.toLocaleString("en-US")} | ${(row.gzip1s / row.rows).toFixed(1)} | ${row.compressedRatio?.toFixed(3) ?? "derived row"} |`), "",
      `At default parameters, these measured ratios and derived-row sizes replace the fallback 1.5x / 250B / 220B assumptions. The raw-envelope CLI multiplier scales calibration relative to 1.5; the derived-byte CLI values scale their measured sizes relative to 250/220. Only a missing calibration uses those fallback sizes literally. Quiet markets may have different envelope amortization; the 2–3x scenarios cover volume sensitivity rather than establish confidence bounds.`] : []), "",
    "Measured legacy archive examples (largest available file/source, not all history):", "", "| Source | Rows | Supported hours | gzip B/row | JSONL B/row |", "|---|---:|---:|---:|---:|",
    ...inputs.measurements.map((row) => `| ${row.source} | ${row.rows.toLocaleString("en-US")} | ${(row.observedSeconds / 3600).toFixed(2)} | ${(row.compressedBytesPerRow ?? 0).toFixed(1)} | ${(row.uncompressedBytesPerRow ?? 0).toFixed(1)} |`), "",
    "Supported hours exclude entire receive-time deltas above 60s for event streams, 2.5s for 1s streams and 2.5 polling periods for low-cadence rows. This deliberately does not divide by the Aug 17–30 calendar span. It is a heuristic: it can exclude genuine sparse-market silence and cannot detect short packet losses. Single seed snapshots have no observation-duration rate; their bytes are used only per snapshot.", "",
    "Compression experiments group the same received rows into synthetic 1s and 60s gzip level-6 members. Larger batches reduce repetitive metadata overhead, especially on quiet feeds; the input JSON retains both measured byte totals. The live production assumption remains 1s batches. Binary/Parquet/Zstd savings are unmeasured and are not credited. Checkpoint gzip is measured as a standalone snapshot, not a long homogeneous tape.", "",
    `For evaluating an optimized fixed-schema binary/Parquet+Zstd archive, an **unmeasured what-if** range of 0.4–0.8 times this gzip base would be **${formatBytes(full.base.compressedBytes * .4)}–${formatBytes(full.base.compressedBytes * .8)}**. This is a sensitivity assumption, not an observed compression ratio or a provisioning recommendation. Preserve raw payloads or a proven lossless reconstruction path and benchmark representative busy and quiet days before claiming those savings.`, "",
    "## Optional participant-observable DEX pilot", "", "Hyperliquid BTC trades/BBO/limited-depth periodic L2 snapshots are separate from the Binance total. The 20s sample excludes the initial replay of historical trades for rate fitting. A 12s watched-address smoke confirms channel accessibility but includes initialization and is not a population-rate estimate. The configurable account allowance is 2 messages/s/address at 400 compressed bytes/message across order/fill/state updates. Ten watched accounts are a bounded pilot, not all exchange participants. Address remains a proxy for identity; economic entity, strategy, non-actions and off-venue information are latent.", "",
    `\`\`\`json\n${JSON.stringify(dex.publicTape ? { publicTapeMonthGzipBytes: dex.publicTape.compressedBytes, publicTape3xBytes: dex.publicTapeStress3x.compressedBytes,
      accountAllowances: dex.accounts.map((row) => ({ watchedAccounts: row.users, monthGzipBytes: row.monthly.compressedBytes })) } : dex, null, 2)}\n\`\`\``, "",
    "## Reproduce", "", "```powershell", `node --import tsx scripts/estimate-collector-storage.ts --universe ${universeFile}`, "node --import tsx --test scripts/lib/collector-storage.test.ts", "```", "",
    "Optional measurement refresh (opens bounded public connections for the live/REST commands; the archive audit is local):", "", "```powershell",
    "node --import tsx scripts/measure-collector-storage.ts --sample-seconds 60",
    "node --import tsx scripts/measure-collector-storage.ts --sample-seconds 20 --output data/research/collector-storage/sector-live-measurements-2026-11-01.json",
    "node --import tsx scripts/measure-collector-storage.ts --rest",
    "node --import tsx scripts/measure-collector-storage.ts --audit-archives",
    "node --import tsx scripts/estimate-collector-storage.ts --refresh-inputs", "```", "",
    "The measurement modes save distinct live/rest/archive files. All live samples stop themselves; none starts a persistent collector. The normal estimate command is offline and reads the committed input basis, including the optional DEX sample budget.", "",
    ...report.caveats.map((text) => `- ${text}`), ""];
  await fs.writeFile(path.resolve(ROOT, `${stem}.md`), lines.join("\n"));
  const columns = Object.keys(estimate.routes[0] ?? {}); const quote = (value: unknown) => `"${String(value ?? "").replaceAll('"', '""')}"`;
  await fs.writeFile(path.resolve(ROOT, `${stem}.csv`), [columns.join(","), ...estimate.routes.map((row) => columns.map((key) => quote(row[key])).join(","))].join("\n") + "\n");
  console.log(JSON.stringify({ assets: estimate.assetCount, routes: estimate.routeCounts, baseGzipBytes: full.base.compressedBytes, stress3xGzipBytes: full.stress3x.compressedBytes }));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) run().catch((error) => { console.error(error); process.exitCode = 1; });

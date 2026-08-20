import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import {
  loadFastAnalysisInputs,
  type FeatureSeries,
} from "./analyze-live-fast-features.ts";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_INPUT = "data/market/mutable/external-live";
const DEFAULT_OUTPUT = "data/runtime-cache/live-component-feature-bases";
const HORIZONS = [1, 5, 15, 60] as const;

interface Candidate {
  id: string;
  family: string;
  source: string;
  seriesIndex: number;
  construction: string;
  lookback: string;
}

export function describeLiveFeature(id: string) {
  const mean = id.match(/_mean_(\d+)s$/);
  if (mean) return {
    lookback: `${mean[1]}s`,
    construction: `trailing ${mean[1]}s arithmetic mean of the completed 1s coordinate; requires at least 80% observed seconds`,
  };
  const event = id.match(/_(count|amount|imbalance)_(\d+)s$/);
  if (event) {
    const construction = event[1] === "count"
      ? `log1p event count over the trailing ${event[2]}s`
      : event[1] === "amount"
        ? `log1p exchange-reported amount summed over the trailing ${event[2]}s`
        : `signed amount divided by total amount over the trailing ${event[2]}s; zero when no event occurs`;
    return { lookback: `${event[2]}s`, construction };
  }
  const descriptions: Record<string, string> = {
    binance_depth_churn_log_quote: "log1p of gross displayed bid/ask additions and removals in the latest completed 1s book-flow bucket",
    binance_depth_pressure: "net bid-supporting minus ask-supporting displayed-depth change, divided by gross depth change",
    binance_add_imbalance: "bid additions minus ask additions, divided by their sum",
    binance_remove_imbalance: "ask removals minus bid removals, divided by their sum",
    spot_mid_dispersion_bps: "cross-venue spot mid-price dispersion in basis points",
    best_executable_spread_bps: "best cross-venue executable ask minus bid spread in basis points",
    binance_perpetual_basis_bps: "Binance perpetual mid minus Binance spot mid in basis points",
  };
  if (descriptions[id]) return { lookback: "latest completed 1s", construction: descriptions[id] };
  if (id.endsWith("_l1_imbalance")) return {
    lookback: "latest completed 1s",
    construction: "best-bid quantity minus best-ask quantity, divided by their sum",
  };
  if (id.endsWith("_top5_imbalance")) return {
    lookback: "latest completed 1s",
    construction: "top-five bid quantity minus top-five ask quantity, divided by their sum",
  };
  if (id.endsWith("_spread_bps")) return {
    lookback: "latest completed 1s",
    construction: "venue best-ask minus best-bid spread in basis points",
  };
  return { lookback: "latest completed 1s", construction: "causal completed-second market-state coordinate" };
}

export function run(args = process.argv.slice(2)) {
  const value = (name: string) => {
    const index = args.indexOf(name);
    return index < 0 ? undefined : args[index + 1];
  };
  const input = resolve(value("--input-dir") ?? DEFAULT_INPUT);
  const output = resolve(value("--output-dir") ?? DEFAULT_OUTPUT);
  const { price, series, diagnostics } = loadFastAnalysisInputs(input);
  const contaminatedPrefixes = Object.entries(diagnostics.book.venues)
    .filter(([, row]: [string, any]) => row.status !== "healthy")
    .map(([name]) => name.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`));
  const candidates = buildCandidates(series).filter((candidate) => (
    !contaminatedPrefixes.some((prefix) => candidate.id.startsWith(prefix))
  ));
  const maps = series.map((item) => new Map(item.observations.map((row) => [row.second, row.values])));
  const start = Math.max(...series.map((item) => item.observations[0]!.second));
  const end = Math.min(...series.map((item) => item.observations.at(-1)!.second));
  const rows: Array<{
    second: number;
    features: number[];
    targets: number[];
    previous: number[];
    volatility: number[];
  }> = [];
  for (let second = start; second <= end; second += 1) {
    const values = maps.map((map) => map.get(second));
    if (values.some((row) => row === undefined)) continue;
    const index = second - price.startSecond - 1;
    const maximumEnd = index + HORIZONS.at(-1)!;
    const maximumPrevious = index - HORIZONS.at(-1)!;
    if (maximumPrevious < 0 || maximumEnd >= price.logPrice.length) continue;
    if (price.invalidPrefix[maximumEnd + 1]! !== price.invalidPrefix[maximumPrevious]!) continue;
    const targets: number[] = [];
    const previous: number[] = [];
    const volatility: number[] = [];
    for (const horizon of HORIZONS) {
      const horizonEnd = index + horizon;
      const horizonPrevious = index - horizon;
      const volatilityStart = Math.max(0, index - Math.max(60, horizon) + 1);
      targets.push(price.logPrice[horizonEnd]! - price.logPrice[index]!);
      previous.push(price.logPrice[index]! - price.logPrice[horizonPrevious]!);
      volatility.push(price.absoluteReturnPrefix[index + 1]! - price.absoluteReturnPrefix[volatilityStart]!);
    }
    rows.push({
      second,
      features: candidates.map((candidate) => Number(values[candidate.seriesIndex]![candidate.id])),
      targets,
      previous,
      volatility,
    });
  }
  if (rows.length < 1_000) throw new Error(`Only ${rows.length} common live rows are available.`);
  fs.mkdirSync(output, { recursive: true });
  writeFloat32(path.join(output, "features.f32"), rows.flatMap((row) => row.features));
  writeFloat32(path.join(output, "targets.f32"), rows.flatMap((row) => row.targets));
  writeFloat32(path.join(output, "previous.f32"), rows.flatMap((row) => row.previous));
  writeFloat32(path.join(output, "volatility.f32"), rows.flatMap((row) => row.volatility));
  writeFloat64(path.join(output, "times.f64"), rows.map((row) => row.second));
  const trainEnd = Math.floor(rows.length * 0.6);
  const primaryEnd = Math.floor(rows.length * 0.8);
  fs.writeFileSync(path.join(output, "splits.u8"), Buffer.from(Uint8Array.from(
    rows.map((_, index) => index < trainEnd ? 0 : index < primaryEnd ? 1 : 2),
  )));
  const manifest = {
    version: 1,
    generatedAt: new Date().toISOString(),
    objective: "Joint finite-universe feature-basis search for separate future return components",
    input: path.relative(repoRoot, input).replaceAll("\\", "/"),
    rows: rows.length,
    featureCount: candidates.length,
    targetCount: HORIZONS.length,
    horizonsSeconds: HORIZONS,
    coverage: {
      firstSecond: rows[0]!.second,
      lastSecond: rows.at(-1)!.second,
      observedHours: rows.length / 3_600,
      wallHours: (rows.at(-1)!.second - rows[0]!.second + 1) / 3_600,
    },
    split: {
      train: { rows: trainEnd, fraction: 0.6 },
      primary: { rows: primaryEnd - trainEnd, fraction: 0.2 },
      transfer: { rows: rows.length - primaryEnd, fraction: 0.2 },
    },
    fixedBaseline: [
      { id: "previous-return", construction: "previous completed return at the same horizon", bins: 3 },
      { id: "trailing-absolute-return", construction: "sum of absolute 1s returns over max(60s, target horizon) through the origin", bins: 3 },
    ],
    features: candidates.map(({ seriesIndex: _seriesIndex, ...candidate }) => candidate),
    files: {
      features: "features.f32",
      targets: "targets.f32",
      previous: "previous.f32",
      volatility: "volatility.f32",
      times: "times.f64",
      splits: "splits.u8",
    },
    diagnostics,
  };
  fs.writeFileSync(path.join(output, "manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`, "utf8");
  console.log(`Wrote ${path.relative(repoRoot, output)} (${rows.length.toLocaleString()} rows, ${candidates.length} features)`);
  return manifest;
}

function buildCandidates(series: FeatureSeries[]): Candidate[] {
  return series.flatMap((item, seriesIndex) => (
    [...new Set(item.observations.flatMap((row) => Object.keys(row.values)))].sort().map((id) => ({
      id,
      family: item.family,
      source: item.source,
      seriesIndex,
      ...describeLiveFeature(id),
    }))
  ));
}

function writeFloat32(file: string, values: number[]) {
  const array = Float32Array.from(values);
  fs.writeFileSync(file, Buffer.from(array.buffer, array.byteOffset, array.byteLength));
}

function writeFloat64(file: string, values: number[]) {
  const array = Float64Array.from(values);
  fs.writeFileSync(file, Buffer.from(array.buffer, array.byteOffset, array.byteLength));
}

function resolve(relativeOrAbsolute: string) {
  return path.isAbsolute(relativeOrAbsolute) ? relativeOrAbsolute : path.resolve(repoRoot, relativeOrAbsolute);
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) run();

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = <T>(relative: string): T => JSON.parse(fs.readFileSync(path.join(root, relative), "utf8")) as T;

interface Definition {
  id: string;
  name?: string;
  family?: string;
  parameters?: string;
  lookback?: string;
  delay?: string;
}

interface ComponentRow {
  horizonId: string;
  componentId: string;
  componentLabel: string;
  status: string;
  selectedBasis: { features?: string[]; primaryBits?: number; transferBits?: number };
}

function main() {
  const recentManifest = read<any>("data/runtime-cache/global-feature-basis-30d/manifest.json");
  const recentDataset = recentManifest.datasets[0];
  const recentDefinitions = recentDataset.features as Definition[];
  const definitions = new Map(recentDefinitions.map((item) => [item.id, item]));
  const tiered = read<any>("data/benchmarks/tiered-component-feature-bases.json");
  const live = read<any>("data/benchmarks/live-component-feature-bases.json");
  const publicAudit = read<any>("data/benchmarks/public-external-feature-information.json");
  const dense = read<any>("data/benchmarks/dense-lagged-indicator-audit.json");
  const fourier = read<any>("data/benchmarks/fourier-return-feature-information.json");
  const fourierManifest = read<any>("data/runtime-cache/fourier-return-features/manifest.json");
  const technical = read<any>("data/benchmarks/technical-indicator-predictiveness.json");
  const tardisAudit = read<any>("data/benchmarks/tardis-monthly-sample-information.json");
  const slowBook = read<any>("data/benchmarks/order-book-return-information.json");
  const externalDir = path.join(root, "data/market/mutable/external");
  const externalCoverage = externalArtifacts(externalDir);
  const bookRefs = countFiles("data/market/immutable/refs/derivatives-book-depth/usdm-futures/btcusdt", ".json");
  const tardis = directoryCounts("data/market/mutable/external/tardis-monthly-samples", ".json.gz");
  const publicCounts = new Map<string, number>();
  for (const group of publicAudit.groups as any[]) {
    const base = String(group.id).replace(/-(1m|5m|15m|30m|60m)$/, "")
      .replace(/-(production|long)$/, "");
    publicCounts.set(base, Math.max(publicCounts.get(base) ?? 0, Number(group.candidateCount)));
  }
  const fourierCount = Math.max(...fourierManifest.datasets.map((row: any) => row.featureCount));
  const horizons = ["1s", "1m", "15m", "1h"];
  const recommended = horizons.map((horizon) => recommendedForHorizon(
    horizon,
    tiered.targets as ComponentRow[],
    definitions,
    fourier,
  ));
  const componentCoverage = horizons.map((horizon) => {
    const rows = (tiered.targets as ComponentRow[]).filter((row) => row.horizonId === horizon);
    return {
      horizon,
      heads: rows.length,
      confirmed: rows.filter((row) => row.status === "confirmed-early").length,
      primaryOnly: rows.filter((row) => row.status === "primary-only").length,
      insufficient: rows.filter((row) => row.status === "insufficient-evaluation").length,
    };
  });
  const inventory = [
    item("recent-broad", recentDataset.featureCount, "all 19 component heads at 1s, 1m, 15m, 60m", "30d chronological", "mixed; 118 archive-backed + 29 local book", 0.94, "selected per head"),
    item("long endogenous basis", 34, "full return distribution at 1m, 15m, 60m; 10-coordinate 1s branch", "2021-07-25..2026-07-25", "candle-derived", 1.0, "promotion/stability reference"),
    item("dense EMA/RSI grid", dense.grid.candidateCount, "full return distribution separately at 1m, 15m, 60m", "2021-07-25..2026-07-25", "candle-derived", 1.0, "no stable conditional addition after volatility/range basis"),
    item("1s technical signals", technical.signals.length, "activity, sign and magnitude targets", technical.window?.start ? `${technical.window.start}..${technical.window.end}` : "long 1s history", "candle-derived", 1.0, "representatives retained only when selected"),
    item("Fourier/FrFT/wavelet", fourierCount, "full distribution, sign, magnitude at 1s, 1m, 15m, 60m", "2021-07-25..2026-07-25", "candle-derived", 1.0, "only three 1s controls survive transfer/tolerance"),
    item("cross-market public", publicCounts.get("cross-market") ?? 104, "full/sign/magnitude at 1m, 15m, 60m", "multi-year", "free kline archives", 0.96, "ETH volatility survives and is represented in broad bases"),
    item("global macro", publicCounts.get("global-macro") ?? 309, "full/sign/magnitude at 1m, 15m, 60m", "2021-03-24..2026-08-19 plus warmup", "official public, slow cadence", 0.9, "recent discoveries fail long-window conditional stability"),
    item("funding", publicCounts.get("binance-funding") ?? 18, "full/sign/magnitude at 1m, 15m, 60m", "2021-03-24..2026-08-19", "official public API", 0.96, "no stable addition"),
    item("DVOL", publicCounts.get("deribit-dvol") ?? 17, "full/sign/magnitude at 15m, 60m", "2021-03-24..2026-08-19", "official public API", 0.94, "no stable addition"),
    item("VIX", publicCounts.get("cboe-vix") ?? 9, "full/sign/magnitude at 1m, 15m, 60m", "2021-03-24..2026-08-18", "FRED public", 0.94, "no stable addition"),
    item("Coin Metrics network/flows", publicCounts.get("coinmetrics") ?? 37, "full/sign/magnitude at 15m, 60m", "2021-03-24..2026-08-18", "community public API; revisions", 0.78, "no stable addition"),
    item("community exchange/whale/miner flows", publicCounts.get("community-daily") ?? 145, "full/sign/magnitude at 15m, 60m", "2022-11-27..2026-08-19", "public derived archive; revisions", 0.65, "no stable addition"),
    item("mempool mining proxies", publicCounts.get("mempool-proxy") ?? 36, "full/sign/magnitude at 15m, 60m", "2023-08-19..2026-08-19", "public proxy history", 0.75, "no stable addition; exact mempool is live-only"),
    item("official futures percentage depth", slowBook.ranked.length, "full/zero/sign for next 1s", "368 official archive days", "free Binance archive, sparse dates", 0.82, "one narrow active-sign result; optional until joint common-window ablation"),
    item("Tardis cross-venue quotes/liquidations", tardisAudit.groups[0]?.candidateCount ?? 118, "full/sign/magnitude at 1s, 1m, 15m, 60m (plus intermediate horizons)", "10 independent first-of-month UTC days", "free monthly samples", 0.6, "no stable candidate at any horizon"),
    item("GDELT news", 8, "early live full-return screen; broad component join pending complete history", "1,162 valid historical buckets through 2026-08-01 plus live", "public rate-limited API", 0.7, "not selected; historical backfill checkpointed after HTTP 429"),
    item("Deribit option surface", 20, "early live full-return screen", "2 fixed surface snapshots plus continuing live summaries", "official current public API; no retrospective surface", 0.4, "insufficient evidence; 12 option-trade-flow coordinates are separately included in fast live clean"),
    item("fast live clean", live.dataset.features, "all 19 heads at 1s, 5s, 15s, 60s", `${isoSeconds(live.dataset.coverage.firstSecond)}..${isoSeconds(live.dataset.coverage.lastSecond)}`, "collector-only", 0.45, "experimental overlay; no fallback promotion over broad confirmed bases"),
    item("matched forward neural tensor", 231, "15m neural ablation", "same sources as forward-market audit", "representation coordinates, not 231 new sources", 0.9, "all-at-once tensor was worse; do not count as a separate feature universe"),
  ];
  const artifact = {
    version: 1,
    generatedAt: new Date().toISOString(),
    objective: "Reconcile all examined feature inventories, verify backfills, and select component-specific inputs with availability-aware near-tie preference",
    horizons: componentCoverage,
    availabilityRule: tiered.selection,
    inventory,
    backfill: {
      externalArtifacts: externalCoverage,
      binanceResearchArchiveAdded: ["2026-08-17", "2026-08-18"],
      binanceCurrentDay: "2026-08-19 daily archives were not yet published; no partial day was stored as a complete archive",
      futuresBookDepthOfficialDays: bookRefs,
      tardisMonthlySamples: tardis,
      gdelt: {
        state: "checkpointed-rate-limited",
        observedRows: 1_162,
        completeThroughExclusive: "2026-08-02T00:00:00.000Z",
        policy: "missing intervals remain missing; incomplete checkpoint is not treated as a finished dataset",
      },
    },
    recommended,
    limitations: [
      "The broad component search scores all coordinates marginally, then searches every subset up to size three among 12 family-aware finalists; it is not a proof over arbitrary transforms or larger subsets.",
      "Slow public sources cannot be identified at genuinely independent 1s cadence; their 1s role is regime conditioning and is redundant with contemporaneous volatility in current evidence.",
      "The 30-day and roughly 20-hour confirmations are regime-limited. Live-only inputs remain optional until multiple independent windows accrue.",
    ],
  };
  const output = path.join(root, "data/benchmarks/all-feature-availability-audit.json");
  const report = path.join(root, "docs/experiments/all-feature-availability-audit-2026-08-19.md");
  fs.mkdirSync(path.dirname(output), { recursive: true });
  fs.mkdirSync(path.dirname(report), { recursive: true });
  fs.writeFileSync(output, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.writeFileSync(report, render(artifact, definitions), "utf8");
  console.log(`Wrote ${path.relative(root, output)}`);
  console.log(`Wrote ${path.relative(root, report)}`);
}

function recommendedForHorizon(horizon: string, rows: ComponentRow[], definitions: Map<string, Definition>, fourier: any) {
  const confirmed = rows.filter((row) => row.horizonId === horizon && row.status === "confirmed-early");
  const componentByFeature = new Map<string, string[]>();
  for (const row of confirmed) for (const id of row.selectedBasis.features ?? []) {
    const list = componentByFeature.get(id) ?? [];
    list.push(row.componentId);
    componentByFeature.set(id, list);
  }
  if (horizon === "1s") {
    componentByFeature.set("return-lag-2s-control", ["full_distribution"]);
    componentByFeature.set("return-difference-1s-control", ["sign"]);
    componentByFeature.set("signed-variance-efficiency-16s", ["sign"]);
  }
  const inputs = [...componentByFeature].map(([id, components]) => {
    const definition = definitions.get(id) ?? spectralDefinition(id);
    const availability = availabilityFor(id);
    return { ...definition, components, ...availability };
  }).sort((left, right) => right.availabilityScore - left.availabilityScore || left.id.localeCompare(right.id));
  return {
    horizon,
    confirmedHeads: confirmed.length,
    requiredArchiveBacked: inputs.filter((row) => row.availabilityScore >= 0.9),
    optionalScarce: inputs.filter((row) => row.availabilityScore < 0.9),
    rejectedEvidence: rejectedForHorizon(horizon, fourier),
  };
}

function spectralDefinition(id: string): Definition {
  if (id === "return-lag-2s-control") return { id, name: "Return lag 2s", family: "return history", parameters: "lag=2s", lookback: "2s", delay: "through origin" };
  if (id === "return-difference-1s-control") return { id, name: "Return first difference", family: "return dynamics", parameters: "r(t)-r(t-1s)", lookback: "2s", delay: "through origin" };
  return { id, name: "Signed variance efficiency", family: "path efficiency", parameters: "signed efficiency of variance over 16s", lookback: "16s", delay: "through origin" };
}

function availabilityFor(id: string) {
  if (id.startsWith("spot-book-")) return { availabilityClass: "optional local snapshot", availabilityScore: 0.45 };
  if (id.startsWith(("spot-flow-"))) return { availabilityClass: "official aggregate-trade archive", availabilityScore: 0.95 };
  if (/^(futures-|open-interest-|top-|global-ratio-|taker-ratio-)/.test(id)) return { availabilityClass: "official futures archive", availabilityScore: 0.95 };
  if (/^(eth-|sol-|bnb-|doge-)/.test(id)) return { availabilityClass: "official alt kline archive", availabilityScore: 0.96 };
  return { availabilityClass: "core candle-derived", availabilityScore: 1.0 };
}

function rejectedForHorizon(horizon: string, fourier: any) {
  const spectral = fourier.horizons.find((row: any) => row.id === horizon);
  return spectral ? Object.entries(spectral.objectives).flatMap(([objective, value]: [string, any]) => (
    value.selected.features.length > 0 && value.selected.conditionalTransferBits <= 0
      ? [{ family: "spectral", objective, features: value.selected.features, reason: "nonpositive transfer addition" }]
      : []
  )) : [];
}

function item(id: string, coordinates: number, targets: string, coverage: string, acquisition: string, availabilityScore: number, conclusion: string) {
  return { id, coordinates, targets, coverage, acquisition, availabilityScore, conclusion };
}

function externalArtifacts(directory: string) {
  return fs.readdirSync(directory).filter((name) => name.endsWith(".json")).flatMap((name) => {
    const artifact = JSON.parse(fs.readFileSync(path.join(directory, name), "utf8"));
    if (!Array.isArray(artifact.rows)) return [];
    const times = artifact.rows.map((row: any) => row.time).filter(Number.isFinite);
    return [{
      file: name,
      rows: artifact.rows.length,
      first: times.length ? new Date(Math.min(...times)).toISOString() : null,
      last: times.length ? new Date(Math.max(...times)).toISOString() : null,
      checksumPresent: typeof artifact.sha256 === "string",
      failedSeries: artifact.request?.failedSeries?.length ?? 0,
    }];
  });
}

function countFiles(relative: string, suffix: string) {
  const directory = path.join(root, relative);
  return fs.existsSync(directory) ? fs.readdirSync(directory).filter((name) => name.endsWith(suffix)).length : 0;
}

function directoryCounts(relative: string, suffix: string) {
  const directory = path.join(root, relative);
  if (!fs.existsSync(directory)) return {};
  return Object.fromEntries(fs.readdirSync(directory, { withFileTypes: true }).filter((item) => item.isDirectory()).map((item) => [
    item.name,
    fs.readdirSync(path.join(directory, item.name)).filter((name) => name.endsWith(suffix)).length,
  ]));
}

function isoSeconds(value: number) {
  return new Date(value * 1_000).toISOString();
}

function render(artifact: any, definitions: Map<string, Definition>) {
  const lines = [
    "# All-feature availability audit and prediction basis",
    "",
    `Generated \`${artifact.generatedAt}\`.`,
    "",
    "## Result",
    "",
    "All inventories are now reconciled by semantic source rather than added as if every tensor coordinate were independent. The 231-input neural tensor is an encoding of already counted sources, while the dense EMA/RSI and spectral inventories are derived transforms of candles.",
    "",
    "The production default is the archive-backed input list below. Local-book features remain optional even when useful because they cannot be reconstructed for arbitrary history. Slow macro/flow inputs stay out because their recent gains did not survive the long conditional test.",
    "",
    "## Prediction heads checked",
    "",
    "| Horizon | Heads | Confirmed | Primary-only | Insufficient |",
    "|---|---:|---:|---:|---:|",
    ...artifact.horizons.map((row: any) => `| ${row.horizon} | ${row.heads} | ${row.confirmed} | ${row.primaryOnly} | ${row.insufficient} |`),
    "",
    "The 19 heads are inactivity; active sign; joint zero/sign/magnitude; four active-magnitude thresholds; sign in three large- and three small-magnitude regimes; and three magnitude thresholds conditional on each sign.",
    "",
    "## Exhaustive inventory ledger",
    "",
    "| Inventory | Coordinates | Targets checked | Evidence coverage | Acquisition | Availability | Conclusion |",
    "|---|---:|---|---|---|---:|---|",
    ...artifact.inventory.map((row: any) => `| ${escape(row.id)} | ${row.coordinates.toLocaleString()} | ${escape(row.targets)} | ${escape(row.coverage)} | ${escape(row.acquisition)} | ${row.availabilityScore.toFixed(2)} | ${escape(row.conclusion)} |`),
    "",
    "Counts overlap intentionally: a lagged EMA, Fourier coefficient, and neural tensor slot can all derive from the same candle. They are separate candidate transforms, not separate data sources.",
    "",
    "## Availability-aware selection effect",
    "",
    `Within the 0.001-bit near-tie band, availability changed ${artifact.availabilityRule.availabilityTieBreakChangedSelections} of ${artifact.availabilityRule.availabilityTieBreakEligibleSelections} eligible broad selections. The mean/max predictive sacrifice was ${artifact.availabilityRule.meanPrimaryBitsSacrificedWhenChanged.toFixed(6)}/${artifact.availabilityRule.maximumPrimaryBitsSacrificedWhenChanged.toFixed(6)} bits per target.`,
    "",
    "## Recommended model inputs",
    "",
  ];
  for (const horizon of artifact.recommended) {
    lines.push(`### ${horizon.horizon}`, "", `Confirmed component heads: ${horizon.confirmedHeads}.`, "", "| Input | Family | Parameters | Lookback | Causal delay | Availability | Used by heads |", "|---|---|---|---|---|---|---|");
    for (const row of horizon.requiredArchiveBacked) lines.push(inputRow(row));
    if (horizon.optionalScarce.length > 0) {
      lines.push("", "Optional scarce overlays (the model must also support a missing mask and an archive-only fallback):", "");
      for (const row of horizon.optionalScarce) lines.push(inputRow(row));
    }
    lines.push("");
  }
  lines.push(
    "## Backfill ledger",
    "",
    "| Artifact | Rows | First | Last | Checksum | Failed series |",
    "|---|---:|---|---|---|---:|",
    ...artifact.backfill.externalArtifacts.map((row: any) => `| ${row.file} | ${row.rows.toLocaleString()} | ${row.first ?? "n/a"} | ${row.last ?? "n/a"} | ${row.checksumPresent ? "yes" : "no"} | ${row.failedSeries} |`),
    "",
    `Binance archive days 2026-08-17 and 2026-08-18 were added for BTC 1s/1m, ETH/SOL/BNB/DOGE 1m, spot aggregate trades, futures klines, and futures metrics. ${artifact.backfill.binanceCurrentDay}.`,
    "",
    `The official Binance futures book-depth scope is complete at ${artifact.backfill.futuresBookDepthOfficialDays} available days. Tardis has ${Object.values(artifact.backfill.tardisMonthlySamples).join("/")} reduced monthly samples per source.`,
    "",
    `GDELT remains checkpointed after ${artifact.backfill.gdelt.observedRows.toLocaleString()} valid rows through ${artifact.backfill.gdelt.completeThroughExclusive}; the API returned HTTP 429. ${artifact.backfill.gdelt.policy}.`,
    "",
    "## What is deliberately excluded",
    "",
    "- Macro, funding, DVOL, VIX, Coin Metrics, community flows, and mempool proxies are not required inputs. Recent macro discoveries did not survive the 2021-2026 conditional stability test; all other slow families had no stable selected addition.",
    "- Dense EMA/RSI variants are not concatenated. All 3,471 variants per larger target were screened, but none produced a stable conditional addition after the volatility/range basis.",
    "- The 1m FrFT entropy addition is below the 0.001-bit tolerance; the 1h FrFT candidate fails transfer. The 1s negative-transfer FFT magnitude feature is excluded.",
    "- The 76 clean live inputs remain an experimental overlay. They cover only about 20 hours and did not rescue a broad component head that lacked confirmation at the matching 1s or 60s horizon.",
    "",
    "## Limits",
    "",
    ...artifact.limitations.map((item: string) => `- ${item}`),
    "",
    "Machine-readable audit: `data/benchmarks/all-feature-availability-audit.json`.",
    "Detailed per-head broad search: `docs/experiments/tiered-component-feature-bases-2026-08-19.md`.",
    "Detailed live-only search: `docs/experiments/live-component-feature-bases-2026-08-19.md`.",
    "",
  );
  return `${lines.join("\n")}\n`;
}

function inputRow(row: any) {
  return `| ${escape(row.id)} | ${escape(row.family ?? "n/a")} | ${escape(row.parameters ?? "n/a")} | ${escape(row.lookback ?? "n/a")} | ${escape(row.delay ?? "n/a")} | ${escape(row.availabilityClass)} | ${escape(row.components.join(", "))} |`;
}

function escape(value: string) {
  return String(value).replaceAll("|", "\\|");
}

main();

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const DEFAULT_OUTPUT = "data/benchmarks/external-feature-basis-audit.json";
const DEFAULT_REPORT = "docs/experiments/external-feature-basis-audit-2026-08-17.md";
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

type EvidenceStatus = "selected" | "provisional" | "rejected" | "collecting-live" | "proxy-only" | "credential-required";
type DataStatus = "historical-local" | "historical-and-live" | "live-local" | "free-backfill" | "live-only" | "paid-pit" | "paid-revised";

interface FeatureFamily {
  id: string;
  label: string;
  requested: string[];
  source: string;
  sourceUrl: string;
  dataStatus: DataStatus;
  history: string;
  features: string[];
  candidateLookbacks: string[];
  targetHorizons: string[];
  timing: string;
  leakageRisk: string;
  evidenceStatus: EvidenceStatus;
  result: string;
  selectedLookbacks: string[];
}

interface ExistingArtifacts {
  forward: any;
  spotBook: any;
  futuresBook: any;
  extended: any;
  control15m: any;
  joint15m: any;
  tardisSamples: any;
  recentJoint: any;
}

export function buildExternalFeatureAudit(root = repoRoot) {
  const existing = readExistingArtifacts(root);
  const forwardById = new Map(existing.forward.ranked.map((row: any) => [row.id, row]));
  const futuresBookById = new Map(existing.futuresBook.ranked.map((row: any) => [row.id, row]));
  const lastSide = forwardById.get("spot-flow-last-side-1s") as any;
  const lastSide2s = forwardById.get("spot-flow-last-side-lag-2s") as any;
  const lastSide3s = forwardById.get("spot-flow-last-side-lag-3s") as any;
  const futuresTradeCount = forwardById.get("futures-log-trade-count-1m") as any;
  const futuresBasis = forwardById.get("futures-basis-level") as any;
  const l1 = existing.spotBook.rankedFull.find((row: any) =>
    row.id === "quantity-imbalance-l1" && row.windowId === "all") as any;
  const l1Ofi = existing.spotBook.rankedFull.find((row: any) =>
    row.id === "normalized-ofi-l1" && row.windowId === "72h") as any;
  const futuresDepth = futuresBookById.get("log-depth-5") as any;
  const control = existing.control15m.bestValidation;
  const joint = existing.joint15m.bestValidation;
  const recentByHorizon = new Map(existing.recentJoint.horizons.map((row: any) => [row.id, row]));
  const families = featureFamilies({
    lastSide,
    lastSide2s,
    lastSide3s,
    futuresTradeCount,
    futuresBasis,
    l1,
    l1Ofi,
    futuresDepth,
    control,
    joint,
    spotBookSource: existing.spotBook.source,
    forwardSource: existing.forward.source,
    tardisSamples: existing.tardisSamples,
  });
  assertRequestedCoverage(families);
  const selected = families.filter((family) => family.evidenceStatus === "selected");
  const provisional = families.filter((family) => family.evidenceStatus === "provisional");
  const collectingLive = families.filter((family) => family.evidenceStatus === "collecting-live");
  const proxyOnly = families.filter((family) => family.evidenceStatus === "proxy-only");
  const credentialRequired = families.filter((family) => family.evidenceStatus === "credential-required");
  return {
    version: 1,
    generatedAt: new Date().toISOString(),
    objective: "Causal BTCUSDT next-return distribution prediction from 1s through 1h",
    policy: {
      targetHorizons: ["1s", "5s", "15s", "1m", "5m", "15m", "30m", "1h"],
      primaryMetric: "held-out incremental log likelihood in bits per target after the current basis",
      secondaryMetrics: ["active-return sign bits", "zero-gate bits", "distribution CRPS", "tail calibration"],
      stabilityRule: "positive in both chronological primary halves and a separated transfer block",
      causalRule: "use source receive time when available; otherwise delay a completed source interval before joining",
      selectionRule: "retain the shortest stable lookback and any longer lookback that adds stable conditional information",
    },
    localCoverage: {
      spotCandlesDays: 1848,
      forwardMatchedDays: existing.forward.source.primary.days + existing.forward.source.transfer.days,
      futuresPercentageDepthDays: existing.futuresBook.source.archiveDays,
      spotBookSnapshots: existing.spotBook.source.audit.validSnapshots,
      spotBookFirst: existing.spotBook.source.audit.firstTime,
      spotBookLast: existing.spotBook.source.audit.lastTime,
    },
    existingMatched15m: {
      candleOnlyValidationObjective: control.selectionObjective,
      allForwardFeaturesValidationObjective: joint.selectionObjective,
      relativeChange: joint.selectionObjective / control.selectionObjective - 1,
      conclusion: "Adding all 231 aggregated trade-flow, basis, and positioning inputs together was worse than the matched candle-only control.",
    },
    recentThirtyDayJoint: {
      start: "2026-07-18",
      end: "2026-08-16",
      candidateFeatures: 147,
      rows: Object.values((recentByHorizon.get("1s") as any).observations).reduce((sum: number, value: any) => sum + Number(value), 0),
      split: "16d train / 7d primary / 7d untouched transfer",
      horizons: Object.fromEntries([...recentByHorizon].map(([id, row]: any) => [id, {
        selected: row.robustOptimum.features,
        primaryBits: row.robustOptimum.primaryBits,
        transferBits: row.robustOptimum.transferBits,
        blockBits: row.robustOptimum.blockBits,
      }])),
      conclusion: "Futures 1m log trade count transfers jointly at the 1m head. No external coordinate is promoted at 15m or 1h; the recent 1h mixture fails transfer.",
    },
    selectedBasis: selected.map(basisRow),
    provisionalBasis: provisional.map(basisRow),
    collectingLive: collectingLive.map((family) => ({ id: family.id, source: family.source, dataStatus: family.dataStatus })),
    proxyOnly: proxyOnly.map((family) => ({ id: family.id, source: family.source, dataStatus: family.dataStatus })),
    credentialRequired: credentialRequired.map((family) => ({ id: family.id, source: family.source, dataStatus: family.dataStatus })),
    families,
    acquisitionBlockers: [
      "Binance no longer publishes its historical liquidationSnapshot archive. Ten free first-of-month Tardis samples now provide sparse event evidence; continuous history still requires a vendor or new live collection.",
      "Ten free first-of-month Tardis quote samples now cover synchronized Binance, Coinbase, Kraken, and Deribit top-of-book state. Continuous L2/L3 books and complete historical Deribit option surfaces remain vendor datasets.",
      "Binance's public BTCUSDT option EOHSummary archive ends on 2023-10-23, so it cannot provide a recent 30-day surface substitute; current BVOL index data remains available but does not contain strike/delta/OI structure.",
      "Macro consensus forecasts need a point-in-time calendar vendor. Official BLS/FRED data supplies actuals and revisions but not historical pre-release consensus.",
      "CryptoQuant documents that exchange-wallet clustering revisions make its historical exchange-flow endpoint non-point-in-time.",
      "The current environment has no Databento, Trading Economics, Blockworks, Glassnode, CryptoQuant, or FRED credentials.",
      "Live point-in-time collection is now running. Order events, option surfaces, GDELT, and true mempool state still need enough forward history before a continuous-history impact test is possible.",
    ],
  };
}

export function writeExternalFeatureAudit(
  root = repoRoot,
  output = DEFAULT_OUTPUT,
  report = DEFAULT_REPORT,
): ReturnType<typeof buildExternalFeatureAudit> {
  const artifact = buildExternalFeatureAudit(root);
  const outputPath = path.resolve(root, output);
  const reportPath = path.resolve(root, report);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, `${JSON.stringify(artifact, null, 2)}\n`, "utf8");
  fs.mkdirSync(path.dirname(reportPath), { recursive: true });
  fs.writeFileSync(reportPath, renderReport(artifact), "utf8");
  return artifact;
}

function featureFamilies(input: any): FeatureFamily[] {
  const windows = {
    micro: ["raw event", "1s", "2s", "5s", "15s", "30s", "1m"],
    flow: ["1s", "5s", "15s", "1m", "5m", "15m", "30m", "1h"],
    minute: ["1m", "2m", "5m", "15m", "30m", "1h", "4h"],
    news: ["5m decay", "15m decay", "30m decay", "1h decay", "4h decay"],
    slow: ["1h", "4h", "1d", "3d", "7d", "30d"],
  };
  const localForward = `${input.forwardSource.primary.start}..${input.forwardSource.primary.end} plus ${input.forwardSource.transfer.start}..${input.forwardSource.transfer.end}`;
  const noLeak = "Timestamp by exchange event and local receive time; fit transforms on training history only.";
  const sampleGroup = (horizonSeconds: number) => input.tardisSamples.groups.find((group: any) => group.horizonSeconds === horizonSeconds);
  const sampleFeature = (horizonSeconds: number, family: string) => sampleGroup(horizonSeconds)?.ranked.find((row: any) => row.family === family && row.stable);
  const crossLead1s = sampleFeature(1, "cross-exchange-lead-lag");
  const crossVol1m = sampleFeature(60, "cross-exchange-volatility");
  const liquidation1s = sampleFeature(1, "liquidations");
  const liquidation1m = sampleFeature(60, "liquidations");
  return [
    {
      id: "spot-order-events",
      label: "Spot order additions, cancellations, and executions",
      requested: ["order additions/cancellations"],
      source: "Binance spot diff-depth stream; existing top-10 snapshots provide only an aggregate OFI proxy",
      sourceUrl: "https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams",
      dataStatus: "live-local",
      history: `${input.spotBookSource.audit.firstTime}..${input.spotBookSource.audit.lastTime} snapshot history; synchronized diff-depth and aggregate-trade events have been retained since 2026-08-17`,
      features: ["bid/ask add notional", "bid/ask cancel notional", "execution notional", "normalized OFI", "queue depletion", "event intensity", "cancel/add ratio", "price-level distance moments"],
      candidateLookbacks: windows.micro,
      targetHorizons: ["1s", "5s", "15s", "1m"],
      timing: noLeak,
      leakageRisk: "A size decrease cannot be separated into cancellation versus execution without joining trades and synchronized depth updates.",
      evidenceStatus: "provisional",
      result: `${metric(input.l1Ofi?.primary?.fullBits)} primary and ${metric(input.l1Ofi?.transfer?.fullBits)} transfer bits/target for the snapshot OFI proxy; one of four chronological blocks is negative.`,
      selectedLookbacks: ["latest synchronized update", "1s", "5s"],
    },
    {
      id: "spot-book-imbalance",
      label: "Spot queue imbalance and microprice",
      requested: ["order additions/cancellations", "cross-exchange books"],
      source: "Locally recorded Binance spot top-10 snapshots",
      sourceUrl: "https://developers.binance.com/docs/binance-spot-api-docs/web-socket-streams",
      dataStatus: "live-local",
      history: `${input.spotBookSource.audit.validSnapshots.toLocaleString()} snapshots across a gappy three-week interval`,
      features: ["L1/L2/L5/L10 quantity imbalance", "notional imbalance", "microprice offset", "spread", "depth slope", "snapshot deltas"],
      candidateLookbacks: ["latest fresh snapshot", "1s", "2s", "5s", "15s", "1m"],
      targetHorizons: ["1s", "5s", "15s", "1m"],
      timing: "Snapshot must predate the target boundary; discard after 5s staleness.",
      leakageRisk: "Local capture gaps and a short calendar span make regime transfer uncertain.",
      evidenceStatus: "selected",
      result: `${metric(input.l1?.primary?.fullBits)} primary and ${metric(input.l1?.transfer?.fullBits)} transfer bits/target for L1 quantity imbalance, positive in 4/4 sub-blocks.`,
      selectedLookbacks: ["latest fresh snapshot", "training history: expanding/all pre-cutoff"],
    },
    {
      id: "cross-exchange-books",
      label: "Cross-exchange spot and perpetual books",
      requested: ["cross-exchange books", "cross-market lead/lag"],
      source: "Binance + Coinbase BTC-USD level2_batch + Kraken BTC/USD book v2 + Deribit BTC-PERPETUAL",
      sourceUrl: "https://docs.cdp.coinbase.com/exchange/websocket-feed/channels",
      dataStatus: "historical-and-live",
      history: "Ten free Tardis first-of-month UTC days across 2025-04..2025-11 and 2026-05..2026-06, plus continuous point-in-time collection since 2026-08-17",
      features: ["venue mid returns", "venue queue imbalance", "spread", "microprice", "cross-venue mid dispersion", "best executable cross-venue spread", "venue lead residuals", "staleness"],
      candidateLookbacks: windows.micro,
      targetHorizons: ["1s", "5s", "15s", "1m", "5m"],
      timing: "Use receive time for every venue and require a maximum age per venue.",
      leakageRisk: "Exchange clocks are not directly comparable; event-time joins create false lead/lag unless receive-time latency is retained.",
      evidenceStatus: "provisional",
      result: `Sparse monthly samples show stable cross-venue information through 1m: Coinbase 5s trailing return adds ${metric(crossLead1s?.primaryBits)} primary / ${metric(crossLead1s?.transferBits)} transfer bits at the 1s target; Coinbase 5m realized volatility adds ${metric(crossVol1m?.primaryBits)} / ${metric(crossVol1m?.transferBits)} at 1m. No cross-venue feature was stable at 5m or longer.`,
      selectedLookbacks: ["Coinbase trailing return 5s", "Coinbase realized volatility 5m"],
    },
    {
      id: "liquidations",
      label: "Liquidations and liquidation bursts",
      requested: ["liquidations", "liquidation bursts"],
      source: "Binance USD-M forceOrder live stream; free monthly Tardis samples; optional paid Tardis/Kaiko continuous history",
      sourceUrl: "https://developers.binance.com/docs/derivatives/usds-margined-futures/websocket-market-streams/Liquidation-Order-Streams",
      dataStatus: "historical-and-live",
      history: "11,064 BTC liquidation events across ten free Tardis first-of-month days, plus point-in-time Binance and Deribit collection since 2026-08-17",
      features: ["long/short liquidation notional", "count", "max event", "signed imbalance", "burst z-score", "inter-arrival gap", "cross-symbol breadth", "post-burst decay"],
      candidateLookbacks: windows.flow,
      targetHorizons: ["1s", "5s", "15s", "1m", "5m", "15m", "30m", "1h"],
      timing: noLeak,
      leakageRisk: "Binance's stream reports only the latest liquidation per symbol in each 1s window, so it is censored burst data.",
      evidenceStatus: "provisional",
      result: `Zero-aware sparse-sample bins show stable liquidation-count information through 1m: a 15m count adds ${metric(liquidation1s?.primaryBits)} primary / ${metric(liquidation1s?.transferBits)} transfer bits for 1s, and a 1h count adds ${metric(liquidation1m?.primaryBits)} / ${metric(liquidation1m?.transferBits)} for 1m. No liquidation feature was stable at 5m or longer.`,
      selectedLookbacks: ["15m count for 1s/5s targets", "1h count for 15s/1m targets"],
    },
    {
      id: "futures-premium",
      label: "Futures premium, basis, funding, and spot/perpetual disagreement",
      requested: ["futures premium", "cross-market lead/lag", "Deribit volatility/skew and broader derivatives state"],
      source: "Existing Binance spot and USD-M 1m histories plus 5m positioning metrics",
      sourceUrl: "https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api",
      dataStatus: "historical-local",
      history: localForward,
      features: ["basis level", "basis change", "basis EMA deviation", "relative spot/futures return", "funding level/change", "premium z-score", "spot/futures activity ratio"],
      candidateLookbacks: windows.minute,
      targetHorizons: ["1s", "1m", "5m", "15m", "30m", "1h"],
      timing: "A futures minute becomes usable only after close; 5m metrics retain a full 5m lag.",
      leakageRisk: "Repeated 5m values reduce effective sample size.",
      evidenceStatus: "rejected",
      result: `Basis level added ${metric(input.futuresBasis?.fullBits)} primary bits/target; no positioning feature was stable. Futures log trade count, not premium, was strongest at ${metric(input.futuresTradeCount?.fullBits)}.`,
      selectedLookbacks: [],
    },
    {
      id: "spot-trade-flow",
      label: "Spot aggressor flow and trade sequence",
      requested: ["cross-market lead/lag", "Minute/second cross-market returns and volatility"],
      source: "Existing Binance spot aggregate-trade archive",
      sourceUrl: "https://github.com/binance/binance-public-data",
      dataStatus: "historical-local",
      history: localForward,
      features: ["last aggressor side", "trade-count imbalance", "quote/base imbalance", "large-trade skew", "arrival centroid", "flow surprise"],
      candidateLookbacks: windows.micro,
      targetHorizons: ["1s", "5s", "15s", "1m"],
      timing: "Only the preceding completed one-second bin is used.",
      leakageRisk: "The effect is execution-latency sensitive and does not imply hour-ahead directional skill.",
      evidenceStatus: "selected",
      result: `Last aggressor side adds ${metric(input.lastSide.fullBits)} primary and ${metric(input.lastSide.transfer.fullBits)} transfer bits/target at 1s; ${metric(input.lastSide2s.fullBits)} at age 2s and ${metric(input.lastSide3s.fullBits)} at age 3s.`,
      selectedLookbacks: ["1s", "2s"],
    },
    {
      id: "cross-market-returns",
      label: "Cross-market returns, volatility, and lead/lag",
      requested: ["cross-market lead/lag", "Minute/second cross-market returns and volatility"],
      source: "Matched Binance BTC/ETH/SOL/BNB/DOGE minute histories; Databento CME/ICE remains optional for macro assets",
      sourceUrl: "https://databento.com/docs/knowledge-base/datasets",
      dataStatus: "historical-local",
      history: "462,240 aligned 1m candles per alt market across the 2025 primary and separated 2026 transfer blocks; macro assets still require a Databento account",
      features: ["lagged return by venue/asset", "realized volatility", "beta residual", "rolling correlation", "lead-lag residual", "session-open flag", "data age"],
      candidateLookbacks: ["1s", "2s", "5s", "15s", "30s", "1m", "2m", "5m", "15m", "30m", "1h", "4h"],
      targetHorizons: ["1m", "5m", "15m"],
      timing: "Use venue receive time where possible and closed bars otherwise; add market-open and staleness masks.",
      leakageRisk: "Back-adjusted continuous futures and forward-filled closed markets can manufacture lead/lag.",
      evidenceStatus: "selected",
      result: "ETH realized volatility adds 0.069524 primary / 0.036094 transfer bits at a 30m lookback for the 1m target; a 60m lookback adds 0.026971 / 0.017613 at 5m and 0.010539 / 0.005732 at 15m. The gain is magnitude information; no feature was stable at 30m or 1h.",
      selectedLookbacks: ["ETH realized volatility 30m for 1m", "ETH realized volatility 60m for 5m/15m"],
    },
    {
      id: "options-atm-iv",
      label: "ATM implied volatility and DVOL",
      requested: ["options IV/skew", "ATM implied volatility", "Deribit volatility/skew and broader derivatives state"],
      source: "Deribit option mark-price/ticker streams and historical BTC DVOL endpoint",
      sourceUrl: "https://docs.deribit.com/api-reference/market-data/public-get_volatility_index_data",
      dataStatus: "free-backfill",
      history: "47,338 hourly BTC DVOL rows are local; full Deribit option-surface collection began 2026-08-17",
      features: ["1d/7d/30d ATM IV", "BTC DVOL", "ATM IV slope", "ATM bid/ask IV spread", "IV z-score"],
      candidateLookbacks: windows.minute,
      targetHorizons: ["1m", "5m", "15m", "30m", "1h"],
      timing: "Surface snapshot must predate prediction time; interpolate in total variance across strike and expiry.",
      leakageRisk: "Trade-implied IV is selection-biased; use quote/mark surfaces rather than only option trades.",
      evidenceStatus: "collecting-live",
      result: "All 17 DVOL level/change/implied-realized candidates failed stability from 5m through 1h; historical ATM surface features remain unavailable and are accumulating live.",
      selectedLookbacks: [],
    },
    {
      id: "options-skew",
      label: "25-delta put/call skew",
      requested: ["25-delta put/call skew", "IV changes and skew changes", "options IV/skew"],
      source: "Deribit full BTC option surface",
      sourceUrl: "https://docs.deribit.com/api-reference/market-data/public-ticker",
      dataStatus: "live-local",
      history: "Full Deribit option-surface summaries and periodic raw snapshots have been recorded since 2026-08-17",
      features: ["25d put IV minus 25d call IV by expiry", "risk reversal", "skew slope", "skew curvature", "skew change"],
      candidateLookbacks: windows.minute,
      targetHorizons: ["1m", "5m", "15m", "30m", "1h"],
      timing: "Calculate delta from the same timestamped surface and freeze the interpolation method.",
      leakageRisk: "Nearest listed option can change discontinuously; interpolate by delta and maturity instead of selecting a contract after the fact.",
      evidenceStatus: "collecting-live",
      result: "Not yet measured: the first point-in-time surface exists, but a chronological test block does not.",
      selectedLookbacks: [],
    },
    {
      id: "options-term-structure",
      label: "1d/7d/30d volatility term structure",
      requested: ["1d versus 7d versus 30d volatility term structure", "Implied minus realized volatility"],
      source: "Deribit full BTC option surface plus local realized volatility",
      sourceUrl: "https://docs.deribit.com/subscriptions/market-data/markpriceoptionsindex_name",
      dataStatus: "live-local",
      history: "Full Deribit option-surface summaries and periodic raw snapshots have been recorded since 2026-08-17",
      features: ["7d-1d forward variance", "30d-7d forward variance", "term slope/curvature", "ATM IV minus realized volatility at matched horizon", "variance-risk premium"],
      candidateLookbacks: windows.minute,
      targetHorizons: ["5m", "15m", "30m", "1h"],
      timing: "Interpolate total variance, then compare with strictly trailing realized variance.",
      leakageRisk: "Comparing annualized IV with differently scaled realized volatility creates unit leakage and misleading levels.",
      evidenceStatus: "collecting-live",
      result: "Not yet measured: the first point-in-time surface exists, but a chronological test block does not.",
      selectedLookbacks: [],
    },
    {
      id: "options-positioning",
      label: "Option OI, major strikes, and expiration pressure",
      requested: ["Call/put open-interest imbalance", "Distance to major strikes and expiration"],
      source: "Deribit instrument ticker/open-interest surface",
      sourceUrl: "https://docs.deribit.com/api-reference/market-data/public-get_book_summary_by_instrument",
      dataStatus: "live-local",
      history: "Full Deribit option OI/strike/expiry snapshots have been recorded since 2026-08-17",
      features: ["call/put OI imbalance", "delta-weighted OI", "gamma-weighted OI", "distance to top OI strikes", "distance to max-pain proxy", "time to expiry", "expiring notional"],
      candidateLookbacks: ["current surface", "5m change", "15m change", "1h change", "4h change", "1d change"],
      targetHorizons: ["5m", "15m", "30m", "1h"],
      timing: "Use only open interest reported before the prediction time and preserve instrument lifecycle metadata.",
      leakageRisk: "Historical OI reconstructed from today's instrument set omits expired contracts and is invalid.",
      evidenceStatus: "collecting-live",
      result: "Not yet measured: the first point-in-time OI surface exists, but a chronological test block does not.",
      selectedLookbacks: [],
    },
    {
      id: "news-gdelt",
      label: "News events, GDELT intensity, and sentiment shocks",
      requested: ["news events", "GDELT article intensity and sentiment shocks"],
      source: "GDELT 2.0 GKG/GCAM, optionally a lower-latency paid news feed",
      sourceUrl: "https://blog.gdeltproject.org/gdelt-2-0-our-global-world-in-realtime/",
      dataStatus: "live-local",
      history: "Filtered GDELT unigram intensity and GKG theme/tone snapshots have been recorded since 2026-08-17; a retrospective DOC proxy has 926 checkpointed 15m observations for 2026-07-18..2026-07-29 but no transfer block",
      features: ["deduplicated story count", "abnormal intensity", "signed sentiment", "absolute sentiment shock", "sentiment change", "topic/entity flags", "source breadth", "novelty"],
      candidateLookbacks: windows.news,
      targetHorizons: ["15m", "30m", "1h"],
      timing: "Use first observed feed time, not an article's editable publication time; deduplicate syndication.",
      leakageRisk: "Backfilled publication timestamps can precede the time an article entered the feed.",
      evidenceStatus: "collecting-live",
      result: "Not yet measured: the DOC proxy was excluded because HTTP 429 throttling stopped it before the primary/transfer span completed. Point-in-time GDELT collection remains active.",
      selectedLookbacks: [],
    },
    {
      id: "macro-surprises",
      label: "Scheduled macro events and standardized surprises",
      requested: ["macro surprises", "news events"],
      source: "BLS/Fed/FRED/ALFRED actuals plus Trading Economics point-in-time consensus calendar",
      sourceUrl: "https://tradingeconomics.com/api/calendar.aspx",
      dataStatus: "paid-pit",
      history: "Official actual/vintage data is free; historical pre-release consensus requires credentials",
      features: ["event countdown", "event type/importance", "actual-consensus surprise", "revision surprise", "post-release age", "simultaneous NQ/DXY/2Y reaction"],
      candidateLookbacks: ["pre-event 4h", "pre-event 1h", "post 1m", "post 5m", "post 15m", "post 30m", "post 1h"],
      targetHorizons: ["5m", "15m", "30m", "1h"],
      timing: "Consensus must be the last value observed strictly before release; actual becomes visible only at release/update time.",
      leakageRisk: "Today's revised previous value and final consensus must never replace what was displayed before a historical release.",
      evidenceStatus: "credential-required",
      result: "Not measured.",
      selectedLookbacks: [],
    },
    {
      id: "onchain-flows",
      label: "Exchange, whale, miner, ETF, and treasury flows",
      requested: ["exchange, whale, miner, ETF and treasury flows"],
      source: "Coin Metrics network data; CryptoQuant/Glassnode labeled flows; issuer ETF holdings; SEC filings",
      sourceUrl: "https://docs.coinmetrics.io/api/v4/",
      dataStatus: "historical-local",
      history: "1,972 Coin Metrics daily rows plus 1,360 CC-BY community daily rows across 29 whale/miner/liquidation/derivatives metrics; values are retrospectively revised",
      features: ["exchange inflow/outflow/netflow", "large transfers", "miner-to-exchange flow", "miner reserves", "ETF net holdings change", "treasury filing impulse", "flow z-scores"],
      candidateLookbacks: windows.slow,
      targetHorizons: ["15m", "30m", "1h"],
      timing: "Use vendor observation/publication time and carry the most recently known value with an explicit age feature.",
      leakageRisk: "Wallet labels are revised retrospectively; vendor history is not point-in-time unless snapshots were archived live.",
      evidenceStatus: "proxy-only",
      result: "No exchange, whale, miner, funding, OI, liquidation, premium, or stablecoin-flow coordinate was stable at 15m, 30m, or 1h. The test is retrospective and ETF/treasury point-in-time histories remain unavailable.",
      selectedLookbacks: [],
    },
    {
      id: "mempool",
      label: "Live Bitcoin mempool pressure",
      requested: ["Live mempool measurements"],
      source: "mempool.space public API or a self-hosted Bitcoin Core node",
      sourceUrl: "https://mempool.space/docs/api/rest",
      dataStatus: "live-local",
      history: "2,195 three-year mined-block proxy rows plus point-in-time mempool snapshots recorded since 2026-08-17",
      features: ["mempool vbytes", "transaction count", "fee histogram", "projected-block depth", "recommended fee curve", "arrival rate", "large-transaction rate", "block-arrival shock"],
      candidateLookbacks: windows.minute,
      targetHorizons: ["1m", "5m", "15m", "30m", "1h"],
      timing: "Archive local receive time and node height; reset rate features after capture gaps.",
      leakageRisk: "Public mempool snapshots are observer-dependent; a later full-node reconstruction cannot reproduce what was unconfirmed earlier.",
      evidenceStatus: "collecting-live",
      result: "All 36 mined-block fee/size/reward proxy candidates failed stability at 15m–1h. True unconfirmed-mempool snapshots are accumulating live for a later test.",
      selectedLookbacks: [],
    },
    {
      id: "futures-percentage-depth",
      label: "Slow futures percentage-depth snapshots",
      requested: ["cross-exchange books"],
      source: "Existing Binance USD-M ±1–5% percentage-depth archive",
      sourceUrl: "https://data.binance.vision/",
      dataStatus: "historical-local",
      history: `${input.futuresDepth ? "368 days" : "local archive"}`,
      features: ["depth imbalance", "total depth", "near/far concentration", "snapshot changes"],
      candidateLookbacks: ["latest 30s snapshot", "1m", "5m", "15m"],
      targetHorizons: ["1s", "1m", "5m", "15m"],
      timing: "Use only a snapshot from an earlier second; discard after 120s.",
      leakageRisk: "Cumulative percentage bands omit top-of-book state and are too slow for queue dynamics.",
      evidenceStatus: "rejected",
      result: `Best full-distribution result was ${metric(input.futuresDepth?.fullBits)} primary bits/target and was unstable.`,
      selectedLookbacks: [],
    },
  ];
}

function readExistingArtifacts(root: string): ExistingArtifacts {
  return {
    forward: readJson(root, "data/benchmarks/forward-market-return-information.json"),
    spotBook: readJson(root, "data/benchmarks/spot-order-book-return-information.json"),
    futuresBook: readJson(root, "data/benchmarks/order-book-return-information.json"),
    extended: readJson(root, "data/benchmarks/extended-market-information-basis.json"),
    control15m: readJson(root, "data/training/runs/causal-forward-market-oracle-15m-control-v1/result.json"),
    joint15m: readJson(root, "data/training/runs/causal-forward-market-oracle-15m-joint-normalized-v2/result.json"),
    tardisSamples: readJson(root, "data/benchmarks/tardis-monthly-sample-information.json"),
    recentJoint: readJson(root, "data/benchmarks/global-return-feature-basis-30d.json"),
  };
}

function readJson(root: string, relative: string): any {
  return JSON.parse(fs.readFileSync(path.resolve(root, relative), "utf8"));
}

function basisRow(family: FeatureFamily) {
  return {
    id: family.id,
    label: family.label,
    features: family.features,
    selectedLookbacks: family.selectedLookbacks,
    targetHorizons: family.targetHorizons,
    evidenceStatus: family.evidenceStatus,
    result: family.result,
  };
}

function assertRequestedCoverage(families: FeatureFamily[]): void {
  const required = [
    "order additions/cancellations", "cross-exchange books", "liquidations", "liquidation bursts",
    "futures premium", "cross-market lead/lag", "news events", "macro surprises", "options IV/skew",
    "ATM implied volatility", "25-delta put/call skew", "1d versus 7d versus 30d volatility term structure",
    "Implied minus realized volatility", "Call/put open-interest imbalance", "Distance to major strikes and expiration",
    "IV changes and skew changes", "exchange, whale, miner, ETF and treasury flows",
    "Minute/second cross-market returns and volatility", "GDELT article intensity and sentiment shocks",
    "Deribit volatility/skew and broader derivatives state", "Live mempool measurements",
  ];
  const covered = new Set(families.flatMap((family) => family.requested));
  const missing = required.filter((item) => !covered.has(item));
  if (missing.length > 0) throw new Error(`Requested feature families missing from audit: ${missing.join(", ")}`);
  const ids = new Set<string>();
  for (const family of families) {
    if (ids.has(family.id)) throw new Error(`Duplicate feature family id: ${family.id}`);
    ids.add(family.id);
    if (family.candidateLookbacks.length === 0) throw new Error(`${family.id} has no lookback screen.`);
    if (family.evidenceStatus === "selected" && family.selectedLookbacks.length === 0) {
      throw new Error(`${family.id} is selected without a selected lookback.`);
    }
  }
}

function renderReport(artifact: ReturnType<typeof buildExternalFeatureAudit>): string {
  const lines = [
    "# External feature basis audit",
    "",
    `Generated ${artifact.generatedAt}. This is the canonical inventory for external BTC return-distribution features from 1s through 1h.`,
    "",
    "## Outcome",
    "",
    `The requested list is fully enumerated. Public backfills, matched crypto minute histories, and ten synchronized monthly quote/liquidation samples have been measured. Selected: ${artifact.selectedBasis.length}; provisional: ${artifact.provisionalBasis.length}; rejected: ${artifact.families.filter((family) => family.evidenceStatus === "rejected").length}; collecting live: ${artifact.collectingLive.length}; proxy-only: ${artifact.proxyOnly.length}; credential-required: ${artifact.credentialRequired.length}. There is no generic awaiting-data bucket.`,
    "",
    `The matched 15m neural test remains negative: adding all 231 existing forward inputs changed validation objective from ${metric(artifact.existingMatched15m.candleOnlyValidationObjective)} to ${metric(artifact.existingMatched15m.allForwardFeaturesValidationObjective)} (${percent(artifact.existingMatched15m.relativeChange)} worse). This is why the basis must be selected conditionally instead of concatenating every candidate.`,
    "",
    "## Thirty-day common-coverage update",
    "",
    `The freely backfillable recent layer is no longer awaiting data. For ${artifact.recentThirtyDayJoint.start} through ${artifact.recentThirtyDayJoint.end}, the local store contains checksum-verified Binance spot 1s/1m candles, spot aggregate-trade flow, USD-M perpetual 1m candles and 5m metrics, and BTC/ETH/SOL/BNB/DOGE 1m candles. These overlap the recent local Binance spot-book capture where that stream is present.`,
    "",
    `The resulting ${artifact.recentThirtyDayJoint.candidateFeatures}-coordinate joint dataset has ${artifact.recentThirtyDayJoint.rows.toLocaleString()} causal minute-boundary origins and uses ${artifact.recentThirtyDayJoint.split}. ${artifact.recentThirtyDayJoint.conclusion} Full results are in \`docs/experiments/global-return-feature-basis-30d-2026-08-17.md\`.`,
    "",
    "This backfill does not manufacture unavailable history. Continuous cross-exchange L2, full option surfaces/OI, exact liquidation history, historical mempool snapshots, and point-in-time macro consensus remain live/vendor-only categories.",
    "",
    "## Evidence-backed basis",
    "",
    "| family | selected lookback | useful target horizons | result |",
    "|---|---|---|---|",
  ];
  for (const row of artifact.selectedBasis) {
    lines.push(`| ${row.label} | ${row.selectedLookbacks.join(", ")} | ${row.targetHorizons.join(", ")} | ${row.result} |`);
  }
  lines.push("", "## Provisional basis", "", "| family | provisional lookback | reason |", "|---|---|---|");
  for (const row of artifact.provisionalBasis) {
    lines.push(`| ${row.label} | ${row.selectedLookbacks.join(", ")} | ${row.result} |`);
  }
  lines.push(
    "",
    "## Unresolved evidence, classified",
    "",
    "| status | families | meaning |",
    "|---|---:|---|",
    `| collecting-live | ${artifact.collectingLive.length} | Causal collection is active, but there is not yet a separated continuous-history test block. |`,
    `| proxy-only | ${artifact.proxyOnly.length} | Public historical proxies were tested; the exact requested point-in-time series remains unavailable. |`,
    `| credential-required | ${artifact.credentialRequired.length} | The exact historical series requires a vendor key or licensed data. |`,
  );
  lines.push(
    "",
    "## Complete feature inventory",
    "",
    "| family | data | evidence | candidate lookbacks | result |",
    "|---|---|---|---|---|",
  );
  for (const family of artifact.families) {
    lines.push(`| ${family.label} | ${family.dataStatus} | ${family.evidenceStatus} | ${family.candidateLookbacks.join(", ")} | ${family.result} |`);
  }
  lines.push(
    "",
    "## Per-family causal contract",
    "",
  );
  for (const family of artifact.families) {
    lines.push(
      `### ${family.label}`,
      "",
      `- Source: [${family.source}](${family.sourceUrl})`,
      `- History: ${family.history}`,
      `- Candidate features: ${family.features.join(", ")}.`,
      `- Target horizons: ${family.targetHorizons.join(", ")}.`,
      `- Timing: ${family.timing}`,
      `- Leakage risk: ${family.leakageRisk}`,
      `- Decision: **${family.evidenceStatus}**. ${family.result}`,
      "",
    );
  }
  lines.push(
    "## Reproduction and live acquisition",
    "",
    "- `npm run fetch:research-forward-market -- --ranges 2026-07-18..2026-08-16` rebuilds the research-only spot-flow and Binance derivatives layer without touching the sealed oracle corpus. Official SHA-256 checksums are verified and source ZIPs are discarded after reduction.",
    "- `npm run fetch:gdelt-history` builds the reduced 15-minute retrospective crypto-news count/tone proxy in sub-72-hour API chunks. Missing intervals remain null, HTTP 429 responses back off, and completed chunks are checkpointed before the immutable artifact is written.",
    "- `npm run analysis:global-feature-basis:export-30d` reconstructs the 147-coordinate common-coverage dataset. `npm run analysis:global-feature-basis -- --input-dir data/runtime-cache/global-feature-basis-30d --output data/benchmarks/global-return-feature-basis-30d.json --report docs/experiments/global-return-feature-basis-30d-2026-08-17.md` reruns the joint search.",
    "- `npm run fetch:external-public` refreshes the free DVOL, Coin Metrics, community daily, mempool-proxy, and current Deribit surface data.",
    "- `npm run analysis:external-public` rebuilds the held-out information screen in `docs/experiments/public-external-feature-information-2026-08-17.md` and its machine-readable JSON artifact.",
    "- `npm run fetch:tardis-samples` streams and reduces the ten free first-of-month cross-venue quote and liquidation samples without retaining raw tick CSV; `npm run analysis:tardis-samples` rebuilds their 1s–1h screen.",
    "- `npm run collect:external-live` records synchronized exchange, liquidation, premium, Deribit surface, GDELT, and mempool observations under `data/market/mutable/external-live`.",
    "- `npm run analysis:external-live-early` rebuilds the horizon-aware live readiness, storage, feature-variation, and early held-out likelihood screen. The active watcher records immutable 1h and 24h checkpoints as collection reaches them.",
    "- The concrete model tensor, horizon routing, causal lookbacks, and selected/provisional/excluded split are summarized in `docs/experiments/recommended-model-input-basis-2026-08-17.md`.",
    "- Live JSONL is stored as concatenated complete gzip members. Each source flushes at least once per second, so files remain readable while the collector runs and a crash loses at most the current in-memory chunk.",
    "- Set `TRADING_ECONOMICS_API_KEY` for point-in-time macro consensus and `BLOCKWORKS_API_KEY` for ETF flows. Every public source operates without credentials.",
    "",
    "## Acquisition blockers recorded",
    "",
    ...artifact.acquisitionBlockers.map((item) => `- ${item}`),
    "",
    "## Live evidence timing and retention",
    "",
    "- One hour is a feed-health and very-large-effect smoke test for 1s–15s targets, not a feature rejection window.",
    "- One day is the first early screen for 1s–1m targets. It is still one market regime, so a negative score alone does not justify deletion.",
    "- GDELT has about 96 observations/day; use at least seven separated days for its first screen. A 1h target has only 24 non-overlapping outcomes/day and needs roughly 30–90 days.",
    "- The unresolved options, premium, GDELT, and mempool feeds are low-volume. The raw exchange books dominate storage; compact those to causal 1s derived state after reconstruction tests instead of cutting the slow candidates early.",
    "",
    "## Selection protocol once data exists",
    "",
    "1. Freeze transformations and quantile cuts on training history only.",
    "2. Evaluate every declared lookback at 1s, 5s, 15s, 1m, 5m, 15m, 30m, and 1h where the source cadence permits.",
    "3. Score incremental held-out log likelihood after the current basis, plus sign, zero-gate, CRPS, and tail calibration.",
    "4. Require positive improvement in two chronological primary halves and one separated transfer block.",
    "5. Within a correlated family, keep the shortest stable lookback; keep another only when it adds conditional information.",
    "6. Repeat greedy forward selection after every accepted family. Individual marginal gains are not additive.",
    "",
    "Machine-readable details are stored in `data/benchmarks/external-feature-basis-audit.json`.",
    "",
  );
  return lines.join("\n");
}

function metric(value: number | undefined): string {
  return value === undefined || !Number.isFinite(value) ? "n/a" : value.toPrecision(8);
}

function percent(value: number): string {
  return `${(100 * value).toFixed(3)}%`;
}

if (process.argv[1] && import.meta.url === pathToFileURL(path.resolve(process.argv[1])).href) {
  const artifact = writeExternalFeatureAudit();
  console.log(`Wrote ${DEFAULT_OUTPUT}`);
  console.log(`Wrote ${DEFAULT_REPORT}`);
  console.log(`${artifact.selectedBasis.length} selected, ${artifact.provisionalBasis.length} provisional, ${artifact.collectingLive.length} collecting live, ${artifact.proxyOnly.length} proxy-only, ${artifact.credentialRequired.length} credential-required.`);
}

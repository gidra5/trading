import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { discoverEconomicAssets, ECONOMIC_COVERAGE_TAGS, INDUSTRY_SECTORS, marketKey, SECTOR_SOURCES,
  type EconomicListing, type EconomicRoute } from "./lib/research-market-sectors.js";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const ARCHIVE = "docs/portfolio/runs/2026-07-22-all-usdt-1m-360c-q4-amp5pct-pearson-k257.json";
const ADDITIONS = ["ETH", "SOL", "XRP", "HYPE"] as const;
const CATEGORIES = ["stablecoins", "wrapped-tokens", "liquid-staking-tokens", "liquid-restaking-tokens"] as const;
// Canonical CoinGecko identities resolve ticker collisions for these named assets.
// Both the expected ticker and the Binance base asset must still match exactly.
const VERIFIED_IDS: Readonly<Record<string, string>> = {
  bitcoin: "BTC", ethereum: "ETH", binancecoin: "BNB", ripple: "XRP",
  solana: "SOL", tron: "TRX", hyperliquid: "HYPE", zcash: "ZEC",
  dogecoin: "DOGE", monero: "XMR", chainlink: "LINK", cardano: "ADA",
  stellar: "XLM", "bitcoin-cash": "BCH", near: "NEAR", uniswap: "UNI",
  litecoin: "LTC", "avalanche-2": "AVAX", sui: "SUI", "hedera-hashgraph": "HBAR",
  "the-open-network": "TON", "shiba-inu": "SHIB", polkadot: "DOT",
  aave: "AAVE", "internet-computer": "ICP", "ethereum-classic": "ETC",
  cosmos: "ATOM", "render-token": "RENDER", "fetch-ai": "FET",
  "ondo-finance": "ONDO", "worldcoin-wld": "WLD", "mantle": "MNT",
};

type Venue = "spot" | "usdm-futures";
type Coin = Readonly<{ id: string; symbol: string; name: string; market_cap: number | null; market_cap_rank: number | null; last_updated: string | null }>;
type CatalogCoin = Readonly<{ id: string; symbol: string; name?: string }>;
type Listing = EconomicListing;
type Market = EconomicRoute;
type FullExchange = Readonly<{ symbols?: readonly unknown[]; optionSymbols?: readonly unknown[]; optionContracts?: readonly Readonly<{ baseAsset: string; underlying: string }>[] }>;
type Snapshot<T> = Readonly<{ source: Readonly<{ url: string; retrievedAt: string; responseSha256: string }>; data: T }>;
type ArchivedAsset = Readonly<{ baseAsset: string; symbol?: string; priceVenue?: Venue; productSymbols?: Readonly<Partial<Record<Venue, readonly string[]>>> }>;
type ArchivedReport = Readonly<{ basis: Readonly<{ entries: readonly Readonly<{ rank: number; baseAsset: string }>[] }>; assets: readonly ArchivedAsset[] }>;

function option(args: readonly string[], name: string, fallback: string): string {
  const index = args.indexOf(name);
  if (index < 0) return fallback;
  const value = args[index + 1];
  if (!value || value.startsWith("--")) throw new Error(`${name} requires a value`);
  return value;
}
function resolve(file: string): string { return path.resolve(ROOT, file); }
function hash(value: string): string { return createHash("sha256").update(value).digest("hex"); }
function message(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function isCoin(value: unknown): value is Coin {
  const row = value as Partial<Coin> | null;
  return !!row && typeof row.id === "string" && typeof row.symbol === "string" && typeof row.name === "string"
    && (row.market_cap === null || (typeof row.market_cap === "number" && Number.isFinite(row.market_cap)));
}
function listings(value: unknown): readonly Listing[] {
  const rows = (value as { symbols?: unknown[] })?.symbols;
  if (!Array.isArray(rows) || !rows.length) throw new Error("Exchange snapshot has no symbols");
  return rows.map((value) => {
    const row = value as Listing;
    if (![row.symbol, row.baseAsset, row.quoteAsset, row.status].every((item) => typeof item === "string")) {
      throw new Error("Invalid exchange listing");
    }
    return { symbol: row.symbol, baseAsset: row.baseAsset, quoteAsset: row.quoteAsset, status: row.status,
      ...(row.contractType ? { contractType: row.contractType } : {}),
      ...(row.underlyingType ? { underlyingType: row.underlyingType } : {}),
      ...(row.underlyingSubType ? { underlyingSubType: row.underlyingSubType } : {}),
      ...(row.isSpotTradingAllowed === undefined ? {} : { isSpotTradingAllowed: row.isSpotTradingAllowed }) };
  });
}
function fullExchange(raw: unknown): FullExchange {
  if (!raw || typeof raw !== "object" || !(Array.isArray((raw as FullExchange).symbols) || Array.isArray((raw as FullExchange).optionSymbols))) {
    throw new Error("Invalid complete Binance product metadata");
  }
  return raw as FullExchange;
}

export async function run(args = process.argv.slice(2)): Promise<void> {
  if (args.includes("--help")) {
    console.log(`Restore historical diversity, crypto cap leaders and every listed noncrypto economic instrument.

Usage: node --import tsx scripts/restore-research-market-universe.ts [options]
  --offline                    Read saved normalized API snapshots; make no requests
  --snapshots-dir DIR           Default docs/portfolio/restored-market-universe-sources
  --output FILE                Default docs/portfolio/restored-market-universe.json
  --historical FILE            Default archived July 22 257-asset report
  --market-pages N             CoinGecko top N * 250 coins (default 4)
  --coverage FRACTION           Cap core share of eligible fetched snapshot (default .95)
  --min-cap-core-size N          Retain at least N cap leaders for redundancy (default 10)

Live inputs: CoinGecko USD crypto market caps, complete named coin catalog,
exclusion categories and complete Binance spot/USD-M/COIN-M/options metadata.
Targets retain historical assets and crypto cap leaders, plus all metadata-backed
TradFi perpetuals, verified stock tokens, gold proxies and fiat/stablecoin pairs.
Actual spot/USD-M routes are listed; COIN-M/options are discovery-only inventory.
Use assetId for identity (CAT stock and CAT crypto differ), venue+symbol for routes.
Global cap is applied only to crypto; noncrypto capitalization remains unavailable.
This does not start or reconfigure a collector.
Replay with --offline and the same options to reproduce membership from snapshots.`);
    return;
  }
  const offline = args.includes("--offline");
  const directory = resolve(option(args, "--snapshots-dir", "docs/portfolio/restored-market-universe-sources"));
  const output = resolve(option(args, "--output", "docs/portfolio/restored-market-universe.json"));
  const archiveFile = resolve(option(args, "--historical", ARCHIVE));
  const pages = Number(option(args, "--market-pages", "4"));
  const coverage = Number(option(args, "--coverage", ".95"));
  const minimumCapCoreSize = Number(option(args, "--min-cap-core-size", "10"));
  if (!Number.isInteger(pages) || pages < 1 || pages > 20) throw new Error("--market-pages must be 1..20");
  if (!(coverage > 0 && coverage <= 1)) throw new Error("--coverage must be >0 and <=1");
  if (!Number.isInteger(minimumCapCoreSize) || minimumCapCoreSize < 1) throw new Error("--min-cap-core-size must be a positive integer");
  const archiveText = await fs.readFile(archiveFile, "utf8");
  const archive = JSON.parse(archiveText) as ArchivedReport;
  const archived = archive.basis.entries.map((row) => row.baseAsset);
  if (archived.length !== 257 || new Set(archived).size !== 257) throw new Error("Expected the archived 257 unique basis assets");
  const historical = [...archived, ...ADDITIONS.filter((asset) => !archived.includes(asset))];
  if (historical.length !== 261) throw new Error("Expected exactly 261 restored historical assets");
  const provenance: Record<string, Snapshot<unknown>["source"]> = {};
  const failures: Array<Readonly<{ input: string; error: string }>> = [];
  if (!offline) await fs.mkdir(directory, { recursive: true });

  async function input<T>(name: string, url: string, normalize: (raw: unknown) => T): Promise<T | null> {
    try {
      let snapshot: Snapshot<T>;
      if (offline) {
        snapshot = JSON.parse(await fs.readFile(path.join(directory, `${name}.json`), "utf8")) as Snapshot<T>;
        if (snapshot.source?.url !== url || !snapshot.source.retrievedAt || !snapshot.source.responseSha256) {
          throw new Error("Snapshot provenance does not match the requested input");
        }
        // Revalidate the normalized data using the same boundary as live input.
        normalize(snapshot.data);
      } else {
        const response = await fetch(url, { signal: AbortSignal.timeout(30_000), headers: { accept: "application/json" } });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        const body = await response.text();
        snapshot = { source: { url, retrievedAt: new Date().toISOString(), responseSha256: hash(body) }, data: normalize(JSON.parse(body)) };
        await fs.writeFile(path.join(directory, `${name}.json`), JSON.stringify(snapshot));
      }
      provenance[name] = snapshot.source;
      return snapshot.data;
    } catch (error) {
      failures.push({ input: name, error: message(error) });
      return null;
    }
  }
  const coinRows = (raw: unknown): readonly Coin[] => {
    if (!Array.isArray(raw) || !raw.every(isCoin)) throw new Error("Invalid CoinGecko markets response");
    return raw.map((row) => ({ id: row.id, symbol: row.symbol, name: row.name, market_cap: row.market_cap,
      market_cap_rank: row.market_cap_rank ?? null, last_updated: row.last_updated ?? null }));
  };
  const [spotFull, usdmFull, coinmFull, optionsFull, catalog, blackberryTickers] = await Promise.all([
    input("spot-full", "https://api.binance.com/api/v3/exchangeInfo", fullExchange),
    input("usdm-full", "https://fapi.binance.com/fapi/v1/exchangeInfo", fullExchange),
    input("coinm-full", "https://dapi.binance.com/dapi/v1/exchangeInfo", fullExchange),
    input("options-full", "https://eapi.binance.com/eapi/v1/exchangeInfo", fullExchange),
    input("coin-catalog-full", "https://api.coingecko.com/api/v3/coins/list", (raw): readonly CatalogCoin[] => {
      if (!Array.isArray(raw) || !raw.length || !raw.every((row) => typeof row.id === "string" && typeof row.symbol === "string")) throw new Error("Invalid coin catalog");
      return raw.map((row) => ({ id: row.id, symbol: row.symbol, name: row.name }));
    }),
    input("blackberry-bstocks-tickers", "https://api.coingecko.com/api/v3/coins/blackberry-bstocks-tokenized-stock/tickers?exchange_ids=binance", (raw) => {
      const value = raw as Readonly<{ tickers?: readonly Readonly<{ base: string; target: string; coin_id: string; market: Readonly<{ identifier: string }> }>[] }>;
      if (!Array.isArray(value?.tickers)) throw new Error("Invalid verified stock-token ticker response");
      return value;
    }),
  ]);
  const spot = spotFull ? listings(spotFull) : null;
  const usdm = usdmFull ? listings(usdmFull) : null;
  const marketPages: Array<readonly Coin[]> = [];
  for (let page = 1; page <= pages; page++) {
    const rows = await input(`markets-${page}`, `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&order=market_cap_desc&per_page=250&page=${page}&sparkline=false`, coinRows);
    if (rows) marketPages.push(rows);
  }
  const coins = [...new Map(marketPages.flat().map((row) => [row.id, row])).values()];
  const floorCap = Math.min(...coins.filter((coin) => coin.market_cap! > 0).map((coin) => coin.market_cap!));
  const excludedIds = new Map<string, string[]>();
  for (const category of CATEGORIES) {
    for (let page = 1; page <= 20; page++) {
      const rows = await input(`exclude-${category}-${page}`, `https://api.coingecko.com/api/v3/coins/markets?vs_currency=usd&category=${category}&order=market_cap_desc&per_page=250&page=${page}&sparkline=false`, coinRows);
      if (!rows) break;
      for (const coin of rows) excludedIds.set(coin.id, [...(excludedIds.get(coin.id) ?? []), category]);
      if (rows.length < 250 || (rows.at(-1)?.market_cap ?? 0) < floorCap) break;
      if (page === 20) failures.push({ input: category, error: "Exclusion category pagination limit reached" });
    }
  }
  const active = (rows: readonly Listing[] | null, venue: Venue): readonly Market[] => (rows ?? [])
    .filter((row) => row.status === "TRADING" && row.quoteAsset === "USDT"
      && (venue === "spot" ? row.isSpotTradingAllowed !== false : ["PERPETUAL", "TRADIFI_PERPETUAL"].includes(row.contractType ?? "")))
    .map((row) => ({ venue, symbol: row.symbol, baseAsset: row.baseAsset, quoteAsset: row.quoteAsset,
      instrumentKind: venue === "spot" ? "crypto-spot" : "crypto-perpetual", priceOrientation: "direct", identityEvidence: ["Binance exchangeInfo physical base/quote; archive provides economic alias where explicitly recorded"],
      ...(row.contractType ? { contractType: row.contractType } : {}),
      ...(row.underlyingType ? { underlyingType: row.underlyingType } : {}),
      ...(row.underlyingSubType ? { underlyingSubType: row.underlyingSubType } : {}) }));
  const markets = [...active(spot, "spot"), ...active(usdm, "usdm-futures")];
  const verifiedStockTickers = new Set(blackberryTickers?.tickers?.filter((row) => row.coin_id === "blackberry-bstocks-tokenized-stock"
    && row.market.identifier === "binance" && row.base === "BBB" && row.target === "USDT").map((row) => row.base));
  const economic = discoverEconomicAssets(spot ?? [], usdm ?? [], catalog ?? [], verifiedStockTickers);
  const economicById = new Map(economic.assets.map((asset) => [asset.assetId, asset]));
  const economicRouteOwner = new Map(economic.assets.flatMap((asset) => asset.markets.map((route) => [marketKey(route), asset.assetId] as const)));
  const byAsset = new Map<string, Market[]>();
  for (const market of markets.filter((market) => !economicRouteOwner.has(marketKey(market)))) {
    const group = byAsset.get(market.baseAsset) ?? [];
    group.push(market);
    byAsset.set(market.baseAsset, group);
  }
  const archiveByAsset = new Map(archive.assets.map((asset) => [asset.baseAsset, asset]));
  const historicMarkets = (asset: string): readonly Market[] => {
    const archivedAsset = archiveByAsset.get(asset);
    const canonicalKey = archivedAsset?.symbol && archivedAsset.priceVenue ? `${archivedAsset.priceVenue}:${archivedAsset.symbol}` : null;
    const primaryOwner = canonicalKey ? economicRouteOwner.get(canonicalKey) : null;
    if (primaryOwner) return economicById.get(primaryOwner)!.markets;
    const canonicalCrypto = canonicalKey && markets.some((market) => marketKey(market) === canonicalKey && !economicRouteOwner.has(canonicalKey));
    if (!canonicalCrypto) {
      const explicitEconomic = economic.assets.find((candidate) => candidate.baseAsset === asset);
      if (explicitEconomic) return explicitEconomic.markets;
    }
    const prior = archivedAsset?.productSymbols;
    return markets.filter((market) => !economicRouteOwner.has(marketKey(market))
      && (market.baseAsset === asset || prior?.[market.venue]?.includes(market.symbol)));
  };
  const historicalRows = historical.map((asset, index) => {
    const routes = historicMarkets(asset);
    return { rank: index + 1, baseAsset: asset, assetId: routes.length ? economicRouteOwner.get(marketKey(routes[0])) ?? `crypto:${asset}` : `unavailable:${asset}`,
      origin: index < 257 ? "archived-1m-QR-basis" : "explicit-research-addition", markets: routes };
  });
  const catalogBySymbol = new Map<string, string[]>();
  for (const coin of catalog ?? []) {
    const symbol = coin.symbol.toUpperCase();
    const ids = catalogBySymbol.get(symbol) ?? [];
    ids.push(coin.id);
    catalogBySymbol.set(symbol, ids);
  }
  const rejected: Array<Readonly<{ id: string; symbol: string; marketCapUsd: number | null; reason: string; candidates?: readonly string[] }>> = [];
  const eligible: Array<Readonly<{ id: string; baseAsset: string; name: string; marketCapUsd: number; mapping: string; markets: readonly Market[] }>> = [];
  for (const coin of coins) {
    const asset = coin.symbol.toUpperCase();
    const candidates = catalogBySymbol.get(asset) ?? [];
    let reason = "";
    if (excludedIds.has(coin.id)) reason = `excluded-category:${excludedIds.get(coin.id)!.join(",")}`;
    else if (!(coin.market_cap! > 0)) reason = "missing-positive-market-cap";
    else if (!byAsset.has(asset)) reason = "no-active-exact-base-USDT-market";
    else if (VERIFIED_IDS[coin.id] !== asset && !(candidates.length === 1 && candidates[0] === coin.id)) reason = "ambiguous-or-unverified-ticker";
    if (reason) rejected.push({ id: coin.id, symbol: asset, marketCapUsd: coin.market_cap, reason,
      ...(reason === "ambiguous-or-unverified-ticker" ? { candidates } : {}) });
    else eligible.push({ id: coin.id, baseAsset: asset, name: coin.name, marketCapUsd: coin.market_cap!,
      mapping: VERIFIED_IDS[coin.id] === asset ? "explicit-canonical-CoinGecko-ID" : "unique-complete-CoinGecko-catalog-ticker", markets: byAsset.get(asset)! });
  }
  // Never assign the same Binance ticker to two CoinGecko identities.
  const duplicateAssets = new Set(eligible.filter((coin, index) => eligible.findIndex((other) => other.baseAsset === coin.baseAsset) !== index).map((coin) => coin.baseAsset));
  const ranked = eligible.filter((coin) => !duplicateAssets.has(coin.baseAsset)).sort((a, b) => b.marketCapUsd - a.marketCapUsd || a.id.localeCompare(b.id));
  for (const coin of eligible.filter((coin) => duplicateAssets.has(coin.baseAsset))) rejected.push({ id: coin.id, symbol: coin.baseAsset, marketCapUsd: coin.marketCapUsd, reason: "multiple-identities-for-Binance-base" });
  const totalCap = ranked.reduce((sum, coin) => sum + coin.marketCapUsd, 0);
  const ready = failures.length === 0 && !!spot && !!usdm && !!catalog && coins.length > 0 && totalCap > 0;
  const core: Array<(typeof ranked)[number] & Readonly<{ cumulativeCapShare: number }>> = [];
  let coreCap = 0;
  if (ready) for (const coin of ranked) {
    coreCap += coin.marketCapUsd;
    core.push({ ...coin, cumulativeCapShare: coreCap / totalCap });
    if (coreCap / totalCap >= coverage && core.length >= minimumCapCoreSize) break;
  }
  const availableHistory = historicalRows.filter((row) => row.markets.length > 0);
  const currentAssets = [...new Set([...availableHistory.map((row) => row.assetId), ...core.map((row) => `crypto:${row.baseAsset}`), ...economic.assets.map((asset) => asset.assetId)])];
  const entries = currentAssets.map((assetId, index) => {
    const historicalRow = historicalRows.find((row) => row.assetId === assetId);
    const sector = economicById.get(assetId);
    const baseAsset = sector?.baseAsset ?? historicalRow?.baseAsset ?? assetId.slice("crypto:".length);
    const cap = assetId.startsWith("crypto:") ? ranked.find((row) => row.baseAsset === baseAsset) : undefined;
    const routes = sector?.markets ?? (historicalRow?.markets.length ? historicalRow.markets : byAsset.get(baseAsset)!);
    return { rank: index + 1, assetId, baseAsset, name: sector?.name ?? cap?.name ?? baseAsset,
      historicalRank: historicalRow?.rank ?? null, mandatorySectorCore: !!sector,
      assetClass: sector?.assetClass ?? "crypto", exposureTags: sector?.exposureTags ?? ["crypto"],
      industrySector: sector?.industrySector ?? null, industryClassification: sector?.industryClassification ?? "not-applicable",
      industrySource: sector?.industrySource ?? null, regions: sector?.regions ?? ["global-crypto"],
      capitalization: sector?.capitalization ?? { valueUsd: cap?.marketCapUsd ?? null, reason: cap ? "CoinGecko global crypto capitalization" : "No identified capitalization in fetched snapshot" },
      capCore: core.some((row) => `crypto:${row.baseAsset}` === assetId), coinGeckoId: cap?.id ?? null,
      marketCapUsd: cap?.marketCapUsd ?? null, products: [...new Set(routes.map((row) => row.venue))],
      routes,
      priceVenue: routes[0].venue, productSymbols: Object.fromEntries((["spot", "usdm-futures"] as const)
        .map((venue) => [venue, routes.filter((row) => row.venue === venue).map((row) => row.symbol)])) };
  });
  const result = {
    version: 2, generatedAt: new Date().toISOString(), status: ready ? "refreshed" : "historical-restored-current-refresh-incomplete",
    purpose: "Broad economic coverage inventory across Binance-supported exposures. Historical diversity, correlated crypto cap leaders, and every metadata-backed noncrypto instrument are retained. This is not a newly optimized representative basis. No trading allocation or collector change.",
    parameters: { marketPages: pages, currency: "usd", targetEligibleSnapshotCapShare: coverage, minimumCapCoreSize, exclusions: CATEGORIES },
    historical: { file: path.relative(ROOT, archiveFile).replaceAll("\\", "/"), sha256: hash(archiveText), basisCount: 257,
      explicitAdditions: ADDITIONS, count: historical.length, entries: historicalRows },
    marketCapCore: { status: ready ? "refreshed" : "unrefreshed", denominator: "Sum of global USD market caps of uniquely mapped, non-excluded coins with active exact-base Binance USDT markets in the fetched CoinGecko snapshot; NOT total global cryptocurrency capitalization",
      fetchedCoinCount: coins.length, eligibleCoinCount: ready ? ranked.length : null, eligibleMarketCapUsd: ready ? totalCap : null,
      selectedMarketCapUsd: ready ? coreCap : null, achievedEligibleSnapshotCapShare: ready ? coreCap / totalCap : null,
      count: core.length, entries: core, verifiedIdMappings: VERIFIED_IDS },
    current: { status: spot && usdm ? "exchange-availability-refreshed" : "exchange-availability-incomplete", role: "broad-economic-coverage-inventory; not a newly optimized representative basis",
      availableHistoricalCount: availableHistory.length, unavailableHistorical: historicalRows.filter((row) => !row.markets.length).map((row) => row.baseAsset),
      addedByCapCore: core.filter((coin) => !historical.includes(coin.baseAsset)).map((coin) => coin.baseAsset), count: entries.length },
    sectorCore: { policy: "Include every discovered active noncrypto economic instrument; no correlation or capitalization cutoff. Spot token proxies and perpetuals remain separate routes to an economic identity.",
      count: economic.assets.length, assetIds: economic.assets.map((asset) => asset.assetId),
      coverage: ECONOMIC_COVERAGE_TAGS.map((tag) => ({ tag, assetIds: economic.assets.filter((asset) => asset.exposureTags.includes(tag)).map((asset) => asset.assetId) })),
      industryWitnesses: INDUSTRY_SECTORS.map((sector) => ({ sector, assetIds: economic.assets.filter((asset) => asset.industrySector === sector).map((asset) => asset.assetId) })),
      scope: "Binance public API spot/USD-M exposure coverage, not a spanning representation of the whole global economy. Issuer industry tags are partial curated witnesses, not a complete GICS classification.",
      capitalization: "Not ranked: missing verified global underlying cap data. No stock-cap, index-cap, ETF-AUM, commodity-stock, turnover or token-wrapper-cap equivalence is assumed.", sources: SECTOR_SOURCES },
    productDiscovery: { collectorTargetVenues: ["spot", "usdm-futures"],
      counts: { spot: spotFull?.symbols?.length ?? 0, usdm: usdmFull?.symbols?.length ?? 0, coinm: coinmFull?.symbols?.length ?? 0, options: optionsFull?.optionSymbols?.length ?? 0 },
      discoveryOnly: { reason: "COIN-M and options are inventoried but not collector targets; derivative strikes/expiries do not become separate economic assets.",
        coinm: (coinmFull?.symbols ?? []).map((value) => { const row = value as Readonly<{ symbol: string; baseAsset: string; contractStatus: string; underlyingType: string }>; return { symbol: row.symbol, baseAsset: row.baseAsset, status: row.contractStatus, underlyingType: row.underlyingType }; }),
        optionUnderlyings: optionsFull?.optionContracts ?? [] } },
    entries, assets: entries, snapshotProvenance: provenance, failures,
    diagnostics: { rejectedCoins: rejected, multiplierListingsNotInferred: markets.filter((market) => /^(1000|10000|1000000)(?=[A-Z])/.test(market.baseAsset)),
      ambiguousStockTokens: economic.ambiguousStockTokens,
      baseTickerCollisions: [...new Set(entries.filter((row, index) => entries.some((other, otherIndex) => otherIndex !== index && row.baseAsset === other.baseAsset)).map((row) => row.baseAsset))],
      archivedIdentityWarning: "The archived report conflated CAT crypto (1000CATUSDT) and CAT equity (CATUSDT). The exact historical name/order is retained; current identities and routes are split using assetId and actual Binance metadata.",
      historicalAliasRoutes: historicalRows.flatMap((row) => row.markets.filter((market) => market.baseAsset !== row.baseAsset).map((market) => ({ historicalAsset: row.baseAsset, ...market }))) },
  };
  if (new Set(entries.map((row) => row.assetId)).size !== entries.length) throw new Error("Duplicate output economic identities");
  const routeKeys = entries.flatMap((row) => row.routes.map(marketKey));
  if (new Set(routeKeys).size !== routeKeys.length) throw new Error("A collector route belongs to multiple identities");
  await fs.mkdir(path.dirname(output), { recursive: true });
  await fs.writeFile(output, `${JSON.stringify(result, null, 2)}\n`);
  const capDescription = ready
    ? `The global-cap core adds leaders regardless of correlation. It selects **${core.length}** assets to cover **${(100 * coreCap / totalCap).toFixed(3)}%** of **${ranked.length} eligible assets in the fetched ${coins.length}-coin snapshot**. This is not a claim of 95% coverage of total global crypto capitalization. Stablecoins, wrapped tokens, liquid staking tokens and liquid restaking tokens are excluded using saved CoinGecko category snapshots.`
    : `The global-cap supplement remains **unrefreshed** because required inputs are unavailable: ${failures.map((failure) => `${failure.input} (${failure.error})`).join("; ")}. The fetched market snapshot contains ${coins.length} unique coins. No cap-core membership or coverage claim is published. The intended policy requires complete stablecoin, wrapped-token, liquid-staking-token and liquid-restaking-token exclusion snapshots before selecting leaders to ${(100 * coverage).toFixed(1)}% of eligible fetched capitalization; that denominator is not total global crypto capitalization.`;
  const markdown = [
    "# Restored research market universe", `Generated ${result.generatedAt}. Status: ${result.status}.`,
    `Historical diversity manifest: 257 archived QR assets + ETH/SOL/XRP/HYPE = **261**, preserved in exact order independently of current availability. **${availableHistory.length}** currently have collector routes.`,
    capDescription + ` At least **${minimumCapCoreSize} crypto cap leaders** are retained for deliberate redundancy, even when the coverage threshold is crossed earlier.`,
    `Broad economic coverage inventory: **${entries.length} economic identities** and **${routeKeys.length} distinct venue/symbol routes**. Mandatory noncrypto inventory: **${economic.assets.length} identities**; every metadata-backed active noncrypto instrument is retained without a correlation cutoff. This is not a newly optimized representative basis or a claim of complete global-economy coverage.`,
    `Crypto cap core: ${core.map((coin) => coin.baseAsset).join(", ") || "unrefreshed"}. Added by crypto cap core beyond the historical list: ${result.current.addedByCapCore.join(", ") || "none"}.`,
    `Unavailable historical names: ${result.current.unavailableHistorical.join(", ") || "none"}.`,
    "Noncrypto targets include all explicitly identified TradFi perpetuals, named bStocks token proxies, gold-backed tokens and currently listed fiat/stablecoin pairs. These are Binance-traded exposures, rather than direct ownership of their referenced securities, commodities or indices. COIN-M and options are inventoried separately and are not collector targets.",
    "Global capitalization applies to the crypto leader selection. Verified global issuer caps, comparable index totals, commodity stocks and FX market sizes are not available in these inputs. ETF AUM, token-wrapper capitalization and derivative open interest are not substituted for underlying capitalization; noncrypto instruments are all retained instead.",
    "Use **assetId** for economic identity and **venue + symbol** for physical routes. `baseAsset` at entry level is the canonical/display ticker; each `routes[]` row preserves the actual Binance baseAsset/quoteAsset, instrumentKind and priceOrientation. A quote-side fiat pair such as USDTBRL remains raw BRL per USDT with priceOrientation=inverse. No prices are transformed by this manifest.",
    "The archived CAT row conflated Simon's Cat (1000CATUSDT) and Caterpillar equity (CATUSDT). Current identities are disjoint: crypto:CAT and tradfi:CAT. The historical name/order remains unchanged. Stock token proxy aliases require named provider identity plus a matching listed TradFi root; unmatched named stock tokens remain separate identities. Differently leveraged funds and regional/ADR contracts are not merged by issuer-name similarity.",
    `Complete Binance metadata and named CoinGecko catalog snapshots are retained in ${path.relative(ROOT, directory).replaceAll("\\", "/")}, alongside market/category inputs. Each source records URL, retrieval time and raw-response SHA-256. Rows retain provider update times. Identity ambiguities and rejected mappings remain visible in the JSON.`,
    "See restored-market-universe-sectors.md for classified exposure coverage, partial industry witnesses and gaps. Binance's supported subset is not a representation of every sector in the global economy.",
    `Reproduce:\n\n\`\`\`powershell\nnode --import tsx scripts/restore-research-market-universe.ts --offline --market-pages ${pages} --coverage ${coverage} --min-cap-core-size ${minimumCapCoreSize}\n\`\`\``,
    "The command does not configure or start collection.",
  ].join("\n\n") + "\n";
  await fs.writeFile(output.replace(/\.json$/i, "") + ".md", markdown);
  const sectorsMarkdown = ["# Binance economic exposure coverage", `Snapshot generated ${result.generatedAt}.`,
    "All directly identified noncrypto instruments are included. Exposure classes are based on current Binance contract metadata and explicitly named provider identities. Crypto projects are not counted as economic sector exposure because of marketing labels.",
    "| Exposure | Economic identities | Examples |\n| --- | ---: | --- |\n" + result.sectorCore.coverage.map((row) => `| ${row.tag} | ${row.assetIds.length} | ${row.assetIds.slice(0, 8).join(", ") || "Not identified in this snapshot"} |`).join("\n"),
    "Counts are not additive: one instrument may carry multiple exposure tags. Broad equity indices are accessed through ETF-linked perpetuals/token proxies, not treated as direct index ownership. Fixed income here is represented by leveraged/inverse long-Treasury ETFs (TMF/TBT), not direct bond markets. Volatility exposure is a VIX-futures ETF proxy (UVXY), not the spot VIX index.",
    "| Curated business sector witness | Included identities |\n| --- | --- |\n" + result.sectorCore.industryWitnesses.map((row) => `| ${row.sector} | ${row.assetIds.join(", ") || "No directly classified witness"} |`).join("\n"),
    "Industry witnesses are a small manually reviewed research taxonomy with issuer source links in the JSON. They are not a licensed/complete GICS classification. Unclassified equity/fund instruments are still retained. A missing witness means no direct verified classification here, rather than proof that a broad ETF has zero exposure to that sector.",
    "Agricultural commodities, direct sovereign bonds, corporate credit and direct real estate were not identified among the classified collector routes. Forex coverage is limited to the currently listed fiat/stablecoin pairs and USDBRL TradFi perpetual; it does not cover the full global FX matrix. Geographic labels come only from explicit regional metadata or reviewed ETF definitions; US listing location is not used as a claim about global revenue exposure.",
    "No noncrypto cap weights are calculated. Stock issuer capitalization, index constituent capitalization, ETF AUM, pre-IPO valuations, commodity inventory and currency turnover are economically different denominators. Obtain appropriate dated global underlying datasets before adding weights; the current policy retains the entire supported noncrypto set.",
    "Sources: [Binance TradFi structure](" + SECTOR_SOURCES.tradfi + "), [Binance energy contracts](" + SECTOR_SOURCES.energy + "), [Binance tokenized bStocks](" + SECTOR_SOURCES.bstocks + "), [CoinGecko named catalog](" + SECTOR_SOURCES.coinCatalog + "). [QQQ issuer](" + SECTOR_SOURCES.qqq + "), [TMF issuer](" + SECTOR_SOURCES.treasuriesBull + "), [TBT issuer](" + SECTOR_SOURCES.treasuriesBear + "), [UVXY issuer](" + SECTOR_SOURCES.volatility + "). Full listing evidence is in the saved Binance metadata snapshots.",
  ].join("\n\n") + "\n";
  await fs.writeFile(output.replace(/\.json$/i, "") + "-sectors.md", sectorsMarkdown);
  console.log(JSON.stringify({ output: path.relative(ROOT, output), status: result.status, historical: historical.length,
    availableHistorical: availableHistory.length, capCore: core.length, sectorCore: economic.assets.length, current: entries.length, routes: routeKeys.length, failures }));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run().catch((error) => { console.error(message(error)); process.exitCode = 1; });
}

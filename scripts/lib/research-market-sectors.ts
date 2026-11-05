/** Economic exposure comes from instrument metadata, not a crypto token's marketing sector. */
export type ResearchVenue = "spot" | "usdm-futures";
export type EconomicListing = Readonly<{
  symbol: string; baseAsset: string; quoteAsset: string; status: string;
  contractType?: string; underlyingType?: string; underlyingSubType?: readonly string[];
  isSpotTradingAllowed?: boolean;
}>;
export type IdentityCoin = Readonly<{ id: string; symbol: string; name?: string }>;
export type EconomicRoute = Readonly<{
  venue: ResearchVenue; symbol: string; baseAsset: string; quoteAsset: string;
  instrumentKind: string; priceOrientation: "direct" | "inverse";
  contractType?: string; underlyingType?: string; underlyingSubType?: readonly string[];
  identityEvidence: readonly string[];
}>;
export type EconomicAsset = Readonly<{
  assetId: string; baseAsset: string; name: string; assetClass: string;
  exposureTags: readonly string[]; regions: readonly string[];
  industrySector: string | null; industryClassification: string;
  industrySource: string | null; capitalization: Readonly<{ valueUsd: null; reason: string }>;
  markets: readonly EconomicRoute[];
}>;

const STABLE_QUOTES = new Set(["USDT", "USDC", "USD1", "FDUSD", "TUSD"]);
const FIAT = new Set("AED ARS AUD BDT BRL CHF COP CZK EUR GBP HUF IDR INR JPY KZT MXN NGN PHP PKR PLN RON RUB TRY UAH VND ZAR".split(" "));
export const SECTOR_SOURCES = {
  tradfi: "https://www.binance.com/en/learn/binance-futures-tradfi-perpetual-contracts",
  energy: "https://www.binance.com/en/support/announcement/detail/e7a10ca855524655ab3dea73add40ef4",
  bstocks: "https://www.binance.com/en-IA/how-to-buy/apple-tokenized-bstocks",
  coinCatalog: "https://api.coingecko.com/api/v3/coins/list",
  currencyCodes: "https://www.six-group.com/en/products-services/financial-information/data-standards.html",
  goldPaxos: "https://www.paxos.com/paxgold",
  goldTether: "https://gold.tether.to/",
  qqq: "https://www.invesco.com/us/en/financial-products/etfs/invesco-qqq-trust-series-1.html",
  treasuriesBull: "https://www.direxion.com/product/daily-20-year-treasury-bull-bear-3x-etfs",
  treasuriesBear: "https://www.proshares.com/our-etfs/leveraged-and-inverse/tbt",
  volatility: "https://www.proshares.com/our-etfs/leveraged-and-inverse/uvxy",
} as const;

const COMMODITIES: Readonly<Record<string, readonly [string, string]>> = {
  XAU: ["Gold", "precious-metals"], XAG: ["Silver", "precious-metals"],
  XPT: ["Platinum", "precious-metals"], XPD: ["Palladium", "precious-metals"],
  COPPER: ["Copper", "industrial-metals"], CL: ["WTI crude oil", "energy-commodities"],
  BZ: ["Brent crude oil", "energy-commodities"], NATGAS: ["Henry Hub natural gas", "energy-commodities"],
};
// These tags document a few verified sector witnesses; the rest remain explicitly
// unclassified by industry. They are research labels, not a licensed GICS dataset.
const INDUSTRIES: Readonly<Record<string, readonly [string, string]>> = {
  MSFT: ["information-technology", "https://www.microsoft.com/en-us/Investor/"],
  JPM: ["financials", "https://www.jpmorganchase.com/about"],
  META: ["communication-services", "https://about.meta.com/"],
  AMZN: ["consumer-discretionary", "https://www.aboutamazon.com/"],
  KO: ["consumer-staples", "https://www.coca-colacompany.com/about-us"],
  WMT: ["consumer-staples", "https://corporate.walmart.com/about"],
  LLY: ["health-care", "https://www.lilly.com/who-we-are"],
  CAT: ["industrials", "https://www.caterpillar.com/en/company.html"],
  CEG: ["utilities", "https://www.constellationenergy.com/our-company.html"],
  VST: ["utilities", "https://vistracorp.com/about/"],
  XOM: ["energy", "https://corporate.exxonmobil.com/who-we-are"],
  MP: ["materials", "https://mpmaterials.com/"],
};
const FUND_TAGS: Readonly<Record<string, readonly string[]>> = {
  SPY: ["equity-index-etf", "US-large-cap"], QQQ: ["equity-index-etf", "Nasdaq-100"],
  IWM: ["equity-index-etf", "US-small-cap"], EWY: ["country-equity-etf", "South-Korea"],
  EWJ: ["country-equity-etf", "Japan"], EWT: ["country-equity-etf", "Taiwan"],
  EWZ: ["country-equity-etf", "Brazil"], EWZS: ["country-equity-etf", "Brazil-small-cap"],
  EIDO: ["country-equity-etf", "Indonesia"], KODEX200: ["country-equity-etf", "South-Korea"],
  XLE: ["sector-equity-etf", "energy-equities"], XBI: ["sector-equity-etf", "biotechnology-equities"],
  SMH: ["sector-equity-etf", "semiconductor-equities"], GDX: ["sector-equity-etf", "gold-mining-equities"],
  URNM: ["sector-equity-etf", "uranium-mining-equities"],
  TMF: ["fixed-income-etf", "US-long-Treasuries", "daily-leverage-3x"],
  TBT: ["fixed-income-etf", "US-long-Treasuries", "daily-inverse-leverage-2x"],
  UVXY: ["volatility-futures-etf", "VIX-futures", "daily-leverage-1.5x"],
  TQQQ: ["equity-index-etf", "Nasdaq-100", "daily-leverage-3x"],
  SQQQ: ["equity-index-etf", "Nasdaq-100", "daily-inverse-leverage-3x"],
  SOXL: ["sector-equity-etf", "semiconductor-equities", "daily-leverage-3x"],
  SOXS: ["sector-equity-etf", "semiconductor-equities", "daily-inverse-leverage-3x"],
};
export const ECONOMIC_COVERAGE_TAGS = [
  "precious-metals", "industrial-metals", "energy-commodities", "foreign-exchange",
  "equities", "pre-IPO-equity", "equity-index-etf", "country-equity-etf",
  "sector-equity-etf", "fixed-income-etf", "volatility-futures-etf",
  "agricultural-commodities", "direct-sovereign-bonds", "corporate-credit", "real-estate-direct",
] as const;
export const INDUSTRY_SECTORS = ["information-technology", "financials", "communication-services", "consumer-discretionary",
  "consumer-staples", "health-care", "industrials", "utilities", "energy", "materials", "real-estate"] as const;

export function isTradFiListing(row: EconomicListing): boolean {
  return row.contractType === "TRADIFI_PERPETUAL" || row.underlyingSubType?.includes("TradFi") === true
    // PREMARKET alone also labels pre-listing crypto tokens. Require the TradFi
    // contract/subtype marker before treating a pre-market instrument as equity.
    || ["EQUITY", "KR_EQUITY", "HK_EQUITY", "CN_EQUITY", "COMMODITY", "FX"].includes(row.underlyingType ?? "");
}
export function marketKey(route: Readonly<{ venue: string; symbol: string }>): string { return `${route.venue}:${route.symbol}`; }

export function discoverEconomicAssets(
  spot: readonly EconomicListing[], usdm: readonly EconomicListing[], catalog: readonly IdentityCoin[],
  verifiedStockTickers: ReadonlySet<string> = new Set(),
): Readonly<{ assets: readonly EconomicAsset[]; ambiguousStockTokens: readonly Readonly<{ symbol: string; candidates: readonly string[] }>[] }> {
  const raw = new Map<string, { baseAsset: string; name: string; type: string; markets: EconomicRoute[] }>();
  const bySymbol = new Map<string, IdentityCoin[]>();
  for (const coin of catalog) {
    const symbol = coin.symbol.toUpperCase();
    const list = bySymbol.get(symbol) ?? [];
    list.push(coin); bySymbol.set(symbol, list);
  }
  const tradfi = usdm.filter((row) => row.status === "TRADING" && isTradFiListing(row));
  const tradfiBases = new Set(tradfi.map((row) => row.baseAsset));
  const add = (id: string, asset: string, name: string, type: string, row: EconomicListing, venue: ResearchVenue, kind: string, evidence: readonly string[], inverse = false) => {
    const group = raw.get(id) ?? { baseAsset: asset, name, type, markets: [] };
    if (group.name === asset && name !== asset) group.name = name;
    if (!group.markets.some((market) => market.venue === venue && market.symbol === row.symbol)) group.markets.push({
      venue, symbol: row.symbol, baseAsset: row.baseAsset, quoteAsset: row.quoteAsset,
      instrumentKind: kind, priceOrientation: inverse ? "inverse" : "direct",
      ...(row.contractType ? { contractType: row.contractType } : {}),
      ...(row.underlyingType ? { underlyingType: row.underlyingType } : {}),
      ...(row.underlyingSubType ? { underlyingSubType: row.underlyingSubType } : {}), identityEvidence: evidence,
    });
    raw.set(id, group);
  };
  for (const row of tradfi) add(`tradfi:${row.baseAsset}`, row.baseAsset, COMMODITIES[row.baseAsset]?.[0] ?? row.baseAsset,
    row.underlyingType ?? "TRADFI", row, "usdm-futures", "tradfi-perpetual", ["Binance exchangeInfo explicit TradFi/underlyingType"]);
  const ambiguousStockTokens: Array<{ symbol: string; candidates: readonly string[] }> = [];
  for (const row of spot.filter((row) => row.status === "TRADING" && row.isSpotTradingAllowed !== false && STABLE_QUOTES.has(row.quoteAsset))) {
    const candidates = bySymbol.get(row.baseAsset) ?? [];
    const stock = candidates.filter((coin) => /bStocks Tokenized|Tokenized bStocks/i.test(coin.name ?? ""));
    if (stock.length === 1 && (candidates.length === 1 || verifiedStockTickers.has(row.baseAsset))) {
      const root = row.baseAsset.endsWith("B") ? row.baseAsset.slice(0, -1) : row.baseAsset;
      const shared = tradfiBases.has(root);
      add(shared ? `tradfi:${root}` : `tradfi-token:${row.baseAsset}`, shared ? root : row.baseAsset,
        stock[0].name!.replace(/\s*\(bStocks Tokenized (?:Stock|ETF)\)/i, ""), "EQUITY", row, "spot", "tokenized-bstock",
        [`CoinGecko ID:${stock[0].id}`, shared ? "Exact Binance TradFi base plus B-Stock suffix; instrument routes retained separately" : "Named stock token; no inferred perpetual alias"]);
    } else if (stock.length) ambiguousStockTokens.push({ symbol: row.symbol, candidates: candidates.map((coin) => coin.id) });
    if (row.baseAsset === "PAXG" || row.baseAsset === "XAUT") add("tradfi:XAU", "XAU", "Gold", "COMMODITY", row, "spot", "gold-backed-token",
      [row.baseAsset === "PAXG" ? SECTOR_SOURCES.goldPaxos : SECTOR_SOURCES.goldTether]);
    if (FIAT.has(row.baseAsset)) add(`fx:${row.baseAsset}`, row.baseAsset, row.baseAsset, "FIAT", row, "spot", "fiat-stablecoin-spot", [SECTOR_SOURCES.currencyCodes]);
  }
  for (const row of spot.filter((row) => row.status === "TRADING" && row.isSpotTradingAllowed !== false && FIAT.has(row.quoteAsset) && STABLE_QUOTES.has(row.baseAsset))) {
    add(`fx:${row.quoteAsset}`, row.quoteAsset, row.quoteAsset, "FIAT", row, "spot", "fiat-stablecoin-spot", [SECTOR_SOURCES.currencyCodes], true);
  }
  // Preserve gold spot/perpetual redundancy while keeping token cap distinct from gold capitalization.
  for (const row of usdm.filter((row) => row.status === "TRADING" && ["PAXG", "XAUT"].includes(row.baseAsset))) {
    add("tradfi:XAU", "XAU", "Gold", "COMMODITY", row, "usdm-futures", "gold-token-perpetual", [row.baseAsset === "PAXG" ? SECTOR_SOURCES.goldPaxos : SECTOR_SOURCES.goldTether]);
  }
  const assets = [...raw].map(([assetId, group]): EconomicAsset => {
    const commodity = COMMODITIES[group.baseAsset];
    const isFx = group.type === "FIAT" || group.type === "FX";
    const tags = commodity ? ["commodities", commodity[1]] : isFx ? ["foreign-exchange"] : ["equities", ...(group.type === "PREMARKET" ? ["pre-IPO-equity"] : []), ...(FUND_TAGS[group.baseAsset] ?? [])];
    const region = group.type === "KR_EQUITY" ? "South-Korea" : group.type === "HK_EQUITY" ? "Hong-Kong" : group.type === "CN_EQUITY" ? "China" : "not-inferred-from-ticker";
    const industry = !commodity && !isFx ? INDUSTRIES[group.baseAsset] : undefined;
    return { assetId, baseAsset: group.baseAsset, name: group.name,
      assetClass: commodity ? "commodity-exposure" : isFx ? "foreign-exchange" : "equity-or-fund-exposure",
      exposureTags: tags, regions: [region], industrySector: industry?.[0] ?? null,
      industryClassification: industry ? "curated-business-sector-witness; not GICS" : "not-classified",
      industrySource: industry?.[1] ?? null,
      capitalization: { valueUsd: null, reason: "No comparable verified global underlying capitalization snapshot. Token wrapper cap, ETF AUM, derivatives open interest and underlying issuer cap are different quantities; all listed noncrypto instruments are retained." },
      markets: group.markets.sort((a, b) => a.venue.localeCompare(b.venue) || a.symbol.localeCompare(b.symbol)) };
  }).sort((a, b) => a.assetId.localeCompare(b.assetId));
  return { assets, ambiguousStockTokens };
}

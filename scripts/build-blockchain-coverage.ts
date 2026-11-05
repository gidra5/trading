import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { fileURLToPath } from "node:url";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const COINS = "https://api.coingecko.com/api/v3/coins/list?include_platform=true";
const PLATFORMS = "https://api.coingecko.com/api/v3/asset_platforms";
const BINANCE_ASSETS = "https://www.binance.com/bapi/asset/v2/public/asset/asset/get-all-asset";
const BINANCE_FUTURES = "https://api.coingecko.com/api/v3/derivatives/exchanges/binance_futures?include_tickers=all";
const EVM_REGISTRY = "https://chainid.network/chains_mini.json";

type Coin = Readonly<{ id: string; symbol: string; name: string; platforms: Readonly<Record<string, string>> }>;
type Platform = Readonly<{ id: string; chain_identifier: number | null; native_coin_id: string | null; name: string }>;
type Route = Readonly<{ venue: string; symbol: string; baseAsset: string; quoteAsset: string }>;
type Asset = Readonly<{
  assetClass: string;
  assetId: string;
  baseAsset: string;
  capCore: boolean;
  coinGeckoId: string | null;
  marketCapUsd: number | null;
  routes: readonly Route[];
}>;
type BinanceAsset = Readonly<{ assetCode: string; assetName: string }>;
type Future = Readonly<{
  base: string; target: string; symbol: string; coin_id: string | null; trade_url: string;
  contract_type: string; expired_at: number | null;
}>;
type RegistryNetwork = Readonly<{ chainId: number; name: string; infoURL: string }>;
type ReviewedIdentity = Readonly<{
  id: string;
  source: string;
  providerAliases?: readonly string[];
  deployment?: Readonly<{ platform: string; address: string; source?: string }>;
}>;

const reviewedIdentities: Readonly<Partial<Record<string, ReviewedIdentity>>> = {
  "crypto:BTTC": {
    id: "bittorrent",
    source: "https://www.binance.com/en/support/announcement/detail/2722e6da4f5141dd9b2fb07b5b1f75",
    deployment: { platform: "tron", address: "TAFjULxiVgT4qWk6UZwjqwZXTSaGaqnVp4" },
  },
  "crypto:DODO": {
    id: "dodo",
    source: "https://www.binance.com/en-UZ/price/dodo",
    deployment: {
      platform: "ethereum",
      address: "0x43Dfc4159D86F3A37A5A4B3D4580b888ad7d4DDd",
      source: "https://docs.dodoex.io/zh/developer/contracts/dodo-v1-v2/contracts-address/ethereum",
    },
  },
  "crypto:BAR": {
    id: "fc-barcelona-fan-token",
    source: "https://www.binance.com/en/support/announcement/detail/bc5b4c305f0a4dcdbef2796b39e252d3",
  },
  "crypto:SATS": {
    id: "sats-ordinals",
    source: "https://www.binance.com/en/support/announcement/detail/67d516a4ecec4c3ca17760d9afe04570",
    providerAliases: ["1000sats-ordinals"],
  },
  "crypto:TKO": {
    id: "tokocrypto",
    source: "https://www.binance.com/en/support/announcement/detail/4620c8a2a87c42978519750964af7aa4",
  },
  "crypto:PORTO": {
    id: "fc-porto",
    source: "https://www.binance.com/en/support/announcement/detail/b4bdead0b59144aca40b9b39902c0c4b",
  },
  "crypto:VTHO": {
    id: "vethor-token",
    source: "https://docs.vechain.org/introduction-to-vechain/dual-token-economic-model/vethor-vtho",
    deployment: { platform: "vechain", address: "0x0000000000000000000000000000456E65726779" },
  },
};

type NativeNetwork = Readonly<{ platform: string; assetId: string; name: string; source: string; chainId?: number }>;
const reviewedNativeNetworks: readonly NativeNetwork[] = [
  { platform: "bitcoin", assetId: "crypto:BTC", name: "Bitcoin", source: "https://developer.bitcoin.org/reference/block_chain.html" },
  { platform: "ethereum", assetId: "crypto:ETH", name: "Ethereum", source: "https://ethereum.org/en/eth/", chainId: 1 },
  { platform: "binance-smart-chain", assetId: "crypto:BNB", name: "BNB Smart Chain", source: "https://www.bnbchain.org/en/blog/bsc-is-now-bnb-chain-the-infrastructure-for-the-metafi-universe", chainId: 56 },
  { platform: "xrp", assetId: "crypto:XRP", name: "XRP Ledger", source: "https://xrpl.org/about/xrp" },
  { platform: "solana", assetId: "crypto:SOL", name: "Solana", source: "https://solana.com/docs/references/terminology" },
  { platform: "tron", assetId: "crypto:TRX", name: "TRON", source: "https://developers.tron.network/docs/token-standards-trx" },
  { platform: "hyperliquid", assetId: "crypto:HYPE", name: "HyperCore", source: "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/hyperevm" },
  { platform: "hyperevm", assetId: "crypto:HYPE", name: "HyperEVM", source: "https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/hyperevm", chainId: 999 },
  { platform: "zcash", assetId: "crypto:ZEC", name: "Zcash", source: "https://zips.z.cash/protocol/protocol.pdf" },
  { platform: "dogecoin", assetId: "crypto:DOGE", name: "Dogecoin", source: "https://github.com/dogecoin/dogecoin" },
  { platform: "monero", assetId: "crypto:XMR", name: "Monero", source: "https://www.getmonero.org/resources/about/" },
  { platform: "nano", assetId: "crypto:XNO", name: "Nano", source: "https://docs.nano.org/protocol-design/introduction/" },
  { platform: "harmony-shard-0", assetId: "crypto:ONE", name: "Harmony Shard 0", source: "https://docs.harmony.one/home/general/ecosystem/wallets/browser-extensions-wallets/metamask-wallet/adding-harmony", chainId: 1666600000 },
  { platform: "digibyte", assetId: "crypto:DGB", name: "DigiByte", source: "https://www.digibyte.org/en/" },
  { platform: "somnia", assetId: "crypto:SOMI", name: "Somnia", source: "https://new.docs.somnia.network/network-info", chainId: 5031 },
  { platform: "henesys", assetId: "crypto:NXPC", name: "Henesys", source: "https://support.msu.io/hc/en-us/articles/12746827359375-Important-Notice-Regarding-NXPC-Deposits-and-Withdrawals-to-from-Centralized-Exchanges-CEX" },
  { platform: "theta", assetId: "crypto:TFUEL", name: "Theta", source: "https://docs.thetatoken.org/docs/web3-stack-metamask", chainId: 361 },
  { platform: "canton-network", assetId: "crypto:CC", name: "Canton Global Synchronizer", source: "https://www.canton.network/global-synchronizer" },
  { platform: "decred", assetId: "crypto:DCR", name: "Decred", source: "https://docs.decred.org/getting-started/obtaining-dcr/" },
  { platform: "ark", assetId: "crypto:ARK", name: "ARK", source: "https://ark.dev/docs/platform-sdk/specification" },
  { platform: "ecash", assetId: "crypto:XEC", name: "eCash", source: "https://www.e.cash/tech" },
  { platform: "monad", assetId: "crypto:MON", name: "Monad", source: "https://github.com/monad-developers", chainId: 143 },
  { platform: "ethereumpow", assetId: "crypto:ETHW", name: "EthereumPoW", source: "https://github.com/ethereumpow", chainId: 10001 },
  { platform: "pharos", assetId: "crypto:PHAROS", name: "Pharos", source: "https://www.binance.com/en/support/announcement/detail/12bcf9092e9c46358de3e49f8ef17f88" },
  { platform: "gunz", assetId: "crypto:GUN", name: "GUNZ", source: "https://gunbygunz.com/documentation/", chainId: 43419 },
  { platform: "the-open-network", assetId: "crypto:GRAM", name: "TON", source: "https://docs.ton.org/applications/payments/overview" },
  { platform: "iota", assetId: "crypto:IOTA", name: "IOTA", source: "https://docs.iota.org/about-iota/tokenomics" },
  { platform: "iotex", assetId: "crypto:IOTX", name: "IoTeX", source: "https://docs.iotex.io/blockchain/learn-iotex/exchange-integration", chainId: 4689 },
];

const indexingOverlays = new Map([["ordinals", "bitcoin"], ["drc-20", "dogecoin"]]);

type Snapshot<T> = Readonly<{ source: string; retrievedAt: string; responseSha256: string; data: T }>;
const normalize = (name: string) => name.normalize("NFKC").trim().toLowerCase();
const sameAddress = (expected: string, candidate: string) => /^0x[a-f0-9]{40}$/i.test(expected)
  ? expected.toLowerCase() === candidate.toLowerCase()
  : expected === candidate;

export async function run(args = process.argv.slice(2)) {
  const directory = path.join(ROOT, "docs/collector/blockchain-sources");
  const universe: { entries: readonly Asset[] } = JSON.parse(
    await fs.readFile(path.join(ROOT, "docs/portfolio/restored-market-universe.json"), "utf8"),
  );
  const entries = universe.entries.filter((entry) => entry.assetClass === "crypto");
  const binanceSymbols = new Set(entries.map((entry) => entry.baseAsset));
  const futuresSymbols = new Set(entries.flatMap((entry) =>
    entry.routes.filter((route) => route.venue === "usdm-futures").map((route) => route.symbol)));
  await fs.mkdir(directory, { recursive: true });
  async function input<T>(name: string, url: string, project: (value: unknown) => T): Promise<Snapshot<T>> {
    const file = path.join(directory, `${name}.json`);
    if (args.includes("--offline")) {
      const snapshot: Snapshot<T> = JSON.parse(await fs.readFile(file, "utf8"));
      if (snapshot.source !== url || !Array.isArray(snapshot.data)) throw new Error(`Invalid snapshot ${name}`);
      return snapshot;
    }
    const response = await fetch(url, { signal: AbortSignal.timeout(45_000) });
    if (!response.ok) throw new Error(`Blockchain inventory source ${name} HTTP ${response.status}`);
    const raw = await response.text();
    const data = project(JSON.parse(raw));
    if (!Array.isArray(data) || data.length === 0) throw new Error(`Invalid blockchain inventory input ${name}`);
    const snapshot = { source: url, retrievedAt: new Date().toISOString(),
      responseSha256: createHash("sha256").update(raw).digest("hex"), data };
    await fs.writeFile(file, JSON.stringify(snapshot) + "\n");
    return snapshot;
  }
  const coinSnapshot = await input("coins-platforms", COINS, (value) => value as readonly Coin[]);
  const platformSnapshot = await input("platforms", PLATFORMS, (value) => value as readonly Platform[]);
  const { data: coins } = coinSnapshot;
  const { data: platforms } = platformSnapshot;
  const platformById = new Map(platforms.map((platform) => [platform.id, platform]));
  const relevantChainIds = new Set(platforms.map((platform) => platform.chain_identifier).filter((id) => id !== null));
  for (const network of reviewedNativeNetworks) if (network.chainId !== undefined) relevantChainIds.add(network.chainId);
  const [assetSnapshot, futuresSnapshot, registrySnapshot] = await Promise.all([
    input("binance-asset-names", BINANCE_ASSETS, (value) =>
      (value as { data: readonly BinanceAsset[] }).data.filter((row) => binanceSymbols.has(row.assetCode))
        .map(({ assetCode, assetName }) => ({ assetCode, assetName }))),
    input("coingecko-binance-futures", BINANCE_FUTURES, (value) =>
      (value as { tickers: readonly Future[] }).tickers.filter((row) => futuresSymbols.has(row.symbol))
        .map(({ base, target, symbol, coin_id, trade_url, contract_type, expired_at }) =>
          ({ base, target, symbol, coin_id, trade_url, contract_type, expired_at }))),
    input("eip155-registry-networks", EVM_REGISTRY, (value) =>
      (value as readonly RegistryNetwork[]).filter((row) => relevantChainIds.has(row.chainId))
        .map(({ chainId, name, infoURL }) => ({ chainId, name, infoURL }))),
  ]);
  const byId = new Map(coins.map((coin) => [coin.id, coin]));
  const bySymbol = Map.groupBy(coins, (coin) => coin.symbol.toUpperCase());
  const assetBySymbol = Map.groupBy(assetSnapshot.data, (asset) => asset.assetCode);
  const futuresBySymbol = Map.groupBy(futuresSnapshot.data, (ticker) => ticker.symbol);
  const registry = Map.groupBy(registrySnapshot.data, (network) => network.chainId);
  const assets = entries.map((entry) => {
    const candidates = bySymbol.get(entry.baseAsset) ?? [];
    const reviewed = reviewedIdentities[entry.assetId];
    const binanceNames = [...new Set((assetBySymbol.get(entry.baseAsset) ?? []).map((asset) => normalize(asset.assetName)))];
    const namedCandidates = candidates.filter((candidate) => binanceNames.includes(normalize(candidate.name)));
    const futuresMatches = entry.routes.filter((route) => route.venue === "usdm-futures").flatMap((route) =>
      (futuresBySymbol.get(route.symbol) ?? []).filter((ticker) =>
        ticker.base === route.baseAsset && ticker.target === route.quoteAsset &&
        ticker.trade_url === `https://www.binance.com/en/futures/${route.symbol}` &&
        ticker.contract_type === "perpetual" && ticker.expired_at === null && ticker.coin_id !== null));
    const routeIds = [...new Set(futuresMatches.flatMap(({ coin_id }) =>
      coin_id ? [reviewed?.providerAliases?.includes(coin_id) ? reviewed.id : coin_id] : []))];
    const existingId = reviewed?.id ?? entry.coinGeckoId;
    const inferredId = routeIds[0] ?? (namedCandidates.length === 1 ? namedCandidates[0]?.id : undefined);
    const preferredId = existingId ?? inferredId;
    const conflict = preferredId !== undefined && preferredId !== null && (
      routeIds.some((id) => id !== preferredId)
      || (namedCandidates.length > 0 && !namedCandidates.some((candidate) => candidate.id === preferredId))
      || (reviewed !== undefined && entry.coinGeckoId !== null && entry.coinGeckoId !== reviewed.id)
    );
    const coin = !conflict && preferredId ? byId.get(preferredId) : undefined;
    const mapping = !coin ? (conflict ? "conflicting-evidence" : "unresolved")
      : reviewed ? "reviewed-listing-or-deployment"
        : existingId ? "existing-identified-ID"
          : routeIds.length ? "provider-exact-binance-futures-route"
            : "exact-binance-name-and-ticker";
    const deployment = coin ? reviewed?.deployment : undefined;
    const contracts = Object.entries(coin?.platforms ?? {}).filter(([platform, address]) => platform && address)
      .map(([platform, address]) => {
        const documented = deployment?.platform === platform && sameAddress(deployment.address, address);
        return { platform, address, chainIdentifier: platformById.get(platform)?.chain_identifier ?? null,
          source: COINS, deploymentVerified: false,
          documentation: documented ? deployment.source ?? reviewed?.source ?? null : null };
      });
    if (reviewed && deployment && !contracts.some((contract) =>
      contract.platform === deployment.platform && sameAddress(deployment.address, contract.address))) {
      const source = deployment.source ?? reviewed.source;
      contracts.push({ platform: deployment.platform, address: deployment.address,
        chainIdentifier: platformById.get(deployment.platform)?.chain_identifier ?? null, source,
        deploymentVerified: false, documentation: source });
    }
    const needsReview = [...new Set([...candidates.map((row) => row.id), ...routeIds, ...(preferredId ? [preferredId] : [])])];
    return { assetId: entry.assetId, coinGeckoId: coin?.id ?? null, mapping,
      evidence: {
        reviewedSource: reviewed?.source ?? null,
        binanceNames: (assetBySymbol.get(entry.baseAsset) ?? []).map((asset) => asset.assetName),
        futuresRoutes: futuresMatches.map(({ symbol, coin_id }) => ({ symbol, coinGeckoId: coin_id })),
      },
      candidates: coin ? [] : needsReview.map((id) => ({ id, name: byId.get(id)?.name ?? null })),
      contracts,
      nativePlatformCandidates: coin ? platforms.filter((row) => row.native_coin_id === coin.id).map((row) => row.id) : [],
      captureEnabled: false };
  });
  const assetById = new Map(assets.map((asset) => [asset.assetId, asset]));
  const deploymentNetworks = new Set(assets.flatMap((asset) => asset.contracts.map((contract) => contract.platform)));
  const providerDeploymentNetworks = new Set(assets.flatMap((asset) =>
    asset.contracts.filter((contract) => contract.source === COINS).map((contract) => contract.platform)));
  const nativeNetworks = reviewedNativeNetworks.filter((network) => assetById.get(network.assetId)?.coinGeckoId);
  const nativeByPlatform = Map.groupBy(nativeNetworks, (network) => network.platform);
  const candidateNetworks = new Set(assets.flatMap((asset) => asset.nativePlatformCandidates));
  const chainInventory = [...new Set([...deploymentNetworks, ...nativeByPlatform.keys(), ...candidateNetworks])]
    .map((platform) => {
      const native = nativeByPlatform.get(platform) ?? [];
      const metadata = platformById.get(platform);
      const chainIdentifier = native[0]?.chainId ?? metadata?.chain_identifier ?? null;
      const registryCandidates = chainIdentifier === null ? [] : registry.get(chainIdentifier) ?? [];
      const registered = registryCandidates.length === 1 && !/\btest\s*net\b/i.test(registryCandidates[0].name)
        ? registryCandidates[0] : null;
      const tokenAssets = assets.filter((asset) => asset.contracts.some((contract) => contract.platform === platform));
      const rankedAssetIds = new Set([...native.map((network) => network.assetId), ...tokenAssets.map((asset) => asset.assetId)]);
      const cap = entries.filter((entry) => rankedAssetIds.has(entry.assetId) && entry.marketCapUsd !== null);
      return {
        platform,
        name: native[0]?.name ?? metadata?.name ?? platform,
        indexingOverlayFor: indexingOverlays.get(platform) ?? null,
        eip155Registry: registered,
        registryCandidates: registered ? [] : registryCandidates,
        documentedChainIdentifier: native[0]?.chainId ?? null,
        providerChainIdentifier: metadata?.chain_identifier ?? null,
        nativeAssets: native.map(({ assetId, source }) => ({ assetId, source })),
        tokenAssets: tokenAssets.length,
        nativeCandidateAssets: assets.filter((asset) => asset.nativePlatformCandidates.includes(platform)).map((asset) => asset.assetId),
        capCoreAssets: cap.filter((entry) => entry.capCore).map((entry) => entry.assetId),
        highestMappedAssetCapUsd: cap.length ? Math.max(...cap.map((entry) => entry.marketCapUsd!)) : null,
        coverageEvidence: native.length ? "reviewed-native-asset"
          : providerDeploymentNetworks.has(platform) ? "provider-reported-token-deployment"
            : tokenAssets.length ? "documented-token-deployment"
              : "provider-native-candidate-only",
      };
    }).sort((a, b) =>
      Number(b.coverageEvidence !== "provider-native-candidate-only") - Number(a.coverageEvidence !== "provider-native-candidate-only")
      || (b.highestMappedAssetCapUsd ?? 0) - (a.highestMappedAssetCapUsd ?? 0)
      || b.tokenAssets - a.tokenAssets || a.platform.localeCompare(b.platform));
  const sources = [coinSnapshot, platformSnapshot, assetSnapshot, futuresSnapshot, registrySnapshot];
  const result = {
    version: 2,
    generatedAt: sources.map((source) => source.retrievedAt).sort().at(-1),
    cryptoAssets: assets.length,
    identifiedAssets: assets.filter((asset) => asset.coinGeckoId).length,
    unresolvedAssets: assets.filter((asset) => !asset.coinGeckoId).length,
    providerReportedDeploymentPlatforms: providerDeploymentNetworks.size,
    reviewedNativeNetworks: nativeNetworks.length,
    deploymentOrReviewedNativeNetworks: chainInventory.filter((network) => network.coverageEvidence !== "provider-native-candidate-only").length,
    eip155RegisteredDeploymentOrReviewedNativeNetworks: chainInventory.filter((network) =>
      network.eip155Registry && network.coverageEvidence !== "provider-native-candidate-only").length,
    sources: sources.map(({ source, retrievedAt, responseSha256 }) => ({ source, retrievedAt, responseSha256 })),
    assets,
    chainInventory,
    limitations: [
      "Exact route and name matches identify provider records; deployment addresses still need direct chain verification.",
      "The EIP-155 registry corroborates chain IDs, not endpoint behavior or which market assets are deployed there. Ambiguous or testnet records require review.",
      "Token deployments include bridged representations and indexing overlays. They are not all separate chains or equivalent economic flows.",
      "Gas-asset candidates without deployment or reviewed native evidence are excluded from the represented coverage counts.",
      "Highest mapped market cap uses the existing global USD market-cap snapshot. Chains can share the same asset; do not sum those caps.",
      "No exchange-wallet identity labels or buy/sell intent are inferred from transfers. This inventory never starts capture.",
    ],
  };
  await fs.writeFile(path.join(ROOT, "docs/collector/blockchain-coverage.json"), JSON.stringify(result, null, 2) + "\n");
  console.log(JSON.stringify({ assets: result.cryptoAssets, identified: result.identifiedAssets,
    unresolved: result.unresolvedAssets, providerDeployments: result.providerReportedDeploymentPlatforms,
    reviewedNativeNetworks: result.reviewedNativeNetworks, representedNetworks: result.deploymentOrReviewedNativeNetworks,
    registryMatches: result.eip155RegisteredDeploymentOrReviewedNativeNetworks }));
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  run().catch((error: Error) => { console.error(error.message); process.exitCode = 1; });
}

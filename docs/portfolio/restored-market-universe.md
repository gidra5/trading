# Restored research market universe

Generated 2026-11-01T19:47:01.779Z. Status: refreshed.

Historical diversity manifest: 257 archived QR assets + ETH/SOL/XRP/HYPE = **261**, preserved in exact order independently of current availability. **255** currently have collector routes.

The global-cap core adds leaders regardless of correlation. It selects **10** assets to cover **95.686%** of **247 eligible assets in the fetched 999-coin snapshot**. This is not a claim of 95% coverage of total global crypto capitalization. Stablecoins, wrapped tokens, liquid staking tokens and liquid restaking tokens are excluded using saved CoinGecko category snapshots. At least **10 crypto cap leaders** are retained for deliberate redundancy, even when the coverage threshold is crossed earlier.

Broad economic coverage inventory: **460 economic identities** and **687 distinct venue/symbol routes**. Mandatory noncrypto inventory: **248 identities**; every metadata-backed active noncrypto instrument is retained without a correlation cutoff. This is not a newly optimized representative basis or a claim of complete global-economy coverage.

Crypto cap core: BTC, ETH, BNB, XRP, SOL, TRX, HYPE, ZEC, DOGE, XMR. Added by crypto cap core beyond the historical list: BNB, ZEC, DOGE.

Unavailable historical names: PYR, PIVX, PUMPBTC, ICX, VANRY, STG.

Noncrypto targets include all explicitly identified TradFi perpetuals, named bStocks token proxies, gold-backed tokens and currently listed fiat/stablecoin pairs. These are Binance-traded exposures, rather than direct ownership of their referenced securities, commodities or indices. COIN-M and options are inventoried separately and are not collector targets.

Global capitalization applies to the crypto leader selection. Verified global issuer caps, comparable index totals, commodity stocks and FX market sizes are not available in these inputs. ETF AUM, token-wrapper capitalization and derivative open interest are not substituted for underlying capitalization; noncrypto instruments are all retained instead.

Use **assetId** for economic identity and **venue + symbol** for physical routes. `baseAsset` at entry level is the canonical/display ticker; each `routes[]` row preserves the actual Binance baseAsset/quoteAsset, instrumentKind and priceOrientation. A quote-side fiat pair such as USDTBRL remains raw BRL per USDT with priceOrientation=inverse. No prices are transformed by this manifest.

The archived CAT row conflated Simon's Cat (1000CATUSDT) and Caterpillar equity (CATUSDT). Current identities are disjoint: crypto:CAT and tradfi:CAT. The historical name/order remains unchanged. Stock token proxy aliases require named provider identity plus a matching listed TradFi root; unmatched named stock tokens remain separate identities. Differently leveraged funds and regional/ADR contracts are not merged by issuer-name similarity.

Complete Binance metadata and named CoinGecko catalog snapshots are retained in docs/portfolio/restored-market-universe-sources, alongside market/category inputs. Each source records URL, retrieval time and raw-response SHA-256. Rows retain provider update times. Identity ambiguities and rejected mappings remain visible in the JSON.

See restored-market-universe-sectors.md for classified exposure coverage, partial industry witnesses and gaps. Binance's supported subset is not a representation of every sector in the global economy.

Reproduce:

```powershell
node --import tsx scripts/restore-research-market-universe.ts --offline --market-pages 4 --coverage 0.95 --min-cap-core-size 10
```

The command does not configure or start collection.

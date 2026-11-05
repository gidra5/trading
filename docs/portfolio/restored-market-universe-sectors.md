# Binance economic exposure coverage

Snapshot generated 2026-11-01T19:47:01.779Z.

All directly identified noncrypto instruments are included. Exposure classes are based on current Binance contract metadata and explicitly named provider identities. Crypto projects are not counted as economic sector exposure because of marketing labels.

| Exposure | Economic identities | Examples |
| --- | ---: | --- |
| precious-metals | 4 | tradfi:XAG, tradfi:XAU, tradfi:XPD, tradfi:XPT |
| industrial-metals | 1 | tradfi:COPPER |
| energy-commodities | 3 | tradfi:BZ, tradfi:CL, tradfi:NATGAS |
| foreign-exchange | 11 | fx:AED, fx:ARS, fx:BRL, fx:COP, fx:EUR, fx:IDR, fx:KZT, fx:MXN |
| equities | 229 | tradfi-token:BBB, tradfi-token:QNTB, tradfi-token:STXB, tradfi-token:USDEB, tradfi:AAOI, tradfi:AAPL, tradfi:ABBV, tradfi:ACN |
| pre-IPO-equity | 4 | tradfi:ANTHROPIC, tradfi:MOONSHOT, tradfi:OPENAI, tradfi:OURA |
| equity-index-etf | 5 | tradfi:IWM, tradfi:QQQ, tradfi:SPY, tradfi:SQQQ, tradfi:TQQQ |
| country-equity-etf | 7 | tradfi:EIDO, tradfi:EWJ, tradfi:EWT, tradfi:EWY, tradfi:EWZ, tradfi:EWZS, tradfi:KODEX200 |
| sector-equity-etf | 7 | tradfi:GDX, tradfi:SMH, tradfi:SOXL, tradfi:SOXS, tradfi:URNM, tradfi:XBI, tradfi:XLE |
| fixed-income-etf | 2 | tradfi:TBT, tradfi:TMF |
| volatility-futures-etf | 1 | tradfi:UVXY |
| agricultural-commodities | 0 | Not identified in this snapshot |
| direct-sovereign-bonds | 0 | Not identified in this snapshot |
| corporate-credit | 0 | Not identified in this snapshot |
| real-estate-direct | 0 | Not identified in this snapshot |

Counts are not additive: one instrument may carry multiple exposure tags. Broad equity indices are accessed through ETF-linked perpetuals/token proxies, not treated as direct index ownership. Fixed income here is represented by leveraged/inverse long-Treasury ETFs (TMF/TBT), not direct bond markets. Volatility exposure is a VIX-futures ETF proxy (UVXY), not the spot VIX index.

| Curated business sector witness | Included identities |
| --- | --- |
| information-technology | tradfi:MSFT |
| financials | tradfi:JPM |
| communication-services | tradfi:META |
| consumer-discretionary | tradfi:AMZN |
| consumer-staples | tradfi:KO, tradfi:WMT |
| health-care | tradfi:LLY |
| industrials | tradfi:CAT |
| utilities | tradfi:CEG, tradfi:VST |
| energy | tradfi:XOM |
| materials | tradfi:MP |
| real-estate | No directly classified witness |

Industry witnesses are a small manually reviewed research taxonomy with issuer source links in the JSON. They are not a licensed/complete GICS classification. Unclassified equity/fund instruments are still retained. A missing witness means no direct verified classification here, rather than proof that a broad ETF has zero exposure to that sector.

Agricultural commodities, direct sovereign bonds, corporate credit and direct real estate were not identified among the classified collector routes. Forex coverage is limited to the currently listed fiat/stablecoin pairs and USDBRL TradFi perpetual; it does not cover the full global FX matrix. Geographic labels come only from explicit regional metadata or reviewed ETF definitions; US listing location is not used as a claim about global revenue exposure.

No noncrypto cap weights are calculated. Stock issuer capitalization, index constituent capitalization, ETF AUM, pre-IPO valuations, commodity inventory and currency turnover are economically different denominators. Obtain appropriate dated global underlying datasets before adding weights; the current policy retains the entire supported noncrypto set.

Sources: [Binance TradFi structure](https://www.binance.com/en/learn/binance-futures-tradfi-perpetual-contracts), [Binance energy contracts](https://www.binance.com/en/support/announcement/detail/e7a10ca855524655ab3dea73add40ef4), [Binance tokenized bStocks](https://www.binance.com/en-IA/how-to-buy/apple-tokenized-bstocks), [CoinGecko named catalog](https://api.coingecko.com/api/v3/coins/list). [QQQ issuer](https://www.invesco.com/us/en/financial-products/etfs/invesco-qqq-trust-series-1.html), [TMF issuer](https://www.direxion.com/product/daily-20-year-treasury-bull-bear-3x-etfs), [TBT issuer](https://www.proshares.com/our-etfs/leveraged-and-inverse/tbt), [UVXY issuer](https://www.proshares.com/our-etfs/leveraged-and-inverse/uvxy). Full listing evidence is in the saved Binance metadata snapshots.

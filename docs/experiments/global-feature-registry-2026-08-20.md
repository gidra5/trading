# Canonical global feature registry

Generated `2026-08-23T19:19:53.679565+00:00`.

## Result

The registry covers **261 assets** and contains **1,001,102 raw ledger coordinates**. Canonical identity removes **10,087 declared duplicate aliases**, leaving **991,015 unique candidate coordinates** represented by **5,629 compact templates**.

The exact raw arithmetic is:

$$
892,047+23,240+26,214+14,280+8,226+31,043+4,242+1,810=1,001,102.
$$

Those terms are, respectively, dense 1m indicators, 1s technical indicators, 1m spectral features, 1s spectral features, long endogenous features, the representative cross-asset catalog, the funding grid, and all other external inventories. They use different availability sets: for example, dense and 1m spectral features bind to 257 assets, while 1s technical and spectral features bind to 140. The 147 existing coordinates are already part of the 31,043-coordinate representative catalog, and the 3,471 dense variants already include their lag grid. Therefore `261 × (147 + 3,471 + ...)` is not a valid count.

After canonical deduplication, the 991,015 coordinates split into **989,853 asset-specific** and **1,162 general** coordinates. Of these, **990,266** are both point-in-time safe and supported by the robust 30-day history used for production search; 222 short live-only coordinates and 527 non-point-in-time/revised coordinates remain documented but excluded from that optimization.

The independent scope and candidate-construction classifications intersect as follows. Templates are feature definitions; coordinates are their concrete subject bindings:

| Scope | Basic candidate templates | Basic expanded coordinates | Derived candidate templates | Derived expanded coordinates |
|---|---:|---:|---:|---:|
| Asset-specific | 10 | 245 | 4,457 | 989,608 |
| General | 66 | 66 | 1,096 | 1,096 |

These are candidate counts, not the complete source-field inventory below. For example, `funding-rate` is one asset-specific base field definition with 236 asset bindings; it is therefore one source feature type and 236 expanded coordinates, not 236 feature types.

A coordinate identity is `(subject kind, subject, venue/instrument, cadence, formula + parameters)`. An asset is bound only when the required source is actually available; no pre-listing or permanently missing columns are invented.

## Base/source feature inventory by origin

Source fields are counted before derived candidate expansion. A field definition is counted once per source schema and cadence; the asset-binding count shows how many concrete asset fields it produces.

| Origin category | Field set | Level | Cadence | Field definitions | Assets | Expanded asset-field bindings |
|---|---|---|---|---:|---:|---:|
| spot-stats | completed-spot-kline-1s | base | 1s | 9 | 140 | 1,260 |
| spot-stats | completed-spot-kline-1m | base | 1m | 9 | 140 | 1,260 |
| futures-stats | completed-usdm-kline-1m | base | 1m | 9 | 233 | 2,097 |
| futures-stats | published-usdm-positioning | base | 5m | 6 | 196 | 1,176 |
| futures-stats | settled-usdm-funding | base | funding-event | 1 | 236 | 236 |
| futures-stats | usdm-book-depth-bands | derived | 5m | 24 | 232 | 5,568 |
| options-stats | deribit-option-surface | derived | live-snapshot | 20 | 1 | 20 |

### Prediction-market base hierarchy

Kalshi `series` is the persistent feature identity. Events and contracts are dynamic instances. At `2026-08-01T12:00:00Z` (`2026-08-01T11:59:59Z` causal model origin), 10 recognized assets plus the separate `CRYPTO-OTHER` bucket had prediction markets.

Each raw trade has **5 base value fields**; each completed minute candle has **10 base value fields**. The stored trade-state and candle-state records expose **10** and **13** subfeatures respectively. The current axis normalizes those to **7 nullable fields per observed contract**.
Each open contract also has **25 contract-definition fields**. That is 25,200 potential asset-contract metadata field slots and 429,025 general-event slots at the snapshot; these identity/text/timing fields are nullable and are not all numeric model inputs.

| Asset/bucket | Persistent series (30d) | Events (30d) | Contracts (30d) | Open series | Open events | Open contracts | Series-field channels | Open contract-field slots | Observed 1m/1s contracts | Observed 1m/1s field-slot upper bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| BNB | 8 | 4,103 | 6,185 | 8 | 12 | 125 | 56 | 875 | 82/11 | 574/77 |
| BTC | 15 | 4,264 | 28,081 | 15 | 19 | 271 | 105 | 1,897 | 178/71 | 1,246/497 |
| CRYPTO-OTHER | 4 | 5 | 45 | 4 | 4 | 38 | 28 | 266 | 30/0 | 210/0 |
| DOGE | 9 | 3,948 | 5,026 | 9 | 13 | 63 | 63 | 441 | 36/0 | 252/0 |
| ETH | 8 | 4,257 | 23,155 | 8 | 12 | 153 | 56 | 1,071 | 75/0 | 525/0 |
| HYPE | 6 | 4,178 | 6,536 | 6 | 10 | 69 | 42 | 483 | 48/0 | 336/0 |
| NEAR | 4 | 2,841 | 2,850 | 4 | 4 | 13 | 28 | 91 | 13/0 | 91/0 |
| SHIB | 2 | 56 | 199 | 2 | 2 | 7 | 14 | 49 | 2/0 | 14/0 |
| SOL | 9 | 4,143 | 13,241 | 9 | 13 | 129 | 63 | 903 | 64/0 | 448/0 |
| XRP | 8 | 4,199 | 7,035 | 8 | 12 | 119 | 56 | 833 | 47/0 | 329/0 |
| ZEC | 5 | 2,842 | 2,870 | 5 | 5 | 21 | 35 | 147 | 16/0 | 112/0 |
| **Asset total** | **78** | **34,836** | **95,223** | **78** | **106** | **1,008** | **546** | **7,056** | **591/82** | **4,137/574** |

Global-event prediction markets use the same contract field schemas but are general features: 

| Scope | Persistent series (30d) | Events (30d) | Contracts (30d) | Open series | Open events | Open contracts | Series-field channels | Open contract-field slots | Observed 1m/1s contracts | Observed 1m/1s field-slot upper bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| General/global events | 1,832 | 9,031 | 89,835 | 1,482 | 3,036 | 17,161 | 10,374 | 120,127 | 12/0 | 84/0 |

Open contract-field slots are the dynamic contract fieldsets available in principle. Observed field-slot upper bounds count only contracts for which the sparse importer emitted a causal update at that origin, multiplied by seven; individual fields remain nullable. They are update counts, not feature counts.

## Availability

| Binding set | Assets |
|---|---:|
| preferred1m99 | 257 |
| spot1m99 | 140 |
| usdm1m99 | 233 |
| spot1s95 | 140 |
| usdmMetrics95 | 196 |
| usdmFunding | 236 |
| usdmBook95 | 232 |
| spotAndUsdm95 | 116 |

## Inventory reconciliation

| Inventory                  | Raw coordinates | Note                                                                                                                                 |
| -------------------------- | --------------: | ------------------------------------------------------------------------------------------------------------------------------------ |
| coinmetrics                |              37 |                                                                                                                                      |
| community-flows            |             145 |                                                                                                                                      |
| cross-market-public        |             104 |                                                                                                                                      |
| dense-minute-indicators    |         892,047 | All 3,471 EMA/RSI period, difference-horizon, and whole-signal-lag variants; the 13 requested lags are already included.             |
| deribit-option-surface     |              20 |                                                                                                                                      |
| dvol                       |              17 |                                                                                                                                      |
| fast-live                  |              76 |                                                                                                                                      |
| full-funding-grid          |           4,242 | The complete 18-template settled-funding grid, availability-filtered by each market's event count.                                   |
| gdelt-news                 |               8 |                                                                                                                                      |
| global-macro               |             309 |                                                                                                                                      |
| long-endogenous            |           8,226 | The 34-coordinate minute ledger, with 32 asset-derived templates expanded by asset and the two UTC calendar coordinates stored once. |
| mempool-proxy              |              36 |                                                                                                                                      |
| prediction-markets         |             931 | Causal Kalshi summaries over every discovered relevant traded asset and global-event market.                                         |
| representative-cross-asset |          31,043 | The 31,043-coordinate study catalog: 147 existing BTC/general entries plus source-supported replicated entries.                      |
| second-technical           |          23,240 | All 166 one-second RSI, EMA value/slope/acceleration, and MACD coordinates.                                                          |
| spectral-1m                |          26,214 | All 102 Fourier, fractional-Fourier, Haar, Morlet, and path-efficiency coordinates.                                                  |
| spectral-1s                |          14,280 | All 102 Fourier, fractional-Fourier, Haar, Morlet, and path-efficiency coordinates.                                                  |
| tardis-cross-venue         |             118 |                                                                                                                                      |
| vix                        |               9 |                                                                                                                                      |

## Search contract

Predictive quality is lexicographically first and the incumbent basis is always admissible. A path is discarded if it loses more than 0.001 bits per eligible target to the incumbent on any chronological fold. Among the survivors, paths inside the paired one-standard-error set of the best mean fold gain are treated as statistically equivalent; only then do broader availability, lower acquisition cost, and fewer features decide the result. The final transfer interval is confirmation only.

The registry does not claim that enumerating every subset of this candidate universe is tractable. The search implementation must publish the exact model class, admissible subset space, pruning bounds, and whether its optimum is proven or approximate.

## Important distinction

These are candidate coordinates. A production input contract is the much smaller selected subset, not the full registry.

import assert from "node:assert/strict";
import { MainClient, USDMClient } from "binance";
import { positive } from "./common.js";
import { BinanceFuturesApi } from "./binance-futures.js";
import { BinanceSpotApi } from "./binance-spot.js";
import { BinanceOptionsApi } from "./binance-options.js";
import {
  BinancePredictionApi,
  type BinancePredictionConfig,
} from "./binance-predictions.js";
import {
  BinanceUnsupportedOperationError,
  DEFAULT_FRICTION_DECAY_RATE,
  validateFrictionDecayRate,
} from "./binance-common.js";
import type {
  BinanceClientConfig,
  BinanceMarketConfig,
  BinanceMarketApi,
  BinanceTransport,
} from "./binance-types.js";

export type {
  BinanceClientConfig,
  BinanceMarketConfig,
  BinanceTransport,
  FuturesTransport,
  SpotTransport,
  SignedTransport,
} from "./binance-types.js";
export type {
  BinancePredictionConfig,
  PredictionOrderInput,
  PredictionQuote,
} from "./binance-predictions.js";
export {
  BinanceOrderSubmissionError,
  BinanceUnsupportedOperationError,
} from "./binance-common.js";
export { BinancePredictionApi } from "./binance-predictions.js";

export type BinanceSymbolConfig = BinanceMarketConfig & {
  readonly product: "spot" | "futures" | "options";
};
export type BinanceTradingApiConfig = BinanceClientConfig & BinanceSymbolConfig;

/** Owns one lazy SDK client per product; market adapters share credentials and transport. */
export class BinanceClient {
  private readonly config: BinanceClientConfig;
  private readonly transport: {
    -readonly [K in keyof BinanceTransport]: BinanceTransport[K];
  } = {};
  private readonly injected: BinanceTransport;
  private readonly markets = new Map<string, Promise<BinanceMarketApi>>();
  constructor(config: BinanceClientConfig, transport: BinanceTransport = {}) {
    assert(
      ["live", "demo", "testnet"].includes(config.environment),
      "Invalid Binance environment",
    );
    assert(
      config.apiKey.trim() && config.apiSecret.trim(),
      "Binance API credentials are required",
    );
    if (config.timeoutMs !== undefined) positive(config.timeoutMs, "timeoutMs");
    this.config = config;
    this.injected = transport;
  }
  private sdkOptions() {
    return {
      api_key: this.config.apiKey,
      api_secret: this.config.apiSecret,
      testnet: this.config.environment === "testnet",
      demoTrading: this.config.environment === "demo",
      disableTimeSync: true,
    };
  }
  private requireLive(product: string) {
    if (this.config.environment !== "live")
      throw new BinanceUnsupportedOperationError(
        product,
        `${this.config.environment} environment`,
      );
  }
  market(
    config: BinanceMarketConfig & { readonly product: "futures" },
  ): Promise<BinanceFuturesApi>;
  market(
    config: BinanceMarketConfig & { readonly product: "spot" },
  ): Promise<BinanceSpotApi>;
  market(
    config: BinanceMarketConfig & { readonly product: "options" },
  ): Promise<BinanceOptionsApi>;
  market(config: BinanceSymbolConfig): Promise<BinanceMarketApi>;
  market(config: BinancePredictionConfig): Promise<BinancePredictionApi>;
  market(
    config: BinanceSymbolConfig | BinancePredictionConfig,
  ): Promise<BinanceMarketApi | BinancePredictionApi> {
    const options = this.sdkOptions(),
      request = { timeout: this.config.timeoutMs ?? 10_000 };
    if (config.product === "predictions") {
      this.requireLive("predictions");
      this.transport.predictions ??=
        this.injected.predictions ?? new MainClient(options, request);
      return Promise.resolve(
        new BinancePredictionApi(config, this.transport.predictions),
      );
    }
    assert(/^[A-Z0-9_-]+$/.test(config.symbol), "Invalid Binance symbol");
    const decay = config.frictionDecayRate ?? DEFAULT_FRICTION_DECAY_RATE;
    validateFrictionDecayRate(decay);
    const key = `${config.product}:${config.symbol}:${decay}`;
    const existing = this.markets.get(key);
    if (existing) return existing;
    const api = (() => {
      switch (config.product) {
        case "futures":
          this.transport.futures ??=
            this.injected.futures ?? new USDMClient(options, request);
          return BinanceFuturesApi.create(config, this.transport.futures);
        case "spot":
          this.transport.spot ??=
            this.injected.spot ?? new MainClient(options, request);
          return BinanceSpotApi.create(config, this.transport.spot);
        case "options":
          this.requireLive("options");
          // The SDK's old voptions URL is for the retired VAPI, not EAPI.
          this.transport.options ??=
            this.injected.options ??
            new MainClient(
              { ...options, baseUrl: "https://eapi.binance.com" },
              request,
            );
          return BinanceOptionsApi.create(config, this.transport.options);
        default:
          assert.fail("Invalid Binance product");
      }
    })();
    this.markets.set(key, api);
    // Share in-flight creation; allow a later attempt after a failed metadata request.
    void api.catch(() => {
      this.markets.delete(key);
    });
    return api;
  }
}

/** Creates a single market adapter. Use BinanceClient.market to share a connection. */
export function getBinanceTradingApi(
  config: BinanceTradingApiConfig & { readonly product: "futures" },
  transport?: BinanceTransport,
): Promise<BinanceFuturesApi>;
export function getBinanceTradingApi(
  config: BinanceTradingApiConfig & { readonly product: "spot" },
  transport?: BinanceTransport,
): Promise<BinanceSpotApi>;
export function getBinanceTradingApi(
  config: BinanceTradingApiConfig & { readonly product: "options" },
  transport?: BinanceTransport,
): Promise<BinanceOptionsApi>;
export function getBinanceTradingApi(
  config: BinanceTradingApiConfig,
  transport?: BinanceTransport,
): Promise<BinanceMarketApi>;
export function getBinanceTradingApi(
  config: BinanceTradingApiConfig,
  transport?: BinanceTransport,
): Promise<BinanceMarketApi> {
  return new BinanceClient(config, transport).market(config);
}

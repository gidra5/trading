import type { WebClient } from "./webclient.js";
import type { BotStore } from "./db/bot-store.js";
import type {
  BotConfigureRequest,
  BotConfigureResponse,
  BotStartResponse,
  BotStopResponse,
  BotResetResponse,
  BotOpenPositionRequest,
  BotClosePositionRequest,
  BotPositionResponse,
} from "./schemas.js";

type Result<T> = T | Promise<T>;
export interface ApiContext {
  addWebClient(client: WebClient): () => void;
  bot: {
    start(symbol: string): Result<BotStartResponse>;
    stop(symbol: string): Result<BotStopResponse>;
    reset(symbol: string): Result<BotResetResponse>;
    configure(input: BotConfigureRequest): Result<BotConfigureResponse>;
    openPosition(input: BotOpenPositionRequest): Result<BotPositionResponse>;
    closePosition(input: BotClosePositionRequest): Result<BotPositionResponse>;
  };
}

export class BotControlsUnavailableError extends Error {
  readonly statusCode = 501;
  constructor() {
    super("Bot controls are not connected yet");
  }
}

const unavailable = (): never => {
  throw new BotControlsUnavailableError();
};
/** Each server owns its connections; no process-wide client registry. */
export function createContext(store: BotStore) {
  const clients = new Set<WebClient>();
  return {
    store,
    addWebClient(client: WebClient) {
      clients.add(client);
      return () => {
        clients.delete(client);
      };
    },
    bot: {
      start: unavailable,
      stop: unavailable,
      reset: unavailable,
      configure(input: BotConfigureRequest): BotConfigureResponse {
        const config = store.setConfig(input);
        return { network: config.network, status: "configured" };
      },
      openPosition: unavailable,
      closePosition: unavailable,
    },
  };
}

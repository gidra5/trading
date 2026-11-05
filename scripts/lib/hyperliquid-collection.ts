/** Public research observations; addresses identify accounts, not economic people or strategies. */
export interface HyperliquidCaptureSpec {
  readonly id: string;
  readonly url: string;
  readonly subscriptions: readonly Readonly<Record<string, unknown>>[];
  readonly heartbeatMessage: Readonly<{ method: "ping" }>;
}

const URL = "wss://api.hyperliquid.xyz/ws";
const MAX_USERS = 10;
const MAX_COINS = 100;

/**
 * The public book is a bounded L2 snapshot feed, not an order-level event tape.
 * User observations cover only explicitly watched addresses. Reconnects require
 * a coverage gap marker even when the provider returns an initial snapshot.
 * Specs: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/websocket/subscriptions
 * Limits: https://hyperliquid.gitbook.io/hyperliquid-docs/for-developers/api/rate-limits-and-user-limits
 */
export function createHyperliquidCaptureSpecs(
  coins: readonly string[],
  users: readonly string[] = [],
): readonly HyperliquidCaptureSpec[] {
  if (coins.some((coin) => !/^[A-Za-z0-9@:_./-]{1,80}$/.test(coin))) {
    throw new Error("Hyperliquid coin names must be exact API identifiers without whitespace");
  }
  if (users.some((user) => !/^0x[0-9a-fA-F]{40}$/.test(user))) {
    throw new Error("Hyperliquid users must be 0x-prefixed, 20-byte account addresses");
  }
  const uniqueCoins = [...new Set(coins)];
  const uniqueUsers = [...new Set(users.map((user) => user.toLowerCase()))];
  if (uniqueCoins.length > MAX_COINS) throw new Error(`At most ${MAX_COINS} Hyperliquid coins per collector`);
  if (uniqueUsers.length > MAX_USERS) {
    throw new Error("Hyperliquid permits 10 unique users across user-specific WebSocket subscriptions per IP");
  }
  const specs: HyperliquidCaptureSpec[] = [];
  if (uniqueCoins.length) {
    specs.push({
      id: "hyperliquid-public",
      url: URL,
      heartbeatMessage: { method: "ping" },
      subscriptions: uniqueCoins.flatMap((coin) => ["trades", "l2Book", "bbo"].map((type) => ({
        method: "subscribe", subscription: { type, coin },
      }))),
    });
  }
  if (uniqueUsers.length) {
    specs.push({
      id: "hyperliquid-participants",
      url: URL,
      heartbeatMessage: { method: "ping" },
      subscriptions: uniqueUsers.flatMap((user) => [
        { type: "orderUpdates", user },
        { type: "userFills", user, aggregateByTime: false },
        { type: "openOrders", user },
        { type: "clearinghouseState", user },
        { type: "spotState", user, isPortfolioMargin: false },
        { type: "userNonFundingLedgerUpdates", user },
      ].map((subscription) => ({ method: "subscribe", subscription }))),
    });
  }
  return specs;
}

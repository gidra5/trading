import type {
  TradingHistoryRequest, TradingOrderSnapshot, TradingQuantityRules,
} from "@trading/bot-algo/trading-api";

export function positive(value: number, name: string): void {
  if (!Number.isFinite(value) || value <= 0) throw new Error(`${name} must be positive and finite`);
}

export function validateHistoryRequest(input: TradingHistoryRequest): void {
  if (!Number.isSafeInteger(input.intervalMs) || input.intervalMs <= 0 ||
      !Number.isSafeInteger(input.count) || input.count < 0) {
    throw new Error("History requires a positive integer intervalMs and nonnegative integer count");
  }
}

export function validateOrder(order: TradingOrderSnapshot): void {
  if (order.side !== "buy" && order.side !== "sell") throw new Error("Invalid order side");
  positive(order.size, "size");
  if (order.price !== null) positive(order.price, "price");
  if (order.stopPrice !== null) positive(order.stopPrice, "stopPrice");
}

export function withinRules(value: number, rules: TradingQuantityRules): boolean {
  if (rules.min !== null && value < rules.min - 1e-12) return false;
  if (rules.max !== null && value > rules.max + 1e-12) return false;
  if (rules.step !== null) {
    const units = value / rules.step;
    if (Math.abs(units - Math.round(units)) > 1e-8) return false;
  }
  return true;
}

export function floorToStep(value: number, step: number | null): number {
  if (step === null) return value;
  return Math.floor(value / step + 1e-10) * step;
}

/** A market quantity must satisfy both LOT_SIZE and MARKET_LOT_SIZE grids. */
export function combineSteps(a: number | null, b: number | null): number | null {
  if (a === null) return b;
  if (b === null) return a;
  const precision = (value: number) => {
    const [coefficient, exponent = "0"] = String(value).split("e");
    return Math.max(0, (coefficient.split(".")[1]?.length ?? 0) - Number(exponent));
  };
  const scale = 10 ** Math.max(precision(a), precision(b));
  const left = Math.round(a * scale);
  const right = Math.round(b * scale);
  if (!Number.isSafeInteger(left) || !Number.isSafeInteger(right)) throw new Error("Quantity step precision exceeds supported range");
  let x = BigInt(left);
  let y = BigInt(right);
  while (y !== 0n) [x, y] = [y, x % y];
  return Number(BigInt(left) / x * BigInt(right)) / scale;
}

export function validateLeverage(leverage: number): void {
  if (!Number.isInteger(leverage) || leverage < 1 || leverage > 125) {
    throw new Error("leverage must be an integer between 1 and 125");
  }
}

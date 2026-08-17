export type QuoteSecond = [
  bucketTime: number,
  observedAt: number,
  bidPrice: number,
  askPrice: number,
  bidAmount: number,
  askAmount: number,
];

export type LiquidationEvent = [
  observedAt: number,
  side: -1 | 0 | 1,
  price: number,
  amount: number,
  symbol: string,
];

export function headerIndexes(header: string) {
  return new Map(header.trim().split(",").map((name, index) => [name, index]));
}

export function parseQuoteRow(line: string, columns: Map<string, number>): QuoteSecond | undefined {
  const values = line.trim().split(",");
  const observedAt = microsToMillis(numberAt(values, columns, "local_timestamp"));
  const askAmount = numberAt(values, columns, "ask_amount");
  const askPrice = numberAt(values, columns, "ask_price");
  const bidPrice = numberAt(values, columns, "bid_price");
  const bidAmount = numberAt(values, columns, "bid_amount");
  if (![observedAt, askAmount, askPrice, bidPrice, bidAmount].every(Number.isFinite)) return undefined;
  if (askPrice <= 0 || bidPrice <= 0 || askPrice < bidPrice || askAmount < 0 || bidAmount < 0) return undefined;
  return [Math.floor(observedAt / 1_000) * 1_000, observedAt, bidPrice, askPrice, bidAmount, askAmount];
}

export function parseLiquidationRow(line: string, columns: Map<string, number>): LiquidationEvent | undefined {
  const values = line.trim().split(",");
  const observedAt = microsToMillis(numberAt(values, columns, "local_timestamp"));
  const price = numberAt(values, columns, "price");
  const amount = numberAt(values, columns, "amount");
  const symbol = textAt(values, columns, "symbol");
  const rawSide = textAt(values, columns, "side");
  const side = rawSide === "buy" ? 1 : rawSide === "sell" ? -1 : 0;
  if (![observedAt, price, amount].every(Number.isFinite) || !symbol) return undefined;
  if (price <= 0 || amount <= 0) return undefined;
  return [observedAt, side, price, amount, symbol];
}

export function retainLastQuotePerSecond(rows: QuoteSecond[]): QuoteSecond[] {
  if (rows.length < 2) return rows;
  const output: QuoteSecond[] = [];
  let current = rows[0]!;
  for (let index = 1; index < rows.length; index += 1) {
    const row = rows[index]!;
    if (row[0] !== current[0]) {
      output.push(current);
      current = row;
    } else if (row[1] >= current[1]) current = row;
  }
  output.push(current);
  return output;
}

function microsToMillis(value: number) {
  return value / 1_000;
}

function numberAt(values: string[], columns: Map<string, number>, name: string) {
  const index = columns.get(name);
  return index === undefined ? Number.NaN : Number(values[index]);
}

function textAt(values: string[], columns: Map<string, number>, name: string) {
  const index = columns.get(name);
  return index === undefined ? "" : values[index] ?? "";
}

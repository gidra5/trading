import assert from "node:assert/strict";
import test from "node:test";
import {
  headerIndexes,
  parseLiquidationRow,
  parseQuoteRow,
  retainLastQuotePerSecond,
} from "./lib/tardis-monthly-samples.ts";

test("quotes are parsed using collector receive time and retain the last observation per second", () => {
  const columns = headerIndexes("exchange,symbol,timestamp,local_timestamp,ask_amount,ask_price,bid_price,bid_amount");
  const first = parseQuoteRow("binance,BTCUSDT,1000000,1250000,2,101,100,3", columns)!;
  const second = parseQuoteRow("binance,BTCUSDT,1000001,1750000,4,102,101,5", columns)!;
  assert.deepEqual(retainLastQuotePerSecond([first, second]), [[1_000, 1_750, 101, 102, 5, 4]]);
});

test("liquidation side distinguishes short and long liquidations", () => {
  const columns = headerIndexes("exchange,symbol,timestamp,local_timestamp,id,side,price,amount");
  assert.deepEqual(
    parseLiquidationRow("binance-futures,BTCUSDT,1632009737493000,1632009737505152,,sell,48283.81,0.01", columns),
    [1_632_009_737_505.152, -1, 48_283.81, 0.01, "BTCUSDT"],
  );
});

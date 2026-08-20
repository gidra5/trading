import assert from "node:assert/strict";
import test from "node:test";
import { describeLiveFeature } from "./export-live-component-feature-basis-dataset.ts";

test("live component basis feature descriptions preserve exact lookbacks", () => {
  assert.deepEqual(describeLiveFeature("deribit_perpetual_trade_count_15s"), {
    lookback: "15s",
    construction: "log1p event count over the trailing 15s",
  });
  assert.equal(describeLiveFeature("binance_spot_l1_imbalance_mean_60s").lookback, "60s");
  assert.equal(describeLiveFeature("binance_depth_churn_log_quote").lookback, "latest completed 1s");
});

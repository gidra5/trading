import assert from "node:assert/strict";
import test from "node:test";
import { createOracleCrraBot, StochasticOracleController } from "../src/bot.js";
import { rational } from "@trading/bot-algo/oracle-crra";
const { ONE, ZERO, rat } = rational;

test("controller requests causal forecasts, retries failed inference, and never duplicates execution", async () => {
  const bot = createOracleCrraBot({ initial: { quote: rat(100), asset: ZERO, price: rat(100),
    previousPrice: rat(100), time: 0, orders: [] }, funding: () => ONE,
    rules: { makerFee: rat("0.0002"), takerFee: rat("0.0005"), reserveLeverage: rat(2),
      entryLower: rat(-2), entryUpper: rat(2), maintenanceLower: rat(-3), maintenanceUpper: rat(3),
      quoteMinimum: rat(1), quoteMaximum: rat(100), priceStep: rat("0.01"), maxOrders: 0 },
    menu: { exposureTargets: [ZERO], limitNotionalFractions: [], limitOffsetsInTicks: [], maximumNewOrders: 0 } });
  let requests = 0, closed = false;
  assert.equal(bot.config.riskAversion, 0);
  const controller = new StochasticOracleController(bot, {
    async predict(input) {
      requests++;
      assert.equal(input.observedPrice, 100);
      if (requests === 1) throw new Error("Temporary inference error");
      return { children: [{ probability: 1, price: 100 }] };
    }, async close() { closed = true; },
  }, 1000);
  const tick = { timestamp: 1000, price: rat(100), features: [] };
  await assert.rejects(controller.onTick(tick), /Temporary/);
  await controller.onTick(tick);
  await assert.rejects(controller.onTick(tick), /exactly one/);
  await controller.onTick({ ...tick, timestamp: 2000 });
  assert.equal(bot.state.time, 1);
  const terminal = await controller.finish(3000, rat(100));
  assert(rational.eq(terminal.wealth, rat(100)));
  assert.equal(closed, true);
});

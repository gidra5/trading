import assert from "node:assert/strict";
import test from "node:test";
import {
  MacdStrategy,
  VolumeImbalanceStrategy,
  createPeakValleyStrategyConfig,
  type PeakValleyStrategyConfig,
  type PositionSide,
  type StrategyDiagnostics,
  type StrategySnapshot,
  type TradingCandle,
  type TradingStrategy,
  type TradingStrategyEntrySignal,
  type TradingStrategyExitSignal,
} from "../src/index.js";

const smallMacdConfig = {
  signalIntervalMs: 60_000,
  fastPeriod: 2,
  slowPeriod: 4,
  signalPeriod: 2,
  rsiPeriod: 2,
  rsiOversold: 35,
  rsiOverbought: 65,
  rsiConfirmationWindowPeriods: 10,
};

test("MACD crossover entries require recent RSI confirmation and exits do not", async () => {
  const prices = [120, 115, 110, 105, 100, 95];
  const history = prices.map((price, index) => candle(index, price));
  const strategy = new MacdStrategy({
    config: createPeakValleyStrategyConfig({ sampleIntervalMs: 60_000 }),
    getHistory: async () => history,
  }, smallMacdConfig);
  await strategy.warmup();

  let enteredLong = false;
  for (let index = 6; index <= 12; index += 1) {
    await strategy.onTick(tick(candle(index, 95 + (index - 5) * 5)));
    if ((await strategy.entrySignal())?.side === "long") {
      enteredLong = true;
      break;
    }
  }
  assert.equal(enteredLong, true);
  assert.equal(strategy.targetExposureSignal, undefined);

  let exitedLong = false;
  for (let index = 13; index <= 30; index += 1) {
    await strategy.onTick(tick(candle(index, 130 - (index - 12) * 3)));
    const entry = await strategy.entrySignal();
    const exit = await strategy.exitSignal();
    if (exit?.side === "long") {
      assert.ok(entry === null || entry.side === "short");
      exitedLong = true;
      break;
    }
  }
  assert.equal(exitedLong, true);
});

test("MACD rejects a crossover that has no recent RSI extreme", async () => {
  const history = Array.from({ length: 6 }, (_, index) => candle(index, 100));
  const strategy = new MacdStrategy({
    config: createPeakValleyStrategyConfig({ sampleIntervalMs: 60_000 }),
    getHistory: async () => history,
  }, smallMacdConfig);
  await strategy.warmup();
  for (let index = 6; index <= 12; index += 1) {
    await strategy.onTick(tick(candle(index, 100 + index - 5)));
    assert.equal(await strategy.entrySignal(), null);
  }
  assert.equal(strategy.getDiagnostics().gates.find((gate) => gate.code === "rsi.long-confirmed")?.passed, false);
});

test("MACD evaluates only completed configured signal bars", async () => {
  const strategy = new MacdStrategy({
    config: createPeakValleyStrategyConfig({ sampleIntervalMs: 60_000 }),
    getHistory: async () => [],
  }, {
    signalIntervalMs: 5 * 60_000,
    fastPeriod: 1,
    slowPeriod: 2,
    signalPeriod: 1,
    rsiPeriod: 2,
    rsiOversold: 30,
    rsiOverbought: 70,
    rsiConfirmationWindowPeriods: 1,
  });
  await strategy.warmup();
  for (let index = 0; index < 4; index += 1) {
    await strategy.onTick(tick(candle(index, 100 + index)));
  }
  assert.equal(strategy.getDiagnostics().indicators["macd.signalIntervalMs"], 300_000);
  assert.equal(strategy.getDiagnostics().gates[0]?.value, 0);
  await strategy.onTick(tick(candle(4, 105)));
  assert.equal(strategy.getDiagnostics().gates[0]?.value, 1);
});

test("aggressor volume gates directional entries but never exits", async () => {
  const base = new StubDirectionalStrategy();
  const strategy = new VolumeImbalanceStrategy({
    config: createPeakValleyStrategyConfig({ sampleIntervalMs: 60_000 }),
    getHistory: async () => [flowCandle(0, 80, 20), flowCandle(1, 70, 30)],
  }, { lookbackPeriods: 1, entryThreshold: 0.3 }, base);
  await strategy.warmup();

  base.entry = signalEntry("long");
  await strategy.onTick(tick(flowCandle(2, 90, 10)));
  assert.equal((await strategy.entrySignal())?.side, "long");

  base.entry = signalEntry("short");
  await strategy.onTick(tick(flowCandle(3, 10, 90)));
  assert.equal((await strategy.entrySignal())?.side, "short");

  base.entry = signalEntry("long");
  assert.equal(await strategy.entrySignal(), null, "opposing flow blocks expansion");
  assert.ok(strategy.getDiagnostics().blockers.includes("aggressor-volume-blocked-long"));

  base.exit = signalExit("long");
  assert.equal((await strategy.exitSignal())?.side, "long", "flow never blocks de-risking");
});

test("volume imbalance refuses to infer aggressor direction from OHLC", async () => {
  const base = new StubDirectionalStrategy();
  base.entry = signalEntry("long");
  const strategy = new VolumeImbalanceStrategy({
    config: createPeakValleyStrategyConfig({ sampleIntervalMs: 60_000 }),
    getHistory: async () => [],
  }, { lookbackPeriods: 1, entryThreshold: 0.1 }, base);
  await strategy.warmup();
  await strategy.onTick(tick({
    ...candle(0, 109),
    open: 100,
    high: 110,
    low: 90,
    volume: 1_000,
  }));
  assert.equal(await strategy.entrySignal(), null);
  assert.ok(strategy.getDiagnostics().blockers.includes("aggressor-volume-warmup"));
  assert.ok(strategy.getDiagnostics().blockers.includes("aggressor-volume-blocked-long"));
});

class StubDirectionalStrategy implements TradingStrategy<
  PeakValleyStrategyConfig,
  StrategySnapshot,
  StrategyDiagnostics
> {
  entry: TradingStrategyEntrySignal | null = null;
  exit: TradingStrategyExitSignal | null = null;

  async warmup() {}
  async onTick() {}
  async entrySignal() { return this.entry; }
  async exitSignal() { return this.exit; }
  staticConfidence() { return 1; }
  async snapshot() { return { version: 1 }; }
  async restore() {}
  async updateConfig() {}
  getDiagnostics(): StrategyDiagnostics {
    return { indicators: {}, gates: [], blockers: [], lastSignal: null };
  }
}

function signalEntry(side: PositionSide): TradingStrategyEntrySignal {
  return { side, size: 1, leverage: 5, price: null, confidence: null };
}

function signalExit(side: PositionSide): TradingStrategyExitSignal {
  return { side, size: 1, price: null, confidence: null };
}

function candle(index: number, close: number): TradingCandle {
  return {
    openTime: index * 60_000,
    closeTime: (index + 1) * 60_000 - 1,
    open: close,
    high: close,
    low: close,
    close,
    volume: 100,
  };
}

function flowCandle(index: number, buy: number, sell: number): TradingCandle {
  return {
    ...candle(index, 100),
    volume: buy + sell,
    aggressiveBuyVolume: buy,
    aggressiveSellVolume: sell,
  };
}

function tick(value: TradingCandle) {
  return {
    timestamp: value.closeTime,
    price: value.close,
    quantity: value.volume,
    candle: value,
  };
}

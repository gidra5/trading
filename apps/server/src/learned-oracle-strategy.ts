import {
  PeakValleyStrategy,
  type PeakValleyStrategyConfig,
  type StrategyOptions,
  type TradingCandle,
  type TradingStrategyTargetExposureContext,
  type TradingStrategyTargetExposureSignal,
  type TradingTick,
} from "@trading/bot-algo";
import {
  HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
  HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
  HINDSIGHT_ORACLE_TEMPERATURE,
  confidenceConditionedHindsightOracleExposure,
  hindsightOracleTargetDecision,
} from "./bot-backtest.js";
import { JointPriceOracleRuntime } from "./joint-price-oracle-runtime.js";

const MINIMUM_CONFIDENCE = 0.05;
const CONTEXT_LENGTH = 3_600;

export class LearnedOracleStrategy extends PeakValleyStrategy {
  private tick: TradingTick | null = null;

  constructor(
    options: StrategyOptions<PeakValleyStrategyConfig>,
    private readonly runtime: JointPriceOracleRuntime,
    private readonly history: (count: number) => Promise<TradingCandle[]>,
    private readonly friction: number,
  ) {
    super(options);
  }

  override async onTick(tick: TradingTick): Promise<void> {
    this.tick = tick;
    await super.onTick(tick);
  }

  async targetExposureSignal(
    context: TradingStrategyTargetExposureContext,
  ): Promise<TradingStrategyTargetExposureSignal | null> {
    if (!this.tick || (context.timestamp + 1) % 60_000 !== 0) return null;
    const candles = await this.history(CONTEXT_LENGTH);
    if (candles.length < CONTEXT_LENGTH
      || candles.at(-1)?.closeTime !== context.timestamp) return null;
    const distribution = await this.runtime.predictLatest(candles);
    if (!distribution) return null;
    const decision = hindsightOracleTargetDecision(
      distribution,
      context.currentExposure,
      this.friction,
      HINDSIGHT_ORACLE_TEMPERATURE,
    );
    if (decision.confidence < MINIMUM_CONFIDENCE) return null;
    const targetExposure = confidenceConditionedHindsightOracleExposure(
      decision.targetExposure,
      decision.confidence,
      context.maxLeverage,
      HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
      HINDSIGHT_ORACLE_CONFIDENCE_LEVERAGE_FLOOR,
    );
    const gridStep = Math.abs(distribution.grid[1]! - distribution.grid[0]!);
    if (Math.abs(targetExposure - context.currentExposure) < gridStep / 2) return null;
    return {
      targetExposure,
      price: this.tick.price,
      confidence: decision.confidence,
    };
  }
}

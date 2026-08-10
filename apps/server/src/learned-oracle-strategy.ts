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
  HINDSIGHT_ORACLE_TEMPERATURE,
  ORACLE_DEFAULT_STATIC_CONFIDENCE_SCALE,
  confidenceScaledHindsightOracleExposure,
  hindsightOracleTargetDecision,
  oracleExecutionExposureScale,
} from "./bot-backtest.js";
import {
  JOINT_PRICE_ORACLE_CONTEXT_LENGTH,
  JointPriceOracleRuntime,
  isJointPriceOracleDecisionTime,
} from "./joint-price-oracle-runtime.js";

export class LearnedOracleStrategy extends PeakValleyStrategy {
  private tick: TradingTick | null = null;

  constructor(
    options: StrategyOptions<PeakValleyStrategyConfig>,
    private readonly runtime: JointPriceOracleRuntime,
    private readonly history: (count: number) => Promise<TradingCandle[]>,
    private readonly friction: number,
    private readonly staticConfidenceScale = ORACLE_DEFAULT_STATIC_CONFIDENCE_SCALE,
  ) {
    super(options);
  }

  override async onTick(tick: TradingTick): Promise<void> {
    this.tick = tick;
    await super.onTick(tick);
  }

  override staticConfidence(): number {
    return this.staticConfidenceScale;
  }

  async targetExposureSignal(
    context: TradingStrategyTargetExposureContext,
  ): Promise<TradingStrategyTargetExposureSignal | null> {
    if (!this.tick || !isJointPriceOracleDecisionTime(context.timestamp)) return null;
    const candles = await this.history(JOINT_PRICE_ORACLE_CONTEXT_LENGTH);
    if (candles.length < JOINT_PRICE_ORACLE_CONTEXT_LENGTH
      || candles.at(-1)?.closeTime !== context.timestamp) return null;
    const distribution = await this.runtime.predictLatest(candles);
    if (!distribution) return null;
    const executionScale = oracleExecutionExposureScale(
      distribution,
      context.maxLeverage,
    );
    const decision = hindsightOracleTargetDecision(
      distribution,
      context.currentExposure / executionScale,
      this.friction,
      HINDSIGHT_ORACLE_TEMPERATURE,
    );
    const targetExposure = confidenceScaledHindsightOracleExposure(
      decision.targetExposure * executionScale,
      decision.confidence,
      HINDSIGHT_ORACLE_CONFIDENCE_EXPOSURE_POWER,
    );
    const gridStep = Math.abs(distribution.grid[1]! - distribution.grid[0]!);
    if (
      Math.abs(targetExposure - context.currentExposure)
      < gridStep * executionScale / 2
    ) return null;
    return {
      targetExposure,
      price: this.tick.price,
      confidence: decision.confidence,
    };
  }
}

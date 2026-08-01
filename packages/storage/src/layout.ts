import path from "node:path";

export class TradingStorageLayout {
  readonly dataRoot: string;

  constructor(dataRoot: string) {
    this.dataRoot = path.resolve(dataRoot);
  }

  get marketRoot(): string {
    return path.join(this.dataRoot, "market");
  }

  get marketStore(): string {
    return path.join(this.marketRoot, "immutable");
  }

  get marketMutable(): string {
    return path.join(this.marketRoot, "mutable");
  }

  get marketTmp(): string {
    return path.join(this.marketMutable, "tmp");
  }

  get trainingRoot(): string {
    return path.join(this.dataRoot, "training");
  }

  get trainingStore(): string {
    return path.join(this.trainingRoot, "immutable");
  }

  get trainingDatasets(): string {
    return path.join(this.trainingRoot, "datasets");
  }

  get trainingRuns(): string {
    return path.join(this.trainingRoot, "runs");
  }

  get trainingCache(): string {
    return path.join(this.trainingRoot, "cache");
  }

  get trainingAnalysis(): string {
    return path.join(this.trainingRoot, "analysis");
  }

  get runtimeRoot(): string {
    return path.join(this.dataRoot, "runtime");
  }

  get runtimeState(): string {
    return path.join(this.runtimeRoot, "state");
  }

  get runtimeBacktests(): string {
    return path.join(this.runtimeRoot, "backtests");
  }

  get runtimeLogs(): string {
    return path.join(this.runtimeRoot, "logs");
  }

  candleReferences(market: string, symbol: string, interval: string): string {
    return path.join(
      this.marketStore,
      "refs",
      "candles",
      identifier(market),
      identifier(symbol),
      identifier(interval),
    );
  }

  dataset(id: string): string {
    return path.join(this.trainingDatasets, identifier(id));
  }

  run(id: string): string {
    return path.join(this.trainingRuns, identifier(id));
  }
}

function identifier(value: string): string {
  if (!/^[a-zA-Z0-9][a-zA-Z0-9._=-]*$/.test(value)) {
    throw new Error(`Unsafe storage identifier: ${value}.`);
  }
  return value;
}

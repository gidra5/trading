export interface LogReturnHistogramSpec {
  binWidthBps: number;
  lowerBps: number;
  upperBps: number;
  binCount: number;
}

export interface LogReturnHistogram {
  observations: number;
  binWidthBps: number;
  lowerBps: number;
  upperBps: number;
  underflowProbability: number;
  overflowProbability: number;
  bins: Array<{
    centerBps: number;
    lowerBps: number;
    upperBps: number;
    probability: number;
  }>;
}

export function createLogReturnHistogramSpec(
  fullHistorySigmaBps: number,
  binWidthSigma = 0.1,
  maximumCenterSigma = 8,
): LogReturnHistogramSpec {
  if (!(fullHistorySigmaBps > 0)
    || !(binWidthSigma > 0)
    || !(maximumCenterSigma > 0)) {
    throw new Error("Histogram sigma, bin width, and range must be positive.");
  }
  const binCount = Math.round(2 * maximumCenterSigma / binWidthSigma) + 1;
  if (binCount < 3 || binCount > 20_001 || binCount % 2 !== 1) {
    throw new Error("Histogram must have a bounded odd number of bins.");
  }
  const binWidthBps = fullHistorySigmaBps * binWidthSigma;
  const lowerBps = -(maximumCenterSigma + binWidthSigma / 2) * fullHistorySigmaBps;
  return {
    binWidthBps,
    lowerBps,
    upperBps: lowerBps + binCount * binWidthBps,
    binCount,
  };
}

export class LogReturnHistogramCounter {
  private readonly counts: Uint32Array;
  private observations = 0;
  private underflow = 0;
  private overflow = 0;

  constructor(readonly spec: LogReturnHistogramSpec) {
    this.counts = new Uint32Array(spec.binCount);
  }

  addLogReturn(value: number): void {
    if (!Number.isFinite(value)) throw new Error("Cannot histogram a non-finite return.");
    this.addBps(value * 10_000);
  }

  addBps(valueBps: number): void {
    this.observations += 1;
    if (valueBps < this.spec.lowerBps) {
      this.underflow += 1;
      return;
    }
    if (valueBps >= this.spec.upperBps) {
      this.overflow += 1;
      return;
    }
    const index = Math.floor(
      (valueBps - this.spec.lowerBps) / this.spec.binWidthBps,
    );
    this.counts[index]! += 1;
  }

  finish(): LogReturnHistogram {
    if (this.observations < 1) throw new Error("Cannot finish an empty histogram.");
    return {
      observations: this.observations,
      binWidthBps: this.spec.binWidthBps,
      lowerBps: this.spec.lowerBps,
      upperBps: this.spec.upperBps,
      underflowProbability: this.underflow / this.observations,
      overflowProbability: this.overflow / this.observations,
      bins: Array.from(this.counts, (count, index) => {
        const lowerBps = this.spec.lowerBps + index * this.spec.binWidthBps;
        return {
          centerBps: lowerBps + this.spec.binWidthBps / 2,
          lowerBps,
          upperBps: lowerBps + this.spec.binWidthBps,
          probability: count / this.observations,
        };
      }),
    };
  }
}

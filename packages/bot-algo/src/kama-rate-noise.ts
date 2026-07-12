export interface KamaRateNoiseSnapshot {
  previous: number | null;
  value: number;
}

export class KamaRateNoise {
  private previous: number | null = null;
  private value = 0;
  private readonly alpha: number;

  constructor(lookbackSamples: number) {
    const period = Math.max(1, Math.round(lookbackSamples));
    this.alpha = 2 / (period + 1);
  }

  update(rate: number): number {
    if (!Number.isFinite(rate)) return this.value;
    if (this.previous !== null) {
      const change = Math.abs(rate - this.previous);
      this.value += this.alpha * (change - this.value);
    }
    this.previous = rate;
    return this.value;
  }

  indicator(): number {
    return this.value;
  }

  snapshot(): KamaRateNoiseSnapshot {
    return { previous: this.previous, value: this.value };
  }

  restore(snapshot: KamaRateNoiseSnapshot | null | undefined): void {
    this.previous = Number.isFinite(snapshot?.previous) ? snapshot!.previous : null;
    this.value = Number.isFinite(snapshot?.value) && snapshot!.value >= 0 ? snapshot!.value : 0;
  }
}

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface TripleReport {
  source: { analysisStartTime: string; analysisEndTime: string };
  histogram: { standardizedEdges: Array<number | null> };
  scales: Array<{
    id: string;
    fullHistory: {
      lag1AbsoluteReturnCorrelation: number;
      lag2AbsoluteReturnCorrelation: number;
      allZeroFraction: number;
      generalizedGaussianApproximation: { power: number };
      shape: {
        projectionHistograms: {
          firstSecond: number[];
          secondThird: number[];
          firstThird: number[];
        };
      };
    };
    selectedFullHistoryModel: { family: string };
  }>;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = parseArgs(process.argv.slice(2));
const inputPath = path.resolve(
  repoRoot,
  args.get("input") ?? "data/benchmarks/consecutive-log-return-triples.json",
);
const outputPath = path.resolve(
  repoRoot,
  args.get("output") ?? "data/benchmarks/consecutive-log-return-triples.svg",
);
const report = JSON.parse(fs.readFileSync(inputPath, "utf8")) as TripleReport;
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, render(report), "utf8");
console.log(outputPath);

function render(report: TripleReport): string {
  const width = 1_590;
  const height = 720;
  const panelWidth = 520;
  const panelHeight = 300;
  const plotSize = 132;
  const top = 115;
  const left = 42;
  const bins = report.histogram.standardizedEdges.length - 1;
  const projections = [
    ["firstSecond", "rₜ", "rₜ₊₁"],
    ["secondThird", "rₜ₊₁", "rₜ₊₂"],
    ["firstThird", "rₜ", "rₜ₊₂"],
  ] as const;
  const panels = report.scales.map((scale, scaleIndex) => {
    const panelX = left + scaleIndex % 3 * panelWidth;
    const panelY = top + Math.floor(scaleIndex / 3) * panelHeight;
    const heatmaps = projections.map(([key, xLabel, yLabel], projectionIndex) => {
      const x = panelX + projectionIndex * 164;
      const y = panelY + 64;
      const histogram = scale.fullHistory.shape.projectionHistograms[key];
      const cells: string[] = [];
      for (let yIndex = 0; yIndex < bins; yIndex += 1) {
        for (let xIndex = 0; xIndex < bins; xIndex += 1) {
          const x0 = finiteEdge(report.histogram.standardizedEdges[xIndex]!, false);
          const x1 = finiteEdge(report.histogram.standardizedEdges[xIndex + 1]!, true);
          const y0 = finiteEdge(report.histogram.standardizedEdges[yIndex]!, false);
          const y1 = finiteEdge(report.histogram.standardizedEdges[yIndex + 1]!, true);
          const cellX = x + coordinate(x0, plotSize);
          const cellY = y + plotSize - coordinate(y1, plotSize);
          cells.push(`<rect x="${cellX.toFixed(2)}" y="${cellY.toFixed(2)}" width="${(coordinate(x1, plotSize) - coordinate(x0, plotSize) + 0.2).toFixed(2)}" height="${(coordinate(y1, plotSize) - coordinate(y0, plotSize) + 0.2).toFixed(2)}" fill="${heatColor(histogram[yIndex * bins + xIndex]!)}"/>`);
        }
      }
      const ticks = [-4, 0, 4].map((tick) => {
        const position = coordinate(tick, plotSize);
        return `<text x="${x + position}" y="${y + plotSize + 17}" class="tick" text-anchor="middle">${tick}</text>
          <text x="${x - 8}" y="${y + plotSize - position + 4}" class="tick" text-anchor="end">${tick}</text>`;
      }).join("");
      return `<g>
        <text x="${x + plotSize / 2}" y="${y - 13}" class="projection" text-anchor="middle">${xLabel} ↔ ${yLabel}</text>
        <rect x="${x}" y="${y}" width="${plotSize}" height="${plotSize}" fill="#08101d"/>
        ${cells.join("")}
        <path d="M ${x + coordinate(0, plotSize)} ${y} V ${y + plotSize} M ${x} ${y + plotSize - coordinate(0, plotSize)} H ${x + plotSize}" class="zero"/>
        <rect x="${x}" y="${y}" width="${plotSize}" height="${plotSize}" class="frame"/>
        ${ticks}
        <text x="${x + plotSize / 2}" y="${y + plotSize + 35}" class="axis" text-anchor="middle">${xLabel} / σ</text>
        <text x="${x - 30}" y="${y + plotSize / 2}" class="axis" text-anchor="middle" transform="rotate(-90 ${x - 30} ${y + plotSize / 2})">${yLabel} / σ</text>
      </g>`;
    }).join("");
    return `<g>
      <text x="${panelX}" y="${panelY}" class="panel-title">${scale.id}</text>
      <text x="${panelX + 56}" y="${panelY}" class="detail">|r| corr ${scale.fullHistory.lag1AbsoluteReturnCorrelation.toFixed(3)} → ${scale.fullHistory.lag2AbsoluteReturnCorrelation.toFixed(3)} · GGD p ${scale.fullHistory.generalizedGaussianApproximation.power.toFixed(3)}</text>
      <text x="${panelX}" y="${panelY + 25}" class="model">${familyLabel(scale.selectedFullHistoryModel.family)}</text>
      ${heatmaps}
    </g>`;
  }).join("\n");
  const legendX = 1_190;
  const legendY = 680;
  const legend = Array.from({ length: 121 }, (_, index) => {
    const probability = 10 ** (-6 + index / 120 * 5);
    return `<rect x="${legendX + 2 * index}" y="${legendY}" width="2.2" height="12" fill="${heatColor(probability)}"/>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    text { font-family: Inter, Segoe UI, sans-serif; fill: #dce7f5; }
    .title { font-size: 25px; font-weight: 700; }
    .subtitle, .detail { font-size: 13px; fill: #8fa5bd; }
    .panel-title { font-size: 21px; font-weight: 700; }
    .model { font-size: 12px; fill: #67d7c4; }
    .projection, .axis { font-size: 11px; fill: #a7bacd; }
    .tick { font-size: 9px; fill: #7e93aa; }
    .zero { stroke: #b6c7d8; stroke-opacity: 0.18; stroke-width: 0.7; }
    .frame { fill: none; stroke: #71869d; stroke-width: 0.8; }
  </style>
  <rect width="100%" height="100%" fill="#07101b"/>
  <text x="32" y="38" class="title">Three consecutive BTCUSDT log returns</text>
  <text x="32" y="63" class="subtitle">Continuous component, standardized within scale · all three pairwise projections · ${report.source.analysisStartTime.slice(0, 10)} to ${report.source.analysisEndTime.slice(0, 10)}</text>
  <text x="32" y="84" class="subtitle">Shared log₁₀ probability scale. The lag-2 view (right subplot in each group) retains magnitude dependence after directional correlation vanishes.</text>
  ${panels}
  <text x="1174" y="691" class="subtitle" text-anchor="end">10⁻⁶</text>
  ${legend}
  <text x="${legendX + 252}" y="691" class="subtitle">10⁻¹</text>
  <text x="32" y="700" class="subtitle">Adjacent projections are almost stationary copies; the first↔third projection isolates two-step persistence.</text>
</svg>\n`;
}

function finiteEdge(value: number | null, upper: boolean): number {
  if (value === null) return upper ? 4.5 : -4.5;
  if (value === Number.NEGATIVE_INFINITY) return -4.5;
  if (value === Number.POSITIVE_INFINITY) return 4.5;
  return upper ? Math.min(4.5, value) : Math.max(-4.5, value);
}

function coordinate(value: number, size: number): number {
  return (value + 4.5) / 9 * size;
}

function heatColor(probability: number): string {
  if (!(probability > 0)) return "#08101d";
  const value = Math.max(0, Math.min(1, (Math.log10(probability) + 6) / 5));
  const stops = [[8, 16, 29], [29, 55, 88], [25, 111, 128], [51, 176, 148], [235, 203, 92]];
  const position = value * (stops.length - 1);
  const lower = Math.min(stops.length - 2, Math.floor(position));
  const weight = position - lower;
  const rgb = stops[lower]!.map((channel, index) => Math.round(
    channel * (1 - weight) + stops[lower + 1]![index]! * weight,
  ));
  return `rgb(${rgb.join(",")})`;
}

function familyLabel(family: string): string {
  return ({
    "trivariate-gaussian": "Trivariate Gaussian",
    "trivariate-student-t": "Trivariate Student t",
    "trivariate-generalized-gaussian": "Trivariate generalized Gaussian",
    "trivariate-radial-lognormal": "Trivariate radial lognormal",
    "trivariate-generalized-t": "Trivariate generalized t",
  } as Record<string, string>)[family] ?? family;
}

function parseArgs(input: string[]): Map<string, string> {
  const output = new Map<string, string>();
  for (let index = 0; index < input.length; index += 2) {
    const key = input[index];
    const value = input[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key ?? "end"}.`);
    }
    output.set(key.slice(2), value);
  }
  return output;
}

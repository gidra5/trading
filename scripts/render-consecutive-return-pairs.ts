import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface PairReport {
  source: {
    analysisStartTime: string;
    analysisEndTime: string;
  };
  histogram: {
    standardizedEdges: Array<number | null>;
  };
  scales: Array<{
    id: string;
    fullHistory: {
      zeroZeroFraction: number;
      absoluteReturnCorrelation: number | null;
      generalizedGaussianApproximation: { power: number };
      shape: { continuousHistogram: number[] };
    };
    selectedFullHistoryModel: { family: string };
  }>;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = parseArgs(process.argv.slice(2));
const inputPath = path.resolve(
  repoRoot,
  args.get("input") ?? "data/benchmarks/consecutive-log-return-pairs.json",
);
const outputPath = path.resolve(
  repoRoot,
  args.get("output") ?? "data/benchmarks/consecutive-log-return-pairs.svg",
);
const report = JSON.parse(fs.readFileSync(inputPath, "utf8")) as PairReport;
fs.mkdirSync(path.dirname(outputPath), { recursive: true });
fs.writeFileSync(outputPath, render(report), "utf8");
console.log(outputPath);

function render(report: PairReport): string {
  const width = 1_160;
  const height = 970;
  const plotSize = 280;
  const panelWidth = 370;
  const panelHeight = 400;
  const originX = 62;
  const originY = 145;
  const bins = report.histogram.standardizedEdges.length - 1;
  const panels = report.scales.map((scale, index) => {
    const column = index % 3;
    const row = Math.floor(index / 3);
    const x = originX + column * panelWidth;
    const y = originY + row * panelHeight;
    const cells: string[] = [];
    for (let yIndex = 0; yIndex < bins; yIndex += 1) {
      for (let xIndex = 0; xIndex < bins; xIndex += 1) {
        const left = finiteEdge(report.histogram.standardizedEdges[xIndex]!, false);
        const right = finiteEdge(report.histogram.standardizedEdges[xIndex + 1]!, true);
        const lower = finiteEdge(report.histogram.standardizedEdges[yIndex]!, false);
        const upper = finiteEdge(report.histogram.standardizedEdges[yIndex + 1]!, true);
        const probability = scale.fullHistory.shape.continuousHistogram[yIndex * bins + xIndex]!;
        const cellX = x + plotCoordinate(left, plotSize);
        const cellY = y + plotSize - plotCoordinate(upper, plotSize);
        const cellWidth = plotCoordinate(right, plotSize) - plotCoordinate(left, plotSize);
        const cellHeight = plotCoordinate(upper, plotSize) - plotCoordinate(lower, plotSize);
        cells.push(`<rect x="${cellX.toFixed(2)}" y="${cellY.toFixed(2)}" width="${(cellWidth + 0.25).toFixed(2)}" height="${(cellHeight + 0.25).toFixed(2)}" fill="${heatColor(probability)}"/>`);
      }
    }
    const tickValues = [-4, -2, 0, 2, 4];
    const grid = tickValues.flatMap((tick) => {
      const position = plotCoordinate(tick, plotSize);
      return [
        `<path d="M ${x + position} ${y} V ${y + plotSize}" class="grid"/>`,
        `<path d="M ${x} ${y + plotSize - position} H ${x + plotSize}" class="grid"/>`,
        `<text x="${x + position}" y="${y + plotSize + 20}" class="tick" text-anchor="middle">${tick}</text>`,
        `<text x="${x - 12}" y="${y + plotSize - position + 4}" class="tick" text-anchor="end">${tick}</text>`,
      ];
    }).join("");
    const selected = familyLabel(scale.selectedFullHistoryModel.family);
    return `<g>
      <text x="${x}" y="${y - 46}" class="panel-title">${escapeXml(scale.id)}</text>
      <text x="${x + 54}" y="${y - 46}" class="panel-detail">|r| corr ${fixed(scale.fullHistory.absoluteReturnCorrelation, 3)} · GGD p ${fixed(scale.fullHistory.generalizedGaussianApproximation.power, 3)}</text>
      <text x="${x}" y="${y - 22}" class="panel-model">${escapeXml(selected)}</text>
      <rect x="${x}" y="${y}" width="${plotSize}" height="${plotSize}" fill="#08101d"/>
      ${cells.join("")}
      ${grid}
      <rect x="${x}" y="${y}" width="${plotSize}" height="${plotSize}" class="frame"/>
      <text x="${x + plotSize / 2}" y="${y + plotSize + 42}" class="axis" text-anchor="middle">rₜ / σ</text>
      <text x="${x - 44}" y="${y + plotSize / 2}" class="axis" text-anchor="middle" transform="rotate(-90 ${x - 44} ${y + plotSize / 2})">rₜ₊₁ / σ</text>
    </g>`;
  }).join("\n");
  const legendX = 800;
  const legendY = 922;
  const legend = Array.from({ length: 121 }, (_, index) => {
    const probability = 10 ** (-6 + index / 120 * 5);
    return `<rect x="${legendX + index * 2}" y="${legendY}" width="2.2" height="12" fill="${heatColor(probability)}"/>`;
  }).join("");
  return `<svg xmlns="http://www.w3.org/2000/svg" width="${width}" height="${height}" viewBox="0 0 ${width} ${height}">
  <style>
    text { font-family: Inter, Segoe UI, sans-serif; fill: #dce7f5; }
    .title { font-size: 25px; font-weight: 700; }
    .subtitle { font-size: 13px; fill: #8fa5bd; }
    .panel-title { font-size: 20px; font-weight: 700; }
    .panel-detail { font-size: 12px; fill: #9fb2c7; }
    .panel-model { font-size: 12px; fill: #67d7c4; }
    .axis { font-size: 12px; fill: #9fb2c7; }
    .tick { font-size: 10px; fill: #7e93aa; }
    .grid { stroke: #9fb2c7; stroke-opacity: 0.16; stroke-width: 0.6; }
    .frame { fill: none; stroke: #71869d; stroke-width: 0.8; }
  </style>
  <rect width="100%" height="100%" fill="#07101b"/>
  <text x="32" y="38" class="title">Consecutive BTCUSDT log-return pairs</text>
  <text x="32" y="63" class="subtitle">Continuous component (both returns nonzero), standardized within scale · ${escapeXml(report.source.analysisStartTime.slice(0, 10))} to ${escapeXml(report.source.analysisEndTime.slice(0, 10))}</text>
  <text x="32" y="84" class="subtitle">Color is log₁₀ bin probability on a shared scale. Exact-zero axes are excluded; the 1s (0,0) atom is ${(100 * report.scales[0]!.fullHistory.zeroZeroFraction).toFixed(2)}%.</text>
  ${panels}
  <text x="708" y="933" class="subtitle" text-anchor="end">10⁻⁶</text>
  ${legend}
  <text x="${legendX + 252}" y="933" class="subtitle">10⁻¹</text>
  <text x="32" y="950" class="subtitle">Elliptical contours and adjacent-tail concentration persist across scales; the center broadens toward daily aggregation.</text>
</svg>\n`;
}

function finiteEdge(value: number | null, upper: boolean): number {
  if (value === null) return upper ? 4.5 : -4.5;
  if (value === Number.NEGATIVE_INFINITY) return -4.5;
  if (value === Number.POSITIVE_INFINITY) return 4.5;
  return upper ? Math.min(4.5, value) : Math.max(-4.5, value);
}

function plotCoordinate(value: number, size: number): number {
  return (value + 4.5) / 9 * size;
}

function heatColor(probability: number): string {
  if (!(probability > 0)) return "#08101d";
  const value = Math.max(0, Math.min(1, (Math.log10(probability) + 6) / 5));
  const stops = [
    [8, 16, 29],
    [29, 55, 88],
    [25, 111, 128],
    [51, 176, 148],
    [235, 203, 92],
  ];
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
    "bivariate-gaussian": "Gaussian",
    "bivariate-student-t": "Student t",
    "elliptical-generalized-gaussian": "Elliptical generalized Gaussian",
    "elliptical-radial-lognormal": "Elliptical radial lognormal",
    "product-generalized-gaussian": "Product generalized Gaussian",
    "elliptical-generalized-t": "Elliptical generalized t",
    "product-generalized-t": "Product generalized t",
  } as Record<string, string>)[family] ?? family;
}

function fixed(value: number | null, digits: number): string {
  return value === null || !Number.isFinite(value) ? "n/a" : value.toFixed(digits);
}

function escapeXml(value: string): string {
  return value.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;");
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

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface GapWindow {
  id: string;
  label: string;
  startTime: string;
  endTime: string;
  zeroReturnProbability: number;
  gaps: number;
  meanSeconds: number;
  quantiles: { p50: number; p90: number; p95: number; p99: number; p999: number };
  maximumSeconds: number;
  longestGaps: Array<{ seconds: number; startTime: string; endTime: string }>;
  pmf: Array<[seconds: number, count: number, probability: number]>;
}

interface GapReport {
  version: number;
  symbol: string;
  commonEndTime: string;
  windows: GapWindow[];
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

try {
  main();
} catch (error: unknown) {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}

function main(): void {
  const values = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(
    repoRoot,
    values.get("input") ?? "data/benchmarks/zero-change-gaps.json",
  );
  const output = values.get("output");
  if (!output) throw new Error("--output is required.");
  const outputPath = path.resolve(output);
  const report = JSON.parse(fs.readFileSync(inputPath, "utf8")) as GapReport;
  if (report.version !== 1 || report.symbol !== "BTCUSDT" || report.windows.length !== 5) {
    throw new Error("Input is not the expected BTCUSDT zero-change gap report.");
  }
  const windows = [...report.windows]
    .sort((left, right) => windowRank(left.id) - windowRank(right.id))
    .map((window) => ({
      id: window.id,
      label: shortWindowLabel(window.id),
      gaps: window.gaps,
      zero: round(window.zeroReturnProbability),
      mean: round(window.meanSeconds),
      q: [window.quantiles.p50, window.quantiles.p90, window.quantiles.p95,
        window.quantiles.p99, window.quantiles.p999],
      max: window.maximumSeconds,
      longest: window.longestGaps.slice(0, 3).map((gap) => [
        gap.seconds,
        gap.startTime.slice(0, 19).replace("T", " ") + "Z",
        gap.endTime.slice(0, 19).replace("T", " ") + "Z",
      ]),
      p: window.pmf.map((entry) => [entry[0], entry[1], round(entry[2])]),
    }));
  const fullWindow = report.windows.find((window) => window.id === "full");
  if (!fullWindow) throw new Error("Gap report is missing the full-history window.");
  const ordinaryPmf = fullWindow.pmf.filter((entry) => entry[0] <= 120);
  const fit = {
    full: fitBetaGeometricMixture(ordinaryPmf),
  };
  const payload = JSON.stringify({ end: report.commonEndTime.slice(0, 10), windows, fit })
    .replaceAll("</script", "<\\/script");
  const fragment = renderFragment(payload);
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, fragment, "utf8");
  console.log(outputPath);
}

function fitBetaGeometricMixture(
  pmf: Array<[seconds: number, count: number, probability: number]>,
): {
  weight: number;
  alpha1: number;
  beta1: number;
  alpha2: number;
  beta2: number;
  fittedThrough: number;
} {
  const total = pmf.reduce((sum, entry) => sum + entry[1], 0);
  const objective = (parameters: number[]): number => {
    const [logitWeight, logAlpha1, logBeta1, logAlpha2, logBeta2] = parameters;
    if (Math.abs(logitWeight) > 30
      || parameters.slice(1).some((value) => Math.abs(value) > 20)) return 1e9;
    const weight = 1 / (1 + Math.exp(-logitWeight));
    const alpha1 = Math.exp(logAlpha1);
    const beta1 = Math.exp(logBeta1);
    const alpha2 = Math.exp(logAlpha2);
    const beta2 = Math.exp(logBeta2);
    let logLikelihood = 0;
    for (const [seconds, count] of pmf) {
      const componentOne = Math.log(weight)
        + betaGeometricLogPmf(seconds, alpha1, beta1);
      const componentTwo = Math.log1p(-weight)
        + betaGeometricLogPmf(seconds, alpha2, beta2);
      logLikelihood += count * logAddExp(componentOne, componentTwo);
    }
    return -logLikelihood / total;
  };
  const starts = [
    [0, Math.log(10), Math.log(7), Math.log(3), Math.log(2)],
    [0, Math.log(30), Math.log(20), Math.log(5), Math.log(4)],
    [1, Math.log(30), Math.log(20), Math.log(3), Math.log(3)],
  ];
  let best: { parameters: number[]; value: number } | undefined;
  for (const start of starts) {
    const result = minimizeNelderMead(objective, start, 0.4);
    if (!best || result.value < best.value) best = result;
  }
  if (!best) throw new Error("Could not fit beta-geometric mixture.");
  let [logitWeight, logAlpha1, logBeta1, logAlpha2, logBeta2] = best.parameters;
  let weight = 1 / (1 + Math.exp(-logitWeight));
  let alpha1 = Math.exp(logAlpha1);
  let beta1 = Math.exp(logBeta1);
  let alpha2 = Math.exp(logAlpha2);
  let beta2 = Math.exp(logBeta2);
  if (alpha1 < alpha2) {
    weight = 1 - weight;
    [alpha1, alpha2] = [alpha2, alpha1];
    [beta1, beta2] = [beta2, beta1];
  }
  return {
    weight: round(weight),
    alpha1: round(alpha1),
    beta1: round(beta1),
    alpha2: round(alpha2),
    beta2: round(beta2),
    fittedThrough: pmf.at(-1)?.[0] ?? 0,
  };
}

function minimizeNelderMead(
  objective: (parameters: number[]) => number,
  start: number[],
  step: number,
): { parameters: number[]; value: number } {
  const dimensions = start.length;
  let simplex = [start.slice()];
  for (let dimension = 0; dimension < dimensions; dimension += 1) {
    const point = start.slice();
    point[dimension] += step;
    simplex.push(point);
  }
  let values = simplex.map(objective);
  for (let iteration = 0; iteration < 20_000; iteration += 1) {
    const order = values.map((_, index) => index).sort((left, right) => values[left] - values[right]);
    simplex = order.map((index) => simplex[index]);
    values = order.map((index) => values[index]);
    const valueSpread = Math.max(...values.map((value) => Math.abs(value - values[0])));
    const pointSpread = Math.max(...simplex.map((point) => Math.max(
      ...point.map((value, index) => Math.abs(value - simplex[0][index])),
    )));
    if (valueSpread < 1e-12 && pointSpread < 1e-7) break;
    const centroid = Array.from({ length: dimensions }, (_, dimension) => (
      simplex.slice(0, dimensions)
        .reduce((sum, point) => sum + point[dimension], 0) / dimensions
    ));
    const reflected = centroid.map((value, index) => value + value - simplex.at(-1)![index]);
    const reflectedValue = objective(reflected);
    if (values[0] <= reflectedValue && reflectedValue < values[dimensions - 1]) {
      simplex[dimensions] = reflected;
      values[dimensions] = reflectedValue;
      continue;
    }
    if (reflectedValue < values[0]) {
      const expanded = centroid.map((value, index) => value + 2 * (reflected[index] - value));
      const expandedValue = objective(expanded);
      if (expandedValue < reflectedValue) {
        simplex[dimensions] = expanded;
        values[dimensions] = expandedValue;
      } else {
        simplex[dimensions] = reflected;
        values[dimensions] = reflectedValue;
      }
      continue;
    }
    const contracted = centroid.map((value, index) => (
      value + 0.5 * (simplex[dimensions][index] - value)
    ));
    const contractedValue = objective(contracted);
    if (contractedValue < values[dimensions]) {
      simplex[dimensions] = contracted;
      values[dimensions] = contractedValue;
      continue;
    }
    for (let index = 1; index <= dimensions; index += 1) {
      simplex[index] = simplex[index].map((value, dimension) => (
        simplex[0][dimension] + 0.5 * (value - simplex[0][dimension])
      ));
      values[index] = objective(simplex[index]);
    }
  }
  const bestIndex = values.indexOf(Math.min(...values));
  return { parameters: simplex[bestIndex], value: values[bestIndex] };
}

function betaGeometricLogPmf(seconds: number, alpha: number, beta: number): number {
  return logBeta(alpha + 1, beta + seconds - 1) - logBeta(alpha, beta);
}

function logBeta(left: number, right: number): number {
  return logGamma(left) + logGamma(right) - logGamma(left + right);
}

function logGamma(value: number): number {
  const coefficients = [
    676.5203681218851,
    -1259.1392167224028,
    771.3234287776531,
    -176.6150291621406,
    12.507343278686905,
    -0.13857109526572012,
    9.984369578019572e-6,
    1.5056327351493116e-7,
  ];
  if (value < 0.5) {
    return Math.log(Math.PI) - Math.log(Math.sin(Math.PI * value)) - logGamma(1 - value);
  }
  const shifted = value - 1;
  let series = 0.9999999999998099;
  for (let index = 0; index < coefficients.length; index += 1) {
    series += coefficients[index] / (shifted + index + 1);
  }
  const t = shifted + coefficients.length - 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (shifted + 0.5) * Math.log(t) - t + Math.log(series);
}

function logAddExp(left: number, right: number): number {
  const maximum = Math.max(left, right);
  return maximum + Math.log(Math.exp(left - maximum) + Math.exp(right - maximum));
}

function renderFragment(payload: string): string {
  return `<div id="btc-zero-change-gap-viz">
  <style>
    #btc-zero-change-gap-viz { position: relative; width: 100%; color: var(--foreground); }
    #btc-zero-change-gap-viz h1,
    #btc-zero-change-gap-viz h2 { font-weight: 500; }
    #btc-zero-change-gap-viz h1 { margin: 0 0 0.25rem; }
    #btc-zero-change-gap-viz h2 { margin: 0 0 0.25rem; }
    #btc-zero-change-gap-viz .plot-note { margin: 0 0 0.75rem; color: var(--muted-foreground); }
    #btc-zero-change-gap-viz .gap-legend {
      display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; margin: 0 0 1rem;
    }
    #btc-zero-change-gap-viz .gap-legend button {
      display: inline-flex; align-items: center; gap: 0.4rem; padding: 0;
      border: 0; background: transparent; color: var(--foreground); font: inherit; cursor: pointer;
    }
    #btc-zero-change-gap-viz .gap-legend button[aria-pressed="false"] { opacity: 0.45; }
    #btc-zero-change-gap-viz .legend-swatch {
      display: inline-block; width: 1rem; height: 0.18rem; background: var(--series-color);
    }
    #btc-zero-change-gap-viz .gap-grid {
      display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.25rem;
    }
    #btc-zero-change-gap-viz figure { min-width: 0; margin: 0; }
    #btc-zero-change-gap-viz .chart-host { min-height: 350px; width: 100%; }
    #btc-zero-change-gap-viz .plot-svg { display: block; width: 100%; height: auto; overflow: visible; }
    #btc-zero-change-gap-viz .plot-svg text { fill: var(--foreground); font-size: 12px; }
    #btc-zero-change-gap-viz .plot-svg .tick text,
    #btc-zero-change-gap-viz .plot-svg .panel-note { fill: var(--muted-foreground); }
    #btc-zero-change-gap-viz .plot-svg .domain,
    #btc-zero-change-gap-viz .plot-svg .tick line { stroke: var(--border); }
    #btc-zero-change-gap-viz .plot-svg .grid line { stroke: var(--border); stroke-opacity: 0.55; }
    #btc-zero-change-gap-viz .plot-svg .grid .domain { display: none; }
    #btc-zero-change-gap-viz [data-chart-frame] { fill: transparent; stroke: var(--border); }
    #btc-zero-change-gap-viz .gap-line {
      fill: none; stroke: var(--series-color); stroke-width: 1.8; vector-effect: non-scaling-stroke;
    }
    #btc-zero-change-gap-viz .gap-line.fit { stroke-dasharray: 6 4; stroke-width: 2; }
    #btc-zero-change-gap-viz .gap-point { fill: var(--series-color); }
    #btc-zero-change-gap-viz [data-chart-hover-guide] {
      stroke: var(--foreground); stroke-width: 1; pointer-events: none;
    }
    #btc-zero-change-gap-viz [data-chart-hover-marker] {
      fill: var(--background); stroke: var(--series-color); stroke-width: 2; pointer-events: none;
    }
    #btc-zero-change-gap-viz [data-chart-hit] { fill: transparent; cursor: crosshair; }
    #btc-zero-change-gap-viz .tooltip {
      position: absolute; z-index: 10; display: none; max-width: 24rem; padding: 0.55rem 0.7rem;
      border: 1px solid var(--border); background: var(--popover); color: var(--popover-foreground);
      pointer-events: none;
    }
    #btc-zero-change-gap-viz .tooltip-title { margin-bottom: 0.1rem; font-weight: 500; }
    #btc-zero-change-gap-viz .tooltip-subtitle { margin-bottom: 0.3rem; color: var(--muted-foreground); }
    #btc-zero-change-gap-viz .tooltip-row {
      display: grid; grid-template-columns: auto 1fr auto; gap: 0.45rem; align-items: center;
    }
    #btc-zero-change-gap-viz .tooltip-dot {
      width: 0.55rem; height: 0.18rem; background: var(--series-color);
    }
    @media (max-width: 800px) {
      #btc-zero-change-gap-viz .gap-grid { grid-template-columns: 1fr; }
    }
  </style>

  <h1>BTCUSDT one-second zero-change gaps</h1>
  <p class="plot-note">Maximal runs of consecutive unchanged closes · boundary-censored runs excluded · log-log axes</p>
  <div class="gap-legend" aria-label="History windows"></div>
  <div class="gap-grid">
    <figure aria-labelledby="gap-pmf-title">
      <h2 id="gap-pmf-title">Probability of an exact gap length</h2>
      <div class="chart-host" data-kind="pmf"></div>
    </figure>
    <figure aria-labelledby="gap-survival-title">
      <h2 id="gap-survival-title">Probability a gap lasts at least this long</h2>
      <div class="chart-host" data-kind="survival"></div>
    </figure>
  </div>
  <div class="tooltip" role="tooltip" aria-hidden="true"></div>

  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
    (() => {
      const root = document.getElementById("btc-zero-change-gap-viz");
      const report = ${payload};
      const tooltip = root.querySelector(".tooltip");
      const styles = [
        { id: "full", color: "var(--viz-series-1)" },
        { id: "365d", color: "var(--viz-series-2)" },
        { id: "90d", color: "var(--viz-series-3)" },
        { id: "30d", color: "var(--viz-series-4)" },
        { id: "7d", color: "var(--viz-series-5)" },
        { id: "full-range-fit", color: "var(--viz-series-6)" }
      ];
      const styleById = new Map(styles.map(style => [style.id, style]));
      const visible = new Set(styles.map(style => style.id));
      const prepared = report.windows.map(window => {
        let remaining = window.gaps;
        const survival = window.p.map(entry => {
          const point = [entry[0], remaining / window.gaps];
          remaining -= entry[1];
          return point;
        });
        return {
          ...window,
          pmf: window.p.map(entry => [entry[0], entry[2]]),
          survival,
          pmfMap: new Map(window.p.map(entry => [entry[0], entry[2]])),
          countMap: new Map(window.p.map(entry => [entry[0], entry[1]]))
        };
      });
      const full = prepared.find(window => window.id === "full");
      const maximumGap = d3.max(prepared, window => window.max);
      const fullFitSurvival = [];
      let componentOneSurvival = 1;
      let componentTwoSurvival = 1;
      for (let seconds = 1; seconds <= maximumGap; seconds += 1) {
        fullFitSurvival.push([seconds,
          report.fit.full.weight * componentOneSurvival
            + (1 - report.fit.full.weight) * componentTwoSurvival]);
        componentOneSurvival *= (report.fit.full.beta1 + seconds - 1)
          / (report.fit.full.alpha1 + report.fit.full.beta1 + seconds - 1);
        componentTwoSurvival *= (report.fit.full.beta2 + seconds - 1)
          / (report.fit.full.alpha2 + report.fit.full.beta2 + seconds - 1);
      }
      const fitted = {
        id: "full-range-fit",
        label: "Full-range beta-geometric survival fit",
        survival: fullFitSurvival,
        fit: true
      };
      const series = [...prepared, fitted];
      const minimumEmpiricalProbability = d3.min(prepared.flatMap(window => [
        ...window.pmf.map(point => point[1]),
        ...window.survival.map(point => point[1])
      ]));
      const legend = d3.select(root.querySelector(".gap-legend"));
      const buttons = legend.selectAll("button")
        .data(series)
        .join("button")
        .attr("type", "button")
        .attr("aria-pressed", "true")
        .style("--series-color", window => styleById.get(window.id).color)
        .on("click", function(event, window) {
          if (visible.has(window.id) && visible.size === 1) return;
          if (visible.has(window.id)) visible.delete(window.id); else visible.add(window.id);
          d3.select(this).attr("aria-pressed", visible.has(window.id) ? "true" : "false");
          drawAll();
        });
      buttons.append("span").attr("class", "legend-swatch").attr("aria-hidden", "true");
      buttons.append("span").text(window => window.label);

      function drawAll() {
        drawChart("pmf");
        drawChart("survival");
      }

      function drawChart(kind) {
        const chartSeries = kind === "survival" ? series : prepared;
        const host = root.querySelector('[data-kind="' + kind + '"]');
        const width = Math.max(320, Math.floor(host.getBoundingClientRect().width || 486));
        const height = 350;
        const margin = { top: 18, right: 20, bottom: 58, left: 72 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        const x = d3.scaleLog().domain([1, maximumGap]).range([margin.left, width - margin.right]);
        const fitViewportFloor = survivalAt(fitted, Math.min(300, maximumGap));
        const minimumProbability = kind === "survival"
          ? Math.min(minimumEmpiricalProbability, fitViewportFloor)
          : minimumEmpiricalProbability;
        const y = d3.scaleLog()
          .domain([minimumProbability / 1.35, kind === "survival" ? 1.08 : 0.8])
          .range([height - margin.bottom, margin.top]);
        const svg = d3.select(host).selectAll("svg").data([null]).join("svg")
          .attr("class", "plot-svg")
          .attr("viewBox", "0 0 " + width + " " + height)
          .attr("role", "img")
          .attr("aria-label", kind === "pmf"
            ? "Probability mass of completed BTCUSDT one-second zero-change gaps across five history windows."
            : "Survival probability of completed BTCUSDT one-second zero-change gaps across five history windows.");
        svg.selectAll("*").remove();
        svg.append("title").text(kind === "pmf" ? "Exact zero-change gap probability" : "Zero-change gap survival probability");
        svg.append("desc").text("Both axes are logarithmic. Gap length is measured in seconds.");
        const clipId = "gap-clip-" + kind;
        svg.append("defs").append("clipPath").attr("id", clipId)
          .append("rect")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", innerWidth).attr("height", innerHeight);
        const yTicks = probabilityTicks(y.domain());
        svg.append("g").attr("class", "grid")
          .attr("transform", "translate(" + margin.left + ",0)")
          .call(d3.axisLeft(y).tickValues(yTicks).tickSize(-innerWidth).tickFormat(""));
        const xTicks = gapTicks(x, width);
        const xAxis = svg.append("g")
          .attr("transform", "translate(0," + (height - margin.bottom) + ")")
          .call(d3.axisBottom(x).tickValues(xTicks).tickFormat(formatDuration));
        const tickTexts = xAxis.selectAll(".tick text");
        tickTexts.filter((_, index) => index === 0).attr("text-anchor", "start");
        tickTexts.filter((_, index, nodes) => index === nodes.length - 1).attr("text-anchor", "end");
        svg.append("g").attr("transform", "translate(" + margin.left + ",0)")
          .call(d3.axisLeft(y).tickValues(yTicks).tickFormat(formatAxisProbability));
        svg.append("rect").attr("data-chart-frame", "")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", innerWidth).attr("height", innerHeight);
        svg.append("text").attr("class", "axis-title").attr("data-axis", "x")
          .attr("x", margin.left + innerWidth / 2).attr("y", height - 8)
          .attr("text-anchor", "middle").text("Zero-change gap length");
        svg.append("text").attr("class", "axis-title").attr("data-axis", "y")
          .attr("transform", "translate(17," + (margin.top + innerHeight / 2) + ") rotate(-90)")
          .attr("text-anchor", "middle")
          .text(kind === "pmf" ? "Probability of exact length" : "Probability gap ≥ length");
        const plot = svg.append("g").attr("clip-path", "url(#" + clipId + ")");
        if (kind === "survival") {
          const survivalExponent = d3.format(".2f")(
            Math.min(report.fit.full.alpha1, report.fit.full.alpha2));
          svg.append("text").attr("class", "panel-note")
            .attr("x", margin.left + 6).attr("y", margin.top + 14)
            .text(width <= 420
              ? "full fit · tail exponent ≈ " + survivalExponent
              : "fit across observed 1–" + report.fit.full.fittedThrough
                + "s · asymptotic survival exponent ≈ " + survivalExponent);
        }
        chartSeries.forEach(window => {
          if (!visible.has(window.id)) return;
          const style = styleById.get(window.id);
          const values = kind === "pmf" ? withPmfGaps(window.pmf) : window.survival;
          const line = d3.line().defined(point => point[1] > 0)
            .x(point => x(point[0])).y(point => y(point[1]))
            .curve(kind === "survival" && !window.fit ? d3.curveStepAfter : d3.curveLinear);
          plot.append("path").datum(values)
            .attr("class", "gap-line" + (window.fit ? " fit" : ""))
            .attr("data-series", window.id)
            .style("--series-color", style.color).attr("d", line);
          if (kind === "pmf" && !window.fit) {
            plot.selectAll("circle.gap-point-" + window.id)
              .data(window.pmf).join("circle")
              .attr("class", "gap-point gap-point-" + window.id)
              .attr("data-series-point", window.id)
              .style("--series-color", style.color)
              .attr("cx", point => x(point[0])).attr("cy", point => y(point[1])).attr("r", 1.6);
          }
        });
        if (kind === "pmf") {
          const longest = full.longest[0];
          const probability = full.pmfMap.get(longest[0]);
          svg.append("text").attr("class", "panel-note")
            .attr("x", width - margin.right - 5).attr("y", margin.top + 14)
            .attr("text-anchor", "end")
            .text("Full: median " + full.q[0] + "s · p99 " + full.q[3] + "s · max " + formatDuration(full.max));
          if (probability) {
            plot.append("text").attr("class", "panel-note")
              .attr("x", x(longest[0]) - 5).attr("y", y(probability) - 7)
              .attr("text-anchor", "end").text(formatDuration(longest[0]));
          }
        }
        const guide = svg.append("line").attr("data-chart-hover-guide", "")
          .attr("y1", margin.top).attr("y2", height - margin.bottom).style("display", "none");
        const markers = svg.append("g");
        svg.append("rect")
          .attr("data-chart-hit", "").attr("data-chart-hover-overlay", "cross-series")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", innerWidth).attr("height", innerHeight)
          .on("pointermove", function(event) {
            const pointer = d3.pointer(event, svg.node());
            const cursorX = Math.max(margin.left, Math.min(width - margin.right, pointer[0]));
            const seconds = Math.max(1, Math.round(x.invert(cursorX)));
            guide.attr("x1", cursorX).attr("x2", cursorX).style("display", null);
            markers.selectAll("circle").remove();
            const rows = chartSeries.filter(window => visible.has(window.id)).map(window => {
              const probability = kind === "pmf"
                ? (window.pmfMap.get(seconds) || 0)
                : survivalAt(window, seconds);
              if (probability > 0) {
                markers.append("circle").attr("data-chart-hover-marker", "")
                  .style("--series-color", styleById.get(window.id).color)
                  .attr("cx", cursorX).attr("cy", y(probability)).attr("r", 3.5);
              }
              return {
                window,
                probability,
                count: kind === "pmf" && !window.fit ? (window.countMap.get(seconds) || 0) : null
              };
            });
            showTooltip(event, kind, seconds, rows);
          })
          .on("pointerleave", () => {
            guide.style("display", "none");
            markers.selectAll("circle").remove();
            hideTooltip();
          });
      }

      function withPmfGaps(values) {
        const points = [];
        let previous = null;
        values.forEach(point => {
          if (previous !== null && point[0] > previous + 1) points.push([point[0], null]);
          points.push(point);
          previous = point[0];
        });
        return points;
      }

      function survivalAt(window, seconds) {
        const index = d3.bisector(point => point[0]).left(window.survival, seconds);
        return index >= window.survival.length ? 0 : window.survival[index][1];
      }

      function showTooltip(event, kind, seconds, rows) {
        const fullEvent = full.longest.find(gap => gap[0] === seconds);
        tooltip.innerHTML = '<div class="tooltip-title">' + formatDuration(seconds) + ' gap</div>'
          + '<div class="tooltip-subtitle">' + (kind === "pmf" ? 'Exact length' : 'At least this length')
          + (fullEvent ? ' · ' + fullEvent[1] : '') + '</div>'
          + rows.map(row => '<div class="tooltip-row"><span class="tooltip-dot" style="--series-color:'
            + styleById.get(row.window.id).color + '"></span><span>' + row.window.label + '</span><span>'
            + (row.probability > 0 ? formatProbability(row.probability) : '0 observed')
            + (row.count === null || row.count === 0 ? '' : ' · ' + d3.format(',')(row.count))
            + '</span></div>').join('');
        tooltip.style.display = "block";
        tooltip.setAttribute("aria-hidden", "false");
        const rootRect = root.getBoundingClientRect();
        const tipRect = tooltip.getBoundingClientRect();
        let left = event.clientX - rootRect.left + 12;
        let top = event.clientY - rootRect.top + 12;
        if (left + tipRect.width > rootRect.width) left -= tipRect.width + 24;
        tooltip.style.left = Math.max(0, left) + "px";
        tooltip.style.top = Math.max(0, top) + "px";
      }

      function hideTooltip() {
        tooltip.style.display = "none";
        tooltip.setAttribute("aria-hidden", "true");
      }

      function gapTicks(scale, width) {
        const candidates = [1, 2, 5, 10, 30, 60, 300, 3600, maximumGap]
          .filter((value, index, values) => value <= maximumGap && values.indexOf(value) === index);
        const minimumSpacing = width <= 420 ? 62 : 48;
        const chosen = [candidates[0]];
        for (const value of candidates.slice(1, -1)) {
          if (scale(value) - scale(chosen.at(-1)) >= minimumSpacing
            && scale(maximumGap) - scale(value) >= minimumSpacing) chosen.push(value);
        }
        if (maximumGap !== chosen.at(-1)) chosen.push(maximumGap);
        return chosen;
      }

      function probabilityTicks(domain) {
        const first = Math.ceil(Math.log10(domain[0]));
        const last = Math.floor(Math.log10(domain[1]));
        const exponents = d3.range(first, last + 1);
        const stride = Math.max(1, Math.ceil(exponents.length / 6));
        return exponents.filter((_, index) => index % stride === 0).map(exponent => 10 ** exponent);
      }

      function formatDuration(seconds) {
        if (seconds < 60) return seconds + "s";
        if (seconds < 3600) return d3.format(".3~g")(seconds / 60) + "m";
        return d3.format(".3~g")(seconds / 3600) + "h";
      }

      function formatProbability(value) {
        return value >= 0.0001 ? d3.format(".4%")(value) : d3.format(".2e")(value);
      }

      function formatAxisProbability(value) {
        if (value >= 0.01) return d3.format(".1%")(value);
        if (value >= 0.0001) return d3.format(".3%")(value);
        return d3.format(".0e")(value);
      }

      let resizeFrame = 0;
      new ResizeObserver(() => {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(drawAll);
      }).observe(root);
      drawAll();
    })();
  </script>
</div>
`;
}

function parseArgs(args: string[]): Map<string, string> {
  const values = new Map<string, string>();
  for (let index = 0; index < args.length; index += 1) {
    const key = args[index];
    const value = args[index + 1];
    if (!key?.startsWith("--") || !value || value.startsWith("--")) {
      throw new Error(`Invalid argument near ${key ?? "end"}.`);
    }
    values.set(key.slice(2), value);
    index += 1;
  }
  return values;
}

function windowRank(id: string): number {
  return ({ full: 0, "365d": 1, "90d": 2, "30d": 3, "7d": 4 } as Record<string, number>)[id]
    ?? Number.MAX_SAFE_INTEGER;
}

function shortWindowLabel(id: string): string {
  return ({
    full: "Full 5 years",
    "365d": "Trailing 365d",
    "90d": "Trailing 90d",
    "30d": "Trailing 30d",
    "7d": "Trailing 7d",
  } as Record<string, string>)[id] ?? id;
}

function round(value: number): number {
  return Number(value.toPrecision(10));
}

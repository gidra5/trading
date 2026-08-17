import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface HistogramBin {
  0: number;
  1: number;
}

interface HistogramWindow {
  id: string;
  label: string;
  zeroProbability: number;
  histogram: {
    observations: number;
    binWidthBps: number;
    lowerBps: number;
    upperBps: number;
    binCount: number;
    underflowProbability: number;
    overflowProbability: number;
    nonzeroBins: HistogramBin[];
  };
}

interface HistogramReport {
  version: number;
  symbol: string;
  commonEndTime: string;
  scales: Array<{
    id: string;
    label: string;
    windows: HistogramWindow[];
  }>;
}

interface ReturnFitReport {
  version: number;
  symbol: string;
  commonEndTime: string;
  scales: Array<{
    id: string;
    family: string;
    empiricalOutsideMass: number;
    excludedBinSigma: [number, number] | null;
    modelOutsideProbability: number;
    parameters: {
      locationSigma: number;
      scaleSigma: number;
      power: number | null;
      tail: number | null;
      degreesFreedom: number | null;
      logNormalizer: number;
    };
    asymptoticPdfExponent: number | null;
  }>;
}

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");

try {
  main();
} catch (error: unknown) {
  console.error(error instanceof Error ? error.stack ?? error.message : String(error));
  process.exitCode = 1;
}

function main(): void {
  const args = parseArgs(process.argv.slice(2));
  const inputPath = path.resolve(
    repoRoot,
    args.get("input") ?? "data/benchmarks/log-return-histograms.json",
  );
  const outputValue = args.get("output");
  if (!outputValue) throw new Error("--output is required.");
  const outputPath = path.resolve(outputValue);
  const report = JSON.parse(fs.readFileSync(inputPath, "utf8")) as HistogramReport;
  const fitInputPath = path.resolve(
    repoRoot,
    args.get("fit-input") ?? "data/benchmarks/log-return-distribution-fits.json",
  );
  const fits = JSON.parse(fs.readFileSync(fitInputPath, "utf8")) as ReturnFitReport;
  if (report.version !== 1 || report.symbol !== "BTCUSDT" || report.scales.length !== 6) {
    throw new Error("Histogram input is not the expected six-scale BTCUSDT report.");
  }
  if (fits.version !== 1 || fits.symbol !== report.symbol
    || fits.commonEndTime !== report.commonEndTime || fits.scales.length !== report.scales.length) {
    throw new Error("Return-fit input does not match the histogram report.");
  }
  const payload = report.scales.map((scale) => {
    const orderedWindows = [...scale.windows].sort(
      (left, right) => windowRank(left.id) - windowRank(right.id),
    );
    const first = orderedWindows[0]!.histogram;
    const fit = fits.scales.find((candidate) => candidate.id === scale.id);
    if (!fit) throw new Error(`Return-fit input is missing scale ${scale.id}.`);
    return {
      id: scale.id,
      label: scale.label,
      lower: round(first.lowerBps),
      upper: round(first.upperBps),
      width: round(first.binWidthBps),
      count: first.binCount,
      zero: round(orderedWindows[0]!.zeroProbability),
      fit: {
        family: fit.family,
        outsideMass: round(fit.empiricalOutsideMass),
        excluded: fit.excludedBinSigma?.map(round) ?? null,
        modelOutside: round(fit.modelOutsideProbability),
        mu: round(fit.parameters.locationSigma),
        s: round(fit.parameters.scaleSigma),
        p: fit.parameters.power === null ? null : round(fit.parameters.power),
        q: fit.parameters.tail === null ? null : round(fit.parameters.tail),
        degrees: fit.parameters.degreesFreedom === null
          ? null
          : round(fit.parameters.degreesFreedom),
        logNormalizer: round(fit.parameters.logNormalizer),
        tailExponent: fit.asymptoticPdfExponent === null
          ? null
          : round(fit.asymptoticPdfExponent),
      },
      windows: orderedWindows.map((window) => ({
        id: window.id,
        label: shortWindowLabel(window.id),
        n: window.histogram.observations,
        under: round(window.histogram.underflowProbability),
        over: round(window.histogram.overflowProbability),
        p: window.histogram.nonzeroBins.map((bin) => [bin[0], round(bin[1])]),
      })),
    };
  });
  const fragment = renderFragment(
    JSON.stringify({ end: report.commonEndTime.slice(0, 10), scales: payload })
      .replaceAll("</script", "<\\/script"),
  );
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, fragment, "utf8");
  console.log(outputPath);
}

function renderFragment(payload: string): string {
  return `<div id="btc-return-probability-viz">
  <style>
    #btc-return-probability-viz { position: relative; width: 100%; color: var(--foreground); }
    #btc-return-probability-viz h1,
    #btc-return-probability-viz h2 { font-weight: 500; }
    #btc-return-probability-viz h1 { margin: 0 0 0.25rem; }
    #btc-return-probability-viz h2 { margin: 0 0 0.25rem; }
    #btc-return-probability-viz .plot-note { margin: 0 0 0.75rem; color: var(--muted-foreground); }
    #btc-return-probability-viz .probability-legend {
      display: flex; flex-wrap: wrap; gap: 0.35rem 1rem; margin: 0 0 1rem;
    }
    #btc-return-probability-viz .probability-legend button {
      display: inline-flex; align-items: center; gap: 0.4rem; padding: 0;
      border: 0; background: transparent; color: var(--foreground); font: inherit; cursor: pointer;
    }
    #btc-return-probability-viz .probability-legend button[aria-pressed="false"] { opacity: 0.45; }
    #btc-return-probability-viz .legend-swatch {
      display: inline-block; width: 1rem; height: 0.18rem; background: var(--series-color);
    }
    #btc-return-probability-viz .legend-swatch.gaussian {
      height: 0; border-top: 2px dashed var(--muted-foreground); background: transparent;
    }
    #btc-return-probability-viz .legend-swatch.fit {
      height: 0; border-top: 2px dashed var(--series-color); background: transparent;
    }
    #btc-return-probability-viz .probability-grid {
      display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.25rem;
    }
    #btc-return-probability-viz figure { min-width: 0; margin: 0; }
    #btc-return-probability-viz .chart-host { min-height: 275px; width: 100%; }
    #btc-return-probability-viz .plot-svg { display: block; width: 100%; height: auto; overflow: visible; }
    #btc-return-probability-viz .plot-svg text { fill: var(--foreground); font-size: 12px; }
    #btc-return-probability-viz .plot-svg .tick text,
    #btc-return-probability-viz .plot-svg .panel-note { fill: var(--muted-foreground); }
    #btc-return-probability-viz .plot-svg .domain,
    #btc-return-probability-viz .plot-svg .tick line { stroke: var(--border); }
    #btc-return-probability-viz .plot-svg .grid line { stroke: var(--border); stroke-opacity: 0.55; }
    #btc-return-probability-viz .plot-svg .grid .domain { display: none; }
    #btc-return-probability-viz [data-chart-frame] { fill: transparent; stroke: var(--border); }
    #btc-return-probability-viz .histogram-line {
      fill: none; stroke: var(--series-color); stroke-width: 1.8; vector-effect: non-scaling-stroke;
    }
    #btc-return-probability-viz .gaussian-line {
      fill: none; stroke: var(--muted-foreground); stroke-width: 1.5;
      stroke-dasharray: 5 4; vector-effect: non-scaling-stroke;
    }
    #btc-return-probability-viz .fit-line {
      fill: none; stroke: var(--series-color); stroke-width: 2;
      stroke-dasharray: 7 4; vector-effect: non-scaling-stroke;
    }
    #btc-return-probability-viz .zero-guide {
      stroke: var(--muted-foreground); stroke-width: 1; stroke-opacity: 0.65;
    }
    #btc-return-probability-viz [data-chart-hover-guide] {
      stroke: var(--foreground); stroke-width: 1; pointer-events: none;
    }
    #btc-return-probability-viz [data-chart-hover-marker] {
      fill: var(--background); stroke: var(--series-color); stroke-width: 2; pointer-events: none;
    }
    #btc-return-probability-viz [data-chart-hit] { fill: transparent; cursor: grab; touch-action: none; }
    #btc-return-probability-viz [data-chart-hit]:active { cursor: grabbing; }
    #btc-return-probability-viz .tooltip {
      position: absolute; z-index: 10; display: none; max-width: 22rem; padding: 0.55rem 0.7rem;
      border: 1px solid var(--border); background: var(--popover); color: var(--popover-foreground);
      pointer-events: none;
    }
    #btc-return-probability-viz .tooltip-title { margin-bottom: 0.1rem; font-weight: 500; }
    #btc-return-probability-viz .tooltip-subtitle { margin-bottom: 0.3rem; color: var(--muted-foreground); }
    #btc-return-probability-viz .tooltip-row {
      display: grid; grid-template-columns: auto 1fr auto; gap: 0.45rem; align-items: center;
    }
    #btc-return-probability-viz .tooltip-dot {
      width: 0.55rem; height: 0.18rem; background: var(--series-color);
    }
    #btc-return-probability-viz .tooltip-dot.gaussian {
      height: 0; border-top: 2px dashed var(--muted-foreground); background: transparent;
    }
    @media (max-width: 800px) {
      #btc-return-probability-viz .probability-grid { grid-template-columns: 1fr; }
    }
  </style>

  <h1>Empirical probability of BTCUSDT log returns</h1>
  <p class="plot-note">Fixed 0.1σ bins · 1s–1m fits exclude the central bin · 15m+ fits include zero</p>
  <div class="viz-controls" aria-label="Axis scale">
    <span>Axis scale</span>
    <button type="button" class="btn" data-axis-mode="linear" aria-pressed="true">Linear</button>
    <button type="button" class="btn" data-axis-mode="log" aria-pressed="false">Log tails</button>
    <span>Return units</span>
    <button type="button" class="btn" data-return-unit="normalized" aria-pressed="true">Normalized σ</button>
    <button type="button" class="btn" data-return-unit="native" aria-pressed="false">Native bp</button>
    <span class="text-small text-muted">Wheel or pinch to zoom · drag to pan · double-click to reset</span>
  </div>
  <div class="probability-legend" aria-label="History windows"></div>
  <div class="probability-grid"></div>
  <div class="tooltip" role="tooltip" aria-hidden="true"></div>

  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
    (() => {
      const root = document.getElementById("btc-return-probability-viz");
      const report = ${payload};
      const tooltip = root.querySelector(".tooltip");
      const windowStyles = [
        { id: "full", label: "Full 5 years", color: "var(--viz-series-1)" },
        { id: "365d", label: "Trailing 365d", color: "var(--viz-series-2)" },
        { id: "90d", label: "Trailing 90d", color: "var(--viz-series-3)" },
        { id: "30d", label: "Trailing 30d", color: "var(--viz-series-4)" },
        { id: "7d", label: "Trailing 7d", color: "var(--viz-series-5)" },
        { id: "fit", label: "Selected fit · full history", color: "var(--viz-series-6)", fit: true },
        { id: "gaussian", label: "Gaussian", color: "var(--muted-foreground)", gaussian: true }
      ];
      const styleById = new Map(windowStyles.map(item => [item.id, item]));
      const visible = new Set(windowStyles.map(item => item.id));
      const legend = d3.select(root.querySelector(".probability-legend"));
      const grid = d3.select(root.querySelector(".probability-grid"));
      const modes = [
        { id: "linear", label: "linear axes" },
        { id: "log", label: "symmetric-log return and logarithmic probability axes" }
      ];
      let activeMode = modes[0];
      let normalizedReturns = true;
      const zoomTransforms = new Map();

      d3.select(root).selectAll("[data-axis-mode]")
        .on("click", function() {
          const nextMode = modes.find(mode => mode.id === this.dataset.axisMode);
          if (!nextMode || nextMode.id === activeMode.id) return;
          activeMode = nextMode;
          zoomTransforms.clear();
          root.querySelectorAll("[data-axis-mode]").forEach(button => {
            button.setAttribute("aria-pressed", button.dataset.axisMode === activeMode.id ? "true" : "false");
          });
          drawAll();
        });

      d3.select(root).selectAll("[data-return-unit]")
        .on("click", function() {
          const nextNormalized = this.dataset.returnUnit === "normalized";
          if (nextNormalized === normalizedReturns) return;
          normalizedReturns = nextNormalized;
          zoomTransforms.clear();
          root.querySelectorAll("[data-return-unit]").forEach(button => {
            button.setAttribute("aria-pressed",
              (button.dataset.returnUnit === "normalized") === normalizedReturns ? "true" : "false");
          });
          updatePanelHeadings();
          drawAll();
        });

      const buttons = legend.selectAll("button")
        .data(windowStyles)
        .join("button")
        .attr("type", "button")
        .attr("aria-pressed", "true")
        .style("--series-color", item => item.color)
        .on("click", function(event, item) {
          if (visible.has(item.id) && visible.size === 1) return;
          if (visible.has(item.id)) visible.delete(item.id); else visible.add(item.id);
          d3.select(this).attr("aria-pressed", visible.has(item.id) ? "true" : "false");
          drawAll();
        });
      buttons.append("span")
        .attr("class", item => "legend-swatch"
          + (item.gaussian ? " gaussian" : "") + (item.fit ? " fit" : ""))
        .attr("aria-hidden", "true");
      buttons.append("span").text(item => item.label);

      const figures = grid.selectAll("figure")
        .data(report.scales)
        .join("figure")
        .attr("aria-labelledby", scale => "probability-title-" + scale.id);
      figures.append("h2")
        .attr("id", scale => "probability-title-" + scale.id)
        .text(scale => scale.label);
      figures.append("div")
        .attr("class", "chart-host")
        .attr("data-scale", scale => scale.id);
      updatePanelHeadings();

      function updatePanelHeadings() {
        figures.select("h2").text(scale => scale.label + " · "
          + (normalizedReturns ? "0.1σ bins" : formatBps(scale.width) + " bins"));
      }

      function normalCdf(value) {
        const sign = value < 0 ? -1 : 1;
        const x = Math.abs(value) / Math.sqrt(2);
        const t = 1 / (1 + 0.3275911 * x);
        const erf = sign * (1 - (((((1.061405429 * t - 1.453152027) * t)
          + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x));
        return 0.5 * (1 + erf);
      }

      function gaussianBins(scale) {
        const sigma = scale.width / 0.1;
        return d3.range(scale.count).flatMap(index => {
          const lower = scale.lower + index * scale.width;
          const upper = lower + scale.width;
          const probability = Math.max(0, normalCdf(upper / sigma) - normalCdf(lower / sigma));
          return probability > 0 ? [[index, probability]] : [];
        });
      }

      function generalizedTFitProbability(scale, index) {
        const sigma = scale.width / 0.1;
        const midpointSigma = (scale.lower + (index + 0.5) * scale.width) / sigma;
        if (scale.fit.excluded && midpointSigma >= scale.fit.excluded[0]
          && midpointSigma < scale.fit.excluded[1]) return 0;
        const distance = Math.abs(midpointSigma - scale.fit.mu) / scale.fit.s;
        let logDensity;
        if (scale.fit.family === "symmetric-generalized-t") {
          logDensity = scale.fit.logNormalizer
            - scale.fit.q * Math.log1p(distance ** scale.fit.p);
        } else if (scale.fit.family === "symmetric-generalized-normal") {
          logDensity = scale.fit.logNormalizer - distance ** scale.fit.p;
        } else if (scale.fit.family === "symmetric-student-t") {
          logDensity = scale.fit.logNormalizer - (scale.fit.degrees + 1) / 2
            * Math.log1p(distance ** 2 / scale.fit.degrees);
        } else {
          logDensity = scale.fit.logNormalizer - distance;
        }
        const density = Math.exp(logDensity);
        return scale.fit.outsideMass / scale.fit.modelOutside * density * 0.1;
      }

      function generalizedTFitPoints(scale, firstIndex, lastIndex) {
        const stride = Math.max(1, Math.floor((lastIndex - firstIndex + 1) / 1_200));
        const excludedIndex = Math.floor((0 - scale.lower) / scale.width);
        const indices = new Set([firstIndex, lastIndex, excludedIndex - 1,
          excludedIndex, excludedIndex + 1]);
        for (let index = firstIndex; index <= lastIndex; index += stride) indices.add(index);
        return [...indices]
          .filter(index => index >= firstIndex && index <= lastIndex)
          .sort((left, right) => left - right)
          .map(index => ({
            x: scale.lower + (index + 0.5) * scale.width,
            y: generalizedTFitProbability(scale, index) || null,
          }));
      }

      function pointsFor(scale, probabilities) {
        const points = [];
        let previousIndex = null;
        probabilities.forEach(pair => {
          const index = pair[0];
          const probability = pair[1];
          const lower = scale.lower + index * scale.width;
          const upper = lower + scale.width;
          if (previousIndex !== null && index > previousIndex + 1) {
            points.push({ x: lower, y: null });
          }
          points.push({ x: lower, y: probability > 0 ? probability : null });
          points.push({ x: upper, y: probability > 0 ? probability : null });
          previousIndex = index;
        });
        return points;
      }

      const empiricalProbabilities = report.scales.flatMap(scale =>
        scale.windows.flatMap(window => window.p.map(pair => pair[1]))
      );
      const probabilityExtent = d3.extent(empiricalProbabilities);
      const sharedLogYDomain = [probabilityExtent[0] / 1.35, Math.min(1, probabilityExtent[1] * 1.2)];

      function drawAll() {
        report.scales.forEach(scale => drawScale(scale, activeMode));
      }

      function drawScale(scale, mode) {
        const host = root.querySelector('[data-scale="' + scale.id + '"]');
        const width = Math.max(320, Math.floor(host.getBoundingClientRect().width || 486));
        const height = 275;
        const margin = { top: 12, right: 18, bottom: 52, left: 68 };
        const innerWidth = width - margin.left - margin.right;
        const innerHeight = height - margin.top - margin.bottom;
        const fullBins = scale.windows.find(window => window.id === "full").p;
        const firstObservedIndex = fullBins[0][0];
        const lastObservedIndex = fullBins[fullBins.length - 1][0];
        const observedLower = scale.lower + firstObservedIndex * scale.width;
        const observedUpper = scale.lower + (lastObservedIndex + 1) * scale.width;
        const sigma = scale.width / 0.1;
        const returnDivisor = normalizedReturns ? sigma : 1;
        const observedLowerX = observedLower / returnDivisor;
        const observedUpperX = observedUpper / returnDivisor;
        const xPadding = (observedUpperX - observedLowerX) * 0.005;
        const fullXDomain = [observedLowerX - xPadding, observedUpperX + xPadding];
        const baseX = (mode.id === "linear"
          ? d3.scaleLinear()
          : d3.scaleSymlog().constant(sigma / returnDivisor))
          .domain(fullXDomain)
          .range([margin.left, width - margin.right]);
        const centralDomain = normalizedReturns
          ? [Math.max(fullXDomain[0], -8), Math.min(fullXDomain[1], 8)]
          : fullXDomain;
        const initialTransform = transformForDomain(baseX, centralDomain);
        let currentTransform = zoomTransforms.get(scale.id) || initialTransform;
        let x = currentTransform.rescaleX(baseX);
        const scaleProbabilityMax = d3.max(scale.windows.flatMap(window => window.p.map(pair => pair[1])));
        let y = mode.id === "linear"
          ? d3.scaleLinear().domain([0, scaleProbabilityMax * 1.06]).nice(4)
            .range([height - margin.bottom, margin.top])
          : d3.scaleLog().domain(sharedLogYDomain)
            .range([height - margin.bottom, margin.top]);
        const svg = d3.select(host).selectAll("svg").data([null]).join("svg")
          .attr("class", "plot-svg")
          .attr("viewBox", "0 0 " + width + " " + height)
          .attr("role", "img")
          .attr("aria-label", "Empirical binned probability of " + scale.label
            + " BTCUSDT log returns across five history windows on " + mode.label
            + (normalizedReturns ? " in full-history standard deviations" : " in basis points")
            + ", ending " + report.end + ".");
        svg.selectAll("*").remove();
        svg.append("title").text("BTCUSDT " + scale.label + " log-return probability");
        svg.append("desc").text("Each line is the probability of a return falling in a "
          + (normalizedReturns ? "0.1 standard deviation" : formatBps(scale.width))
          + " interval. The chart uses " + mode.label + ".");
        const clipId = "probability-clip-" + scale.id;
        svg.append("defs").append("clipPath").attr("id", clipId)
          .append("rect")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", innerWidth).attr("height", innerHeight);
        const yGridGroup = svg.append("g").attr("class", "grid")
          .attr("transform", "translate(" + margin.left + ",0)");
        const xAxisGroup = svg.append("g")
          .attr("transform", "translate(0," + (height - margin.bottom) + ")")
          .call(axisForX(x));
        const yAxisGroup = svg.append("g")
          .attr("transform", "translate(" + margin.left + ",0)");
        svg.append("rect").attr("data-chart-frame", "")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", innerWidth).attr("height", innerHeight);
        svg.append("text").attr("class", "axis-title").attr("data-axis", "x")
          .attr("x", margin.left + innerWidth / 2).attr("y", height - 7)
          .attr("text-anchor", "middle")
          .text(normalizedReturns ? "Log return / full-history σ" : "Log return (bp)");
        svg.append("text").attr("class", "axis-title").attr("data-axis", "y")
          .attr("transform", "translate(16," + (margin.top + innerHeight / 2) + ") rotate(-90)")
          .attr("text-anchor", "middle").text("Probability per bin");
        const plot = svg.append("g").attr("clip-path", "url(#" + clipId + ")");
        const zeroGuide = plot.append("line").attr("class", "zero-guide")
          .attr("x1", x(0)).attr("x2", x(0))
          .attr("y1", margin.top).attr("y2", height - margin.bottom);
        const line = d3.line()
          .defined(point => point.y !== null
            && (mode.id === "linear" || point.y > 0))
          .x(point => x(point.x / returnDivisor)).y(point => y(point.y));
        scale.windows.forEach(window => {
          if (!visible.has(window.id)) return;
          const style = styleById.get(window.id);
          plot.append("path").datum(pointsFor(scale, window.p))
            .attr("class", "histogram-line")
            .attr("data-series", window.id)
            .style("--series-color", style.color)
            .attr("d", line);
        });
        const gaussian = gaussianBins(scale);
        const fitPoints = generalizedTFitPoints(scale, firstObservedIndex, lastObservedIndex);
        const probabilityMaps = new Map(scale.windows.map(window => [window.id, new Map(window.p)]));
        const gaussianMap = new Map(gaussian);
        if (visible.has("fit")) {
          plot.append("path").datum(fitPoints)
            .attr("class", "fit-line").attr("data-series", "fit")
            .style("--series-color", styleById.get("fit").color).attr("d", line);
        }
        if (visible.has("gaussian")) {
          plot.append("path").datum(pointsFor(scale, gaussian))
            .attr("class", "gaussian-line").attr("data-series", "gaussian").attr("d", line);
        }
        if (scale.id === "1s") {
          svg.append("text").attr("class", "panel-note")
            .attr("x", width - margin.right - 5).attr("y", margin.top + 14)
            .attr("text-anchor", "end")
            .text("P(r = 0) = " + d3.format(".2%")(scale.zero));
        }
        const guide = svg.append("line").attr("data-chart-hover-guide", "")
          .attr("y1", margin.top).attr("y2", height - margin.bottom).style("display", "none");
        const markers = svg.append("g");
        const overlay = svg.append("rect")
          .attr("data-chart-hit", "")
          .attr("data-chart-hover-overlay", "cross-series")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", innerWidth).attr("height", innerHeight)
          .on("pointermove", function(event) {
            const pointer = d3.pointer(event, svg.node());
            const cursorX = Math.max(margin.left, Math.min(width - margin.right, pointer[0]));
            const returnValue = x.invert(cursorX);
            const returnBps = returnValue * returnDivisor;
            const binIndex = Math.max(0, Math.min(
              scale.count - 1,
              Math.floor((returnBps - scale.lower) / scale.width)
            ));
            const binLower = scale.lower + binIndex * scale.width;
            const binUpper = binLower + scale.width;
            guide.attr("x1", cursorX).attr("x2", cursorX).style("display", null);
            markers.selectAll("circle").remove();
            const rows = [];
            scale.windows.forEach(window => {
              if (!visible.has(window.id)) return;
              const probability = probabilityMaps.get(window.id).get(binIndex) || 0;
              const style = styleById.get(window.id);
              rows.push({ label: style.label, color: style.color, probability });
              if (probability > 0 && probability >= y.domain()[0] && probability <= y.domain()[1]) {
                markers.append("circle").attr("data-chart-hover-marker", "")
                  .style("--series-color", style.color)
                  .attr("cx", cursorX).attr("cy", y(probability)).attr("r", 3.5);
              }
            });
            if (visible.has("fit")) {
              const probability = generalizedTFitProbability(scale, binIndex);
              const style = styleById.get("fit");
              rows.push({ label: style.label, color: style.color, probability });
              if (probability > 0 && probability >= y.domain()[0] && probability <= y.domain()[1]) {
                markers.append("circle").attr("data-chart-hover-marker", "")
                  .style("--series-color", style.color)
                  .attr("cx", cursorX).attr("cy", y(probability)).attr("r", 3.5);
              }
            }
            if (visible.has("gaussian")) {
              const probability = gaussianMap.get(binIndex) || 0;
              rows.push({ label: "Gaussian", color: "var(--muted-foreground)", probability, gaussian: true });
              if (probability > 0 && probability >= y.domain()[0] && probability <= y.domain()[1]) {
                markers.append("circle").attr("data-chart-hover-marker", "")
                  .style("--series-color", "var(--muted-foreground)")
                  .attr("cx", cursorX).attr("cy", y(probability)).attr("r", 3.5);
              }
            }
            showTooltip(event, returnBps, returnValue, binLower, binUpper, rows, returnDivisor);
          })
          .on("pointerleave", function() {
            guide.style("display", "none");
            markers.selectAll("circle").remove();
            hideTooltip();
          });

        const zoom = d3.zoom()
          .scaleExtent([1, 128])
          .extent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]])
          .translateExtent([[margin.left, margin.top], [width - margin.right, height - margin.bottom]])
          .filter(event => event.type !== "dblclick" && !event.button)
          .on("zoom", event => {
            currentTransform = event.transform;
            zoomTransforms.set(scale.id, currentTransform);
            x = currentTransform.rescaleX(baseX);
            renderHorizontalView();
          });

        overlay.call(zoom)
          .on("dblclick.zoom", null)
          .on("dblclick.reset", () => {
            zoomTransforms.delete(scale.id);
            overlay.call(zoom.transform, initialTransform);
          });
        overlay.call(zoom.transform, currentTransform);

        function renderHorizontalView() {
          fitYToVisibleX();
          xAxisGroup.call(axisForX(x));
          zeroGuide.attr("x1", x(0)).attr("x2", x(0));
          plot.selectAll("[data-series]").attr("d", line);
          guide.style("display", "none");
          markers.selectAll("circle").remove();
          hideTooltip();
        }

        function fitYToVisibleX() {
          const visibleXDomain = x.domain();
          const visibleLowerBps = Math.min(...visibleXDomain) * returnDivisor;
          const visibleUpperBps = Math.max(...visibleXDomain) * returnDivisor;
          const probabilities = [];
          const collectVisibleProbabilities = pairs => {
            pairs.forEach(pair => {
              const binLower = scale.lower + pair[0] * scale.width;
              const binUpper = binLower + scale.width;
              if (pair[1] > 0 && binUpper >= visibleLowerBps && binLower <= visibleUpperBps) {
                probabilities.push(pair[1]);
              }
            });
          };
          scale.windows.forEach(window => {
            if (visible.has(window.id)) collectVisibleProbabilities(window.p);
          });
          if (visible.has("fit")) {
            fitPoints.forEach(point => {
              if (point.y > 0 && point.x >= visibleLowerBps && point.x <= visibleUpperBps) {
                probabilities.push(point.y);
              }
            });
          }
          if (probabilities.length === 0 && visible.has("gaussian")) {
            collectVisibleProbabilities(gaussian);
          }

          if (mode.id === "linear") {
            const maximum = d3.max(probabilities) || scaleProbabilityMax;
            y = d3.scaleLinear().domain([0, maximum * 1.06]).nice(4)
              .range([height - margin.bottom, margin.top]);
          } else {
            const extent = d3.extent(probabilities);
            let minimum = extent[0] || sharedLogYDomain[0];
            let maximum = extent[1] || sharedLogYDomain[1];
            if (minimum === maximum) {
              minimum /= 3;
              maximum *= 3;
            }
            y = d3.scaleLog()
              .domain([Math.max(Number.MIN_VALUE, minimum / 1.2), Math.min(1, maximum * 1.2)])
              .range([height - margin.bottom, margin.top]);
          }
          const ticks = tickValuesForY(y);
          yGridGroup.call(d3.axisLeft(y).tickValues(ticks).tickSize(-innerWidth).tickFormat(""));
          yAxisGroup.call(d3.axisLeft(y).tickValues(ticks).tickFormat(formatProbability));
        }

        function axisForX(scaleX) {
          return d3.axisBottom(scaleX)
            .tickValues(tickValuesForX(scaleX))
            .tickFormat(value => formatAxisReturn(value, normalizedReturns));
        }

        function tickValuesForX(scaleX) {
          if (currentTransform.k <= 1.0001 && mode.id === "linear") {
            return scaleX.ticks(width <= 420 ? 3 : 4);
          }
          if (currentTransform.k <= 1.0001 && mode.id === "log") {
            const displayedSigma = sigma / returnDivisor;
            return width <= 420 ? [observedLowerX, 0, observedUpperX] : [
              observedLowerX,
              ...[-10 * displayedSigma, 10 * displayedSigma].filter(value =>
                value >= observedLowerX && value <= observedUpperX
                && scaleX(value) - scaleX(observedLowerX) >= 52
                && scaleX(observedUpperX) - scaleX(value) >= 52
                && Math.abs(scaleX(value) - scaleX(0)) >= 52
              ),
              0,
              observedUpperX
            ].sort((left, right) => left - right);
          }
          const maximumTicks = width <= 420 ? 3 : 5;
          const candidates = scaleX.ticks(maximumTicks);
          if (candidates.length <= maximumTicks) return candidates;
          return d3.range(maximumTicks).map(index =>
            candidates[Math.round(index * (candidates.length - 1) / (maximumTicks - 1))]
          );
        }

        function tickValuesForY(scaleY) {
          if (mode.id === "linear") return scaleY.ticks(4);
          const domain = scaleY.domain();
          const firstExponent = Math.ceil(Math.log10(domain[0]));
          const lastExponent = Math.floor(Math.log10(domain[1]));
          const exponents = d3.range(firstExponent, lastExponent + 1);
          const stride = Math.max(1, Math.ceil(exponents.length / 5));
          return exponents.filter((_, index) => index % stride === 0).map(exponent => 10 ** exponent);
        }
      }

      function transformForDomain(scale, domain) {
        const range = scale.range();
        const left = scale(domain[0]);
        const right = scale(domain[1]);
        const zoom = (range[1] - range[0]) / (right - left);
        if (!Number.isFinite(zoom) || zoom <= 1.0001) return d3.zoomIdentity;
        return d3.zoomIdentity.translate(range[0] - zoom * left, 0).scale(zoom);
      }

      function showTooltip(event, returnBps, returnValue, lower, upper, rows, returnDivisor) {
        const title = normalizedReturns
          ? formatSigma(returnValue) + " · " + formatBps(returnBps)
          : formatBps(returnBps);
        const binLabel = normalizedReturns
          ? "Bin [" + formatSigma(lower / returnDivisor) + ", "
            + formatSigma(upper / returnDivisor) + ")"
          : "Bin [" + formatBinBps(lower, upper - lower) + ", "
            + formatBinBps(upper, upper - lower) + ")";
        tooltip.innerHTML = '<div class="tooltip-title">r = ' + title + '</div>'
          + '<div class="tooltip-subtitle">' + binLabel + '</div>'
          + rows.map(row => '<div class="tooltip-row"><span class="tooltip-dot'
            + (row.gaussian ? ' gaussian' : '') + '" style="--series-color:' + row.color
            + '"></span><span>' + row.label + '</span><span>'
            + formatTooltipProbability(row.probability) + '</span></div>').join("");
        tooltip.style.display = "block";
        tooltip.setAttribute("aria-hidden", "false");
        const rootRect = root.getBoundingClientRect();
        const tipRect = tooltip.getBoundingClientRect();
        let left = event.clientX - rootRect.left + 12;
        let top = event.clientY - rootRect.top + 12;
        if (left + tipRect.width > rootRect.width) {
          left = event.clientX - rootRect.left - tipRect.width - 12;
        }
        tooltip.style.left = Math.max(0, left) + "px";
        tooltip.style.top = Math.max(0, top) + "px";
      }

      function hideTooltip() {
        tooltip.style.display = "none";
        tooltip.setAttribute("aria-hidden", "true");
      }

      function formatBps(value) {
        const absolute = Math.abs(value);
        const digits = absolute < 0.1 ? 3 : absolute < 10 ? 2 : 1;
        return d3.format("." + digits + "f")(value) + " bp";
      }

      function formatSigma(value) {
        const absolute = Math.abs(value);
        const digits = absolute < 1 ? 2 : absolute < 10 ? 1 : 0;
        return d3.format("." + digits + "f")(value) + "σ";
      }

      function formatAxisReturn(value, normalized) {
        if (!normalized) return d3.format(Math.abs(value) < 1 ? ".2f" : ".0f")(value);
        const absolute = Math.abs(value);
        return d3.format(absolute < 1 ? ".2f" : absolute < 10 ? ".1f" : ".0f")(value);
      }

      function formatBinBps(value, width) {
        const digits = width < 0.1 ? 3 : width < 1 ? 2 : width < 10 ? 1 : 0;
        return d3.format("." + digits + "f")(value) + " bp";
      }

      function formatProbability(value) {
        if (value === 0) return "0%";
        return value >= 0.001 ? d3.format(".1%")(value) : d3.format(".0e")(value);
      }

      function formatTooltipProbability(value) {
        if (value === 0) return "0 observed";
        return value >= 0.0001 ? d3.format(".4%")(value) : d3.format(".2e")(value);
      }

      let resizeFrame = 0;
      let observedRootWidth = Math.round(root.getBoundingClientRect().width);
      new ResizeObserver(() => {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(() => {
          const nextWidth = Math.round(root.getBoundingClientRect().width);
          if (nextWidth !== observedRootWidth) zoomTransforms.clear();
          observedRootWidth = nextWidth;
          drawAll();
        });
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

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

interface Comparison {
  sigmaBps: number;
  varianceRatio: number;
  integratedReturnCorrelation: number;
  jsDivergenceBits: number;
  centralMass: { observed: number; independent: number; ratio: number };
  threeSigmaTail: {
    observed: number;
    independent: number;
    ratio: number;
    observedCount: number;
    independentExpectedCount: number;
  };
  fiveSigmaTail: {
    observed: number;
    independent: number;
    ratio: number;
    observedCount: number;
    independentExpectedCount: number;
  };
}

interface HierarchyReport {
  version: number;
  symbol: string;
  commonEndTime: string;
  hierarchyValidation: {
    sampledDays: number;
    minuteEndpoints: number;
    mismatchedEndpoints: number;
    maximumCloseDifferenceBps: number;
  };
  scales: Array<{
    id: string;
    label: string;
    seconds: number;
    parentId: string;
    parentFactor: number;
    observations: number;
    sigmaBps: number;
    oneSecondIndependent: Comparison;
    parentIndependent: Comparison;
    points: Array<[number, number, number, number, number | null, number | null]>;
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
    args.get("input") ?? "data/benchmarks/return-aggregation-hierarchy.json",
  );
  const outputValue = args.get("output");
  if (!outputValue) throw new Error("--output is required.");
  const outputPath = path.resolve(outputValue);
  const report = JSON.parse(fs.readFileSync(inputPath, "utf8")) as HierarchyReport;
  if (report.version !== 1 || report.symbol !== "BTCUSDT" || report.scales.length !== 5) {
    throw new Error("Input is not the expected BTCUSDT aggregation hierarchy report.");
  }
  const payload = {
    end: report.commonEndTime.slice(0, 10),
    validation: report.hierarchyValidation,
    scales: report.scales,
  };
  const serialized = JSON.stringify(payload).replaceAll("</script", "<\\/script");
  fs.mkdirSync(path.dirname(outputPath), { recursive: true });
  fs.writeFileSync(outputPath, renderFragment(serialized), "utf8");
  console.log(outputPath);
}

function renderFragment(payload: string): string {
  return `<div id="btc-return-hierarchy-viz">
  <style>
    #btc-return-hierarchy-viz {
      --actual: #0b8f87;
      --parent-iid: #d56b36;
      --second-iid: #7064c2;
      --variance: #2878b5;
      --peak: #d56b36;
      --tail: #bd3b4f;
      position: relative; width: 100%; color: var(--foreground);
    }
    #btc-return-hierarchy-viz h1,
    #btc-return-hierarchy-viz h2 { font-weight: 500; }
    #btc-return-hierarchy-viz h1 { margin: 0 0 0.25rem; }
    #btc-return-hierarchy-viz h2 { margin: 0 0 0.25rem; font-size: 1.05rem; }
    #btc-return-hierarchy-viz p { line-height: 1.45; }
    #btc-return-hierarchy-viz .subtle { margin: 0 0 0.85rem; color: var(--muted-foreground); }
    #btc-return-hierarchy-viz .finding {
      margin: 0 0 1rem; padding: 0.7rem 0.85rem; border-left: 3px solid var(--actual);
      background: color-mix(in srgb, var(--actual) 8%, transparent);
    }
    #btc-return-hierarchy-viz .controls,
    #btc-return-hierarchy-viz .legend {
      display: flex; flex-wrap: wrap; align-items: center; gap: 0.4rem 0.85rem;
      margin: 0 0 0.85rem;
    }
    #btc-return-hierarchy-viz .controls > span { color: var(--muted-foreground); }
    #btc-return-hierarchy-viz button {
      border: 1px solid var(--border); border-radius: 0.3rem; padding: 0.25rem 0.5rem;
      background: var(--background); color: var(--foreground); font: inherit; cursor: pointer;
    }
    #btc-return-hierarchy-viz button[aria-pressed="true"] {
      border-color: var(--foreground); background: var(--secondary);
    }
    #btc-return-hierarchy-viz .legend button {
      display: inline-flex; align-items: center; gap: 0.35rem; border: 0; padding: 0;
      background: transparent;
    }
    #btc-return-hierarchy-viz .legend button[aria-pressed="false"] { opacity: 0.42; }
    #btc-return-hierarchy-viz .swatch { width: 1.05rem; height: 0; border-top: 2px solid var(--c); }
    #btc-return-hierarchy-viz .swatch.parent { border-top-style: dashed; }
    #btc-return-hierarchy-viz .swatch.second { border-top-style: dotted; }
    #btc-return-hierarchy-viz .summary-section { margin: 0 0 1.3rem; }
    #btc-return-hierarchy-viz .summary-host { width: 100%; min-height: 260px; }
    #btc-return-hierarchy-viz .chart-grid {
      display: grid; grid-template-columns: minmax(0, 1fr) minmax(0, 1fr); gap: 1.15rem;
    }
    #btc-return-hierarchy-viz figure { min-width: 0; margin: 0; }
    #btc-return-hierarchy-viz figcaption { margin: 0 0 0.15rem; }
    #btc-return-hierarchy-viz .panel-meta { color: var(--muted-foreground); font-size: 0.82rem; }
    #btc-return-hierarchy-viz .chart-host { width: 100%; min-height: 390px; }
    #btc-return-hierarchy-viz svg { display: block; width: 100%; height: auto; overflow: visible; }
    #btc-return-hierarchy-viz svg text { fill: var(--foreground); font-size: 11px; }
    #btc-return-hierarchy-viz svg .muted { fill: var(--muted-foreground); }
    #btc-return-hierarchy-viz svg .domain,
    #btc-return-hierarchy-viz svg .tick line { stroke: var(--border); }
    #btc-return-hierarchy-viz svg .grid line { stroke: var(--border); stroke-opacity: 0.55; }
    #btc-return-hierarchy-viz svg .grid .domain { display: none; }
    #btc-return-hierarchy-viz .frame { fill: transparent; stroke: var(--border); }
    #btc-return-hierarchy-viz .zero-line { stroke: var(--muted-foreground); stroke-opacity: 0.65; }
    #btc-return-hierarchy-viz .actual-line,
    #btc-return-hierarchy-viz .parent-line,
    #btc-return-hierarchy-viz .second-line,
    #btc-return-hierarchy-viz .residual-parent,
    #btc-return-hierarchy-viz .residual-second,
    #btc-return-hierarchy-viz .metric-line { fill: none; vector-effect: non-scaling-stroke; }
    #btc-return-hierarchy-viz .actual-line { stroke: var(--actual); stroke-width: 2; }
    #btc-return-hierarchy-viz .parent-line { stroke: var(--parent-iid); stroke-width: 1.8; stroke-dasharray: 7 4; }
    #btc-return-hierarchy-viz .second-line { stroke: var(--second-iid); stroke-width: 1.6; stroke-dasharray: 2 4; }
    #btc-return-hierarchy-viz .residual-parent { stroke: var(--parent-iid); stroke-width: 1.5; }
    #btc-return-hierarchy-viz .residual-second { stroke: var(--second-iid); stroke-width: 1.35; }
    #btc-return-hierarchy-viz .metric-line { stroke: var(--c); stroke-width: 2; }
    #btc-return-hierarchy-viz .metric-dot { fill: var(--background); stroke: var(--c); stroke-width: 2; }
    #btc-return-hierarchy-viz .hover-guide { stroke: var(--foreground); stroke-width: 1; pointer-events: none; }
    #btc-return-hierarchy-viz .hover-dot { fill: var(--background); stroke: var(--c); stroke-width: 2; pointer-events: none; }
    #btc-return-hierarchy-viz .hit { fill: transparent; cursor: grab; touch-action: none; }
    #btc-return-hierarchy-viz .hit:active { cursor: grabbing; }
    #btc-return-hierarchy-viz .tooltip {
      position: absolute; z-index: 20; display: none; max-width: 22rem; padding: 0.55rem 0.7rem;
      border: 1px solid var(--border); border-radius: 0.25rem;
      background: var(--popover); color: var(--popover-foreground); pointer-events: none;
      box-shadow: 0 0.25rem 1rem color-mix(in srgb, var(--foreground) 12%, transparent);
    }
    #btc-return-hierarchy-viz .tip-title { font-weight: 500; margin-bottom: 0.25rem; }
    #btc-return-hierarchy-viz .tip-row {
      display: grid; grid-template-columns: 0.7rem minmax(7.5rem, 1fr) auto;
      gap: 0.4rem; align-items: center; white-space: nowrap;
    }
    #btc-return-hierarchy-viz .tip-dot { width: 0.55rem; height: 0.18rem; background: var(--c); }
    #btc-return-hierarchy-viz .tip-note { margin-top: 0.25rem; color: var(--muted-foreground); font-size: 0.8rem; }
    @media (max-width: 800px) {
      #btc-return-hierarchy-viz .chart-grid { grid-template-columns: 1fr; }
    }
  </style>

  <h1>What aggregation adds beyond the 1-second marginal</h1>
  <p class="subtle">BTCUSDT full history through <span data-end></span> · exact contiguous returns versus IID convolution</p>
  <p class="finding"><strong>The hierarchy separates two effects:</strong> variance is nearly additive above 1 minute, while the sharper center and heavier tails keep rebuilding at every aggregation step.</p>

  <section class="summary-section" aria-labelledby="hierarchy-summary-title">
    <h2 id="hierarchy-summary-title">Dependence signal added at each step</h2>
    <p class="subtle">Observed statistic divided by a convolution of independent parent blocks; 1 means the parent marginal fully explains the child.</p>
    <div class="summary-host" data-summary></div>
  </section>

  <div class="controls" aria-label="Chart controls">
    <span>Probability axis</span>
    <button type="button" data-axis="log" aria-pressed="true">Log tails</button>
    <button type="button" data-axis="linear" aria-pressed="false">Linear</button>
    <span>Wheel/pinch to zoom · drag to pan · double-click for full observed range</span>
  </div>
  <div class="legend" aria-label="Visible probability series">
    <button type="button" data-series-toggle="actual" aria-pressed="true"><span class="swatch" style="--c:var(--actual)"></span>Observed contiguous</button>
    <button type="button" data-series-toggle="parent" aria-pressed="true"><span class="swatch parent" style="--c:var(--parent-iid)"></span>Parent blocks IID</button>
    <button type="button" data-series-toggle="second" aria-pressed="true"><span class="swatch second" style="--c:var(--second-iid)"></span>1s returns IID</button>
  </div>
  <p class="subtle">Lower strips show log₁₀(observed / IID). Residuals stop where either side represents fewer than 3 events at that scale.</p>
  <div class="chart-grid" data-grid></div>
  <div class="tooltip" role="tooltip" aria-hidden="true"></div>

  <script src="https://cdn.jsdelivr.net/npm/d3@7.9.0/dist/d3.min.js"></script>
  <script>
    (() => {
      const report = ${payload};
      const root = document.getElementById("btc-return-hierarchy-viz");
      const d3 = window.d3;
      root.querySelector("[data-end]").textContent = report.end;
      const tooltip = root.querySelector(".tooltip");
      const visible = new Set(["actual", "parent", "second"]);
      const viewDomains = new Map();
      let axisMode = "log";

      const summaryMetrics = [
        { id: "variance", label: "Variance", color: "var(--variance)", value: s => s.parentIndependent.varianceRatio },
        { id: "peak", label: "Mass inside ±0.25σ", color: "var(--peak)", value: s => s.parentIndependent.centralMass.ratio },
        { id: "tail", label: ">3σ tail", color: "var(--tail)", value: s => s.parentIndependent.threeSigmaTail.ratio },
      ];

      const summaryLegend = document.createElement("div");
      summaryLegend.className = "legend";
      summaryMetrics.forEach(metric => {
        const item = document.createElement("span");
        item.innerHTML = '<span class="swatch" style="--c:' + metric.color + '"></span> ' + metric.label;
        item.style.display = "inline-flex";
        item.style.alignItems = "center";
        item.style.gap = "0.35rem";
        summaryLegend.appendChild(item);
      });
      root.querySelector("[data-summary]").before(summaryLegend);

      const grid = root.querySelector("[data-grid]");
      report.scales.forEach(scale => {
        const figure = document.createElement("figure");
        const local = scale.parentIndependent;
        figure.innerHTML = '<figcaption><h2>' + scale.parentId + ' × ' + scale.parentFactor + ' → ' + scale.id
          + '</h2><div class="panel-meta">variance ' + d3.format(".3~f")(local.varianceRatio)
          + '× · center ' + d3.format(".2f")(local.centralMass.ratio)
          + '× · >3σ tail ' + d3.format(".2f")(local.threeSigmaTail.ratio) + '×</div></figcaption>'
          + '<div class="chart-host" data-scale="' + scale.id + '"></div>';
        grid.appendChild(figure);
      });

      root.querySelectorAll("[data-axis]").forEach(button => {
        button.addEventListener("click", () => {
          axisMode = button.dataset.axis;
          root.querySelectorAll("[data-axis]").forEach(candidate => {
            candidate.setAttribute("aria-pressed", String(candidate === button));
          });
          drawPanels();
        });
      });
      root.querySelectorAll("[data-series-toggle]").forEach(button => {
        button.addEventListener("click", () => {
          const id = button.dataset.seriesToggle;
          if (visible.has(id) && visible.size > 1) visible.delete(id); else visible.add(id);
          button.setAttribute("aria-pressed", String(visible.has(id)));
          drawPanels();
        });
      });

      function drawSummary() {
        const host = root.querySelector("[data-summary]");
        const width = Math.max(320, Math.floor(host.getBoundingClientRect().width || 900));
        const height = 260;
        const margin = { top: 12, right: 22, bottom: 52, left: 58 };
        const x = d3.scalePoint().domain(report.scales.map(s => s.id))
          .range([margin.left, width - margin.right]).padding(0.35);
        const allValues = summaryMetrics.flatMap(metric => report.scales.map(metric.value));
        const y = d3.scaleLog().domain([0.85, Math.max(3.6, d3.max(allValues) * 1.12)])
          .range([height - margin.bottom, margin.top]);
        const svg = d3.select(host).selectAll("svg").data([null]).join("svg")
          .attr("viewBox", "0 0 " + width + " " + height)
          .attr("role", "img")
          .attr("aria-label", "Observed to independent-parent ratios for variance, central mass, and three-sigma tails at five aggregation steps.");
        svg.selectAll("*").remove();
        svg.append("title").text("Dependence signal at each aggregation step");
        svg.append("desc").text("Variance becomes nearly additive above one minute, but central mass and tail probability remain above an independent-parent convolution.");
        const ticks = [1, 1.5, 2, 3].filter(value => value <= y.domain()[1]);
        svg.append("g").attr("class", "grid").attr("transform", "translate(" + margin.left + ",0)")
          .call(d3.axisLeft(y).tickValues(ticks).tickSize(-(width - margin.left - margin.right)).tickFormat(""));
        svg.append("g").attr("transform", "translate(0," + (height - margin.bottom) + ")")
          .call(d3.axisBottom(x).tickFormat((id, index) => report.scales[index].parentId + "→" + id));
        svg.append("g").attr("transform", "translate(" + margin.left + ",0)")
          .call(d3.axisLeft(y).tickValues(ticks).tickFormat(value => d3.format(".2~f")(value) + "×"));
        svg.append("line").attr("class", "zero-line")
          .attr("x1", margin.left).attr("x2", width - margin.right).attr("y1", y(1)).attr("y2", y(1));
        const plot = svg.append("g");
        summaryMetrics.forEach(metric => {
          const points = report.scales.map(scale => ({ id: scale.id, value: metric.value(scale) }));
          plot.append("path").datum(points).attr("class", "metric-line")
            .style("--c", metric.color)
            .attr("d", d3.line().x(point => x(point.id)).y(point => y(point.value)));
          plot.selectAll("circle." + metric.id).data(points).join("circle")
            .attr("class", "metric-dot " + metric.id).style("--c", metric.color)
            .attr("cx", point => x(point.id)).attr("cy", point => y(point.value)).attr("r", 3.5);
        });
        const guide = svg.append("line").attr("class", "hover-guide")
          .attr("y1", margin.top).attr("y2", height - margin.bottom).style("display", "none");
        svg.append("rect").attr("class", "hit")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", width - margin.left - margin.right).attr("height", height - margin.top - margin.bottom)
          .on("pointermove", event => {
            const px = d3.pointer(event, svg.node())[0];
            const nearest = report.scales.reduce((best, scale) =>
              Math.abs(x(scale.id) - px) < Math.abs(x(best.id) - px) ? scale : best
            , report.scales[0]);
            guide.attr("x1", x(nearest.id)).attr("x2", x(nearest.id)).style("display", null);
            showTip(event, nearest.parentId + " × " + nearest.parentFactor + " → " + nearest.id,
              summaryMetrics.map(metric => ({ label: metric.label, color: metric.color, value: d3.format(".3~f")(metric.value(nearest)) + "×" })),
              "Ratios above 1 are structure not reproduced by independent parent blocks.");
          })
          .on("pointerleave", () => { guide.style("display", "none"); hideTip(); });
      }

      function drawPanels() {
        report.scales.forEach(drawPanel);
      }

      function drawPanel(scale) {
        const host = root.querySelector('[data-scale="' + scale.id + '"]');
        const width = Math.max(320, Math.floor(host.getBoundingClientRect().width || 480));
        const height = 390;
        const margin = { top: 12, right: 16, bottom: 44, left: 61 };
        const probabilityBottom = 238;
        const residualTop = 276;
        const residualBottom = height - margin.bottom;
        const fullDomain = d3.extent(scale.points, point => point[0]);
        const initialDomain = [Math.max(fullDomain[0], -12), Math.min(fullDomain[1], 12)];
        const desiredDomain = viewDomains.get(scale.id) || initialDomain;
        const baseX = d3.scaleLinear().domain(fullDomain).range([margin.left, width - margin.right]);
        let currentTransform = transformForDomain(baseX, desiredDomain);
        let x = currentTransform.rescaleX(baseX);
        let yProbability;
        let yResidual;
        const svg = d3.select(host).selectAll("svg").data([null]).join("svg")
          .attr("viewBox", "0 0 " + width + " " + height)
          .attr("role", "img")
          .attr("aria-label", scale.label + " observed return probabilities versus independent parent-block and independent one-second convolutions.");
        svg.selectAll("*").remove();
        svg.append("title").text(scale.parentId + " to " + scale.id + " return aggregation");
        svg.append("desc").text("The upper chart compares binned probabilities. The lower chart is the log-ratio residual, which is zero when the IID baseline matches.");
        const clipId = "hierarchy-clip-" + scale.id;
        svg.append("defs").append("clipPath").attr("id", clipId).append("rect")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", width - margin.left - margin.right).attr("height", residualBottom - margin.top);
        svg.append("rect").attr("class", "frame").attr("x", margin.left).attr("y", margin.top)
          .attr("width", width - margin.left - margin.right).attr("height", probabilityBottom - margin.top);
        svg.append("rect").attr("class", "frame").attr("x", margin.left).attr("y", residualTop)
          .attr("width", width - margin.left - margin.right).attr("height", residualBottom - residualTop);
        svg.append("text").attr("class", "muted").attr("x", 14)
          .attr("y", margin.top + (probabilityBottom - margin.top) / 2)
          .attr("text-anchor", "middle").attr("transform", "rotate(-90,14," + (margin.top + (probabilityBottom - margin.top) / 2) + ")")
          .text("Probability per 0.1σ bin");
        svg.append("text").attr("class", "muted").attr("x", 14)
          .attr("y", residualTop + (residualBottom - residualTop) / 2)
          .attr("text-anchor", "middle").attr("transform", "rotate(-90,14," + (residualTop + (residualBottom - residualTop) / 2) + ")")
          .text("log₁₀ ratio");
        svg.append("text").attr("x", margin.left + (width - margin.left - margin.right) / 2)
          .attr("y", height - 6).attr("text-anchor", "middle").text("Log return / observed σ");

        const probabilityGrid = svg.append("g").attr("class", "grid")
          .attr("transform", "translate(" + margin.left + ",0)");
        const probabilityAxis = svg.append("g").attr("transform", "translate(" + margin.left + ",0)");
        const residualAxis = svg.append("g").attr("transform", "translate(" + margin.left + ",0)");
        const xAxis = svg.append("g").attr("transform", "translate(0," + residualBottom + ")");
        const plot = svg.append("g").attr("clip-path", "url(#" + clipId + ")");
        const zeroReturn = plot.append("line").attr("class", "zero-line")
          .attr("y1", margin.top).attr("y2", residualBottom);
        const residualZero = plot.append("line").attr("class", "zero-line")
          .attr("x1", margin.left).attr("x2", width - margin.right);
        const probabilityPaths = {
          second: plot.append("path").datum(scale.points).attr("class", "second-line"),
          parent: plot.append("path").datum(scale.points).attr("class", "parent-line"),
          actual: plot.append("path").datum(scale.points).attr("class", "actual-line"),
        };
        const residualPaths = {
          second: plot.append("path").datum(scale.points).attr("class", "residual-second"),
          parent: plot.append("path").datum(scale.points).attr("class", "residual-parent"),
        };
        const guide = svg.append("line").attr("class", "hover-guide")
          .attr("y1", margin.top).attr("y2", residualBottom).style("display", "none");
        const markers = svg.append("g");

        function renderView() {
          x = currentTransform.rescaleX(baseX);
          viewDomains.set(scale.id, x.domain());
          fitVerticalScales();
          xAxis.call(d3.axisBottom(x).ticks(width <= 420 ? 4 : 6).tickFormat(d3.format("~g")));
          zeroReturn.attr("x1", x(0)).attr("x2", x(0));
          residualZero.attr("y1", yResidual(0)).attr("y2", yResidual(0));
          const probabilityLine = index => d3.line()
            .defined(point => visible.has(index.id) && point[index.column] > 0)
            .x(point => x(point[0])).y(point => yProbability(point[index.column]));
          probabilityPaths.actual.attr("d", probabilityLine({ id: "actual", column: 1 }));
          probabilityPaths.second.attr("d", probabilityLine({ id: "second", column: 2 }));
          probabilityPaths.parent.attr("d", probabilityLine({ id: "parent", column: 3 }));
          residualPaths.second.style("display", visible.has("actual") && visible.has("second") ? null : "none")
            .attr("d", d3.line().defined(point => point[4] !== null)
              .x(point => x(point[0])).y(point => yResidual(point[4])));
          residualPaths.parent.style("display", visible.has("actual") && visible.has("parent") ? null : "none")
            .attr("d", d3.line().defined(point => point[5] !== null)
              .x(point => x(point[0])).y(point => yResidual(point[5])));
          guide.style("display", "none");
          markers.selectAll("*").remove();
          hideTip();
        }

        function fitVerticalScales() {
          const domain = x.domain();
          const visiblePoints = scale.points.filter(point => point[0] >= domain[0] && point[0] <= domain[1]);
          const columns = [["actual", 1], ["second", 2], ["parent", 3]].filter(item => visible.has(item[0]));
          const probabilities = visiblePoints.flatMap(point => columns.map(item => point[item[1]]).filter(value => value > 0));
          const maxProbability = d3.max(probabilities) || 1;
          if (axisMode === "linear") {
            yProbability = d3.scaleLinear().domain([0, maxProbability * 1.06]).nice(4)
              .range([probabilityBottom, margin.top]);
          } else {
            const minProbability = d3.min(probabilities) || maxProbability / 100;
            yProbability = d3.scaleLog().domain([Math.max(Number.MIN_VALUE, minProbability / 1.25), Math.min(1, maxProbability * 1.18)])
              .range([probabilityBottom, margin.top]);
          }
          const probabilityTicks = axisMode === "linear"
            ? yProbability.ticks(4)
            : logTicks(yProbability.domain(), 5);
          probabilityGrid.call(d3.axisLeft(yProbability).tickValues(probabilityTicks)
            .tickSize(-(width - margin.left - margin.right)).tickFormat(""));
          probabilityAxis.call(d3.axisLeft(yProbability).tickValues(probabilityTicks).tickFormat(formatProbability));
          const residualValues = visiblePoints.flatMap(point => [
            visible.has("actual") && visible.has("second") ? point[4] : null,
            visible.has("actual") && visible.has("parent") ? point[5] : null,
          ]).filter(value => value !== null && Number.isFinite(value));
          let extent = residualValues.length ? d3.extent(residualValues) : [-0.5, 0.5];
          const bound = Math.max(0.35, Math.min(3, Math.max(Math.abs(extent[0]), Math.abs(extent[1])) * 1.12));
          yResidual = d3.scaleLinear().domain([-bound, bound]).range([residualBottom, residualTop]);
          residualAxis.call(d3.axisLeft(yResidual).ticks(3).tickFormat(d3.format(".1f")));
        }

        const zoom = d3.zoom().scaleExtent([1, 256])
          .extent([[margin.left, margin.top], [width - margin.right, residualBottom]])
          .translateExtent([[margin.left, margin.top], [width - margin.right, residualBottom]])
          .filter(event => event.type !== "dblclick" && !event.button)
          .on("zoom", event => { currentTransform = event.transform; renderView(); });
        const overlay = svg.append("rect").attr("class", "hit")
          .attr("x", margin.left).attr("y", margin.top)
          .attr("width", width - margin.left - margin.right).attr("height", residualBottom - margin.top)
          .on("pointermove.tip", event => {
            const cursor = d3.pointer(event, svg.node())[0];
            const value = x.invert(cursor);
            const index = d3.bisector(point => point[0]).center(scale.points, value);
            const point = scale.points[index];
            guide.attr("x1", x(point[0])).attr("x2", x(point[0])).style("display", null);
            markers.selectAll("*").remove();
            const rows = [];
            const series = [
              { id: "actual", label: "Observed", color: "var(--actual)", column: 1 },
              { id: "parent", label: scale.parentId + " blocks IID", color: "var(--parent-iid)", column: 3 },
              { id: "second", label: "1s returns IID", color: "var(--second-iid)", column: 2 },
            ];
            series.filter(item => visible.has(item.id)).forEach(item => {
              const probability = point[item.column];
              rows.push({ label: item.label, color: item.color, value: formatProbabilityTip(probability) });
              if (probability > 0 && probability >= yProbability.domain()[0] && probability <= yProbability.domain()[1]) {
                markers.append("circle").attr("class", "hover-dot").style("--c", item.color)
                  .attr("cx", x(point[0])).attr("cy", yProbability(probability)).attr("r", 3.2);
              }
            });
            if (point[5] !== null) rows.push({ label: "Observed / parent IID", color: "var(--parent-iid)", value: d3.format(".3~f")(10 ** point[5]) + "×" });
            if (point[4] !== null) rows.push({ label: "Observed / 1s IID", color: "var(--second-iid)", value: d3.format(".3~f")(10 ** point[4]) + "×" });
            showTip(event, d3.format("+.2f")(point[0]) + "σ", rows, "0.1σ-wide bin · n = " + d3.format(",")(scale.observations));
          })
          .on("pointerleave.tip", () => { guide.style("display", "none"); markers.selectAll("*").remove(); hideTip(); });
        overlay.call(zoom).on("dblclick.zoom", null).on("dblclick.reset", () => {
          viewDomains.set(scale.id, fullDomain);
          overlay.call(zoom.transform, d3.zoomIdentity);
        });
        overlay.call(zoom.transform, currentTransform);
      }

      function transformForDomain(scale, domain) {
        const range = scale.range();
        const left = scale(domain[0]);
        const right = scale(domain[1]);
        const k = (range[1] - range[0]) / (right - left);
        if (!Number.isFinite(k) || k <= 1.000001) return d3.zoomIdentity;
        return d3.zoomIdentity.translate(range[0] - k * left, 0).scale(k);
      }

      function logTicks(domain, maximum) {
        const first = Math.ceil(Math.log10(domain[0]));
        const last = Math.floor(Math.log10(domain[1]));
        const exponents = d3.range(first, last + 1);
        const stride = Math.max(1, Math.ceil(exponents.length / maximum));
        return exponents.filter((_, index) => index % stride === 0).map(exponent => 10 ** exponent);
      }

      function formatProbability(value) {
        return value >= 0.001 ? d3.format(".1%")(value) : d3.format(".0e")(value);
      }

      function formatProbabilityTip(value) {
        if (value === 0) return "0";
        return value >= 0.0001 ? d3.format(".4%")(value) : d3.format(".3e")(value);
      }

      function showTip(event, title, rows, note) {
        tooltip.innerHTML = '<div class="tip-title">' + title + '</div>'
          + rows.map(row => '<div class="tip-row"><span class="tip-dot" style="--c:' + row.color
            + '"></span><span>' + row.label + '</span><span>' + row.value + '</span></div>').join("")
          + (note ? '<div class="tip-note">' + note + '</div>' : "");
        tooltip.style.display = "block";
        tooltip.setAttribute("aria-hidden", "false");
        const rootRect = root.getBoundingClientRect();
        const tipRect = tooltip.getBoundingClientRect();
        let left = event.clientX - rootRect.left + 12;
        let top = event.clientY - rootRect.top + 12;
        if (left + tipRect.width > rootRect.width) left = event.clientX - rootRect.left - tipRect.width - 12;
        if (top + tipRect.height > rootRect.height) top = event.clientY - rootRect.top - tipRect.height - 12;
        tooltip.style.left = Math.max(0, left) + "px";
        tooltip.style.top = Math.max(0, top) + "px";
      }

      function hideTip() {
        tooltip.style.display = "none";
        tooltip.setAttribute("aria-hidden", "true");
      }

      let resizeFrame = 0;
      let observedWidth = Math.round(root.getBoundingClientRect().width);
      new ResizeObserver(() => {
        cancelAnimationFrame(resizeFrame);
        resizeFrame = requestAnimationFrame(() => {
          const nextWidth = Math.round(root.getBoundingClientRect().width);
          if (nextWidth === observedWidth) return;
          observedWidth = nextWidth;
          drawSummary();
          drawPanels();
        });
      }).observe(root);
      drawSummary();
      drawPanels();
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

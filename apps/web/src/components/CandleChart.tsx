import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import type {
  BacktestExtremumTrace,
  BacktestOraclePoint,
  BacktestTrace,
  BacktestChartAnnotation,
  BacktestChartSmaSeries,
  Candle,
  TradingOrderSnapshot,
} from "@trading/bot-algo";
import { formatQuote, formatTime } from "../format";

interface CandleChartProps {
  candles: Candle[];
  orders: readonly TradingOrderSnapshot[];
  lastPrice: number;
  smaSeries?: BacktestChartSmaSeries[];
  annotations?: BacktestChartAnnotation[];
  maxCandles?: number;
  minInteractiveCandles?: number;
  interactive?: boolean;
  timeNavigation?: boolean;
  emptyLabel?: string;
  selectedTime?: number;
  cursorTime?: number;
  viewport?: CandleChartViewport;
  priceDisplay?: CandleChartPriceDisplay;
  trace?: BacktestTrace;
  overlays?: Partial<CandleChartOverlayVisibility>;
  highlightedPositionId?: string;
  highlightedAnnotation?: BacktestChartAnnotation;
  stateBands?: CandleChartStateBand[];
  onSelectionChange?: (selection: CandleChartSelection | undefined) => void;
  onExtremumHoverChange?: (extremum: BacktestExtremumTrace | undefined) => void;
  onOracleHoverChange?: (point: BacktestOraclePoint | undefined) => void;
  onViewportChange?: (viewport: CandleChartViewport) => void;
  onCursorTimeChange?: (time: number | undefined) => void;
}

export interface CandleChartStateBandPoint {
  time: number;
  state: BacktestOraclePoint["state"];
  exposure?: number;
}

export interface CandleChartStateBand {
  label: string;
  points: CandleChartStateBandPoint[];
}

export type CandleChartPriceDisplay = "candles" | "line";

export interface CandleChartOverlayVisibility {
  averages: boolean;
  signals: boolean;
  orders: boolean;
  fills: boolean;
  positions: boolean;
  extrema: boolean;
  oracle: boolean;
}

export interface CandleChartViewport {
  start: number;
  end: number;
}

export interface CandleChartSelection {
  time: number;
  candle: Candle;
  annotations: BacktestChartAnnotation[];
  extremum?: BacktestExtremumTrace;
  oracle?: BacktestOraclePoint;
}

const MIN_INTERACTIVE_CANDLES = 12;
const WHEEL_ZOOM_FACTOR = 0.18;
const MAX_BACKGROUND_ANNOTATION_MARKERS = 360;
const CANDIDATE_SIGNAL_COLOR = "#a78bfa";
export const CANDLE_CHART_PLOT_LEFT = 18;
export const CANDLE_CHART_PLOT_RIGHT = 84;

export function CandleChart(props: CandleChartProps) {
  let canvas!: HTMLCanvasElement;
  let observer: ResizeObserver | undefined;
  let lastSeriesKey = "";
  let dragState:
    | {
        pointerId: number;
        startX: number;
        viewport: CandleChartViewport;
        timeRange?: { start: number; end: number };
      }
    | undefined;
  const [viewport, setViewport] = createSignal<CandleChartViewport>();
  const [isDragging, setIsDragging] = createSignal(false);
  const [internalSelectedTime, setInternalSelectedTime] = createSignal<number>();
  const [hoveredExtremumId, setHoveredExtremumId] = createSignal<string>();
  const [hoveredOracle, setHoveredOracle] = createSignal<BacktestOraclePoint>();

  const draw = () => {
    if (!canvas) {
      return;
    }

    const parent = canvas.parentElement;
    const width = Math.max(320, parent?.clientWidth ?? canvas.clientWidth);
    const height = Math.max(320, parent?.clientHeight ?? 360);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      return;
    }

    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#101217";
    ctx.fillRect(0, 0, width, height);

    const chartViewport = currentViewport();
    const sourceCandles = props.candles;
    const candles = sourceCandles.slice(chartViewport.start, chartViewport.end);
    if (candles.length < 2) {
      drawEmpty(ctx, width, height, props.emptyLabel);
      return;
    }

    const openOrders = (overlay("orders") ? props.orders : []).filter(
      (order) => order.status === "open" && order.price !== null,
    );
    const showLastPrice = props.lastPrice > 0 && chartViewport.end >= sourceCandles.length;
    const startTime = candles[0]?.openTime ?? 0;
    const endTime = candles.at(-1)?.closeTime ?? startTime + 1;
    const smaSeries = (overlay("averages") ? props.smaSeries ?? [] : [])
      .map((series) => ({
        ...series,
        points: series.points.filter(
          (point) => point.time >= startTime && point.time <= endTime,
        ),
      }))
      .filter((series) => series.points.length > 0);
    const annotations = (props.annotations ?? []).filter(
      (annotation) => annotation.time >= startTime && annotation.time <= endTime,
    ).filter(annotationVisible);
    const extrema = (overlay("extrema") ? props.trace?.extrema ?? [] : []).filter(
      (item) => item.time >= startTime && item.time <= endTime,
    );
    const signals = (overlay("signals") ? props.trace?.signals ?? [] : []).filter(
      (item) => item.time >= startTime && item.time <= endTime,
    );
    const oracle = oraclePointsInRange(
      overlay("oracle") ? props.trace?.oracle.points ?? [] : [],
      startTime,
      endTime,
    );
    const hoveredExtremum = extrema.find((item) => item.id === hoveredExtremumId());
    const selectedCandle = findCandleForTime(candles, crosshairTime());
    const selectedAnnotations = selectedCandle
      ? annotationsForCandle(annotations, selectedCandle)
      : [];
    const values = candles.flatMap((candle) => [candle.high, candle.low]);
    if (showLastPrice) {
      values.push(props.lastPrice);
    }
    values.push(...openOrders.map((order) => order.price as number));
    values.push(...smaSeries.flatMap((series) => series.points.map((point) => point.value)));
    values.push(...selectedAnnotations.map((annotation) => annotation.price));

    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = Math.max((max - min) * 0.08, max * 0.0005);
    const priceMin = min - padding;
    const priceMax = max + padding;
    const canvasPlot = getPlotBounds(width, height);
    const stateBands = (props.stateBands ?? []).filter((band) => band.points.length > 0);
    const stateBandAreaHeight = stateBands.length > 0 ? stateBands.length * 18 + 2 : 0;
    const plot = {
      ...canvasPlot,
      bottom: Math.max(canvasPlot.top + 40, canvasPlot.bottom - stateBandAreaHeight),
    };
    const plotWidth = plot.right - plot.left;
    const plotHeight = plot.bottom - plot.top;
    const priceToY = (price: number) =>
      plot.top + ((priceMax - price) / (priceMax - priceMin)) * plotHeight;
    const timeToX = (time: number) =>
      plot.left + ((time - startTime) / Math.max(1, endTime - startTime)) * plotWidth;

    drawGrid(ctx, width, height, plot, priceMin, priceMax, priceToY);
    drawSignalActivity(ctx, signals, plot, timeToX);
    if (stateBands.length === 0) drawOracleBands(ctx, oracle, plot, timeToX);
    for (const series of smaSeries) {
      drawLineSeries(ctx, series, plot, priceToY, timeToX);
    }

    if ((props.priceDisplay ?? "candles") === "line") {
      drawContinuousPriceLine(ctx, candles, plot, priceToY, timeToX);
    } else {
      candles.forEach((candle) => {
        const openX = clamp(timeToX(candle.openTime), plot.left, plot.right);
        const closeX = clamp(timeToX(candle.closeTime), plot.left, plot.right);
        const x = (openX + closeX) / 2;
        const slotWidth =
          closeX > openX
            ? closeX - openX
            : plotWidth / Math.max(1, candles.length);
        const bodyWidth = Math.max(1, Math.min(12, slotWidth * 0.68));
        const openY = priceToY(candle.open);
        const closeY = priceToY(candle.close);
        const highY = priceToY(candle.high);
        const lowY = priceToY(candle.low);
        const rising = candle.close >= candle.open;
        const color = rising ? "#22c55e" : "#f05252";

        ctx.strokeStyle = color;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, highY);
        ctx.lineTo(x, lowY);
        ctx.stroke();

        ctx.fillStyle = color;
        const bodyTop = Math.min(openY, closeY);
        const bodyHeight = Math.max(2, Math.abs(openY - closeY));
        ctx.fillRect(x - bodyWidth / 2, bodyTop, bodyWidth, bodyHeight);
      });
    }

    if (showLastPrice) {
      drawPriceLine(ctx, plot, priceToY(props.lastPrice), props.lastPrice, "#38bdf8");
    }

    for (const order of openOrders) {
      drawPriceLine(
        ctx,
        plot,
        priceToY(order.price as number),
        order.price as number,
        order.side === "buy" ? "#22c55e" : "#f5b84b",
        order.side.toUpperCase(),
      );
    }

    drawAnnotationMarkers(ctx, annotations, selectedCandle, plot, priceToY, timeToX);
    drawOracleTransitions(
      ctx,
      oracle,
      hoveredOracle(),
      props.highlightedAnnotation,
      plot,
      priceToY,
      timeToX,
    );
    if (overlay("positions")) {
      drawHighlightedPosition(ctx, props.trace, props.highlightedPositionId, plot, priceToY, timeToX);
    }
    drawExtremaMarkers(ctx, extrema, hoveredExtremum, plot, priceToY, timeToX);
    if (hoveredExtremum) drawExtremumError(ctx, hoveredExtremum, plot, priceToY, timeToX);
    drawAnnotations(ctx, selectedAnnotations, plot, priceToY, timeToX);
    if (stateBands.length > 0) {
      drawStateBands(ctx, stateBands, {
        left: canvasPlot.left,
        right: canvasPlot.right,
        top: plot.bottom + 2,
        bottom: canvasPlot.bottom,
      }, startTime, endTime);
    }
    drawTimeLabels(ctx, candles, canvasPlot, timeToX);
    if (selectedCandle) {
      drawSelectedCandle(
        ctx,
        selectedCandle,
        crosshairTime(),
        plot,
        priceToY,
        timeToX,
        smaSeries,
        stateBands,
        canvasPlot.bottom,
      );
    }
  };

  createEffect(() => {
    const candles = props.candles;
    const first = candles[0]?.openTime ?? 0;
    const last = candles.at(-1)?.closeTime ?? 0;
    const seriesKey = `${candles.length}:${first}:${last}:${props.maxCandles ?? ""}:${
      props.interactive ? 1 : 0
    }:${props.minInteractiveCandles ?? ""}`;
    if (seriesKey !== lastSeriesKey) {
      lastSeriesKey = seriesKey;
      if (dragState && canvas?.hasPointerCapture(dragState.pointerId)) {
        canvas.releasePointerCapture(dragState.pointerId);
      }
      dragState = undefined;
      setIsDragging(false);
      const controlled = props.viewport;
      const nextViewport = controlled
        ? normalizeViewport(controlled, candles.length, minimumInteractiveCandles())
        : defaultViewport(candles.length);
      setViewport(nextViewport);
      if (!controlled) props.onViewportChange?.(nextViewport);
      setInternalSelectedTime(undefined);
    }
  });

  createEffect(() => {
    const candles = props.candles;
    const orders = props.orders;
    const smaSeries = props.smaSeries;
    const annotations = props.annotations;
    const highlightedAnnotation = props.highlightedAnnotation;
    const trace = props.trace;
    const first = candles[0]?.openTime;
    const last = candles.at(-1)?.closeTime;
    candles.length;
    first;
    last;
    orders.length;
    props.lastPrice;
    smaSeries?.length;
    annotations?.length;
    highlightedAnnotation?.time;
    highlightedAnnotation?.price;
    props.stateBands?.length;
    trace?.positions.length;
    trace?.orders.length;
    trace?.signals.length;
    trace?.extrema.length;
    trace?.oracle.points.length;
    props.overlays?.averages;
    props.overlays?.signals;
    props.overlays?.orders;
    props.overlays?.fills;
    props.overlays?.positions;
    props.overlays?.extrema;
    props.overlays?.oracle;
    props.highlightedPositionId;
    props.selectedTime;
    props.cursorTime;
    props.viewport?.start;
    props.viewport?.end;
    props.priceDisplay;
    internalSelectedTime();
    hoveredExtremumId();
    hoveredOracle();
    viewport();
    draw();
  });

  onMount(() => {
    observer = new ResizeObserver(draw);
    observer.observe(canvas.parentElement ?? canvas);
    draw();
  });

  onCleanup(() => {
    observer?.disconnect();
    dragState = undefined;
  });

  const handleWheel = (event: WheelEvent) => {
    if (!props.interactive || props.candles.length < 2) {
      return;
    }

    event.preventDefault();
    canvas.focus();

    if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      panByPixels(event.deltaX || event.deltaY);
      updateSelectionFromPointer(event);
      return;
    }

    const plot = getPlotBounds(canvas.clientWidth, canvas.clientHeight);
    const anchor = clamp((event.offsetX - plot.left) / Math.max(1, plot.right - plot.left), 0, 1);
    zoomBy(Math.exp(Math.sign(event.deltaY) * WHEEL_ZOOM_FACTOR), anchor);
    updateSelectionFromPointer(event);
  };

  const handlePointerDown = (event: PointerEvent) => {
    if (!props.interactive || event.button !== 0 || props.candles.length < 2) {
      return;
    }

    event.preventDefault();
    canvas.focus();
    updateSelectionFromPointer(event);
    dragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      viewport: currentViewport(),
      timeRange: props.timeNavigation ? viewportTimeRange(currentViewport()) : undefined,
    };
    setIsDragging(true);
    canvas.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: PointerEvent) => {
    if (!dragState || event.pointerId !== dragState.pointerId) {
      if (props.interactive) {
        updateSelectionFromPointer(event);
      }
      return;
    }

    event.preventDefault();
    const plot = getPlotBounds(canvas.clientWidth, canvas.clientHeight);
    if (props.timeNavigation && dragState.timeRange) {
      const duration = dragState.timeRange.end - dragState.timeRange.start;
      const shift = (event.clientX - dragState.startX)
        / Math.max(1, plot.right - plot.left) * duration;
      commitTimeRange(dragState.timeRange.start - shift, dragState.timeRange.end - shift);
      return;
    }
    const visible = dragState.viewport.end - dragState.viewport.start;
    const candleWidth = Math.max(1, (plot.right - plot.left) / Math.max(1, visible));
    const shift = Math.round((event.clientX - dragState.startX) / candleWidth);
    const start = dragState.viewport.start - shift;
    commitViewport({ start, end: start + visible });
  };

  const handlePointerUp = (event: PointerEvent) => {
    if (!dragState || event.pointerId !== dragState.pointerId) {
      return;
    }

    if (canvas.hasPointerCapture(event.pointerId)) {
      canvas.releasePointerCapture(event.pointerId);
    }
    dragState = undefined;
    setIsDragging(false);
  };

  const handleKeyDown = (event: KeyboardEvent) => {
    if (!props.interactive || props.candles.length < 2) {
      return;
    }

    if (event.key === "+" || event.key === "=") {
      event.preventDefault();
      zoomBy(0.8);
    } else if (event.key === "-") {
      event.preventDefault();
      zoomBy(1.25);
    } else if (event.key === "ArrowLeft") {
      event.preventDefault();
      const plot = getPlotBounds(canvas.clientWidth, canvas.clientHeight);
      panByPixels(-(plot.right - plot.left) * 0.12);
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      const plot = getPlotBounds(canvas.clientWidth, canvas.clientHeight);
      panByPixels((plot.right - plot.left) * 0.12);
    } else if (event.key === "Home" || event.key === "0") {
      event.preventDefault();
      resetViewport();
    }
  };

  const resetViewport = () => commitViewport(defaultViewport(props.candles.length));

  const zoomBy = (scale: number, anchor = 0.5) => {
    const total = props.candles.length;
    if (total < 2) {
      return;
    }

    const current = currentViewport();
    if (props.timeNavigation) {
      const range = viewportTimeRange(current);
      const duration = range.end - range.start;
      const anchorTime = range.start + duration * clamp(anchor, 0, 1);
      const targetDuration = Math.max(1, Math.round(duration * scale));
      commitTimeRange(
        anchorTime - targetDuration * clamp(anchor, 0, 1),
        anchorTime + targetDuration * (1 - clamp(anchor, 0, 1)),
      );
      return;
    }
    const visible = current.end - current.start;
    const minVisible = Math.min(total, minimumInteractiveCandles());
    const targetVisible = clamp(Math.round(visible * scale), minVisible, total);
    if (targetVisible === visible) {
      return;
    }

    const anchorIndex = current.start + visible * clamp(anchor, 0, 1);
    const start = Math.round(anchorIndex - targetVisible * clamp(anchor, 0, 1));
    commitViewport({ start, end: start + targetVisible });
  };

  const panByPixels = (deltaPixels: number) => {
    const current = currentViewport();
    const plot = getPlotBounds(canvas.clientWidth, canvas.clientHeight);
    if (props.timeNavigation) {
      const range = viewportTimeRange(current);
      const shift = deltaPixels / Math.max(1, plot.right - plot.left) * (range.end - range.start);
      commitTimeRange(range.start + shift, range.end + shift);
      return;
    }
    const visible = current.end - current.start;
    const candleWidth = Math.max(1, (plot.right - plot.left) / Math.max(1, visible));
    panByCandles(Math.round(deltaPixels / candleWidth));
  };

  const panByCandles = (delta: number) => {
    if (delta === 0) {
      return;
    }

    const current = currentViewport();
    commitViewport({
      start: current.start + delta,
      end: current.end + delta,
    });
  };

  const commitViewport = (next: CandleChartViewport) => {
    const normalized = normalizeViewport(next, props.candles.length, minimumInteractiveCandles());
    setViewport(normalized);
    props.onViewportChange?.(normalized);
  };

  const commitTimeRange = (start: number, end: number) => {
    const first = props.candles[0];
    const last = props.candles.at(-1);
    if (!first || !last) return;
    const lower = first.openTime;
    const upper = last.closeTime + 1;
    const duration = Math.min(upper - lower, Math.max(1, Math.round(end - start)));
    const normalizedStart = clamp(Math.round(start), lower, upper - duration);
    commitViewport(viewportForTimeRange(normalizedStart, normalizedStart + duration));
  };

  const selectedTime = () => props.selectedTime ?? internalSelectedTime();
  const crosshairTime = () => props.cursorTime ?? selectedTime();

  const updateSelectionFromPointer = (event: { offsetX: number; offsetY?: number }) => {
    const selection = selectionAtOffset(event.offsetX, event.offsetY);
    props.onCursorTimeChange?.(selection?.time);
    if (!selection) {
      return;
    }
    setHoveredExtremumId(selection.extremum?.id);
    setHoveredOracle(selection.oracle);
    props.onExtremumHoverChange?.(selection.extremum);
    props.onOracleHoverChange?.(selection.oracle);
    if (selection.time === selectedTime()) {
      if (selection.extremum || selection.oracle) props.onSelectionChange?.(selection);
      return;
    }
    setInternalSelectedTime(selection.time);
    props.onSelectionChange?.(selection);
  };

  const selectionAtOffset = (offsetX: number, offsetY?: number): CandleChartSelection | undefined => {
    const chartViewport = currentViewport();
    const candles = props.candles.slice(chartViewport.start, chartViewport.end);
    if (candles.length === 0) {
      return undefined;
    }

    const plot = getPlotBounds(canvas.clientWidth, canvas.clientHeight);
    const startTime = candles[0]?.openTime ?? 0;
    const endTime = candles.at(-1)?.closeTime ?? startTime + 1;
    const normalized = clamp((offsetX - plot.left) / Math.max(1, plot.right - plot.left), 0, 1);
    const time = startTime + normalized * Math.max(1, endTime - startTime);
    const candle = findCandleForTime(candles, time);
    if (!candle) {
      return undefined;
    }

    const extremum = extremumAtOffset(
      overlay("extrema") ? props.trace?.extrema ?? [] : [],
      candles,
      plot,
      offsetX,
      offsetY,
    );
    const oracle = oracleAtOffset(
      overlay("oracle") ? props.trace?.oracle.points ?? [] : [],
      candles,
      plot,
      offsetX,
      offsetY,
    );
    return {
      time: oracle?.time ?? extremum?.time ?? candle.closeTime,
      candle,
      annotations: annotationsForCandle((props.annotations ?? []).filter(annotationVisible), candle),
      extremum,
      oracle,
    };
  };

  const currentViewport = () =>
    props.interactive
      ? normalizeViewport(
          props.viewport ?? viewport() ?? defaultViewport(props.candles.length),
          props.candles.length,
          minimumInteractiveCandles(),
        )
      : defaultViewport(props.candles.length);

  const viewportTimeRange = (value: CandleChartViewport): { start: number; end: number } => ({
    start: props.candles[value.start]?.openTime ?? props.candles[0]?.openTime ?? 0,
    end: (props.candles[Math.max(value.start, value.end - 1)]?.closeTime
      ?? props.candles.at(-1)?.closeTime
      ?? 0) + 1,
  });

  const viewportForTimeRange = (startTime: number, endTime: number): CandleChartViewport => {
    const candles = props.candles;
    let low = 0;
    let high = candles.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (candles[middle]!.closeTime < startTime) low = middle + 1;
      else high = middle;
    }
    const start = low;
    low = start;
    high = candles.length;
    while (low < high) {
      const middle = (low + high) >>> 1;
      if (candles[middle]!.openTime < endTime) low = middle + 1;
      else high = middle;
    }
    return normalizeViewport(
      { start, end: Math.max(start + 1, low) },
      candles.length,
      minimumInteractiveCandles(),
    );
  };

  const overlay = (key: keyof CandleChartOverlayVisibility) => props.overlays?.[key] ?? true;
  const annotationVisible = (annotation: BacktestChartAnnotation) => annotation.kind.endsWith("signal")
    ? overlay("signals")
    : annotation.kind.endsWith("order")
      ? overlay("orders")
      : overlay("fills");

  const minimumInteractiveCandles = () => Math.max(
    2,
    Math.round(props.minInteractiveCandles ?? MIN_INTERACTIVE_CANDLES),
  );

  const defaultViewport = (total: number): CandleChartViewport => {
    const maxCandles = props.maxCandles ?? 140;
    if (maxCandles > 0) {
      return normalizeViewport(
        { start: total - maxCandles, end: total },
        total,
        props.interactive ? minimumInteractiveCandles() : 1,
      );
    }

    return normalizeViewport(
      { start: 0, end: total },
      total,
      props.interactive ? minimumInteractiveCandles() : 1,
    );
  };

  return (
    <canvas
      ref={canvas}
      aria-label={props.interactive ? "Interactive candle chart" : undefined}
      class="h-full min-h-80 w-full rounded-2 outline-none focus-visible:ring-2 focus-visible:ring-accent/45"
      classList={{
        "cursor-grab": Boolean(props.interactive) && !isDragging(),
        "cursor-grabbing": isDragging(),
      }}
      tabIndex={props.interactive ? 0 : undefined}
      title={props.interactive ? "Drag to pan. Wheel to zoom. Hover extrema for order error or oracle markers for exact actions. Double-click to reset." : undefined}
      style={{ "touch-action": props.interactive ? "none" : "auto" }}
      onDblClick={props.interactive ? resetViewport : undefined}
      onKeyDown={handleKeyDown}
      onPointerCancel={handlePointerUp}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerLeave={() => {
        setHoveredExtremumId(undefined);
        setHoveredOracle(undefined);
        props.onExtremumHoverChange?.(undefined);
        props.onOracleHoverChange?.(undefined);
        if (!dragState) {
          props.onCursorTimeChange?.(undefined);
          if (props.onCursorTimeChange) setInternalSelectedTime(undefined);
        }
      }}
      onPointerUp={handlePointerUp}
      onWheel={handleWheel}
    />
  );
}

function getPlotBounds(width: number, height: number): {
  left: number;
  right: number;
  top: number;
  bottom: number;
} {
  return {
    left: CANDLE_CHART_PLOT_LEFT,
    right: width - CANDLE_CHART_PLOT_RIGHT,
    top: 18,
    bottom: height - 34,
  };
}

function normalizeViewport(
  viewport: CandleChartViewport,
  total: number,
  minCandles: number,
): CandleChartViewport {
  if (total <= 0) {
    return { start: 0, end: 0 };
  }

  const minVisible = Math.min(total, Math.max(1, minCandles));
  const visible = clamp(
    Math.round(viewport.end - viewport.start),
    minVisible,
    Math.max(minVisible, total),
  );
  const start = clamp(Math.round(viewport.start), 0, Math.max(0, total - visible));

  return {
    start,
    end: Math.min(total, start + visible),
  };
}

function drawEmpty(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  label = "Waiting for Binance candle data",
): void {
  ctx.fillStyle = "#aeb6c8";
  ctx.font = "14px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText(label, width / 2, height / 2);
}

function drawGrid(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  plot: { left: number; right: number; top: number; bottom: number },
  priceMin: number,
  priceMax: number,
  priceToY: (price: number) => number,
): void {
  ctx.strokeStyle = "#242833";
  ctx.lineWidth = 1;
  ctx.font = "12px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "middle";

  for (let i = 0; i <= 4; i += 1) {
    const price = priceMin + ((priceMax - priceMin) * i) / 4;
    const y = priceToY(price);
    ctx.beginPath();
    ctx.moveTo(plot.left, y);
    ctx.lineTo(plot.right, y);
    ctx.stroke();
    ctx.fillStyle = "#aeb6c8";
    ctx.fillText(formatQuote(price, 2), plot.right + 10, y);
  }

  ctx.strokeStyle = "#2b303b";
  ctx.strokeRect(plot.left, plot.top, plot.right - plot.left, plot.bottom - plot.top);
  ctx.fillStyle = "#101217";
  ctx.fillRect(0, height - 26, width, 26);
}

function drawPriceLine(
  ctx: CanvasRenderingContext2D,
  plot: { left: number; right: number },
  y: number,
  price: number,
  color: string,
  label = "LAST",
): void {
  ctx.strokeStyle = color;
  ctx.setLineDash([4, 4]);
  ctx.beginPath();
  ctx.moveTo(plot.left, y);
  ctx.lineTo(plot.right, y);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = color;
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "right";
  ctx.textBaseline = "bottom";
  ctx.fillText(`${label} ${formatQuote(price, 2)}`, plot.right - 6, y - 4);
}

function drawLineSeries(
  ctx: CanvasRenderingContext2D,
  series: BacktestChartSmaSeries,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  if (series.points.length < 2) {
    return;
  }

  ctx.save();
  ctx.strokeStyle = series.color;
  ctx.lineWidth = 1.5;
  ctx.globalAlpha = 0.86;
  ctx.beginPath();
  series.points.forEach((point, index) => {
    const x = clamp(timeToX(point.time), plot.left, plot.right);
    const y = clamp(priceToY(point.value), plot.top, plot.bottom);
    if (index === 0) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  });
  ctx.stroke();
  ctx.globalAlpha = 1;
  const last = series.points.at(-1);
  if (last) {
    ctx.fillStyle = series.color;
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillText(
      series.label,
      plot.right - 6,
      clamp(priceToY(last.value), plot.top + 8, plot.bottom - 8),
    );
  }
  ctx.restore();
}

function drawContinuousPriceLine(
  ctx: CanvasRenderingContext2D,
  candles: Candle[],
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  if (candles.length === 0) {
    return;
  }

  ctx.save();
  ctx.strokeStyle = "#38bdf8";
  ctx.lineWidth = 2;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.beginPath();

  candles.forEach((candle, index) => {
    const openX = clamp(timeToX(candle.openTime), plot.left, plot.right);
    const openY = clamp(priceToY(candle.open), plot.top, plot.bottom);
    const closeX = clamp(timeToX(candle.closeTime), plot.left, plot.right);
    const closeY = clamp(priceToY(candle.close), plot.top, plot.bottom);

    if (index === 0) {
      ctx.moveTo(openX, openY);
    } else {
      ctx.lineTo(openX, openY);
    }
    ctx.lineTo(closeX, closeY);
  });

  ctx.globalAlpha = 0.24;
  ctx.lineWidth = 6;
  ctx.stroke();
  ctx.globalAlpha = 1;
  ctx.lineWidth = 2;
  ctx.stroke();
  ctx.restore();
}

function drawSignalActivity(
  ctx: CanvasRenderingContext2D,
  signals: NonNullable<BacktestTrace["signals"]>,
  plot: { left: number; right: number; top: number; bottom: number },
  timeToX: (time: number) => number,
): void {
  ctx.save();
  for (const signal of signals) {
    const x = clamp(timeToX(signal.time), plot.left, plot.right);
    const activeSignals = signal.active.length > 0
      ? signal.active
      : signal.gates
          .filter((gate) => gate.passed && (gate.code.includes(".confirmation.") || gate.code.includes(".source.")))
          .slice(0, 2)
          .map((gate) => ({
            side: gate.code.startsWith("long.") ? "long" as const : "short" as const,
            type: gate.code.includes(".exit.") ? "exit" as const : "entry" as const,
          }));
    for (const [index, active] of activeSignals.entries()) {
      const buy = active.side === "long" === (active.type === "entry");
      ctx.fillStyle = buy ? "#22c55e" : "#f05252";
      ctx.globalAlpha = signal.active.length > 0 ? (active.type === "entry" ? 0.11 : 0.06) : 0.035;
      ctx.fillRect(x - 3 + index * 3, plot.top, 5, plot.bottom - plot.top);
    }
  }
  ctx.restore();
}

function drawOracleBands(
  ctx: CanvasRenderingContext2D,
  points: BacktestTrace["oracle"]["points"],
  plot: { left: number; right: number; top: number; bottom: number },
  timeToX: (time: number) => number,
): void {
  if (points.length < 2) return;
  const top = plot.bottom - 9;
  const width = Math.max(1, Math.floor(plot.right - plot.left));
  ctx.save();
  ctx.fillStyle = "#090a0d";
  ctx.fillRect(plot.left, top - 1, plot.right - plot.left, 9);
  ctx.globalAlpha = 0.82;
  let state = oracleStateAtX(points, plot.left + 0.5, timeToX);
  let runStart = 0;
  for (let pixel = 1; pixel < width; pixel += 1) {
    const next = oracleStateAtX(points, plot.left + pixel + 0.5, timeToX);
    if (next === state) continue;
    ctx.fillStyle = oracleStateColor(state);
    ctx.fillRect(plot.left + runStart, top, pixel - runStart, 7);
    state = next;
    runStart = pixel;
  }
  ctx.fillStyle = oracleStateColor(state);
  ctx.fillRect(plot.left + runStart, top, width - runStart, 7);
  ctx.restore();
}

function drawStateBands(
  ctx: CanvasRenderingContext2D,
  bands: CandleChartStateBand[],
  bounds: { left: number; right: number; top: number; bottom: number },
  startTime: number,
  endTime: number,
): void {
  if (bands.length === 0 || endTime <= startTime) return;
  const width = Math.max(1, Math.floor(bounds.right - bounds.left));
  const rowHeight = Math.max(1, (bounds.bottom - bounds.top) / bands.length);
  ctx.save();
  bands.forEach((band, row) => {
    const top = bounds.top + row * rowHeight;
    const height = Math.max(1, rowHeight - 2);
    ctx.fillStyle = "#090a0d";
    ctx.fillRect(bounds.left, top, width, height);
    const segments = stateBandSegments(band.points, startTime, endTime);
    segments.forEach((segment, index) => {
      const rawLeft = bounds.left + (segment.start - startTime) / (endTime - startTime) * width;
      const rawRight = bounds.left + (segment.end - startTime) / (endTime - startTime) * width;
      const left = Math.floor(rawLeft);
      const right = Math.ceil(rawRight);
      ctx.globalAlpha = 1;
      ctx.fillStyle = stateBandColor(segment.state);
      ctx.fillRect(left, top, Math.max(1, right - left), height);
      if (index > 0) {
        ctx.strokeStyle = "#090a0d";
        ctx.globalAlpha = 0.72;
        ctx.beginPath();
        ctx.moveTo(left + 0.5, top);
        ctx.lineTo(left + 0.5, top + height);
        ctx.stroke();
      }
      drawStateBandRunLabel(ctx, segment.state, rawLeft, rawRight, top, height);
    });
    if (segments.length === 0) {
      ctx.globalAlpha = 1;
      ctx.fillStyle = stateBandColor("flat");
      ctx.fillRect(bounds.left, top, width, height);
    }
    ctx.globalAlpha = 1;
    ctx.fillStyle = "#aeb6c8";
    ctx.font = "600 10px Inter, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(band.label, bounds.right + 7, top + height / 2);
  });
  ctx.restore();
}

function stateBandSegments(
  points: CandleChartStateBandPoint[],
  startTime: number,
  endTime: number,
): Array<{ state: BacktestOraclePoint["state"]; start: number; end: number }> {
  if (endTime <= startTime) return [];
  const initial = stateBandPointAtTime(points, startTime);
  let state = initial?.state ?? "flat";
  let cursor = startTime;
  const segments: Array<{ state: BacktestOraclePoint["state"]; start: number; end: number }> = [];
  for (const point of points) {
    if (point.time <= startTime) continue;
    if (point.time >= endTime) break;
    if (point.state === state) continue;
    segments.push({ state, start: cursor, end: point.time });
    cursor = point.time;
    state = point.state;
  }
  segments.push({ state, start: cursor, end: endTime });
  return segments;
}

function drawStateBandRunLabel(
  ctx: CanvasRenderingContext2D,
  state: BacktestOraclePoint["state"],
  left: number,
  right: number,
  top: number,
  height: number,
): void {
  if (right - left < 26) return;
  ctx.globalAlpha = 1;
  ctx.fillStyle = state === "flat" ? "#e5e7eb" : "#071018";
  ctx.font = "600 9px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(state === "long" ? "L" : state === "short" ? "S" : "F", (left + right) / 2, top + height / 2);
}

function stateBandPointAtTime(
  points: CandleChartStateBandPoint[],
  time: number,
): CandleChartStateBandPoint | undefined {
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if ((points[middle]?.time ?? Number.POSITIVE_INFINITY) <= time) low = middle + 1;
    else high = middle;
  }
  return low > 0 ? points[low - 1] : undefined;
}

function stateBandColor(state: BacktestOraclePoint["state"]): string {
  return state === "long" ? "#22c55e" : state === "short" ? "#f05252" : "#4b5563";
}

function drawOracleTransitions(
  ctx: CanvasRenderingContext2D,
  points: BacktestTrace["oracle"]["points"],
  hovered: BacktestOraclePoint | undefined,
  matched: BacktestChartAnnotation | undefined,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  const transitions = points.filter((point) => point.action !== "hold");
  if (transitions.length === 0) return;
  const width = Math.max(1, plot.right - plot.left);
  const dense = transitions.length > width / 7;
  const renderEvery = dense ? Math.max(1, Math.ceil(transitions.length / (width * 2))) : 1;
  const occupied = new Set<string>();
  ctx.save();
  for (let index = 0; index < transitions.length; index += renderEvery) {
    const point = transitions[index];
    if (!point) continue;
    const x = timeToX(point.time);
    if (x < plot.left || x > plot.right) continue;
    const y = clamp(priceToY(point.price), plot.top, plot.bottom);
    const highlighted = point.time === hovered?.time && point.state === hovered.state;
    if (dense && !highlighted) {
      const key = `${Math.round(x)}:${point.state}`;
      if (occupied.has(key)) continue;
      occupied.add(key);
      ctx.globalAlpha = 0.76;
      ctx.strokeStyle = oracleStateColor(point.state);
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x, y - 5);
      ctx.lineTo(x, y + 5);
      ctx.stroke();
      continue;
    }
    drawOracleMarker(ctx, point, x, y, highlighted, !dense);
  }
  if (hovered) {
    const x = timeToX(hovered.time);
    if (x >= plot.left && x <= plot.right) {
      const y = clamp(priceToY(hovered.price), plot.top, plot.bottom);
      drawOracleMarker(ctx, hovered, x, y, true, true);
      if (matched) drawMatchedCandidate(ctx, matched, x, y, plot, priceToY, timeToX);
      drawOracleTooltip(ctx, hovered, matched, x, y, plot);
    }
  }
  ctx.restore();
}

function oracleStateAtX(
  points: BacktestOraclePoint[],
  x: number,
  timeToX: (time: number) => number,
): BacktestOraclePoint["state"] {
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (timeToX(points[middle]?.time ?? Number.POSITIVE_INFINITY) <= x) low = middle + 1;
    else high = middle;
  }
  return points[Math.max(0, low - 1)]?.state ?? "flat";
}

function drawOracleMarker(
  ctx: CanvasRenderingContext2D,
  point: BacktestOraclePoint,
  x: number,
  y: number,
  highlighted: boolean,
  showLabel: boolean,
): void {
  const radius = highlighted ? 7 : 5;
  const color = oracleStateColor(point.state);
  ctx.save();
  ctx.globalAlpha = 1;
  ctx.fillStyle = color;
  ctx.strokeStyle = highlighted ? "#f4f6fb" : "#090a0d";
  ctx.lineWidth = highlighted ? 2.5 : 1.5;
  ctx.beginPath();
  if (point.state === "long") {
    ctx.moveTo(x, y - radius);
    ctx.lineTo(x - radius, y + radius * 0.72);
    ctx.lineTo(x + radius, y + radius * 0.72);
  } else if (point.state === "short") {
    ctx.moveTo(x, y + radius);
    ctx.lineTo(x - radius, y - radius * 0.72);
    ctx.lineTo(x + radius, y - radius * 0.72);
  } else {
    ctx.moveTo(x, y - radius);
    ctx.lineTo(x + radius, y);
    ctx.lineTo(x, y + radius);
    ctx.lineTo(x - radius, y);
  }
  ctx.closePath();
  ctx.stroke();
  ctx.fill();
  if (showLabel) {
    ctx.fillStyle = color;
    ctx.font = `${highlighted ? 700 : 600} 10px Inter, sans-serif`;
    ctx.textAlign = "center";
    ctx.textBaseline = point.state === "short" ? "bottom" : "top";
    ctx.fillText(
      point.state === "long" ? "L" : point.state === "short" ? "S" : "F",
      x,
      point.state === "short" ? y - radius - 3 : y + radius + 3,
    );
  }
  ctx.restore();
}

function drawOracleTooltip(
  ctx: CanvasRenderingContext2D,
  point: BacktestOraclePoint,
  matched: BacktestChartAnnotation | undefined,
  markerX: number,
  markerY: number,
  plot: { left: number; right: number; top: number; bottom: number },
): void {
  const title = `${oracleActionLabel(point)} · ${point.fromState} → ${point.state}`;
  const detail = `$${formatQuote(point.price, 4)} · ${formatTime(point.time)}`;
  const match = matched
    ? `Matched candidate ${signedChartDuration(matched.time - point.time)}`
    : "No one-to-one candidate match";
  ctx.font = "600 11px Inter, sans-serif";
  const width = Math.max(
    ctx.measureText(title).width,
    ctx.measureText(detail).width,
    ctx.measureText(match).width,
  ) + 18;
  const height = 58;
  const left = clamp(markerX + 12, plot.left, plot.right - width);
  const top = clamp(markerY - height - 12, plot.top, plot.bottom - height);
  ctx.fillStyle = "rgba(9, 10, 13, 0.96)";
  ctx.strokeStyle = oracleStateColor(point.state);
  ctx.lineWidth = 1;
  ctx.fillRect(left, top, width, height);
  ctx.strokeRect(left, top, width, height);
  ctx.fillStyle = "#f4f6fb";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(title, left + 9, top + 7);
  ctx.fillStyle = "#aeb6c8";
  ctx.font = "11px Inter, sans-serif";
  ctx.fillText(detail, left + 9, top + 23);
  ctx.fillStyle = matched ? CANDIDATE_SIGNAL_COLOR : "#aeb6c8";
  ctx.fillText(match, left + 9, top + 39);
}

function drawMatchedCandidate(
  ctx: CanvasRenderingContext2D,
  annotation: BacktestChartAnnotation,
  oracleX: number,
  oracleY: number,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  const x = timeToX(annotation.time);
  if (x < plot.left || x > plot.right) return;
  const y = clamp(priceToY(annotation.price), plot.top, plot.bottom);
  ctx.save();
  ctx.strokeStyle = CANDIDATE_SIGNAL_COLOR;
  ctx.fillStyle = CANDIDATE_SIGNAL_COLOR;
  ctx.lineWidth = 2;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.moveTo(oracleX, oracleY);
  ctx.lineTo(x, y);
  ctx.stroke();
  ctx.setLineDash([]);
  ctx.beginPath();
  ctx.arc(x, y, 7, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = "#f4f6fb";
  ctx.stroke();
  ctx.fillStyle = "#090a0d";
  ctx.font = "700 9px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(annotation.signalState === "long" ? "L" : annotation.signalState === "short" ? "S" : "F", x, y);
  ctx.restore();
}

function signedChartDuration(ms: number): string {
  const sign = ms > 0 ? "+" : ms < 0 ? "−" : "";
  const seconds = Math.abs(ms) / 1_000;
  if (seconds < 60) return `${sign}${formatQuote(seconds, seconds < 10 ? 1 : 0)}s`;
  const minutes = seconds / 60;
  if (minutes < 60) return `${sign}${formatQuote(minutes, minutes < 10 ? 1 : 0)}m`;
  return `${sign}${formatQuote(minutes / 60, 1)}h`;
}

function oracleStateColor(state: BacktestOraclePoint["state"]): string {
  return state === "long" ? "#4ade80" : state === "short" ? "#fb7185" : "#94a3b8";
}

function oracleActionLabel(point: BacktestOraclePoint): string {
  if (point.state === "flat") return "GO FLAT";
  return point.fromState === "flat"
    ? `GO ${point.state.toUpperCase()}`
    : `SWITCH TO ${point.state.toUpperCase()}`;
}

function drawHighlightedPosition(
  ctx: CanvasRenderingContext2D,
  trace: BacktestTrace | undefined,
  positionId: string | undefined,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  if (!trace || !positionId) return;
  ctx.save();
  for (const order of trace.orders.filter((item) => item.positionId === positionId)) {
    const color = order.side === "buy" ? "#22c55e" : "#f05252";
    const price = order.price ?? order.stopPrice ?? order.fills[0]?.price;
    if (!price) continue;
    const fromX = clamp(timeToX(order.createdAt), plot.left, plot.right);
    const fromY = clamp(priceToY(price), plot.top, plot.bottom);
    ctx.strokeStyle = color;
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.82;
    ctx.setLineDash(order.grid === "entry" ? [] : [4, 3]);
    for (const fill of order.fills) {
      const toX = clamp(timeToX(fill.time), plot.left, plot.right);
      const toY = clamp(priceToY(fill.price), plot.top, plot.bottom);
      ctx.beginPath();
      ctx.moveTo(fromX, fromY);
      ctx.lineTo(toX, toY);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(toX, toY, 4, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.setLineDash([]);
    ctx.strokeRect(fromX - 3, fromY - 3, 6, 6);
  }
  ctx.restore();
}

function drawExtremaMarkers(
  ctx: CanvasRenderingContext2D,
  extrema: BacktestExtremumTrace[],
  hovered: BacktestExtremumTrace | undefined,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  ctx.save();
  for (const extremum of extrema) {
    const selected = extremum.id === hovered?.id;
    const x = clamp(timeToX(extremum.time), plot.left, plot.right);
    const y = clamp(priceToY(extremum.price), plot.top, plot.bottom);
    ctx.fillStyle = extremum.kind === "valley" ? "#38bdf8" : "#f5b84b";
    ctx.strokeStyle = selected ? "#f4f6fb" : "#090a0d";
    ctx.lineWidth = selected ? 2 : 1;
    ctx.beginPath();
    ctx.arc(x, y, selected ? 6 : 3.5, 0, Math.PI * 2);
    ctx.stroke();
    ctx.fill();
  }
  ctx.restore();
}

function drawExtremumError(
  ctx: CanvasRenderingContext2D,
  extremum: BacktestExtremumTrace,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  const centerX = clamp(timeToX(extremum.time), plot.left, plot.right);
  const centerY = clamp(priceToY(extremum.price), plot.top, plot.bottom);
  ctx.save();
  drawErrorBox(
    ctx,
    extremum.time - extremum.p99TimeDistanceMs,
    extremum.time + extremum.p99TimeDistanceMs,
    extremum.price * (1 - extremum.p99PriceDistancePct / 100),
    extremum.price * (1 + extremum.p99PriceDistancePct / 100),
    plot,
    priceToY,
    timeToX,
    "#a78bfa",
    true,
  );
  if (extremum.errorBox) {
    drawErrorBox(
      ctx,
      extremum.time + extremum.errorBox.minTimeErrorMs,
      extremum.time + extremum.errorBox.maxTimeErrorMs,
      extremum.price * (1 + extremum.errorBox.minPriceErrorPct / 100),
      extremum.price * (1 + extremum.errorBox.maxPriceErrorPct / 100),
      plot,
      priceToY,
      timeToX,
      "#38bdf8",
      false,
    );
  }
  for (const order of extremum.orders) {
    const x = clamp(timeToX(order.time), plot.left, plot.right);
    const y = clamp(priceToY(order.price), plot.top, plot.bottom);
    ctx.strokeStyle = order.withinThreshold ? "#22c55e" : "#f05252";
    ctx.globalAlpha = 0.84;
    ctx.beginPath();
    ctx.moveTo(centerX, centerY);
    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.fillStyle = ctx.strokeStyle;
    ctx.beginPath();
    ctx.arc(x, y, 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.globalAlpha = 0.95;
  ctx.fillStyle = "#d6dbea";
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillText(
    `${extremum.kind.toUpperCase()} · ${extremum.orders.length} fills · P99 error box`,
    clamp(centerX + 9, plot.left, plot.right - 190),
    clamp(centerY + 9, plot.top, plot.bottom - 16),
  );
  ctx.restore();
}

function drawErrorBox(
  ctx: CanvasRenderingContext2D,
  startTime: number,
  endTime: number,
  lowPrice: number,
  highPrice: number,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
  color: string,
  dashed: boolean,
): void {
  const left = clamp(timeToX(startTime), plot.left, plot.right);
  const right = clamp(timeToX(endTime), plot.left, plot.right);
  const top = clamp(priceToY(highPrice), plot.top, plot.bottom);
  const bottom = clamp(priceToY(lowPrice), plot.top, plot.bottom);
  ctx.strokeStyle = color;
  ctx.fillStyle = color;
  ctx.globalAlpha = dashed ? 0.62 : 0.2;
  ctx.setLineDash(dashed ? [5, 4] : []);
  ctx.strokeRect(left, top, Math.max(1, right - left), Math.max(1, bottom - top));
  if (!dashed) ctx.fillRect(left, top, Math.max(1, right - left), Math.max(1, bottom - top));
  ctx.setLineDash([]);
}

function oraclePointsInRange(
  points: BacktestOraclePoint[],
  startTime: number,
  endTime: number,
): BacktestOraclePoint[] {
  if (points.length === 0) return [];
  const start = Math.max(0, oracleLowerBound(points, startTime) - 1);
  const end = Math.min(points.length, oracleUpperBound(points, endTime) + 1);
  return points.slice(start, end);
}

function oracleAtOffset(
  points: BacktestOraclePoint[],
  candles: Candle[],
  plot: { left: number; right: number; top: number; bottom: number },
  offsetX: number,
  offsetY: number | undefined,
): BacktestOraclePoint | undefined {
  if (candles.length === 0 || points.length === 0) return undefined;
  const startTime = candles[0]?.openTime ?? 0;
  const endTime = candles.at(-1)?.closeTime ?? startTime + 1;
  const width = Math.max(1, plot.right - plot.left);
  const targetTime = startTime
    + clamp((offsetX - plot.left) / width, 0, 1) * Math.max(1, endTime - startTime);
  const timeRadius = Math.max(1, endTime - startTime) * 14 / width;
  const start = oracleLowerBound(points, Math.max(startTime, targetTime - timeRadius));
  const end = oracleUpperBound(points, Math.min(endTime, targetTime + timeRadius));
  const values = candles.flatMap((candle) => [candle.low, candle.high]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = Math.max((max - min) * 0.08, max * 0.0005);
  const low = min - padding;
  const high = max + padding;
  let nearest: BacktestOraclePoint | undefined;
  let nearestDistance = 14;
  for (let index = start; index < end; index += 1) {
    const point = points[index];
    if (!point || point.action === "hold" || point.time < startTime || point.time > endTime) continue;
    const x = plot.left + (point.time - startTime) / Math.max(1, endTime - startTime) * width;
    const y = plot.top + (high - point.price) / Math.max(Number.EPSILON, high - low) * (plot.bottom - plot.top);
    const distance = offsetY === undefined ? Math.abs(offsetX - x) : Math.hypot(offsetX - x, offsetY - y);
    if (distance <= nearestDistance) {
      nearest = point;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function oracleLowerBound(points: BacktestOraclePoint[], time: number): number {
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if ((points[middle]?.time ?? Number.POSITIVE_INFINITY) < time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function oracleUpperBound(points: BacktestOraclePoint[], time: number): number {
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if ((points[middle]?.time ?? Number.POSITIVE_INFINITY) <= time) low = middle + 1;
    else high = middle;
  }
  return low;
}

function extremumAtOffset(
  extrema: BacktestExtremumTrace[],
  candles: Candle[],
  plot: { left: number; right: number; top: number; bottom: number },
  offsetX: number,
  offsetY: number | undefined,
): BacktestExtremumTrace | undefined {
  if (candles.length === 0 || extrema.length === 0) return undefined;
  const startTime = candles[0]?.openTime ?? 0;
  const endTime = candles.at(-1)?.closeTime ?? startTime + 1;
  const visible = extrema.filter((item) => item.time >= startTime && item.time <= endTime);
  const values = candles.flatMap((candle) => [candle.low, candle.high]);
  const min = Math.min(...values);
  const max = Math.max(...values);
  const padding = Math.max((max - min) * 0.08, max * 0.0005);
  const low = min - padding;
  const high = max + padding;
  let nearest: BacktestExtremumTrace | undefined;
  let nearestDistance = 12;
  for (const extremum of visible) {
    const x = plot.left + (extremum.time - startTime) / Math.max(1, endTime - startTime) * (plot.right - plot.left);
    const y = plot.top + (high - extremum.price) / Math.max(Number.EPSILON, high - low) * (plot.bottom - plot.top);
    const distance = offsetY === undefined ? Math.abs(offsetX - x) : Math.hypot(offsetX - x, offsetY - y);
    if (distance <= nearestDistance) {
      nearest = extremum;
      nearestDistance = distance;
    }
  }
  return nearest;
}

function findCandleForTime(candles: Candle[], time: number | undefined): Candle | undefined {
  if (time === undefined || candles.length === 0) {
    return undefined;
  }

  let nearest = candles[0];
  let nearestDistance = Number.POSITIVE_INFINITY;
  for (const candle of candles) {
    if (time >= candle.openTime && time <= candle.closeTime) {
      return candle;
    }

    const distance = Math.min(
      Math.abs(time - candle.openTime),
      Math.abs(time - candle.closeTime),
    );
    if (distance < nearestDistance) {
      nearest = candle;
      nearestDistance = distance;
    }
  }

  return nearest;
}

function annotationsForCandle(
  annotations: BacktestChartAnnotation[],
  candle: Candle,
): BacktestChartAnnotation[] {
  return annotations.filter(
    (annotation) => annotation.time >= candle.openTime && annotation.time <= candle.closeTime,
  );
}

function drawAnnotationMarkers(
  ctx: CanvasRenderingContext2D,
  annotations: BacktestChartAnnotation[],
  selectedCandle: Candle | undefined,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  if (annotations.length === 0) return;
  const signals = annotations.filter((annotation) => annotation.kind.endsWith("signal"));
  const events = annotations.filter((annotation) => !annotation.kind.endsWith("signal"));
  const width = Math.max(1, plot.right - plot.left);
  const dense = signals.length > width / 7;
  const signalEvery = dense ? Math.max(1, Math.ceil(signals.length / (width * 2))) : 1;
  const occupied = new Set<string>();

  ctx.save();
  for (let index = 0; index < signals.length; index += signalEvery) {
    const annotation = signals[index];
    if (!annotation) continue;
    if (
      selectedCandle &&
      annotation.time >= selectedCandle.openTime &&
      annotation.time <= selectedCandle.closeTime
    ) continue;

    const x = clamp(timeToX(annotation.time), plot.left, plot.right);
    const y = clamp(priceToY(annotation.price), plot.top, plot.bottom);
    const isBuy = annotation.kind.startsWith("buy");
    if (dense) {
      const key = `${Math.round(x)}:${annotation.signalState ?? (isBuy ? "buy" : "sell")}`;
      if (occupied.has(key)) continue;
      occupied.add(key);
    }
    drawCandidateSignalMarker(ctx, x, y, isBuy, dense, annotation.signalState);
  }

  const eventEvery = Math.max(1, Math.ceil(events.length / MAX_BACKGROUND_ANNOTATION_MARKERS));
  for (let index = 0; index < events.length; index += eventEvery) {
    const annotation = events[index];
    if (!annotation) continue;
    if (
      selectedCandle &&
      annotation.time >= selectedCandle.openTime &&
      annotation.time <= selectedCandle.closeTime
    ) continue;

    const x = clamp(timeToX(annotation.time), plot.left, plot.right);
    const y = clamp(priceToY(annotation.price), plot.top, plot.bottom);
    const isBuy = annotation.kind.startsWith("buy");
    ctx.globalAlpha = 0.38;
    ctx.strokeStyle = isBuy ? "#22c55e" : "#f05252";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, isBuy ? y + 7 : y - 7);
    ctx.lineTo(x, isBuy ? y + 1 : y - 1);
    ctx.stroke();
  }
  ctx.restore();
}

function drawCandidateSignalMarker(
  ctx: CanvasRenderingContext2D,
  x: number,
  y: number,
  isBuy: boolean,
  dense: boolean,
  state?: BacktestChartAnnotation["signalState"],
): void {
  ctx.save();
  ctx.globalAlpha = dense ? 0.86 : 1;
  ctx.strokeStyle = dense ? CANDIDATE_SIGNAL_COLOR : "#090a0d";
  ctx.lineWidth = dense ? 4 : 1.5;
  ctx.beginPath();
  if (dense) {
    ctx.moveTo(x, y - 5);
    ctx.lineTo(x, y + 5);
  } else if (state === "flat") {
    const radius = 6;
    ctx.fillStyle = CANDIDATE_SIGNAL_COLOR;
    ctx.moveTo(x, y - radius);
    ctx.lineTo(x + radius, y);
    ctx.lineTo(x, y + radius);
    ctx.lineTo(x - radius, y);
    ctx.closePath();
  } else {
    const radius = 7;
    ctx.fillStyle = CANDIDATE_SIGNAL_COLOR;
    ctx.moveTo(x, isBuy ? y - radius : y + radius);
    ctx.lineTo(x - radius, isBuy ? y + radius * 0.72 : y - radius * 0.72);
    ctx.lineTo(x + radius, isBuy ? y + radius * 0.72 : y - radius * 0.72);
    ctx.closePath();
  }
  ctx.stroke();
  if (!dense) ctx.fill();
  if (!dense && state) {
    ctx.fillStyle = CANDIDATE_SIGNAL_COLOR;
    ctx.font = "bold 10px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = state === "long" ? "bottom" : "top";
    ctx.fillText(
      state === "long" ? "L" : state === "short" ? "S" : "F",
      x,
      state === "long" ? y - 9 : y + 9,
    );
  }
  ctx.restore();
}

function drawSelectedCandle(
  ctx: CanvasRenderingContext2D,
  candle: Candle,
  selectedTime: number | undefined,
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
  smaSeries: BacktestChartSmaSeries[],
  stateBands: CandleChartStateBand[],
  lineBottom = plot.bottom,
): void {
  const markerTime = selectedTime ?? (candle.openTime + candle.closeTime) / 2;
  const x = clamp(timeToX(markerTime), plot.left, plot.right);
  const candleX = clamp(
    timeToX((candle.openTime + candle.closeTime) / 2),
    plot.left,
    plot.right,
  );
  const highY = clamp(priceToY(candle.high), plot.top, plot.bottom);
  const lowY = clamp(priceToY(candle.low), plot.top, plot.bottom);

  ctx.save();
  ctx.strokeStyle = "#d6dbea";
  ctx.globalAlpha = 0.74;
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 5]);
  ctx.beginPath();
  ctx.moveTo(x, plot.top);
  ctx.lineTo(x, lineBottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.globalAlpha = 0.95;
  ctx.strokeStyle = "#f4f6fb";
  ctx.strokeRect(candleX - 6, highY - 4, 12, Math.max(8, lowY - highY + 8));
  const label = formatCursorTime(markerTime);
  ctx.font = "11px Inter, sans-serif";
  const labelWidth = ctx.measureText(label).width + 12;
  const labelLeft = clamp(x - labelWidth / 2, plot.left, plot.right - labelWidth);
  ctx.fillStyle = "#252a35";
  ctx.fillRect(labelLeft, lineBottom - 18, labelWidth, 18);
  ctx.fillStyle = "#eef1f8";
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(label, labelLeft + labelWidth / 2, lineBottom - 9);

  const values = [
    `O ${formatQuote(candle.open, 2)}  H ${formatQuote(candle.high, 2)}  L ${formatQuote(candle.low, 2)}  C ${formatQuote(candle.close, 2)}`,
    ...smaSeries.flatMap((series) => {
      const point = nearestTimedPoint(series.points, markerTime);
      return point ? [`${series.label} ${formatQuote(point.value, 4)}`] : [];
    }),
    ...stateBands.flatMap((band) => {
      const point = stateBandPointAtTime(band.points, markerTime);
      if (!point) return [];
      const exposure = point.exposure === undefined ? "" : ` ${formatQuote(point.exposure, 3)}`;
      return [`${band.label} ${point.state}${exposure}`];
    }),
  ];
  ctx.font = "11px Inter, sans-serif";
  const valueWidth = Math.min(
    plot.right - plot.left,
    Math.max(...values.map((value) => ctx.measureText(value).width), 0) + 16,
  );
  const valueHeight = values.length * 17 + 8;
  const valueLeft = x + 10 + valueWidth <= plot.right
    ? x + 10
    : Math.max(plot.left, x - valueWidth - 10);
  const valueTop = plot.top + 6;
  ctx.fillStyle = "rgba(22, 25, 32, 0.94)";
  ctx.fillRect(valueLeft, valueTop, valueWidth, valueHeight);
  ctx.fillStyle = "#eef1f8";
  ctx.textAlign = "left";
  values.forEach((value, index) => {
    ctx.fillText(value, valueLeft + 8, valueTop + 5 + index * 17 + 8.5, valueWidth - 16);
  });
  ctx.restore();
}

function nearestTimedPoint<T extends { time: number }>(points: readonly T[], time: number): T | undefined {
  if (points.length === 0) return undefined;
  let low = 0;
  let high = points.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (points[middle]!.time < time) low = middle + 1;
    else high = middle;
  }
  const right = points[Math.min(points.length - 1, low)]!;
  const left = points[Math.max(0, low - 1)]!;
  return Math.abs(left.time - time) <= Math.abs(right.time - time) ? left : right;
}

function formatCursorTime(time: number): string {
  return `${new Date(time).toISOString().replace("T", " ").slice(0, 19)} UTC`;
}

function drawAnnotations(
  ctx: CanvasRenderingContext2D,
  annotations: BacktestChartAnnotation[],
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  for (const annotation of annotations) {
    const x = clamp(timeToX(annotation.time), plot.left, plot.right);
    const y = clamp(priceToY(annotation.price), plot.top, plot.bottom);
    const isBuy = annotation.kind.startsWith("buy");
    const isSignal = annotation.kind.endsWith("signal");
    const color = isSignal ? CANDIDATE_SIGNAL_COLOR : isBuy ? "#22c55e" : "#f05252";
    const markerUp = annotation.signalState === "long" || (annotation.signalState === undefined && isBuy);

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = "#090a0d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (annotation.signalState === "flat") {
      ctx.moveTo(x, y - 7);
      ctx.lineTo(x + 7, y);
      ctx.lineTo(x, y + 7);
      ctx.lineTo(x - 7, y);
    } else if (markerUp) {
      ctx.moveTo(x, y - 8);
      ctx.lineTo(x - 5, y + 4);
      ctx.lineTo(x + 5, y + 4);
    } else {
      ctx.moveTo(x, y + 8);
      ctx.lineTo(x - 5, y - 4);
      ctx.lineTo(x + 5, y - 4);
    }
    ctx.closePath();
    ctx.stroke();
    ctx.fill();

    if (isSignal) {
      ctx.strokeStyle = color;
      ctx.setLineDash([2, 5]);
      ctx.beginPath();
      ctx.moveTo(x, plot.top);
      ctx.lineTo(x, plot.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    ctx.fillStyle = color;
    ctx.font = "10px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = markerUp ? "bottom" : "top";
    ctx.fillText(annotationShortLabel(annotation), x, markerUp ? y - 9 : y + 9);
    ctx.restore();
  }
}

function annotationShortLabel(annotation: BacktestChartAnnotation): string {
  if (annotation.kind.includes("signal")) {
    return annotation.signalState === "long" ? "L"
      : annotation.signalState === "short" ? "S"
        : annotation.signalState === "flat" ? "F" : "SIG";
  }
  if (annotation.kind.includes("fill")) {
    return "F";
  }
  return "O";
}

function drawTimeLabels(
  ctx: CanvasRenderingContext2D,
  candles: Candle[],
  plot: { left: number; right: number; bottom: number },
  timeToX: (time: number) => number,
): void {
  ctx.fillStyle = "#aeb6c8";
  ctx.font = "11px Inter, sans-serif";
  ctx.textBaseline = "top";

  const count = Math.min(5, candles.length);
  for (let i = 0; i < count; i += 1) {
    const index = Math.round((candles.length - 1) * (i / Math.max(1, count - 1)));
    const candle = candles[index];
    const x = clamp(timeToX(candle.openTime), plot.left, plot.right);
    ctx.textAlign = i === 0 ? "left" : i === count - 1 ? "right" : "center";
    ctx.fillText(formatTime(candle.openTime), x, plot.bottom + 8);
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

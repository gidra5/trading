import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import type {
  BacktestChartAnnotation,
  BacktestChartSmaSeries,
  Candle,
  TradingOrder,
} from "@trading/bot-algo";
import { formatQuote, formatTime } from "../format";

interface CandleChartProps {
  candles: Candle[];
  orders: TradingOrder[];
  lastPrice: number;
  smaSeries?: BacktestChartSmaSeries[];
  annotations?: BacktestChartAnnotation[];
  maxCandles?: number;
  interactive?: boolean;
  emptyLabel?: string;
  selectedTime?: number;
  viewport?: CandleChartViewport;
  priceDisplay?: CandleChartPriceDisplay;
  onSelectionChange?: (selection: CandleChartSelection | undefined) => void;
  onViewportChange?: (viewport: CandleChartViewport) => void;
}

export type CandleChartPriceDisplay = "candles" | "line";

export interface CandleChartViewport {
  start: number;
  end: number;
}

export interface CandleChartSelection {
  time: number;
  candle: Candle;
  annotations: BacktestChartAnnotation[];
}

const MIN_INTERACTIVE_CANDLES = 12;
const WHEEL_ZOOM_FACTOR = 0.18;
const MAX_BACKGROUND_ANNOTATION_MARKERS = 360;

export function CandleChart(props: CandleChartProps) {
  let canvas!: HTMLCanvasElement;
  let observer: ResizeObserver | undefined;
  let lastSeriesKey = "";
  let dragState:
    | {
        pointerId: number;
        startX: number;
        viewport: CandleChartViewport;
      }
    | undefined;
  const [viewport, setViewport] = createSignal<CandleChartViewport>();
  const [isDragging, setIsDragging] = createSignal(false);
  const [internalSelectedTime, setInternalSelectedTime] = createSignal<number>();

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

    const openOrders = props.orders.filter((order) => order.status === "open");
    const showLastPrice = props.lastPrice > 0 && chartViewport.end >= sourceCandles.length;
    const startTime = candles[0]?.openTime ?? 0;
    const endTime = candles.at(-1)?.closeTime ?? startTime + 1;
    const smaSeries = (props.smaSeries ?? [])
      .map((series) => ({
        ...series,
        points: series.points.filter(
          (point) => point.time >= startTime && point.time <= endTime,
        ),
      }))
      .filter((series) => series.points.length > 0);
    const annotations = (props.annotations ?? []).filter(
      (annotation) => annotation.time >= startTime && annotation.time <= endTime,
    );
    const selectedCandle = findCandleForTime(candles, selectedTime());
    const selectedAnnotations = selectedCandle
      ? annotationsForCandle(annotations, selectedCandle)
      : [];
    const values = candles.flatMap((candle) => [candle.high, candle.low]);
    if (showLastPrice) {
      values.push(props.lastPrice);
    }
    values.push(...openOrders.map((order) => order.price));
    values.push(...smaSeries.flatMap((series) => series.points.map((point) => point.value)));
    values.push(...selectedAnnotations.map((annotation) => annotation.price));

    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = Math.max((max - min) * 0.08, max * 0.0005);
    const priceMin = min - padding;
    const priceMax = max + padding;
    const plot = getPlotBounds(width, height);
    const plotWidth = plot.right - plot.left;
    const plotHeight = plot.bottom - plot.top;
    const priceToY = (price: number) =>
      plot.top + ((priceMax - price) / (priceMax - priceMin)) * plotHeight;
    const timeToX = (time: number) =>
      plot.left + ((time - startTime) / Math.max(1, endTime - startTime)) * plotWidth;

    drawGrid(ctx, width, height, plot, priceMin, priceMax, priceToY);
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
        priceToY(order.price),
        order.price,
        order.side === "buy" ? "#22c55e" : "#f5b84b",
        order.side.toUpperCase(),
      );
    }

    drawAnnotationMarkers(ctx, annotations, selectedCandle, plot, priceToY, timeToX);
    if (selectedCandle) {
      drawSelectedCandle(ctx, selectedCandle, selectedTime(), plot, priceToY, timeToX);
    }
    drawAnnotations(ctx, selectedAnnotations, plot, priceToY, timeToX);
    drawTimeLabels(ctx, candles, plot, timeToX);
  };

  createEffect(() => {
    const candles = props.candles;
    const first = candles[0]?.openTime ?? 0;
    const last = candles.at(-1)?.closeTime ?? 0;
    const seriesKey = `${candles.length}:${first}:${last}:${props.maxCandles ?? ""}:${
      props.interactive ? 1 : 0
    }`;
    if (seriesKey !== lastSeriesKey) {
      lastSeriesKey = seriesKey;
      const nextViewport = defaultViewport(candles.length);
      setViewport(nextViewport);
      props.onViewportChange?.(nextViewport);
      setInternalSelectedTime(undefined);
    }
  });

  createEffect(() => {
    const candles = props.candles;
    const orders = props.orders;
    const smaSeries = props.smaSeries;
    const annotations = props.annotations;
    const first = candles[0]?.openTime;
    const last = candles.at(-1)?.closeTime;
    candles.length;
    first;
    last;
    orders.length;
    props.lastPrice;
    smaSeries?.length;
    annotations?.length;
    props.selectedTime;
    props.viewport?.start;
    props.viewport?.end;
    props.priceDisplay;
    internalSelectedTime();
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
      const visible = currentViewport().end - currentViewport().start;
      panByCandles(-Math.max(1, Math.round(visible * 0.12)));
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      const visible = currentViewport().end - currentViewport().start;
      panByCandles(Math.max(1, Math.round(visible * 0.12)));
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
    const visible = current.end - current.start;
    const minVisible = Math.min(total, MIN_INTERACTIVE_CANDLES);
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
    const visible = current.end - current.start;
    const plot = getPlotBounds(canvas.clientWidth, canvas.clientHeight);
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
    const normalized = normalizeViewport(next, props.candles.length, MIN_INTERACTIVE_CANDLES);
    setViewport(normalized);
    props.onViewportChange?.(normalized);
  };

  const selectedTime = () => props.selectedTime ?? internalSelectedTime();

  const updateSelectionFromPointer = (event: { offsetX: number }) => {
    const selection = selectionAtOffset(event.offsetX);
    if (!selection) {
      return;
    }
    if (selection.time === selectedTime()) {
      return;
    }

    setInternalSelectedTime(selection.time);
    props.onSelectionChange?.(selection);
  };

  const selectionAtOffset = (offsetX: number): CandleChartSelection | undefined => {
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

    return {
      time,
      candle,
      annotations: annotationsForCandle(props.annotations ?? [], candle),
    };
  };

  const currentViewport = () =>
    props.interactive
      ? normalizeViewport(
          props.viewport ?? viewport() ?? defaultViewport(props.candles.length),
          props.candles.length,
          MIN_INTERACTIVE_CANDLES,
        )
      : defaultViewport(props.candles.length);

  const defaultViewport = (total: number): CandleChartViewport => {
    const maxCandles = props.maxCandles ?? 140;
    if (maxCandles > 0) {
      return normalizeViewport(
        { start: total - maxCandles, end: total },
        total,
        props.interactive ? MIN_INTERACTIVE_CANDLES : 1,
      );
    }

    return normalizeViewport(
      { start: 0, end: total },
      total,
      props.interactive ? MIN_INTERACTIVE_CANDLES : 1,
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
      title={props.interactive ? "Drag to pan. Wheel to zoom. Double-click to reset." : undefined}
      style={{ "touch-action": props.interactive ? "none" : "auto" }}
      onDblClick={props.interactive ? resetViewport : undefined}
      onKeyDown={handleKeyDown}
      onPointerCancel={handlePointerUp}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
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
    left: 18,
    right: width - 84,
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

function findCandleForTime(candles: Candle[], time: number | undefined): Candle | undefined {
  if (!time || candles.length === 0) {
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
  if (annotations.length === 0) {
    return;
  }

  const sampleEvery = Math.max(
    1,
    Math.ceil(annotations.length / MAX_BACKGROUND_ANNOTATION_MARKERS),
  );

  ctx.save();
  for (let index = 0; index < annotations.length; index += sampleEvery) {
    const annotation = annotations[index];
    if (!annotation) {
      continue;
    }
    if (
      selectedCandle &&
      annotation.time >= selectedCandle.openTime &&
      annotation.time <= selectedCandle.closeTime
    ) {
      continue;
    }

    const x = clamp(timeToX(annotation.time), plot.left, plot.right);
    const y = clamp(priceToY(annotation.price), plot.top, plot.bottom);
    const isBuy = annotation.kind.startsWith("buy");
    const isSignal = annotation.kind.endsWith("signal");
    ctx.globalAlpha = isSignal ? 0.62 : 0.38;
    ctx.strokeStyle = isBuy ? "#22c55e" : "#f05252";
    ctx.lineWidth = isSignal ? 1.5 : 1;
    ctx.beginPath();
    ctx.moveTo(x, isBuy ? y + 7 : y - 7);
    ctx.lineTo(x, isBuy ? y + 1 : y - 1);
    ctx.stroke();
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
  ctx.lineTo(x, plot.bottom);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.globalAlpha = 0.95;
  ctx.strokeStyle = "#f4f6fb";
  ctx.strokeRect(candleX - 6, highY - 4, 12, Math.max(8, lowY - highY + 8));
  ctx.restore();
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
    const color = isBuy ? "#22c55e" : "#f05252";

    ctx.save();
    ctx.fillStyle = color;
    ctx.strokeStyle = "#090a0d";
    ctx.lineWidth = 2;
    ctx.beginPath();
    if (isBuy) {
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
    ctx.textBaseline = isBuy ? "bottom" : "top";
    ctx.fillText(annotationShortLabel(annotation), x, isBuy ? y - 9 : y + 9);
    ctx.restore();
  }
}

function annotationShortLabel(annotation: BacktestChartAnnotation): string {
  if (annotation.kind.includes("signal")) {
    return "SIG";
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

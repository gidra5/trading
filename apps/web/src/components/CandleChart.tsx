import { createEffect, onCleanup, onMount } from "solid-js";
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
  emptyLabel?: string;
}

export function CandleChart(props: CandleChartProps) {
  let canvas!: HTMLCanvasElement;
  let observer: ResizeObserver | undefined;

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

    const maxCandles = props.maxCandles ?? 140;
    const candles = maxCandles > 0 ? props.candles.slice(-maxCandles) : props.candles;
    if (candles.length < 2) {
      drawEmpty(ctx, width, height, props.emptyLabel);
      return;
    }

    const openOrders = props.orders.filter((order) => order.status === "open");
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
    const values = candles.flatMap((candle) => [candle.high, candle.low]);
    if (props.lastPrice > 0) {
      values.push(props.lastPrice);
    }
    values.push(...openOrders.map((order) => order.price));
    values.push(...smaSeries.flatMap((series) => series.points.map((point) => point.value)));
    values.push(...annotations.map((annotation) => annotation.price));

    const min = Math.min(...values);
    const max = Math.max(...values);
    const padding = Math.max((max - min) * 0.08, max * 0.0005);
    const priceMin = min - padding;
    const priceMax = max + padding;
    const plot = {
      left: 12,
      right: width - 74,
      top: 16,
      bottom: height - 28,
    };
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

    const step = plotWidth / candles.length;
    const bodyWidth = Math.max(3, Math.min(10, step * 0.62));

    candles.forEach((candle, index) => {
      const x = plot.left + index * step + step / 2;
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

    if (props.lastPrice > 0) {
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

    drawAnnotations(ctx, annotations, plot, priceToY, timeToX);
    drawTimeLabels(ctx, candles, plot);
  };

  createEffect(() => {
    props.candles.length;
    props.orders.length;
    props.lastPrice;
    props.smaSeries?.length;
    props.annotations?.length;
    draw();
  });

  onMount(() => {
    observer = new ResizeObserver(draw);
    observer.observe(canvas.parentElement ?? canvas);
    draw();
  });

  onCleanup(() => observer?.disconnect());

  return <canvas ref={canvas} class="h-full min-h-80 w-full rounded-2" />;
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
    ctx.fillText(series.label, plot.right - 6, clamp(priceToY(last.value), plot.top + 8, plot.bottom - 8));
  }
  ctx.restore();
}

function drawAnnotations(
  ctx: CanvasRenderingContext2D,
  annotations: BacktestChartAnnotation[],
  plot: { left: number; right: number; top: number; bottom: number },
  priceToY: (price: number) => number,
  timeToX: (time: number) => number,
): void {
  const visible = annotations.slice(-160);
  for (const annotation of visible) {
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
    ctx.fillText(annotation.kind.includes("signal") ? "SIG" : annotation.kind.includes("fill") ? "F" : "O", x, isBuy ? y - 9 : y + 9);
    ctx.restore();
  }
}

function drawTimeLabels(
  ctx: CanvasRenderingContext2D,
  candles: Candle[],
  plot: { left: number; right: number; bottom: number },
): void {
  ctx.fillStyle = "#aeb6c8";
  ctx.font = "11px Inter, sans-serif";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";

  const count = Math.min(5, candles.length);
  for (let i = 0; i < count; i += 1) {
    const index = Math.round((candles.length - 1) * (i / Math.max(1, count - 1)));
    const candle = candles[index];
    const x = plot.left + (plot.right - plot.left) * (i / Math.max(1, count - 1));
    ctx.fillText(formatTime(candle.openTime), x, plot.bottom + 8);
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

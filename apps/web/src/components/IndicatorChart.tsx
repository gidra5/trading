import { createEffect, createSignal, onCleanup, onMount } from "solid-js";
import { formatQuote } from "../format";
import { CANDLE_CHART_PLOT_LEFT, CANDLE_CHART_PLOT_RIGHT } from "./CandleChart";

export interface IndicatorChartPoint {
  time: number;
  value: number;
}

export interface IndicatorChartReference {
  value: number;
  color?: string;
}

export interface IndicatorChartOverlay {
  label: string;
  color: string;
  points: IndicatorChartPoint[];
  dashed?: boolean;
}

export interface IndicatorChartEvent {
  time: number;
  seriesId: string;
  label: string;
  color?: string;
}

export interface IndicatorChartSeries {
  id: string;
  label: string;
  color: string;
  points: IndicatorChartPoint[];
  minimum?: number;
  maximum?: number;
  symmetric?: boolean;
  references?: IndicatorChartReference[];
  overlays?: IndicatorChartOverlay[];
  decimals?: number;
  suffix?: string;
}

interface IndicatorChartProps {
  series: IndicatorChartSeries[];
  events?: IndicatorChartEvent[];
  start: number;
  end: number;
  cursorTime?: number;
  onCursorTimeChange?: (time: number | undefined) => void;
  onZoom: (scale: number, anchor: number) => void;
  onPan: (fraction: number) => void;
}

export function IndicatorChart(props: IndicatorChartProps) {
  let canvas!: HTMLCanvasElement;
  let observer: ResizeObserver | undefined;
  let drag: { pointerId: number; x: number } | undefined;
  const [dragging, setDragging] = createSignal(false);
  const [internalCursorTime, setInternalCursorTime] = createSignal<number>();

  const draw = () => {
    if (!canvas) return;
    const width = Math.max(320, canvas.parentElement?.clientWidth ?? canvas.clientWidth);
    const height = Math.max(120, canvas.parentElement?.clientHeight ?? canvas.clientHeight);
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.floor(width * ratio);
    canvas.height = Math.floor(height * ratio);
    canvas.style.width = `${width}px`;
    canvas.style.height = `${height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    context.fillStyle = "#101217";
    context.fillRect(0, 0, width, height);

    const right = Math.max(CANDLE_CHART_PLOT_LEFT + 1, width - CANDLE_CHART_PLOT_RIGHT);
    const lanes = Math.max(1, props.series.length);
    const laneHeight = height / lanes;
    const duration = Math.max(1, props.end - props.start);
    const xFor = (time: number) => CANDLE_CHART_PLOT_LEFT
      + (time - props.start) / duration * (right - CANDLE_CHART_PLOT_LEFT);

    context.strokeStyle = "#242833";
    context.lineWidth = 1;
    for (let index = 0; index <= 4; index += 1) {
      const x = CANDLE_CHART_PLOT_LEFT
        + index / 4 * (right - CANDLE_CHART_PLOT_LEFT);
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
    }

    props.series.forEach((series, lane) => {
      const laneTop = lane * laneHeight;
      const top = laneTop + 6;
      const bottom = laneTop + laneHeight - 6;
      const visible = series.points.filter((point) =>
        point.time >= props.start && point.time < props.end && Number.isFinite(point.value));
      const visibleOverlays = (series.overlays ?? []).map((overlay) => ({
        ...overlay,
        points: overlay.points.filter((point) =>
          point.time >= props.start && point.time < props.end && Number.isFinite(point.value)),
      }));
      const values = [
        ...visible.map((point) => point.value),
        ...visibleOverlays.flatMap((overlay) => overlay.points.map((point) => point.value)),
      ];
      let minimum = series.minimum ?? Math.min(...values, 0);
      let maximum = series.maximum ?? Math.max(...values, 1);
      if (series.symmetric) {
        const magnitude = Math.max(Math.abs(minimum), Math.abs(maximum), Number.EPSILON);
        minimum = -magnitude;
        maximum = magnitude;
      }
      if (!(maximum > minimum)) {
        const padding = Math.max(1, Math.abs(maximum) * 0.05);
        minimum -= padding;
        maximum += padding;
      } else if (series.minimum === undefined || series.maximum === undefined) {
        const padding = (maximum - minimum) * 0.08;
        if (series.minimum === undefined) minimum -= padding;
        if (series.maximum === undefined) maximum += padding;
      }
      const yFor = (value: number) => top + (maximum - value) / (maximum - minimum) * (bottom - top);

      if (lane > 0) {
        context.strokeStyle = "#2b303b";
        context.beginPath();
        context.moveTo(CANDLE_CHART_PLOT_LEFT, laneTop);
        context.lineTo(right, laneTop);
        context.stroke();
      }
      for (const reference of series.references ?? []) {
        if (reference.value < minimum || reference.value > maximum) continue;
        const y = yFor(reference.value);
        context.save();
        context.setLineDash([4, 4]);
        context.strokeStyle = reference.color ?? "#505868";
        context.beginPath();
        context.moveTo(CANDLE_CHART_PLOT_LEFT, y);
        context.lineTo(right, y);
        context.stroke();
        context.restore();
      }

      for (const overlay of visibleOverlays) {
        drawSeriesLine(context, overlay.points, overlay.color, top, bottom, right, xFor, yFor, overlay.dashed);
        const last = overlay.points.at(-1);
        if (last) {
          context.font = "10px Inter, sans-serif";
          context.textAlign = "right";
          context.textBaseline = "middle";
          context.fillStyle = overlay.color;
          context.fillText(
            formatIndicator(last.value, series),
            right - 5,
            Math.max(top + 7, Math.min(bottom - 7, yFor(last.value))),
          );
        }
      }

      drawSeriesLine(context, visible, series.color, top, bottom, right, xFor, yFor);

      context.font = "600 11px Inter, sans-serif";
      context.textAlign = "left";
      context.textBaseline = "top";
      context.fillStyle = series.color;
      context.fillText(series.label, CANDLE_CHART_PLOT_LEFT + 5, laneTop + 5);
      const last = visible.at(-1);
      context.font = "11px Inter, sans-serif";
      context.fillStyle = "#aeb6c8";
      context.fillText(formatIndicator(last?.value, series), right + 7, laneTop + 5);
      context.fillStyle = "#6f788a";
      context.fillText(formatIndicator(maximum, series), right + 7, laneTop + 20);
      if (laneHeight >= 54) {
        context.textBaseline = "bottom";
        context.fillText(formatIndicator(minimum, series), right + 7, laneTop + laneHeight - 5);
      }
    });

    for (const event of props.events ?? []) {
      if (event.time < props.start || event.time >= props.end) continue;
      const lane = props.series.findIndex((series) => series.id === event.seriesId);
      if (lane < 0) continue;
      const laneTop = lane * laneHeight;
      const x = Math.max(CANDLE_CHART_PLOT_LEFT, Math.min(right, xFor(event.time)));
      context.save();
      context.strokeStyle = event.color ?? "#fb923c";
      context.globalAlpha = 0.46;
      context.setLineDash([2, 3]);
      context.beginPath();
      context.moveTo(x, laneTop + 18);
      context.lineTo(x, laneTop + laneHeight - 4);
      context.stroke();
      context.setLineDash([]);
      context.globalAlpha = 1;
      context.fillStyle = event.color ?? "#fb923c";
      context.beginPath();
      context.moveTo(x - 4, laneTop + 17);
      context.lineTo(x + 4, laneTop + 17);
      context.lineTo(x, laneTop + 22);
      context.closePath();
      context.fill();
      context.restore();
    }

    const cursorTime = props.cursorTime ?? internalCursorTime();
    if (cursorTime !== undefined && cursorTime >= props.start && cursorTime < props.end) {
      const x = Math.max(CANDLE_CHART_PLOT_LEFT, Math.min(right, xFor(cursorTime)));
      context.save();
      context.strokeStyle = "#d6dbea";
      context.globalAlpha = 0.78;
      context.setLineDash([4, 5]);
      context.beginPath();
      context.moveTo(x, 0);
      context.lineTo(x, height);
      context.stroke();
      context.setLineDash([]);
      drawCursorLabel(context, x, height, right, cursorTime);
      const nearest = nearestEvent(props.events ?? [], cursorTime, props.start, props.end, right);
      if (nearest) {
        const lane = props.series.findIndex((series) => series.id === nearest.seriesId);
        if (lane >= 0) drawEventTooltip(context, x, lane * laneHeight + 24, right, nearest.label);
      }
      context.restore();
    }
  };

  createEffect(() => {
    props.start;
    props.end;
    for (const series of props.series) {
      series.points.length;
      series.references?.length;
      series.overlays?.forEach((overlay) => overlay.points.length);
    }
    props.events?.length;
    props.cursorTime;
    internalCursorTime();
    draw();
  });

  onMount(() => {
    observer = new ResizeObserver(draw);
    observer.observe(canvas.parentElement ?? canvas);
    draw();
  });
  onCleanup(() => observer?.disconnect());

  const handleWheel = (event: WheelEvent) => {
    event.preventDefault();
    const bounds = canvas.getBoundingClientRect();
    if (event.shiftKey || Math.abs(event.deltaX) > Math.abs(event.deltaY)) {
      props.onPan((event.deltaX || event.deltaY) / Math.max(1, bounds.width));
      return;
    }
    const anchor = Math.max(0, Math.min(1,
      (event.clientX - bounds.left - CANDLE_CHART_PLOT_LEFT)
        / Math.max(1, bounds.width - CANDLE_CHART_PLOT_LEFT - CANDLE_CHART_PLOT_RIGHT)));
    props.onZoom(Math.exp(Math.sign(event.deltaY) * 0.18), anchor);
  };
  const handlePointerDown = (event: PointerEvent) => {
    if (event.button !== 0) return;
    event.preventDefault();
    drag = { pointerId: event.pointerId, x: event.clientX };
    setDragging(true);
    updateCursor(event);
    canvas.setPointerCapture(event.pointerId);
  };
  const handlePointerMove = (event: PointerEvent) => {
    updateCursor(event);
    if (!drag || event.pointerId !== drag.pointerId) return;
    event.preventDefault();
    const distance = drag.x - event.clientX;
    drag.x = event.clientX;
    props.onPan(distance / Math.max(
      1,
      canvas.clientWidth - CANDLE_CHART_PLOT_LEFT - CANDLE_CHART_PLOT_RIGHT,
    ));
  };
  const updateCursor = (event: PointerEvent) => {
    const bounds = canvas.getBoundingClientRect();
    const right = Math.max(CANDLE_CHART_PLOT_LEFT + 1, bounds.width - CANDLE_CHART_PLOT_RIGHT);
    const fraction = Math.max(0, Math.min(1,
      (event.clientX - bounds.left - CANDLE_CHART_PLOT_LEFT)
        / Math.max(1, right - CANDLE_CHART_PLOT_LEFT)));
    const time = props.start + fraction * Math.max(1, props.end - props.start);
    setInternalCursorTime(time);
    props.onCursorTimeChange?.(time);
  };
  const clearCursor = () => {
    if (drag) return;
    setInternalCursorTime(undefined);
    props.onCursorTimeChange?.(undefined);
  };
  const handlePointerUp = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
    drag = undefined;
    setDragging(false);
  };

  return (
    <canvas
      ref={canvas}
      class={`h-full w-full rounded-2 select-none ${dragging() ? "cursor-grabbing" : "cursor-grab"}`}
      aria-label="Synchronized KAMA indicator chart"
      onPointerCancel={handlePointerUp}
      onPointerDown={handlePointerDown}
      onPointerMove={handlePointerMove}
      onPointerLeave={clearCursor}
      onPointerUp={handlePointerUp}
      onWheel={handleWheel}
    />
  );
}

function drawSeriesLine(
  context: CanvasRenderingContext2D,
  points: IndicatorChartPoint[],
  color: string,
  top: number,
  bottom: number,
  right: number,
  xFor: (time: number) => number,
  yFor: (value: number) => number,
  dashed = false,
): void {
  if (points.length < 2) return;
  context.save();
  if (dashed) context.setLineDash([5, 4]);
  context.beginPath();
  points.forEach((point, index) => {
    const x = Math.max(CANDLE_CHART_PLOT_LEFT, Math.min(right, xFor(point.time)));
    const y = Math.max(top, Math.min(bottom, yFor(point.value)));
    if (index === 0) context.moveTo(x, y);
    else context.lineTo(x, y);
  });
  context.strokeStyle = color;
  context.lineWidth = dashed ? 1 : 1.5;
  context.stroke();
  context.restore();
}

function drawCursorLabel(
  context: CanvasRenderingContext2D,
  x: number,
  height: number,
  right: number,
  time: number,
): void {
  const label = formatCursorTime(time);
  context.font = "11px Inter, sans-serif";
  const width = context.measureText(label).width + 12;
  const left = Math.max(CANDLE_CHART_PLOT_LEFT, Math.min(right - width, x - width / 2));
  context.fillStyle = "#252a35";
  context.fillRect(left, height - 18, width, 18);
  context.fillStyle = "#eef1f8";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(label, left + width / 2, height - 9);
}

function drawEventTooltip(
  context: CanvasRenderingContext2D,
  x: number,
  y: number,
  right: number,
  label: string,
): void {
  context.font = "11px Inter, sans-serif";
  const width = Math.min(300, context.measureText(label).width + 14);
  const left = Math.max(CANDLE_CHART_PLOT_LEFT, Math.min(right - width, x + 8));
  context.fillStyle = "#2a211b";
  context.fillRect(left, y, width, 22);
  context.fillStyle = "#fdba74";
  context.textAlign = "left";
  context.textBaseline = "middle";
  context.fillText(label, left + 7, y + 11, width - 14);
}

function nearestEvent(
  events: IndicatorChartEvent[],
  cursorTime: number,
  start: number,
  end: number,
  right: number,
): IndicatorChartEvent | undefined {
  const tolerance = Math.max(1, end - start) * 8
    / Math.max(1, right - CANDLE_CHART_PLOT_LEFT);
  return events.reduce<IndicatorChartEvent | undefined>((nearest, event) =>
    event.time >= start && event.time < end
      && Math.abs(event.time - cursorTime) <= tolerance
      && (!nearest || Math.abs(event.time - cursorTime) < Math.abs(nearest.time - cursorTime))
      ? event
      : nearest, undefined);
}

function formatCursorTime(time: number): string {
  return `${new Date(time).toISOString().replace("T", " ").slice(0, 19)} UTC`;
}

function formatIndicator(value: number | undefined, series: IndicatorChartSeries): string {
  return value === undefined || !Number.isFinite(value)
    ? "—"
    : `${formatQuote(value, series.decimals ?? 2)}${series.suffix ?? ""}`;
}

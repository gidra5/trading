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

export interface IndicatorChartSeries {
  id: string;
  label: string;
  color: string;
  points: IndicatorChartPoint[];
  minimum?: number;
  maximum?: number;
  symmetric?: boolean;
  references?: IndicatorChartReference[];
  decimals?: number;
  suffix?: string;
}

interface IndicatorChartProps {
  series: IndicatorChartSeries[];
  start: number;
  end: number;
  onZoom: (scale: number, anchor: number) => void;
  onPan: (fraction: number) => void;
}

export function IndicatorChart(props: IndicatorChartProps) {
  let canvas!: HTMLCanvasElement;
  let observer: ResizeObserver | undefined;
  let drag: { pointerId: number; x: number } | undefined;
  const [dragging, setDragging] = createSignal(false);

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
      const values = visible.map((point) => point.value);
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

      if (visible.length > 1) {
        context.save();
        context.beginPath();
        visible.forEach((point, index) => {
          const x = Math.max(CANDLE_CHART_PLOT_LEFT, Math.min(right, xFor(point.time)));
          const y = Math.max(top, Math.min(bottom, yFor(point.value)));
          if (index === 0) context.moveTo(x, y);
          else context.lineTo(x, y);
        });
        context.strokeStyle = series.color;
        context.lineWidth = 1.5;
        context.stroke();
        context.restore();
      }

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
  };

  createEffect(() => {
    props.start;
    props.end;
    for (const series of props.series) {
      series.points.length;
      series.references?.length;
    }
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
    canvas.setPointerCapture(event.pointerId);
  };
  const handlePointerMove = (event: PointerEvent) => {
    if (!drag || event.pointerId !== drag.pointerId) return;
    event.preventDefault();
    const distance = drag.x - event.clientX;
    drag.x = event.clientX;
    props.onPan(distance / Math.max(
      1,
      canvas.clientWidth - CANDLE_CHART_PLOT_LEFT - CANDLE_CHART_PLOT_RIGHT,
    ));
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
      onPointerUp={handlePointerUp}
      onWheel={handleWheel}
    />
  );
}

function formatIndicator(value: number | undefined, series: IndicatorChartSeries): string {
  return value === undefined || !Number.isFinite(value)
    ? "—"
    : `${formatQuote(value, series.decimals ?? 2)}${series.suffix ?? ""}`;
}

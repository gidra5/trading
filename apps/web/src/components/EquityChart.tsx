import { createEffect, onCleanup, onMount } from "solid-js";
import type { EquityPoint } from "@trading/bot-algo";
import { formatQuote } from "../format";

interface EquityChartProps {
  points: EquityPoint[];
  emptyText?: string;
}

export function EquityChart(props: EquityChartProps) {
  let canvas!: HTMLCanvasElement;
  let observer: ResizeObserver | undefined;

  const draw = () => {
    const parent = canvas.parentElement;
    const width = Math.max(260, parent?.clientWidth ?? 320);
    const height = Math.max(160, parent?.clientHeight ?? 180);
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

    const points = props.points;
    if (points.length < 2) {
      ctx.fillStyle = "#aeb6c8";
      ctx.font = "13px Inter, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(props.emptyText ?? "No equity history yet", width / 2, height / 2);
      return;
    }

    const equities = points.map((point) => point.equity);
    const min = Math.min(...equities);
    const max = Math.max(...equities);
    const padding = Math.max((max - min) * 0.08, 1);
    const low = min - padding;
    const high = max + padding;
    const left = 10;
    const right = width - 70;
    const top = 12;
    const bottom = height - 20;
    const yFor = (value: number) => top + ((high - value) / (high - low)) * (bottom - top);

    ctx.strokeStyle = "#242833";
    ctx.lineWidth = 1;
    for (let i = 0; i <= 3; i += 1) {
      const value = low + ((high - low) * i) / 3;
      const y = yFor(value);
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
      ctx.fillStyle = "#aeb6c8";
      ctx.font = "11px Inter, sans-serif";
      ctx.textAlign = "left";
      ctx.fillText(formatQuote(value, 0), right + 8, y + 3);
    }

    ctx.strokeStyle = points[points.length - 1].equity >= points[0].equity ? "#22c55e" : "#f05252";
    ctx.lineWidth = 2;
    ctx.beginPath();
    points.forEach((point, index) => {
      const x = left + (index / (points.length - 1)) * (right - left);
      const y = yFor(point.equity);
      if (index === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });
    ctx.stroke();
  };

  createEffect(() => {
    props.points.length;
    draw();
  });

  onMount(() => {
    observer = new ResizeObserver(draw);
    observer.observe(canvas.parentElement ?? canvas);
    draw();
  });

  onCleanup(() => observer?.disconnect());

  return <canvas ref={canvas} class="h-full min-h-40 w-full rounded-2" />;
}

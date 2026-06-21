import {
  defineConfig,
  presetAttributify,
  presetUno,
  transformerDirectives,
} from "unocss";

export default defineConfig({
  presets: [presetUno(), presetAttributify()],
  transformers: [transformerDirectives()],
  theme: {
    colors: {
      ink: {
        950: "#090a0d",
        900: "#101217",
        800: "#171a21",
        700: "#242833",
        600: "#323846",
        300: "#aeb6c8",
        200: "#d6dbea",
        100: "#f4f6fb",
      },
      gain: "#22c55e",
      loss: "#f05252",
      warn: "#f5b84b",
      line: "#2b303b",
      accent: "#38bdf8",
    },
  },
  shortcuts: {
    panel:
      "border border-line bg-ink-900/88 rounded-2 p-4 shadow-[0_12px_40px_rgba(0,0,0,0.26)]",
    "panel-tight": "border border-line bg-ink-900/88 rounded-2 p-3",
    "muted-label": "text-ink-300 text-xs uppercase tracking-wide",
    "metric-value": "text-ink-100 text-xl font-semibold tabular-nums",
    btn:
      "inline-flex min-h-9 select-none items-center justify-center gap-2 whitespace-nowrap rounded-2 border border-line bg-ink-800 px-3 py-2 text-sm font-semibold text-ink-100 shadow-[inset_0_1px_0_rgba(255,255,255,0.04)] transition hover:border-accent/70 hover:bg-ink-700 active:translate-y-px focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent/45 disabled:cursor-not-allowed disabled:opacity-45 disabled:active:translate-y-0",
    "table-head": "whitespace-nowrap pr-3 text-left text-xs uppercase tracking-wide text-ink-300 font-medium",
    "td-cell": "whitespace-nowrap border-t border-line py-2 pr-3 align-top text-sm tabular-nums",
  },
});

import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { VW_KAMA_SCORE_VERSION, type VwKamaPreset } from "../packages/bot-algo/src/index.js";
import { KamaInspectorEngine } from "../apps/server/src/kama-inspector.js";

void main();

async function main(): Promise<void> {
  const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
  const input = path.resolve(root, process.argv[2] ?? "data/benchmarks/vw-kama-window-presets.json");
  const output = path.resolve(
    root,
    process.argv[3] ?? `data/benchmarks/vw-kama-window-presets-v${VW_KAMA_SCORE_VERSION}-rescored.json`,
  );
  const parsed = JSON.parse(await fs.readFile(input, "utf8")) as VwKamaPreset | VwKamaPreset[];
  const presets = (Array.isArray(parsed) ? parsed : [parsed])
    .filter((preset) => preset.scope === "window" && preset.intervalMs === 1_000);
  const engine = new KamaInspectorEngine(path.join(root, "data"));
  const defaults = engine.catalog().defaults;
  const results = [];

  for (const [index, preset] of presets.entries()) {
    const evaluated = await engine.analyze({
      ...defaults,
      windowId: preset.windowId!,
      intervalMs: preset.intervalMs!,
      parameters: preset.parameters,
    });
    results.push({
      id: preset.id,
      windowId: preset.windowId,
      intervalMs: preset.intervalMs,
      historicalScore: preset.score,
      historicalScoreVersion: preset.scoreVersion,
      scoreVersion: VW_KAMA_SCORE_VERSION,
      metrics: evaluated.metrics,
      parameters: preset.parameters,
    });
    console.error(`${index + 1}/${presets.length} ${preset.windowId}: ${(evaluated.metrics.score * 100).toFixed(2)}%`);
  }

  await fs.writeFile(output, `${JSON.stringify(results, null, 2)}\n`);
  console.error(`Results: ${path.relative(root, output)}`);
  process.exit(0);
}

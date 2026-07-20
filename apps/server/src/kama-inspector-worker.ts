import { parentPort } from "node:worker_threads";
import type {
  VwKamaCandleRangeRequest,
  VwKamaInspectorRequest,
  VwKamaPredictorFitRequest,
} from "@trading/bot-algo";
import { KamaInspectorEngine } from "./kama-inspector.js";

const port = parentPort;
if (!port) throw new Error("VW-KAMA inspector worker requires a parent port.");

let engine: KamaInspectorEngine | null = null;

type InspectorWorkerRequest =
  | { type: "init"; dataDir: string }
  | { type: "analyze"; id: number; input: VwKamaInspectorRequest }
  | { type: "candles"; id: number; input: VwKamaCandleRangeRequest }
  | { type: "fit-predictor"; id: number; input: VwKamaPredictorFitRequest };

port.on("message", async (message: InspectorWorkerRequest) => {
  if (message.type === "init") {
    if (!message.dataDir) throw new Error("VW-KAMA worker data directory is required.");
    engine = new KamaInspectorEngine(message.dataDir);
    return;
  }
  if (!engine) return;
  try {
    const result = message.type === "analyze"
      ? await engine.analyze(message.input)
      : message.type === "candles"
        ? await engine.candles(message.input)
        : await engine.fitPredictor(message.input);
    port.postMessage({ id: message.id, result });
  } catch (error) {
    port.postMessage({
      id: message.id,
      error: error instanceof Error ? error.message : "VW-KAMA worker request failed",
    });
  }
});

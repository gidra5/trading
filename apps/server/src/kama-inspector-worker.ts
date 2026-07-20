import { parentPort } from "node:worker_threads";
import type {
  VwKamaCandleRangeRequest,
  VwKamaHistoricalAveragesRequest,
  VwKamaInspectorRequest,
  VwKamaPredictorFitRequest,
} from "@trading/bot-algo";
import { KamaInspectorEngine } from "./kama-inspector.js";

const port = parentPort;
if (!port) throw new Error("VW-KAMA inspector worker requires a parent port.");

let engine: KamaInspectorEngine | null = null;

type InspectorWorkerRequest =
  | { type: "init"; dataDir: string }
  | { type: "analyze"; id: number; input: VwKamaInspectorRequest; cancelFlag?: Int32Array }
  | { type: "candles"; id: number; input: VwKamaCandleRangeRequest; cancelFlag?: Int32Array }
  | { type: "fit-predictor"; id: number; input: VwKamaPredictorFitRequest; cancelFlag?: Int32Array }
  | { type: "historical-averages"; id: number; input: VwKamaHistoricalAveragesRequest; cancelFlag?: Int32Array };

port.on("message", async (message: InspectorWorkerRequest) => {
  if (message.type === "init") {
    if (!message.dataDir) throw new Error("VW-KAMA worker data directory is required.");
    engine = new KamaInspectorEngine(message.dataDir);
    return;
  }
  if (!engine) return;
  try {
    const result = message.type === "analyze"
      ? await engine.analyze(message.input, message.cancelFlag)
      : message.type === "candles"
        ? await engine.candles(message.input, message.cancelFlag)
        : message.type === "fit-predictor"
          ? await engine.fitPredictor(message.input, message.cancelFlag)
          : await engine.historicalAverages(message.input, message.cancelFlag);
    port.postMessage({ id: message.id, result });
  } catch (error) {
    port.postMessage({
      id: message.id,
      error: error instanceof Error ? error.message : "VW-KAMA worker request failed",
    });
  }
});

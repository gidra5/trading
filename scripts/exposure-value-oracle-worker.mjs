import { parentPort } from "node:worker_threads";
import {
  directOracleDiagnosticsCuda,
  exposureHoldingCutoffsCuda,
  prepareExposureValueOracleCuda,
} from "../packages/bot-algo/dist/index.js";

if (!parentPort) {
  throw new Error("Exposure-value oracle worker requires a parent thread.");
}

let queue = Promise.resolve();

parentPort.on("message", (message) => {
  if (message?.type !== "prepare") return;
  queue = queue.then(async () => {
  const started = performance.now();
  try {
    const prepared = await prepareExposureValueOracleCuda(
      message.prices,
      message.options,
      true,
    );
    let directDiagnostics;
    if (message.directDiagnostics) {
      const diagnosticStarted = performance.now();
      const count = message.directDiagnostics.exampleCount;
      const execution = {
        friction: message.options.friction,
        minExposure: message.options.minExposure,
        maxExposure: message.options.maxExposure,
        maxEffectiveExposure: message.options.maxEffectiveExposure,
        quoteBorrowRate: message.options.quoteBorrowRate ?? 0,
        assetBorrowRate: message.options.assetBorrowRate ?? 0,
      };
      const cutoffs = await exposureHoldingCutoffsCuda(
        message.prices,
        count,
        message.options.holdingPeriodSteps,
        execution,
        true,
      );
      const { cutoffLowers, cutoffUppers } = cutoffs;
      const probabilities = prepared.oracle.probabilities?.subarray(
        0,
        count * prepared.oracle.grid.length,
      );
      if (!probabilities) {
        throw new Error("Direct-oracle diagnostics require retained oracle probabilities.");
      }
      const diagnostic = await directOracleDiagnosticsCuda(
        probabilities,
        prepared.oracle.grid,
        prepared.oracle.currentGrid,
        cutoffLowers,
        cutoffUppers,
        {
          visibleLower: message.directDiagnostics.visibleLower,
          visibleUpper: message.directDiagnostics.visibleUpper,
          friction: message.options.friction,
          transitionLogScale: 1 / message.options.temperature,
          distanceEpsilon: message.directDiagnostics.distanceEpsilon,
        },
        true,
      );
      directDiagnostics = {
        ...diagnostic,
        cutoffLowers,
        cutoffUppers,
        cutoffKernelMs: cutoffs.kernelMs,
        wallMs: performance.now() - diagnosticStarted,
      };
    }
    parentPort.postMessage({
      type: "complete",
      id: message.id,
      prepared: {
        ...prepared,
        ...(directDiagnostics ? { directDiagnostics } : {}),
      },
      workerWallMs: performance.now() - started,
    });
  } catch (error) {
    parentPort.postMessage({
      type: "error",
      id: message.id,
      message: error instanceof Error ? error.stack ?? error.message : String(error),
    });
  }
  });
});

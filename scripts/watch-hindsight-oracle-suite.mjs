import fs from "node:fs";
import path from "node:path";

const suiteDir = path.resolve("data/benchmarks/hindsight-oracle-stored-v3");
const workerFile = path.join(suiteDir, "workers.json");
const outputFile = path.join(suiteDir, "hindsight-oracle-bot-suite.json");
const summaryFile = path.join(suiteDir, "hindsight-oracle-bot-suite.summary.json");
const smokeFile = path.join(suiteDir, "stored-smoke.json");
const expectedByBatch = new Map([
  ["fit-all", 5],
  ["shapes-rest", 9],
  ["sharpe", 8],
  ["sideways", 5],
  ["regimes-failures", 5],
  ["latest", 1],
]);

await main();

async function main() {
  const workers = parseJson(fs.readFileSync(workerFile, "utf8"));
  while (true) {
    let complete = true;
    const statuses = [];
    for (const worker of workers) {
      const expected = expectedByBatch.get(worker.Batch);
      if (expected === undefined) throw new Error(`Unknown suite batch ${worker.Batch}.`);
      const report = readJson(worker.Report);
      const resultCount = report?.results?.length ?? 0;
      const running = processExists(worker.Id);
      statuses.push({ batch: worker.Batch, running, resultCount, expected });
      if (resultCount < expected) {
        complete = false;
        if (!running) {
          const stderr = readText(worker.Stderr).trim();
          throw new Error(
            `${worker.Batch} stopped at ${resultCount}/${expected}. ${stderr || "No error was logged."}`,
          );
        }
      }
    }
    console.log(JSON.stringify({ event: "suite-status", at: new Date().toISOString(), statuses }));
    if (complete) break;
    await new Promise((resolve) => setTimeout(resolve, 60_000));
  }

  const reports = [readJson(smokeFile), ...workers.map((worker) => readJson(worker.Report))];
  const results = reports
    .flatMap((report) => report?.results ?? [])
    .sort((left, right) => left.startTime - right.startTime || left.id.localeCompare(right.id));
  const ids = new Set(results.map((result) => result.id));
  if (results.length !== 34 || ids.size !== 34) {
    throw new Error(`Expected 34 unique suite results, received ${results.length}/${ids.size}.`);
  }
  const template = reports.find(Boolean);
  const combined = {
    ...template,
    generatedAt: new Date().toISOString(),
    results,
  };
  const returns = results.map((result) => result.summary.returnPct).sort((left, right) => left - right);
  const summary = {
    generatedAt: combined.generatedAt,
    windows: results.length,
    profitableWindows: results.filter((result) => result.summary.returnPct > 0).length,
    meanReturnPct: mean(returns),
    medianReturnPct: median(returns),
    meanMaxDrawdownPct: mean(results.map((result) => result.summary.maxDrawdownPct)),
    medianMaxDrawdownPct: median(
      results.map((result) => result.summary.maxDrawdownPct).sort((left, right) => left - right),
    ),
    totalTrades: results.reduce((sum, result) => sum + result.summary.tradeCount, 0),
    totalCandles: results.reduce((sum, result) => sum + result.summary.candlesProcessed, 0),
    wallDurationMs: results.reduce((sum, result) => sum + result.wallDurationMs, 0),
    best: [...results]
      .sort((left, right) => right.summary.returnPct - left.summary.returnPct)
      .slice(0, 5)
      .map(compact),
    worst: [...results]
      .sort((left, right) => left.summary.returnPct - right.summary.returnPct)
      .slice(0, 5)
      .map(compact),
  };
  writeJson(outputFile, combined);
  writeJson(summaryFile, summary);
  console.log(JSON.stringify({ event: "suite-complete", outputFile, summaryFile, summary }));
}

function compact(result) {
  return {
    id: result.id,
    returnPct: result.summary.returnPct,
    maxDrawdownPct: result.summary.maxDrawdownPct,
    tradeCount: result.summary.tradeCount,
    maxEffectiveLeverage: result.summary.maxEffectiveLeverage,
  };
}

function mean(values) {
  return values.reduce((sum, value) => sum + value, 0) / Math.max(1, values.length);
}

function median(sorted) {
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function processExists(pid) {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

function readJson(file) {
  try {
    return parseJson(fs.readFileSync(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT") return undefined;
    throw error;
  }
}

function parseJson(value) {
  return JSON.parse(value.replace(/^\uFEFF|^ï»¿|^п»ї/, ""));
}

function readText(file) {
  try {
    return fs.readFileSync(file, "utf8");
  } catch (error) {
    if (error?.code === "ENOENT") return "";
    throw error;
  }
}

function writeJson(file, value) {
  fs.writeFileSync(file, `${JSON.stringify(value, null, 2)}\n`, "utf8");
}

/** Replay the prespecified specialist comparison and verify every decision. */
import { spawn } from "node:child_process";
import { mkdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
const outputIndex = args.indexOf("--output");
const output = path.resolve(root, outputIndex < 0 ? "data/backtests/oracle-neural-specialists" : args[outputIndex + 1]);
const plan = JSON.parse(await readFile(path.join(output, "replay-plan.json"), "utf8"));

async function run(args, logFile) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, args, { cwd: root, windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "", stderr = "";
    child.stdout.setEncoding("utf8").on("data", (part) => { stdout += part; });
    child.stderr.setEncoding("utf8").on("data", (part) => { stderr += part; });
    child.once("error", reject);
    child.once("close", async (code) => {
      try {
        await writeFile(logFile, stdout + (stderr ? `\nSTDERR\n${stderr}` : ""));
        if (code !== 0) throw new Error(`Command failed (${code}): ${args.join(" ")}\n${stderr.slice(-2000)}`);
        resolve();
      } catch (error) { reject(error); }
    });
  });
}

const jobs = plan.horizonsSeconds.flatMap((horizon) => plan.variants.flatMap((variant) =>
  variant.feeScales.map((feeScale) => ({ horizon, variant: variant.name, feeScale }))));
let next = 0;
async function worker() {
  while (next < jobs.length) {
    const { horizon, variant, feeScale } = jobs[next++];
    const name = `${variant}-h${horizon}-fees${feeScale}`;
    const directory = path.join(output, name);
    await mkdir(directory, { recursive: true });
    const count = String(plan.decisionCountsByHorizon[horizon]);
    await run(["--conditions=development", "--import", "tsx", "scripts/backtest-oracle-endpoint-hold.ts",
      "--forecasts", path.join(output, `${variant}-h${horizon}.jsonl`),
      "--count", count, "--certificate-count", count, "--fee-scale", String(feeScale),
      "--output", directory], path.join(directory, "run.log"));
    const report = JSON.parse(await readFile(path.join(directory, "report.json"), "utf8"));
    console.log(JSON.stringify({ replay: name, returnPercent: report.returnPercent,
      markets: report.markets, certificates: report.decisionCertificates }));
  }
}
await Promise.all([worker(), worker()]);
await run([plan.reportScript ?? "scripts/report-oracle-neural-specialists.mjs", "--output", output], path.join(output, "artifact-checks.log"));
await run(["scripts/verify-oracle-crra.mjs", "--certificates", path.join(output, "decisions.jsonl"),
  "--output", path.join(output, "lean-verification.json")], path.join(output, "lean-verification.log"));
const comparison = JSON.parse(await readFile(path.join(output, "comparison.json"), "utf8"));
console.log(JSON.stringify({ artifactChecksPassed: comparison.artifactChecksPassed,
  decisionsCheckedInLean: comparison.certificateRecords,
  report: path.join(output, "comparison.json") }));

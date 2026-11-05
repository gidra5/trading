import { spawn } from "node:child_process";
import { createReadStream, existsSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const args = process.argv.slice(2);
function option(key, fallback) { const i = args.indexOf(key); return i < 0 ? fallback : args[i + 1]; }
const project = path.resolve(option("--lean-project", path.join(root, "docs/theory/lean")));
const elan = process.env.ELAN_HOME || path.join(root, ".tools/elan");
const lake = path.join(elan, "bin", process.platform === "win32" ? "lake.exe" : "lake");
if (!existsSync(lake)) throw new Error("Run npm run theory:setup before certificate verification");
const inputs = [option("--fixtures", "data/backtests/stochastic-crra/oracle-conformance.jsonl"),
  option("--certificates", "data/backtests/stochastic-crra/first-hour-gamma0/decisions.jsonl")].map((file) => path.resolve(root, file));
const output = path.resolve(root, option("--output", "data/backtests/stochastic-crra/lean-verification.json"));
const reports = [];
for (const input of inputs) {
  if (!existsSync(input)) throw new Error(`Missing checker input: ${input}`);
  reports.push(await new Promise((resolve, reject) => {
    const child = spawn(lake, ["env", "lean", "--run", "ExactCRRACheck.lean"], {
      cwd: project, windowsHide: true, stdio: ["pipe", "pipe", "pipe"],
      env: { ...process.env, ELAN_HOME: elan },
    });
    let stdout = "", stderr = "";
    child.stdout.setEncoding("utf8").on("data", (chunk) => { stdout += chunk; });
    child.stderr.setEncoding("utf8").on("data", (chunk) => { stderr += chunk; });
    child.once("error", reject);
    const source = createReadStream(input);
    const hash = createHash("sha256");
    source.on("data", (chunk) => hash.update(chunk));
    source.once("error", (error) => { child.kill(); reject(error); });
    child.stdin.on("error", (error) => { if (error.code !== "EPIPE") reject(error); });
    source.pipe(child.stdin);
    child.once("close", (code) => {
      try {
        if (code !== 0) throw new Error(`Lean checker exited ${code}: ${stderr}\n${stdout.slice(-1000)}`);
        const results = stdout.trim().split(/\r?\n/).filter(Boolean).map((line) => JSON.parse(line));
        if (!results.length || results.some((result) => result.ok !== true))
          throw new Error(`Lean rejected certificate: ${stdout.slice(-1000)}`);
        resolve({ input, sha256: hash.digest("hex"), checked: results.length, passed: true, results });
      } catch (error) { reject(error); }
    });
  }));
}
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, JSON.stringify({ project, reports }, null, 2) + "\n");
console.log(JSON.stringify(reports.map(({ input, checked, passed }) => ({ input, checked, passed })), null, 2));

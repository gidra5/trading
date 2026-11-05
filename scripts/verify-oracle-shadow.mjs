import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { createReadStream } from "node:fs";
import { readFile, writeFile } from "node:fs/promises";
import { createHash } from "node:crypto";
import path from "node:path";

const root = path.resolve(process.argv[2] ?? "data/backtests/oracle-finite-cash-replay");
const elan = path.resolve(process.env.ELAN_HOME ?? ".tools/elan");
const source = path.join(root, "shadow-certificates.jsonl");
async function run(file, negative = false) {
  return new Promise((resolve, reject) => {
    const child = spawn(path.join(elan, "bin", process.platform === "win32" ? "lake.exe" : "lake"),
      ["env", "lean", "--run", "ExactShadowCheck.lean"], {
        cwd: path.resolve("docs/theory/lean"), env: { ...process.env, ELAN_HOME: elan },
        windowsHide: true, stdio: ["pipe", "pipe", "pipe"],
      });
    let out = "", err = ""; const hash = createHash("sha256");
    child.stdout.setEncoding("utf8").on("data", b => out += b);
    child.stderr.setEncoding("utf8").on("data", b => err += b);
    child.once("error", reject);
    child.stdin.on("error", e => { if (e.code !== "EPIPE") reject(e); });
    if (negative) child.stdin.end('{"lowerTick":"100000","upperTick":"100100"}\n');
    else {
      const stream = createReadStream(file); stream.on("data", b => hash.update(b));
      stream.once("error", reject); stream.pipe(child.stdin);
    }
    child.once("close", code => {
      try {
        if (negative) {
          assert.notEqual(code, 0); assert.match(out+err, /Invalid 2-bps shadow-price envelope/);
          resolve({ rejectedInvalidEnvelope: true });
        } else {
          assert.equal(code, 0, out+err); const result = JSON.parse(out.trim()); assert.equal(result.ok, true);
          resolve({ ...result, sourceSha256: hash.digest("hex") });
        }
      } catch (e) { reject(e); }
    });
  });
}
const [positive, negative] = await Promise.all([run(source), run(undefined, true)]);
const report = JSON.parse(await readFile(path.join(root, "report.json"), "utf8"));
assert.equal(positive.checked, report.shadowBoundDecisions);
const hashes = {};
for (const file of ["docs/theory/lean/ExactShadowCheck.lean", "docs/theory/lean/TradingTheory/OracleShadowBound.lean"])
  hashes[file] = createHash("sha256").update(await readFile(file)).digest("hex");
await writeFile(path.join(root, "lean-shadow-verification.json"), JSON.stringify({ ...positive, ...negative,
  hashes, scope: "Exact envelope inequalities; Oracle trace bound proved separately; envelope-to-forecast matching checked by execution verifier" }, null, 2)+"\n");
console.log(JSON.stringify({ ...positive, ...negative }));

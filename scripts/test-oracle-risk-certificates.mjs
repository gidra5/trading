import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const elan = process.env.ELAN_HOME || path.join(root, ".tools/elan");
const candidates = [
  { id: "cash", feasible: true, branches: [{ weight: "1", wealth: "100" }] },
  { id: "risky", feasible: true, branches: [{ weight: "1/2", wealth: "300" }, { weight: "1/2", wealth: "0" }] },
];
const records = [
  { riskAversion: 0, chosenId: "risky", candidates },
  { riskAversion: 0, chosenId: "cash", candidates },
  { riskAversion: 2, chosenId: "risky", candidates },
  { riskAversion: 2, chosenId: "cash", candidates },
  { riskAversion: 1, chosenId: "cash", candidates },
  { riskAversion: 0, chosenId: "only", candidates: [{ id: "only", feasible: true, branches: [
    { weight: "1", child: { riskAversion: 2, chosenId: "cash", candidates } },
  ] }] },
];
const result = spawnSync(path.join(elan, "bin", process.platform === "win32" ? "lake.exe" : "lake"),
  ["env", "lean", "--run", "ExactCRRACheck.lean"], { cwd: path.join(root, "docs/theory/lean"),
    env: { ...process.env, ELAN_HOME: elan }, windowsHide: true, encoding: "utf8",
    input: records.map((record) => JSON.stringify(record)).join("\n") + "\n" });
if (result.error) throw result.error;
assert(result.stdout.trimStart().startsWith("{"), result.stdout + result.stderr);
const outputs = result.stdout.trim().split(/\r?\n/).map((line) => JSON.parse(line));
assert.deepEqual(outputs.map((output) => output.ok), [true, false, false, true, false, false], result.stderr);
assert.equal(outputs[0].score, "149/1");
assert.equal(outputs[3].score, "99/100");
assert.equal(result.status, 1, "The checker must exit unsuccessfully when any certificate is rejected");
console.log("Six Lean risk-certificate regressions passed, including four expected rejections.");

import assert from "node:assert/strict";
import test from "node:test";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";

test("replay consumes its first row and all declared ticks, including terminal close metrics", async () => {
  const directory = await mkdtemp(path.join(tmpdir(), "oracle-crra-test-"));
  const input = path.join(directory, "forecast.jsonl");
  const rows = [1000, 2000, 3000].map((timestamp) => ({ timestamp, observedPrice: 100,
    realizedNextTimestamp: timestamp + 1000, realizedNextPrice: 100,
    tree: { children: [{ probability: 1, price: 100, children: [{ probability: 1, price: 100 }] }] },
  }));
  await writeFile(input, rows.map((row) => JSON.stringify(row)).join("\n") + "\n");
  const result = spawnSync(process.execPath, ["--conditions=development", "--import", "tsx", "scripts/backtest-oracle-crra.ts",
    "--forecasts", input, "--output", directory, "--count", "3", "--certificates", "1"], { encoding: "utf8", windowsHide: true });
  assert.equal(result.status, 0, result.stderr);
  const report = JSON.parse(await readFile(path.join(directory, "report.json"), "utf8"));
  assert.equal(report.rows, 3);
  assert.equal(report.firstTimestamp, 1000);
  assert.equal(report.terminalTimestamp, 4000);
  assert.equal(report.terminalWealth, 10000);
  assert.equal(report.gamma, 0);
  assert.equal(report.terminalUtility, 9999);
  const certificate = JSON.parse((await readFile(path.join(directory, "decisions.jsonl"), "utf8")).trim());
  assert.equal(certificate.riskAversion, 0);
  assert(report.buyHoldBaselineWealth < report.cashBaselineWealth);
  assert.match(report.evaluatedForecastPrefixSha256, /^[a-f0-9]{64}$/);
});

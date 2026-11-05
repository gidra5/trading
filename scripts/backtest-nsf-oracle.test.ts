import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { mkdtemp, readFile, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import path from "node:path";
import { spawnSync } from "node:child_process";
import test from "node:test";

test("NSF replay reports infeasibility when the realized terminal price cannot be closed", async () => {
  for (const actualPrice of [110, 300]) {
    const directory = await mkdtemp(path.join(tmpdir(), "oracle-flat-replay-"));
    const forecast = JSON.stringify({ index: 0, timestamp: 1000, price: 100, terminalTimestamp: 2000,
      tree: { children: [{ probability: 1, price: 110 }] } }) + "\n";
    const marks = JSON.stringify({ startTimestamp: 1000, prices: [100, actualPrice] });
    const sha = (text: string) => createHash("sha256").update(text).digest("hex");
    await writeFile(path.join(directory, "forecasts.jsonl"), forecast);
    await writeFile(path.join(directory, "marks.json"), marks);
    await writeFile(path.join(directory, "forecast-plan.json"), JSON.stringify({ terminalTimestamp: 2000,
      forecastSha256: sha(forecast), marksSha256: sha(marks) }));
    const out = path.join(directory, "result");
    const result = spawnSync(process.execPath, ["--import", "tsx", "scripts/backtest-nsf-oracle.ts",
      "--source", directory, "--output", out, "--action-level", "3", "--limits", "0"],
      { encoding: "utf8", windowsHide: true });
    const report = JSON.parse(await readFile(path.join(out, "report.json"), "utf8"));
    assert.equal(report.terminalRequirement, "zero-asset-and-no-orders");
    assert.equal(report.reachedHorizon, true);
    if (actualPrice === 110) {
      assert.equal(result.status, 0, result.stderr);
      assert.equal(report.complete, true);
      assert.equal(report.residualAsset, 0);
      assert(report.terminalWealth > 10400);
    } else {
      assert.equal(result.status, 1);
      assert.equal(report.complete, false);
      assert.match(report.terminalFailure, /terminal-exposure/);
      assert.equal(report.terminalWealth, null);
      assert.equal(report.returnPercent, null);
      assert(report.residualAsset > 0);
    }
  }
});

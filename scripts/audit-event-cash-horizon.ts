import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { createHash } from "node:crypto";
import { restoreEventPolicy } from "../packages/bot-algo/src/event-log-policy.js";
import { eventCashHorizon } from "../packages/bot-algo/src/event-cash-horizon.js";
const arg = (key: string) => { const i = process.argv.indexOf(`--${key}`); return i < 0 ? "" : process.argv[i + 1]; };
const root = path.resolve(__dirname, ".."), output = path.resolve(root, "data/benchmarks", arg("output"));
assert.ok(arg("sources") && arg("output") && !fs.existsSync(output), "Specify --sources and new --output");
const sources = arg("sources").split(",").map(s => path.resolve(root, "data/benchmarks", s));
const maximumDepth = Number(arg("depth") || 10000);
fs.mkdirSync(output, { recursive: true });
const save = (f: string, value: unknown) => fs.writeFileSync(path.join(output, f), JSON.stringify(value, null, 2));
save("config.json", { contract: "event-cash-horizon-audit-v1", sources, maximumDepth,
  method: "Compute the forecast-measure terminal-price martingale backwards. A shadow price inside the fee band at every remaining depth is a sufficient upper certificate of zero expected log growth from cash. Retain joint return/successor dependence and actual costs. A failed certificate does not prove profitable trading. No fitting or next-open-execution guarantee.",
  reference: "https://www.mat.univie.ac.at/~schachermayer/pubs/preprnts/prpr0156.pdf" });
save("sources.json", Object.fromEntries(["scripts/audit-event-cash-horizon.ts", "packages/bot-algo/src/event-cash-horizon.ts"]
  .map(f => [f, fs.readFileSync(path.join(root, f), "utf8")])));
const results = sources.map((source, i) => {
  const bytes = fs.readFileSync(path.join(source, "model.json")), p = restoreEventPolicy(JSON.parse(bytes.toString()));
  const started = performance.now(), result = eventCashHorizon(p.model, p.costs, maximumDepth), elapsedSec = (performance.now() - started) / 1000;
  save(`source-${i}-bounds.json`, result);
  return { source, modelHash: createHash("sha256").update(bytes).digest("hex"), verifiedDepth: result.verifiedDepth,
    firstUnverified: result.stoppedAtFirstFailure ? result.rows.at(-1) : null, elapsedSec };
});
save("summary.json", { results }); console.log(JSON.stringify({ results }));

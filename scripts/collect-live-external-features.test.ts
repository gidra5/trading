import assert from "node:assert/strict";
import test from "node:test";
import fsp from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { run } from "./collect-live-external-features.ts";
import { summarizeGdeltGkg, summarizeGdeltNgrams } from "./lib/external-live-data.ts";

test("GDELT unigram summaries normalize crypto intensity", () => {
  const summary = summarizeGdeltNgrams([
    "20260101000000\tENGLISH\tbitcoin\t4",
    "20260101000000\tENGLISH\tmarket\t6",
    "20260101000000\tSPANISH\tbitcoin\t100",
  ].join("\n"));
  assert.equal(summary.englishTokenCount, 10);
  assert.equal(summary.cryptoTermCount, 4);
  assert.equal(summary.cryptoTermsPerMillion, 400_000);
});

test("GDELT GKG summaries retain matching URLs and tone", () => {
  const fields = new Array(16).fill("");
  fields[0] = "record";
  fields[1] = "20260101000000";
  fields[3] = "example.com";
  fields[4] = "https://example.com/bitcoin-news";
  fields[7] = "ECON_CRYPTOCURRENCY";
  fields[15] = "-2,3,5,8";
  const summary = summarizeGdeltGkg(fields.join("\t"));
  assert.equal(summary.storyCount, 1);
  assert.equal(summary.sourceCount, 1);
  assert.equal(summary.meanTone, -2);
  assert.equal(summary.matches[0]!.url, fields[4]);
});

test("bounded collection aborts slow supplemental requests before closing the sink", async () => {
  const root = await fsp.mkdtemp(path.join(os.tmpdir(), "trading-collector-stop-"));
  const previousFetch = globalThis.fetch;
  let requested = 0;
  let aborted = 0;
  globalThis.fetch = ((_input: unknown, init?: RequestInit) => {
    requested++;
    return new Promise((_resolve, reject) => {
      init?.signal?.addEventListener("abort", () => { aborted++; reject(init.signal!.reason); }, { once: true });
    });
  }) as typeof fetch;
  const start = Date.now();
  try {
    await run(["--rest-only", "--duration-seconds", "0.05", "--output-dir", root]);
    assert.ok(requested >= 4);
    assert.equal(aborted, requested);
    assert.ok(Date.now() - start < 2_000);
  } finally {
    globalThis.fetch = previousFetch;
    await fsp.rm(root, { recursive: true, force: true });
  }
});

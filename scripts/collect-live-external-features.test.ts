import assert from "node:assert/strict";
import test from "node:test";
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

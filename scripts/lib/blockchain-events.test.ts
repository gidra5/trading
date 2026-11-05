import assert from "node:assert/strict";
import test from "node:test";
import { collectChainStep, createChainAdapter, validateChainConfigs, type CapturedBlock, type ChainAdapter, type ChainConfig, type ChainState } from "./blockchain-events.ts";

const config: ChainConfig = { id: "test", adapter: "evm", chainId: "0x1", url: "http://localhost", startHeight: 0 };
const block = (height: number, branch = "a") => ({ height, hash: `${branch}${height}`, parentHash: height ? `${branch}${height - 1}` : "zero", payload: { exact: "0001.23000", transaction: "900719925474099312345" } });

test("all committed blocks are ordered; durable sink flush precedes cursor advancement", async () => {
  const events: string[] = [];
  const adapter: ChainAdapter = { head: async () => 2, identity: async (height) => block(height), capture: async (height) => block(height) };
  const state = await collectChainStep(config, adapter, { version: 1, chain: "test", tail: [] }, {
    write: (_source, value) => { events.push(`block:${(value as any).height}`); }, flush: async () => { events.push("flush"); },
  }, new AbortController().signal, async (next) => { events.push(`cursor:${next.tail.at(-1)!.height}`); });
  assert.deepEqual(events, ["block:0", "flush", "cursor:0", "block:1", "flush", "cursor:1", "block:2", "flush", "cursor:2"]);
  assert.equal(state.tail.at(-1)!.height, 2);
});

test("reorganizations append detach observations and retain canonical order", async () => {
  const prior: ChainState = { version: 1, chain: "test", tail: [block(0), block(1), block(2)] };
  const canonical = [block(0), { ...block(1, "b"), parentHash: "a0" }, block(2, "b")];
  const output: any[] = [];
  const adapter: ChainAdapter = { head: async () => 2, identity: async (height) => canonical[height]!, capture: async (height) => canonical[height]! };
  const next = await collectChainStep(config, adapter, prior, { write: (_source, value) => { output.push(value); }, flush: async () => {} }, new AbortController().signal, async () => {});
  assert.equal(output[0].kind, "reorg");
  assert.deepEqual(output[0].detached.map((item: any) => item.hash), ["a1", "a2"]);
  assert.deepEqual(next.tail.map((item) => item.hash), ["a0", "b1", "b2"]);
  assert.equal(prior.tail[1]!.hash, "a1");
});

test("provider lag or an unavailable saved block cannot detach durable history", async () => {
  const state: ChainState = { version: 1, chain: "test", tail: [block(0), block(1)] };
  let head = 0;
  const adapter: ChainAdapter = { head: async () => head, identity: async () => null, capture: async (height) => block(height) };
  const sink = { write: () => assert.fail("unexpected journal write"), flush: async () => {} };
  const collect = () => collectChainStep(config, adapter, state, sink, new AbortController().signal,
    async () => { assert.fail("unexpected cursor write"); });
  await assert.rejects(collect(), /behind the durable cursor/);
  head = 2;
  await assert.rejects(collect(), /cannot identify the durable cursor/);
});

test("missing block data or a failed flush never commits that block's cursor", async () => {
  let saves = 0;
  const adapter: ChainAdapter = { head: async () => 0, identity: async (height) => block(height), capture: async (height) => block(height) };
  await assert.rejects(collectChainStep(config, adapter, { version: 1, chain: "test", tail: [] }, {
    write: () => {}, flush: async () => { throw new Error("disk failure"); },
  }, new AbortController().signal, async () => { saves++; }), /disk failure/);
  assert.equal(saves, 0);
});

test("implicit starting boundary is durable even if the first block is unavailable", async () => {
  const startAtTip = { ...config, startHeight: undefined };
  let state: ChainState = { version: 1, chain: "test", tail: [] };
  let head = 50, available = false;
  const adapter: ChainAdapter = {
    head: async () => head,
    identity: async (height) => block(height),
    capture: async (height) => {
      if (!available) throw new Error("unavailable");
      return block(height);
    },
  };
  const records: unknown[] = [];
  const sink = { write: (_source: string, record: unknown) => { records.push(record); }, flush: async () => {} };
  const save = async (next: ChainState) => { state = next; };
  const signal = new AbortController().signal;
  await assert.rejects(collectChainStep(startAtTip, adapter, state, sink, signal, save), /unavailable/);
  assert.equal(state.initialHeight, 50);
  assert.equal(state.tail.length, 0);
  head = 52; available = true;
  await collectChainStep(startAtTip, adapter, state, sink, signal, save);
  assert.deepEqual(state.tail.map(({ height }) => height), [49, 50, 51, 52]);
  assert.deepEqual(records.map((record: any) => record.kind), ["capture_start", "ancestry_anchor", "block", "block", "block"]);
});

test("sparse capture journals only parent-proven skipped slots and commits a successful prefix in order", async () => {
  const sparse = { ...config, id: "sparse", startHeight: 8 };
  const sparseBlock = (height: number, parentHeight: number): CapturedBlock =>
    ({ ...block(height), parentHeight, parentHash: `a${parentHeight}` });
  const known = new Map([[6, sparseBlock(6, 5)], [9, sparseBlock(9, 6)], [10, sparseBlock(10, 9)], [12, sparseBlock(12, 10)]]);
  let state: ChainState = { version: 1, chain: "sparse", tail: [] };
  let failing = true, heights = [9, 10, 12];
  const adapter: ChainAdapter = {
    head: async () => 12, identity: async (height) => known.get(height) ?? null,
    findHeights: async () => heights,
    capture: async (height) => {
      if (failing && height === 10) throw new Error("unavailable");
      return known.get(height)!;
    },
  };
  const records: any[] = [];
  const sink = { write: (_source: string, record: unknown) => { records.push(record); }, flush: async () => {} };
  const save = async (next: ChainState) => { state = next; };
  const signal = new AbortController().signal;
  await assert.rejects(collectChainStep(sparse, adapter, state, sink, signal, save), /unavailable/);
  assert.deepEqual(state.tail.map(({ height }) => height), [6, 9]);
  assert.deepEqual(records.map((record) => record.kind), ["ancestry_anchor", "skipped_slots", "block"]);
  assert.deepEqual([records[1].fromHeight, records[1].throughHeight], [8, 8]);
  failing = false; heights = [12];
  await assert.rejects(collectChainStep(sparse, adapter, state, sink, signal, save), /last captured parent/);
  assert.equal(records.length, 3);
  assert.equal(state.tail.at(-1)?.height, 9);
  heights = [10, 12];
  await collectChainStep(sparse, adapter, state, sink, signal, save);
  assert.deepEqual(state.tail.map(({ height }) => height), [6, 9, 10, 12]);
  assert.deepEqual([records[4].fromHeight, records[4].throughHeight], [11, 11]);
});

test("identity and endpoint declarations cannot silently select another network", () => {
  assert.doesNotThrow(() => validateChainConfigs([config]));
  assert.throws(() => validateChainConfigs([{ ...config, chainId: undefined }]), /chainId/);
  assert.throws(() => validateChainConfigs([config, config]), /unique/);
  assert.throws(() => validateChainConfigs([{ ...config, adapter: "bitcoin-esplora" }]), /genesisHash/);
});

test("EVM receipts retain raw fields and incomplete/reorged receipts fail capture", async () => {
  const original = globalThis.fetch;
  const hash = `0x${"a".repeat(64)}`, tx = `0x${"b".repeat(64)}`;
  let wrongBlock = false;
  globalThis.fetch = async (_input, init) => {
    const request = JSON.parse(String(init?.body));
    const result = request.method === "eth_chainId" ? "0x1" : request.method === "eth_getBlockByNumber"
      ? { number: "0x0", hash, parentHash: `0x${"0".repeat(64)}`, transactions: [{ hash: tx, value: "0x10000000000000001" }] }
      : { transactionHash: tx, blockHash: wrongBlock ? `0x${"c".repeat(64)}` : hash, logs: [{ data: "0x00001200", unknown: "0001.23000" }] };
    return new Response(JSON.stringify({ jsonrpc: "2.0", id: request.id, result }));
  };
  try {
    const adapter = createChainAdapter(config);
    const captured = await adapter.capture(0, new AbortController().signal);
    assert.match(String(captured.payload.block), /0x10000000000000001/);
    assert.match((captured.payload.receipts as string[])[0]!, /0001.23000/);
    wrongBlock = true;
    await assert.rejects(adapter.capture(0, new AbortController().signal), /different block/);
  } finally { globalThis.fetch = original; }
});

import assert from "node:assert/strict";
import test from "node:test";
import type { ChainConfig, ChainRpc } from "./blockchain-events.ts";
import { createSolanaAdapter, isSolanaHash } from "./blockchain-solana.ts";

const genesisHash = "5eykt4UsFv8P8NJdTREpY1vzqKqZKvdpKuc147dw2N9d";
const blockhash = "J8mfvgG5xJ1hfGVxcrJ9f3BwdYPuXNfY3cJzKMwHyFTe";
const previousBlockhash = "9gAzd1Mv43CptYG8cjy5bE7uNq6P9SFNZFTKToHcQHGD";
const config: ChainConfig = { id: "solana", adapter: "solana", urlEnv: "SOLANA_RPC_URL", genesisHash };
const signal = new AbortController().signal;

test("Solana verifies the cluster and keeps finalized block and transaction metadata exactly as received", async () => {
  const raw = ` {"jsonrpc":"2.0", "id":99,"result":{"blockhash":"${blockhash}","previousBlockhash":"${previousBlockhash}",` +
    `"parentSlot":97,"blockHeight":80,"blockTime":null,"rewards":[],"transactions":[` +
    `{"meta":{"preBalances":[18446744073709551615],"unknown":1.2300},"transaction":["gQE=","base64"],"version":1},` +
    `{"meta":null,"transaction":["AAI=","base64"],"version":"legacy"}]}}\n`;
  const calls: Array<readonly [string, readonly unknown[]]> = [];
  const rpc: ChainRpc = async (method, params, receivedSignal) => {
    assert.equal(receivedSignal, signal);
    calls.push([method, params]);
    if (method === "getGenesisHash") return { raw: "", result: genesisHash };
    if (method === "getSlot") return { raw: "", result: 105 };
    if (method === "getBlocksWithLimit") return { raw: "", result: [100] };
    return { raw, result: JSON.parse(raw).result };
  };
  const adapter = createSolanaAdapter(config, rpc);
  assert.equal(await adapter.head(signal), 105);
  assert.deepEqual(await adapter.findHeights!(98, 105, 8, signal), [100]);
  assert.deepEqual(await adapter.findHeights!(106, 105, 8, signal), []);
  assert.deepEqual(await adapter.findHeights!(98, 99, 8, signal), []);
  const identity = { height: 100, parentHeight: 97, hash: blockhash, parentHash: previousBlockhash };
  assert.deepEqual(await adapter.identity(100, signal), identity);
  const captured = await adapter.capture(100, signal);
  assert.deepEqual({ ...captured, payload: undefined }, { ...identity, payload: undefined });
  assert.equal(captured.payload.block, raw);
  assert.equal(captured.payload.transactionCount, 2);
  assert.equal(captured.payload.rawBytes, Buffer.byteLength(raw));
  assert.deepEqual(calls, [
    ["getGenesisHash", []], ["getSlot", [{ commitment: "finalized" }]],
    ["getBlocksWithLimit", [98, 8, { commitment: "finalized", minContextSlot: 105 }]],
    ["getBlocksWithLimit", [98, 2, { commitment: "finalized", minContextSlot: 99 }]],
    ["getBlock", [100, { commitment: "finalized", encoding: "base64", transactionDetails: "none", rewards: false, maxSupportedTransactionVersion: 1 }]],
    ["getBlock", [100, { commitment: "finalized", encoding: "base64", transactionDetails: "full", rewards: true, maxSupportedTransactionVersion: 1 }]],
  ]);
});

test("Solana rejects malformed or wrong-cluster identities and does not convert unavailable blocks into skips", async () => {
  assert.equal(isSolanaHash(genesisHash), true);
  assert.equal(isSolanaHash("1".repeat(32)), true);
  assert.equal(isSolanaHash("2".repeat(32)), false);
  assert.equal(isSolanaHash("z".repeat(44)), false);
  assert.throws(() => createSolanaAdapter({ ...config, genesisHash: undefined }, async () => { throw new Error("unexpected RPC"); }), /genesisHash/);

  let cluster = previousBlockhash;
  let result: unknown = null;
  const adapter = createSolanaAdapter(config, async (method) => ({ raw: "", result: method === "getGenesisHash" ? cluster : result }));
  await assert.rejects(adapter.head(signal), /genesis hash mismatch/);
  cluster = genesisHash;
  await assert.rejects(adapter.identity(100, signal), /block unavailable/);
  await assert.rejects(adapter.capture(100, signal), /block unavailable/);
  result = [96];
  await assert.rejects(adapter.findHeights!(97, 105, 8, signal), /outside the requested range/);
  result = [100, 98];
  await assert.rejects(adapter.findHeights!(97, 105, 8, signal), /out of order/);
  result = [];
  assert.deepEqual(await adapter.findHeights!(97, 105, 8, signal), []);
  result = { blockhash, previousBlockhash, parentSlot: 100 };
  await assert.rejects(adapter.identity(100, signal), /block identity/);
  result = { blockhash, previousBlockhash, parentSlot: 97, transactions: [{ transaction: ["AAE=", "base64"] }], rewards: [] };
  await assert.rejects(adapter.capture(100, signal), /Incomplete Solana transactions/);
});

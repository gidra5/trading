import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import type { ChainHttpJson } from "./blockchain-events.ts";
import { createTronAdapter } from "./blockchain-tron.ts";

const blockId = (height: number, digit = "a") => height.toString(16).padStart(16, "0") + digit.repeat(48);
const genesisHash = blockId(0);
const timestamp = 1_793_887_347_000;
const trieRoot = "f".repeat(64);
const rawDataHex = "abcd";
const txId = createHash("sha256").update(rawDataHex, "hex").digest("hex");
const transaction = { txID: txId, raw_data_hex: rawDataHex };
const block = {
  blockID: blockId(1), block_header: { raw_data: { number: 1, parentHash: genesisHash, txTrieRoot: trieRoot, timestamp } },
  transactions: [transaction],
};
const genesis = {
  blockID: genesisHash, block_header: { raw_data: { parentHash: "e".repeat(64), txTrieRoot: trieRoot } }, transactions: [transaction],
};
const receipt = { id: txId, blockNumber: 1, blockTimeStamp: timestamp };
const signal = new AbortController().signal;

test("TRON preserves exact block and matched solidified receipts, including genesis without execution", async () => {
  const blockRaw = ` ${JSON.stringify(block)}\n`;
  const receiptsRaw = `[{"id":"${txId}","blockNumber":1,"blockTimeStamp":${timestamp},"fee":18446744073709551615}]\n`;
  const calls: Array<readonly [string, unknown]> = [];
  const http: ChainHttpJson = async (route, body, receivedSignal) => {
    assert.equal(receivedSignal, signal);
    calls.push([route, body]);
    const height = (body as { num?: number }).num;
    const raw = route.endsWith("gettransactioninfobyblocknum") ? height === 0 ? "{}\n" : receiptsRaw
      : height === 0 ? JSON.stringify(genesis) : blockRaw;
    return { raw, result: JSON.parse(raw) };
  };
  const adapter = createTronAdapter({ genesisHash }, http);
  assert.equal(await adapter.head(signal), 1);
  assert.deepEqual(await adapter.identity(1, signal), { height: 1, hash: block.blockID, parentHash: genesisHash });
  const captured = await adapter.capture(1, signal);
  assert.equal(captured.payload.block, blockRaw);
  assert.equal(captured.payload.receipts, receiptsRaw);
  assert.equal(captured.payload.receiptCount, 1);
  assert.equal(captured.payload.transactionCount, 1);
  const allocations = await adapter.capture(0, signal);
  assert.equal(allocations.payload.receipts, "{}\n");
  assert.equal(allocations.payload.receiptCount, 0);
  assert.equal(allocations.payload.transactionCount, 1);
  assert.deepEqual(calls, [
    ["/walletsolidity/getblockbynum", { num: 0 }], ["/walletsolidity/getnowblock", {}],
    ["/walletsolidity/getblockbynum", { num: 1 }], ["/walletsolidity/getblockbynum", { num: 1 }],
    ["/walletsolidity/gettransactioninfobyblocknum", { num: 1 }], ["/walletsolidity/getblockbynum", { num: 0 }],
    ["/walletsolidity/gettransactioninfobyblocknum", { num: 0 }],
  ]);
});

test("TRON rejects partial or unrelated data and recognizes a genuine empty block", async () => {
  let blockResult: unknown = block, receiptsResult: unknown = [];
  const http: ChainHttpJson = async (route, body) => {
    const result = (body as { num?: number }).num === 0 ? genesis
      : route.endsWith("gettransactioninfobyblocknum") ? receiptsResult : blockResult;
    return { raw: JSON.stringify(result), result };
  };
  const adapter = createTronAdapter({ genesisHash }, http);
  await assert.rejects(createTronAdapter({ genesisHash: blockId(0, "b") }, http).head(signal), /genesis hash mismatch/);
  await assert.rejects(adapter.capture(1, signal), /Incomplete TRON receipt set/);
  receiptsResult = [{ ...receipt, blockNumber: 2 }];
  await assert.rejects(adapter.capture(1, signal), /receipt identity mismatch/);
  receiptsResult = [{ ...receipt, id: "c".repeat(64) }];
  await assert.rejects(adapter.capture(1, signal), /receipt identity mismatch/);
  receiptsResult = [{ ...receipt, blockTimeStamp: timestamp + 3_000 }];
  await assert.rejects(adapter.capture(1, signal), /receipt identity mismatch/);

  blockResult = { ...block, transactions: [{ ...transaction, raw_data_hex: "abce" }] };
  await assert.rejects(adapter.capture(1, signal), /transaction ID mismatch/);
  blockResult = { ...block, transactions: undefined };
  await assert.rejects(adapter.capture(1, signal), /transaction list does not match/);
  blockResult = { ...block, transactions: undefined,
    block_header: { raw_data: { ...block.block_header.raw_data, txTrieRoot: "0".repeat(64) } } };
  receiptsResult = {};
  await assert.rejects(adapter.capture(1, signal), /Incomplete TRON receipt set/);
  receiptsResult = [];
  const empty = await adapter.capture(1, signal);
  assert.equal(empty.payload.transactionCount, 0);
  assert.equal(empty.payload.receiptCount, 0);
  blockResult = {};
  await assert.rejects(adapter.identity(1, signal), /block unavailable/);
});

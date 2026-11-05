import assert from "node:assert/strict";
import test from "node:test";
import type { ChainRpc } from "./blockchain-events.ts";
import { captureEvmBlockReceipts } from "./blockchain-evm-receipts.ts";

const blockHash = `0x${"a".repeat(64)}`;
const firstTx = `0x${"b".repeat(64)}`;
const secondTx = `0x${"c".repeat(64)}`;
const block = { height: 9, hash: blockHash, parentHash: `0x${"0".repeat(64)}` };
const transactions = [{ hash: firstTx }, { hash: secondTx }] as const;
const receipts = [
  { transactionHash: secondTx, blockHash, logs: [{ transactionHash: secondTx, blockHash, data: "0x0001" }] },
  { transactionHash: firstTx, blockHash, logs: [] },
] as const;
const signal = new AbortController().signal;

test("block receipt capture requests by hash and preserves raw receipt response", async () => {
  const raw = `{ "result" : ${JSON.stringify(receipts)}, "extra" : 900719925474099312345 }\n`;
  const rpc: ChainRpc = async (method, params, requestSignal) => {
    assert.equal(method, "eth_getBlockReceipts");
    assert.deepEqual(params, [blockHash]);
    assert.equal(requestSignal, signal);
    return { raw, result: JSON.parse(raw).result };
  };
  assert.equal(await captureEvmBlockReceipts(rpc, block, transactions, signal), raw);
});

test("block receipt capture rejects missing, duplicate, cross-block and inconsistent receipt data", async () => {
  const cases = [
    [null, /missing or incomplete/],
    [[receipts[0]], /missing or incomplete/],
    [[receipts[0], receipts[0]], /unexpected transaction/],
    [[receipts[0], { ...receipts[1], blockHash: block.parentHash }], /unexpected transaction/],
    [[{ ...receipts[0], logs: undefined }, receipts[1]], /logs are missing/],
    [[{ ...receipts[0], logs: [{ transactionHash: firstTx, blockHash }] }, receipts[1]], /inconsistent identity/],
  ] as const;
  for (const [result, error] of cases) {
    const rpc: ChainRpc = async () => ({ raw: JSON.stringify({ result }), result });
    await assert.rejects(captureEvmBlockReceipts(rpc, block, transactions, signal), error);
  }
});

test("a block with zero transactions still requires an explicit empty receipt set", async () => {
  const raw = '{"jsonrpc":"2.0","id":1,"result":[]}';
  assert.equal(await captureEvmBlockReceipts(async () => ({ raw, result: [] }), block, [], signal), raw);
  await assert.rejects(captureEvmBlockReceipts(async () => ({ raw, result: null }), block, [], signal), /missing or incomplete/);
});

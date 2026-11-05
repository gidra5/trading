import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import test from "node:test";
import { createXrplAdapter } from "./blockchain-xrpl.ts";

const sha512Half = (prefix: string, bytes: Buffer) => createHash("sha512").update(prefix).update(bytes).digest("hex").slice(0, 64);

test("XRP Ledger verifies validated header and transaction hashes while retaining raw binary metadata", async () => {
  const header = Buffer.alloc(118, 0x12);
  header.writeUInt32BE(88, 0);
  const ledgerHash = sha512Half("LWR\0", header), txBlob = "1234", meta = "01000200";
  const txHash = sha512Half("TXN\0", Buffer.from(txBlob, "hex"));
  const tx = { hash: txHash, tx_blob: txBlob, meta_blob: meta };
  let validated = true, ledgerData = header.toString("hex"), networkId = 1;
  let transactions: readonly unknown[] = [tx];
  let lastRaw = "";
  const adapter = createXrplAdapter({ networkId: 0 }, async (method, params) => {
    const result = method === "server_info" ? { status: "success", info: { network_id: networkId } } : {
      status: "success", validated, ledger_index: 88, ledger_hash: ledgerHash.toUpperCase(),
      ledger: { closed: true, ledger_data: ledgerData, transactions },
    };
    if (method === "ledger") assert.equal((params[0] as { binary: boolean }).binary, true);
    lastRaw = ` { "result": ${JSON.stringify(result)} }\n`;
    return { result, raw: lastRaw };
  });
  const signal = new AbortController().signal;
  await assert.rejects(adapter.head(signal), /network identity mismatch/);
  networkId = 0;
  assert.equal(await adapter.head(signal), 88);
  const captured = await adapter.capture(88, signal);
  assert.equal(captured.hash, ledgerHash);
  assert.equal(captured.parentHash, "12".repeat(32));
  assert.equal(captured.payload.ledger, lastRaw);
  assert.equal(captured.payload.transactionCount, 1);
  validated = false;
  await assert.rejects(adapter.capture(88, signal), /unvalidated/);
  validated = true; ledgerData = "00" + ledgerData.slice(2, -2) + "00";
  await assert.rejects(adapter.identity(88, signal), /header hash or height mismatch/);
  ledgerData = header.toString("hex"); transactions = [];
  await assert.rejects(adapter.capture(88, signal), /transaction set missing/);
  transactions = [tx, tx];
  await assert.rejects(adapter.capture(88, signal), /Invalid XRP Ledger transaction/);
  transactions = [{ ...tx, meta_blob: undefined }];
  await assert.rejects(adapter.capture(88, signal), /Invalid XRP Ledger transaction/);
});

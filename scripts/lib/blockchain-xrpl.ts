import { createHash } from "node:crypto";
import type { BlockIdentity, ChainAdapter, ChainConfig, ChainRpc } from "./blockchain-events.ts";

const hex256 = /^[0-9a-f]{64}$/i;
const hexBlob = /^(?:[0-9a-f]{2})+$/i;
const emptyHash = "0".repeat(64);

function record(value: unknown): Readonly<Record<string, unknown>> {
  if (typeof value !== "object" || value === null || Array.isArray(value)) throw new Error("Invalid XRP Ledger response");
  return value as Readonly<Record<string, unknown>>;
}

function hashWithPrefix(prefix: string, value: Buffer): string {
  return createHash("sha512").update(prefix, "ascii").update(value).digest("hex").slice(0, 64);
}

export function createXrplAdapter(config: Pick<ChainConfig, "networkId">, rpc: ChainRpc): ChainAdapter {
  let verified = false;
  async function verify(signal: AbortSignal): Promise<void> {
    if (verified) return;
    const reply = record((await rpc("server_info", [{ api_version: 2 }], signal)).result);
    if (reply.status !== "success" || record(reply.info).network_id !== config.networkId)
      throw new Error("XRP Ledger network identity mismatch");
    verified = true;
  }

  async function ledgerAt(requested: number | "validated", full: boolean, signal: AbortSignal) {
    await verify(signal);
    const { raw, result } = await rpc("ledger", [{
      ledger_index: requested, api_version: 2, binary: true, transactions: full, expand: full,
    }], signal);
    const reply = record(result);
    if (reply.status !== "success" || reply.validated !== true)
      throw new Error(`XRP Ledger response is unavailable or unvalidated at ${requested}`);
    const ledger = record(reply.ledger);
    const { ledger_index: height, ledger_hash: reportedHash } = reply;
    if (typeof height !== "number" || !Number.isSafeInteger(height) || height < 0
      || (requested !== "validated" && height !== requested) || typeof reportedHash !== "string" || !hex256.test(reportedHash)
      || ledger.closed !== true || typeof ledger.ledger_data !== "string" || ledger.ledger_data.length !== 236 || !hexBlob.test(ledger.ledger_data))
      throw new Error(`Invalid XRP Ledger header at ${requested}`);
    const bytes = Buffer.from(ledger.ledger_data, "hex"), hash = reportedHash.toLowerCase();
    // Rippled ledger-master hash prefix, followed by the serialized 118-byte ledger header.
    if (bytes.readUInt32BE(0) !== height || hashWithPrefix("LWR\0", bytes) !== hash)
      throw new Error(`XRP Ledger header hash or height mismatch at ${requested}`);
    const identity: BlockIdentity = { height, hash, parentHash: bytes.subarray(12, 44).toString("hex") };
    return { raw, identity, transactions: ledger.transactions, transactionRoot: bytes.subarray(44, 76).toString("hex") };
  }

  return {
    head: async (signal) => (await ledgerAt("validated", false, signal)).identity.height,
    identity: async (height, signal) => (await ledgerAt(height, false, signal)).identity,
    capture: async (height, signal) => {
      const { raw, identity, transactions, transactionRoot } = await ledgerAt(height, true, signal);
      const entries = transactions === undefined && transactionRoot === emptyHash ? [] : transactions;
      if (!Array.isArray(entries) || (entries.length === 0) !== (transactionRoot === emptyHash))
        throw new Error(`XRP Ledger transaction set missing at ${height}`);
      const hashes = new Set<string>();
      for (const entry of entries) {
        const { hash, tx_blob: tx, meta_blob: meta } = record(entry);
        if (typeof hash !== "string" || !hex256.test(hash) || typeof tx !== "string" || !hexBlob.test(tx)
          || typeof meta !== "string" || !hexBlob.test(meta) || hashWithPrefix("TXN\0", Buffer.from(tx, "hex")) !== hash.toLowerCase()
          || hashes.has(hash.toLowerCase())) throw new Error(`Invalid XRP Ledger transaction or metadata at ${height}`);
        hashes.add(hash.toLowerCase());
      }
      return { ...identity, payload: { format: "xrpl-json-rpc-raw", ledger: raw, transactionCount: entries.length,
        rawBytes: Buffer.byteLength(raw), coverage: "Validated ledger header and provider-returned serialized transactions with execution metadata" } };
    },
  };
}

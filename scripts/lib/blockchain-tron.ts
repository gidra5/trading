import { createHash } from "node:crypto";
import type { BlockIdentity, ChainAdapter, ChainConfig, ChainHttpJson } from "./blockchain-events.ts";

const blockRoute = "/walletsolidity/getblockbynum";
const zeroHash = "0".repeat(64);

function isHexHash(value: unknown): value is string {
  return typeof value === "string" && /^[0-9a-f]{64}$/i.test(value);
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function tronBlock(result: unknown): {
  readonly identity: BlockIdentity;
  readonly header: Readonly<Record<string, unknown>>;
  readonly result: Readonly<Record<string, unknown>>;
} {
  if (!isRecord(result) || !isRecord(result.block_header) || !isRecord(result.block_header.raw_data)) {
    throw new Error("TRON solidified block unavailable");
  }
  const header = result.block_header.raw_data;
  const height = header.number === undefined ? 0 : header.number;
  if (typeof height !== "number" || !Number.isSafeInteger(height) || height < 0 || !isHexHash(result.blockID)
    || !isHexHash(header.parentHash) || !isHexHash(header.txTrieRoot)) throw new Error("Invalid TRON block identity");
  if (height > 0 && (typeof header.timestamp !== "number" || !Number.isSafeInteger(header.timestamp) || header.timestamp <= 0)) {
    throw new Error("Invalid TRON block timestamp");
  }
  const hash = result.blockID.toLowerCase(), parentHash = header.parentHash.toLowerCase();
  if (BigInt(`0x${hash.slice(0, 16)}`) !== BigInt(height)
    || (height > 0 && BigInt(`0x${parentHash.slice(0, 16)}`) !== BigInt(height - 1))) {
    throw new Error("TRON block ID height mismatch");
  }
  return { identity: { height, hash, parentHash }, header, result };
}

function transactionIds(result: Readonly<Record<string, unknown>>, trieRoot: unknown): Set<string> {
  // Protobuf omits empty repeated fields. A zero transaction root independently confirms an empty block.
  const transactions = result.transactions === undefined && trieRoot === zeroHash ? [] : result.transactions;
  if (!Array.isArray(transactions) || (transactions.length === 0) !== (trieRoot === zeroHash)) {
    throw new Error("TRON block transaction list does not match its transaction root");
  }
  const ids = new Set<string>();
  for (const tx of transactions) {
    if (!isRecord(tx) || !isHexHash(tx.txID) || typeof tx.raw_data_hex !== "string" || !/^(?:[0-9a-f]{2})+$/i.test(tx.raw_data_hex)) {
      throw new Error("TRON block transaction body missing");
    }
    const id = tx.txID.toLowerCase();
    if (ids.has(id) || createHash("sha256").update(tx.raw_data_hex, "hex").digest("hex") !== id) {
      throw new Error("TRON transaction ID mismatch or duplicate");
    }
    ids.add(id);
  }
  return ids;
}

export function createTronAdapter(config: Pick<ChainConfig, "genesisHash">, http: ChainHttpJson): ChainAdapter {
  const genesisHash = config.genesisHash?.toLowerCase();
  if (!isHexHash(genesisHash)) throw new Error("Expected TRON genesisHash is required");

  async function blockAt(height: number, signal: AbortSignal) {
    const { raw, result } = await http(blockRoute, { num: height }, signal);
    const block = tronBlock(result);
    if (block.identity.height !== height) throw new Error("TRON block height mismatch");
    return { ...block, raw };
  }

  let verified = false;
  async function verify(signal: AbortSignal): Promise<void> {
    if (verified) return;
    if ((await blockAt(0, signal)).identity.hash !== genesisHash) throw new Error("TRON genesis hash mismatch");
    verified = true;
  }

  return {
    head: async (signal) => {
      await verify(signal);
      return tronBlock((await http("/walletsolidity/getnowblock", {}, signal)).result).identity.height;
    },
    identity: async (height, signal) => {
      await verify(signal);
      return (await blockAt(height, signal)).identity;
    },
    capture: async (height, signal) => {
      await verify(signal);
      const block = await blockAt(height, signal);
      const ids = transactionIds(block.result, block.header.txTrieRoot), transactionCount = ids.size;
      const receipts = await http("/walletsolidity/gettransactioninfobyblocknum", { num: height }, signal);
      // java-tron does not execute the genesis allocations and returns {} for a receipt query at height zero.
      const results = height === 0 && isRecord(receipts.result) && Object.keys(receipts.result).length === 0 ? [] : receipts.result;
      if (!Array.isArray(results) || results.length !== (height === 0 ? 0 : transactionCount)) {
        throw new Error(`Incomplete TRON receipt set at height ${height}`);
      }
      for (const receipt of results) {
        if (!isRecord(receipt) || receipt.blockNumber !== height || receipt.blockTimeStamp !== block.header.timestamp
          || !isHexHash(receipt.id) || !ids.delete(receipt.id.toLowerCase())) {
          throw new Error(`TRON receipt identity mismatch at height ${height}`);
        }
      }
      return { ...block.identity, payload: { format: "tron-solidity-http-raw", block: block.raw, receipts: receipts.raw,
        transactionCount, receiptCount: results.length, rawBytes: Buffer.byteLength(block.raw) + Buffer.byteLength(receipts.raw),
        coverage: height === 0 ? "Genesis allocation records without execution receipts"
          : "Solidified transactions and matching execution receipts; internal transactions are included when provided by the node" } };
    },
  };
}

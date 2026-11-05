import type { BlockIdentity, ChainAdapter, ChainConfig, ChainRpc } from "./blockchain-events.ts";

const alphabet = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz";
const commitment = "finalized";
const maxSupportedTransactionVersion = 1;

export function isSolanaHash(value: unknown): value is string {
  if (typeof value !== "string" || value.length < 32 || value.length > 44) return false;
  let decoded = 0n;
  for (const digit of value) {
    const index = alphabet.indexOf(digit);
    if (index < 0) return false;
    decoded = decoded * 58n + BigInt(index);
  }
  const leadingZeros = value.match(/^1*/)?.[0].length ?? 0;
  const bytes = decoded === 0n ? 0 : Math.ceil(decoded.toString(16).length / 2);
  return leadingZeros + bytes === 32;
}

function slotOf(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0) throw new Error("Invalid Solana slot");
  return value;
}

function isRecord(value: unknown): value is Readonly<Record<string, unknown>> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function hasEncodedTransaction(value: unknown): boolean {
  if (!isRecord(value)) return false;
  const { transaction, meta } = value;
  return Array.isArray(transaction) && transaction.length === 2 && transaction[1] === "base64"
    && typeof transaction[0] === "string" && transaction[0].length > 0 && (meta === null || isRecord(meta));
}

export function createSolanaAdapter(config: ChainConfig, rpc: ChainRpc): ChainAdapter {
  const genesisHash = config.genesisHash;
  if (!isSolanaHash(genesisHash)) throw new Error("Expected Solana genesisHash is required");
  let verified = false;
  async function verify(signal: AbortSignal): Promise<void> {
    if (verified) return;
    if ((await rpc("getGenesisHash", [], signal)).result !== genesisHash) throw new Error("Solana genesis hash mismatch");
    verified = true;
  }

  async function blockAt(height: number, full: boolean, signal: AbortSignal) {
    await verify(signal);
    const { raw, result } = await rpc("getBlock", [height, {
      commitment, encoding: "base64", transactionDetails: full ? "full" : "none", rewards: full,
      maxSupportedTransactionVersion,
    }], signal);
    if (!isRecord(result)) throw new Error(`Finalized Solana block unavailable at slot ${height}`);
    const { blockhash, previousBlockhash } = result;
    const parentHeight = slotOf(result.parentSlot);
    if (!isSolanaHash(blockhash) || !isSolanaHash(previousBlockhash) || (height === 0 ? parentHeight !== 0 : parentHeight >= height)) {
      throw new Error(`Invalid Solana block identity at slot ${height}`);
    }
    const identity: BlockIdentity = { height, hash: blockhash, parentHash: previousBlockhash, parentHeight };
    return { identity, raw, result };
  }

  return {
    head: async (signal) => {
      await verify(signal);
      return slotOf((await rpc("getSlot", [{ commitment }], signal)).result);
    },
    findHeights: async (from, through, limit, signal) => {
      if (from > through) return [];
      await verify(signal);
      const count = Math.min(limit, through - from + 1);
      const { result } = await rpc("getBlocksWithLimit", [from, count, { commitment, minContextSlot: through }], signal);
      if (!Array.isArray(result) || result.length > count) throw new Error("Invalid Solana block slot list");
      const heights = result.map(slotOf);
      if (heights.some((height, index) => height < from || (index > 0 && height <= heights[index - 1]!)))
        throw new Error("Solana block slots are outside the requested range or out of order");
      return heights.filter((height) => height <= through);
    },
    identity: async (height, signal) => (await blockAt(height, false, signal)).identity,
    capture: async (height, signal) => {
      const { identity, raw, result } = await blockAt(height, true, signal);
      if (!Array.isArray(result.transactions) || !result.transactions.every(hasEncodedTransaction) || !Array.isArray(result.rewards)) {
        throw new Error(`Incomplete Solana transactions or rewards at slot ${height}`);
      }
      return { ...identity, payload: { format: "solana-json-rpc-raw", block: raw, commitment,
        transactionEncoding: "base64", transactionCount: result.transactions.length, rawBytes: Buffer.byteLength(raw),
        coverage: "Finalized block with serialized transactions, block rewards and RPC-provided transaction metadata" } };
    },
  };
}

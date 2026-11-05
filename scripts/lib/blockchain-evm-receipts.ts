import type { BlockIdentity, ChainRpc } from "./blockchain-events.ts";

function hashOf(value: unknown): string | null {
  return typeof value === "string" && /^0x[0-9a-f]{64}$/i.test(value) ? value.toLowerCase() : null;
}

type ReceiptFields = {
  readonly blockHash?: unknown;
  readonly transactionHash?: unknown;
  readonly logs?: unknown;
};

/** Preserve the provider's raw block receipt response after validating completeness and identity. */
export async function captureEvmBlockReceipts(rpc: ChainRpc, block: BlockIdentity, transactions: readonly unknown[],
  signal: AbortSignal): Promise<string> {
  const outstanding = new Set<string>();
  for (const transaction of transactions) {
    const transactionHash = hashOf((transaction as { readonly hash?: unknown } | null)?.hash);
    if (!transactionHash || outstanding.has(transactionHash)) throw new Error("Invalid or duplicate EVM block transaction hash");
    outstanding.add(transactionHash);
  }
  const { raw, result } = await rpc("eth_getBlockReceipts", [block.hash], signal);
  if (!Array.isArray(result) || result.length !== outstanding.size) {
    throw new Error("EVM block receipt set is missing or incomplete; retry without advancing cursor");
  }
  for (const receipt of result as readonly (ReceiptFields | null)[]) {
    const transactionHash = hashOf(receipt?.transactionHash);
    if (!transactionHash || !outstanding.delete(transactionHash) || hashOf(receipt?.blockHash) !== block.hash) {
      throw new Error("EVM block receipt has an unexpected transaction or block hash; retry without advancing cursor");
    }
    if (!Array.isArray(receipt?.logs) || (receipt.logs as readonly (ReceiptFields | null)[]).some((log) =>
      hashOf(log?.transactionHash) !== transactionHash || hashOf(log?.blockHash) !== block.hash)) {
      throw new Error("EVM block receipt logs are missing or have inconsistent identity; retry without advancing cursor");
    }
  }
  return raw;
}

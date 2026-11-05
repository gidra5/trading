import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import { captureEvmBlockReceipts } from "./blockchain-evm-receipts.ts";
import { createSolanaAdapter, isSolanaHash } from "./blockchain-solana.ts";
import { createTronAdapter } from "./blockchain-tron.ts";
import { createXrplAdapter } from "./blockchain-xrpl.ts";

export interface ChainConfig {
  readonly id: string;
  readonly adapter: "bitcoin-esplora" | "evm" | "solana" | "tron" | "xrpl";
  readonly url?: string;
  readonly urlEnv?: string;
  readonly chainId?: string;
  readonly genesisHash?: string;
  readonly networkId?: number;
  readonly evmReceiptMode?: "per-transaction" | "block";
  readonly tronApiKeyEnv?: string;
  readonly confirmationLag?: number;
  readonly startHeight?: number;
}
export interface BlockIdentity { readonly height: number; readonly hash: string; readonly parentHash: string; readonly parentHeight?: number; }
export interface CapturedBlock extends BlockIdentity { readonly payload: Readonly<Record<string, unknown>>; }
export type ChainRpc = (method: string, params: readonly unknown[], signal: AbortSignal) => Promise<{ raw: string; result: unknown }>;
export type ChainHttpJson = (route: string, body: unknown, signal: AbortSignal) => Promise<{ raw: string; result: unknown }>;
export interface ChainAdapter {
  head(signal: AbortSignal): Promise<number>;
  identity(height: number, signal: AbortSignal): Promise<BlockIdentity | null>;
  capture(height: number, signal: AbortSignal): Promise<CapturedBlock>;
  findHeights?(from: number, through: number, limit: number, signal: AbortSignal): Promise<readonly number[]>;
}
export interface ChainState { readonly version: 1; readonly chain: string; readonly tail: readonly BlockIdentity[]; readonly initialHeight?: number; readonly observedTarget?: number; }
export interface ChainHealth { readonly chain: string; readonly cursorHeight: number | null; readonly observedTarget: number | null; readonly error: string | null; }
interface Sink { write(source: string, payload: unknown): void; flush(): Promise<void>; }
class ProviderError extends Error { constructor(message: string, readonly retryAfterMs: number) { super(message); } }
const hash = (bytes: Buffer) => createHash("sha256").update(bytes).digest();
const utf8 = new TextDecoder("utf-8", { fatal: true, ignoreBOM: true });
export function bitcoinHeaderIdentity(raw: Buffer): Readonly<{ hash: string; parentHash: string }> {
  if (raw.length < 81) throw new Error("Truncated Bitcoin block");
  return { hash: Buffer.from(hash(hash(raw.subarray(0, 80)))).reverse().toString("hex"),
    parentHash: Buffer.from(raw.subarray(4, 36)).reverse().toString("hex") };
}
const heightOf = (value: unknown): number => {
  const height = typeof value === "string" && /^(0x[0-9a-f]+|[0-9]+)$/i.test(value) ? Number(BigInt(value)) : value;
  if (typeof height !== "number" || !Number.isSafeInteger(height) || height < 0) throw new Error("Invalid chain height");
  return height;
};
export function validateChainConfigs(configs: readonly ChainConfig[]): void {
  const ids = new Set<string>();
  for (const config of configs) {
    if (!/^[a-z0-9-]{1,80}$/.test(config.id) || ids.has(config.id)) throw new Error("Chain IDs must be unique lowercase identifiers");
    ids.add(config.id);
    if (!["evm", "bitcoin-esplora", "solana", "tron", "xrpl"].includes(config.adapter)) throw new Error(`Unsupported chain adapter: ${config.id}`);
    if (!!config.url === !!config.urlEnv) throw new Error(`Set exactly one url or urlEnv: ${config.id}`);
    if (config.adapter === "evm" && !/^0x[0-9a-f]+$/i.test(config.chainId ?? "")) throw new Error(`Expected EVM chainId is required: ${config.id}`);
    if ((config.adapter === "bitcoin-esplora" || config.adapter === "tron") && !/^[0-9a-f]{64}$/i.test(config.genesisHash ?? ""))
      throw new Error(`Expected genesisHash is required: ${config.id}`);
    if (config.adapter === "solana" && !isSolanaHash(config.genesisHash)) throw new Error(`Expected Solana genesisHash is required: ${config.id}`);
    if (config.adapter === "xrpl" && (!Number.isSafeInteger(config.networkId) || config.networkId! < 0))
      throw new Error(`Expected XRPL networkId is required: ${config.id}`);
    if (config.evmReceiptMode !== undefined && (config.adapter !== "evm" || !["per-transaction", "block"].includes(config.evmReceiptMode)))
      throw new Error(`Invalid EVM receipt mode: ${config.id}`);
    if (config.tronApiKeyEnv !== undefined && (config.adapter !== "tron" || !/^[A-Z_][A-Z0-9_]*$/i.test(config.tronApiKeyEnv)))
      throw new Error(`Invalid TRON API key environment variable name: ${config.id}`);
    for (const value of [config.confirmationLag, config.startHeight]) {
      if (value !== undefined && (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0))
        throw new Error(`Invalid confirmationLag or startHeight: ${config.id}`);
    }
  }
}
export function createChainAdapter(config: ChainConfig): ChainAdapter {
  const endpoint = config.url ?? process.env[config.urlEnv!];
  if (!endpoint) throw new Error(`Missing configured endpoint environment variable for ${config.id}`);
  const url = new URL(endpoint);
  if (!["http:", "https:"].includes(url.protocol)) throw new Error("Chain endpoints must use HTTP(S)");
  const tronApiKey = config.tronApiKeyEnv === undefined ? undefined : process.env[config.tronApiKeyEnv];
  if (config.tronApiKeyEnv !== undefined && (!tronApiKey || url.protocol !== "https:"))
    throw new Error(`TRON API key requires a configured value and HTTPS: ${config.id}`);
  let requestId = 0;
  async function request(route: string, signal: AbortSignal, body?: unknown): Promise<Buffer> {
    const requestUrl = route ? new URL(url) : url;
    if (route) requestUrl.pathname = `${requestUrl.pathname.replace(/\/$/, "")}${route}`;
    const response = await fetch(requestUrl, {
      method: body === undefined ? "GET" : "POST", body: body === undefined ? undefined : JSON.stringify(body),
      headers: { ...(body !== undefined ? { "content-type": "application/json" } : {}),
        ...(tronApiKey ? { "TRON-PRO-API-KEY": tronApiKey } : {}) },
      redirect: tronApiKey ? "error" : "follow",
      signal: AbortSignal.any([signal, AbortSignal.timeout(30_000)]),
    });
    if (!response.ok) {
      const retry = response.headers.get("retry-after");
      const seconds = retry === null ? 15 : Number(retry);
      const delay = Number.isFinite(seconds) ? seconds * 1000 : Date.parse(retry!) - Date.now();
      throw new ProviderError(`Chain provider HTTP ${response.status}`, Math.max(15_000, Number.isFinite(delay) ? delay : 15_000));
    }
    return Buffer.from(await response.arrayBuffer());
  }
  async function rpc(method: string, params: readonly unknown[], signal: AbortSignal) {
    const body = config.adapter === "xrpl" ? { method, params } : { jsonrpc: "2.0", id: ++requestId, method, params };
    const raw = utf8.decode(await request("", signal, body));
    const parsed = JSON.parse(raw) as { result?: any; error?: { code: number } };
    if (parsed.error) throw new Error(`Chain RPC ${method} failed (${parsed.error.code})`);
    if (!("result" in parsed)) throw new Error(`Missing RPC result: ${method}`);
    return { raw, result: parsed.result };
  }
  if (config.adapter === "solana") return createSolanaAdapter(config, rpc);
  if (config.adapter === "tron") return createTronAdapter(config, async (route, body, signal) => {
    const raw = utf8.decode(await request(route, signal, body));
    return { raw, result: JSON.parse(raw) };
  });
  if (config.adapter === "xrpl") return createXrplAdapter(config, rpc);
  if (config.adapter === "bitcoin-esplora") {
    let verified = false;
    const verify = async (signal: AbortSignal) => {
      if (!verified) {
        if ((await request("/block-height/0", signal)).toString().trim() !== config.genesisHash) throw new Error("Bitcoin network identity mismatch");
        verified = true;
      }
    };
    const identity = async (height: number, signal: AbortSignal): Promise<BlockIdentity> => {
      await verify(signal);
      const blockHash = (await request(`/block-height/${height}`, signal)).toString().trim();
      if (!/^[0-9a-f]{64}$/.test(blockHash)) throw new Error("Invalid Bitcoin block hash");
      const metadata = JSON.parse((await request(`/block/${blockHash}`, signal)).toString());
      if (metadata.id !== blockHash || metadata.height !== height) throw new Error("Bitcoin metadata identity mismatch");
      return { height, hash: blockHash, parentHash: metadata.previousblockhash ?? "0".repeat(64) };
    };
    return {
      head: async (signal) => { await verify(signal); return heightOf((await request("/blocks/tip/height", signal)).toString().trim()); }, identity,
      capture: async (height, signal) => {
        const block = await identity(height, signal);
        const raw = await request(`/block/${block.hash}/raw`, signal);
        const header = bitcoinHeaderIdentity(raw);
        if (header.hash !== block.hash || header.parentHash !== block.parentHash) throw new Error("Bitcoin raw header identity mismatch");
        return { ...block, payload: { format: "bitcoin-raw-block-base64", rawBase64: raw.toString("base64"),
          rawBytes: raw.length, rawSha256: hash(raw).toString("hex"),
          coverage: "All serialized transactions in this block; previous UTXOs and observer mempool history are separate inputs" } };
      },
    };
  }
  let verified = false;
  async function verify(signal: AbortSignal) {
    if (!verified) {
      if (BigInt((await rpc("eth_chainId", [], signal)).result) !== BigInt(config.chainId!)) throw new Error("EVM chain identity mismatch");
      verified = true;
    }
  }
  async function blockAt(height: number, full: boolean, signal: AbortSignal) {
    await verify(signal);
    return rpc("eth_getBlockByNumber", [`0x${height.toString(16)}`, full], signal);
  }
  const blockIdentity = (block: any): BlockIdentity => {
    if (!/^0x[0-9a-f]{64}$/i.test(block?.hash ?? "") || !/^0x[0-9a-f]{64}$/i.test(block?.parentHash ?? "")) throw new Error("Invalid EVM block identity");
    return { height: heightOf(block.number), hash: block.hash.toLowerCase(), parentHash: block.parentHash.toLowerCase() };
  };
  return {
    head: async (signal) => { await verify(signal); return heightOf((await rpc("eth_blockNumber", [], signal)).result); },
    identity: async (height, signal) => { const { result } = await blockAt(height, false, signal); return result === null ? null : blockIdentity(result); },
    capture: async (height, signal) => {
      const block = await blockAt(height, true, signal);
      const identity = blockIdentity(block.result);
      if (identity.height !== height || !Array.isArray(block.result.transactions)) throw new Error("EVM block contents missing");
      const txs = block.result.transactions as readonly { hash: string }[];
      if (config.evmReceiptMode === "block") {
        const blockReceipts = await captureEvmBlockReceipts(rpc, identity, txs, signal);
        return { ...identity, payload: { format: "evm-json-rpc-raw-block-receipts", block: block.raw, blockReceipts,
          coverage: "Full block transactions and every transaction receipt/log; internal call traces and pending transactions are not included" } };
      }
      const receipts = new Array<string>(txs.length);
      let nextReceipt = 0;
      let receiptError: unknown;
      // Bounded concurrency; no incomplete receipt set advances the block cursor.
      await Promise.all(Array.from({ length: Math.min(8, txs.length) }, async () => {
        while (!receiptError && nextReceipt < txs.length) {
          const index = nextReceipt++, tx = txs[index]!;
          try {
            const receipt = await rpc("eth_getTransactionReceipt", [tx.hash], signal);
            if (receipt.result?.transactionHash?.toLowerCase() !== tx.hash.toLowerCase()
              || receipt.result?.blockHash?.toLowerCase() !== identity.hash) throw new Error("Receipt is missing or belongs to a different block; retry without advancing cursor");
            receipts[index] = receipt.raw;
          } catch (error) { receiptError ??= error; }
        }
      }));
      if (receiptError) throw receiptError;
      return { ...identity, payload: { format: "evm-json-rpc-raw", block: block.raw, receipts,
        coverage: "Full block transactions and every transaction receipt/log; internal call traces and pending transactions are not included" } };
    },
  };
}

/** A reorg appends a detach record; original orphan observations remain in the journal. */
export async function collectChainStep(config: ChainConfig, adapter: ChainAdapter, state: ChainState, sink: Sink,
  signal: AbortSignal, save: (state: ChainState) => Promise<void>): Promise<ChainState> {
  const head = await adapter.head(signal), target = head - (config.confirmationLag ?? 0);
  if (target < 0) return state;
  state = { ...state, observedTarget: target };
  const source = `blockchain/${config.id}`;
  let tail = state.tail;
  const last = tail.at(-1);
  if (last) {
    if (last.height > head) throw new Error("Provider head is behind the durable cursor; retry on a synchronized provider");
    const current = await adapter.identity(last.height, signal);
    if (!current || current.height !== last.height) throw new Error("Provider cannot identify the durable cursor block");
    if (current.hash !== last.hash) {
      let ancestor = -1;
      for (let index = tail.length - 2; index >= 0; index--) {
        const prior = tail[index]!;
        const candidate = await adapter.identity(prior.height, signal);
        if (!candidate || candidate.height !== prior.height) throw new Error("Provider cannot identify a retained ancestor");
        if (candidate.hash === prior.hash) { ancestor = index; break; }
      }
      if (ancestor < 0) throw new Error(`Reorg exceeds retained ${tail.length}-block ancestry; explicit rescan is required`);
      sink.write(source, { kind: "reorg", observedAt: Date.now(), detached: tail.slice(ancestor + 1), ancestor: tail[ancestor] });
      tail = tail.slice(0, ancestor + 1);
      await sink.flush(); await save({ ...state, tail });
    }
  }
  if (!tail.length && config.startHeight === undefined && state.initialHeight === undefined) {
    state = { ...state, initialHeight: target };
    sink.write(source, { kind: "capture_start", observedAt: Date.now(), startHeight: target });
    await sink.flush(); await save(state);
  }
  const start = config.startHeight ?? state.initialHeight ?? 0;
  let from = Math.max(start, (tail.at(-1)?.height ?? -1) + 1);
  async function commit(next: number, block: CapturedBlock) {
    if (block.height !== next) throw new Error("Provider returned a different block height");
    if (adapter.findHeights && next > 0 && (!Number.isSafeInteger(block.parentHeight) || block.parentHeight! < 0 || block.parentHeight! >= next))
      throw new Error("Sparse block is missing a valid parent height");
    if (!tail.length && next > 0) {
      const anchorHeight = block.parentHeight ?? next - 1;
      if (anchorHeight >= from) throw new Error("Provider omitted a block at or after the requested starting height");
      const anchor = await adapter.identity(anchorHeight, signal);
      if (!anchor || anchor.height !== anchorHeight || anchor.hash !== block.parentHash) throw new Error("Missing or inconsistent starting ancestry anchor");
      sink.write(source, { kind: "ancestry_anchor", observedAt: Date.now(), ...anchor,
        note: "Identity only; transaction history starts at the requested height" });
      tail = [anchor];
    }
    const prior = tail.at(-1);
    if (prior && (block.parentHash !== prior.hash || (block.parentHeight ?? next - 1) !== prior.height))
      throw new Error("Block does not link to the last captured parent; retry canonical ancestry");
    if (next > from) sink.write(source, { kind: "skipped_slots", observedAt: Date.now(),
      fromHeight: from, throughHeight: next - 1, parent: prior, nextBlockHash: block.hash });
    sink.write(source, { kind: "block", observedAt: Date.now(), chain: config.id,
      blockId: `${config.id}:${block.hash}`, ...block });
    const identity: BlockIdentity = { height: block.height, hash: block.hash, parentHash: block.parentHash,
      ...(block.parentHeight !== undefined ? { parentHeight: block.parentHeight } : {}) };
    tail = [...tail, identity].slice(-128);
    await sink.flush(); await save({ ...state, tail });
    from = next + 1;
  }
  const parallelReads = adapter.findHeights ? 8 : config.evmReceiptMode === "block" ? 4 : 1;
  if (parallelReads > 1 && from <= target && !signal.aborted) {
    const limit = Math.min(parallelReads, target - from + 1);
    const heights = adapter.findHeights ? await adapter.findHeights(from, target, limit, signal)
      : Array.from({ length: limit }, (_, index) => from + index);
    if (heights.length > limit || heights.some((next, index) => !Number.isSafeInteger(next) || next < from || next > target
      || (index > 0 && next <= heights[index - 1]!))) throw new Error("Provider returned an invalid block range");
    const blocks = await Promise.allSettled(heights.map((next) => adapter.capture(next, signal)));
    for (const [index, result] of blocks.entries()) {
      if (result.status === "rejected") throw result.reason;
      await commit(heights[index]!, result.value);
    }
  } else {
    for (let count = 0; from <= target && count < 8 && !signal.aborted; count++) await commit(from, await adapter.capture(from, signal));
  }
  return { ...state, tail };
}

export async function runBlockchainCapture(configs: readonly ChainConfig[], sink: Sink, directory: string,
  signal: AbortSignal, onHealth?: (health: readonly ChainHealth[]) => void): Promise<void> {
  validateChainConfigs(configs);
  await fs.mkdir(directory, { recursive: true });
  const health = new Map<string, ChainHealth>();
  const update = (config: ChainConfig, state: ChainState, error: string | null) => {
    health.set(config.id, { chain: config.id, cursorHeight: state.tail.at(-1)?.height ?? null,
      observedTarget: state.observedTarget ?? null, error });
    onHealth?.([...health.values()]);
  };
  await Promise.all(configs.map(async (config) => {
    const file = path.join(directory, `${config.id}.json`), adapter = createChainAdapter(config);
    const configHash = createHash("sha256").update(JSON.stringify(config)).digest("hex");
    let state: ChainState = { version: 1, chain: config.id, tail: [] };
    try {
      const saved = JSON.parse(await fs.readFile(file, "utf8"));
      if (saved.configHash !== configHash || saved.state?.chain !== config.id || !Array.isArray(saved.state.tail)) throw new Error("Chain checkpoint config mismatch");
      state = saved.state;
    } catch (error) { if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error; }
    update(config, state, null);
    const save = async (next: ChainState) => {
      const temporary = `${file}.tmp`;
      const handle = await fs.open(temporary, "w");
      try { await handle.writeFile(JSON.stringify({ configHash, state: next })); await handle.sync(); } finally { await handle.close(); }
      await fs.rename(temporary, file);
      state = next;
    };
    sink.write(`blockchain/${config.id}`, { kind: "session", observedAt: Date.now(), adapter: config.adapter,
      chainId: config.chainId ?? null, genesisHash: config.genesisHash ?? null, networkId: config.networkId ?? null,
      evmReceiptMode: config.adapter === "evm" ? config.evmReceiptMode ?? "per-transaction" : null,
      startHeight: config.startHeight ?? "current-confirmed-tip", confirmationLag: config.confirmationLag ?? 0,
      note: "Polling observer. No pre-start mempool history or previously unobserved orphan blocks is claimed." });
    while (!signal.aborted) {
      let retryDelay = 0;
      try { state = await collectChainStep(config, adapter, state, sink, signal, save); update(config, state, null); }
      catch (error) {
        if (signal.aborted) break;
        retryDelay = error instanceof ProviderError ? error.retryAfterMs : 15_000;
        update(config, state, error instanceof Error ? error.message : String(error));
        sink.write(`blockchain/${config.id}`, { kind: "coverage_error", observedAt: Date.now(), message: error instanceof Error ? error.message : String(error) });
        await sink.flush();
      }
      await new Promise<void>((resolve) => {
        const done = () => { clearTimeout(timer); signal.removeEventListener("abort", done); resolve(); };
        const catchingUp = state.tail.length > 0 && state.observedTarget !== undefined && state.tail.at(-1)!.height < state.observedTarget;
        const timer = setTimeout(done, retryDelay || (catchingUp ? 50 : 15_000)); signal.addEventListener("abort", done, { once: true });
        if (signal.aborted) done();
      });
    }
  }));
}

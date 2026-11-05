import { randomUUID } from "node:crypto";
import WebSocket from "ws";

export type BinanceVenue = "spot" | "usdm-futures";
export interface CaptureMarket {
  readonly venue: BinanceVenue;
  readonly symbol: string;
  readonly baseAsset: string;
  readonly assetId: string;
  readonly physicalBaseAsset: string;
  readonly quoteAsset: string;
  readonly priceOrientation: "direct" | "inverse";
  readonly instrumentKind?: string;
  readonly contractType?: string;
}
interface UniverseRoute {
  readonly venue: BinanceVenue;
  readonly symbol: string;
  readonly baseAsset: string;
  readonly quoteAsset: string;
  readonly priceOrientation: "direct" | "inverse";
  readonly instrumentKind?: string;
  readonly contractType?: string;
}
interface UniverseEntry {
  readonly baseAsset: string;
  readonly assetId: string;
  readonly routes: readonly UniverseRoute[];
}
export interface CaptureUniverse { readonly entries: readonly UniverseEntry[]; }
export function selectCaptureMarkets(universe: CaptureUniverse, symbols?: readonly string[], venues: readonly BinanceVenue[] = ["spot", "usdm-futures"]): readonly CaptureMarket[] {
  if (!Array.isArray(universe.entries)) throw new Error("Universe must have entries with explicit physical routes");
  const selected = new Map<string, CaptureMarket>();
  const wanted = symbols ? new Set(symbols.map((symbol) => symbol.toUpperCase())) : undefined;
  for (const entry of universe.entries) {
    if (typeof entry.baseAsset !== "string" || typeof entry.assetId !== "string" || !Array.isArray(entry.routes)) throw new Error("Invalid universe entry: assetId, baseAsset and routes are required");
    for (const route of entry.routes) {
      const { venue, symbol } = route;
      if (venue !== "spot" && venue !== "usdm-futures") throw new Error(`Invalid Binance venue: ${venue}`);
      if (!venues.includes(venue)) continue;
      if (route.priceOrientation !== "direct" && route.priceOrientation !== "inverse") throw new Error(`Missing explicit priceOrientation for ${symbol}`);
      if (!route.baseAsset || !route.quoteAsset) throw new Error(`Missing physical base/quote for ${symbol}`);
        if (typeof symbol !== "string" || !/^[\p{L}\p{N}_]{2,50}$/u.test(symbol)) throw new Error(`Invalid Binance symbol: ${symbol}`);
        if (wanted && !wanted.has(symbol)) continue;
        const key = `${venue}:${symbol}`;
        const previous = selected.get(key);
        const market = { venue, symbol, baseAsset: entry.baseAsset, assetId: entry.assetId,
          physicalBaseAsset: route.baseAsset, quoteAsset: route.quoteAsset, priceOrientation: route.priceOrientation,
          instrumentKind: route.instrumentKind, contractType: route.contractType };
        if (previous && previous.assetId !== market.assetId) throw new Error(`Conflicting identities for ${key}`);
        selected.set(key, market);
    }
  }
  if (wanted) for (const symbol of wanted) if (![...selected.values()].some((market) => market.symbol === symbol)) throw new Error(`Requested symbol is absent from selected universe routes: ${symbol}`);
  if (selected.size === 0) throw new Error("No Binance markets selected");
  return [...selected.values()];
}

export interface CaptureBatch {
  readonly id: string;
  readonly venue: BinanceVenue;
  readonly url: string;
  readonly markets: readonly CaptureMarket[];
  readonly streams: readonly string[];
  readonly hasDepth: boolean;
}
export function createCaptureBatches(markets: readonly CaptureMarket[], marketsPerSocket = 20): readonly CaptureBatch[] {
  if (!Number.isInteger(marketsPerSocket) || marketsPerSocket < 1 || marketsPerSocket > 100) throw new Error("markets-per-socket must be 1..100");
  const batches: CaptureBatch[] = [];
  for (const venue of ["spot", "usdm-futures"] as const) {
    const selected = markets.filter((market) => market.venue === venue);
    for (let offset = 0; offset < selected.length; offset += marketsPerSocket) {
      const group = selected.slice(offset, offset + marketsPerSocket);
      const index = String(offset / marketsPerSocket).padStart(3, "0");
      const channels = venue === "spot" ? ["depth@100ms", "bookTicker", "trade"] : ["depth@100ms", "bookTicker"];
      batches.push({ id: `${venue}-public-${index}`, venue, markets: group, hasDepth: true,
        url: venue === "spot" ? "wss://stream.binance.com:9443/stream" : "wss://fstream.binance.com/public/stream",
        streams: group.flatMap((market) => channels.map((channel) => `${market.symbol.toLowerCase()}@${channel}`)) });
      if (venue === "usdm-futures") batches.push({ id: `${venue}-market-${index}`, venue, markets: group, hasDepth: false,
        url: "wss://fstream.binance.com/market/stream", streams: group.flatMap((market) => ["aggTrade", "markPrice@1s"].map((channel) => `${market.symbol.toLowerCase()}@${channel}`)) });
    }
  }
  return batches;
}

type Level = readonly [string, string];
export interface DepthEvent {
  readonly U: number;
  readonly u: number;
  readonly pu?: number;
  readonly E: number;
  readonly T?: number;
  readonly b: readonly Level[];
  readonly a: readonly Level[];
}
export interface DepthSnapshot {
  readonly lastUpdateId: number;
  readonly bids: readonly Level[];
  readonly asks: readonly Level[];
  readonly E?: number;
  readonly T?: number;
}
export interface EventBookView {
  readonly updateId: number;
  readonly eventTime: number;
  readonly transactionTime: number | null;
  readonly bid: string;
  readonly bidQty: string;
  readonly ask: string;
  readonly askQty: string;
  readonly mid: number;
  readonly microprice: number;
  readonly imbalance: number;
}
interface DepthResult {
  readonly status: "buffered" | "ignored" | "applied" | "gap";
  readonly reason?: string;
  readonly bridge?: boolean;
  readonly book?: EventBookView;
}
function sequence(value: number) {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error("Unsafe or invalid Binance sequence ID; raw wire record retained, state cannot be certified");
  return value;
}

/** Private mutable reconstruction; source messages/snapshots remain immutable. */
export class SequencedDepthBook {
  private readonly bids = new Map<string, string>();
  private readonly asks = new Map<string, string>();
  private bestBid: string | undefined;
  private bestAsk: string | undefined;
  private pending: DepthEvent[] = [];
  private lastUpdateId: number | undefined;
  private bridged = false;
  private view: EventBookView | undefined;
  private readonly knownLevelLimit: number;
  constructor(private readonly venue: BinanceVenue, private readonly maxBufferedEvents = 1_000) {
    this.knownLevelLimit = venue === "spot" ? 10_000 : 2_000;
  }

  reset() {
    this.lastUpdateId = undefined; this.bridged = false; this.pending = []; this.view = undefined;
    this.bids.clear(); this.asks.clear(); this.bestBid = undefined; this.bestAsk = undefined;
  }
  needsSnapshot() { return this.lastUpdateId === undefined; }
  isValid() { return this.bridged && this.view !== undefined; }
  current() { return this.isValid() ? this.view : undefined; }
  bufferedCount() { return this.pending.length; }
  observe(event: DepthEvent, snapshotInFlight: boolean): DepthResult {
    if (this.needsSnapshot() && !snapshotInFlight) return { status: "ignored", reason: "awaiting_snapshot_request" };
    return this.consume(event);
  }

  consume(event: DepthEvent): DepthResult {
    sequence(event.U); sequence(event.u);
    if (event.U > event.u || !Number.isFinite(event.E)) throw new Error("Malformed depth sequence or event timestamp");
    if (this.lastUpdateId === undefined) {
      if (this.pending.length >= this.maxBufferedEvents) {
        this.pending = [event];
        return { status: "gap", reason: "reconstruction_buffer_overflow" };
      }
      this.pending.push(event);
      return { status: "buffered" };
    }
    if (this.venue === "spot" ? event.u <= this.lastUpdateId : event.u < this.lastUpdateId || (this.bridged && event.u === this.lastUpdateId)) return { status: "ignored" };
    const bridge = !this.bridged;
    const next = this.venue === "spot" ? this.lastUpdateId + 1 : this.lastUpdateId;
    const joins = bridge ? event.U <= next && next <= event.u
      : this.venue === "spot" ? event.U <= next && next <= event.u : sequence(event.pu!) === this.lastUpdateId;
    if (!joins) {
      this.reset(); this.pending.push(event);
      return { status: "gap", reason: bridge ? "snapshot_does_not_bridge" : "depth_sequence_gap" };
    }
    this.apply("bid", event.b); this.apply("ask", event.a);
    this.lastUpdateId = event.u;
    if (!this.bestBid || !this.bestAsk || Number(this.bestBid) >= Number(this.bestAsk)) {
      this.reset();
      return { status: "gap", reason: "crossed_or_empty_book" };
    }
    this.bridged = true;
    const bidQty = this.bids.get(this.bestBid)!; const askQty = this.asks.get(this.bestAsk)!;
    const bid = Number(this.bestBid); const ask = Number(this.bestAsk); const total = Number(bidQty) + Number(askQty);
    this.view = { updateId: event.u, eventTime: event.E, transactionTime: event.T ?? null,
      bid: this.bestBid, bidQty, ask: this.bestAsk, askQty, mid: (bid + ask) / 2,
      microprice: (bid * Number(askQty) + ask * Number(bidQty)) / total, imbalance: (Number(bidQty) - Number(askQty)) / total };
    return { status: "applied", bridge, book: this.view };
  }

  snapshot(snapshot: DepthSnapshot): readonly DepthResult[] {
    const buffered = this.pending;
    this.reset();
    this.lastUpdateId = sequence(snapshot.lastUpdateId);
    this.apply("bid", snapshot.bids); this.apply("ask", snapshot.asks);
    const results: DepthResult[] = [];
    for (const event of buffered) results.push(this.consume(event));
    return results;
  }
  checkpoint() {
    if (!this.isValid()) return undefined;
    return { lastUpdateId: this.lastUpdateId, eventTime: this.view!.eventTime,
      bids: [...this.bids] as readonly Level[], asks: [...this.asks] as readonly Level[],
      knownLevelLimitPerSide: this.knownLevelLimit,
      coverage: "Initial REST depth plus observed changed levels, bounded to nearest known levels; unchanged levels beyond the snapshot are unknown" };
  }
  private apply(side: "bid" | "ask", levels: readonly Level[]) {
    const target = side === "bid" ? this.bids : this.asks;
    let best = side === "bid" ? this.bestBid : this.bestAsk;
    for (const [price, quantity] of levels) {
      if (!(Number(price) > 0) || !(Number(quantity) >= 0) || !Number.isFinite(Number(quantity))) throw new Error("Invalid depth level");
      if (Number(quantity) === 0) { target.delete(price); if (best === price) best = undefined; }
      else { target.set(price, quantity); if (best !== undefined && (side === "bid" ? Number(price) > Number(best) : Number(price) < Number(best))) best = price; }
    }
    if (target.size > this.knownLevelLimit) {
      const ordered = [...target.keys()].sort((left, right) => side === "bid" ? Number(right) - Number(left) : Number(left) - Number(right));
      for (const price of ordered.slice(this.knownLevelLimit)) target.delete(price);
    }
    if (best === undefined) for (const price of target.keys()) if (best === undefined || (side === "bid" ? Number(price) > Number(best) : Number(price) < Number(best))) best = price;
    if (side === "bid") this.bestBid = best; else this.bestAsk = best;
  }
}

interface Sink { write(source: string, payload: unknown, recordedAt?: number): void; }
export interface BinanceCaptureOptions {
  readonly markets: readonly CaptureMarket[];
  readonly sink: Sink;
  readonly sessionId: string;
  readonly marketsPerSocket?: number;
  readonly snapshotIntervalMs?: number;
  readonly checkpointSeconds?: number;
  readonly derivedRows?: boolean;
  readonly onFatal: (error: Error) => void;
}
interface MarketState {
  readonly market: CaptureMarket;
  readonly book: SequencedDepthBook;
  history: EventBookView[];
  generation: number;
  snapshotPending: boolean;
  bufferingSnapshot: boolean;
  depthConnected: boolean;
  lastTradeId?: number;
  lastDepthAt?: number;
  lastTradeAt?: number;
  lastBboAt?: number;
  depthEpoch: number;
  tradeEpoch: number;
}

/** Public feeds preserve exchange aggregation and missing timestamps; no synthetic L3/identities. */
export class BinanceEventCollector {
  private stopping = false;
  private readonly sockets = new Set<WebSocket>();
  private readonly requests = new Set<AbortController>();
  private readonly tasks = new Set<Promise<void>>();
  private readonly states = new Map<string, MarketState>();
  private receiveSequence = 0;
  private snapshotQueue: Promise<void> = Promise.resolve();
  private snapshotNotBefore = 0;
  private readonly timers: NodeJS.Timeout[] = [];

  constructor(private readonly options: BinanceCaptureOptions) {
    for (const market of options.markets) this.states.set(`${market.venue}:${market.symbol}`, {
      market, book: new SequencedDepthBook(market.venue), history: [], generation: 0,
      snapshotPending: false, bufferingSnapshot: false, depthConnected: false, depthEpoch: 0, tradeEpoch: 0,
    });
    if ((options.snapshotIntervalMs ?? 3_000) < 3_000) throw new Error("snapshot-interval-ms must be >=3000 to bound 5000-level spot snapshot weight");
  }

  async run() {
    this.record("session_start", { markets: this.options.markets, capabilities: {
      spotTrades: "individual public trades", futuresTrades: "exchange-aggregated 100ms, excludes ADL/insurance trades",
      orderFlow: "net price-level changes; no exact cancellation attribution, L3, participant IDs, or inventories",
      timestamps: "exchange milliseconds where supplied; spot bookTicker has no exchange timestamp",
      alignment: "last already-received reconstructed depth state with eventTime <= tradeTime; age and missing state explicit",
      depth: "5000 spot / 1000 USD-M initial levels; unchanged levels outside snapshot unknown; RPI depth excluded",
    } });
    const batches = createCaptureBatches(this.options.markets, this.options.marketsPerSocket);
    const coverage = setInterval(() => this.safe(() => this.record("coverage", { markets: this.health() })), 5_000);
    const checkpoints = setInterval(() => this.safe(() => {
      for (const state of this.states.values()) {
        const checkpoint = state.book.checkpoint();
        if (checkpoint) this.record("book_checkpoint", { ...state.market, depthEpoch: state.depthEpoch, ...checkpoint });
      }
    }), (this.options.checkpointSeconds ?? 300) * 1_000);
    this.timers.push(coverage, checkpoints);
    // Opening every market immediately would overflow reconstruction buffers while weight-limited snapshots queue.
    let precedingMarkets = 0;
    let groupStagger = 0;
    for (const batch of batches) {
      if (batch.hasDepth) { groupStagger = precedingMarkets * (this.options.snapshotIntervalMs ?? 3_000); precedingMarkets += batch.markets.length; }
      const task = this.supervise(batch, groupStagger).catch((error: unknown) => this.options.onFatal(asError(error)));
      this.tasks.add(task); void task.finally(() => this.tasks.delete(task));
    }
    for (const task of [this.clockLoop(), this.openInterestLoop()]) {
      const guarded = task.catch((error: unknown) => this.options.onFatal(asError(error)));
      this.tasks.add(guarded); void guarded.finally(() => this.tasks.delete(guarded));
    }
    await Promise.all(this.tasks);
    await this.snapshotQueue;
    this.record("session_stop", { markets: this.health() });
  }
  stop() {
    this.stopping = true;
    for (const timer of this.timers) clearInterval(timer);
    for (const request of this.requests) request.abort();
    for (const socket of this.sockets) socket.terminate();
  }
  health() {
    return [...this.states.values()].map((state) => ({ ...state.market, bookValid: state.book.isValid(), snapshotPending: state.snapshotPending,
      bufferedDepthEvents: state.book.bufferedCount(), lastDepthAt: state.lastDepthAt ?? null, lastTradeAt: state.lastTradeAt ?? null,
      lastBboAt: state.lastBboAt ?? null, lastTradeId: state.lastTradeId ?? null, depthEpoch: state.depthEpoch, tradeEpoch: state.tradeEpoch }));
  }
  private safe(task: () => void) { try { task(); } catch (error) { this.options.onFatal(asError(error)); } }
  private record(kind: string, fields: Readonly<Record<string, unknown>>, source = "binance-event-tape/control", receivedAt = Date.now(), monotonicNs = process.hrtime.bigint().toString()) {
    this.options.sink.write(source, { schemaVersion: 1, sessionId: this.options.sessionId, kind, receivedAt, monotonicNs,
      receiveSequence: ++this.receiveSequence, ...fields }, receivedAt);
  }
  private async pause(milliseconds: number) {
    const end = Date.now() + milliseconds;
    while (!this.stopping && Date.now() < end) await new Promise((resolve) => setTimeout(resolve, Math.min(250, end - Date.now())));
  }
  private async supervise(batch: CaptureBatch, stagger: number) {
    await this.pause(stagger);
    let attempt = 0;
    while (!this.stopping) {
      await this.connect(batch);
      if (!this.stopping) await this.pause(Math.min(30_000, 1_000 * 2 ** Math.min(attempt++, 5)));
    }
  }
  private connect(batch: CaptureBatch) {
    return new Promise<void>((resolve) => {
      const connectionId = randomUUID();
      const socket = new WebSocket(batch.url, { handshakeTimeout: 15_000 });
      this.sockets.add(socket);
      let lastMessageAt = Date.now();
      let opened = false;
      const states = batch.markets.map((market) => this.states.get(`${market.venue}:${market.symbol}`)!);
      const ping = setInterval(() => {
        if (Date.now() - lastMessageAt > 90_000) socket.terminate();
        else if (socket.readyState === WebSocket.OPEN) socket.ping();
      }, 20_000);
      socket.on("pong", () => { lastMessageAt = Date.now(); });
      // Reconnect before Binance's 24h connection lifetime. The new session is explicitly a coverage boundary.
      const rotate = setTimeout(() => socket.close(1000, "scheduled connection rotation"), 23 * 60 * 60_000);
      socket.on("open", () => this.safe(() => {
        opened = true;
        for (const state of states) {
          if (batch.hasDepth) {
            state.book.reset(); state.history = []; state.generation++; state.depthEpoch++; state.depthConnected = true;
          }
          if (!batch.hasDepth || batch.venue === "spot") state.tradeEpoch++;
        }
        this.record("connection_open", { connectionId, batch: batch.id, url: batch.url, streams: batch.streams });
        socket.send(JSON.stringify({ method: "SUBSCRIBE", params: batch.streams, id: connectionId }));
        if (batch.hasDepth) for (const state of states) this.requestSnapshot(state);
      }));
      socket.on("message", (bytes) => this.safe(() => {
        const receivedAt = Date.now(); const monotonicNs = process.hrtime.bigint().toString(); lastMessageAt = receivedAt;
        const raw = bytes.toString();
        const envelope = JSON.parse(raw) as { readonly stream?: string; readonly data?: Record<string, unknown>; readonly code?: number; readonly msg?: string; readonly id?: string; readonly result?: unknown };
        if (!envelope.stream || !envelope.data) {
          this.record("subscription_response", { connectionId, batch: batch.id, raw }, undefined, receivedAt, monotonicNs);
          if (envelope.code !== undefined) socket.close(4000, "subscription error");
          return;
        }
        const message = envelope.data;
        const symbol = String(message.s ?? envelope.stream.split("@")[0]).toUpperCase();
        const state = this.states.get(`${batch.venue}:${symbol}`);
        if (!state || !batch.markets.some((market) => market.symbol === symbol)) throw new Error(`Unexpected Binance stream ${envelope.stream}`);
        const stream = envelope.stream.split("@").slice(1).join("@");
        const source = `binance-event-tape/${batch.id}`;
        const eventTime = typeof message.E === "number" ? message.E : null;
        const tradeTime = typeof message.T === "number" ? message.T : null;
        this.record("wire", { ...state.market, connectionId, depthEpoch: state.depthEpoch, tradeEpoch: state.tradeEpoch,
          stream: envelope.stream, exchangeEventTime: eventTime, exchangeTransactionTime: tradeTime,
          exchangeTimeAvailable: eventTime !== null, raw }, source, receivedAt, monotonicNs);
        if (stream.startsWith("depth")) {
          state.lastDepthAt = receivedAt;
          // The wire tape always retains this message. Only buffer reconstruction during the actual
          // REST request, not while its rate-limited turn is queued (including mass reconnects).
          const result = state.book.observe(message as unknown as DepthEvent, state.bufferingSnapshot);
          this.depthResult(state, result, connectionId);
          if (state.book.needsSnapshot()) this.requestSnapshot(state);
        } else if (stream === "trade" || stream === "aggTrade") {
          state.lastTradeAt = receivedAt;
          const id = sequence(Number(stream === "trade" ? message.t : message.a));
          if (state.lastTradeId !== undefined && id !== state.lastTradeId + 1) this.record("trade_id_discontinuity", { ...state.market, connectionId,
            expected: state.lastTradeId + 1, actual: id, note: "Detectable discontinuity, not a claim that all missing underlying fills are public" });
          state.lastTradeId = id;
          const at = tradeTime ?? eventTime;
          if (this.options.derivedRows) {
          let book: EventBookView | undefined;
          if (at !== null) for (let index = state.history.length - 1; index >= 0; index--) {
            if (state.history[index]!.eventTime <= at) { book = state.history[index]; break; }
          }
          this.record("trade_book_alignment", { ...state.market, connectionId, tradeId: id, tradeTime: at,
            depthEpoch: state.depthEpoch, tradeEpoch: state.tradeEpoch, book: state.book.isValid() ? book ?? null : null,
            bookEventAgeMs: at !== null && book && state.book.isValid() ? at - book.eventTime : null,
            policy: "eventTime<=tradeTime and received before trade; not matching-engine total ordering" }, source, receivedAt, monotonicNs);
          }
        } else if (stream === "bookTicker") state.lastBboAt = receivedAt;
      }));
      socket.on("error", (error) => this.safe(() => this.record("connection_error", { connectionId, batch: batch.id, message: error.message })));
      socket.once("close", (code, reason) => {
        clearInterval(ping); clearTimeout(rotate); this.sockets.delete(socket);
        this.safe(() => {
          for (const state of states) if (batch.hasDepth) { state.depthConnected = false; state.generation++; state.book.reset(); state.history = []; }
          this.record("connection_close", { connectionId, batch: batch.id, code, opened, reason: reason.toString(), coverage: "not certified across reconnect" });
        });
        resolve();
      });
    });
  }

  private depthResult(state: MarketState, result: DepthResult, connectionId?: string) {
    if (result.status === "gap") {
      state.history = [];
      this.record("book_gap", { ...state.market, connectionId, depthEpoch: state.depthEpoch, reason: result.reason });
      if (result.reason === "reconstruction_buffer_overflow") throw new Error(`Reconstruction buffer overflow for ${state.market.venue}:${state.market.symbol}; stop rather than certify partial state`);
    }
    if (result.book) {
      if (this.options.derivedRows) {
        state.history.push(result.book);
        if (state.history.length > 1_000) state.history.shift();
      }
      if (result.bridge) this.record("book_bridge", { ...state.market, connectionId, depthEpoch: state.depthEpoch, book: result.book });
      // This timestamped BBO is derived from sequence-checked depth; spot bookTicker itself has no exchange timestamp.
      if (this.options.derivedRows) this.record("depth_bbo", { ...state.market, depthEpoch: state.depthEpoch, book: result.book }, `binance-event-tape/${state.market.venue}-depth-bbo`);
    }
  }
  private async clockLoop() {
    const venues = [...new Set(this.options.markets.map((market) => market.venue))];
    while (!this.stopping) {
      for (const venue of venues) {
        if (this.stopping) break;
        const requestStartedAt = Date.now();
        await this.publicRest("server_time_calibration", venue === "spot" ? "https://api.binance.com/api/v3/time" : "https://fapi.binance.com/fapi/v1/time", { venue, requestStartedAt }, (raw, receivedAt) => {
          const serverTime = Number(JSON.parse(raw).serverTime);
          return { serverTime, roundTripMs: receivedAt - requestStartedAt,
            serverMinusLocalBoundsMs: [serverTime - receivedAt, serverTime - requestStartedAt],
            note: "Network-bound clock estimate, not an assurance that venue matching-engine clocks agree" };
        });
      }
      await this.pause(60_000);
    }
  }
  private async openInterestLoop() {
    const markets = this.options.markets.filter((market) => market.venue === "usdm-futures");
    if (markets.length === 0) return;
    const spacing = Math.max(250, 300_000 / markets.length);
    let index = 0;
    while (!this.stopping) {
      const market = markets[index++ % markets.length]!;
      await this.publicRest("open_interest", `https://fapi.binance.com/fapi/v1/openInterest?symbol=${encodeURIComponent(market.symbol)}`, market);
      await this.pause(spacing);
    }
  }
  private async publicRest(kind: string, url: string, fields: Readonly<Record<string, unknown>> | CaptureMarket,
    extra?: (raw: string, receivedAt: number) => Readonly<Record<string, unknown>>) {
    if (this.stopping) return;
    const controller = new AbortController(); this.requests.add(controller);
    const timeout = setTimeout(() => controller.abort(), 15_000);
    const requestStartedAt = Date.now();
    try {
      const response = await fetch(url, { signal: controller.signal });
      if (!response.ok) {
        if (response.status === 429 || response.status === 418) await this.pause(Math.max(60, Number(response.headers.get("retry-after") ?? 60)) * 1_000);
        throw new Error(`HTTP ${response.status}`);
      }
      const raw = await response.text(); const receivedAt = Date.now();
      if (!this.stopping) this.record(kind, { ...fields, url, requestStartedAt, raw, ...extra?.(raw, receivedAt) }, undefined, receivedAt);
    } catch (error) {
      if (!this.stopping) this.record(`${kind}_error`, { ...fields, requestStartedAt, url, message: asError(error).message });
    } finally { clearTimeout(timeout); this.requests.delete(controller); }
  }
  private requestSnapshot(state: MarketState) {
    if (state.snapshotPending || !state.depthConnected || this.stopping) return;
    state.snapshotPending = true;
    const generation = state.generation;
    this.snapshotQueue = this.snapshotQueue.then(async () => {
      if (this.stopping || generation !== state.generation) return;
      await this.pause(Math.max(0, this.snapshotNotBefore - Date.now()));
      if (this.stopping || generation !== state.generation) return;
      this.snapshotNotBefore = Date.now() + (this.options.snapshotIntervalMs ?? 3_000);
      const requestStartedAt = Date.now();
      state.book.reset(); state.history = []; state.bufferingSnapshot = true;
      const limit = state.market.venue === "spot" ? 5_000 : 1_000;
      const url = state.market.venue === "spot" ? "https://api.binance.com/api/v3/depth" : "https://fapi.binance.com/fapi/v1/depth";
      const controller = new AbortController(); this.requests.add(controller);
      const timeout = setTimeout(() => controller.abort(), 15_000);
      try {
        const response = await fetch(`${url}?symbol=${encodeURIComponent(state.market.symbol)}&limit=${limit}`, { signal: controller.signal });
        if (!response.ok) {
          const retryAfterSeconds = Number(response.headers.get("retry-after") ?? 60);
          if (response.status === 429 || response.status === 418) this.snapshotNotBefore = Date.now() + Math.max(60, retryAfterSeconds) * 1_000;
          throw new Error(`Depth snapshot HTTP ${response.status}`);
        }
        const raw = await response.text();
        if (generation !== state.generation || this.stopping) return;
        this.record("rest_snapshot", { ...state.market, depthEpoch: state.depthEpoch, requestStartedAt, limit, url, raw });
        const snapshot = JSON.parse(raw) as DepthSnapshot;
        for (const result of state.book.snapshot(snapshot)) this.depthResult(state, result);
      } catch (error) {
        state.book.reset(); state.history = [];
        if (!this.stopping) this.record("snapshot_error", { ...state.market, depthEpoch: state.depthEpoch, message: asError(error).message });
      } finally { state.bufferingSnapshot = false; clearTimeout(timeout); this.requests.delete(controller); }
    }).catch((error: unknown) => this.options.onFatal(asError(error))).finally(() => {
      state.snapshotPending = false;
      if (!this.stopping && state.depthConnected && state.book.needsSnapshot()) {
        const timer = setTimeout(() => this.requestSnapshot(state), 3_000); timer.unref();
      }
    });
  }
}
function asError(error: unknown) { return error instanceof Error ? error : new Error(String(error)); }

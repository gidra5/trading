import assert from "node:assert/strict";
import { once } from "node:events";
import test from "node:test";
import { createServer } from "../src/index.js";
import type { ApiContext } from "../src/app.js";
import type { WebClient } from "../src/webclient.js";
import { ApiErrorResponseSchema, HealthRouteSchema } from "../src/schemas.js";

function context(overrides: Partial<ApiContext["bot"]> = {}): ApiContext {
  return {
    addWebClient: () => () => {},
    bot: {
      start: (symbol) => ({ symbol, status: "running" }),
      stop: (symbol) => ({ symbol, status: "stopped" }),
      reset: (symbol) => ({ symbol, status: "reset" }),
      openPosition: ({ symbol }) => ({ symbol, status: "accepted" }),
      closePosition: ({ symbol }) => ({ symbol, status: "accepted" }),
      configure: ({ network }) => ({ network, status: "configured" }),
      ...overrides,
    },
  };
}

test("valid bodies reach typed bot controls and responses match their contracts", async (t) => {
  const calls: unknown[] = [];
  const server = await createServer(
    {},
    context({
      start: (symbol) => {
        calls.push(["start", symbol]);
        return { symbol, status: "running" };
      },
      stop: (symbol) => {
        calls.push(["stop", symbol]);
        return { symbol, status: "stopped" };
      },
      configure: (input) => {
        calls.push(["configure", input]);
        return { network: input.network, status: "configured" };
      },
    }),
  );
  t.after(() => server.close());
  const health = await server.inject({ method: "GET", url: "/health" });
  assert.equal(health.statusCode, 200);
  assert.deepEqual(health.json(), { status: "ok" });
  for (const action of ["start", "stop"] as const) {
    const response = await server.inject({
      method: "POST",
      url: `/bot/${action}`,
      payload: { symbol: "BTCUSDT" },
    });
    assert.equal(response.statusCode, 200);
    assert.deepEqual(response.json(), {
      symbol: "BTCUSDT",
      status: action === "start" ? "running" : "stopped",
    });
  }
  const input = {
    network: "demo",
    credentials: { apiKey: "test-key", apiSecret: "test-secret" },
  };
  const response = await server.inject({
    method: "POST",
    url: "/bot/configure",
    payload: input,
  });
  assert.equal(response.statusCode, 200);
  assert.deepEqual(response.json(), { network: "demo", status: "configured" });
  assert.deepEqual(calls, [
    ["start", "BTCUSDT"],
    ["stop", "BTCUSDT"],
    ["configure", input],
  ]);
});

test("invalid or unknown body fields fail with 400 before invoking bot controls", async (t) => {
  let calls = 0;
  const unexpected = (): never => {
    calls++;
    throw new Error("Must not run");
  };
  const server = await createServer(
    {},
    context({ start: unexpected, stop: unexpected, configure: unexpected }),
  );
  t.after(() => server.close());
  for (const url of ["/bot/start", "/bot/stop"]) {
    for (const payload of [
      {},
      { symbol: "" },
      { symbol: "btcusdt" },
      { symbol: 123 },
      { symbol: "BTCUSDT", leverage: 5 },
      null,
      [],
    ]) {
      const response = await server.inject({
        method: "POST",
        url,
        payload: JSON.stringify(payload),
        headers: { "content-type": "application/json" },
      });
      assert.equal(response.statusCode, 400);
      assert.equal(
        ApiErrorResponseSchema.parse(response.json()).statusCode,
        400,
      );
    }
  }
  for (const payload of [
    { network: "paper", credentials: { apiKey: "key", apiSecret: "secret" } },
    { network: "demo", credentials: { apiKey: "key" } },
    { network: "live", credentials: { apiKey: " ", apiSecret: "secret" } },
    {
      network: "demo",
      credentials: {
        apiKey: "key",
        apiSecret: "secret",
        password: "unexpected",
      },
    },
  ]) {
    const response = await server.inject({
      method: "POST",
      url: "/bot/configure",
      payload,
    });
    assert.equal(response.statusCode, 400);
  }
  assert.equal(calls, 0);
});

test("missing and malformed JSON bodies return consistent client errors", async (t) => {
  const server = await createServer({}, context());
  t.after(() => server.close());
  const missing = await server.inject({ method: "POST", url: "/bot/start" });
  assert.equal(missing.statusCode, 400);
  const malformed = await server.inject({
    method: "POST",
    url: "/bot/start",
    payload: "{",
    headers: { "content-type": "application/json" },
  });
  assert.equal(malformed.statusCode, 400);
  assert.equal(
    ApiErrorResponseSchema.parse(malformed.json()).error,
    "Bad Request",
  );
  const oldMethod = await server.inject({ method: "GET", url: "/bot/start" });
  assert.equal(oldMethod.statusCode, 404);
});

test("response schemas strip undeclared data, including credentials", async (t) => {
  const server = await createServer(
    {},
    context({
      configure: (input) => ({
        network: input.network,
        status: "configured",
        credentials: input.credentials,
      }),
    }),
  );
  t.after(() => server.close());
  const response = await server.inject({
    method: "POST",
    url: "/bot/configure",
    payload: {
      network: "live",
      credentials: { apiKey: "private-key", apiSecret: "private-secret" },
    },
  });
  assert.equal(response.statusCode, 200);
  assert.deepEqual(response.json(), { network: "live", status: "configured" });
  assert.ok(!response.body.includes("private"));
});

test("invalid server responses fail validation without exposing internal data", async (t) => {
  const server = await createServer({}, context());
  t.after(() => server.close());
  // Deliberately bypass static typing to exercise the runtime serializer guard.
  server.get(
    "/invalid-response",
    { schema: HealthRouteSchema },
    async () => ({ status: "private-internal-value" }) as never,
  );
  const response = await server.inject({
    method: "GET",
    url: "/invalid-response",
  });
  assert.equal(response.statusCode, 500);
  assert.deepEqual(response.json(), {
    statusCode: 500,
    error: "Internal Server Error",
    message: "Internal Server Error",
  });
});

test("unconnected controls return 501 instead of reporting successful bot actions", async (t) => {
  const server = await createServer({ databasePath: ":memory:" });
  t.after(() => server.close());
  const response = await server.inject({
    method: "POST",
    url: "/bot/start",
    payload: { symbol: "BTCUSDT" },
  });
  assert.equal(response.statusCode, 501);
  assert.deepEqual(response.json(), {
    statusCode: 501,
    error: "Not Implemented",
    message: "Bot controls are not connected yet",
  });
});

test(
  "WebSocket status messages use the registered client and disconnect removes it",
  { timeout: 5000 },
  async (t) => {
    let client!: WebClient;
    let cleanupCount = 0;
    let removed!: () => void;
    const disconnected = new Promise<void>((resolve) => {
      removed = resolve;
    });
    const server = await createServer(
      {},
      {
        ...context(),
        addWebClient(value) {
          client = value;
          return () => {
            cleanupCount++;
            removed();
          };
        },
      },
    );
    t.after(() => server.close());
    await server.ready();
    const socket = await server.injectWS("/ws");
    try {
      for (const type of [
        "botStatus",
        "connectionStatus",
        "marketStatus",
        "backtestStatus",
        "accountStatus",
      ] as const) {
        const received = once(socket, "message");
        assert.equal(client[type]({ status: "ready" }), true);
        const [message, binary] = await received;
        assert.deepEqual(JSON.parse(message.toString()), {
          type,
          message: { status: "ready" },
        });
        assert.equal(binary, false);
      }
    } finally {
      const closed = once(socket, "close");
      socket.terminate();
      await closed;
    }
    await disconnected;
    assert.equal(cleanupCount, 1);
    assert.equal(client.botStatus({ status: "stopped" }), false);
  },
);

test("reset and position endpoints dispatch validated requests to distinct controls", async (t) => {
  const calls: unknown[] = [];
  const server = await createServer(
    {},
    context({
      stop: () => {
        throw new Error("Reset must not call stop");
      },
      reset: (symbol) => {
        calls.push(["reset", symbol]);
        return { symbol, status: "reset" };
      },
      openPosition: (input) => {
        calls.push(["open", input]);
        return { symbol: input.symbol, status: "accepted" };
      },
      closePosition: (input) => {
        calls.push(["close", input]);
        return { symbol: input.symbol, status: "accepted" };
      },
    }),
  );
  t.after(() => server.close());
  const reset = await server.inject({
    method: "POST",
    url: "/bot/reset",
    payload: { symbol: "BTCUSDT" },
  });
  assert.equal(reset.statusCode, 200);
  assert.deepEqual(reset.json(), { symbol: "BTCUSDT", status: "reset" });
  const open = {
    symbol: "BTCUSDT",
    side: "long",
    quantity: 0.1,
    lifecycle: { lifetimeMs: 60000, stopLossPrice: 90, takeProfitPrice: 110 },
  };
  const close = {
    symbol: "BTCUSDT",
    side: "long",
    quantity: 0.05,
    positionId: "position-1",
  };
  for (const [action, payload] of [
    ["open", open],
    ["close", close],
  ] as const) {
    const response = await server.inject({
      method: "POST",
      url: `/bot/position/${action}`,
      payload,
    });
    assert.equal(response.statusCode, 200);
    assert.deepEqual(response.json(), {
      symbol: "BTCUSDT",
      status: "accepted",
    });
  }
  assert.deepEqual(calls, [
    ["reset", "BTCUSDT"],
    ["open", open],
    ["close", close],
  ]);
});

test("position schemas reject invalid sizes, sides and lifecycle fields before controls run", async (t) => {
  const unexpected = (): never => {
    throw new Error("Invalid input reached controls");
  };
  const server = await createServer(
    {},
    context({
      openPosition: unexpected,
      closePosition: unexpected,
      reset: unexpected,
    }),
  );
  t.after(() => server.close());
  for (const action of ["open", "close"]) {
    for (const payload of [
      {},
      { symbol: "BTCUSDT", side: "buy", quantity: 1 },
      { symbol: "BTCUSDT", side: "long", quantity: 0 },
      { symbol: "BTCUSDT", side: "short", quantity: -1 },
      { symbol: "BTCUSDT", side: "long", quantity: "1" },
      { symbol: "BTCUSDT", side: "long", quantity: 1, extra: true },
    ]) {
      const response = await server.inject({
        method: "POST",
        url: `/bot/position/${action}`,
        payload,
      });
      assert.equal(response.statusCode, 400);
    }
  }
  for (const lifecycle of [
    { lifetimeMs: -1 },
    { stopLossPrice: 0 },
    { takeProfitPrice: 0 },
    { unknown: 1 },
  ]) {
    const response = await server.inject({
      method: "POST",
      url: "/bot/position/open",
      payload: { symbol: "BTCUSDT", side: "long", quantity: 1, lifecycle },
    });
    assert.equal(response.statusCode, 400);
  }
  const reset = await server.inject({
    method: "POST",
    url: "/bot/reset",
    payload: {},
  });
  assert.equal(reset.statusCode, 400);
});

test("default reset and position controls return 501", async (t) => {
  const server = await createServer({ databasePath: ":memory:" });
  t.after(() => server.close());
  for (const [url, payload] of [
    ["/bot/reset", { symbol: "BTCUSDT" }],
    ["/bot/position/open", { symbol: "BTCUSDT", side: "long", quantity: 1 }],
    ["/bot/position/close", { symbol: "BTCUSDT", side: "long", quantity: 1 }],
  ] as const) {
    const response = await server.inject({ method: "POST", url, payload });
    assert.equal(response.statusCode, 501);
  }
});

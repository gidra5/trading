import assert from "node:assert/strict";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { sql } from "drizzle-orm";
import { z } from "zod";
import type { Immutable } from "@trading/bot-algo/trading-api";
import { createContext } from "../src/app.js";
import { createServer } from "../src/index.js";
import { botSettings, botStates, openDatabase } from "../src/db/index.js";
import { createBotStore } from "../src/db/bot-store.js";
import {
  BotSettingsSchema,
  BotStateSchema,
  type BotConfigureRequest,
} from "../src/schemas.js";

const config: BotConfigureRequest = {
  network: "demo",
  credentials: { apiKey: "fixture-key", apiSecret: "fixture-secret" },
};

// Both directions must agree; changing a Zod property cannot silently drift from SQL row types.
type Equal<A, B> =
  (<T>() => T extends A ? 1 : 2) extends <T>() => T extends B ? 1 : 2
    ? true
    : false;
const stateTypesMatch: Equal<
  typeof botStates.$inferSelect,
  z.infer<typeof BotStateSchema>
> = true;
const settingsTypesMatch: Equal<
  Immutable<typeof botSettings.$inferSelect>,
  Immutable<z.infer<typeof BotSettingsSchema>>
> = true;

test("migrations and SQLite settings initialize an empty database", (t) => {
  const connection = openDatabase(":memory:");
  t.after(() => connection.close());
  const store = createBotStore(connection.db);
  assert.equal(stateTypesMatch, true);
  assert.equal(settingsTypesMatch, true);
  assert.equal(store.getConfig(), null);
  assert.deepEqual(store.listStates(), []);
  assert.equal(store.getState("BTCUSDT"), null);
  assert.deepEqual(connection.db.get(sql`PRAGMA foreign_keys`), {
    foreign_keys: 1,
  });
  assert.deepEqual(connection.db.get(sql`PRAGMA busy_timeout`), {
    timeout: 5000,
  });
  assert.deepEqual(
    connection.db.get(sql`SELECT count(*) AS count FROM __drizzle_migrations`),
    { count: 1 },
  );
});

test("configuration and symbol state survive restart and migration replay", (t) => {
  const directory = mkdtempSync(join(tmpdir(), "trading-db-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const filename = join(directory, "nested", "state.sqlite");
  const first = openDatabase(filename);
  try {
    const context = createContext(createBotStore(first.db));
    assert.deepEqual(context.bot.configure(config), {
      network: "demo",
      status: "configured",
    });
    context.store.setActive("BTCUSDT", true);
    context.store.setActive("ETHUSDT", false);
    assert.deepEqual(first.db.get(sql`PRAGMA journal_mode`), {
      journal_mode: "wal",
    });
  } finally {
    first.close();
  }
  const second = openDatabase(filename);
  try {
    const store = createBotStore(second.db);
    assert.deepEqual(store.getConfig(), config);
    assert.equal(store.getState("BTCUSDT")?.active, true);
    assert.equal(store.getState("ETHUSDT")?.active, false);
    store.setConfig({ ...config, network: "testnet" });
    store.setActive("BTCUSDT", false);
    assert.equal(store.getConfig()?.network, "testnet");
    assert.equal(store.listStates().length, 2);
    assert.equal(second.db.select().from(botSettings).all().length, 1);
    assert.deepEqual(
      second.db.get(sql`SELECT count(*) AS count FROM __drizzle_migrations`),
      { count: 1 },
    );
    store.clearConfig();
    assert.equal(store.getConfig(), null);
  } finally {
    second.close();
  }
});

test("Zod validates direct Drizzle writes and rejects corrupt data on reads", (t) => {
  const connection = openDatabase(":memory:");
  t.after(() => connection.close());
  const { db } = connection;
  assert.throws(() =>
    db
      .insert(botStates)
      .values({ symbol: "btc", active: true, updatedAt: 0 })
      .run(),
  );
  assert.throws(() =>
    db
      .insert(botSettings)
      .values({
        id: 1,
        configuration: { ...config, network: "invalid" as "demo" },
        updatedAt: 0,
      })
      .run(),
  );
  assert.throws(() =>
    db
      .insert(botSettings)
      .values({
        id: 1,
        configuration: {
          ...config,
          credentials: { apiKey: " ", apiSecret: "" },
        },
        updatedAt: 0,
      })
      .run(),
  );
  assert.throws(() =>
    db
      .insert(botStates)
      .values({ symbol: "BTCUSDT", active: true, updatedAt: -1 })
      .run(),
  );
  assert.deepEqual(db.select().from(botStates).all(), []);
  assert.throws(() =>
    db.run(sql`INSERT INTO bot_states VALUES ('BTCUSDT', 2, 0)`),
  );
  assert.throws(() =>
    db.run(sql`INSERT INTO bot_settings VALUES (2, '{}', 0)`),
  );
  assert.throws(() =>
    db.run(sql`INSERT INTO bot_settings VALUES (1, 'bad-json', 0)`),
  );
  db.run(sql`INSERT INTO bot_settings VALUES (1, '{}', 0)`);
  assert.throws(() => createBotStore(db).getConfig());
});

test("transactions roll back all writes and connection close is idempotent", () => {
  const connection = openDatabase(":memory:");
  try {
    assert.throws(
      () =>
        connection.db.transaction((tx) => {
          tx.insert(botStates)
            .values({ symbol: "BTCUSDT", active: true, updatedAt: 0 })
            .run();
          tx.insert(botSettings)
            .values({ id: 1, configuration: config, updatedAt: 0 })
            .run();
          throw new Error("abort");
        }),
      /abort/,
    );
    assert.deepEqual(connection.db.select().from(botStates).all(), []);
    assert.deepEqual(connection.db.select().from(botSettings).all(), []);
  } finally {
    connection.close();
  }
  connection.close();
  assert.throws(() => connection.db.select().from(botStates).all(), /not open/);
});

test("the default configure endpoint persists settings without returning credentials", async (t) => {
  const directory = mkdtempSync(join(tmpdir(), "trading-server-db-"));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const databasePath = join(directory, "state.sqlite");
  const server = await createServer({ databasePath });
  try {
    const response = await server.inject({
      method: "POST",
      url: "/bot/configure",
      payload: config,
    });
    assert.equal(response.statusCode, 200);
    assert.deepEqual(response.json(), {
      network: "demo",
      status: "configured",
    });
    assert.ok(!response.body.includes("fixture"));
    const invalid = await server.inject({
      method: "POST",
      url: "/bot/configure",
      payload: { ...config, network: "invalid" },
    });
    assert.equal(invalid.statusCode, 400);
    const start = await server.inject({
      method: "POST",
      url: "/bot/start",
      payload: { symbol: "BTCUSDT" },
    });
    assert.equal(start.statusCode, 501);
  } finally {
    await server.close();
  }
  const connection = openDatabase(databasePath);
  try {
    assert.deepEqual(createBotStore(connection.db).getConfig(), config);
  } finally {
    connection.close();
  }
});

test("in-memory servers have independent configuration", async () => {
  const first = openDatabase(":memory:");
  const second = openDatabase(":memory:");
  try {
    const store = createBotStore(first.db);
    store.setConfig(config);
    assert.equal(createBotStore(second.db).getConfig(), null);
    assert.deepEqual(store.getConfig(), config);
  } finally {
    first.close();
    second.close();
  }
});

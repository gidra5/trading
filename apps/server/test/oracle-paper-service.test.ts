import assert from "node:assert/strict";
import { existsSync, mkdtempSync, readdirSync, unlinkSync, rmdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import Fastify from "fastify";
import { serializerCompiler, validatorCompiler } from "fastify-type-provider-zod";
import { OraclePaperService } from "../src/oracle-paper-service.js";
import { oraclePaperRoutes } from "../src/oracle-paper-routes.js";
import type { DenseDecision, DenseObservation } from "../src/oracle-dense-controller.js";

const observation = (t: number): DenseObservation => ({ features: [Array(59).fill(0), Array(59).fill(0)],
  prices: [100, 100], timestamps: [t - 1, t] });
const initial = { observation: observation(0), account: { quote: 100, asset: 0, previousPrice: 100, orders: [] },
  terminalSeconds: 2, consumedQuote: 0, filled: false };

function fixture(t: TestContext) {
  const directory = mkdtempSync(path.join(tmpdir(), "oracle-paper-service-"));
  let saved: DenseDecision | undefined;
  let closes = 0, checkpoints = 0;
  let breakWorker: (reason: string) => void;
  let rejectCheckpoint = false;
  const decision = (time: number): DenseDecision => ({ phase: time === 2 ? "complete" : "active", update: time,
    timestamp: time, account: { quote: 100, asset: 0, price: 100, previousPrice: 100, orders: [] },
    terminalWealth: time === 2 ? 100 : null, diagnostics: {} });
  const service = new OraclePaperService({ symbol: "BTCUSDT", checkpointEvery: 1,
    worker: { pythonExecutable: "unused", configPath: "unused", cwd: directory, stateDirectory: directory } }, () => {
    let last = decision(0);
    let failure: string | undefined;
    breakWorker = reason => { failure = reason; };
    return {
      get failureReason() { return failure; },
      async initialize() { return last; },
      async onTick(input) {
        if (failure) throw new Error(failure);
        if (input.timestamps[1] !== last.timestamp + 1) throw new Error("Rejected history");
        last = decision(input.timestamps[1]); return last;
      },
      async checkpoint() {
        if (rejectCheckpoint) {
          rejectCheckpoint = false;
          failure = "Checkpoint write failed";
          throw new Error(failure);
        }
        saved = last; checkpoints++;
        writeFileSync(path.join(directory, "current.json"), "{}");
        return last;
      },
      async restore() { if (!saved) throw new Error("No checkpoint"); last = saved; return saved; },
      async close() { closes++; },
    };
  });
  t.after(async () => {
    await service.close();
    for (const file of readdirSync(directory)) unlinkSync(path.join(directory, file));
    rmdirSync(directory);
  });
  return { service, directory, counts: () => ({ closes, checkpoints }),
    failWorker: (reason: string) => breakWorker(reason),
    failNextCheckpoint: () => { rejectCheckpoint = true; } };
}

test("paper service reports an idle worker failure and resumes its saved boundary", async t => {
  const { service, failWorker, counts } = fixture(t);
  await service.start(initial);
  failWorker("Forecast process exited");
  assert.equal(service.status().running, false);
  assert.equal(service.status().failure, "Forecast process exited");
  assert.equal((await service.resume()).update, 0);
  assert.equal(counts().closes, 1);
  assert.equal(service.status().failure, null);
  assert.equal(service.status().running, true);
  await service.tick(observation(1));
});

test("a failed tick checkpoint is not published as an accepted decision", async t => {
  const { service, failNextCheckpoint, counts } = fixture(t);
  await service.start(initial);
  failNextCheckpoint();
  await assert.rejects(service.tick(observation(1)), /Checkpoint write failed/);
  assert.equal(service.status().running, false);
  assert.equal(service.status().decision?.update, 0);
  assert.equal(service.status().failure, "Checkpoint write failed");
  assert.equal(counts().closes, 1);
  assert.equal((await service.resume()).update, 0);
  assert.equal((await service.tick(observation(1))).update, 1);
});

test("rejected history preserves the healthy worker; stopping a failed worker preserves its checkpoint", async t => {
  const { service, failWorker, counts } = fixture(t);
  await service.start(initial);
  await assert.rejects(service.tick(observation(2)), /Rejected history/);
  assert.equal(service.status().running, true);
  assert.equal(service.status().failure, null);
  assert.equal(counts().closes, 0);
  await service.tick(observation(1));
  failWorker("Input pipe failed");
  await service.stop();
  assert.equal(service.status().failure, "Input pipe failed");
  assert.equal((await service.resume()).update, 1);
});

test("paper service saves, stops, resumes, and refuses to replay a completed episode", async t => {
  const { service, directory, counts } = fixture(t);
  await service.start(initial);
  assert.equal(counts().checkpoints, 1);
  await assert.rejects(service.start(initial), /exists/);
  await service.tick(observation(1));
  await service.stop();
  assert.equal(service.status().running, false);
  assert.equal((await service.resume()).update, 1);
  const final = await service.tick(observation(2));
  assert.equal(final.terminalWealth, 100);
  assert.equal(service.status().running, false);
  assert.ok(existsSync(path.join(directory, "completed.json")));
  await assert.rejects(service.resume(), /completed/);
  await service.reset();
  assert.equal(service.status().checkpointAvailable, false);
  assert.equal(service.status().completed, false);
  await service.start(initial);
  await assert.rejects(service.reset(), /Stop/);
});

test("paper routes validate market inputs and expose the complete lifecycle", async t => {
  const { service } = fixture(t);
  const server = Fastify();
  server.setValidatorCompiler(validatorCompiler);
  server.setSerializerCompiler(serializerCompiler);
  await server.register(oraclePaperRoutes, { service });
  t.after(() => server.close());
  const post = (suffix: string, payload?: unknown) => server.inject({ method: "POST", url: "/bot/oracle-paper/" + suffix, payload });
  assert.equal((await post("start", { ...initial, observation: { ...observation(0), features: [[], []] } })).statusCode, 400);
  assert.equal((await post("start", initial)).statusCode, 200);
  assert.equal((await post("start", initial)).statusCode, 409);
  assert.equal((await post("tick", observation(1))).statusCode, 200);
  assert.equal((await post("checkpoint")).statusCode, 200);
  assert.equal((await post("stop")).statusCode, 200);
  assert.equal((await post("resume")).json().update, 1);
  const result = await post("tick", observation(2));
  assert.equal(result.statusCode, 200);
  assert.equal(result.json().phase, "complete");
  assert.equal((await server.inject({ method: "GET", url: "/bot/oracle-paper" })).json().completed, true);
  assert.equal((await post("reset")).statusCode, 200);
});

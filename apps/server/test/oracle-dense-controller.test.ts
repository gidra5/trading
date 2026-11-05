import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtempSync, rmdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import { DenseOracleController, type DenseObservation, type DenseInitialization } from "../src/oracle-dense-controller.js";

const observation = (time: number): DenseObservation => ({
  features: [Array(59).fill(0), Array(59).fill(0)], prices: [100, 100], timestamps: [time - 1, time],
});
const initial: DenseInitialization = { observation: observation(0), terminalSeconds: 3,
  account: { quote: 100, asset: 0, previousPrice: 100, orders: [] }, consumedQuote: 0, filled: false };

function worker(t: TestContext, body = "process.stdout.write(JSON.stringify(reply)+'\\n');", timeout = 2000) {
  const directory = mkdtempSync(path.join(tmpdir(), "oracle-dense-controller-"));
  const script = path.join(directory, "worker.cjs");
  writeFileSync(script, `const lines=require('node:readline').createInterface({input:process.stdin});
lines.on('line',line=>{const request=JSON.parse(line);const time=request.observation.timestamps[1];
const reply={id:request.id,phase:time===3?'complete':'active',update:time,timestamp:time,
account:{quote:100,asset:0,price:100,previousPrice:100,orders:[]},terminalWealth:time===3?100:null,diagnostics:{}};
${body}});`);
  const launches: { args: readonly string[]; hidden?: boolean }[] = [];
  const controller = new DenseOracleController({ pythonExecutable: "configured-python", configPath: "config.json",
    cwd: directory, initializationTimeoutMs: timeout, tickTimeoutMs: timeout }, {
    spawn: (executable, args, options) => {
      assert.equal(executable, "configured-python");
      launches.push({ args, hidden: options.windowsHide });
      return spawn(process.execPath, [script], options);
    },
  });
  t.after(async () => { await controller.close(); unlinkSync(script); rmdirSync(directory); });
  return { controller, launches };
}

test("dense worker persists through serial decisions and enforces flat completion", async t => {
  const { controller, launches } = worker(t);
  assert.equal((await controller.initialize(initial)).account.quote, 100);
  for (let i = 1; i <= 3; i++) {
    const result = await controller.onTick(observation(i));
    assert.equal(result.update, i);
    assert.equal(result.phase, i === 3 ? "complete" : "active");
  }
  await assert.rejects(controller.onTick(observation(4)), /not ready/);
  assert.equal(launches.length, 1);
  assert.equal(launches[0].hidden, true);
  assert.deepEqual(launches[0].args.slice(-2), ["--config", "config.json"]);
});

test("dense controller rejects concurrent, duplicate and skipped ticks without double execution", async t => {
  const { controller } = worker(t, "setTimeout(()=>process.stdout.write(JSON.stringify(reply)+'\\n'),30);");
  const starting = controller.initialize(initial);
  await assert.rejects(controller.initialize(initial), /busy/);
  await starting;
  await assert.rejects(controller.onTick(observation(2)), /exactly one second/);
  const first = controller.onTick(observation(1));
  await assert.rejects(controller.onTick(observation(1)), /not ready/);
  await first;
  await assert.rejects(controller.onTick(observation(1)), /exactly one second/);
  assert.equal((await controller.onTick(observation(2))).update, 2);
});

test("malformed, mismatched and duplicate replies permanently stop the controller", async t => {
  for (const body of ["process.stdout.write('not-json\\n');",
    "reply.id='wrong';process.stdout.write(JSON.stringify(reply)+'\\n');",
    "reply.account.price=-1;process.stdout.write(JSON.stringify(reply)+'\\n');",
    "const s=JSON.stringify(reply)+'\\n';process.stdout.write(s+s);"] ) {
    const { controller } = worker(t, body);
    await assert.rejects(controller.initialize(initial));
    await assert.rejects(controller.initialize(initial));
  }
});

test("terminal success cannot retain an asset or pending order", async t => {
  const { controller } = worker(t, "if(time===3)reply.account.asset=.01;process.stdout.write(JSON.stringify(reply)+'\\n');");
  await controller.initialize(initial);
  await controller.onTick(observation(1));
  await controller.onTick(observation(2));
  await assert.rejects(controller.onTick(observation(3)), /terminal account/);
});

test("forecast errors and timeouts do not permit an ambiguous retry", async t => {
  const errored = worker(t, "process.stdout.write(JSON.stringify({id:request.id,error:{message:'forecast failed',failed:true}})+'\\n');").controller;
  await assert.rejects(errored.initialize(initial), /forecast failed/);
  await assert.rejects(errored.initialize(initial), /forecast failed/);
  const hung = worker(t, "", 100).controller;
  await assert.rejects(hung.initialize(initial), /timed out/);
  await assert.rejects(hung.initialize(initial), /timed out/);
});

test("shutdown rejects pending work and waits for process exit", async t => {
  const { controller } = worker(t, "");
  const pending = assert.rejects(controller.initialize(initial), /closed during evaluation/);
  await controller.close();
  await pending;
  await controller.close();
});

test("invalid inputs are rejected locally without losing the worker", async t => {
  const { controller } = worker(t);
  await assert.rejects(controller.initialize({ ...initial, consumedQuote: 1, filled: false }), /Invalid/);
  await assert.rejects(controller.initialize({ ...initial, observation: { ...observation(0), features: [[], []] } }), /Invalid/);
  await controller.initialize(initial);
  await assert.rejects(controller.onTick({ ...observation(1), prices: [101, 100] }), /accepted history/);
  assert.equal((await controller.onTick(observation(1))).account.quote, 100);
});

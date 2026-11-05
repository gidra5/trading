import assert from "node:assert/strict";
import { spawn } from "node:child_process";
import { mkdtempSync, rmdirSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import test, { type TestContext } from "node:test";
import { OracleForecastProvider, type OracleForecastInput } from "../src/oracle-forecast-provider.js";

const input: OracleForecastInput = { observedPrice: 100, features: Array.from({ length: 178 }, () => 0) };
const treeExpression = `(price) => ({children:[{probability:1,price,children:[{probability:1,price:price+1}]}]})`;

function worker(t: TestContext, body: string, timeoutMs = 2000) {
  const directory = mkdtempSync(path.join(tmpdir(), "oracle-forecast-provider-"));
  const script = path.join(directory, "worker.cjs");
  writeFileSync(script, `const readline=require('node:readline');
const tree=${treeExpression};
const lines=readline.createInterface({input:process.stdin});
lines.on('line',line=>{const request=JSON.parse(line);${body}});
`);
  const launches: { executable: string; args: readonly string[]; hidden: boolean | undefined }[] = [];
  const provider = new OracleForecastProvider({
    pythonExecutable: "configured-python", checkpoint: "configured-checkpoint", cwd: directory, timeoutMs,
  }, { spawn: (executable, args, options) => {
    launches.push({ executable, args, hidden: options.windowsHide });
    return spawn(process.execPath, [script], options);
  } });
  t.after(async () => {
    await provider.close();
    unlinkSync(script);
    rmdirSync(directory);
  });
  return { provider, launches };
}

test("persistent provider correlates concurrent JSONL responses and launches a hidden configured worker", async (t) => {
  const { provider, launches } = worker(t, `setTimeout(()=>process.stdout.write(JSON.stringify({id:request.id,tree:tree(request.observedPrice)})+String.fromCharCode(10)),request.observedPrice===100?30:0);`);
  const first = provider.predict(input);
  const second = provider.predict({ ...input, observedPrice: 200 });
  const [firstTree, secondTree] = await Promise.all([first, second]);
  assert.equal(secondTree.children[0].price, 200);
  assert.equal(firstTree.children[0].price, 100);
  assert.equal(launches.length, 1);
  assert.equal(launches[0].executable, "configured-python");
  assert.equal(launches[0].hidden, true);
  assert.deepEqual(launches[0].args.slice(-4), ["--checkpoint", "configured-checkpoint", "--device", "auto"]);
});

test("an inference error rejects only its request and later inference remains available", async (t) => {
  const { provider } = worker(t, `process.stdout.write(JSON.stringify(request.observedPrice===100?{id:request.id,error:{code:'invalid-request',message:'bad sample'}}:{id:request.id,tree:tree(request.observedPrice)})+String.fromCharCode(10));`);
  await assert.rejects(provider.predict(input), /invalid-request.*bad sample/);
  assert.equal((await provider.predict({ ...input, observedPrice: 200 })).children[0].price, 200);
});

test("malformed and mismatched worker responses fail every pending request", async (t) => {
  for (const output of ["not-json", JSON.stringify({ id: "not-requested", tree: { children: [] } }),
    JSON.stringify({ id: "1", tree: { children: [{ probability: 1, price: -1 }] } })]) {
    const { provider } = worker(t, `process.stdout.write(${JSON.stringify(output)}+String.fromCharCode(10));`);
    const failures = [provider.predict(input), provider.predict(input)].map((promise) => assert.rejects(promise));
    await Promise.all(failures);
    await assert.rejects(provider.predict(input));
    await provider.close();
  }
});

test("timeout terminates a hung worker and rejects subsequent requests", async (t) => {
  const { provider } = worker(t, "", 100);
  await assert.rejects(provider.predict(input), /timed out/);
  await assert.rejects(provider.predict(input), /timed out/);
  await provider.close();
});

test("closing cancels pending inference, waits for process exit, and is idempotent", async (t) => {
  const { provider } = worker(t, "");
  const rejected = assert.rejects(provider.predict(input), /closed before inference/);
  await provider.close();
  await rejected;
  await provider.close();
  await assert.rejects(provider.predict(input), /closed/);
});

test("worker exit rejects inference and includes only bounded diagnostic output", async (t) => {
  const { provider } = worker(t, `process.stderr.write('x'.repeat(10000));process.exit(7);`);
  await assert.rejects(provider.predict(input), (error: Error) => {
    assert.match(error.message, /exited \(7\)/);
    assert.ok(error.message.length < 4200);
    return true;
  });
});

test("invalid inputs are rejected locally without poisoning the persistent worker", async (t) => {
  const { provider } = worker(t, `process.stdout.write(JSON.stringify({id:request.id,tree:tree(request.observedPrice)})+String.fromCharCode(10));`);
  for (const invalid of [
    { ...input, observedPrice: 0 }, { ...input, observedPrice: Infinity },
    { ...input, features: [] }, { ...input, features: [...input.features.slice(1), NaN] },
  ]) await assert.rejects(provider.predict(invalid), /178 finite features/);
  assert.equal((await provider.predict(input)).children[0].price, 100);
});

test("missing executables reject pending work without hanging shutdown", async () => {
  const provider = new OracleForecastProvider({ pythonExecutable: path.join(tmpdir(), "missing-oracle-python.exe"),
    checkpoint: "missing.pt", cwd: tmpdir() });
  await assert.rejects(provider.predict(input), /could not run/);
  await provider.close();
});

test("worker dimensions are explicit instead of silently accepting unsupported horizons", () => {
  assert.throws(() => new OracleForecastProvider({ pythonExecutable: "python", checkpoint: "test.pt", cwd: tmpdir(), horizon: 1 }), /horizon 2/);
  assert.throws(() => new OracleForecastProvider({ pythonExecutable: "python", checkpoint: "test.pt", cwd: tmpdir(), bins: 4 }), /3 continuous bins/);
});

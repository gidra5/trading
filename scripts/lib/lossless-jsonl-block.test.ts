import assert from "node:assert/strict";
import test from "node:test";
import { encodeLosslessJsonlBlock, decodeLosslessJsonlBlock } from "./lossless-jsonl-block.ts";

test("lexical block roundtrips entire lines including unsafe IDs, spelling and unknown raw fields", () => {
  const raw = '{ "s":"币安人生USDT", "u":9223372036854775807, "b":[["0001.23000","-0.000"]], "future":{"unknown":1e+300} }';
  const lines = [
    ` {"recordedAt":1793664000000,"raw":${JSON.stringify(raw)},"unknown":[true,null,"a\\nb",{"escaped":"\\u0041"}]} \n`,
    '{"recordedAt":1793664000001,"number":-0,"decimal":"00.00120","raw":"{malformed unknown future payload"}\n',
    '{"recordedAt":1793664000002,"number":-1.230,"decimal":"000.01000","raw":"{\\n  \\"unknown\\": true\\n}"}',
  ];
  assert.deepEqual(decodeLosslessJsonlBlock(encodeLosslessJsonlBlock(lines)), lines);
});
test("changing arrays, schema, numeric types and decimal scales preserve every byte and row order", () => {
  const lines = Array.from({ length: 200 }, (_, index) => JSON.stringify({ recordedAt: 1000 + index,
    payload: { raw: `{"E":${index},"a":[${Array.from({ length: index % 8 }, (_, level) => `["${100 + level}.000","${index}.${"0".repeat(index % 4)}1"]`).join(",")}]}`,
      string: index % 2 ? "-0.00" : "001.2300", optional: index % 3 ? true : null } }) + "\n");
  assert.deepEqual(decodeLosslessJsonlBlock(encodeLosslessJsonlBlock(lines)), lines);
});
test("invalid input and truncated blocks fail instead of silently dropping data", () => {
  assert.throws(() => encodeLosslessJsonlBlock(['{"missing":}']));
  const block = encodeLosslessJsonlBlock(['{"x":1}\n', '{"x":2}\n']);
  assert.throws(() => decodeLosslessJsonlBlock(block.subarray(0, block.length - 1)));
});

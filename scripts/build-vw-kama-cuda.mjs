import { mkdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(repoRoot, "packages/bot-algo/native/cuda/vw-kama-cuda.cu");
const outputDirectory = path.join(repoRoot, "packages/bot-algo/native/cuda/build");
const output = path.join(outputDirectory, "libvw_kama_cuda.so");
const nvcc = process.env.NVCC ?? "/usr/local/cuda/bin/nvcc";

mkdirSync(outputDirectory, { recursive: true });
const result = spawnSync(nvcc, [
  "-O3",
  "--use_fast_math",
  "--std=c++17",
  "--shared",
  "-Xcompiler",
  "-fPIC",
  "-gencode",
  "arch=compute_86,code=sm_86",
  "-gencode",
  "arch=compute_86,code=compute_86",
  source,
  "-o",
  output,
], { cwd: repoRoot, encoding: "utf8" });

if (result.error) throw result.error;
if (result.status !== 0) {
  process.stderr.write(result.stderr);
  process.exit(result.status ?? 1);
}
process.stdout.write(`Built ${path.relative(repoRoot, output)}\n`);

import { copyFileSync, existsSync, mkdirSync } from "node:fs";
import { spawnSync } from "node:child_process";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const source = path.join(repoRoot, "packages/bot-algo/native/cuda/vw-kama-cuda.cu");
const outputDirectory = path.join(repoRoot, "packages/bot-algo/native/cuda/build");
const windows = process.platform === "win32";
const output = path.resolve(
  process.env.VW_KAMA_CUDA_OUTPUT
    ?? path.join(outputDirectory, windows ? "vw_kama_cuda.dll" : "libvw_kama_cuda.so"),
);

mkdirSync(outputDirectory, { recursive: true });

const localCudaRoot = path.join(repoRoot, ".tools", "cuda-12.8");
const localNvccRoot = path.join(
  localCudaRoot,
  "cuda_nvcc-windows-x86_64-12.8.93-archive",
);
const localCudartRoot = path.join(
  localCudaRoot,
  "cuda_cudart-windows-x86_64-12.8.90-archive",
);
const systemCudaRoot = process.env.CUDA_PATH;
const nvcc = process.env.NVCC
  ?? (windows
    ? firstExisting(
        path.join(localNvccRoot, "bin", "nvcc.exe"),
        systemCudaRoot && path.join(systemCudaRoot, "bin", "nvcc.exe"),
      ) ?? "nvcc.exe"
    : "/usr/local/cuda/bin/nvcc");

const args = [
  "-O3",
  "--use_fast_math",
  "--std=c++20",
  "--shared",
  ...(windows ? ["-Xcompiler", "/MD"] : ["-Xcompiler", "-fPIC"]),
  "-gencode",
  "arch=compute_86,code=sm_86",
  "-gencode",
  "arch=compute_86,code=compute_86",
];

let environment = { ...process.env };
if (windows) {
  const cudartRoot = firstExisting(
    localCudartRoot,
    systemCudaRoot,
  );
  if (!cudartRoot) {
    throw new Error(
      "CUDA runtime headers are missing. Run `npm run mlp:bootstrap` or set CUDA_PATH.",
    );
  }
  args.push(
    "-I", path.join(cudartRoot, "include"),
    "-L", path.join(cudartRoot, "lib", "x64"),
    "-lcudart",
  );
  environment = { ...environment, ...visualStudioEnvironment() };
}
args.push(source, "-o", output);

const result = spawnSync(nvcc, args, {
  cwd: repoRoot,
  env: environment,
  encoding: "utf8",
});

if (result.error) throw result.error;
if (result.status !== 0) {
  process.stderr.write(result.stdout ?? "");
  process.stderr.write(result.stderr ?? "");
  process.exit(result.status ?? 1);
}
if (windows) {
  const cudart = path.join(localCudartRoot, "bin", "cudart64_12.dll");
  const runtimeOutput = path.join(path.dirname(output), path.basename(cudart));
  if (existsSync(cudart) && !existsSync(runtimeOutput)) {
    copyFileSync(cudart, runtimeOutput);
  }
}
process.stdout.write(`Built ${path.relative(repoRoot, output)}\n`);

function firstExisting(...candidates) {
  return candidates.find((candidate) => candidate && existsSync(candidate));
}

function visualStudioEnvironment() {
  const roots = [
    process.env.VSINSTALLDIR,
    "C:\\Program Files (x86)\\Microsoft Visual Studio\\2022\\BuildTools",
    "C:\\Program Files\\Microsoft Visual Studio\\2022\\BuildTools",
  ].filter(Boolean);
  const setup = roots
    .map((root) => path.join(root, "VC", "Auxiliary", "Build", "vcvars64.bat"))
    .find(existsSync);
  if (!setup) {
    throw new Error(
      "Visual Studio 2022 C++ Build Tools are missing; install the x64 C++ toolchain.",
    );
  }
  const result = spawnSync(
    process.env.ComSpec ?? "cmd.exe",
    ["/d", "/s", "/c", `""${setup}" >nul && set"`],
    { encoding: "utf8", windowsVerbatimArguments: true },
  );
  if (result.status !== 0) {
    throw new Error(`Failed to load the Visual Studio build environment: ${result.stderr}`);
  }
  return Object.fromEntries(
    result.stdout
      .split(/\r?\n/)
      .flatMap((line) => {
        const equals = line.indexOf("=");
        return equals > 0 ? [[line.slice(0, equals), line.slice(equals + 1)]] : [];
      }),
  );
}

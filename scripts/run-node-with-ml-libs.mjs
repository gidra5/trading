import { spawn } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";

const repositoryRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const arguments_ = process.argv.slice(2);
const development = arguments_[0] === "--development";
if (development) arguments_.shift();
const [entryArgument, ...entryArguments] = arguments_;
if (!entryArgument) throw new Error("A repository-relative Node entry point is required.");

const environment = { ...process.env };
environment.TMPDIR ||= os.tmpdir();
environment.PYTHONUTF8 ||= "1";
environment.PYTHONIOENCODING ||= "utf-8";
if (development) {
  environment.NODE_OPTIONS = [
    environment.NODE_OPTIONS,
    "--conditions=development",
  ].filter(Boolean).join(" ");
}
const cudnnDirectory = environment.TRADING_MLP_CUDNN_DIR?.trim()
  || bundledCudnnDirectory();
if (cudnnDirectory) {
  if (!fs.existsSync(cudnnDirectory) || !fs.statSync(cudnnDirectory).isDirectory()) {
    throw new Error(`TRADING_MLP_CUDNN_DIR is not a directory: ${cudnnDirectory}`);
  }
  const libraryPathVariable = process.platform === "win32"
    ? Object.keys(environment).find((key) => key.toLowerCase() === "path") ?? "Path"
    : "LD_LIBRARY_PATH";
  environment[libraryPathVariable] = [
    cudnnDirectory,
    environment[libraryPathVariable],
  ]
    .filter(Boolean)
    .join(path.delimiter);
}

const bundledNode = path.join(
  repositoryRoot,
  ".node-22",
  process.platform === "win32" ? "node.exe" : "bin/node",
);
const nodeExecutable = fs.existsSync(bundledNode) ? bundledNode : process.execPath;
const child = spawn(
  nodeExecutable,
  [path.resolve(repositoryRoot, entryArgument), ...entryArguments],
  { cwd: repositoryRoot, env: environment, stdio: "inherit" },
);

for (const signal of ["SIGINT", "SIGTERM"]) {
  process.once(signal, () => child.kill(signal));
}
child.on("error", (error) => {
  console.error(error);
  process.exitCode = 1;
});
child.on("exit", (code) => {
  process.exitCode = code ?? 1;
});

function bundledCudnnDirectory() {
  if (process.platform === "win32") {
    const torchLibraries = path.join(
      repositoryRoot,
      ".venv-ml",
      "Lib",
      "site-packages",
      "torch",
      "lib",
    );
    return fs.existsSync(torchLibraries) ? torchLibraries : undefined;
  }
  const libraryRoot = path.join(repositoryRoot, ".venv-ml", "lib");
  if (!fs.existsSync(libraryRoot)) return undefined;
  const pythonDirectories = fs.readdirSync(libraryRoot)
    .filter((name) => /^python\d+\.\d+$/.test(name))
    .sort()
    .reverse();
  for (const pythonDirectory of pythonDirectories) {
    const candidate = path.join(
      libraryRoot,
      pythonDirectory,
      "site-packages",
      "nvidia",
      "cudnn",
      "lib",
    );
    if (fs.existsSync(candidate)) return candidate;
  }
  return undefined;
}

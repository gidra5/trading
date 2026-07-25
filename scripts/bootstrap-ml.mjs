import { spawnSync } from "node:child_process";
import {
  createWriteStream,
  existsSync,
  mkdirSync,
  rmSync,
} from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import AdmZip from "adm-zip";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const windows = process.platform === "win32";
const toolsRoot = path.join(repoRoot, ".tools");
const uv = await ensureUv();
const python = path.join(
  repoRoot,
  ".venv-ml",
  windows ? "Scripts/python.exe" : "bin/python",
);

if (!existsSync(python)) {
  run(uv, ["venv", "--python", "3.12", path.join(repoRoot, ".venv-ml")]);
}

const installArguments = [
  "pip",
  "install",
  "--python",
  python,
  "--link-mode",
  "copy",
  ...(windows
    ? [
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu130",
        "--index-strategy",
        "unsafe-best-match",
      ]
    : []),
  "-r",
  path.join(repoRoot, "ml", "requirements.txt"),
];
run(uv, installArguments);
run(uv, ["pip", "check", "--python", python]);

if (windows) await ensureWindowsCuda();
run(process.execPath, [path.join(repoRoot, "scripts", "build-vw-kama-cuda.mjs")]);
run(python, [
  "-c",
  "import torch, triton; "
    + "assert torch.cuda.is_available(), 'PyTorch cannot see the CUDA GPU'; "
    + "print({'torch': torch.__version__, 'cuda': torch.version.cuda, "
    + "'triton': triton.__version__, 'gpu': torch.cuda.get_device_name(0)})",
]);

process.stdout.write("ML bootstrap complete.\n");

async function ensureUv() {
  const local = path.join(
    toolsRoot,
    windows ? "uv" : "uv-linux",
    windows ? "uv.exe" : "uv",
  );
  if (existsSync(local)) return local;
  const system = spawnSync("uv", ["--version"], { encoding: "utf8" });
  if (!system.error && system.status === 0) return "uv";

  const version = "0.11.32";
  const directory = path.dirname(local);
  mkdirSync(directory, { recursive: true });
  if (windows) {
    const archive = path.join(toolsRoot, `uv-${version}-windows.zip`);
    await download(
      `https://github.com/astral-sh/uv/releases/download/${version}/uv-x86_64-pc-windows-msvc.zip`,
      archive,
    );
    new AdmZip(archive).extractAllTo(directory, true);
    rmSync(archive, { force: true });
  } else {
    const archive = path.join(toolsRoot, `uv-${version}-linux.tar.gz`);
    await download(
      `https://github.com/astral-sh/uv/releases/download/${version}/uv-x86_64-unknown-linux-gnu.tar.gz`,
      archive,
    );
    run("tar", ["-xzf", archive, "-C", directory, "--strip-components=1"]);
    rmSync(archive, { force: true });
  }
  if (!existsSync(local)) throw new Error(`uv bootstrap did not create ${local}`);
  return local;
}

async function ensureWindowsCuda() {
  const root = path.join(toolsRoot, "cuda-12.8");
  const components = [
    {
      expected: path.join(
        root,
        "cuda_nvcc-windows-x86_64-12.8.93-archive",
        "bin",
        "nvcc.exe",
      ),
      url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/windows-x86_64/cuda_nvcc-windows-x86_64-12.8.93-archive.zip",
    },
    {
      expected: path.join(
        root,
        "cuda_cudart-windows-x86_64-12.8.90-archive",
        "bin",
        "cudart64_12.dll",
      ),
      url: "https://developer.download.nvidia.com/compute/cuda/redist/cuda_cudart/windows-x86_64/cuda_cudart-windows-x86_64-12.8.90-archive.zip",
    },
  ];
  mkdirSync(root, { recursive: true });
  for (const [index, component] of components.entries()) {
    if (existsSync(component.expected)) continue;
    const archive = path.join(root, `.component-${index}.zip`);
    await download(component.url, archive);
    new AdmZip(archive).extractAllTo(root, true);
    rmSync(archive, { force: true });
    if (!existsSync(component.expected)) {
      throw new Error(`CUDA bootstrap did not create ${component.expected}`);
    }
  }
}

async function download(url, file) {
  process.stdout.write(`Downloading ${url}\n`);
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok || !response.body) {
    throw new Error(`${url}: HTTP ${response.status}`);
  }
  await pipeline(Readable.fromWeb(response.body), createWriteStream(file));
}

function run(command, args) {
  const result = spawnSync(command, args, {
    cwd: repoRoot,
    env: { ...process.env },
    stdio: "inherit",
  });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} exited with code ${result.status ?? 1}`);
  }
}

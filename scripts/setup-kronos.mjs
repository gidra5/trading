import { existsSync } from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
process.env.HF_HUB_DISABLE_XET ||= "1";
const windows = process.platform === "win32";
const python = path.join(repoRoot, ".venv-ml", windows ? "Scripts/python.exe" : "bin/python");
const uv = path.join(repoRoot, ".tools", windows ? "uv/uv.exe" : "uv-linux/uv");

if (!existsSync(python) || !existsSync(uv)) {
  run(process.execPath, [path.join(repoRoot, "scripts", "bootstrap-ml.mjs")]);
} else {
  run(uv, [
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
  ]);
  run(uv, ["pip", "check", "--python", python]);
}

run(process.execPath, [
  path.join(repoRoot, "scripts", "run-ml-python.mjs"),
  "ml/setup_kronos.py",
  ...process.argv.slice(2),
]);

function run(command, args) {
  const result = spawnSync(command, args, { cwd: repoRoot, stdio: "inherit" });
  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${command} exited with code ${result.status ?? 1}`);
  }
}

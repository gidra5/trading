import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const root = path.resolve(path.dirname(process.argv[1]), "..");
const bundledPython = path.join(root, ".venv-ml", "bin", "python");
const python = fs.existsSync(bundledPython) ? bundledPython : "python3";
const script = path.join(root, "ml", "backtest_binance_multiscale_index.py");
const child = spawn(python, [script, ...process.argv.slice(2)], {
  cwd: root,
  env: {
    ...process.env,
    PYTHONPATH: [
      path.join(root, "ml"),
      process.env.PYTHONPATH,
    ]
      .filter(Boolean)
      .join(path.delimiter),
  },
  stdio: "inherit",
});

child.on("error", (error) => {
  console.error(error);
  process.exitCode = 1;
});
child.on("exit", (code, signal) => {
  if (signal) {
    console.error(`Backtest terminated by ${signal}.`);
    process.exitCode = 1;
  } else {
    process.exitCode = code ?? 1;
  }
});

import { spawnSync } from "node:child_process";
import { createWriteStream, existsSync, mkdirSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const project = path.join(root, "docs/theory/lean");
const windows = process.platform === "win32";
const elanHome = process.env.ELAN_HOME || path.join(root, ".tools/elan");
const bin = path.join(elanHome, "bin");
const executable = (name) => path.join(bin, `${name}${windows ? ".exe" : ""}`);
const env = {
  ...process.env,
  ELAN_HOME: elanHome,
  MATHLIB_CACHE_DIR: process.env.MATHLIB_CACHE_DIR || path.join(root, ".tools/mathlib-cache"),
  PATH: `${bin}${path.delimiter}${process.env.PATH || ""}`,
};
const mode = process.argv[2];
if (!["setup", "check"].includes(mode)) {
  throw new Error("Usage: node scripts/theory-lean.mjs setup|check");
}

function run(command, args, capture = false) {
  const result = spawnSync(command, args, {
    cwd: project, env, stdio: capture ? "pipe" : "inherit", encoding: "utf8",
    windowsHide: true,
  });
  if (capture) {
    process.stdout.write(result.stdout || "");
    process.stderr.write(result.stderr || "");
  }
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.status})`);
  return result.stdout || "";
}

async function installElan() {
  if (existsSync(executable("elan"))) return;
  const targets = {
    "win32-x64": "x86_64-pc-windows-msvc.zip",
    "linux-x64": "x86_64-unknown-linux-gnu.tar.gz",
    "linux-arm64": "aarch64-unknown-linux-gnu.tar.gz",
    "darwin-x64": "x86_64-apple-darwin.tar.gz",
    "darwin-arm64": "aarch64-apple-darwin.tar.gz",
  };
  const target = targets[`${process.platform}-${process.arch}`];
  if (!target) throw new Error("Unsupported platform; install elan and set ELAN_HOME to its directory.");
  const bootstrap = path.join(root, ".tools/lean-bootstrap");
  mkdirSync(bootstrap, { recursive: true });
  const archive = path.join(bootstrap, `elan-${target}`);
  const response = await fetch(`https://github.com/leanprover/elan/releases/download/v4.2.4/elan-${target}`);
  if (!response.ok) throw new Error(`Elan download failed: ${response.status}`);
  await pipeline(Readable.fromWeb(response.body), createWriteStream(archive));
  // Both Windows tar.exe and Unix tar support the corresponding release archive.
  run("tar", ["-xf", archive, "-C", bootstrap]);
  run(path.join(bootstrap, windows ? "elan-init.exe" : "elan-init"),
    ["-y", "--no-modify-path", "--default-toolchain", "none"]);
}

if (mode === "setup") {
  await installElan();
  const toolchain = readFileSync(path.join(project, "lean-toolchain"), "utf8").trim();
  run(executable("elan"), ["toolchain", "install", toolchain]);
  // The committed Lake manifest pins transitive dependencies; reserve `lake update`
  // for deliberate dependency upgrades.
  // Mathlib's cache does not traverse imports through downstream namespaces.
  // Collect the direct Mathlib imports in our library instead.
  const sources = readdirSync(path.join(project, "TradingTheory"), { recursive: true })
    .filter((name) => name.endsWith(".lean"));
  const imports = new Set(sources.flatMap((name) =>
    [...readFileSync(path.join(project, "TradingTheory", name), "utf8")
      .matchAll(/^import (Mathlib(?:\.[A-Za-z0-9_]+)*)\s*$/gm)].map((m) => m[1])));
  if (imports.size === 0) throw new Error("No Mathlib imports found for the proof cache.");
  run(executable("lake"), ["exe", "cache", "get", ...imports]);
} else if (!existsSync(executable("lake"))) {
  throw new Error("Lean is not installed locally. Run npm run theory:setup first.");
}

run(executable("lake"), ["build"]);
const audit = run(executable("lake"), ["env", "lean", "Audit.lean"], true);
const allowed = new Set(["propext", "Classical.choice", "Quot.sound"]);
const reports = [...audit.matchAll(/depends on axioms:\s*\[([^\]]*)\]/g)];
if (reports.length === 0 && !audit.includes("does not depend on any axioms")) {
  throw new Error("Missing Lean axiom report.");
}
for (const report of reports) {
  const unexpected = report[1].split(",").map((s) => s.trim()).filter((s) => s && !allowed.has(s));
  if (unexpected.length) throw new Error(`Unapproved proof axioms: ${unexpected.join(", ")}`);
}
console.log("Theory proofs built and axiom audit passed.");

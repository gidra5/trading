import fs from "node:fs";
import path from "node:path";

interface CalibrationResult {
  policyId: string;
  [key: string]: unknown;
}

interface CalibrationProgress {
  version: 1;
  contract: "kronos-bot-policy-calibration-progress-v1";
  forecastRunSignature: string;
  policyGridSignature: string;
  candidateCount: number;
  completedCandidates: number;
  results: CalibrationResult[];
}

interface CalibrationShard {
  version: 1;
  contract: "foundation-bot-policy-calibration-shard-v1";
  forecastRunSignature: string;
  policyGridSignature: string;
  candidateCount: number;
  start: number;
  end: number;
  results: CalibrationResult[];
}

const progressFile = path.resolve(requiredArgument("progress"));
const shardFiles = process.argv
  .filter((item) => item.startsWith("--shard="))
  .map((item) => path.resolve(item.slice("--shard=".length)));
if (shardFiles.length === 0) throw new Error("At least one --shard=... file is required.");
const progress = JSON.parse(fs.readFileSync(progressFile, "utf8")) as CalibrationProgress;
if (progress.version !== 1
  || progress.contract !== "kronos-bot-policy-calibration-progress-v1"
  || progress.results.length !== progress.completedCandidates) {
  throw new Error("Base calibration progress is invalid.");
}
const merged = new Array<CalibrationResult | undefined>(progress.candidateCount);
for (let index = 0; index < progress.results.length; index += 1) merged[index] = progress.results[index];
for (const file of shardFiles) {
  const shard = JSON.parse(fs.readFileSync(file, "utf8")) as CalibrationShard;
  if (shard.version !== 1
    || shard.contract !== "foundation-bot-policy-calibration-shard-v1"
    || shard.forecastRunSignature !== progress.forecastRunSignature
    || shard.policyGridSignature !== progress.policyGridSignature
    || shard.candidateCount !== progress.candidateCount
    || shard.start < 0
    || shard.end <= shard.start
    || shard.end > shard.candidateCount
    || shard.results.length !== shard.end - shard.start) {
    throw new Error(`Calibration shard does not match the base progress: ${file}`);
  }
  for (let offset = 0; offset < shard.results.length; offset += 1) {
    const index = shard.start + offset;
    const existing = merged[index];
    const candidate = shard.results[offset]!;
    if (existing && existing.policyId !== candidate.policyId) {
      throw new Error(`Conflicting policy result at candidate ${index}.`);
    }
    merged[index] = candidate;
  }
}
const missing = merged.flatMap((item, index) => item ? [] : [index]);
if (missing.length > 0) {
  throw new Error(`Merged calibration misses ${missing.length} candidates; first is ${missing[0]}.`);
}
const output: CalibrationProgress = {
  ...progress,
  completedCandidates: progress.candidateCount,
  results: merged as CalibrationResult[],
};
const temporary = `${progressFile}.merge.tmp`;
fs.writeFileSync(temporary, `${JSON.stringify(output, null, 2)}\n`);
fs.renameSync(temporary, progressFile);
console.log(JSON.stringify({
  progressFile,
  candidateCount: output.candidateCount,
  shards: shardFiles.length,
}, null, 2));

function requiredArgument(name: string): string {
  const prefix = `--${name}=`;
  const value = process.argv.find((item) => item.startsWith(prefix))?.slice(prefix.length).trim();
  if (!value) throw new Error(`--${name}=... is required.`);
  return value;
}

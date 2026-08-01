import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_STUDY_ROOT = "data/training/runs/studies";

export function pruneCompletedTrainingState(directory, { dryRun = false } = {}) {
  const manifestFile = path.join(directory, "manifest.json");
  const manifest = readJson(manifestFile);
  const study = readJson(path.join(directory, "study.json"));
  const modelFile = path.join(directory, manifest?.modelFile ?? "model.onnx");
  if (!manifest?.verification?.verifiedAt
    || !study?.bestValidationMetrics
    || !regularNonemptyFile(modelFile)) {
    return { directory, eligible: false, removedBytes: 0, removedFiles: [] };
  }

  const candidates = new Set([
    path.join(directory, "checkpoints", "last.json"),
    path.join(directory, "checkpoints", "best.json"),
  ]);
  const removedFiles = [];
  let removedBytes = 0;
  for (const file of candidates) {
    let stat;
    try {
      stat = fs.statSync(file);
    } catch (error) {
      if (error?.code === "ENOENT") continue;
      throw error;
    }
    if (!stat.isFile()) continue;
    if (!dryRun) fs.unlinkSync(file);
    removedBytes += stat.size;
    removedFiles.push(path.basename(file));
  }

  if (!dryRun) {
    const retainedManifest = {
      ...manifest,
      training: {
        ...manifest.training,
        retention: {
          inferenceArtifact: manifest.modelFile ?? "model.onnx",
          optimizerCheckpoint: "pruned-after-artifact-verification",
          pytorchBestWeights: "pruned-after-artifact-verification",
        },
      },
    };
    delete retainedManifest.checkpointFile;
    atomicWrite(manifestFile, `${JSON.stringify(retainedManifest, null, 2)}\n`);
  }
  return { directory, eligible: true, removedBytes, removedFiles };
}

export function pruneCompletedMetricsOnlyTrainingState(
  directory,
  { dryRun = false } = {},
) {
  const studyFile = path.join(directory, "study.json");
  const study = readJson(studyFile);
  if (!study?.bestValidationMetrics || study.finalizedEarly === true) {
    return { directory, eligible: false, removedBytes: 0, removedFiles: [] };
  }
  const removed = removeFiles(trainingStateFiles(directory), { dryRun });
  if (!dryRun) {
    atomicWrite(studyFile, `${JSON.stringify({
      ...study,
      retention: {
        ...(study.retention ?? {}),
        optimizerCheckpoint: "pruned-after-completed-screen",
        pytorchBestWeights: "pruned-after-completed-screen",
      },
    }, null, 2)}\n`);
  }
  return { directory, eligible: true, ...removed };
}

export function pruneStudyArtifact(directory, { dryRun = false } = {}) {
  const study = readJson(path.join(directory, "study.json"));
  if (!study?.bestValidationMetrics || study.finalizedEarly === true) {
    return { directory, eligible: false, removedBytes: 0, removedFiles: [] };
  }
  const manifestFile = path.join(directory, "manifest.json");
  const manifest = readJson(manifestFile);
  const candidates = new Set([
    manifestFile,
    path.join(directory, manifest?.modelFile ?? "model.onnx"),
    path.join(directory, manifest?.verificationFixture?.inputFile ?? "verification-input.f32"),
    path.join(directory, manifest?.verificationFixture?.outputFile ?? "verification-output.f32"),
  ]);
  const removed = removeFiles(candidates, { dryRun });
  return { directory, eligible: true, ...removed };
}

export function pruneCompletedStudies(root, { dryRun = false } = {}) {
  const directories = findManifestDirectories(root);
  const results = directories.map(
    (directory) => pruneCompletedTrainingState(directory, { dryRun }),
  );
  return {
    root,
    dryRun,
    scanned: directories.length,
    eligible: results.filter((result) => result.eligible).length,
    removedFiles: results.reduce((sum, result) => sum + result.removedFiles.length, 0),
    removedBytes: results.reduce((sum, result) => sum + result.removedBytes, 0),
  };
}

export function retainBestArtifactsPerDelay(root, { dryRun = false } = {}) {
  const entries = findStudyDirectories(root).map((directory) => ({
    directory,
    study: readJson(path.join(directory, "study.json")),
  })).filter((entry) =>
    entry.study?.bestValidationMetrics
    && entry.study.finalizedEarly !== true
    && Number.isFinite(entry.study.predictionDelayMs)
    && Number.isFinite(entry.study.bestValidationMetrics.klDivergence));
  const winners = new Map();
  for (const entry of entries) {
    const previous = winners.get(entry.study.predictionDelayMs);
    if (!previous
      || entry.study.bestValidationMetrics.klDivergence
        < previous.study.bestValidationMetrics.klDivergence) {
      winners.set(entry.study.predictionDelayMs, entry);
    }
  }
  let removedFiles = 0;
  let removedBytes = 0;
  for (const entry of entries) {
    const winner = winners.get(entry.study.predictionDelayMs);
    const training = fs.existsSync(path.join(entry.directory, "manifest.json"))
      ? pruneCompletedTrainingState(entry.directory, { dryRun })
      : pruneCompletedMetricsOnlyTrainingState(entry.directory, { dryRun });
    removedFiles += training.removedFiles.length;
    removedBytes += training.removedBytes;
    if (entry.directory === winner?.directory) continue;
    const artifact = pruneStudyArtifact(entry.directory, { dryRun });
    removedFiles += artifact.removedFiles.length;
    removedBytes += artifact.removedBytes;
  }
  return {
    root,
    dryRun,
    scanned: entries.length,
    retainedArtifacts: [...winners.values()].map((entry) => ({
      delayMs: entry.study.predictionDelayMs,
      validationKl: entry.study.bestValidationMetrics.klDivergence,
      directory: entry.directory,
    })).sort((left, right) => left.delayMs - right.delayMs),
    removedFiles,
    removedBytes,
  };
}

function findManifestDirectories(root) {
  if (!fs.existsSync(root)) return [];
  const found = [];
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const child = path.join(directory, entry.name);
      if (fs.existsSync(path.join(child, "manifest.json"))) found.push(child);
      else pending.push(child);
    }
  }
  return found;
}

function findStudyDirectories(root) {
  if (!fs.existsSync(root)) return [];
  const found = [];
  const pending = [root];
  while (pending.length > 0) {
    const directory = pending.pop();
    for (const entry of fs.readdirSync(directory, { withFileTypes: true })) {
      if (!entry.isDirectory()) continue;
      const child = path.join(directory, entry.name);
      if (fs.existsSync(path.join(child, "study.json"))) found.push(child);
      else pending.push(child);
    }
  }
  return found;
}

function regularNonemptyFile(file) {
  try {
    const stat = fs.statSync(file);
    return stat.isFile() && stat.size > 0;
  } catch (error) {
    if (error?.code === "ENOENT") return false;
    throw error;
  }
}

function trainingStateFiles(directory) {
  return new Set([
    path.join(directory, "checkpoints", "last.json"),
    path.join(directory, "checkpoints", "best.json"),
  ]);
}

function removeFiles(files, { dryRun }) {
  const removedFiles = [];
  let removedBytes = 0;
  for (const file of files) {
    let stat;
    try {
      stat = fs.statSync(file);
    } catch (error) {
      if (error?.code === "ENOENT") continue;
      throw error;
    }
    if (!stat.isFile()) continue;
    if (!dryRun) fs.unlinkSync(file);
    removedBytes += stat.size;
    removedFiles.push(path.basename(file));
  }
  return { removedBytes, removedFiles };
}

function readJson(file) {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8"));
  } catch (error) {
    if (error?.code === "ENOENT" || error instanceof SyntaxError) return undefined;
    throw error;
  }
}

function atomicWrite(file, contents) {
  const temporary = `${file}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, contents, { flag: "wx" });
  fs.renameSync(temporary, file);
}

const invokedFile = process.argv[1] ? path.resolve(process.argv[1]) : undefined;
if (invokedFile === fileURLToPath(import.meta.url)) {
  const positional = process.argv.slice(2).filter(
    (argument) => !["--dry-run", "--best-per-delay"].includes(argument),
  );
  const root = path.resolve(positional[0] ?? DEFAULT_STUDY_ROOT);
  const dryRun = process.argv.includes("--dry-run");
  const result = process.argv.includes("--best-per-delay")
    ? retainBestArtifactsPerDelay(root, { dryRun })
    : pruneCompletedStudies(root, { dryRun });
  process.stdout.write(`${JSON.stringify({
    event: "mlp-study-retention-pruned",
    ...result,
    removedGiB: result.removedBytes / (1024 ** 3),
  })}\n`);
}

import fs from "node:fs";
import path from "node:path";
import { spawnSync } from "node:child_process";
import { fileURLToPath } from "node:url";

const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const basePlanFile = path.join(
  repoRoot,
  "ml/training-plans/mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle-mi-continuation.json",
);
const basePlan = JSON.parse(fs.readFileSync(basePlanFile, "utf8"));
const queueRunDir = path.join(
  repoRoot,
  "data/ml-runs/mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle-mi-ablation",
);
const queueStatusFile = path.join(queueRunDir, "status.json");
const featureStatisticsSource = path.join(
  repoRoot,
  "data/ml-runs/mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle-mi-continuation/feature-statistics.npz",
);
const variants = [
  {
    key: "temporal-mi-only",
    label: "Temporal MI only",
    temporalMutualInformation: 1,
    oracleMutualInformation: 0,
  },
  {
    key: "oracle-mi-only",
    label: "Oracle MI only",
    temporalMutualInformation: 0,
    oracleMutualInformation: 1,
  },
  {
    key: "oracle-mi-half-scratch",
    label: "0.5 Oracle MI · fresh initialization",
    temporalMutualInformation: 0,
    oracleMutualInformation: 0.5,
    patience: 8,
    freshInitialization: true,
  },
];

fs.mkdirSync(queueRunDir, { recursive: true });

function writeQueueStatus(value) {
  const temporary = `${queueStatusFile}.${process.pid}.tmp`;
  fs.writeFileSync(temporary, `${JSON.stringify({
    updatedAt: new Date().toISOString(),
    ...value,
  }, null, 2)}\n`);
  fs.renameSync(temporary, queueStatusFile);
}

function linkFeatureStatistics(runDir) {
  if (!fs.existsSync(featureStatisticsSource)) return;
  fs.mkdirSync(runDir, { recursive: true });
  const destination = path.join(runDir, "feature-statistics.npz");
  if (fs.existsSync(destination)) return;
  try {
    fs.linkSync(featureStatisticsSource, destination);
  } catch (error) {
    if (error?.code !== "EXDEV") throw error;
    fs.copyFileSync(featureStatisticsSource, destination);
  }
}

writeQueueStatus({
  stage: "running",
  pid: process.pid,
  variants: variants.map(({ key, label }) => ({ key, label })),
});

for (const [index, variant] of variants.entries()) {
  const suffix = variant.key;
  const planFile = path.join(
    repoRoot,
    `ml/training-plans/mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle-${suffix}.json`,
  );
  const artifactDir = `data/models/mlp/mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle-${suffix}`;
  const runDir = `data/ml-runs/mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle-${suffix}`;
  const plan = JSON.parse(JSON.stringify(basePlan));
  plan.id = `mlp-direct-oracle-temporal-v12-delay-3600s-full-minute-oracle-${suffix}`;
  plan.label = `Direct 255-action MLP v12 · 60-minute full hindsight · CE+pMSE+${variant.label}`;
  plan.artifactDir = artifactDir;
  plan.runDir = runDir;
  plan.training.epochs = 64;
  plan.training.patience = variant.patience ?? 4;
  plan.training.lossWeights = {
    crossEntropy: 1,
    probabilityMse: 1,
    excessEntropy: 0,
    temporalMutualInformation: variant.temporalMutualInformation,
    oracleMutualInformation: variant.oracleMutualInformation,
  };
  if (variant.freshInitialization) {
    delete plan.training.initializeFromCheckpoint;
  }
  fs.writeFileSync(planFile, `${JSON.stringify(plan, null, 2)}\n`);
  linkFeatureStatistics(path.join(repoRoot, runDir));

  const manifestFile = path.join(repoRoot, artifactDir, "manifest.json");
  if (fs.existsSync(manifestFile)) {
    writeQueueStatus({
      stage: "running",
      pid: process.pid,
      activeIndex: index + 1,
      activeVariant: variant.key,
      message: "Completed artifact already exists; skipping.",
    });
    continue;
  }

  writeQueueStatus({
    stage: "running",
    pid: process.pid,
    activeIndex: index + 1,
    activeVariant: variant.key,
    planFile,
  });
  const result = spawnSync(
    process.execPath,
    [
      path.join(repoRoot, "scripts/run-node-with-ml-libs.mjs"),
      path.join(repoRoot, "scripts/run-mlp-training.mjs"),
      "--training-only",
      "--plan",
      planFile,
    ],
    { cwd: repoRoot, stdio: "inherit" },
  );
  if (result.status !== 0) {
    writeQueueStatus({
      stage: "failed",
      pid: process.pid,
      activeIndex: index + 1,
      activeVariant: variant.key,
      exitCode: result.status,
      signal: result.signal,
    });
    process.exit(result.status ?? 1);
  }
}

writeQueueStatus({
  stage: "complete",
  pid: process.pid,
  completedAt: new Date().toISOString(),
});

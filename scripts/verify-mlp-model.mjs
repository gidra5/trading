import fs from "node:fs/promises";
import path from "node:path";
import * as ort from "onnxruntime-node";
import {
  MLP_FEATURE_SCHEMA_VERSION,
  MLP_INPUT_FEATURE_COUNT,
  MLP_OUTPUT_ACTION_COUNT,
  predictMlpDistribution,
} from "@trading/bot-algo";

const PYTORCH_ABSOLUTE_TOLERANCE = 5e-5;
const PYTORCH_RELATIVE_TOLERANCE = 1e-5;
const PROVIDER_ABSOLUTE_TOLERANCE = 5e-3;
const PROVIDER_RELATIVE_TOLERANCE = 2e-3;
const PROVIDER_MAX_PROBABILITY_ERROR = 5e-4;
const PROVIDER_ACTION_MEAN_RMSE = 5e-2;

const artifact = path.resolve(process.argv[2] ?? "");
if (!process.argv[2]) throw new Error("Usage: npm run mlp:verify -- /path/to/model-artifact");
const manifestFile = path.join(artifact, "manifest.json");
const manifest = JSON.parse(await fs.readFile(manifestFile, "utf8"));
if (manifest.featureSchemaVersion !== MLP_FEATURE_SCHEMA_VERSION
  || manifest.inputFeatureCount !== MLP_INPUT_FEATURE_COUNT
  || manifest.outputRepresentation !== "base-action-logits"
  || manifest.outputActionCount !== MLP_OUTPUT_ACTION_COUNT
  || manifest.actionGrid?.length !== MLP_OUTPUT_ACTION_COUNT
  || manifest.hiddenLayerCount !== 16 || manifest.hiddenWidth !== 1024) {
  throw new Error("MLP manifest does not match the current 901 -> 16x1024 -> 255 contract.");
}
const modelFile = artifactFile(manifest.modelFile);
const batch = manifest.verificationFixture?.batchSize ?? 7;
const values = manifest.verificationFixture
  ? await readFloat32(artifactFile(manifest.verificationFixture.inputFile))
  : Float32Array.from({ length: batch * manifest.inputFeatureCount }, (_, index) =>
      Math.sin(index * 0.173) * 0.3 + Math.cos(index * 0.019) * 0.1);
if (values.length !== batch * manifest.inputFeatureCount) {
  throw new Error("MLP verification input fixture has an invalid shape.");
}
const input = new ort.Tensor("float32", values, [batch, manifest.inputFeatureCount]);
const cpu = await ort.InferenceSession.create(modelFile, {
  executionProviders: ["cpu"],
  graphOptimizationLevel: "all",
});
const cpuResult = await cpu.run({ features: input }, ["action_logits"]);
const cpuValues = cpuResult.action_logits.data;
if (!(cpuValues instanceof Float32Array)
  || cpuValues.length !== batch * manifest.outputActionCount
  || !cpuValues.every(Number.isFinite)) {
  throw new Error("MLP CPU verification output has an invalid shape or non-finite values.");
}
let maximumPyTorchError = 0;
let maximumPyTorchRelativeError = 0;
let pytorchPolicyParity = {
  maxProbabilityError: 0,
  probabilityMse: 0,
  actionMeanRmse: 0,
};
if (manifest.verificationFixture) {
  const expected = await readFloat32(artifactFile(manifest.verificationFixture.outputFile));
  if (expected.length !== cpuValues.length || !expected.every(Number.isFinite)) {
    throw new Error("MLP PyTorch verification fixture has an invalid shape or non-finite values.");
  }
  for (let index = 0; index < expected.length; index += 1) {
    const absoluteError = Math.abs(expected[index] - cpuValues[index]);
    const relativeError = absoluteError / Math.max(1, Math.abs(expected[index]));
    maximumPyTorchError = Math.max(maximumPyTorchError, absoluteError);
    maximumPyTorchRelativeError = Math.max(maximumPyTorchRelativeError, relativeError);
    const tolerance = PYTORCH_ABSOLUTE_TOLERANCE
      + PYTORCH_RELATIVE_TOLERANCE * Math.abs(expected[index]);
    if (!(absoluteError <= tolerance)) {
      throw new Error(
        `MLP PyTorch/ONNX parity error ${absoluteError} at output ${index} exceeds `
        + `${tolerance} (atol=${PYTORCH_ABSOLUTE_TOLERANCE}, `
        + `rtol=${PYTORCH_RELATIVE_TOLERANCE}).`,
      );
    }
  }
  pytorchPolicyParity = measurePolicyParity(expected, cpuValues, manifest);
  if (!(pytorchPolicyParity.maxProbabilityError <= PROVIDER_MAX_PROBABILITY_ERROR)
    || !(pytorchPolicyParity.actionMeanRmse <= PROVIDER_ACTION_MEAN_RMSE)) {
    throw new Error(
      `MLP PyTorch/ONNX policy parity failed: max probability error `
      + `${pytorchPolicyParity.maxProbabilityError}, action-mean RMSE `
      + `${pytorchPolicyParity.actionMeanRmse}.`,
    );
  }
}
let provider = "cpu";
let maximumProviderError = 0;
let maximumProviderRelativeError = 0;
let providerPolicyParity = {
  maxProbabilityError: 0,
  probabilityMse: 0,
  actionMeanRmse: 0,
};
const cudaVerification = process.env.TRADING_MLP_VERIFY_CUDA?.trim().toLowerCase() ?? "auto";
if (!["auto", "true", "false"].includes(cudaVerification)) {
  throw new Error("TRADING_MLP_VERIFY_CUDA must be auto, true, or false.");
}
if (cudaVerification !== "false") {
  try {
    const cuda = await ort.InferenceSession.create(modelFile, {
      executionProviders: [{ name: "cuda", deviceId: 0 }],
      graphOptimizationLevel: "all",
    });
    const cudaResult = await cuda.run({ features: input }, ["action_logits"]);
    const expected = cpuValues;
    const actual = cudaResult.action_logits.data;
    if (!(actual instanceof Float32Array)
      || expected.length !== batch * manifest.outputActionCount
      || actual.length !== expected.length
      || !actual.every(Number.isFinite)) {
      throw new Error("MLP verification output has an invalid shape.");
    }
    for (let index = 0; index < expected.length; index += 1) {
      const absoluteError = Math.abs(expected[index] - actual[index]);
      const relativeError = absoluteError / Math.max(1, Math.abs(expected[index]));
      maximumProviderError = Math.max(maximumProviderError, absoluteError);
      maximumProviderRelativeError = Math.max(maximumProviderRelativeError, relativeError);
      const tolerance = PROVIDER_ABSOLUTE_TOLERANCE
        + PROVIDER_RELATIVE_TOLERANCE * Math.abs(expected[index]);
      if (!(absoluteError <= tolerance)) {
        throw new Error(
          `MLP CUDA/CPU raw parity error ${absoluteError} at output ${index} exceeds `
          + `${tolerance} (atol=${PROVIDER_ABSOLUTE_TOLERANCE}, `
          + `rtol=${PROVIDER_RELATIVE_TOLERANCE}).`,
        );
      }
    }
    providerPolicyParity = measurePolicyParity(expected, actual, manifest);
    if (!(providerPolicyParity.maxProbabilityError <= PROVIDER_MAX_PROBABILITY_ERROR)
      || !(providerPolicyParity.actionMeanRmse <= PROVIDER_ACTION_MEAN_RMSE)) {
      throw new Error(
        `MLP CUDA/CPU policy parity failed: max probability error `
        + `${providerPolicyParity.maxProbabilityError}, action-mean RMSE `
        + `${providerPolicyParity.actionMeanRmse}.`,
      );
    }
    provider = "cuda";
  } catch (error) {
    if (cudaVerification === "true") throw error;
    process.stderr.write(
      `MLP CUDA verification unavailable; CPU artifact verification succeeded: `
      + `${error instanceof Error ? error.message : String(error)}\n`,
    );
  }
}
manifest.verification = {
  executionProvider: provider,
  maxAbsolutePyTorchError: maximumPyTorchError,
  maxRelativePyTorchError: maximumPyTorchRelativeError,
  pytorchAbsoluteTolerance: PYTORCH_ABSOLUTE_TOLERANCE,
  pytorchRelativeTolerance: PYTORCH_RELATIVE_TOLERANCE,
  pytorchPolicyParity,
  maxAbsoluteProviderError: maximumProviderError,
  maxRelativeProviderError: maximumProviderRelativeError,
  providerAbsoluteTolerance: PROVIDER_ABSOLUTE_TOLERANCE,
  providerRelativeTolerance: PROVIDER_RELATIVE_TOLERANCE,
  providerPolicyParity,
  verifiedAt: new Date().toISOString(),
};
const temporary = `${manifestFile}.${process.pid}.tmp`;
await fs.writeFile(temporary, `${JSON.stringify(manifest, null, 2)}\n`, { flag: "wx" });
await fs.rename(temporary, manifestFile);
process.stdout.write(
  `${manifest.id}: ${provider.toUpperCase()} verified; `
  + `max |PyTorch-ONNX| = ${maximumPyTorchError}; `
  + `max relative PyTorch/ONNX = ${maximumPyTorchRelativeError}; `
  + `max |CPU-GPU| = ${maximumProviderError}; `
  + `policy pMSE = ${providerPolicyParity.probabilityMse}; `
  + `action-mean RMSE = ${providerPolicyParity.actionMeanRmse}\n`,
);

function measurePolicyParity(expectedRaw, actualRaw, modelManifest) {
  const support = modelManifest.policySupport;
  const actions = Float64Array.from({ length: 255 }, (_, index) =>
    support.visible_lower
    + index * (support.visible_upper - support.visible_lower) / 254);
  let maximumProbabilityError = 0;
  let squaredProbabilityError = 0;
  let squaredActionMeanError = 0;
  let probabilityCells = 0;
  let policyRows = 0;
  for (let batchIndex = 0; batchIndex < batch; batchIndex += 1) {
    const offset = batchIndex * modelManifest.outputActionCount;
    const expectedPrediction = predictMlpDistribution(
      actions,
      expectedRaw.subarray(offset, offset + modelManifest.outputActionCount),
      modelManifest.actionGrid,
    );
    const actualPrediction = predictMlpDistribution(
      actions,
      actualRaw.subarray(offset, offset + modelManifest.outputActionCount),
      modelManifest.actionGrid,
    );
    for (let action = 0; action < actions.length; action += 1) {
      const difference = actualPrediction.probabilities[action]
        - expectedPrediction.probabilities[action];
      maximumProbabilityError = Math.max(maximumProbabilityError, Math.abs(difference));
      squaredProbabilityError += difference * difference;
      probabilityCells += 1;
    }
    squaredActionMeanError += (
      actualPrediction.meanExposure - expectedPrediction.meanExposure
    ) ** 2;
    policyRows += 1;
  }
  return {
    maxProbabilityError: maximumProbabilityError,
    probabilityMse: squaredProbabilityError / probabilityCells,
    actionMeanRmse: Math.sqrt(squaredActionMeanError / policyRows),
  };
}

async function readFloat32(file) {
  const buffer = await fs.readFile(file);
  if (buffer.byteLength % 4 !== 0) throw new Error(`Invalid float32 fixture: ${file}`);
  return new Float32Array(
    buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength),
  );
}

function artifactFile(relativeFile) {
  const resolved = path.resolve(artifact, relativeFile);
  if (!resolved.startsWith(`${artifact}${path.sep}`)) {
    throw new Error(`MLP artifact path escapes its directory: ${relativeFile}`);
  }
  return resolved;
}

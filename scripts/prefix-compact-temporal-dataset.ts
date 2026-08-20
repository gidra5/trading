import fs from "node:fs";
import path from "node:path";

type Manifest = {
  storageLayout?: string;
  examplesBySplit: Record<string, number>;
  files: Record<string, Record<string, string>>;
  [key: string]: unknown;
};

function argument(name: string): string {
  const index = process.argv.indexOf(name);
  const value = index >= 0 ? process.argv[index + 1] : undefined;
  if (!value) throw new Error(`Missing ${name}`);
  return value;
}

const repo = path.resolve(import.meta.dirname, "..");
const source = path.resolve(repo, argument("--source"));
const output = path.resolve(repo, argument("--output"));
const trainExamples = Number.parseInt(argument("--train-examples"), 10);
if (!Number.isSafeInteger(trainExamples) || trainExamples < 1) {
  throw new Error("--train-examples must be a positive integer");
}
if (fs.existsSync(output)) throw new Error(`Output already exists: ${output}`);

const manifest = JSON.parse(
  fs.readFileSync(path.join(source, "manifest.json"), "utf8"),
) as Manifest;
if (manifest.storageLayout !== "temporal-channel-timeline-v1") {
  throw new Error("Source is not a compact temporal dataset");
}
if (trainExamples > manifest.examplesBySplit.train!) {
  throw new Error("Requested prefix is larger than the source train split");
}

fs.mkdirSync(output, { recursive: true });
for (const split of ["train", "validation", "test"]) {
  const files = manifest.files[split]!;
  fs.linkSync(
    path.join(source, files.timelineFeatures!),
    path.join(output, files.timelineFeatures!),
  );
  for (const key of ["origins", "targets", "times"] as const) {
    const filename = files[key]!;
    fs.copyFileSync(path.join(source, filename), path.join(output, filename));
  }
}

for (const [filename, bytesPerExample] of [
  [manifest.files.train!.origins!, 4],
  [manifest.files.train!.targets!, 4],
  [manifest.files.train!.times!, 8],
] as const) {
  fs.truncateSync(path.join(output, filename), trainExamples * bytesPerExample);
}

manifest.examplesBySplit = {
  ...manifest.examplesBySplit,
  train: trainExamples,
};
manifest.generatedAt = new Date().toISOString();
manifest.derivedFrom = path.relative(repo, source).replaceAll("\\", "/");
manifest.derivation = "exact-clean-train-prefix-with-shared-temporal-timelines";
fs.writeFileSync(
  path.join(output, "manifest.json"),
  `${JSON.stringify(manifest, null, 2)}\n`,
  "utf8",
);
console.log(
  `Wrote ${path.relative(repo, output)} as an exact ${trainExamples.toLocaleString()}-example prefix.`,
);

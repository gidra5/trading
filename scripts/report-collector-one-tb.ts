import fs from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const DEFAULT_STEM = "docs/experiments/collector-one-tb-2026-11-02";
type Row = Readonly<{ source: string; records: number; compressedBytes: number }>;
type CodecMeasurement = Readonly<{ component: string; originalGzipBytes: number; encodedBytes: number; roundTripByteExact: boolean;
  scope: string; source: string }>;
type TransferBasis = Readonly<{ candidate: string; note: string; measurements: readonly CodecMeasurement[];
  baselineCorrection?: Readonly<{ components: readonly string[]; factor: number; evidence: string }> }>;
type Benchmark = Readonly<{ generatedAt: string; componentRatios: readonly Readonly<{ component: string; candidate: string;
  originalGzipBytes: number; encodedBytes: number; roundTripByteExact: boolean; evidence: readonly Readonly<{ dataset: string; spanSeconds: number }>[] }>[];
  datasets: readonly Readonly<{ id: string; rows: number; spanSeconds: number; streams: readonly Readonly<{ kind: string;
    baselineGzip1sBytes: number; candidates: readonly Readonly<{ name: string; compressedBytes: number; ratioToGzip1s: number; roundTripByteExact: boolean }>[] }>[] }>[] }>;

export function component(source: string): string {
  if (source.startsWith("BTC-peripherals/")) return source;
  const separator = source.indexOf("/");
  const venue = source.slice(0, separator).split(":")[0];
  return `${venue}/${source.slice(separator + 1)}`;
}

export function capacityRequirement(rows: readonly Row[], activityMultiplier = 3, capBytes = 1e12, reservedFraction = .2) {
  if (!(activityMultiplier > 0) || !(capBytes > 0) || !(reservedFraction >= 0 && reservedFraction < 1)) throw new Error("Invalid capacity settings");
  const baselineBytes = rows.reduce((sum, row) => sum + row.compressedBytes, 0),
    baselineRecords = rows.reduce((sum, row) => sum + row.records, 0);
  const archiveBudget = capBytes * (1 - reservedFraction), stressBytes = baselineBytes * activityMultiplier;
  return { capBytes, reservedFraction, workingReserveBytes: capBytes - archiveBudget, archiveBudget, activityMultiplier,
    baselineBytes, stressBytes, baselineRecords, stressRecords: baselineRecords * activityMultiplier,
    requiredWholeArchiveRatio: archiveBudget / stressBytes, requiredReductionFactor: stressBytes / archiveBudget,
    availableBytesPerModeledRecord: archiveBudget / (baselineRecords * activityMultiplier) };
}

/** Unknown components receive no saving. Transfers are assumptions, not an entropy bound or hard traffic guarantee. */
export function projectMeasuredTransfer(rows: readonly Row[], basis: TransferBasis, capBytes = 1e12, reservedFraction = .2) {
  const ratios = new Map<string, number>();
  for (const measurement of basis.measurements) {
    if (!measurement.roundTripByteExact) throw new Error(`Not a verified lossless measurement: ${measurement.component}`);
    if (!(measurement.originalGzipBytes > 0) || !(measurement.encodedBytes > 0)) throw new Error("Invalid measured byte counts");
    const ratio = measurement.encodedBytes / measurement.originalGzipBytes;
    // Retain the worst observed ratio for this component instead of cherry-picking one busy sample.
    ratios.set(measurement.component, Math.max(ratios.get(measurement.component) ?? 0, ratio));
  }
  const correction = basis.baselineCorrection;
  if (correction && !(correction.factor > 0 && Number.isFinite(correction.factor))) throw new Error("Invalid baseline correction");
  const projectedRows = rows.map((row) => {
    const key = component(row.source), ratio = ratios.get(key) ?? 1;
    const baselineCorrection = correction?.components.includes(key) ? correction.factor : 1;
    return { source: row.source, component: key, originalGzipBytes: row.compressedBytes, measuredRatio: ratios.has(key) ? ratio : null,
      baselineCorrection, correctedBaselineBytes: row.compressedBytes * baselineCorrection,
      projectedBytes: row.compressedBytes * baselineCorrection * ratio };
  });
  const correctedBaselineBytes = projectedRows.reduce((sum, row) => sum + row.correctedBaselineBytes, 0),
    baseBytes = projectedRows.reduce((sum, row) => sum + row.projectedBytes, 0), workingReserveBytes = capBytes * reservedFraction;
  return { candidate: basis.candidate, note: basis.note, correction: correction ?? null,
    originalBaselineBytes: rows.reduce((sum, row) => sum + row.compressedBytes, 0), correctedBaselineBytes, baseBytes,
    codecRatioAgainstCorrectedBaseline: baseBytes / correctedBaselineBytes,
    scenarios: [1, 2, 3].map((activityMultiplier) => ({ activityMultiplier, archiveBytes: baseBytes * activityMultiplier,
      archivePlusReserveBytes: baseBytes * activityMultiplier + workingReserveBytes,
      fitsPhysicalCapWithReserve: baseBytes * activityMultiplier + workingReserveBytes <= capBytes })),
    coveredOriginalBytes: projectedRows.filter((row) => row.measuredRatio !== null).reduce((sum, row) => sum + row.originalGzipBytes, 0),
    unmeasuredOriginalBytes: projectedRows.filter((row) => row.measuredRatio === null).reduce((sum, row) => sum + row.originalGzipBytes, 0), rows: projectedRows };
}

const option = (args: readonly string[], name: string, fallback: string) => { const index = args.indexOf(name); return index < 0 ? fallback : args[index + 1] ?? fallback; };
const tb = (bytes: number) => `${(bytes / 1e12).toFixed(3)} TB`;
const gb = (bytes: number) => `${(bytes / 1e9).toFixed(1)} GB`;
export async function run(args = process.argv.slice(2)) {
  if (args.includes("--help")) {
    console.log("Read-only feasibility analysis: --estimate FILE --transfer-basis FILE --benchmark FILE --refresh-codec-basis --output-stem FILE. Refresh copies existing verified benchmark mappings; no collection or retention settings change.");
    return;
  }
  const estimateFile = option(args, "--estimate", "docs/experiments/collector-storage-capacity-2026-11-01.json"),
    stem = option(args, "--output-stem", DEFAULT_STEM), basisFile = option(args, "--transfer-basis", `${DEFAULT_STEM}-codec-basis.json`),
    benchmarkFile = option(args, "--benchmark", "docs/experiments/collector-lossless-codec-2026-11-02.json");
  const estimateText = await fs.readFile(path.resolve(ROOT, estimateFile), "utf8"), estimate = JSON.parse(estimateText),
    rows = estimate.profiles["all-routes-full-public-replay"].base.rows as readonly Row[];
  const requirement = capacityRequirement(rows);
  const contribution = new Map<string, { records: number; bytes: number }>();
  for (const row of rows) {
    const key = component(row.source), current = contribution.get(key) ?? { records: 0, bytes: 0 };
    current.records += row.records; current.bytes += row.compressedBytes; contribution.set(key, current);
  }
  const components = [...contribution].map(([component, values]) => ({ component, baseRecords: values.records, baseGzipBytes: values.bytes,
    stress3xGzipBytes: values.bytes * 3, share: values.bytes / requirement.baselineBytes })).sort((a, b) => b.baseGzipBytes - a.baseGzipBytes);
  const checkpointScenarios = [300, 3600, 86_400].map((cadenceSeconds) => {
    const factor = estimate.parameters.checkpointSeconds / cadenceSeconds;
    const projectedRows = rows.map((row) => row.source.endsWith("/local-checkpoint")
      ? { ...row, records: row.records * factor, compressedBytes: row.compressedBytes * factor } : row);
    const capacity = capacityRequirement(projectedRows);
    return { cadenceSeconds, eventInformationPreserved: true, baseBytes: capacity.baselineBytes, stress3xBytes: capacity.stressBytes,
      stress3xSavingBytes: requirement.stressBytes - capacity.stressBytes, rows: projectedRows };
  });
  let transfer: ReturnType<typeof projectMeasuredTransfer> | null = null, pending = "Verified codec transfer basis is not yet available.";
  let basis: TransferBasis | null = null, benchmark: Benchmark | null = null;
  try { basis = JSON.parse(await fs.readFile(path.resolve(ROOT, basisFile), "utf8")); } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  try { benchmark = JSON.parse(await fs.readFile(path.resolve(ROOT, benchmarkFile), "utf8")); } catch (error) {
    if ((error as NodeJS.ErrnoException).code !== "ENOENT") throw error;
  }
  if (args.includes("--refresh-codec-basis")) {
    if (!benchmark?.componentRatios.length) throw new Error("No measured codec mappings are available");
    const candidates = [...new Set(benchmark.componentRatios.map((row) => row.candidate))];
    if (candidates.length !== 1) throw new Error("A single consistent codec candidate is required for this projection");
    basis = { candidate: candidates[0]!, note: "Worst measured ratio per component across corresponding full-recorder and legacy windows. Legacy aggregate trades are a conservative granularity transfer, not a new individual-trade measurement. Unmeasured sources remain unchanged. No additional global multi-market baseline correction is credited.",
      measurements: benchmark.componentRatios.map((row) => ({ component: row.component, originalGzipBytes: row.originalGzipBytes,
        encodedBytes: row.encodedBytes, roundTripByteExact: row.roundTripByteExact,
        scope: row.evidence.map((evidence) => `${evidence.dataset} (${evidence.spanSeconds}s)`).join("; "), source: `${benchmarkFile}#componentRatios` })) };
    await fs.writeFile(path.resolve(ROOT, basisFile), JSON.stringify(basis, null, 2) + "\n");
  }
  if (basis) { transfer = projectMeasuredTransfer(rows, basis); pending = ""; }
  const matchedShards = benchmark?.datasets.flatMap((dataset) => {
    const whole = dataset.streams.find((row) => row.kind === "whole-actual-shard"), candidate = whole?.candidates.find((row) => row.name === basis?.candidate);
    if (!whole || !candidate) return [];
    const separatedKindGzipBytes = dataset.streams.filter((row) => row.kind !== "whole-actual-shard").reduce((sum, row) => sum + row.baselineGzip1sBytes, 0);
    return [{ dataset: dataset.id, rows: dataset.rows, spanSeconds: dataset.spanSeconds, separatedKindGzipBytes,
      actualShardGzipBytes: whole.baselineGzip1sBytes, baselineGroupingRatio: whole.baselineGzip1sBytes / separatedKindGzipBytes,
      codecBytes: candidate.compressedBytes, codecRatioAgainstActualShard: candidate.ratioToGzip1s, roundTripByteExact: candidate.roundTripByteExact }];
  }) ?? [];
  const report = { version: 1, generatedAt: new Date().toISOString(), estimate: { file: estimateFile, sha256: createHash("sha256").update(estimateText).digest("hex") },
    scope: { assets: estimate.assetCount, routes: estimate.routeCount, preserveEveryPublishedEvent: true, changeCollectorSettings: false,
    changeUniverse: false, dataDeletion: false, durationDays: 30 }, requirement, components,
    checkpointScenarios: checkpointScenarios.map(({ rows: _rows, ...scenario }) => ({ ...scenario,
      measuredTransfer: basis ? projectMeasuredTransfer(checkpointScenarios.find((row) => row.cadenceSeconds === scenario.cadenceSeconds)!.rows, basis) : null })),
    benchmarkFile: benchmark ? benchmarkFile : null, basisFile: basis ? basisFile : null, matchedShards, transfer, pending: pending || null };
  await fs.writeFile(path.resolve(ROOT, `${stem}.json`), JSON.stringify(report, null, 2) + "\n");
  const lines = ["# Can all-event, all-market collection fit 1 TB/month?", "",
    `Scope remains **${estimate.assetCount} economic identities / ${estimate.routeCount} physical routes**, every published event, 30 days. This report changes no collector, universe, or retention settings. All amounts use decimal TB/GB.`, "",
    `The earlier per-route gzip model is **${tb(requirement.baselineBytes)} base / ${tb(requirement.stressBytes)} at 3× activity**. A 1 TB physical cap with 20% reserved for transient WAL, indices, rotation and working space permits **${gb(requirement.archiveBudget)} of monthly archive data**. The complete archive would need to retain at most **${(requirement.requiredWholeArchiveRatio * 100).toFixed(2)}%** of the earlier gzip footprint, a **${requirement.requiredReductionFactor.toFixed(2)}× reduction**. This is a feasibility target, not an observed compression result.`, "",
    `The model contains ${(requirement.stressRecords / 1e9).toFixed(2)} billion records at 3×: the 800 GB archive allowance averages **${requirement.availableBytesPerModeledRecord.toFixed(2)} bytes per modeled record**. That is not an entropy lower bound—batching can share metadata—but it makes a 10× claim demanding.`, "",
    "## Where the modeled bytes go", "", "| Component | Base gzip | 3× gzip | Share |", "|---|---:|---:|---:|",
    ...components.map((row) => `| ${row.component} | ${gb(row.baseGzipBytes)} | ${gb(row.stress3xGzipBytes)} | ${(100 * row.share).toFixed(1)}% |`), "",
    "## Measured lossless result", "",
    ...(transfer ? [`Candidate: **${transfer.candidate}**. ${transfer.note}`, "",
      `Measured component mappings cover ${gb(transfer.coveredOriginalBytes)} of the earlier base footprint; ${gb(transfer.unmeasuredOriginalBytes)} remains unmeasured and receives no compression saving.`, "",
      transfer.correction ? `Earlier modeled baseline: ${tb(transfer.originalBaselineBytes)}. Matched physical-shard baseline correction: ${tb(transfer.correctedBaselineBytes)}. Codec ratio relative to that corrected baseline: ${(100 * transfer.codecRatioAgainstCorrectedBaseline).toFixed(1)}%. These effects are separate; improved modeling of shared-shard headers is not a codec gain.`
        : `No global shard-baseline correction is applied. The measured-component projection retains ${(100 * transfer.codecRatioAgainstCorrectedBaseline).toFixed(1)}% of the earlier conservative modeled baseline, including unchanged unmeasured sources. This is a conditional transfer estimate, not a direct month-long 687-route measurement.`, "",
      "| Component | Worst measured encoded / 1s gzip | Byte-exact round trip |", "|---|---:|---|",
      ...basis!.measurements.map((row) => `| ${row.component} | ${(100 * row.encodedBytes / row.originalGzipBytes).toFixed(1)}% | ${row.roundTripByteExact ? "yes" : "no"} |`), "",
      "| Activity scenario | Projected archive | Archive + 200 GB reserve | Within 1 TB? |", "|---|---:|---:|---|",
      ...transfer.scenarios.map((row) => `| ${row.activityMultiplier}× | ${tb(row.archiveBytes)} | ${tb(row.archivePlusReserveBytes)} | ${row.fitsPhysicalCapWithReserve ? "yes, conditional projection" : "no"} |`), ""] : [pending, "", "No unmeasured 10× saving is credited and no all-market compliance claim is made.", ""]),
    "The existing 3.31 TB baseline prices gzip using separate per-route/per-stream 1s groups. Production shares metadata across multi-market socket shards; especially the 1s mark/funding rows can be cheaper in a matched shard. Benchmarking must distinguish this baseline correction from the additional gain of the lossless codec. BTC-only busy-stream ratios cannot establish quiet TradFi or 687-route traffic costs.", "",
    ...(matchedShards.length ? ["Matched physical-shard measurements (each candidate uses exactly the baseline's same rows; short windows are not full-hour observations):", "",
      "| Dataset | Observed seconds | Kind-separated 1s gzip | Physical-shard 1s gzip | Encoded / physical shard |", "|---|---:|---:|---:|---:|",
      ...matchedShards.map((row) => `| ${row.dataset} | ${row.spanSeconds.toFixed(2)} | ${row.separatedKindGzipBytes.toLocaleString("en-US")} B | ${row.actualShardGzipBytes.toLocaleString("en-US")} B | ${(100 * row.codecRatioAgainstActualShard).toFixed(1)}% |`), "",
      "These are matching selected wire records regrouped by physical source, not extrapolated bootstrap totals. Optional derived rows and separately stored checkpoints/control records are outside these codec samples and receive no unmeasured saving. The multi-market benchmark's per-kind groups already combine markets. Their grouping comparison isolates additional sharing across stream kinds, not the full difference between 20 markets and 20 isolated one-market files. No further unmeasured sharing factor is multiplied into the global projection.", ""] : []),
    "All accepted codec ratios must be measured against the same input records and pass byte-exact round-trip checks, including raw public payload strings, timestamps, ordering, identifiers, session/sequence fields and gaps. Recompression, dictionary/schema references, reversible integer/delta encoding and longer archive frames can reduce repetition. Lossless recompression does not supply missing exchange events or resolve public L2 into cancellation identities.", "",
    "## Operational interpretation", "",
    "A future archiver can persist short durable WAL chunks, repack only sealed chunks into longer compressed frames, verify exact reconstruction, and retain originals until verification finishes. Transient duplicated chunks and indices need working capacity. Increasing the live flush interval alone trades away crash durability; it is not required to obtain a longer offline compression window.", "",
    "Local full-book checkpoints are derived caches of retained REST seeds plus L2 deltas. Replacing full checkpoint copies with replay offsets, or constructing them on demand, preserves all published events while changing restore/random-access work; that additional unmeasured encoding gain is not credited. Independent BBO events cannot generally be reconstructed from the 100ms depth feed and must remain in an all-event archive.", "",
    "The following future-cadence scenarios reduce only duplicate local checkpoint snapshots. All published wire messages, REST initial/resync seeds and session/gap/quality records remain. This preserves market-event information; a longer initial replay is required when seeking or restarting. Existing archived checkpoint rows are not removed or rewritten by this report.", "",
    "| Future local checkpoint cadence | Base before codec | 3× before codec | 3× saving vs 5m |", "|---|---:|---:|---:|",
    ...checkpointScenarios.map((row) => `| ${row.cadenceSeconds === 300 ? "5 min" : row.cadenceSeconds === 3600 ? "1 hour" : "1 day"} | ${tb(row.baseBytes)} | ${tb(row.stress3xBytes)} | ${gb(row.stress3xSavingBytes)} |`), "",
    ...(basis ? ["With the same measured codec transfers, changing only the future checkpoint cadence gives:", "", "| Cadence | Base archive | 3× archive | 3× + working reserve |", "|---|---:|---:|---:|",
      ...report.checkpointScenarios.map((row) => `| ${row.cadenceSeconds}s | ${tb(row.measuredTransfer!.baseBytes)} | ${tb(row.measuredTransfer!.baseBytes * 3)} | ${tb(row.measuredTransfer!.baseBytes * 3 + requirement.workingReserveBytes)} |`), ""] : []),
    "A physical 1 TB cap cannot guarantee 30 days of unrestricted future event volume. If the measured all-event projection exceeds the cap, the answer is additional lossless engineering, additional capacity, or an explicit user-approved change of scope. This report does not discard assets, downsample the book, shorten 30-day raw history, or silently drop records to enforce a cap.", "",
    "## Reproduce", "", "```powershell", "node --import tsx scripts/report-collector-one-tb.ts", "node --import tsx --test scripts/report-collector-one-tb.test.ts", "```", "",
    `Inputs: [earlier capacity model](${path.basename(estimateFile)})${benchmark ? `, [actual lossless benchmark](${path.basename(benchmarkFile)})` : ""}${basis ? `, [explicit transfer assumptions](${path.basename(basisFile)})` : ""}. Machine-readable result: [JSON](${path.basename(stem)}.json).`, ""];
  await fs.writeFile(path.resolve(ROOT, `${stem}.md`), lines.join("\n"));
  console.log(JSON.stringify({ requirement, pending: report.pending, scenarios: transfer?.scenarios ?? null }));
}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) run().catch((error) => { console.error(error); process.exitCode = 1; });

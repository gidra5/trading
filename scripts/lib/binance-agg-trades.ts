import { createHash } from "node:crypto";
import fs from "node:fs/promises";
import path from "node:path";
import type { TradeFlowAggregate } from "@trading/storage";

const DAY_MS = 86_400_000;
const DATE_PATTERN = /^\d{4}-\d{2}-\d{2}$/;
const EXPECTED_HEADERS = new Set([
  "agg_trade_id,price,quantity,first_trade_id,last_trade_id,transact_time,is_buyer_maker,is_best_match",
  "aggregate_trade_id,price,quantity,first_trade_id,last_trade_id,timestamp,was_the_buyer_the_maker,was_the_trade_the_best_price_match",
]);

export interface OracleScopedDates {
  dates: string[];
  targetDates: Set<string>;
  contextDates: Set<string>;
  sealedTestStart: string;
  sealedTestEnd: string;
  targetReferenceCount: number;
}

export interface OracleCorpusSplitContract {
  schemaVersion: 1;
  targetContract: string;
  referenceCount: number;
  referenceFilenameSha256: string;
  filenameHashEncoding: "utf8-lf-with-trailing-lf";
  train: { count: number; first: string; last: string };
  validation: { count: number; first: string; last: string };
  test: {
    count: number;
    first: string;
    last: string;
    policy: "sealed-never-load";
  };
}

export interface ParsedBinanceAggregateTrade {
  aggregateTradeId: bigint;
  firstTradeId: bigint;
  lastTradeId: bigint;
  timestampUnit: "millisecond" | "microsecond";
  bestPriceMatch: boolean;
  aggregate: TradeFlowAggregate;
}

export type ParsedBinanceAggregateTradeRow =
  | { kind: "header"; columns: string[] }
  | { kind: "invalid-sentinel" }
  | { kind: "trade"; value: ParsedBinanceAggregateTrade };

export async function oracleScopedTradeFlowDates(
  targetReferenceDirectory: string,
  requestedDates?: readonly string[],
  splitContract?: OracleCorpusSplitContract,
): Promise<OracleScopedDates> {
  if (!splitContract) throw new Error("An immutable oracle split contract is required.");
  validateSplitContract(splitContract);
  if (path.basename(path.resolve(targetReferenceDirectory)) !== splitContract.targetContract) {
    throw new Error("Oracle target directory differs from the immutable split contract.");
  }
  const referenceNames = (await fs.readdir(targetReferenceDirectory, { withFileTypes: true }))
    .filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
    .sort((left, right) => left.name.localeCompare(right.name));
  if (referenceNames.some((entry) => !DATE_PATTERN.test(entry.name.slice(0, -5)))) {
    throw new Error("Oracle target directory contains a non-date JSON reference.");
  }
  const targetDates = referenceNames.map((entry) => entry.name.slice(0, -5));
  const actualHash = createHash("sha256")
    .update(`${targetDates.map((date) => `${date}.json`).join("\n")}\n`)
    .digest("hex");
  if (targetDates.length !== splitContract.referenceCount
    || actualHash !== splitContract.referenceFilenameSha256) {
    throw new Error("Oracle target filenames differ from the immutable split contract.");
  }
  targetDates.forEach(parseUtcDay);
  const trainEnd = splitContract.train.count;
  const validationEnd = trainEnd + splitContract.validation.count;
  const sealedDates = targetDates.slice(validationEnd);
  const eligibleTargets = targetDates.slice(0, validationEnd);
  if (targetDates[0] !== splitContract.train.first
    || targetDates[trainEnd - 1] !== splitContract.train.last
    || targetDates[trainEnd] !== splitContract.validation.first
    || targetDates[validationEnd - 1] !== splitContract.validation.last
    || sealedDates[0] !== splitContract.test.first
    || sealedDates.at(-1) !== splitContract.test.last) {
    throw new Error("Oracle split boundaries differ from the immutable contract.");
  }
  const eligibleSet = new Set(eligibleTargets);
  const allAllowed = new Set(eligibleTargets);
  const contextDates = new Set<string>();
  for (const date of eligibleTargets) {
    const prior = formatUtcDay(parseUtcDay(date) - DAY_MS);
    allAllowed.add(prior);
    if (!eligibleSet.has(prior)) contextDates.add(prior);
  }
  const dates = requestedDates === undefined
    ? [...allAllowed].sort()
    : [...new Set(requestedDates.map((date) => {
        parseUtcDay(date);
        return date;
      }))].sort();
  for (const date of dates) {
    if (!allAllowed.has(date)) {
      const sealed = date >= sealedDates[0]! && date <= sealedDates.at(-1)!;
      throw new Error(
        sealed
          ? `${date} is in the sealed oracle test split and cannot be ingested.`
          : `${date} is not a train/validation target or required predecessor context day.`,
      );
    }
  }
  return {
    dates,
    targetDates: eligibleSet,
    contextDates,
    sealedTestStart: sealedDates[0]!,
    sealedTestEnd: sealedDates.at(-1)!,
    targetReferenceCount: targetDates.length,
  };
}

export async function readOracleCorpusSplitContract(
  file: string,
): Promise<OracleCorpusSplitContract> {
  const value = JSON.parse(await fs.readFile(file, "utf8")) as OracleCorpusSplitContract;
  validateSplitContract(value);
  return value;
}

export function parseBinanceAggregateTradeCsvRow(
  line: string,
): ParsedBinanceAggregateTradeRow {
  const columns = line.replace(/^\ufeff/, "").split(",").map((value) => value.trim());
  if (columns.length !== 8) throw new Error(`Expected 8 aggTrade CSV columns, found ${columns.length}.`);
  if (!/^-?\d+$/.test(columns[0]!)) {
    const normalized = columns.map((column) => column.toLowerCase().replaceAll(" ", "_"));
    if (!EXPECTED_HEADERS.has(normalized.join(","))) {
      throw new Error("aggTrade CSV header differs from the ordered Binance schema.");
    }
    return { kind: "header", columns };
  }
  const aggregateTradeId = integer(columns[0]!, "aggregate trade ID");
  const price = Number(columns[1]);
  const quantity = Number(columns[2]);
  const firstTradeId = integer(columns[3]!, "first trade ID");
  const lastTradeId = integer(columns[4]!, "last trade ID");
  if (price === 0 && quantity === 0 && firstTradeId === -1n && lastTradeId === -1n) {
    return { kind: "invalid-sentinel" };
  }
  if (aggregateTradeId < 0n
    || !Number.isFinite(price) || price <= 0
    || !Number.isFinite(quantity) || quantity <= 0
    || firstTradeId < 0n || lastTradeId < firstTradeId) {
    throw new Error("Invalid aggTrade identifiers, price, or quantity.");
  }
  const rawTimestamp = integer(columns[5]!, "timestamp");
  if (rawTimestamp < 0n) throw new Error("Invalid aggTrade timestamp.");
  const timestampUnit = rawTimestamp >= 1_000_000_000_000_000n
    ? "microsecond"
    : "millisecond";
  const timeMicrosBig = timestampUnit === "microsecond"
    ? rawTimestamp
    : rawTimestamp * 1_000n;
  const timeMicros = Number(timeMicrosBig);
  const rawTradeCount = lastTradeId - firstTradeId + 1n;
  if (!Number.isSafeInteger(timeMicros)
    || rawTradeCount > 0xffff_ffffn) {
    throw new Error("aggTrade timestamp or constituent trade count exceeds the storage schema.");
  }
  const buyerMaker = bool(columns[6]!, "buyer-maker flag");
  return {
    kind: "trade",
    value: {
      aggregateTradeId,
      firstTradeId,
      lastTradeId,
      timestampUnit,
      bestPriceMatch: bool(columns[7]!, "best-price-match flag"),
      aggregate: {
        timeMicros,
        price,
        quantity,
        tradeCount: Number(rawTradeCount),
        aggressorSide: buyerMaker ? -1 : 1,
      },
    },
  };
}

export function parseDateList(input: string): string[] {
  const dates = input.split(",").filter(Boolean);
  dates.forEach(parseUtcDay);
  return dates;
}

function integer(value: string, label: string): bigint {
  if (!/^-?\d+$/.test(value)) throw new Error(`Invalid ${label}.`);
  return BigInt(value);
}

function bool(value: string, label: string): boolean {
  if (/^true$/i.test(value)) return true;
  if (/^false$/i.test(value)) return false;
  throw new Error(`Invalid ${label}.`);
}

function parseUtcDay(date: string): number {
  if (!DATE_PATTERN.test(date)) throw new Error(`Invalid UTC date: ${date}.`);
  const timestamp = Date.parse(`${date}T00:00:00.000Z`);
  if (!Number.isFinite(timestamp) || formatUtcDay(timestamp) !== date) {
    throw new Error(`Invalid UTC date: ${date}.`);
  }
  return timestamp;
}

function formatUtcDay(timestamp: number): string {
  return new Date(timestamp).toISOString().slice(0, 10);
}

export function targetContractId(targetReferenceDirectory: string): string {
  return path.basename(path.resolve(targetReferenceDirectory));
}

function validateSplitContract(value: OracleCorpusSplitContract): void {
  const splits = [value?.train, value?.validation, value?.test];
  if (value?.schemaVersion !== 1
    || !value.targetContract
    || !Number.isSafeInteger(value.referenceCount)
    || value.referenceCount < 1
    || !/^[a-f0-9]{64}$/.test(value.referenceFilenameSha256)
    || value.filenameHashEncoding !== "utf8-lf-with-trailing-lf"
    || splits.some((split) => !split
      || !Number.isSafeInteger(split.count) || split.count < 1
      || !DATE_PATTERN.test(split.first) || !DATE_PATTERN.test(split.last))
    || splits.reduce((total, split) => total + split.count, 0) !== value.referenceCount
    || value.test.policy !== "sealed-never-load") {
    throw new Error("Invalid immutable oracle split contract.");
  }
  for (const split of splits) {
    parseUtcDay(split.first);
    parseUtcDay(split.last);
  }
}

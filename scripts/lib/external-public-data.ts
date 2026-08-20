export interface DvolRow {
  time: number;
  availableAt: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface VixRow {
  time: number;
  availableAt: number;
  close: number;
}

export type MacroFrequency = "daily" | "event" | "monthly" | "quarterly";

export interface MacroSeriesDefinition {
  id: string;
  label: string;
  economy: string;
  frequency: MacroFrequency;
  availabilityLagDays: number;
  provider: string;
  sourceUrl: string;
}

export interface MacroRow extends MacroSeriesDefinition {
  time: number;
  availableAt: number;
  value: number;
}

export interface BinanceFundingRateRow {
  time: number;
  availableAt: number;
  symbol: string;
  fundingRate: number;
  markPrice: number | null;
}

export interface CoinMetricsRow {
  time: number;
  availableAt: number;
  latestRevisionAt: number | null;
  values: Record<string, number | null>;
  statuses: Record<string, string>;
}

export interface MempoolMiningRow {
  time: number;
  availableAt: number;
  avgHeight: number;
  values: Record<string, number>;
}

export interface CommunityCryptoMetricPayload {
  name?: string;
  description?: string;
  data?: Array<{ timestamp?: number; value?: number; last_modified?: number }>;
}

export interface CommunityCryptoDailyRow {
  time: number;
  availableAt: number;
  latestRevisionAt: number | null;
  values: Record<string, number>;
}

export interface DeribitOptionSummary {
  observedAt: number;
  underlyingPrice: number | null;
  instrumentCount: number;
  quotedInstrumentCount: number;
  totalCallOpenInterest: number;
  totalPutOpenInterest: number;
  callPutOpenInterestImbalance: number | null;
  nextExpiryAt: number | null;
  hoursToNextExpiry: number | null;
  nearestMajorStrike: number | null;
  logDistanceToNearestMajorStrike: number | null;
  majorStrikes: Array<{ strike: number; openInterest: number }>;
  expiries: DeribitExpirySummary[];
  term: Record<string, number | null>;
}

export interface DeribitExpirySummary {
  expiryAt: number;
  daysToExpiry: number;
  underlyingPrice: number;
  atmIv: number | null;
  put25Iv: number | null;
  call25Iv: number | null;
  putCall25Skew: number | null;
  callOpenInterest: number;
  putOpenInterest: number;
  callPutOpenInterestImbalance: number | null;
  nearestAtmLogMoneyness: number | null;
  put25DeltaError: number | null;
  call25DeltaError: number | null;
}

export interface DeribitBookSummary {
  instrument_name: string;
  open_interest?: number | null;
  mark_iv?: number | null;
  underlying_price?: number | null;
  bid_price?: number | null;
  ask_price?: number | null;
}

const HOUR_MS = 3_600_000;
const YEAR_MS = 365.25 * 86_400_000;
const MONTHS = new Map([
  ["JAN", 0], ["FEB", 1], ["MAR", 2], ["APR", 3], ["MAY", 4], ["JUN", 5],
  ["JUL", 6], ["AUG", 7], ["SEP", 8], ["OCT", 9], ["NOV", 10], ["DEC", 11],
]);

export function normalizeDvolRows(data: unknown[]): DvolRow[] {
  const rows: DvolRow[] = [];
  for (const item of data) {
    if (!Array.isArray(item) || item.length < 5) continue;
    const values = item.slice(0, 5).map(Number);
    if (!values.every(Number.isFinite)) continue;
    const [time, open, high, low, close] = values as [number, number, number, number, number];
    rows.push({ time, availableAt: time + HOUR_MS, open, high, low, close });
  }
  return deduplicate(rows, (row) => row.time);
}

export function normalizeFredVixCsv(csv: string): VixRow[] {
  const rows: VixRow[] = [];
  for (const line of csv.replace(/^\uFEFF/, "").split(/\r?\n/).slice(1)) {
    const [date, rawClose] = line.split(",");
    if (!date || rawClose === undefined) continue;
    const time = Date.parse(`${date.trim()}T00:00:00.000Z`);
    const close = Number(rawClose.trim());
    if (!Number.isFinite(time) || !Number.isFinite(close) || close <= 0) continue;
    // The daily close is known after the US session. The following UTC
    // boundary is a conservative, DST-independent point-in-time timestamp.
    rows.push({ time, availableAt: time + 86_400_000, close });
  }
  return deduplicate(rows, (row) => row.time);
}

export function normalizeMacroCsv(
  csv: string,
  definition: MacroSeriesDefinition,
  columns: { period?: string; value?: string; filter?: Record<string, string> } = {},
): MacroRow[] {
  const records = parseCsvRecords(csv);
  if (records.length === 0) return [];
  const header = records[0]!;
  const periodIndex = header.indexOf(columns.period ?? "observation_date");
  const valueIndex = header.indexOf(columns.value ?? definition.id);
  if (periodIndex < 0 || valueIndex < 0) return [];
  const filters = Object.entries(columns.filter ?? {}).map(([name, expected]) => [header.indexOf(name), expected] as const);
  const rows: MacroRow[] = [];
  for (const record of records.slice(1)) {
    if (filters.some(([index, expected]) => index < 0 || record[index] !== expected)) continue;
    const rawValue = record[valueIndex];
    if (!rawValue || rawValue.trim() === "." || rawValue.trim() === "..") continue;
    const time = parseMacroPeriod(record[periodIndex] ?? "");
    const value = Number(rawValue.trim().replace(",", "."));
    if (!Number.isFinite(time) || !Number.isFinite(value)) continue;
    rows.push({
      ...definition,
      time,
      availableAt: time + definition.availabilityLagDays * 86_400_000,
      value,
    });
  }
  return deduplicate(rows, (row) => row.time);
}

export function normalizeMacroHtmlTable(
  html: string,
  definition: MacroSeriesDefinition,
  columns: { period: number; value: number },
): MacroRow[] {
  const rows: MacroRow[] = [];
  for (const match of html.matchAll(/<tr\b[^>]*>([\s\S]*?)<\/tr>/gi)) {
    const cells = [...match[1]!.matchAll(/<t[dh]\b[^>]*>([\s\S]*?)<\/t[dh]>/gi)]
      .map((cell) => decodeHtml(cell[1]!.replace(/<[^>]+>/g, "")).trim());
    const rawPeriod = cells[columns.period];
    const rawValue = cells[columns.value];
    if (!rawPeriod || !rawValue) continue;
    const time = parseMacroPeriod(rawPeriod);
    const value = Number(rawValue.replace(/\s/g, "").replace(",", "."));
    if (!Number.isFinite(time) || !Number.isFinite(value)) continue;
    rows.push({
      ...definition,
      time,
      availableAt: time + definition.availabilityLagDays * 86_400_000,
      value,
    });
  }
  return deduplicate(rows, (row) => row.time);
}

export function normalizeBinanceFundingRows(payload: unknown[]): BinanceFundingRateRow[] {
  const rows: BinanceFundingRateRow[] = [];
  for (const raw of payload) {
    if (!raw || typeof raw !== "object") continue;
    const record = raw as Record<string, unknown>;
    const time = Number(record.fundingTime);
    const fundingRate = Number(record.fundingRate);
    const rawMarkPrice = record.markPrice;
    const markPrice = rawMarkPrice === "" || rawMarkPrice == null ? null : Number(rawMarkPrice);
    const symbol = String(record.symbol ?? "");
    if (!symbol || !Number.isFinite(time) || !Number.isFinite(fundingRate)) continue;
    rows.push({
      time,
      // Funding is usable only after settlement; one minute also prevents the
      // settlement instant from sharing a target candle with the feature.
      availableAt: time + 60_000,
      symbol,
      fundingRate,
      markPrice: markPrice !== null && Number.isFinite(markPrice) ? markPrice : null,
    });
  }
  return deduplicate(rows, (row) => row.time);
}

export function normalizeCoinMetricsRows(
  data: Array<Record<string, unknown>>,
  metrics: string[],
): CoinMetricsRow[] {
  return data.flatMap((raw) => {
    const time = Date.parse(String(raw.time ?? ""));
    if (!Number.isFinite(time)) return [];
    const values: Record<string, number | null> = {};
    const statuses: Record<string, string> = {};
    const statusTimes: number[] = [];
    for (const metric of metrics) {
      const value = Number(raw[metric]);
      values[metric] = Number.isFinite(value) ? value : null;
      const status = raw[`${metric}-status`];
      if (typeof status === "string") statuses[metric] = status;
      const statusTime = Date.parse(String(raw[`${metric}-status-time`] ?? ""));
      if (Number.isFinite(statusTime)) statusTimes.push(statusTime);
    }
    // Community daily metrics are normally published after the UTC day closes.
    // Coin Metrics' per-row status-time is the latest retrospective revision,
    // not the first historical publication time. Preserve it for the PIT-risk
    // audit, but do not pretend the value was unavailable until that revision.
    // The next-day boundary is therefore an explicitly retrospective proxy.
    const nextDay = time + 86_400_000;
    const latestRevisionAt = statusTimes.length > 0 ? Math.max(...statusTimes) : null;
    return [{ time, availableAt: nextDay, latestRevisionAt, values, statuses }];
  }).sort((left, right) => left.time - right.time);
}

export function mergeMempoolMiningSeries(
  series: Record<string, unknown>,
): MempoolMiningRow[] {
  const merged = new Map<string, MempoolMiningRow>();
  const arrays: Array<[string, unknown[]]> = [];
  for (const [family, payload] of Object.entries(series)) {
    if (Array.isArray(payload)) arrays.push([family, payload]);
    else if (payload && typeof payload === "object") {
      for (const [subFamily, value] of Object.entries(payload as Record<string, unknown>)) {
        if (Array.isArray(value)) arrays.push([`${family}.${subFamily}`, value]);
      }
    }
  }
  for (const [family, rows] of arrays) for (const raw of rows) {
    if (!raw || typeof raw !== "object") continue;
    const record = raw as Record<string, unknown>;
    const timestampSeconds = Number(record.timestamp);
    const avgHeight = Number(record.avgHeight);
    if (!Number.isFinite(timestampSeconds) || !Number.isFinite(avgHeight)) continue;
    const time = timestampSeconds * 1_000;
    const key = `${time}:${avgHeight}`;
    const row = merged.get(key) ?? { time, availableAt: time, avgHeight, values: {} };
    for (const [name, value] of Object.entries(record)) {
      if (name === "timestamp" || name === "avgHeight") continue;
      const numeric = Number(value);
      if (Number.isFinite(numeric)) row.values[`${family}.${name}`] = numeric;
    }
    merged.set(key, row);
  }
  const result = [...merged.values()].sort((left, right) => left.time - right.time);
  const gaps = result.slice(1).map((row, index) => row.time - result[index]!.time)
    .filter((gap) => gap > 0).sort((left, right) => left - right);
  const medianGap = gaps.length > 0 ? gaps[Math.floor(gaps.length / 2)]! : 12 * HOUR_MS;
  // These are trailing aggregates. Delay them by one observed bucket because
  // the API does not expose the exact historical publication timestamp.
  for (const row of result) row.availableAt = row.time + medianGap;
  return result;
}

export function normalizeCommunityCryptoDaily(
  payloads: Record<string, CommunityCryptoMetricPayload>,
): CommunityCryptoDailyRow[] {
  const rows = new Map<number, CommunityCryptoDailyRow>();
  for (const [metric, payload] of Object.entries(payloads)) for (const point of payload.data ?? []) {
    const time = Number(point.timestamp);
    const value = Number(point.value);
    const revision = Number(point.last_modified);
    if (!Number.isFinite(time) || !Number.isFinite(value)) continue;
    const row = rows.get(time) ?? {
      time,
      availableAt: time + 86_400_000,
      latestRevisionAt: null,
      values: {},
    };
    row.values[metric] = value;
    if (Number.isFinite(revision)) row.latestRevisionAt = Math.max(row.latestRevisionAt ?? 0, revision);
    rows.set(time, row);
  }
  return [...rows.values()].sort((left, right) => left.time - right.time);
}

export function deriveDeribitOptionSummary(
  books: DeribitBookSummary[],
  observedAt: number,
): DeribitOptionSummary {
  const parsed = books.flatMap((book) => {
    const instrument = parseOptionInstrument(book.instrument_name);
    const iv = finiteOrNull(book.mark_iv);
    const underlying = finiteOrNull(book.underlying_price);
    const openInterest = Math.max(0, finiteOrNull(book.open_interest) ?? 0);
    if (!instrument || iv === null || iv <= 0 || underlying === null || underlying <= 0) return [];
    const years = (instrument.expiryAt - observedAt) / YEAR_MS;
    if (years <= 0) return [];
    const sigma = iv / 100;
    const standardDeviation = sigma * Math.sqrt(years);
    const d1 = Math.log(underlying / instrument.strike) / standardDeviation + standardDeviation / 2;
    const delta = instrument.kind === "C" ? normalCdf(d1) : normalCdf(d1) - 1;
    return [{ ...instrument, iv, underlying, openInterest, delta }];
  });
  const expiries = new Map<number, typeof parsed>();
  for (const option of parsed) {
    const rows = expiries.get(option.expiryAt) ?? [];
    rows.push(option);
    expiries.set(option.expiryAt, rows);
  }
  const expirySummaries = [...expiries.entries()].map(([expiryAt, options]) => {
    const underlying = median(options.map((option) => option.underlying));
    const atm = minimumBy(options, (option) => Math.abs(Math.log(option.strike / underlying)));
    const put25 = minimumBy(options.filter((option) => option.kind === "P"), (option) => Math.abs(option.delta + 0.25));
    const call25 = minimumBy(options.filter((option) => option.kind === "C"), (option) => Math.abs(option.delta - 0.25));
    const callOpenInterest = sum(options.filter((option) => option.kind === "C").map((option) => option.openInterest));
    const putOpenInterest = sum(options.filter((option) => option.kind === "P").map((option) => option.openInterest));
    return {
      expiryAt,
      daysToExpiry: (expiryAt - observedAt) / 86_400_000,
      underlyingPrice: underlying,
      atmIv: atm?.iv ?? null,
      put25Iv: put25?.iv ?? null,
      call25Iv: call25?.iv ?? null,
      putCall25Skew: put25 && call25 ? put25.iv - call25.iv : null,
      callOpenInterest,
      putOpenInterest,
      callPutOpenInterestImbalance: symmetricImbalance(callOpenInterest, putOpenInterest),
      nearestAtmLogMoneyness: atm ? Math.log(atm.strike / underlying) : null,
      put25DeltaError: put25 ? Math.abs(put25.delta + 0.25) : null,
      call25DeltaError: call25 ? Math.abs(call25.delta - 0.25) : null,
    } satisfies DeribitExpirySummary;
  }).sort((left, right) => left.expiryAt - right.expiryAt);
  const callOpenInterest = sum(parsed.filter((option) => option.kind === "C").map((option) => option.openInterest));
  const putOpenInterest = sum(parsed.filter((option) => option.kind === "P").map((option) => option.openInterest));
  const strikeInterest = new Map<number, number>();
  for (const option of parsed) strikeInterest.set(
    option.strike,
    (strikeInterest.get(option.strike) ?? 0) + option.openInterest,
  );
  const majorStrikes = [...strikeInterest].map(([strike, openInterest]) => ({ strike, openInterest }))
    .sort((left, right) => right.openInterest - left.openInterest).slice(0, 10);
  const underlyingPrice = parsed.length > 0 ? median(parsed.map((option) => option.underlying)) : null;
  const nearestMajorStrike = underlyingPrice === null
    ? null
    : minimumBy(majorStrikes, (item) => Math.abs(Math.log(item.strike / underlyingPrice)))?.strike ?? null;
  const term: Record<string, number | null> = {};
  for (const days of [1, 7, 30]) {
    term[`atmIv${days}d`] = interpolateTerm(expirySummaries, observedAt, days, "atmIv", true);
    term[`putCall25Skew${days}d`] = interpolateTerm(expirySummaries, observedAt, days, "putCall25Skew", false);
  }
  term.atmIvSlope7dMinus1d = difference(term.atmIv7d, term.atmIv1d);
  term.atmIvSlope30dMinus7d = difference(term.atmIv30d, term.atmIv7d);
  term.skewSlope7dMinus1d = difference(term.putCall25Skew7d, term.putCall25Skew1d);
  term.skewSlope30dMinus7d = difference(term.putCall25Skew30d, term.putCall25Skew7d);
  const nextExpiryAt = expirySummaries[0]?.expiryAt ?? null;
  return {
    observedAt,
    underlyingPrice,
    instrumentCount: books.length,
    quotedInstrumentCount: parsed.length,
    totalCallOpenInterest: callOpenInterest,
    totalPutOpenInterest: putOpenInterest,
    callPutOpenInterestImbalance: symmetricImbalance(callOpenInterest, putOpenInterest),
    nextExpiryAt,
    hoursToNextExpiry: nextExpiryAt === null ? null : (nextExpiryAt - observedAt) / HOUR_MS,
    nearestMajorStrike,
    logDistanceToNearestMajorStrike: underlyingPrice !== null && nearestMajorStrike !== null
      ? Math.log(underlyingPrice / nearestMajorStrike)
      : null,
    majorStrikes,
    expiries: expirySummaries,
    term,
  };
}

function parseOptionInstrument(name: string) {
  const match = /^BTC-(\d{1,2})([A-Z]{3})(\d{2})-(\d+(?:\.\d+)?)-([CP])$/.exec(name);
  if (!match) return null;
  const month = MONTHS.get(match[2]!);
  if (month === undefined) return null;
  const expiryAt = Date.UTC(2_000 + Number(match[3]), month, Number(match[1]), 8);
  const strike = Number(match[4]);
  if (!Number.isFinite(expiryAt) || !Number.isFinite(strike) || strike <= 0) return null;
  return { expiryAt, strike, kind: match[5] as "C" | "P" };
}

function interpolateTerm(
  rows: DeribitExpirySummary[],
  observedAt: number,
  targetDays: number,
  field: "atmIv" | "putCall25Skew",
  totalVariance: boolean,
): number | null {
  const valid = rows.filter((row) => row[field] !== null);
  if (valid.length === 0) return null;
  const targetYears = targetDays / 365.25;
  const points = valid.map((row) => ({
    years: (row.expiryAt - observedAt) / YEAR_MS,
    value: row[field]!,
  })).filter((point) => point.years > 0);
  if (points.length === 0) return null;
  if (targetYears <= points[0]!.years) return points[0]!.value;
  if (targetYears >= points.at(-1)!.years) return points.at(-1)!.value;
  const upperIndex = points.findIndex((point) => point.years >= targetYears);
  const lower = points[upperIndex - 1]!;
  const upper = points[upperIndex]!;
  const weight = (targetYears - lower.years) / (upper.years - lower.years);
  if (!totalVariance) return lower.value + weight * (upper.value - lower.value);
  const lowerVariance = (lower.value / 100) ** 2 * lower.years;
  const upperVariance = (upper.value / 100) ** 2 * upper.years;
  const targetVariance = lowerVariance + weight * (upperVariance - lowerVariance);
  return Math.sqrt(Math.max(0, targetVariance) / targetYears) * 100;
}

function normalCdf(value: number): number {
  const sign = value < 0 ? -1 : 1;
  const x = Math.abs(value) / Math.sqrt(2);
  const t = 1 / (1 + 0.3275911 * x);
  const polynomial = (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t
    - 0.284496736) * t + 0.254829592) * t;
  const erf = sign * (1 - polynomial * Math.exp(-x * x));
  return (1 + erf) / 2;
}

function finiteOrNull(value: unknown): number | null {
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

function symmetricImbalance(left: number, right: number): number | null {
  const total = left + right;
  return total > 0 ? (left - right) / total : null;
}

function minimumBy<T>(items: T[], score: (item: T) => number): T | undefined {
  let best: T | undefined;
  let bestScore = Number.POSITIVE_INFINITY;
  for (const item of items) {
    const value = score(item);
    if (value < bestScore) {
      best = item;
      bestScore = value;
    }
  }
  return best;
}

function median(values: number[]): number {
  const sorted = [...values].sort((left, right) => left - right);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function sum(values: number[]): number {
  return values.reduce((total, value) => total + value, 0);
}

function difference(left: number | null | undefined, right: number | null | undefined): number | null {
  return left == null || right == null ? null : left - right;
}

function parseCsvRecords(csv: string): string[][] {
  const records: string[][] = [];
  let record: string[] = [];
  let field = "";
  let quoted = false;
  const input = csv.replace(/^\uFEFF/, "");
  for (let index = 0; index < input.length; index += 1) {
    const character = input[index]!;
    if (character === '"') {
      if (quoted && input[index + 1] === '"') {
        field += '"';
        index += 1;
      } else quoted = !quoted;
    } else if (character === "," && !quoted) {
      record.push(field);
      field = "";
    } else if ((character === "\n" || character === "\r") && !quoted) {
      if (character === "\r" && input[index + 1] === "\n") index += 1;
      record.push(field);
      if (record.some((value) => value.length > 0)) records.push(record);
      record = [];
      field = "";
    } else field += character;
  }
  if (field.length > 0 || record.length > 0) {
    record.push(field);
    if (record.some((value) => value.length > 0)) records.push(record);
  }
  return records;
}

function parseMacroPeriod(raw: string): number {
  const value = raw.trim();
  let match = /^(\d{4})-Q([1-4])$/.exec(value);
  if (match) return Date.UTC(Number(match[1]), (Number(match[2]) - 1) * 3, 1);
  match = /^(\d{4})-(\d{2})$/.exec(value);
  if (match) return Date.UTC(Number(match[1]), Number(match[2]) - 1, 1);
  match = /^(\d{2})\.(\d{2})\.(\d{4})$/.exec(value);
  if (match) return Date.UTC(Number(match[3]), Number(match[2]) - 1, Number(match[1]));
  match = /^(\d{2})\.(\d{4})$/.exec(value);
  if (match) return Date.UTC(Number(match[2]), Number(match[1]) - 1, 1);
  match = /^(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2}|\d{4})$/.exec(value);
  if (match) {
    const month = MONTHS.get(match[2]!.toUpperCase());
    const rawYear = Number(match[3]);
    const year = rawYear < 100 ? 2_000 + rawYear : rawYear;
    if (month !== undefined) return Date.UTC(year, month, Number(match[1]));
  }
  const parsed = Date.parse(`${value}T00:00:00.000Z`);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function decodeHtml(value: string): string {
  return value
    .replace(/&nbsp;|&#160;/gi, " ")
    .replace(/&amp;/gi, "&")
    .replace(/&lt;/gi, "<")
    .replace(/&gt;/gi, ">");
}

function deduplicate<T>(rows: T[], key: (row: T) => number): T[] {
  return [...new Map(rows.map((row) => [key(row), row])).values()]
    .sort((left, right) => key(left) - key(right));
}

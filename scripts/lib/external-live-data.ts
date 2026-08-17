export interface GdeltNgramSummary {
  batch: string | null;
  englishTokenCount: number;
  termCounts: Record<string, number>;
  cryptoTermCount: number;
  cryptoTermsPerMillion: number | null;
}

export interface GdeltGkgSummary {
  batch: string | null;
  storyCount: number;
  sourceCount: number;
  meanTone: number | null;
  meanPositive: number | null;
  meanNegative: number | null;
  meanPolarity: number | null;
  matches: Array<{
    recordId: string;
    date: string;
    source: string;
    url: string;
    themes: string[];
    tone: number | null;
  }>;
}

const CRYPTO_TERMS = new Set([
  "bitcoin", "btc", "cryptocurrency", "cryptocurrencies", "crypto",
  "blockchain", "ethereum", "ether", "stablecoin", "stablecoins",
]);
const CRYPTO_PATTERN = /(?:bitcoin|cryptocurrenc|blockchain|digital[_ -]?asset|stablecoin|(?:^|[^a-z])btc(?:[^a-z]|$))/i;

export function summarizeGdeltNgrams(text: string): GdeltNgramSummary {
  let batch: string | null = null;
  let englishTokenCount = 0;
  const termCounts: Record<string, number> = {};
  for (const line of text.split(/\r?\n/)) {
    if (!line) continue;
    const fields = line.split("\t");
    if (fields.length < 4) continue;
    batch ??= fields[0] ?? null;
    if (fields[1]?.toUpperCase() !== "ENGLISH") continue;
    const term = fields[2]!.trim().toLowerCase();
    const count = Number(fields[3]);
    if (!Number.isFinite(count) || count < 0) continue;
    englishTokenCount += count;
    if (CRYPTO_TERMS.has(term)) termCounts[term] = (termCounts[term] ?? 0) + count;
  }
  const cryptoTermCount = Object.values(termCounts).reduce((total, count) => total + count, 0);
  return {
    batch,
    englishTokenCount,
    termCounts,
    cryptoTermCount,
    cryptoTermsPerMillion: englishTokenCount > 0 ? cryptoTermCount * 1_000_000 / englishTokenCount : null,
  };
}

export function summarizeGdeltGkg(text: string): GdeltGkgSummary {
  let batch: string | null = null;
  const matches: GdeltGkgSummary["matches"] = [];
  const sources = new Set<string>();
  const tones: number[][] = [];
  for (const line of text.split(/\r?\n/)) {
    if (!line) continue;
    const fields = line.split("\t");
    if (fields.length < 16) continue;
    batch ??= fields[1] ?? null;
    const source = fields[3] ?? "";
    const url = fields[4] ?? "";
    const themeText = `${fields[7] ?? ""};${fields[8] ?? ""}`;
    const organizationText = `${fields[13] ?? ""};${fields[14] ?? ""}`;
    if (!CRYPTO_PATTERN.test(`${url};${themeText};${organizationText}`)) continue;
    const tone = (fields[15] ?? "").split(",").slice(0, 4).map(Number);
    if (tone.length === 4 && tone.every(Number.isFinite)) tones.push(tone);
    sources.add(source);
    matches.push({
      recordId: fields[0] ?? "",
      date: fields[1] ?? "",
      source,
      url,
      themes: themeText.split(";").filter((item) => CRYPTO_PATTERN.test(item)).slice(0, 20),
      tone: tone.length > 0 && Number.isFinite(tone[0]) ? tone[0]! : null,
    });
  }
  return {
    batch,
    storyCount: matches.length,
    sourceCount: sources.size,
    meanTone: columnMean(tones, 0),
    meanPositive: columnMean(tones, 1),
    meanNegative: columnMean(tones, 2),
    meanPolarity: columnMean(tones, 3),
    matches,
  };
}

function columnMean(rows: number[][], column: number): number | null {
  return rows.length > 0
    ? rows.reduce((total, row) => total + row[column]!, 0) / rows.length
    : null;
}

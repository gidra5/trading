/** Experimental lexical JSONL block codec. Every input byte, including raw wire
 * strings, decimal spelling, key order, whitespace and unsafe integer lexemes,
 * is reconstructed. Numeric storage uses lexical strings and BigInt, never
 * Number; JSON.parse is used only for keys/strings and syntax validation. */
type Piece = readonly ["l", string] | readonly ["v", number] | readonly ["j", readonly Piece[]];
type DecimalStyle = readonly [boolean, number, number, boolean];
type ColumnHeader = Readonly<{ path: string; mode: "dictionary"; dictionary: readonly string[]; count: number; bytes: number }>
  | Readonly<{ path: string; mode: "decimal"; styles: readonly DecimalStyle[]; count: number; bytes: number }>;
interface Header { readonly version: 1; readonly rows: number; readonly rowBytes: number; readonly templates: readonly (readonly Piece[])[]; readonly columns: readonly ColumnHeader[]; }
const MAGIC = Buffer.from("LJLC1\n");
function uint(value: bigint, output: number[]) {
  if (value < 0n) throw new Error("Negative unsigned varint");
  while (value >= 128n) { output.push(Number(value & 127n) | 128); value >>= 7n; }
  output.push(Number(value));
}
function readUint(buffer: Buffer, cursor: { value: number }): bigint {
  let output = 0n, shift = 0n;
  for (;;) {
    if (cursor.value >= buffer.length || shift > 4096n) throw new Error("Invalid varint");
    const value = buffer[cursor.value++]!; output |= BigInt(value & 127) << shift;
    if (!(value & 128)) return output;
    shift += 7n;
  }
}
const zig = (value: bigint) => value >= 0n ? value * 2n : -value * 2n - 1n;
const unzig = (value: bigint) => value & 1n ? -(value + 1n) / 2n : value / 2n;
function decimal(value: string): Readonly<{ coefficient: bigint; style: DecimalStyle }> | null {
  if (value.length > 80) return null;
  const match = /^("?)(-?)(\d+)(?:\.(\d+))?\1$/.exec(value);
  if (!match) return null;
  const coefficient = BigInt(`${match[2]}${match[3]}${match[4] ?? ""}`);
  return { coefficient, style: [!!match[1], match[4] === undefined ? -1 : match[4].length,
    match[3]!.length - BigInt(match[3]!).toString().length, match[2] === "-" && coefficient === 0n] };
}
function restoreDecimal(coefficient: bigint, style: DecimalStyle): string {
  const [quoted, places, zeros, negativeZero] = style;
  const absolute = (coefficient < 0n ? -coefficient : coefficient).toString().padStart(Math.max(0, places) + 1, "0");
  const integer = places > 0 ? absolute.slice(0, -places) : absolute;
  const body = `${coefficient < 0n || negativeZero ? "-" : ""}${"0".repeat(zeros)}${integer}${places < 0 ? "" : `.${places ? absolute.slice(-places) : ""}`}`;
  return quoted ? `"${body}"` : body;
}

export function encodeLosslessJsonlBlock(lines: readonly string[]): Buffer {
  const columns: Array<{ path: string; values: string[] }> = [];
  const paths = new Map<string, number>();
  const templates: Array<readonly Piece[]> = [];
  const templateIds = new Map<string, number>();
  const rowIds: number[] = [];
  const variable = (path: readonly string[], value: string): Piece => {
    const key = JSON.stringify(path);
    let index = paths.get(key);
    if (index === undefined) { index = columns.length; paths.set(key, index); columns.push({ path: key, values: [] }); }
    columns[index]!.values.push(value);
    return ["v", index];
  };
  function parse(text: string, prefix: readonly string[]): readonly Piece[] {
    let cursor = 0;
    const pieces: Piece[] = [];
    const literal = (value: string) => {
      const last = pieces.at(-1);
      if (last?.[0] === "l") pieces[pieces.length - 1] = ["l", last[1] + value];
      else if (value) pieces.push(["l", value]);
    };
    const whitespace = () => { const start = cursor; while (/\s/.test(text[cursor] ?? "") && cursor < text.length) cursor++; literal(text.slice(start, cursor)); };
    const string = () => {
      const start = cursor++;
      while (cursor < text.length) {
        const character = text[cursor++];
        if (character === "\\") cursor++;
        else if (character === '"') return text.slice(start, cursor);
      }
      throw new Error("Unterminated JSON string");
    };
    function value(path: readonly string[]) {
      whitespace();
      const character = text[cursor];
      if (character === "{") {
        literal(text[cursor++]!); whitespace();
        if (text[cursor] !== "}") for (;;) {
          if (text[cursor] !== '"') throw new Error("Invalid object key");
          const key = string(); literal(key); whitespace();
          if (text[cursor] !== ":") throw new Error("Missing colon");
          literal(text[cursor++]!); value([...path, JSON.parse(key) as string]); whitespace();
          if (text[cursor] !== ",") break;
          literal(text[cursor++]!); whitespace();
        }
        if (text[cursor] !== "}") throw new Error("Unclosed object");
        literal(text[cursor++]!);
      } else if (character === "[") {
        literal(text[cursor++]!); whitespace(); let index = 0;
        if (text[cursor] !== "]") for (;;) {
          // Price-level arrays retain separate price/quantity columns; arrays of
          // objects or tuples share a wildcard path across repeated rows.
          const child = text[cursor] === "[" || text[cursor] === "{" ? "*" : String(index);
          value([...path, child]); index++; whitespace();
          if (text[cursor] !== ",") break;
          literal(text[cursor++]!); whitespace();
        }
        if (text[cursor] !== "]") throw new Error("Unclosed array");
        literal(text[cursor++]!);
      } else if (character === '"') {
        const token = string();
        if (path.at(-1) === "raw") {
          const decoded: unknown = JSON.parse(token);
          if (typeof decoded === "string" && /^[\s]*[\[{]/.test(decoded) && JSON.stringify(decoded) === token) {
            let valid = true;
            try { JSON.parse(decoded); } catch { valid = false; }
            if (valid) { pieces.push(["j", parse(decoded, [...path, "<wire>"])]); return; }
          }
        }
        pieces.push(variable(path, token));
      } else {
        const match = /^(?:-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)/.exec(text.slice(cursor));
        if (!match) throw new Error("Invalid JSON primitive");
        cursor += match[0].length; pieces.push(variable(path, match[0]));
      }
    }
    value(prefix); whitespace();
    if (cursor !== text.length) throw new Error("Trailing JSON content");
    return pieces;
  }
  for (const line of lines) {
    const template = parse(line, []);
    const identity = JSON.stringify(template);
    let index = templateIds.get(identity);
    if (index === undefined) { index = templates.length; templateIds.set(identity, index); templates.push(template); }
    uint(BigInt(index), rowIds);
  }
  const bodies: Buffer[] = [Buffer.from(rowIds)];
  const headers: ColumnHeader[] = [];
  for (const column of columns) {
    const dictionary: string[] = []; const ids = new Map<string, number>(); const dictionaryBytes: number[] = [];
    for (const value of column.values) {
      let index = ids.get(value);
      if (index === undefined) { index = dictionary.length; ids.set(value, index); dictionary.push(value); }
      uint(BigInt(index), dictionaryBytes);
    }
    const decimals = column.values.map(decimal);
    const decimalBytes: number[] = [], styles: DecimalStyle[] = [], styleIds = new Map<string, number>();
    if (decimals.every((value) => value !== null)) {
      let previous = 0n;
      for (const value of decimals) {
        const key = JSON.stringify(value.style); let style = styleIds.get(key);
        if (style === undefined) { style = styles.length; styleIds.set(key, style); styles.push(value.style); }
        uint(BigInt(style), decimalBytes); uint(zig(value.coefficient - previous), decimalBytes); previous = value.coefficient;
      }
    }
    if (decimalBytes.length && decimalBytes.length + JSON.stringify(styles).length < dictionaryBytes.length + Buffer.byteLength(JSON.stringify(dictionary))) {
      headers.push({ path: column.path, mode: "decimal", styles, count: column.values.length, bytes: decimalBytes.length }); bodies.push(Buffer.from(decimalBytes));
    } else {
      headers.push({ path: column.path, mode: "dictionary", dictionary, count: column.values.length, bytes: dictionaryBytes.length }); bodies.push(Buffer.from(dictionaryBytes));
    }
  }
  const header: Header = { version: 1, rows: lines.length, rowBytes: rowIds.length, templates, columns: headers };
  const encodedHeader = Buffer.from(JSON.stringify(header)); const length = Buffer.alloc(4); length.writeUInt32LE(encodedHeader.length);
  return Buffer.concat([MAGIC, length, encodedHeader, ...bodies]);
}

export function decodeLosslessJsonlBlock(buffer: Buffer): readonly string[] {
  if (!buffer.subarray(0, MAGIC.length).equals(MAGIC)) throw new Error("Unknown lexical block format");
  const headerLength = buffer.readUInt32LE(MAGIC.length);
  const headerStart = MAGIC.length + 4;
  const header = JSON.parse(buffer.toString("utf8", headerStart, headerStart + headerLength)) as Header;
  if (header.version !== 1) throw new Error("Unsupported lexical block version");
  let offset = headerStart + headerLength;
  const rowIds = buffer.subarray(offset, offset + header.rowBytes); offset += header.rowBytes;
  const columns = header.columns.map((column) => {
    const source = buffer.subarray(offset, offset + column.bytes); offset += column.bytes;
    const cursor = { value: 0 }; const values: string[] = []; let previous = 0n;
    for (let index = 0; index < column.count; index++) {
      const first = Number(readUint(source, cursor));
      if (column.mode === "dictionary") {
        const value = column.dictionary[first]; if (value === undefined) throw new Error("Invalid dictionary index"); values.push(value);
      } else {
        const style = column.styles[first]; if (!style) throw new Error("Invalid decimal style");
        previous += unzig(readUint(source, cursor)); values.push(restoreDecimal(previous, style));
      }
    }
    if (cursor.value !== source.length) throw new Error("Column length mismatch");
    return { values, cursor: 0 };
  });
  if (offset !== buffer.length) throw new Error("Block length mismatch");
  function render(pieces: readonly Piece[]): string {
    return pieces.map((piece) => {
      if (piece[0] === "l") return piece[1];
      if (piece[0] === "j") return JSON.stringify(render(piece[1]));
      const column = columns[piece[1]]; const value = column?.values[column.cursor++];
      if (value === undefined) throw new Error("Column exhausted");
      return value;
    }).join("");
  }
  const cursor = { value: 0 }; const lines: string[] = [];
  for (let index = 0; index < header.rows; index++) {
    const template = header.templates[Number(readUint(rowIds, cursor))];
    if (!template) throw new Error("Invalid row template"); lines.push(render(template));
  }
  if (cursor.value !== rowIds.length || columns.some((column) => column.cursor !== column.values.length)) throw new Error("Unused block data");
  return lines;
}

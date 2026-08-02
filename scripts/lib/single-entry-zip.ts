import { createReadStream } from "node:fs";
import fs from "node:fs/promises";
import { PassThrough, Transform, type Readable } from "node:stream";
import { pipeline } from "node:stream/promises";
import { createInflateRaw } from "node:zlib";

const END_OF_CENTRAL_DIRECTORY = 0x0605_4b50;
const CENTRAL_DIRECTORY_ENTRY = 0x0201_4b50;
const LOCAL_FILE_HEADER = 0x0403_4b50;
const MAX_END_RECORD_BYTES = 65_535 + 22;

export interface SingleZipEntry {
  name: string;
  stream: Readable;
  expectedUncompressedBytes: number;
  completed: Promise<number>;
}

/**
 * Opens the sole regular entry in a non-Zip64 archive without materializing
 * either the compressed or uncompressed file in memory.
 */
export async function openSingleZipEntry(
  archiveFile: string,
  expectedName?: string,
): Promise<SingleZipEntry> {
  const handle = await fs.open(archiveFile, "r");
  let metadata: ReturnType<typeof parseCentralDirectory>;
  try {
    const stat = await handle.stat();
    if (!stat.isFile() || stat.size < 22) throw new Error("ZIP archive is truncated.");
    const tailBytes = Math.min(stat.size, MAX_END_RECORD_BYTES);
    const tail = Buffer.allocUnsafe(tailBytes);
    await readExactly(handle, tail, stat.size - tailBytes);
    let endOffset = -1;
    for (let offset = tail.length - 22; offset >= 0; offset -= 1) {
      if (tail.readUInt32LE(offset) === END_OF_CENTRAL_DIRECTORY) {
        endOffset = offset;
        break;
      }
    }
    if (endOffset < 0) throw new Error("ZIP end-of-central-directory record is missing.");
    const disk = tail.readUInt16LE(endOffset + 4);
    const centralDisk = tail.readUInt16LE(endOffset + 6);
    const diskEntries = tail.readUInt16LE(endOffset + 8);
    const totalEntries = tail.readUInt16LE(endOffset + 10);
    const centralBytes = tail.readUInt32LE(endOffset + 12);
    const centralOffset = tail.readUInt32LE(endOffset + 16);
    const commentBytes = tail.readUInt16LE(endOffset + 20);
    if (disk !== 0 || centralDisk !== 0 || diskEntries !== 1 || totalEntries !== 1) {
      throw new Error("ZIP archive must contain exactly one entry on one disk.");
    }
    if (endOffset + 22 + commentBytes !== tail.length
      || centralBytes < 46
      || centralBytes > 1_048_576
      || centralOffset + centralBytes > stat.size) {
      throw new Error("ZIP central-directory bounds are invalid.");
    }
    const central = Buffer.allocUnsafe(centralBytes);
    await readExactly(handle, central, centralOffset);
    metadata = parseCentralDirectory(central, stat.size);

    const local = Buffer.allocUnsafe(30);
    await readExactly(handle, local, metadata.localHeaderOffset);
    if (local.readUInt32LE(0) !== LOCAL_FILE_HEADER) {
      throw new Error("ZIP local-file header is invalid.");
    }
    const localFlags = local.readUInt16LE(6);
    const localMethod = local.readUInt16LE(8);
    const localNameBytes = local.readUInt16LE(26);
    const localExtraBytes = local.readUInt16LE(28);
    if (localFlags !== metadata.flags || localMethod !== metadata.method) {
      throw new Error("ZIP local and central entry metadata disagree.");
    }
    const localName = Buffer.allocUnsafe(localNameBytes);
    await readExactly(handle, localName, metadata.localHeaderOffset + 30);
    if (localName.toString("utf8") !== metadata.name) {
      throw new Error("ZIP local and central entry names disagree.");
    }
    metadata.dataOffset = metadata.localHeaderOffset + 30 + localNameBytes + localExtraBytes;
    if (metadata.dataOffset + metadata.compressedBytes > centralOffset) {
      throw new Error("ZIP entry overlaps its central directory.");
    }
  } finally {
    await handle.close();
  }

  if (expectedName !== undefined && metadata.name !== expectedName) {
    throw new Error(`ZIP entry is ${metadata.name}; expected ${expectedName}.`);
  }
  const compressed = createReadStream(archiveFile, {
    start: metadata.dataOffset,
    end: metadata.dataOffset + metadata.compressedBytes - 1,
  });
  const output = new PassThrough();
  let decodedBytes = 0;
  const counter = new Transform({
    transform(chunk: Buffer, _encoding, callback) {
      decodedBytes += chunk.byteLength;
      callback(null, chunk);
    },
    flush(callback) {
      callback(decodedBytes === metadata.uncompressedBytes
        ? undefined
        : new Error(
            `ZIP entry decoded to ${decodedBytes} bytes; expected ${metadata.uncompressedBytes}.`,
          ));
    },
  });
  const completed = (metadata.method === 8
    ? pipeline(compressed, createInflateRaw(), counter, output)
    : pipeline(compressed, counter, output))
    .then(() => decodedBytes);
  return {
    name: metadata.name,
    stream: output,
    expectedUncompressedBytes: metadata.uncompressedBytes,
    completed,
  };
}

function parseCentralDirectory(central: Buffer, archiveBytes: number): {
  name: string;
  flags: number;
  method: number;
  compressedBytes: number;
  uncompressedBytes: number;
  localHeaderOffset: number;
  dataOffset: number;
} {
  if (central.readUInt32LE(0) !== CENTRAL_DIRECTORY_ENTRY) {
    throw new Error("ZIP central-directory entry is invalid.");
  }
  const flags = central.readUInt16LE(8);
  const method = central.readUInt16LE(10);
  const compressedBytes = central.readUInt32LE(20);
  const uncompressedBytes = central.readUInt32LE(24);
  const nameBytes = central.readUInt16LE(28);
  const extraBytes = central.readUInt16LE(30);
  const commentBytes = central.readUInt16LE(32);
  const disk = central.readUInt16LE(34);
  const localHeaderOffset = central.readUInt32LE(42);
  if ((flags & 1) !== 0) throw new Error("Encrypted ZIP entries are unsupported.");
  if (method !== 0 && method !== 8) throw new Error(`Unsupported ZIP compression method ${method}.`);
  if (disk !== 0
    || compressedBytes === 0xffff_ffff
    || uncompressedBytes === 0xffff_ffff
    || localHeaderOffset === 0xffff_ffff) {
    throw new Error("Multi-disk and Zip64 archives are unsupported.");
  }
  const expectedBytes = 46 + nameBytes + extraBytes + commentBytes;
  if (expectedBytes !== central.length || localHeaderOffset + 30 > archiveBytes) {
    throw new Error("ZIP central-directory entry bounds are invalid.");
  }
  const name = central.subarray(46, 46 + nameBytes).toString("utf8");
  if (!name || name.endsWith("/") || name.includes("\\") || name.split("/").includes("..")) {
    throw new Error("ZIP entry name is unsafe.");
  }
  return {
    name,
    flags,
    method,
    compressedBytes,
    uncompressedBytes,
    localHeaderOffset,
    dataOffset: 0,
  };
}

async function readExactly(
  handle: fs.FileHandle,
  buffer: Buffer,
  position: number,
): Promise<void> {
  const { bytesRead } = await handle.read(buffer, 0, buffer.length, position);
  if (bytesRead !== buffer.length) throw new Error("ZIP archive is truncated.");
}

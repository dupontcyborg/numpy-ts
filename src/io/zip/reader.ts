/**
 * Minimal ZIP file reader
 *
 * Reads ZIP files with STORED or DEFLATE compression.
 * Uses the Compression Streams API (built into modern browsers and Node.js 18+).
 */

import {
  ZIP_LOCAL_SIGNATURE,
  ZIP_END_SIGNATURE,
  ZIP_STORED,
  ZIP_DEFLATED,
  type RawZipEntry,
} from './types';

/**
 * Read a ZIP file and return its entries
 *
 * @param buffer - ZIP file contents
 * @returns Map of file names to their uncompressed data
 */
export async function readZip(buffer: ArrayBuffer | Uint8Array): Promise<Map<string, Uint8Array>> {
  const entries = parseZipEntries(buffer);
  const result = new Map<string, Uint8Array>();

  for (const entry of entries) {
    const data = await decompressEntry(entry);
    result.set(entry.name, data);
  }

  return result;
}

/**
 * Synchronously read a ZIP file (only works for STORED entries)
 *
 * @param buffer - ZIP file contents
 * @returns Map of file names to their uncompressed data
 * @throws Error if any entry uses compression
 */
export function readZipSync(buffer: ArrayBuffer | Uint8Array): Map<string, Uint8Array> {
  const entries = parseZipEntries(buffer);
  const result = new Map<string, Uint8Array>();

  for (const entry of entries) {
    if (entry.compressionMethod !== ZIP_STORED) {
      throw new Error(
        `Cannot read compressed entry synchronously: ${entry.name}. ` +
          `Use readZip() (async) for DEFLATE-compressed files.`
      );
    }
    result.set(entry.name, entry.compressedData);
  }

  return result;
}

/**
 * Central directory entry info
 */
interface CentralDirEntry {
  name: string;
  compressionMethod: number;
  crc32: number;
  compressedSize: number;
  uncompressedSize: number;
  localHeaderOffset: number;
}

/**
 * Parse ZIP entries without decompressing
 *
 * Uses central directory for reliable size information, as some ZIP writers
 * (including Python's zipfile module used by NumPy) set local header sizes to
 * 0xFFFFFFFF when streaming.
 */
function parseZipEntries(buffer: ArrayBuffer | Uint8Array): RawZipEntry[] {
  const bytes = buffer instanceof ArrayBuffer ? new Uint8Array(buffer) : buffer;
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const entries: RawZipEntry[] = [];

  // Find end of central directory
  let eocdOffset = -1;
  for (let i = bytes.length - 22; i >= 0; i--) {
    if (view.getUint32(i, true) === ZIP_END_SIGNATURE) {
      eocdOffset = i;
      break;
    }
  }

  if (eocdOffset === -1) {
    throw new Error('Invalid ZIP file: end of central directory not found');
  }

  // Read central directory location
  const centralDirOffset = view.getUint32(eocdOffset + 16, true);
  const numEntries = view.getUint16(eocdOffset + 10, true);

  // Parse central directory entries first to get reliable sizes
  const centralEntries: CentralDirEntry[] = [];
  let cdOffset = centralDirOffset;

  for (let i = 0; i < numEntries; i++) {
    const signature = view.getUint32(cdOffset, true);
    if (signature !== 0x02014b50) {
      // Central directory signature
      break;
    }

    const compressionMethod = view.getUint16(cdOffset + 10, true);
    const crc32 = view.getUint32(cdOffset + 16, true);
    const compressedSize = view.getUint32(cdOffset + 20, true);
    const uncompressedSize = view.getUint32(cdOffset + 24, true);
    const fileNameLength = view.getUint16(cdOffset + 28, true);
    const extraFieldLength = view.getUint16(cdOffset + 30, true);
    const commentLength = view.getUint16(cdOffset + 32, true);
    const localHeaderOffset = view.getUint32(cdOffset + 42, true);

    const fileNameBytes = bytes.slice(cdOffset + 46, cdOffset + 46 + fileNameLength);
    const fileName = new TextDecoder('utf-8').decode(fileNameBytes);

    centralEntries.push({
      name: fileName,
      compressionMethod,
      crc32,
      compressedSize,
      uncompressedSize,
      localHeaderOffset,
    });

    cdOffset = cdOffset + 46 + fileNameLength + extraFieldLength + commentLength;
  }

  // Now extract data using local headers for data location, but central directory for sizes
  for (const ce of centralEntries) {
    const localOffset = ce.localHeaderOffset;
    const signature = view.getUint32(localOffset, true);

    if (signature !== ZIP_LOCAL_SIGNATURE) {
      throw new Error(`Invalid local file header at offset ${localOffset}`);
    }

    const fileNameLength = view.getUint16(localOffset + 26, true);
    const extraFieldLength = view.getUint16(localOffset + 28, true);

    const dataStart = localOffset + 30 + fileNameLength + extraFieldLength;
    const compressedData = bytes.slice(dataStart, dataStart + ce.compressedSize);

    entries.push({
      name: ce.name,
      compressedData,
      compressionMethod: ce.compressionMethod,
      crc32: ce.crc32,
      compressedSize: ce.compressedSize,
      uncompressedSize: ce.uncompressedSize,
    });
  }

  return entries;
}

/**
 * Decompress a single ZIP entry
 */
async function decompressEntry(entry: RawZipEntry): Promise<Uint8Array> {
  if (entry.compressionMethod === ZIP_STORED) {
    return entry.compressedData;
  }

  if (entry.compressionMethod === ZIP_DEFLATED) {
    return await inflateRaw(entry.compressedData);
  }

  throw new Error(`Unsupported compression method: ${entry.compressionMethod}`);
}

/**
 * Decompress raw DEFLATE data using DecompressionStream
 */
async function inflateRaw(data: Uint8Array): Promise<Uint8Array> {
  // Check if DecompressionStream is available
  if (typeof DecompressionStream === 'undefined') {
    throw new Error(
      'DecompressionStream is not available. ' +
        'This environment does not support the Compression Streams API. ' +
        'Please use a modern browser or Node.js 18+.'
    );
  }

  // DEFLATE in ZIP is "raw" DEFLATE (no zlib header)
  // DecompressionStream expects the "deflate-raw" format
  const ds = new DecompressionStream('deflate-raw');

  // Create a copy to ensure we have a clean ArrayBuffer (avoids SharedArrayBuffer issues)
  const dataCopy = new Uint8Array(data.length);
  dataCopy.set(data);

  const writer = ds.writable.getWriter();
  void writer.write(dataCopy);
  void writer.close();

  const reader = ds.readable.getReader();
  const chunks: Uint8Array[] = [];

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
  }

  // Concatenate chunks
  const totalLength = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
  const result = new Uint8Array(totalLength);
  let offset = 0;
  for (const chunk of chunks) {
    result.set(chunk, offset);
    offset += chunk.length;
  }

  return result;
}

/**
 * Minimal ZIP file writer
 *
 * Creates ZIP files with optional DEFLATE compression.
 * Uses the Compression Streams API (built into modern browsers and Node.js 18+).
 */

import {
  ZIP_LOCAL_SIGNATURE,
  ZIP_CENTRAL_SIGNATURE,
  ZIP_END_SIGNATURE,
  ZIP_STORED,
  ZIP_DEFLATED,
  crc32,
} from './types';

/**
 * Options for writing a ZIP file
 */
export interface ZipWriteOptions {
  /** Whether to compress files (default: false for NPZ compatibility) */
  compress?: boolean;
}

/**
 * Create a ZIP file from a map of file names to data
 *
 * @param files - Map of file names to their data
 * @param options - Write options
 * @returns ZIP file as Uint8Array
 */
export async function writeZip(
  files: Map<string, Uint8Array>,
  options: ZipWriteOptions = {}
): Promise<Uint8Array> {
  const compress = options.compress ?? false;
  const entries: {
    name: string;
    data: Uint8Array;
    compressedData: Uint8Array;
    crc: number;
    compressionMethod: number;
    offset: number;
  }[] = [];

  // Prepare entries
  for (const [name, data] of files) {
    const crc = crc32(data);
    let compressedData: Uint8Array;
    let compressionMethod: number;

    if (compress) {
      compressedData = await deflateRaw(data);
      // Only use compression if it actually makes the data smaller
      if (compressedData.length < data.length) {
        compressionMethod = ZIP_DEFLATED;
      } else {
        compressedData = data;
        compressionMethod = ZIP_STORED;
      }
    } else {
      compressedData = data;
      compressionMethod = ZIP_STORED;
    }

    entries.push({
      name,
      data,
      compressedData,
      crc,
      compressionMethod,
      offset: 0, // Will be set during writing
    });
  }

  // Calculate total size
  let localHeadersSize = 0;
  for (const entry of entries) {
    const nameBytes = new TextEncoder().encode(entry.name);
    localHeadersSize += 30 + nameBytes.length + entry.compressedData.length;
  }

  let centralDirSize = 0;
  for (const entry of entries) {
    const nameBytes = new TextEncoder().encode(entry.name);
    centralDirSize += 46 + nameBytes.length;
  }

  const eocdSize = 22;
  const totalSize = localHeadersSize + centralDirSize + eocdSize;

  // Allocate buffer
  const output = new Uint8Array(totalSize);
  const view = new DataView(output.buffer);

  // Write local file headers and data
  let offset = 0;
  for (const entry of entries) {
    entry.offset = offset;
    offset = writeLocalHeader(output, view, offset, entry);
  }

  // Write central directory
  const centralDirOffset = offset;
  for (const entry of entries) {
    offset = writeCentralHeader(output, view, offset, entry);
  }

  // Write end of central directory
  writeEndOfCentralDirectory(view, offset, entries.length, centralDirSize, centralDirOffset);

  return output;
}

/**
 * Create a ZIP file synchronously (no compression)
 *
 * @param files - Map of file names to their data
 * @returns ZIP file as Uint8Array
 */
export function writeZipSync(files: Map<string, Uint8Array>): Uint8Array {
  const entries: {
    name: string;
    data: Uint8Array;
    compressedData: Uint8Array;
    crc: number;
    compressionMethod: number;
    offset: number;
  }[] = [];

  // Prepare entries (no compression in sync mode)
  for (const [name, data] of files) {
    const crc = crc32(data);
    entries.push({
      name,
      data,
      compressedData: data,
      crc,
      compressionMethod: ZIP_STORED,
      offset: 0,
    });
  }

  // Calculate total size
  let localHeadersSize = 0;
  for (const entry of entries) {
    const nameBytes = new TextEncoder().encode(entry.name);
    localHeadersSize += 30 + nameBytes.length + entry.compressedData.length;
  }

  let centralDirSize = 0;
  for (const entry of entries) {
    const nameBytes = new TextEncoder().encode(entry.name);
    centralDirSize += 46 + nameBytes.length;
  }

  const eocdSize = 22;
  const totalSize = localHeadersSize + centralDirSize + eocdSize;

  // Allocate buffer
  const output = new Uint8Array(totalSize);
  const view = new DataView(output.buffer);

  // Write local file headers and data
  let offset = 0;
  for (const entry of entries) {
    entry.offset = offset;
    offset = writeLocalHeader(output, view, offset, entry);
  }

  // Write central directory
  const centralDirOffset = offset;
  for (const entry of entries) {
    offset = writeCentralHeader(output, view, offset, entry);
  }

  // Write end of central directory
  writeEndOfCentralDirectory(view, offset, entries.length, centralDirSize, centralDirOffset);

  return output;
}

/**
 * Write a local file header and data
 */
function writeLocalHeader(
  output: Uint8Array,
  view: DataView,
  offset: number,
  entry: {
    name: string;
    compressedData: Uint8Array;
    data: Uint8Array;
    crc: number;
    compressionMethod: number;
  }
): number {
  const nameBytes = new TextEncoder().encode(entry.name);

  // Signature
  view.setUint32(offset, ZIP_LOCAL_SIGNATURE, true);
  offset += 4;

  // Version needed to extract (2.0 for DEFLATE)
  view.setUint16(offset, entry.compressionMethod === ZIP_DEFLATED ? 20 : 10, true);
  offset += 2;

  // General purpose bit flag
  view.setUint16(offset, 0, true);
  offset += 2;

  // Compression method
  view.setUint16(offset, entry.compressionMethod, true);
  offset += 2;

  // Last mod time (use a fixed value)
  view.setUint16(offset, 0, true);
  offset += 2;

  // Last mod date (use a fixed value)
  view.setUint16(offset, 0x21, true); // Jan 1, 1980
  offset += 2;

  // CRC-32
  view.setUint32(offset, entry.crc, true);
  offset += 4;

  // Compressed size
  view.setUint32(offset, entry.compressedData.length, true);
  offset += 4;

  // Uncompressed size
  view.setUint32(offset, entry.data.length, true);
  offset += 4;

  // File name length
  view.setUint16(offset, nameBytes.length, true);
  offset += 2;

  // Extra field length
  view.setUint16(offset, 0, true);
  offset += 2;

  // File name
  output.set(nameBytes, offset);
  offset += nameBytes.length;

  // File data
  output.set(entry.compressedData, offset);
  offset += entry.compressedData.length;

  return offset;
}

/**
 * Write a central directory header
 */
function writeCentralHeader(
  output: Uint8Array,
  view: DataView,
  offset: number,
  entry: {
    name: string;
    compressedData: Uint8Array;
    data: Uint8Array;
    crc: number;
    compressionMethod: number;
    offset: number;
  }
): number {
  const nameBytes = new TextEncoder().encode(entry.name);

  // Signature
  view.setUint32(offset, ZIP_CENTRAL_SIGNATURE, true);
  offset += 4;

  // Version made by
  view.setUint16(offset, 20, true);
  offset += 2;

  // Version needed to extract
  view.setUint16(offset, entry.compressionMethod === ZIP_DEFLATED ? 20 : 10, true);
  offset += 2;

  // General purpose bit flag
  view.setUint16(offset, 0, true);
  offset += 2;

  // Compression method
  view.setUint16(offset, entry.compressionMethod, true);
  offset += 2;

  // Last mod time
  view.setUint16(offset, 0, true);
  offset += 2;

  // Last mod date
  view.setUint16(offset, 0x21, true);
  offset += 2;

  // CRC-32
  view.setUint32(offset, entry.crc, true);
  offset += 4;

  // Compressed size
  view.setUint32(offset, entry.compressedData.length, true);
  offset += 4;

  // Uncompressed size
  view.setUint32(offset, entry.data.length, true);
  offset += 4;

  // File name length
  view.setUint16(offset, nameBytes.length, true);
  offset += 2;

  // Extra field length
  view.setUint16(offset, 0, true);
  offset += 2;

  // File comment length
  view.setUint16(offset, 0, true);
  offset += 2;

  // Disk number start
  view.setUint16(offset, 0, true);
  offset += 2;

  // Internal file attributes
  view.setUint16(offset, 0, true);
  offset += 2;

  // External file attributes
  view.setUint32(offset, 0, true);
  offset += 4;

  // Relative offset of local header
  view.setUint32(offset, entry.offset, true);
  offset += 4;

  // File name
  output.set(nameBytes, offset);
  offset += nameBytes.length;

  return offset;
}

/**
 * Write end of central directory record
 */
function writeEndOfCentralDirectory(
  view: DataView,
  offset: number,
  numEntries: number,
  centralDirSize: number,
  centralDirOffset: number
): void {
  // Signature
  view.setUint32(offset, ZIP_END_SIGNATURE, true);
  offset += 4;

  // Number of this disk
  view.setUint16(offset, 0, true);
  offset += 2;

  // Disk where central directory starts
  view.setUint16(offset, 0, true);
  offset += 2;

  // Number of central directory records on this disk
  view.setUint16(offset, numEntries, true);
  offset += 2;

  // Total number of central directory records
  view.setUint16(offset, numEntries, true);
  offset += 2;

  // Size of central directory
  view.setUint32(offset, centralDirSize, true);
  offset += 4;

  // Offset of central directory
  view.setUint32(offset, centralDirOffset, true);
  offset += 4;

  // Comment length
  view.setUint16(offset, 0, true);
}

/**
 * Compress data using raw DEFLATE
 */
async function deflateRaw(data: Uint8Array): Promise<Uint8Array> {
  // Check if CompressionStream is available
  if (typeof CompressionStream === 'undefined') {
    throw new Error(
      'CompressionStream is not available. ' +
        'This environment does not support the Compression Streams API. ' +
        'Please use a modern browser or Node.js 18+.'
    );
  }

  const cs = new CompressionStream('deflate-raw');

  // Create a copy to ensure we have a clean ArrayBuffer (avoids SharedArrayBuffer issues)
  const dataCopy = new Uint8Array(data.length);
  dataCopy.set(data);

  const writer = cs.writable.getWriter();
  void writer.write(dataCopy);
  void writer.close();

  const reader = cs.readable.getReader();
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

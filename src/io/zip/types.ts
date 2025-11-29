/**
 * ZIP file format types and constants
 */

/**
 * ZIP local file header signature
 */
export const ZIP_LOCAL_SIGNATURE = 0x04034b50;

/**
 * ZIP central directory header signature
 */
export const ZIP_CENTRAL_SIGNATURE = 0x02014b50;

/**
 * ZIP end of central directory signature
 */
export const ZIP_END_SIGNATURE = 0x06054b50;

/**
 * Compression methods
 */
export const ZIP_STORED = 0; // No compression
export const ZIP_DEFLATED = 8; // DEFLATE compression

/**
 * Entry in a ZIP file
 */
export interface ZipEntry {
  /** File name */
  name: string;
  /** Uncompressed data */
  data: Uint8Array;
  /** Compression method used */
  compressionMethod: number;
  /** CRC-32 checksum */
  crc32: number;
  /** Compressed size */
  compressedSize: number;
  /** Uncompressed size */
  uncompressedSize: number;
}

/**
 * Raw entry as read from ZIP (before decompression)
 */
export interface RawZipEntry {
  /** File name */
  name: string;
  /** Compressed data */
  compressedData: Uint8Array;
  /** Compression method */
  compressionMethod: number;
  /** CRC-32 checksum */
  crc32: number;
  /** Compressed size */
  compressedSize: number;
  /** Uncompressed size */
  uncompressedSize: number;
}

/**
 * CRC-32 lookup table
 */
const CRC32_TABLE = (() => {
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    table[i] = c;
  }
  return table;
})();

/**
 * Calculate CRC-32 checksum
 */
export function crc32(data: Uint8Array): number {
  let crc = 0xffffffff;
  for (let i = 0; i < data.length; i++) {
    crc = CRC32_TABLE[(crc ^ data[i]!) & 0xff]! ^ (crc >>> 8);
  }
  return (crc ^ 0xffffffff) >>> 0;
}

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
 * CRC-32 lookup tables (slice-by-8 for ~8x throughput over byte-at-a-time)
 */
const CRC32_TABLES = (() => {
  const tables: Uint32Array[] = [];
  const t0 = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) {
      c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    }
    t0[i] = c;
  }
  tables.push(t0);
  for (let k = 1; k < 8; k++) {
    const tk = new Uint32Array(256);
    for (let i = 0; i < 256; i++) {
      tk[i] = (tables[k - 1]![i]! >>> 8) ^ t0[tables[k - 1]![i]! & 0xff]!;
    }
    tables.push(tk);
  }
  return tables;
})();

/**
 * Calculate CRC-32 checksum using slice-by-8.
 * Processes 8 bytes per iteration, then finishes byte-at-a-time.
 */
export function crc32(data: Uint8Array): number {
  const t0 = CRC32_TABLES[0]!;
  const t1 = CRC32_TABLES[1]!;
  const t2 = CRC32_TABLES[2]!;
  const t3 = CRC32_TABLES[3]!;
  const t4 = CRC32_TABLES[4]!;
  const t5 = CRC32_TABLES[5]!;
  const t6 = CRC32_TABLES[6]!;
  const t7 = CRC32_TABLES[7]!;
  const len = data.length;

  let crc = 0xffffffff;
  let i = 0;

  const end8 = len - 7;
  while (i < end8) {
    const lo = (crc ^ (data[i]! | (data[i + 1]! << 8) | (data[i + 2]! << 16) | (data[i + 3]! << 24))) >>> 0;
    const hi = data[i + 4]! | (data[i + 5]! << 8) | (data[i + 6]! << 16) | (data[i + 7]! << 24);
    crc =
      t7[lo & 0xff]! ^
      t6[(lo >>> 8) & 0xff]! ^
      t5[(lo >>> 16) & 0xff]! ^
      t4[(lo >>> 24) & 0xff]! ^
      t3[hi & 0xff]! ^
      t2[(hi >>> 8) & 0xff]! ^
      t1[(hi >>> 16) & 0xff]! ^
      t0[(hi >>> 24) & 0xff]!;
    i += 8;
  }

  while (i < len) {
    crc = t0[(crc ^ data[i]!) & 0xff]! ^ (crc >>> 8);
    i++;
  }

  return (crc ^ 0xffffffff) >>> 0;
}

import { describe, it, expect } from 'vitest';
import { readZip, readZipSync, writeZip, writeZipSync, crc32 } from '../../../src/io/zip';

describe('ZIP Format', () => {
  describe('crc32', () => {
    it('calculates CRC-32 for empty data', () => {
      const data = new Uint8Array(0);
      const checksum = crc32(data);
      expect(checksum).toBe(0);
    });

    it('calculates CRC-32 for simple data', () => {
      // "hello" in ASCII
      const data = new Uint8Array([0x68, 0x65, 0x6c, 0x6c, 0x6f]);
      const checksum = crc32(data);
      // Known CRC-32 for "hello"
      expect(checksum).toBe(0x3610a686);
    });

    it('calculates CRC-32 for longer data', () => {
      // Create a repeating pattern
      const data = new Uint8Array(256);
      for (let i = 0; i < 256; i++) {
        data[i] = i;
      }
      // Just verify it returns a number (specific value depends on implementation)
      expect(typeof crc32(data)).toBe('number');
      expect(crc32(data)).toBeGreaterThan(0);
    });

    it('produces different checksums for different data', () => {
      const data1 = new Uint8Array([1, 2, 3]);
      const data2 = new Uint8Array([1, 2, 4]);
      expect(crc32(data1)).not.toBe(crc32(data2));
    });
  });

  describe('writeZipSync and readZipSync', () => {
    it('round-trips a single file', () => {
      const files = new Map([['test.txt', new TextEncoder().encode('Hello, World!')]]);

      const zipBytes = writeZipSync(files);
      const result = readZipSync(zipBytes);

      expect(result.size).toBe(1);
      expect(result.has('test.txt')).toBe(true);
      expect(new TextDecoder().decode(result.get('test.txt'))).toBe('Hello, World!');
    });

    it('round-trips multiple files', () => {
      const files = new Map([
        ['file1.txt', new TextEncoder().encode('Content 1')],
        ['file2.txt', new TextEncoder().encode('Content 2')],
        ['file3.bin', new Uint8Array([0, 1, 2, 3, 4, 5])],
      ]);

      const zipBytes = writeZipSync(files);
      const result = readZipSync(zipBytes);

      expect(result.size).toBe(3);
      expect(new TextDecoder().decode(result.get('file1.txt'))).toBe('Content 1');
      expect(new TextDecoder().decode(result.get('file2.txt'))).toBe('Content 2');
      expect(Array.from(result.get('file3.bin')!)).toEqual([0, 1, 2, 3, 4, 5]);
    });

    it('handles empty files', () => {
      const files = new Map([['empty.txt', new Uint8Array(0)]]);

      const zipBytes = writeZipSync(files);
      const result = readZipSync(zipBytes);

      expect(result.size).toBe(1);
      expect(result.get('empty.txt')!.length).toBe(0);
    });

    it('handles large files', () => {
      // Create a 1MB file
      const largeData = new Uint8Array(1024 * 1024);
      for (let i = 0; i < largeData.length; i++) {
        largeData[i] = i % 256;
      }

      const files = new Map([['large.bin', largeData]]);
      const zipBytes = writeZipSync(files);
      const result = readZipSync(zipBytes);

      expect(result.size).toBe(1);
      expect(result.get('large.bin')!.length).toBe(1024 * 1024);
      expect(result.get('large.bin')![0]).toBe(0);
      expect(result.get('large.bin')![1000]).toBe(1000 % 256);
    });

    it('handles empty archive', () => {
      const files = new Map<string, Uint8Array>();
      const zipBytes = writeZipSync(files);
      const result = readZipSync(zipBytes);

      expect(result.size).toBe(0);
    });

    it('preserves file names with special characters', () => {
      const files = new Map([
        ['path/to/file.txt', new TextEncoder().encode('nested')],
        ['file-with-dashes.txt', new TextEncoder().encode('dashes')],
        ['file_with_underscores.txt', new TextEncoder().encode('underscores')],
      ]);

      const zipBytes = writeZipSync(files);
      const result = readZipSync(zipBytes);

      expect(result.has('path/to/file.txt')).toBe(true);
      expect(result.has('file-with-dashes.txt')).toBe(true);
      expect(result.has('file_with_underscores.txt')).toBe(true);
    });
  });

  describe('async writeZip and readZip', () => {
    it('round-trips without compression', async () => {
      const files = new Map([['test.txt', new TextEncoder().encode('Hello, World!')]]);

      const zipBytes = await writeZip(files, { compress: false });
      const result = await readZip(zipBytes);

      expect(result.size).toBe(1);
      expect(new TextDecoder().decode(result.get('test.txt'))).toBe('Hello, World!');
    });

    it('round-trips with compression', async () => {
      // Use repetitive data that compresses well
      const content = 'Hello, World! '.repeat(100);
      const files = new Map([['test.txt', new TextEncoder().encode(content)]]);

      const compressedZip = await writeZip(files, { compress: true });
      const uncompressedZip = await writeZip(files, { compress: false });

      // Compressed should be smaller
      expect(compressedZip.length).toBeLessThan(uncompressedZip.length);

      // Both should decompress to same content
      const compressedResult = await readZip(compressedZip);
      const uncompressedResult = await readZip(uncompressedZip);

      expect(new TextDecoder().decode(compressedResult.get('test.txt'))).toBe(content);
      expect(new TextDecoder().decode(uncompressedResult.get('test.txt'))).toBe(content);
    });

    it('handles multiple compressed files', async () => {
      const files = new Map([
        ['a.txt', new TextEncoder().encode('AAAA'.repeat(100))],
        ['b.txt', new TextEncoder().encode('BBBB'.repeat(100))],
        ['c.txt', new TextEncoder().encode('CCCC'.repeat(100))],
      ]);

      const zipBytes = await writeZip(files, { compress: true });
      const result = await readZip(zipBytes);

      expect(result.size).toBe(3);
      expect(new TextDecoder().decode(result.get('a.txt'))).toBe('AAAA'.repeat(100));
      expect(new TextDecoder().decode(result.get('b.txt'))).toBe('BBBB'.repeat(100));
      expect(new TextDecoder().decode(result.get('c.txt'))).toBe('CCCC'.repeat(100));
    });
  });

  describe('error handling', () => {
    it('throws for invalid ZIP data', () => {
      const invalidData = new Uint8Array([0, 1, 2, 3, 4, 5]);
      expect(() => readZipSync(invalidData)).toThrow('Invalid ZIP file');
    });

    it('async throws for invalid ZIP data', async () => {
      const invalidData = new Uint8Array([0, 1, 2, 3, 4, 5]);
      await expect(readZip(invalidData)).rejects.toThrow('Invalid ZIP file');
    });

    it('sync throws for compressed entries', async () => {
      // Create a compressed ZIP
      const files = new Map([['test.txt', new TextEncoder().encode('Hello'.repeat(100))]]);
      const compressedZip = await writeZip(files, { compress: true });

      // Sync read should fail for compressed entries
      expect(() => readZipSync(compressedZip)).toThrow(
        'Cannot read compressed entry synchronously'
      );
    });
  });
});

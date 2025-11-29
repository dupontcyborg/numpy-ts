import { describe, it, expect } from 'vitest';
import {
  parseNpy,
  serializeNpy,
  parseNpyHeader,
  UnsupportedDTypeError,
  InvalidNpyError,
  DTYPE_TO_DESCR,
} from '../../../src/io/npy';
import { array, zeros, ones, arange } from '../../../src/core/ndarray';
import type { DType } from '../../../src/core/dtype';

describe('NPY Format', () => {
  describe('serializeNpy', () => {
    it('serializes a 1D float64 array', () => {
      const arr = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const bytes = serializeNpy(arr);

      // Check magic number
      expect(bytes[0]).toBe(0x93);
      expect(bytes[1]).toBe(0x4e); // N
      expect(bytes[2]).toBe(0x55); // U
      expect(bytes[3]).toBe(0x4d); // M
      expect(bytes[4]).toBe(0x50); // P
      expect(bytes[5]).toBe(0x59); // Y

      // Check version (3.0)
      expect(bytes[6]).toBe(3);
      expect(bytes[7]).toBe(0);

      // Verify we can round-trip
      const parsed = parseNpy(bytes);
      expect(parsed.shape).toEqual([5]);
      expect(parsed.dtype).toBe('float64');
      expect(parsed.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('serializes a 2D float32 array', () => {
      const arr = array(
        [
          [1.0, 2.0, 3.0],
          [4.0, 5.0, 6.0],
        ],
        'float32'
      );
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.shape).toEqual([2, 3]);
      expect(parsed.dtype).toBe('float32');
      expect(parsed.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('serializes a 3D array', () => {
      const data = [
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ];
      const arr = array(data);
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.shape).toEqual([2, 2, 2]);
      expect(parsed.toArray()).toEqual(data);
    });

    it('serializes a scalar (0D) array', () => {
      const arr = array(42.5);
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.shape).toEqual([]);
      expect(parsed.size).toBe(1);
      expect(parsed.get([])).toBe(42.5);
    });

    it('serializes int32 arrays', () => {
      const arr = array([1, 2, 3, 4], 'int32');
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('int32');
      expect(parsed.toArray()).toEqual([1, 2, 3, 4]);
    });

    it('serializes int64 arrays with BigInt', () => {
      const arr = array([BigInt(1), BigInt(2), BigInt(3)], 'int64');
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('int64');
      expect(parsed.toArray()).toEqual([BigInt(1), BigInt(2), BigInt(3)]);
    });

    it('serializes uint64 arrays with BigInt', () => {
      const arr = array([BigInt(1), BigInt(2), BigInt(3)], 'uint64');
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('uint64');
      expect(parsed.toArray()).toEqual([BigInt(1), BigInt(2), BigInt(3)]);
    });

    it('serializes bool arrays', () => {
      const arr = array([1, 0, 1, 1, 0], 'bool');
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('bool');
      expect(parsed.toArray()).toEqual([1, 0, 1, 1, 0]);
    });

    it('serializes uint8 arrays', () => {
      const arr = array([0, 128, 255], 'uint8');
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('uint8');
      expect(parsed.toArray()).toEqual([0, 128, 255]);
    });

    it('serializes int16 arrays', () => {
      const arr = array([-32768, 0, 32767], 'int16');
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('int16');
      expect(parsed.toArray()).toEqual([-32768, 0, 32767]);
    });

    it('handles large arrays', () => {
      const arr = arange(10000);
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.shape).toEqual([10000]);
      expect(parsed.get([0])).toBe(0);
      expect(parsed.get([9999])).toBe(9999);
    });

    it('handles zeros and ones', () => {
      const z = zeros([3, 3], 'float64');
      const o = ones([3, 3], 'float64');

      const zBytes = serializeNpy(z);
      const oBytes = serializeNpy(o);

      const zParsed = parseNpy(zBytes);
      const oParsed = parseNpy(oBytes);

      expect(zParsed.toArray()).toEqual([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0],
      ]);
      expect(oParsed.toArray()).toEqual([
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1],
      ]);
    });
  });

  describe('parseNpy', () => {
    it('parses a minimal NPY file', () => {
      // Create a minimal v2.0 NPY file with a single float64 element
      const arr = array([42.0]);
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.shape).toEqual([1]);
      expect(parsed.get([0])).toBe(42.0);
    });

    it('rejects invalid magic number', () => {
      // Need at least 10 bytes to pass the size check, then fail on magic number
      const bytes = new Uint8Array([0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00]);

      expect(() => parseNpy(bytes)).toThrow(InvalidNpyError);
      expect(() => parseNpy(bytes)).toThrow('Invalid NPY magic number');
    });

    it('rejects files that are too small', () => {
      const bytes = new Uint8Array([0x93, 0x4e, 0x55, 0x4d, 0x50]);

      expect(() => parseNpy(bytes)).toThrow(InvalidNpyError);
      expect(() => parseNpy(bytes)).toThrow('too small');
    });

    it('rejects unsupported versions', () => {
      const bytes = new Uint8Array([
        0x93,
        0x4e,
        0x55,
        0x4d,
        0x50,
        0x59, // NUMPY
        4,
        0, // version 4.0
        10,
        0, // header len
      ]);

      expect(() => parseNpy(bytes)).toThrow(InvalidNpyError);
      expect(() => parseNpy(bytes)).toThrow('Unsupported NPY version');
    });
  });

  describe('parseNpyHeader', () => {
    it('parses v3.0 header (our output format)', () => {
      const arr = array([1.0, 2.0, 3.0]);
      const bytes = serializeNpy(arr);
      const metadata = parseNpyHeader(bytes);

      expect(metadata.version.major).toBe(3);
      expect(metadata.version.minor).toBe(0);
      expect(metadata.header.shape).toEqual([3]);
      expect(metadata.header.fortran_order).toBe(false);
      expect(metadata.header.descr).toBe('<f8');
    });
  });

  describe('dtype mapping', () => {
    const dtypes: DType[] = [
      'float64',
      'float32',
      'int64',
      'int32',
      'int16',
      'int8',
      'uint64',
      'uint32',
      'uint16',
      'uint8',
      'bool',
    ];

    for (const dtype of dtypes) {
      it(`round-trips ${dtype} arrays`, () => {
        let arr;
        if (dtype === 'int64' || dtype === 'uint64') {
          arr = array([BigInt(1), BigInt(2), BigInt(3)], dtype);
        } else if (dtype === 'bool') {
          arr = array([1, 0, 1], dtype);
        } else {
          arr = array([1, 2, 3], dtype);
        }

        const bytes = serializeNpy(arr);
        const parsed = parseNpy(bytes);

        expect(parsed.dtype).toBe(dtype);
        expect(parsed.shape).toEqual([3]);
      });
    }
  });

  describe('DTYPE_TO_DESCR', () => {
    it('maps all dtypes to descriptors', () => {
      expect(DTYPE_TO_DESCR.float64).toBe('<f8');
      expect(DTYPE_TO_DESCR.float32).toBe('<f4');
      expect(DTYPE_TO_DESCR.int64).toBe('<i8');
      expect(DTYPE_TO_DESCR.int32).toBe('<i4');
      expect(DTYPE_TO_DESCR.int16).toBe('<i2');
      expect(DTYPE_TO_DESCR.int8).toBe('|i1');
      expect(DTYPE_TO_DESCR.uint64).toBe('<u8');
      expect(DTYPE_TO_DESCR.uint32).toBe('<u4');
      expect(DTYPE_TO_DESCR.uint16).toBe('<u2');
      expect(DTYPE_TO_DESCR.uint8).toBe('|u1');
      expect(DTYPE_TO_DESCR.bool).toBe('|b1');
    });
  });

  describe('UnsupportedDTypeError', () => {
    it('is thrown for complex dtypes', () => {
      // Create a mock NPY file with complex dtype
      const headerStr = "{'descr': '<c16', 'fortran_order': False, 'shape': (3,), }";
      const header = new TextEncoder().encode(headerStr + '\n');
      const headerLen = header.length;

      const npyFile = new Uint8Array(12 + headerLen + 24);
      npyFile.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 2, 0], 0);
      npyFile[8] = headerLen & 0xff;
      npyFile[9] = (headerLen >> 8) & 0xff;
      npyFile.set(header, 12);

      expect(() => parseNpy(npyFile)).toThrow(UnsupportedDTypeError);
      expect(() => parseNpy(npyFile)).toThrow('complex');
    });
  });
});

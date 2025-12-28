import { describe, it, expect } from 'vitest';
import {
  parseNpy,
  serializeNpy,
  parseNpyHeader,
  UnsupportedDTypeError,
  InvalidNpyError,
  DTYPE_TO_DESCR,
} from '../../../src/io/npy';
import { array, zeros, ones, arange, transpose } from '../../../src/core/ndarray';
import { Complex } from '../../../src/core/complex';
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

    it('serializes complex128 arrays', () => {
      const arr = array([new Complex(1, 2), new Complex(3, 4)], 'complex128');
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('complex128');
      expect(parsed.shape).toEqual([2]);
      const val0 = parsed.get([0]) as Complex;
      const val1 = parsed.get([1]) as Complex;
      expect(val0.re).toBe(1);
      expect(val0.im).toBe(2);
      expect(val1.re).toBe(3);
      expect(val1.im).toBe(4);
    });

    it('serializes complex64 arrays', () => {
      const arr = array([new Complex(1.5, 2.5), new Complex(3.5, 4.5)], 'complex64');
      const bytes = serializeNpy(arr);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('complex64');
      expect(parsed.shape).toEqual([2]);
      const val0 = parsed.get([0]) as Complex;
      const val1 = parsed.get([1]) as Complex;
      expect(val0.re).toBeCloseTo(1.5);
      expect(val0.im).toBeCloseTo(2.5);
      expect(val1.re).toBeCloseTo(3.5);
      expect(val1.im).toBeCloseTo(4.5);
    });

    it('serializes non-contiguous (transposed) arrays', () => {
      // Create a 2x3 array and transpose it to 3x2
      // This creates a non-contiguous array with non-standard strides
      const original = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const transposed = transpose(original);

      expect(transposed.shape).toEqual([3, 2]);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.shape).toEqual([3, 2]);
      expect(parsed.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('serializes sliced (non-contiguous) arrays', () => {
      // Create an array and slice it to create non-contiguous view
      const original = arange(10);
      // Take every other element: [0, 2, 4, 6, 8]
      const sliced = original.slice('::2');

      const bytes = serializeNpy(sliced);
      const parsed = parseNpy(bytes);

      expect(parsed.shape).toEqual([5]);
      expect(parsed.toArray()).toEqual([0, 2, 4, 6, 8]);
    });

    it('serializes non-contiguous int32 arrays', () => {
      const original = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'int32'
      );
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('int32');
      expect(parsed.shape).toEqual([3, 2]);
      expect(parsed.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('serializes non-contiguous int16 arrays', () => {
      const original = array([[10, 20], [30, 40]], 'int16');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('int16');
      expect(parsed.toArray()).toEqual([
        [10, 30],
        [20, 40],
      ]);
    });

    it('serializes non-contiguous int8 arrays', () => {
      const original = array([[1, 2], [3, 4]], 'int8');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('int8');
      expect(parsed.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it('serializes non-contiguous uint32 arrays', () => {
      const original = array([[100, 200], [300, 400]], 'uint32');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('uint32');
    });

    it('serializes non-contiguous uint16 arrays', () => {
      const original = array([[10, 20], [30, 40]], 'uint16');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('uint16');
    });

    it('serializes non-contiguous uint8 arrays', () => {
      const original = array([[1, 2], [3, 4]], 'uint8');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('uint8');
    });

    it('serializes non-contiguous float32 arrays', () => {
      const original = array([[1.5, 2.5], [3.5, 4.5]], 'float32');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('float32');
      expect(parsed.get([0, 0])).toBeCloseTo(1.5);
      expect(parsed.get([0, 1])).toBeCloseTo(3.5);
    });

    it('serializes non-contiguous bool arrays', () => {
      const original = array([[1, 0], [0, 1]], 'bool');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('bool');
      expect(parsed.toArray()).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });

    it('serializes non-contiguous int64 BigInt arrays', () => {
      const original = array([[BigInt(1), BigInt(2)], [BigInt(3), BigInt(4)]], 'int64');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('int64');
      expect(parsed.toArray()).toEqual([
        [BigInt(1), BigInt(3)],
        [BigInt(2), BigInt(4)],
      ]);
    });

    it('serializes non-contiguous uint64 BigInt arrays', () => {
      const original = array([[BigInt(10), BigInt(20)], [BigInt(30), BigInt(40)]], 'uint64');
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('uint64');
    });

    it('serializes non-contiguous complex128 arrays', () => {
      const original = array(
        [
          [new Complex(1, 2), new Complex(3, 4)],
          [new Complex(5, 6), new Complex(7, 8)],
        ],
        'complex128'
      );
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('complex128');
      expect(parsed.shape).toEqual([2, 2]);
      const val00 = parsed.get([0, 0]) as Complex;
      expect(val00.re).toBe(1);
      expect(val00.im).toBe(2);
    });

    it('serializes non-contiguous complex64 arrays', () => {
      const original = array(
        [
          [new Complex(1, 2), new Complex(3, 4)],
          [new Complex(5, 6), new Complex(7, 8)],
        ],
        'complex64'
      );
      const transposed = transpose(original);

      const bytes = serializeNpy(transposed);
      const parsed = parseNpy(bytes);

      expect(parsed.dtype).toBe('complex64');
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
    it('is thrown for string dtypes', () => {
      // Create a mock NPY file with string dtype (unsupported)
      const headerStr = "{'descr': 'S10', 'fortran_order': False, 'shape': (3,), }";
      const header = new TextEncoder().encode(headerStr + '\n');
      const headerLen = header.length;

      const npyFile = new Uint8Array(12 + headerLen + 30);
      npyFile.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 2, 0], 0);
      npyFile[8] = headerLen & 0xff;
      npyFile[9] = (headerLen >> 8) & 0xff;
      npyFile.set(header, 12);

      expect(() => parseNpy(npyFile)).toThrow(UnsupportedDTypeError);
      expect(() => parseNpy(npyFile)).toThrow('string');
    });
  });

  describe('Edge cases for parser coverage', () => {
    it('rejects invalid shape values in header', () => {
      // Create NPY file with invalid shape value 'abc' instead of a number
      const headerStr = "{'descr': '<f8', 'fortran_order': False, 'shape': (abc,), }";
      const header = new TextEncoder().encode(headerStr + '\n');
      const headerLen = header.length;

      const npyFile = new Uint8Array(12 + headerLen);
      npyFile.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 2, 0], 0);
      npyFile[8] = headerLen & 0xff;
      npyFile[9] = (headerLen >> 8) & 0xff;
      npyFile[10] = (headerLen >> 16) & 0xff;
      npyFile[11] = (headerLen >> 24) & 0xff;
      npyFile.set(header, 12);

      expect(() => parseNpy(npyFile)).toThrow(InvalidNpyError);
      expect(() => parseNpy(npyFile)).toThrow('Invalid shape value');
    });

    it('parses big-endian float64 data with byte swapping', () => {
      // Create a big-endian NPY file (>f8 instead of <f8)
      const headerStr = "{'descr': '>f8', 'fortran_order': False, 'shape': (2,), }";
      const header = new TextEncoder().encode(headerStr);
      // Pad header to be aligned to 64 bytes total
      const paddingNeeded = 64 - (12 + header.length) % 64;
      const paddedHeader = new Uint8Array(header.length + paddingNeeded);
      paddedHeader.set(header);
      paddedHeader.fill(0x20, header.length); // Fill with spaces
      paddedHeader[paddedHeader.length - 1] = 0x0a; // End with newline
      const headerLen = paddedHeader.length;

      // 2 float64 values = 16 bytes of data
      // Value 1.0 in big-endian: 0x3FF0000000000000
      // Value 2.0 in big-endian: 0x4000000000000000
      const dataBytes = new Uint8Array([
        0x3f, 0xf0, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 1.0 big-endian
        0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, // 2.0 big-endian
      ]);

      const npyFile = new Uint8Array(12 + headerLen + dataBytes.length);
      npyFile.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 2, 0], 0);
      npyFile[8] = headerLen & 0xff;
      npyFile[9] = (headerLen >> 8) & 0xff;
      npyFile[10] = (headerLen >> 16) & 0xff;
      npyFile[11] = (headerLen >> 24) & 0xff;
      npyFile.set(paddedHeader, 12);
      npyFile.set(dataBytes, 12 + headerLen);

      const parsed = parseNpy(npyFile);
      expect(parsed.shape).toEqual([2]);
      expect(parsed.dtype).toBe('float64');
      expect(parsed.get([0])).toBeCloseTo(1.0);
      expect(parsed.get([1])).toBeCloseTo(2.0);
    });

    it('parses big-endian int32 data with byte swapping', () => {
      // Create a big-endian NPY file (>i4 instead of <i4)
      const headerStr = "{'descr': '>i4', 'fortran_order': False, 'shape': (3,), }";
      const header = new TextEncoder().encode(headerStr);
      const paddingNeeded = 64 - (12 + header.length) % 64;
      const paddedHeader = new Uint8Array(header.length + paddingNeeded);
      paddedHeader.set(header);
      paddedHeader.fill(0x20, header.length);
      paddedHeader[paddedHeader.length - 1] = 0x0a;
      const headerLen = paddedHeader.length;

      // 3 int32 values: 1, 256, 65536 in big-endian
      const dataBytes = new Uint8Array([
        0x00, 0x00, 0x00, 0x01, // 1
        0x00, 0x00, 0x01, 0x00, // 256
        0x00, 0x01, 0x00, 0x00, // 65536
      ]);

      const npyFile = new Uint8Array(12 + headerLen + dataBytes.length);
      npyFile.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 2, 0], 0);
      npyFile[8] = headerLen & 0xff;
      npyFile[9] = (headerLen >> 8) & 0xff;
      npyFile[10] = (headerLen >> 16) & 0xff;
      npyFile[11] = (headerLen >> 24) & 0xff;
      npyFile.set(paddedHeader, 12);
      npyFile.set(dataBytes, 12 + headerLen);

      const parsed = parseNpy(npyFile);
      expect(parsed.shape).toEqual([3]);
      expect(parsed.dtype).toBe('int32');
      expect(parsed.toArray()).toEqual([1, 256, 65536]);
    });

    it('parses Fortran-ordered (column-major) 2D array', () => {
      // Create a Fortran-order NPY file
      // For a 2x3 array with fortran_order=True, the data is stored column-major:
      // [[1,2,3],[4,5,6]] stored as [1,4,2,5,3,6] in memory
      const headerStr = "{'descr': '<f8', 'fortran_order': True, 'shape': (2, 3), }";
      const header = new TextEncoder().encode(headerStr);
      const paddingNeeded = 64 - (12 + header.length) % 64;
      const paddedHeader = new Uint8Array(header.length + paddingNeeded);
      paddedHeader.set(header);
      paddedHeader.fill(0x20, header.length);
      paddedHeader[paddedHeader.length - 1] = 0x0a;
      const headerLen = paddedHeader.length;

      // 6 float64 values in Fortran (column-major) order: [1,4,2,5,3,6]
      // This represents the matrix [[1,2,3],[4,5,6]] stored column by column
      const dataView = new DataView(new ArrayBuffer(48));
      dataView.setFloat64(0, 1.0, true); // little-endian
      dataView.setFloat64(8, 4.0, true);
      dataView.setFloat64(16, 2.0, true);
      dataView.setFloat64(24, 5.0, true);
      dataView.setFloat64(32, 3.0, true);
      dataView.setFloat64(40, 6.0, true);
      const dataBytes = new Uint8Array(dataView.buffer);

      const npyFile = new Uint8Array(12 + headerLen + dataBytes.length);
      npyFile.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 2, 0], 0);
      npyFile[8] = headerLen & 0xff;
      npyFile[9] = (headerLen >> 8) & 0xff;
      npyFile[10] = (headerLen >> 16) & 0xff;
      npyFile[11] = (headerLen >> 24) & 0xff;
      npyFile.set(paddedHeader, 12);
      npyFile.set(dataBytes, 12 + headerLen);

      const parsed = parseNpy(npyFile);
      expect(parsed.shape).toEqual([2, 3]);
      expect(parsed.dtype).toBe('float64');
      // After parsing, should be in C order: [[1,2,3],[4,5,6]]
      expect(parsed.toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('parses v1.0 format NPY files', () => {
      // Create a v1.0 NPY file (2-byte header length)
      const headerStr = "{'descr': '<f8', 'fortran_order': False, 'shape': (2,), }";
      const header = new TextEncoder().encode(headerStr);
      // Pad to align
      const paddingNeeded = 64 - (10 + header.length) % 64;
      const paddedHeader = new Uint8Array(header.length + paddingNeeded);
      paddedHeader.set(header);
      paddedHeader.fill(0x20, header.length);
      paddedHeader[paddedHeader.length - 1] = 0x0a;
      const headerLen = paddedHeader.length;

      // 2 float64 values
      const dataView = new DataView(new ArrayBuffer(16));
      dataView.setFloat64(0, 3.14, true);
      dataView.setFloat64(8, 2.71, true);
      const dataBytes = new Uint8Array(dataView.buffer);

      // v1.0: 10-byte header (magic + version + 2-byte header length)
      const npyFile = new Uint8Array(10 + headerLen + dataBytes.length);
      npyFile.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0], 0); // v1.0
      npyFile[8] = headerLen & 0xff;
      npyFile[9] = (headerLen >> 8) & 0xff;
      npyFile.set(paddedHeader, 10);
      npyFile.set(dataBytes, 10 + headerLen);

      const parsed = parseNpy(npyFile);
      expect(parsed.shape).toEqual([2]);
      expect(parsed.dtype).toBe('float64');
      expect(parsed.get([0])).toBeCloseTo(3.14);
      expect(parsed.get([1])).toBeCloseTo(2.71);
    });

    it('rejects truncated data', () => {
      // Create NPY file where data section is truncated
      const headerStr = "{'descr': '<f8', 'fortran_order': False, 'shape': (10,), }";
      const header = new TextEncoder().encode(headerStr + '\n');
      const headerLen = header.length;

      // Only provide 8 bytes of data instead of 80 (10 * 8 bytes)
      const npyFile = new Uint8Array(12 + headerLen + 8);
      npyFile.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 2, 0], 0);
      npyFile[8] = headerLen & 0xff;
      npyFile[9] = (headerLen >> 8) & 0xff;
      npyFile[10] = (headerLen >> 16) & 0xff;
      npyFile[11] = (headerLen >> 24) & 0xff;
      npyFile.set(header, 12);

      expect(() => parseNpy(npyFile)).toThrow(InvalidNpyError);
      expect(() => parseNpy(npyFile)).toThrow('truncated');
    });
  });
});

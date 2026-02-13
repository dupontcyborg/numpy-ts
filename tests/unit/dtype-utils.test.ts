/**
 * Unit tests for dtype utility functions
 * Tests the low-level dtype module functions directly
 */

import { describe, it, expect } from 'vitest';
import {
  inferDType,
  promoteDTypes,
  castValue,
  isValidDType,
  getDTypeSize,
  getTypedArrayConstructor,
  isIntegerDType,
  isFloatDType,
  isBigIntDType,
  toStdlibDType,
} from '../../src/common/dtype';

describe('dtype utility functions', () => {
  describe('inferDType()', () => {
    it('infers int64 from bigint', () => {
      expect(inferDType(BigInt(42))).toBe('int64');
      expect(inferDType(BigInt(0))).toBe('int64');
      expect(inferDType(BigInt(-100))).toBe('int64');
    });

    it('infers bool from boolean', () => {
      expect(inferDType(true)).toBe('bool');
      expect(inferDType(false)).toBe('bool');
    });

    it('infers float64 from floating point numbers', () => {
      expect(inferDType(3.14)).toBe('float64');
      expect(inferDType(0.5)).toBe('float64');
      expect(inferDType(-2.7)).toBe('float64');
    });

    it('infers appropriate integer types from integer ranges', () => {
      // uint8 range: 0-255
      expect(inferDType(0)).toBe('uint8');
      expect(inferDType(255)).toBe('uint8');
      expect(inferDType(100)).toBe('uint8');

      // int8 range: -128-127 (negative in range)
      expect(inferDType(-128)).toBe('int8');
      expect(inferDType(-1)).toBe('int8');
      expect(inferDType(127)).toBe('uint8'); // Positive, so uint8

      // uint16 range: 256-65535
      expect(inferDType(256)).toBe('uint16');
      expect(inferDType(65535)).toBe('uint16');
      expect(inferDType(1000)).toBe('uint16');

      // int16 range: -32768 to -129
      expect(inferDType(-129)).toBe('int16');
      expect(inferDType(-32768)).toBe('int16');

      // uint32 range: 65536-4294967295
      expect(inferDType(65536)).toBe('uint32');
      expect(inferDType(4294967295)).toBe('uint32');
      expect(inferDType(1000000)).toBe('uint32');

      // int32 range: -2147483648 to -32769
      expect(inferDType(-32769)).toBe('int32');
      expect(inferDType(-2147483648)).toBe('int32');

      // Large integers beyond int32 range fallback to float64
      expect(inferDType(4294967296)).toBe('float64');
      expect(inferDType(Number.MAX_SAFE_INTEGER)).toBe('float64');
      expect(inferDType(-2147483649)).toBe('float64');
    });

    it('infers float64 for unknown types', () => {
      expect(inferDType(undefined)).toBe('float64');
      expect(inferDType(null)).toBe('float64');
      expect(inferDType('string')).toBe('float64');
      expect(inferDType({})).toBe('float64');
      expect(inferDType([])).toBe('float64');
    });
  });

  describe('promoteDTypes()', () => {
    it('returns same dtype when both are identical', () => {
      expect(promoteDTypes('float64', 'float64')).toBe('float64');
      expect(promoteDTypes('int32', 'int32')).toBe('int32');
      expect(promoteDTypes('uint8', 'uint8')).toBe('uint8');
    });

    it('promotes bool to other type', () => {
      expect(promoteDTypes('bool', 'float64')).toBe('float64');
      expect(promoteDTypes('int32', 'bool')).toBe('int32');
      expect(promoteDTypes('bool', 'uint8')).toBe('uint8');
    });

    it('promotes to float64 when one operand is float64', () => {
      expect(promoteDTypes('float64', 'float32')).toBe('float64');
      expect(promoteDTypes('float64', 'int32')).toBe('float64');
      expect(promoteDTypes('uint8', 'float64')).toBe('float64');
    });

    it('promotes float32 with small integers to float32', () => {
      expect(promoteDTypes('float32', 'int8')).toBe('float32');
      expect(promoteDTypes('float32', 'uint8')).toBe('float32');
      expect(promoteDTypes('float32', 'int16')).toBe('float32');
      expect(promoteDTypes('float32', 'uint16')).toBe('float32');
      expect(promoteDTypes('int8', 'float32')).toBe('float32');
      expect(promoteDTypes('uint16', 'float32')).toBe('float32');
    });

    it('promotes float32 with large integers to float64', () => {
      // float32 has 24-bit mantissa, can't safely hold all int32/int64 values
      expect(promoteDTypes('float32', 'int32')).toBe('float64');
      expect(promoteDTypes('float32', 'uint32')).toBe('float64');
      expect(promoteDTypes('float32', 'int64')).toBe('float64');
      expect(promoteDTypes('float32', 'uint64')).toBe('float64');
      expect(promoteDTypes('int32', 'float32')).toBe('float64');
      expect(promoteDTypes('uint64', 'float32')).toBe('float64');
    });

    it('promotes int64 + uint64 to float64', () => {
      expect(promoteDTypes('int64', 'uint64')).toBe('float64');
      expect(promoteDTypes('uint64', 'int64')).toBe('float64');
    });

    it('promotes signed and unsigned of same size to larger signed', () => {
      expect(promoteDTypes('int8', 'uint8')).toBe('int16');
      expect(promoteDTypes('uint8', 'int8')).toBe('int16');
      expect(promoteDTypes('int16', 'uint16')).toBe('int32');
      expect(promoteDTypes('uint16', 'int16')).toBe('int32');
      expect(promoteDTypes('int32', 'uint32')).toBe('int64');
      expect(promoteDTypes('uint32', 'int32')).toBe('int64');
    });

    it('promotes same signedness to larger size', () => {
      // Signed integers
      expect(promoteDTypes('int8', 'int16')).toBe('int16');
      expect(promoteDTypes('int8', 'int32')).toBe('int32');
      expect(promoteDTypes('int16', 'int64')).toBe('int64');

      // Unsigned integers
      expect(promoteDTypes('uint8', 'uint16')).toBe('uint16');
      expect(promoteDTypes('uint8', 'uint32')).toBe('uint32');
      expect(promoteDTypes('uint16', 'uint64')).toBe('uint64');
    });

    it('promotes different signedness with larger signed to signed', () => {
      expect(promoteDTypes('int16', 'uint8')).toBe('int16');
      expect(promoteDTypes('uint8', 'int16')).toBe('int16');
      expect(promoteDTypes('int32', 'uint16')).toBe('int32');
      expect(promoteDTypes('uint16', 'int32')).toBe('int32');
    });

    it('promotes different signedness with smaller signed conservatively', () => {
      expect(promoteDTypes('int8', 'uint16')).toBe('int32');
      expect(promoteDTypes('uint16', 'int8')).toBe('int32');
      expect(promoteDTypes('int8', 'uint32')).toBe('int64');
      expect(promoteDTypes('int16', 'uint32')).toBe('int64');
      expect(promoteDTypes('int8', 'uint64')).toBe('float64');
      expect(promoteDTypes('uint64', 'int16')).toBe('float64');
    });
  });

  describe('castValue()', () => {
    it('casts to BigInt for BigInt dtypes', () => {
      expect(castValue(42, 'int64')).toBe(BigInt(42));
      expect(castValue(100, 'uint64')).toBe(BigInt(100));
      expect(castValue(BigInt(50), 'int64')).toBe(BigInt(50));
    });

    it('casts bool to 0 or 1', () => {
      expect(castValue(true, 'bool')).toBe(1);
      expect(castValue(false, 'bool')).toBe(0);
      expect(castValue(1, 'bool')).toBe(1);
      expect(castValue(0, 'bool')).toBe(0);
    });

    it('casts to Number for numeric dtypes', () => {
      expect(castValue(42, 'float64')).toBe(42);
      expect(castValue(3.14, 'float32')).toBe(3.14);
      expect(castValue(100, 'int32')).toBe(100);
      expect(castValue(BigInt(50), 'float64')).toBe(50);
    });
  });

  describe('isValidDType()', () => {
    it('returns true for valid dtypes', () => {
      expect(isValidDType('float64')).toBe(true);
      expect(isValidDType('float32')).toBe(true);
      expect(isValidDType('int64')).toBe(true);
      expect(isValidDType('int32')).toBe(true);
      expect(isValidDType('int16')).toBe(true);
      expect(isValidDType('int8')).toBe(true);
      expect(isValidDType('uint64')).toBe(true);
      expect(isValidDType('uint32')).toBe(true);
      expect(isValidDType('uint16')).toBe(true);
      expect(isValidDType('uint8')).toBe(true);
      expect(isValidDType('bool')).toBe(true);
      expect(isValidDType('complex64')).toBe(true);
      expect(isValidDType('complex128')).toBe(true);
    });

    it('returns false for invalid dtypes', () => {
      expect(isValidDType('string')).toBe(false);
      expect(isValidDType('float16')).toBe(false);
      expect(isValidDType('int128')).toBe(false);
      expect(isValidDType('')).toBe(false);
      expect(isValidDType('Float64')).toBe(false); // Case sensitive
    });
  });

  describe('getDTypeSize()', () => {
    it('returns correct byte size for 8-byte types', () => {
      expect(getDTypeSize('float64')).toBe(8);
      expect(getDTypeSize('int64')).toBe(8);
      expect(getDTypeSize('uint64')).toBe(8);
    });

    it('returns correct byte size for 4-byte types', () => {
      expect(getDTypeSize('float32')).toBe(4);
      expect(getDTypeSize('int32')).toBe(4);
      expect(getDTypeSize('uint32')).toBe(4);
    });

    it('returns correct byte size for 2-byte types', () => {
      expect(getDTypeSize('int16')).toBe(2);
      expect(getDTypeSize('uint16')).toBe(2);
    });

    it('returns correct byte size for 1-byte types', () => {
      expect(getDTypeSize('int8')).toBe(1);
      expect(getDTypeSize('uint8')).toBe(1);
      expect(getDTypeSize('bool')).toBe(1);
    });

    it('throws error for unknown dtype', () => {
      expect(() => getDTypeSize('invalid' as any)).toThrow('Unknown dtype');
    });
  });

  describe('getTypedArrayConstructor()', () => {
    it('returns correct constructor for float types', () => {
      expect(getTypedArrayConstructor('float64')).toBe(Float64Array);
      expect(getTypedArrayConstructor('float32')).toBe(Float32Array);
    });

    it('returns correct constructor for signed integer types', () => {
      expect(getTypedArrayConstructor('int64')).toBe(BigInt64Array);
      expect(getTypedArrayConstructor('int32')).toBe(Int32Array);
      expect(getTypedArrayConstructor('int16')).toBe(Int16Array);
      expect(getTypedArrayConstructor('int8')).toBe(Int8Array);
    });

    it('returns correct constructor for unsigned integer types', () => {
      expect(getTypedArrayConstructor('uint64')).toBe(BigUint64Array);
      expect(getTypedArrayConstructor('uint32')).toBe(Uint32Array);
      expect(getTypedArrayConstructor('uint16')).toBe(Uint16Array);
      expect(getTypedArrayConstructor('uint8')).toBe(Uint8Array);
    });

    it('returns Uint8Array for bool', () => {
      expect(getTypedArrayConstructor('bool')).toBe(Uint8Array);
    });

    it('throws error for unknown dtype', () => {
      expect(() => getTypedArrayConstructor('invalid' as any)).toThrow('Unknown dtype');
    });
  });

  describe('isIntegerDType()', () => {
    it('returns true for integer dtypes', () => {
      expect(isIntegerDType('int64')).toBe(true);
      expect(isIntegerDType('int32')).toBe(true);
      expect(isIntegerDType('int16')).toBe(true);
      expect(isIntegerDType('int8')).toBe(true);
      expect(isIntegerDType('uint64')).toBe(true);
      expect(isIntegerDType('uint32')).toBe(true);
      expect(isIntegerDType('uint16')).toBe(true);
      expect(isIntegerDType('uint8')).toBe(true);
    });

    it('returns false for non-integer dtypes', () => {
      expect(isIntegerDType('float64')).toBe(false);
      expect(isIntegerDType('float32')).toBe(false);
      expect(isIntegerDType('bool')).toBe(false);
    });
  });

  describe('isFloatDType()', () => {
    it('returns true for float dtypes', () => {
      expect(isFloatDType('float64')).toBe(true);
      expect(isFloatDType('float32')).toBe(true);
    });

    it('returns false for non-float dtypes', () => {
      expect(isFloatDType('int32')).toBe(false);
      expect(isFloatDType('uint8')).toBe(false);
      expect(isFloatDType('bool')).toBe(false);
    });
  });

  describe('isBigIntDType()', () => {
    it('returns true for BigInt dtypes', () => {
      expect(isBigIntDType('int64')).toBe(true);
      expect(isBigIntDType('uint64')).toBe(true);
    });

    it('returns false for non-BigInt dtypes', () => {
      expect(isBigIntDType('float64')).toBe(false);
      expect(isBigIntDType('int32')).toBe(false);
      expect(isBigIntDType('uint32')).toBe(false);
      expect(isBigIntDType('bool')).toBe(false);
    });
  });

  describe('toStdlibDType()', () => {
    it('maps int64/uint64 to generic', () => {
      expect(toStdlibDType('int64')).toBe('generic');
      expect(toStdlibDType('uint64')).toBe('generic');
    });

    it('returns same dtype for other types', () => {
      expect(toStdlibDType('float64')).toBe('float64');
      expect(toStdlibDType('float32')).toBe('float32');
      expect(toStdlibDType('int32')).toBe('int32');
      expect(toStdlibDType('uint8')).toBe('uint8');
      expect(toStdlibDType('bool')).toBe('bool');
    });
  });
});

/**
 * Unit tests for type checking functions
 */

import { describe, it, expect } from 'vitest';
import { array } from '../../src/core/ndarray';
import {
  can_cast,
  common_type,
  result_type,
  min_scalar_type,
  issubdtype,
  typename,
  mintypecode,
} from '../../src/core/ndarray';

describe('Type Checking Functions', () => {
  describe('can_cast()', () => {
    describe('safe casting (default)', () => {
      it('allows same type casting', () => {
        expect(can_cast('float64', 'float64')).toBe(true);
        expect(can_cast('int32', 'int32')).toBe(true);
      });

      it('allows bool to any type', () => {
        expect(can_cast('bool', 'int32')).toBe(true);
        expect(can_cast('bool', 'float64')).toBe(true);
      });

      it('allows integer upcasting', () => {
        expect(can_cast('int8', 'int16')).toBe(true);
        expect(can_cast('int16', 'int32')).toBe(true);
        expect(can_cast('int32', 'int64')).toBe(true);
        expect(can_cast('uint8', 'uint16')).toBe(true);
      });

      it('allows float upcasting', () => {
        expect(can_cast('float32', 'float64')).toBe(true);
      });

      it('denies float downcasting', () => {
        expect(can_cast('float64', 'float32')).toBe(false);
      });

      it('allows int to float when safe', () => {
        expect(can_cast('int8', 'float32')).toBe(true);
        expect(can_cast('int16', 'float32')).toBe(true);
        expect(can_cast('int8', 'float64')).toBe(true);
        expect(can_cast('int32', 'float64')).toBe(true);
      });

      it('denies int32 to float32 (precision loss)', () => {
        expect(can_cast('int32', 'float32')).toBe(false);
      });

      it('allows any real to complex', () => {
        expect(can_cast('float64', 'complex128')).toBe(true);
        expect(can_cast('int32', 'complex64')).toBe(true);
      });

      it('handles NDArray input', () => {
        const arr = array([1, 2, 3], 'int32');
        expect(can_cast(arr, 'float64')).toBe(true);
        expect(can_cast(arr, 'int8')).toBe(false);
      });
    });

    describe('unsafe casting', () => {
      it('allows any casting with unsafe', () => {
        expect(can_cast('float64', 'int32', 'unsafe')).toBe(true);
        expect(can_cast('complex128', 'float32', 'unsafe')).toBe(true);
      });
    });

    describe('same_kind casting', () => {
      it('allows same kind types', () => {
        expect(can_cast('int32', 'int8', 'same_kind')).toBe(true);
        expect(can_cast('float64', 'float32', 'same_kind')).toBe(true);
      });

      it('allows int/uint mixing', () => {
        expect(can_cast('int32', 'uint32', 'same_kind')).toBe(true);
        expect(can_cast('uint16', 'int64', 'same_kind')).toBe(true);
      });

      it('allows float to complex', () => {
        expect(can_cast('float32', 'complex64', 'same_kind')).toBe(true);
      });

      it('denies int to float in same_kind', () => {
        expect(can_cast('int32', 'float32', 'same_kind')).toBe(false);
      });
    });

    describe('no casting', () => {
      it('only allows same type', () => {
        expect(can_cast('int32', 'int32', 'no')).toBe(true);
        expect(can_cast('int32', 'int64', 'no')).toBe(false);
      });
    });

    describe('equiv casting', () => {
      it('only allows same type', () => {
        expect(can_cast('float64', 'float64', 'equiv')).toBe(true);
        expect(can_cast('float64', 'float32', 'equiv')).toBe(false);
      });
    });
  });

  describe('common_type()', () => {
    it('returns float64 for no inputs', () => {
      expect(common_type()).toBe('float64');
    });

    it('returns float32 for small integer arrays', () => {
      const a = array([1, 2], 'int16');
      const b = array([3, 4], 'int8');
      expect(common_type(a, b)).toBe('float32');
    });

    it('returns float64 for float64 input', () => {
      const a = array([1.0, 2.0], 'float64');
      const b = array([3, 4], 'int16');
      expect(common_type(a, b)).toBe('float64');
    });

    it('returns float64 for 64-bit integers', () => {
      const a = array([1n, 2n], 'int64');
      expect(common_type(a)).toBe('float64');
    });

    it('returns complex type for complex input', () => {
      const a = array([1, 2], 'complex64');
      expect(common_type(a)).toBe('complex64');
    });

    it('returns complex128 when complex128 is present', () => {
      const a = array([1, 2], 'complex128');
      const b = array([3, 4], 'float32');
      expect(common_type(a, b)).toBe('complex128');
    });
  });

  describe('result_type()', () => {
    it('returns float64 for no inputs', () => {
      expect(result_type()).toBe('float64');
    });

    it('promotes bool to other type', () => {
      expect(result_type('bool', 'int32')).toBe('int32');
    });

    it('promotes integers correctly', () => {
      expect(result_type('int8', 'int16')).toBe('int16');
      expect(result_type('int32', 'uint32')).toBe('int64');
    });

    it('promotes float with integer', () => {
      expect(result_type('float32', 'int16')).toBe('float32');
      expect(result_type('float32', 'int32')).toBe('float64');
    });

    it('handles NDArray inputs', () => {
      const a = array([1, 2], 'int8');
      expect(result_type(a, 'float32')).toBe('float32');
    });

    it('promotes complex correctly', () => {
      expect(result_type('complex64', 'float64')).toBe('complex128');
      expect(result_type('complex64', 'float32')).toBe('complex64');
    });
  });

  describe('min_scalar_type()', () => {
    it('returns bool for booleans', () => {
      expect(min_scalar_type(true)).toBe('bool');
      expect(min_scalar_type(false)).toBe('bool');
    });

    it('returns uint8 for small positive integers', () => {
      expect(min_scalar_type(0)).toBe('uint8');
      expect(min_scalar_type(10)).toBe('uint8');
      expect(min_scalar_type(255)).toBe('uint8');
    });

    it('returns uint16 for larger positive integers', () => {
      expect(min_scalar_type(256)).toBe('uint16');
      expect(min_scalar_type(65535)).toBe('uint16');
    });

    it('returns int8 for small negative integers', () => {
      expect(min_scalar_type(-1)).toBe('int8');
      expect(min_scalar_type(-128)).toBe('int8');
    });

    it('returns int16 for medium negative integers', () => {
      expect(min_scalar_type(-129)).toBe('int16');
      expect(min_scalar_type(-32768)).toBe('int16');
    });

    it('returns float64 for floats', () => {
      expect(min_scalar_type(3.14)).toBe('float64');
      expect(min_scalar_type(Infinity)).toBe('float64');
      expect(min_scalar_type(NaN)).toBe('float64');
    });

    it('handles bigint values', () => {
      expect(min_scalar_type(10n)).toBe('uint8');
      expect(min_scalar_type(1000n)).toBe('uint16');
      expect(min_scalar_type(-100n)).toBe('int8');
    });
  });

  describe('issubdtype()', () => {
    it('matches same types', () => {
      expect(issubdtype('int32', 'int32')).toBe(true);
      expect(issubdtype('float64', 'float64')).toBe(true);
    });

    it('returns false for different specific types', () => {
      // In NumPy, specific types are only subtypes of themselves
      expect(issubdtype('int8', 'int64')).toBe(false);
      expect(issubdtype('int64', 'int8')).toBe(false);
      expect(issubdtype('float32', 'float64')).toBe(false);
    });

    it('handles category strings', () => {
      expect(issubdtype('int32', 'integer')).toBe(true);
      expect(issubdtype('uint8', 'integer')).toBe(true); // All integers
      expect(issubdtype('uint8', 'unsignedinteger')).toBe(true);
      expect(issubdtype('int32', 'signedinteger')).toBe(true);
      expect(issubdtype('float32', 'floating')).toBe(true);
      expect(issubdtype('complex64', 'complexfloating')).toBe(true);
      expect(issubdtype('float64', 'inexact')).toBe(true);
      expect(issubdtype('int32', 'number')).toBe(true);
    });

    it('handles NDArray input', () => {
      const arr = array([1, 2, 3], 'int32');
      expect(issubdtype(arr, 'integer')).toBe(true);
      expect(issubdtype(arr, 'floating')).toBe(false);
    });
  });

  describe('typename()', () => {
    it('returns dtype name', () => {
      expect(typename('float64')).toBe('float64');
      expect(typename('int32')).toBe('int32');
      expect(typename('complex128')).toBe('complex128');
      expect(typename('bool')).toBe('bool');
    });
  });

  describe('mintypecode()', () => {
    it('returns minimum type character', () => {
      expect(mintypecode('ff')).toBe('f'); // float32 + float32 = float32
      expect(mintypecode('fd')).toBe('d'); // float32 + float64 = float64
      expect(mintypecode('if')).toBe('d'); // int32 + float32 = float64 (safe promotion)
    });

    it('handles complex types', () => {
      expect(mintypecode('FF')).toBe('F'); // complex64
      expect(mintypecode('FD')).toBe('D'); // complex128
    });

    it('returns default for empty input', () => {
      expect(mintypecode('')).toBe('d');
      expect(mintypecode('', 'GDFgdf', 'f')).toBe('f');
    });

    it('respects typeset filter', () => {
      expect(mintypecode('bb', 'fd')).toBe('f'); // bool+bool but only float allowed
    });
  });
});

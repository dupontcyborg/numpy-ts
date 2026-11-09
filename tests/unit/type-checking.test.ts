/**
 * Unit tests for type checking functions
 */

import { describe, expect, it } from 'vitest';
import {
  array,
  Complex,
  can_cast,
  common_type,
  isdtype,
  isfortran,
  isneginf,
  isposinf,
  isscalar,
  issubdtype,
  iterable,
  min_scalar_type,
  mintypecode,
  promote_types,
  result_type,
  typename,
} from '../../src';
import { promoteDTypeSequence } from '../../src/common/dtype';

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

      it('allows real to complex only when the component holds every value', () => {
        expect(can_cast('float64', 'complex128')).toBe(true);
        expect(can_cast('int16', 'complex64')).toBe(true);
        expect(can_cast('int32', 'complex64')).toBe(false);
        expect(can_cast('int32', 'complex128')).toBe(true);
        expect(can_cast('float64', 'complex64')).toBe(false);
      });

      it('denies signed to unsigned and int64 to float64 is safe', () => {
        expect(can_cast('int8', 'uint8')).toBe(false);
        expect(can_cast('uint8', 'int16')).toBe(true);
        expect(can_cast('uint8', 'int8')).toBe(false);
        expect(can_cast('int64', 'float64')).toBe(true);
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

      it('allows unsigned to signed but not signed to unsigned', () => {
        expect(can_cast('int32', 'uint32', 'same_kind')).toBe(false);
        expect(can_cast('uint64', 'int8', 'same_kind')).toBe(true);
        expect(can_cast('uint16', 'int64', 'same_kind')).toBe(true);
      });

      it('allows float to complex', () => {
        expect(can_cast('float32', 'complex64', 'same_kind')).toBe(true);
      });

      it('allows int to float and bool to anything, but not float to int', () => {
        expect(can_cast('int32', 'float32', 'same_kind')).toBe(true);
        expect(can_cast('bool', 'float16', 'same_kind')).toBe(true);
        expect(can_cast('float32', 'int64', 'same_kind')).toBe(false);
        expect(can_cast('complex64', 'float64', 'same_kind')).toBe(false);
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
    it('throws for no inputs', () => {
      expect(() => result_type()).toThrow('at least one array or dtype is required');
    });

    it('does not widen integers before joining them with a float', () => {
      expect(result_type('int8', 'uint16', 'float32')).toBe('float32');
      expect(result_type('int8', 'uint8', 'float16')).toBe('float16');
    });

    it('promotes bool to other type', () => {
      expect(result_type('bool', 'int32')).toBe('int32');
    });

    it('promotes integers correctly', () => {
      expect(result_type('int8', 'int16')).toBe('int16');
      expect(result_type('int32', 'uint32')).toBe('int64');
      expect(result_type('int16', 'uint8')).toBe('int16');
      expect(result_type('int64', 'uint64')).toBe('float64');
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

    it('picks the float width by magnitude', () => {
      expect(min_scalar_type(3.14)).toBe('float16');
      expect(min_scalar_type(Infinity)).toBe('float16');
      expect(min_scalar_type(NaN)).toBe('float16');
      expect(min_scalar_type(-0)).toBe('float16');
      expect(min_scalar_type(65000.5)).toBe('float32');
      expect(min_scalar_type(3.5e38)).toBe('float64');
      expect(min_scalar_type(1e300)).toBe('float64');
    });

    it('returns uint64 for integers above the uint32 range', () => {
      expect(min_scalar_type(4294967296)).toBe('uint64');
      expect(min_scalar_type(-2147483649)).toBe('int64');
      expect(min_scalar_type(2n ** 64n - 1n)).toBe('uint64');
      expect(min_scalar_type(-129n)).toBe('int16');
      expect(min_scalar_type(-32769n)).toBe('int32');
      expect(min_scalar_type(-2147483649n)).toBe('int64');
      expect(min_scalar_type(-(2n ** 63n))).toBe('int64');
    });

    it('throws for bigints outside the 64-bit range', () => {
      expect(() => min_scalar_type(2n ** 64n)).toThrow(RangeError);
      expect(() => min_scalar_type(-(2n ** 63n) - 1n)).toThrow(RangeError);
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
    it('describes type character codes', () => {
      expect(typename('d')).toBe('double precision');
      expect(typename('f')).toBe('single precision');
      expect(typename('S1')).toBe('character');
      expect(typename('?')).toBe('bool');
      expect(typename('G')).toBe('complex long double precision');
    });

    it('throws for unknown codes and dtype names', () => {
      expect(() => typename('float64')).toThrow(/unknown type code/);
      expect(() => typename('e')).toThrow(/unknown type code/);
      expect(() => typename('')).toThrow(/unknown type code/);
    });
  });

  describe('mintypecode()', () => {
    it('picks the largest code present from the typeset', () => {
      expect(mintypecode('ff')).toBe('f');
      expect(mintypecode('fd')).toBe('d');
      expect(mintypecode('FF')).toBe('F');
      expect(mintypecode('FD')).toBe('D');
    });

    it('ignores codes outside the typeset', () => {
      expect(mintypecode('if')).toBe('f');
      expect(mintypecode(['i', 'f'])).toBe('f');
      expect(mintypecode(['d', 'f', 'S'])).toBe('d');
    });

    it('returns D for single and double precision complex mixed', () => {
      expect(mintypecode(['F', 'd'])).toBe('D');
      expect(mintypecode('dF')).toBe('D');
    });

    it('returns default when nothing matches', () => {
      expect(mintypecode('')).toBe('d');
      expect(mintypecode([])).toBe('d');
      expect(mintypecode('', 'GDFgdf', 'f')).toBe('f');
      expect(mintypecode('abceh', 'GDFgdf', 'G')).toBe('G');
      expect(mintypecode('bb', 'fd')).toBe('d');
    });

    it('accepts the typeset as an array', () => {
      expect(mintypecode(['f', 'd'], ['f', 'd', 'D'])).toBe('d');
      expect(mintypecode(['f', 'd'], ['f'])).toBe('f');
    });

    it('reads an empty-string element as a unicode code, which is outside the default typeset', () => {
      expect(mintypecode(['', 'f'])).toBe('f');
    });

    it('throws when a matching code has no size ranking, as NumPy does', () => {
      expect(() => mintypecode(['x'], 'x')).toThrow(/not a known type code/);
    });
  });

  describe('isscalar()', () => {
    it('returns true for numbers, bigints, booleans, strings and Complex', () => {
      expect(isscalar(42)).toBe(true);
      expect(isscalar(3.14)).toBe(true);
      expect(isscalar(5n)).toBe(true);
      expect(isscalar(false)).toBe(true);
      expect(isscalar('x')).toBe(true);
      expect(isscalar(new Complex(1, 2))).toBe(true);
    });

    it('returns false for arrays and objects', () => {
      expect(isscalar([1, 2])).toBe(false);
      expect(isscalar(array([1, 2]))).toBe(false);
      expect(isscalar(array(5))).toBe(false);
      expect(isscalar({ re: 1, im: 2 })).toBe(false);
    });
  });

  describe('iterable()', () => {
    it('returns true for iterable objects', () => {
      expect(iterable([1, 2])).toBe(true);
    });

    it('returns false for non-iterables', () => {
      expect(iterable(42)).toBe(false);
    });
  });

  describe('isdtype()', () => {
    it('identifies kinds', () => {
      expect(isdtype('float64', 'real floating')).toBe(true);
      expect(isdtype('float16', 'real floating')).toBe(true);
      expect(isdtype('int32', 'integral')).toBe(true);
      expect(isdtype('uint8', 'signed integer')).toBe(false);
      expect(isdtype('bool', 'bool')).toBe(true);
      expect(isdtype('bool', 'numeric')).toBe(false);
      expect(isdtype('complex64', 'complex floating')).toBe(true);
    });

    it('accepts a dtype or an array of kinds', () => {
      expect(isdtype('float32', 'float32')).toBe(true);
      expect(isdtype('float32', ['float32', 'float64'])).toBe(true);
      expect(isdtype('int8', ['float32', 'unsigned integer'])).toBe(false);
      expect(isdtype('int8', [])).toBe(false);
    });

    it('rejects the single-letter kinds and unknown names', () => {
      expect(() => isdtype('float64', 'f')).toThrow(/not a known kind name/);
    });
  });

  describe('promote_types()', () => {
    it('promotes int32 and float32 to float64', () => {
      expect(promote_types('int32', 'float32')).toBe('float64');
      expect(promote_types('uint8', 'int8')).toBe('int16');
      expect(promote_types('int64', 'uint64')).toBe('float64');
      expect(promote_types('int16', 'float16')).toBe('float32');
      expect(promote_types('complex64', 'float64')).toBe('complex128');
    });
  });

  describe('isneginf()', () => {
    it('detects negative infinity', () => {
      const a = array([-Infinity, 0, Infinity]);
      const r = isneginf(a);
      expect(r.size).toBe(3);
    });
  });

  describe('isposinf()', () => {
    it('detects positive infinity', () => {
      const a = array([-Infinity, 0, Infinity]);
      const r = isposinf(a);
      expect(r.size).toBe(3);
    });
  });

  describe('isfortran()', () => {
    it('returns boolean for array', () => {
      const a = array([1, 2, 3]);
      expect(typeof isfortran(a)).toBe('boolean');
    });
  });
});

describe('promoteDTypeSequence()', () => {
  it('throws for an empty sequence', () => {
    expect(() => promoteDTypeSequence([])).toThrow('need at least one dtype');
  });

  it('matches a pairwise promotion for one or two dtypes', () => {
    expect(promoteDTypeSequence(['int8'])).toBe('int8');
    expect(promoteDTypeSequence(['int8', 'uint8'])).toBe('int16');
  });
});

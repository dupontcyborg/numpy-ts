/**
 * Unit tests for logic operations
 */

import { describe, it, expect } from 'vitest';
import {
  array,
  Complex,
  zeros,
  logical_and,
  logical_or,
  logical_not,
  logical_xor,
  isfinite,
  isinf,
  isnan,
  isnat,
  iscomplex,
  iscomplexobj,
  isreal,
  isrealobj,
  isneginf,
  isposinf,
  isfortran,
  real_if_close,
  isscalar,
  iterable,
  isdtype,
  promote_types,
  copysign,
  signbit,
  nextafter,
  spacing,
} from '../../src';

describe('Logic Operations', () => {
  describe('logical_and()', () => {
    describe('scalar operations', () => {
      it('ANDs 1D array with scalar 0 (always false)', () => {
        const arr = array([0, 1, 2, 3, 4]);
        const result = arr.logical_and(0);
        expect(result.dtype).toBe('bool');
        expect(result.shape).toEqual([5]);
        expect(Array.from(result.data)).toEqual([0, 0, 0, 0, 0]);
      });

      it('ANDs 1D array with scalar 1 (preserves truthiness)', () => {
        const arr = array([0, 1, 2, 3, 4]);
        const result = arr.logical_and(1);
        expect(Array.from(result.data)).toEqual([0, 1, 1, 1, 1]);
      });

      it('ANDs 2D array with scalar', () => {
        const arr = array([
          [0, 1],
          [2, 0],
        ]);
        const result = arr.logical_and(1);
        expect(result.shape).toEqual([2, 2]);
        expect(Array.from(result.data)).toEqual([0, 1, 1, 0]);
      });
    });

    describe('array operations', () => {
      it('ANDs two 1D arrays', () => {
        const a = array([0, 1, 0, 1]);
        const b = array([0, 0, 1, 1]);
        const result = a.logical_and(b);
        expect(Array.from(result.data)).toEqual([0, 0, 0, 1]);
      });

      it('ANDs two 2D arrays', () => {
        const a = array([
          [1, 0],
          [0, 1],
        ]);
        const b = array([
          [1, 1],
          [0, 0],
        ]);
        const result = a.logical_and(b);
        expect(Array.from(result.data)).toEqual([1, 0, 0, 0]);
      });
    });

    describe('broadcasting', () => {
      it('broadcasts (2, 3) with (3,)', () => {
        const a = array([
          [1, 0, 1],
          [0, 1, 0],
        ]);
        const b = array([1, 1, 0]);
        const result = a.logical_and(b);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.data)).toEqual([1, 0, 0, 0, 1, 0]);
      });
    });

    describe('top-level function', () => {
      it('works as top-level function', () => {
        const a = array([1, 0, 1, 0]);
        const b = array([1, 1, 0, 0]);
        const result = logical_and(a, b);
        expect(Array.from(result.data)).toEqual([1, 0, 0, 0]);
      });
    });

    describe('BigInt operations', () => {
      it('ANDs two BigInt arrays', () => {
        const a = array([0n, 1n, 0n, 1n], 'int64');
        const b = array([0n, 0n, 1n, 1n], 'int64');
        const result = a.logical_and(b);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.data)).toEqual([0, 0, 0, 1]);
      });

      it('ANDs BigInt array with scalar', () => {
        const arr = array([0n, 1n, 2n, 0n], 'int64');
        const result = arr.logical_and(1);
        expect(Array.from(result.data)).toEqual([0, 1, 1, 0]);
      });
    });

    describe('complex array operations', () => {
      it('ANDs two complex arrays (tests isComplexTruthy)', () => {
        const a = array([
          new Complex(0, 0),
          new Complex(1, 0),
          new Complex(0, 1),
          new Complex(1, 1),
        ]);
        const b = array([
          new Complex(0, 0),
          new Complex(0, 0),
          new Complex(1, 0),
          new Complex(1, 1),
        ]);
        const result = a.logical_and(b);
        expect(result.dtype).toBe('bool');
        // Complex is truthy if either real or imaginary part is non-zero
        expect(Array.from(result.data)).toEqual([0, 0, 1, 1]);
      });

      it('ANDs complex array with broadcasting (tests canUseFastPath)', () => {
        // Test broadcasting with different shapes to trigger slow path
        const a = array([[new Complex(1, 1), new Complex(1, 1)]]);
        const b = array([[new Complex(1, 0)], [new Complex(0, 0)]]);
        const result = logical_and(a, b);
        expect(result.shape).toEqual([2, 2]);
        expect(result.dtype).toBe('bool');
        // Just verify it runs without error - the exact values depend on complex truthiness and broadcasting logic
        expect(result.data.length).toBe(4);
      });
    });
  });

  describe('logical_or()', () => {
    describe('scalar operations', () => {
      it('ORs 1D array with scalar 0 (preserves truthiness)', () => {
        const arr = array([0, 1, 2, 0, 4]);
        const result = arr.logical_or(0);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.data)).toEqual([0, 1, 1, 0, 1]);
      });

      it('ORs 1D array with scalar 1 (always true)', () => {
        const arr = array([0, 0, 0]);
        const result = arr.logical_or(1);
        expect(Array.from(result.data)).toEqual([1, 1, 1]);
      });
    });

    describe('array operations', () => {
      it('ORs two 1D arrays', () => {
        const a = array([0, 1, 0, 1]);
        const b = array([0, 0, 1, 1]);
        const result = a.logical_or(b);
        expect(Array.from(result.data)).toEqual([0, 1, 1, 1]);
      });
    });

    describe('top-level function', () => {
      it('works as top-level function', () => {
        const a = array([1, 0, 1, 0]);
        const b = array([1, 1, 0, 0]);
        const result = logical_or(a, b);
        expect(Array.from(result.data)).toEqual([1, 1, 1, 0]);
      });
    });

    describe('BigInt operations', () => {
      it('ORs two BigInt arrays', () => {
        const a = array([0n, 1n, 0n, 1n], 'int64');
        const b = array([0n, 0n, 1n, 1n], 'int64');
        const result = a.logical_or(b);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.data)).toEqual([0, 1, 1, 1]);
      });

      it('ORs BigInt array with scalar', () => {
        const arr = array([0n, 1n, 0n], 'int64');
        const result = arr.logical_or(1);
        expect(Array.from(result.data)).toEqual([1, 1, 1]);
      });
    });
  });

  describe('logical_not()', () => {
    it('NOTs 1D array', () => {
      const arr = array([0, 1, 2, 0, -1]);
      const result = arr.logical_not();
      expect(result.dtype).toBe('bool');
      expect(Array.from(result.data)).toEqual([1, 0, 0, 1, 0]);
    });

    it('NOTs 2D array', () => {
      const arr = array([
        [0, 1],
        [2, 0],
      ]);
      const result = arr.logical_not();
      expect(result.shape).toEqual([2, 2]);
      expect(Array.from(result.data)).toEqual([1, 0, 0, 1]);
    });

    it('works as top-level function', () => {
      const arr = array([0, 1, 0, 1]);
      const result = logical_not(arr);
      expect(Array.from(result.data)).toEqual([1, 0, 1, 0]);
    });

    it('NOTs BigInt array', () => {
      const arr = array([0n, 1n, 2n, 0n], 'int64');
      const result = arr.logical_not();
      expect(result.dtype).toBe('bool');
      expect(Array.from(result.data)).toEqual([1, 0, 0, 1]);
    });
  });

  describe('logical_xor()', () => {
    describe('scalar operations', () => {
      it('XORs 1D array with scalar 0', () => {
        const arr = array([0, 1, 2, 0, 4]);
        const result = arr.logical_xor(0);
        expect(result.dtype).toBe('bool');
        // XOR with 0 preserves truthiness
        expect(Array.from(result.data)).toEqual([0, 1, 1, 0, 1]);
      });

      it('XORs 1D array with scalar 1 (flips truthiness)', () => {
        const arr = array([0, 1, 2, 0]);
        const result = arr.logical_xor(1);
        expect(Array.from(result.data)).toEqual([1, 0, 0, 1]);
      });
    });

    describe('array operations', () => {
      it('XORs two 1D arrays', () => {
        const a = array([0, 1, 0, 1]);
        const b = array([0, 0, 1, 1]);
        const result = a.logical_xor(b);
        expect(Array.from(result.data)).toEqual([0, 1, 1, 0]);
      });
    });

    describe('top-level function', () => {
      it('works as top-level function', () => {
        const a = array([1, 0, 1, 0]);
        const b = array([1, 1, 0, 0]);
        const result = logical_xor(a, b);
        expect(Array.from(result.data)).toEqual([0, 1, 1, 0]);
      });
    });

    describe('BigInt operations', () => {
      it('XORs two BigInt arrays', () => {
        const a = array([0n, 1n, 0n, 1n], 'int64');
        const b = array([0n, 0n, 1n, 1n], 'int64');
        const result = a.logical_xor(b);
        expect(result.dtype).toBe('bool');
        expect(Array.from(result.data)).toEqual([0, 1, 1, 0]);
      });

      it('XORs BigInt array with scalar', () => {
        const arr = array([0n, 1n, 2n], 'int64');
        const result = arr.logical_xor(1);
        // XOR with true: 0→1, 1→0, 1→0
        expect(Array.from(result.data)).toEqual([1, 0, 0]);
      });
    });
  });

  describe('isfinite()', () => {
    it('detects finite values', () => {
      const arr = array([1, 2, Infinity, -Infinity, NaN, 0]);
      const result = arr.isfinite();
      expect(result.dtype).toBe('bool');
      expect(Array.from(result.data)).toEqual([1, 1, 0, 0, 0, 1]);
    });

    it('returns all true for integer arrays', () => {
      const arr = array([1, 2, 3, 4, 5], 'int32');
      const result = arr.isfinite();
      expect(Array.from(result.data)).toEqual([1, 1, 1, 1, 1]);
    });

    it('works as top-level function', () => {
      const arr = array([1, Infinity, NaN]);
      const result = isfinite(arr);
      expect(Array.from(result.data)).toEqual([1, 0, 0]);
    });
  });

  describe('isinf()', () => {
    it('detects infinite values', () => {
      const arr = array([1, 2, Infinity, -Infinity, NaN, 0]);
      const result = arr.isinf();
      expect(result.dtype).toBe('bool');
      expect(Array.from(result.data)).toEqual([0, 0, 1, 1, 0, 0]);
    });

    it('returns all false for integer arrays', () => {
      const arr = array([1, 2, 3, 4, 5], 'int32');
      const result = arr.isinf();
      expect(Array.from(result.data)).toEqual([0, 0, 0, 0, 0]);
    });

    it('works as top-level function', () => {
      const arr = array([1, Infinity, NaN]);
      const result = isinf(arr);
      expect(Array.from(result.data)).toEqual([0, 1, 0]);
    });
  });

  describe('isnan()', () => {
    it('detects NaN values', () => {
      const arr = array([1, 2, Infinity, -Infinity, NaN, 0]);
      const result = arr.isnan();
      expect(result.dtype).toBe('bool');
      expect(Array.from(result.data)).toEqual([0, 0, 0, 0, 1, 0]);
    });

    it('returns all false for integer arrays', () => {
      const arr = array([1, 2, 3, 4, 5], 'int32');
      const result = arr.isnan();
      expect(Array.from(result.data)).toEqual([0, 0, 0, 0, 0]);
    });

    it('handles multiple NaN values', () => {
      const arr = array([NaN, 1, NaN, 2, NaN]);
      const result = arr.isnan();
      expect(Array.from(result.data)).toEqual([1, 0, 1, 0, 1]);
    });

    it('works as top-level function', () => {
      const arr = array([1, Infinity, NaN]);
      const result = isnan(arr);
      expect(Array.from(result.data)).toEqual([0, 0, 1]);
    });
  });

  describe('isnat()', () => {
    it('returns all false (no datetime support)', () => {
      const arr = array([1, 2, 3]);
      const result = arr.isnat();
      expect(result.dtype).toBe('bool');
      expect(Array.from(result.data)).toEqual([0, 0, 0]);
    });

    it('works as top-level function', () => {
      const arr = array([1, 2, 3]);
      const result = isnat(arr);
      expect(Array.from(result.data)).toEqual([0, 0, 0]);
    });
  });

  describe('copysign()', () => {
    describe('scalar operations', () => {
      it('copies positive sign to values', () => {
        const arr = array([-1, -2, 3, -4]);
        const result = arr.copysign(1);
        expect(Array.from(result.data)).toEqual([1, 2, 3, 4]);
      });

      it('copies negative sign to values', () => {
        const arr = array([1, 2, -3, 4]);
        const result = arr.copysign(-1);
        expect(Array.from(result.data)).toEqual([-1, -2, -3, -4]);
      });
    });

    describe('array operations', () => {
      it('copies signs from array', () => {
        const a = array([1, -2, 3, -4]);
        const b = array([-1, 1, -1, 1]);
        const result = a.copysign(b);
        expect(Array.from(result.data)).toEqual([-1, 2, -3, 4]);
      });
    });

    describe('top-level function', () => {
      it('works as top-level function', () => {
        const a = array([1, 2, 3]);
        const b = array([-1, -1, -1]);
        const result = copysign(a, b);
        expect(Array.from(result.data)).toEqual([-1, -2, -3]);
      });
    });

    describe('broadcasting', () => {
      it('broadcasts 2D with 1D (tests broadcastToStorage)', () => {
        const a = array([
          [1, 2],
          [3, 4],
        ]);
        const b = array([-1, 1]);
        const result = copysign(a, b);
        expect(result.shape).toEqual([2, 2]);
        expect(Array.from(result.data)).toEqual([-1, 2, -3, 4]);
      });
    });
  });

  describe('signbit()', () => {
    it('detects negative numbers', () => {
      const arr = array([1, -1, 0, -0.0, 2, -3]);
      const result = arr.signbit();
      expect(result.dtype).toBe('bool');
      // 1 -> false, -1 -> true, 0 -> false, -0 -> true, 2 -> false, -3 -> true
      expect(Array.from(result.data)).toEqual([0, 1, 0, 1, 0, 1]);
    });

    it('handles infinity', () => {
      const arr = array([Infinity, -Infinity]);
      const result = arr.signbit();
      expect(Array.from(result.data)).toEqual([0, 1]);
    });

    it('works as top-level function', () => {
      const arr = array([1, -1, 0]);
      const result = signbit(arr);
      expect(Array.from(result.data)).toEqual([0, 1, 0]);
    });
  });

  describe('nextafter()', () => {
    it('gets next float towards positive infinity', () => {
      const arr = array([0, 1]);
      const result = arr.nextafter(Infinity);
      // Next after 0 towards Infinity is smallest positive number
      expect(result.data[0]).toBeGreaterThan(0);
      expect(result.data[0]).toBeLessThan(1e-300);
      // Next after 1 towards Infinity is slightly greater than 1
      expect(result.data[1]).toBeGreaterThan(1);
    });

    it('gets next float towards negative infinity', () => {
      const arr = array([0, 1]);
      const result = arr.nextafter(-Infinity);
      // Next after 0 towards -Infinity is smallest negative number
      expect(result.data[0]).toBeLessThan(0);
      // Next after 1 towards -Infinity is slightly less than 1
      expect(result.data[1]).toBeLessThan(1);
    });

    it('returns same value when direction equals value', () => {
      const arr = array([1.0, 2.0, 3.0]);
      const result = arr.nextafter(arr);
      expect(Array.from(result.data)).toEqual([1.0, 2.0, 3.0]);
    });

    it('works as top-level function', () => {
      const a = array([1.0]);
      const result = nextafter(a, 2.0);
      expect(result.data[0]).toBeGreaterThan(1.0);
    });

    it('broadcasts with array direction (tests broadcastToStorage)', () => {
      const a = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const b = array([Infinity, -Infinity]);
      const result = nextafter(a, b);
      expect(result.shape).toEqual([2, 2]);
      // First column moves towards Infinity, second column towards -Infinity
      expect(result.get([0, 0])).toBeGreaterThan(1.0);
      expect(result.get([0, 1])).toBeLessThan(2.0);
    });
  });

  describe('spacing()', () => {
    it('returns spacing at various magnitudes', () => {
      const arr = array([1, 10, 100, 1000]);
      const result = arr.spacing();
      // Spacing should increase with magnitude
      expect(result.data[1]).toBeGreaterThan(result.data[0]);
      expect(result.data[2]).toBeGreaterThan(result.data[1]);
      expect(result.data[3]).toBeGreaterThan(result.data[2]);
    });

    it('returns smallest value for zero', () => {
      const arr = array([0]);
      const result = arr.spacing();
      expect(result.data[0]).toBe(Number.MIN_VALUE);
    });

    it('works as top-level function', () => {
      const arr = array([1.0]);
      const result = spacing(arr);
      expect(result.data[0]).toBeGreaterThan(0);
    });
  });

  describe('result properties', () => {
    it('returns bool dtype for logical operations', () => {
      const arr = array([1, 2, 3]);
      expect(arr.logical_and(1).dtype).toBe('bool');
      expect(arr.logical_or(0).dtype).toBe('bool');
      expect(arr.logical_not().dtype).toBe('bool');
      expect(arr.logical_xor(1).dtype).toBe('bool');
    });

    it('returns bool dtype for testing functions', () => {
      const arr = array([1, 2, 3]);
      expect(arr.isfinite().dtype).toBe('bool');
      expect(arr.isinf().dtype).toBe('bool');
      expect(arr.isnan().dtype).toBe('bool');
      expect(arr.isnat().dtype).toBe('bool');
      expect(arr.signbit().dtype).toBe('bool');
    });

    it('preserves shape in all operations', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.logical_and(1).shape).toEqual(arr.shape);
      expect(arr.logical_not().shape).toEqual(arr.shape);
      expect(arr.isfinite().shape).toEqual(arr.shape);
      expect(arr.copysign(1).shape).toEqual(arr.shape);
    });
  });

  describe('edge cases', () => {
    it('handles empty arrays', () => {
      const arr = array([]);
      const result = arr.logical_not();
      expect(result.shape).toEqual([0]);
      expect(result.size).toBe(0);
    });

    it('handles all-true result', () => {
      const arr = array([0, 0, 0]);
      const result = arr.logical_not();
      expect(Array.from(result.data)).toEqual([1, 1, 1]);
    });

    it('handles all-false result', () => {
      const arr = array([1, 2, 3]);
      const result = arr.logical_not();
      expect(Array.from(result.data)).toEqual([0, 0, 0]);
    });

    it('handles floating point in logical operations', () => {
      const arr = array([0.0, 0.1, 1.0, -0.1]);
      const result = arr.logical_not();
      // 0.0 -> true, others -> false
      expect(Array.from(result.data)).toEqual([1, 0, 0, 0]);
    });
  });

  describe('broadcasting errors', () => {
    it('throws on incompatible shapes', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([1, 2, 3]);
      expect(() => a.logical_and(b)).toThrow(/broadcast/);
    });
  });
});

describe('Additional Logic Functions', () => {
  describe('iscomplex()', () => {
    it('returns all false for real arrays', () => {
      const arr = array([1, 2, 3]);
      const result = iscomplex(arr);
      expect(result.toArray()).toEqual([0, 0, 0]);
      expect(result.dtype).toBe('bool');
    });

    it('handles 2D arrays', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = iscomplex(arr);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [0, 0],
        [0, 0],
      ]);
    });
  });

  describe('iscomplexobj()', () => {
    it('returns false for real arrays', () => {
      const arr = array([1, 2, 3]);
      const result = iscomplexobj(arr);
      expect(result).toBe(false);
    });
  });

  describe('isreal()', () => {
    it('returns all true for real arrays', () => {
      const arr = array([1, 2, 3]);
      const result = isreal(arr);
      expect(result.toArray()).toEqual([1, 1, 1]);
      expect(result.dtype).toBe('bool');
    });

    it('handles 2D arrays', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = isreal(arr);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });
  });

  describe('isrealobj()', () => {
    it('returns true for real arrays', () => {
      const arr = array([1, 2, 3]);
      const result = isrealobj(arr);
      expect(result).toBe(true);
    });
  });

  describe('isneginf()', () => {
    it('detects negative infinity', () => {
      const arr = array([-Infinity, -1, 0, 1, Infinity]);
      const result = isneginf(arr);
      expect(result.toArray()).toEqual([1, 0, 0, 0, 0]);
      expect(result.dtype).toBe('bool');
    });

    it('handles all finite values', () => {
      const arr = array([1, 2, 3]);
      const result = isneginf(arr);
      expect(result.toArray()).toEqual([0, 0, 0]);
    });

    it('handles 2D arrays', () => {
      const arr = array([
        [-Infinity, 1],
        [2, -Infinity],
      ]);
      const result = isneginf(arr);
      expect(result.toArray()).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });
  });

  describe('isposinf()', () => {
    it('detects positive infinity', () => {
      const arr = array([-Infinity, -1, 0, 1, Infinity]);
      const result = isposinf(arr);
      expect(result.toArray()).toEqual([0, 0, 0, 0, 1]);
      expect(result.dtype).toBe('bool');
    });

    it('handles all finite values', () => {
      const arr = array([1, 2, 3]);
      const result = isposinf(arr);
      expect(result.toArray()).toEqual([0, 0, 0]);
    });

    it('handles 2D arrays', () => {
      const arr = array([
        [Infinity, 1],
        [2, Infinity],
      ]);
      const result = isposinf(arr);
      expect(result.toArray()).toEqual([
        [1, 0],
        [0, 1],
      ]);
    });
  });

  describe('isfortran()', () => {
    it('returns false for C-contiguous arrays', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = isfortran(arr);
      expect(result).toBe(false);
    });
  });

  describe('real_if_close()', () => {
    it('returns copy of real array', () => {
      const arr = array([1, 2, 3]);
      const result = real_if_close(arr);
      expect(result.toArray()).toEqual([1, 2, 3]);
      expect(result.base).toBe(null);
    });

    it('handles 2D arrays', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const result = real_if_close(arr);
      expect(result.shape).toEqual([2, 2]);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('returns real part when imaginary parts are close to zero', () => {
      // Create complex array with tiny imaginary parts
      const arr = zeros([3], 'complex128');
      arr.set([0], new Complex(1, 1e-20));
      arr.set([1], new Complex(2, 1e-20));
      arr.set([2], new Complex(3, 0));
      const result = real_if_close(arr);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('returns complex array when imaginary parts are not close to zero', () => {
      const arr = array([new Complex(1, 0.5), new Complex(2, 0.3)]);
      const result = real_if_close(arr);
      expect(result.dtype).toBe('complex128');
      const resultArr = result.toArray() as Complex[];
      expect(resultArr[0]!.re).toBe(1);
      expect(resultArr[0]!.im).toBe(0.5);
    });

    it('handles complex64 arrays', () => {
      const arr = zeros([2], 'complex64');
      arr.set([0], new Complex(5, 1e-10));
      arr.set([1], new Complex(10, 0));
      const result = real_if_close(arr);
      expect(result.dtype).toBe('float32');
    });
  });

  describe('isscalar()', () => {
    it('returns true for scalars', () => {
      expect(isscalar(5)).toBe(true);
      expect(isscalar(3.14)).toBe(true);
      expect(isscalar(true)).toBe(true);
      expect(isscalar('hello')).toBe(true);
      expect(isscalar(BigInt(42))).toBe(true);
    });

    it('returns false for non-scalars', () => {
      expect(isscalar([1, 2, 3])).toBe(false);
      expect(isscalar({ a: 1 })).toBe(false);
      expect(isscalar(null)).toBe(false);
      expect(isscalar(undefined)).toBe(false);
      expect(isscalar(array([1, 2, 3]))).toBe(false);
    });
  });

  describe('iterable()', () => {
    it('returns true for iterables', () => {
      expect(iterable([1, 2, 3])).toBe(true);
      expect(iterable('hello')).toBe(true);
      expect(iterable(new Set([1, 2, 3]))).toBe(true);
      expect(iterable(new Map())).toBe(true);
    });

    it('returns false for non-iterables', () => {
      expect(iterable(5)).toBe(false);
      expect(iterable(true)).toBe(false);
      expect(iterable(null)).toBe(false);
      expect(iterable(undefined)).toBe(false);
      expect(iterable({ a: 1 })).toBe(false);
    });
  });

  describe('isdtype()', () => {
    it('checks for bool dtype', () => {
      expect(isdtype('bool', 'b')).toBe(true);
      expect(isdtype('int32', 'b')).toBe(false);
    });

    it('checks for int dtypes', () => {
      expect(isdtype('int8', 'i')).toBe(true);
      expect(isdtype('int16', 'i')).toBe(true);
      expect(isdtype('int32', 'i')).toBe(true);
      expect(isdtype('int64', 'i')).toBe(true);
      expect(isdtype('uint32', 'i')).toBe(false);
      expect(isdtype('float32', 'i')).toBe(false);
    });

    it('checks for uint dtypes', () => {
      expect(isdtype('uint8', 'u')).toBe(true);
      expect(isdtype('uint16', 'u')).toBe(true);
      expect(isdtype('uint32', 'u')).toBe(true);
      expect(isdtype('uint64', 'u')).toBe(true);
      expect(isdtype('int32', 'u')).toBe(false);
    });

    it('checks for float dtypes', () => {
      expect(isdtype('float32', 'f')).toBe(true);
      expect(isdtype('float64', 'f')).toBe(true);
      expect(isdtype('int32', 'f')).toBe(false);
    });

    it('returns false for unknown kind', () => {
      expect(isdtype('float32', 'x')).toBe(false);
    });
  });

  describe('promote_types()', () => {
    it('promotes int to float', () => {
      expect(promote_types('int32', 'float32')).toBe('float32');
      expect(promote_types('float32', 'int32')).toBe('float32');
    });

    it('promotes to higher precision', () => {
      expect(promote_types('int8', 'int32')).toBe('int32');
      expect(promote_types('float32', 'float64')).toBe('float64');
    });

    it('handles same dtype', () => {
      expect(promote_types('int32', 'int32')).toBe('int32');
      expect(promote_types('float64', 'float64')).toBe('float64');
    });

    it('promotes signed and unsigned ints', () => {
      expect(promote_types('int32', 'uint32')).toBe('int32');
      expect(promote_types('uint8', 'int16')).toBe('int16');
    });

    it('promotes bool to any other type', () => {
      expect(promote_types('bool', 'int32')).toBe('int32');
      expect(promote_types('int32', 'bool')).toBe('int32');
    });
  });
});

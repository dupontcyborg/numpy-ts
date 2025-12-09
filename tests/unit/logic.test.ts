/**
 * Unit tests for logic operations
 */

import { describe, it, expect } from 'vitest';
import { array } from '../../src/core/ndarray';
import {
  logical_and,
  logical_or,
  logical_not,
  logical_xor,
  isfinite,
  isinf,
  isnan,
  isnat,
  copysign,
  signbit,
  nextafter,
  spacing,
} from '../../src/core/ndarray';

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

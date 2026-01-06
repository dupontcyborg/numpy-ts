import { describe, it, expect } from 'vitest';
import {
  array,
  sqrt,
  power,
  absolute,
  negative,
  sign,
  mod,
  floor_divide,
  positive,
  reciprocal,
} from '../../src/core/ndarray';

describe('Mathematical Operations', () => {
  describe('sqrt', () => {
    it('computes square root of positive values', () => {
      const arr = array([4, 9, 16, 25]);
      const result = sqrt(arr);
      expect(result.toArray()).toEqual([2, 3, 4, 5]);
    });

    it('computes square root as method', () => {
      const arr = array([1, 4, 9]);
      const result = arr.sqrt();
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([4, 9], 'int32');
      const result = sqrt(arr);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([2, 3]);
    });

    it('preserves float32 dtype', () => {
      const arr = array([4, 9], 'float32');
      const result = sqrt(arr);
      expect(result.dtype).toBe('float32');
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [1, 4],
        [9, 16],
      ]);
      const result = sqrt(arr);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });
  });

  describe('power', () => {
    it('raises array to scalar power', () => {
      const arr = array([2, 3, 4]);
      const result = power(arr, 2);
      expect(result.toArray()).toEqual([4, 9, 16]);
    });

    it('raises array to power as method', () => {
      const arr = array([2, 3, 4]);
      const result = arr.power(3);
      expect(result.toArray()).toEqual([8, 27, 64]);
    });

    it('preserves integer dtype for positive integer exponent', () => {
      const arr = array([2, 3, 4], 'int32');
      const result = power(arr, 2);
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([4, 9, 16]);
    });

    it('promotes to float64 for negative exponent', () => {
      const arr = array([2, 4, 8], 'int32');
      const result = power(arr, -1);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([0.5, 0.25, 0.125]);
    });

    it('promotes to float64 for fractional exponent', () => {
      const arr = array([4, 9, 16], 'int32');
      const result = power(arr, 0.5);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([2, 3, 4]);
    });

    it('raises to zero power', () => {
      const arr = array([1, 2, 3, 0]);
      const result = power(arr, 0);
      expect(result.toArray()).toEqual([1, 1, 1, 1]);
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [2, 3],
        [4, 5],
      ]);
      const result = power(arr, 2);
      expect(result.toArray()).toEqual([
        [4, 9],
        [16, 25],
      ]);
    });

    it('works with BigInt arrays', () => {
      const arr = array([2n, 3n, 4n], 'int64');
      const result = power(arr, 2);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([4n, 9n, 16n]);
    });

    it('promotes BigInt to float64 for negative exponent', () => {
      const arr = array([2n, 4n, 8n], 'int64');
      const result = power(arr, -1);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([0.5, 0.25, 0.125]);
    });

    it('promotes BigInt to float64 for fractional exponent', () => {
      const arr = array([4n, 9n, 16n], 'int64');
      const result = power(arr, 0.5);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([2, 3, 4]);
    });
  });

  describe('absolute', () => {
    it('computes absolute value of mixed signs', () => {
      const arr = array([-3, -1, 0, 1, 3]);
      const result = absolute(arr);
      expect(result.toArray()).toEqual([3, 1, 0, 1, 3]);
    });

    it('computes absolute value as method', () => {
      const arr = array([-5, 5, -10]);
      const result = arr.absolute();
      expect(result.toArray()).toEqual([5, 5, 10]);
    });

    it('preserves dtype', () => {
      const arr = array([-5, -10, 15], 'int32');
      const result = absolute(arr);
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([5, 10, 15]);
    });

    it('works with float arrays', () => {
      const arr = array([-3.5, 2.1, -0.5], 'float64');
      const result = absolute(arr);
      expect(result.toArray()).toEqual([3.5, 2.1, 0.5]);
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [-1, 2],
        [-3, 4],
      ]);
      const result = absolute(arr);
      expect(result.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('works with BigInt arrays', () => {
      const arr = array([-5n, 10n, -15n], 'int64');
      const result = absolute(arr);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([5n, 10n, 15n]);
    });
  });

  describe('negative', () => {
    it('negates positive and negative values', () => {
      const arr = array([1, -2, 3, -4, 0]);
      const result = negative(arr);
      const resultArray = result.toArray() as number[];
      expect(resultArray[0]).toBe(-1);
      expect(resultArray[1]).toBe(2);
      expect(resultArray[2]).toBe(-3);
      expect(resultArray[3]).toBe(4);
      // JavaScript -0 === 0 is true, but Object.is(-0, 0) is false
      // Both -0 and 0 are acceptable for negating 0
      expect(Math.abs(resultArray[4]!)).toBe(0);
    });

    it('negates as method', () => {
      const arr = array([5, -10, 15]);
      const result = arr.negative();
      expect(result.toArray()).toEqual([-5, 10, -15]);
    });

    it('preserves dtype', () => {
      const arr = array([5, -10, 15], 'int32');
      const result = negative(arr);
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([-5, 10, -15]);
    });

    it('works with float arrays', () => {
      const arr = array([3.5, -2.1, 0.5], 'float64');
      const result = negative(arr);
      expect(result.toArray()).toEqual([-3.5, 2.1, -0.5]);
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [1, -2],
        [-3, 4],
      ]);
      const result = negative(arr);
      expect(result.toArray()).toEqual([
        [-1, 2],
        [3, -4],
      ]);
    });

    it('works with BigInt arrays', () => {
      const arr = array([5n, -10n, 15n], 'int64');
      const result = negative(arr);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([-5n, 10n, -15n]);
    });
  });

  describe('sign', () => {
    it('returns sign of values (-1, 0, 1)', () => {
      const arr = array([-5, -0.5, 0, 0.5, 5]);
      const result = sign(arr);
      expect(result.toArray()).toEqual([-1, -1, 0, 1, 1]);
    });

    it('returns sign as method', () => {
      const arr = array([-10, 0, 10]);
      const result = arr.sign();
      expect(result.toArray()).toEqual([-1, 0, 1]);
    });

    it('preserves dtype', () => {
      const arr = array([-5, 0, 5], 'int32');
      const result = sign(arr);
      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([-1, 0, 1]);
    });

    it('works with float arrays', () => {
      const arr = array([-3.5, 0, 2.1], 'float64');
      const result = sign(arr);
      expect(result.toArray()).toEqual([-1, 0, 1]);
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [-1, 0],
        [0, 1],
      ]);
      const result = sign(arr);
      expect(result.toArray()).toEqual([
        [-1, 0],
        [0, 1],
      ]);
    });

    it('works with BigInt arrays', () => {
      const arr = array([-5n, 0n, 5n], 'int64');
      const result = sign(arr);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([-1n, 0n, 1n]);
    });
  });

  describe('mod', () => {
    it('computes modulo with scalar', () => {
      const arr = array([10, 11, 12, 13]);
      const result = mod(arr, 3);
      expect(result.toArray()).toEqual([1, 2, 0, 1]);
    });

    it('computes modulo as method', () => {
      const arr = array([7, 8, 9]);
      const result = arr.mod(4);
      expect(result.toArray()).toEqual([3, 0, 1]);
    });

    it('preserves dtype', () => {
      const arr = array([10, 11, 12], 'int32');
      const result = mod(arr, 3);
      expect(result.dtype).toBe('int32');
    });

    it('works with negative numbers (NumPy floor modulo)', () => {
      // NumPy uses floor modulo: result has same sign as divisor
      // -10 % 3 = 2 (not -1 like JavaScript)
      const arr = array([-10, -7, -3, 0, 3, 7, 10]);
      const result = mod(arr, 3);
      const resultArray = result.toArray() as number[];
      expect(resultArray[0]).toBe(2); // -10 % 3 = 2 (floor modulo)
      expect(resultArray[1]).toBe(2); // -7 % 3 = 2
      expect(Math.abs(resultArray[2]!)).toBe(0); // -3 % 3 = 0
      expect(resultArray[3]).toBe(0); // 0 % 3 = 0
      expect(resultArray[4]).toBe(0); // 3 % 3 = 0
      expect(resultArray[5]).toBe(1); // 7 % 3 = 1
      expect(resultArray[6]).toBe(1); // 10 % 3 = 1
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [10, 11],
        [12, 13],
      ]);
      const result = mod(arr, 3);
      expect(result.toArray()).toEqual([
        [1, 2],
        [0, 1],
      ]);
    });

    it('works with BigInt arrays', () => {
      const arr = array([10n, 11n, 12n], 'int64');
      const result = mod(arr, 3);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([1n, 2n, 0n]);
    });
  });

  describe('floor_divide', () => {
    it('computes floor division with scalar', () => {
      const arr = array([10, 11, 12, 13]);
      const result = floor_divide(arr, 3);
      expect(result.toArray()).toEqual([3, 3, 4, 4]);
    });

    it('computes floor division as method', () => {
      const arr = array([7, 8, 9]);
      const result = arr.floor_divide(2);
      expect(result.toArray()).toEqual([3, 4, 4]);
    });

    it('preserves dtype for integers', () => {
      const arr = array([10, 11, 12], 'int32');
      const result = floor_divide(arr, 3);
      expect(result.dtype).toBe('int32');
    });

    it('works with negative numbers', () => {
      const arr = array([-10, -7, -3, 0, 3, 7, 10]);
      const result = floor_divide(arr, 3);
      expect(result.toArray()).toEqual([-4, -3, -1, 0, 1, 2, 3]);
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [10, 15],
        [20, 25],
      ]);
      const result = floor_divide(arr, 4);
      expect(result.toArray()).toEqual([
        [2, 3],
        [5, 6],
      ]);
    });

    it('works with BigInt arrays', () => {
      const arr = array([10n, 15n, 20n], 'int64');
      const result = floor_divide(arr, 3);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([3n, 5n, 6n]);
    });
  });

  describe('positive', () => {
    it('returns a copy of the array', () => {
      const arr = array([1, -2, 3, -4, 0]);
      const result = positive(arr);
      expect(result.toArray()).toEqual([1, -2, 3, -4, 0]);
      // Verify it's a copy, not the same object
      expect(result).not.toBe(arr);
    });

    it('works as method', () => {
      const arr = array([5, -10, 15]);
      const result = arr.positive();
      expect(result.toArray()).toEqual([5, -10, 15]);
    });

    it('preserves dtype', () => {
      const arr = array([5, -10, 15], 'int32');
      const result = positive(arr);
      expect(result.dtype).toBe('int32');
    });

    it('works with float arrays', () => {
      const arr = array([3.5, -2.1, 0.5], 'float64');
      const result = positive(arr);
      expect(result.toArray()).toEqual([3.5, -2.1, 0.5]);
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [1, -2],
        [-3, 4],
      ]);
      const result = positive(arr);
      expect(result.toArray()).toEqual([
        [1, -2],
        [-3, 4],
      ]);
    });

    it('works with BigInt arrays', () => {
      const arr = array([5n, -10n, 15n], 'int64');
      const result = positive(arr);
      expect(result.dtype).toBe('int64');
      expect(result.toArray()).toEqual([5n, -10n, 15n]);
    });
  });

  describe('reciprocal', () => {
    it('computes reciprocal (1/x)', () => {
      const arr = array([2, 4, 8]);
      const result = reciprocal(arr);
      expect(result.toArray()).toEqual([0.5, 0.25, 0.125]);
    });

    it('works as method', () => {
      const arr = array([5, 10, 20]);
      const result = arr.reciprocal();
      expect(result.toArray()).toEqual([0.2, 0.1, 0.05]);
    });

    it('promotes integer arrays to float64', () => {
      const arr = array([2, 4, 8], 'int32');
      const result = reciprocal(arr);
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([0.5, 0.25, 0.125]);
    });

    it('preserves float32 dtype', () => {
      const arr = array([2, 4], 'float32');
      const result = reciprocal(arr);
      expect(result.dtype).toBe('float32');
    });

    it('handles negative numbers', () => {
      const arr = array([-2, -4, 2, 4]);
      const result = reciprocal(arr);
      expect(result.toArray()).toEqual([-0.5, -0.25, 0.5, 0.25]);
    });

    it('works with 2D arrays', () => {
      const arr = array([
        [2, 4],
        [5, 10],
      ]);
      const result = reciprocal(arr);
      expect(result.toArray()).toEqual([
        [0.5, 0.25],
        [0.2, 0.1],
      ]);
    });

    it('returns Infinity for zero', () => {
      const arr = array([0, 1, 2]);
      const result = reciprocal(arr);
      expect(result.get([0])).toBe(Infinity);
      expect(result.get([1])).toBe(1);
      expect(result.get([2])).toBe(0.5);
    });
  });
});

/**
 * Unit tests for internal computation engine
 * Tests element-wise operations with different dtypes
 */

import { describe, it, expect } from 'vitest';
import { array, zeros } from '../../src';

describe('Compute engine - element-wise operations', () => {
  describe('Binary operations with BigInt dtypes', () => {
    it('performs addition with int64 dtype', () => {
      const a = array([1, 2, 3], 'int64');
      const b = array([4, 5, 6], 'int64');
      const result = a.add(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(BigInt(5));
      expect(result.data[1]).toBe(BigInt(7));
      expect(result.data[2]).toBe(BigInt(9));
    });

    it('performs subtraction with int64 dtype', () => {
      const a = array([10, 20, 30], 'int64');
      const b = array([1, 2, 3], 'int64');
      const result = a.subtract(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(BigInt(9));
      expect(result.data[1]).toBe(BigInt(18));
      expect(result.data[2]).toBe(BigInt(27));
    });

    it('performs multiplication with int64 dtype', () => {
      const a = array([2, 3, 4], 'int64');
      const b = array([5, 6, 7], 'int64');
      const result = a.multiply(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(BigInt(10));
      expect(result.data[1]).toBe(BigInt(18));
      expect(result.data[2]).toBe(BigInt(28));
    });

    it('performs division with int64 dtype (promotes to float64)', () => {
      const a = array([10, 20, 30], 'int64');
      const b = array([2, 4, 5], 'int64');
      const result = a.divide(b);

      // Division always promotes to float64 in NumPy
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([5, 5, 6]);
    });

    it('performs addition with uint64 dtype', () => {
      const a = array([1, 2, 3], 'uint64');
      const b = array([4, 5, 6], 'uint64');
      const result = a.add(b);

      expect(result.dtype).toBe('uint64');
      expect(result.data[0]).toBe(BigInt(5));
      expect(result.data[1]).toBe(BigInt(7));
      expect(result.data[2]).toBe(BigInt(9));
    });

    it('performs subtraction with uint64 dtype', () => {
      const a = array([10, 20, 30], 'uint64');
      const b = array([1, 2, 3], 'uint64');
      const result = a.subtract(b);

      expect(result.dtype).toBe('uint64');
      expect(result.data[0]).toBe(BigInt(9));
      expect(result.data[1]).toBe(BigInt(18));
      expect(result.data[2]).toBe(BigInt(27));
    });

    it('performs multiplication with uint64 dtype', () => {
      const a = array([2, 3, 4], 'uint64');
      const b = array([5, 6, 7], 'uint64');
      const result = a.multiply(b);

      expect(result.dtype).toBe('uint64');
      expect(result.data[0]).toBe(BigInt(10));
      expect(result.data[1]).toBe(BigInt(18));
      expect(result.data[2]).toBe(BigInt(28));
    });

    it('performs division with uint64 dtype (promotes to float64)', () => {
      const a = array([10, 20, 30], 'uint64');
      const b = array([2, 4, 5], 'uint64');
      const result = a.divide(b);

      // Division always promotes to float64 in NumPy
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([5, 5, 6]);
    });
  });

  describe('Mixed dtype operations with BigInt', () => {
    it('promotes int64 + float64 to float64', () => {
      const a = array([1, 2, 3], 'int64');
      const b = array([1.5, 2.5, 3.5], 'float64');
      const result = a.add(b);

      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([2.5, 4.5, 6.5]);
    });

    it('promotes int64 + int32 to int64', () => {
      const a = array([10, 20, 30], 'int64');
      const b = array([1, 2, 3], 'int32');
      const result = a.add(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(BigInt(11));
      expect(result.data[1]).toBe(BigInt(22));
      expect(result.data[2]).toBe(BigInt(33));
    });

    it('converts from int32 to int64 in promotion', () => {
      const a = array([1, 2, 3], 'int32');
      const b = array([4, 5, 6], 'int64');
      const result = a.add(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(BigInt(5));
    });
  });

  describe('Unary operations', () => {
    it('performs absolute on float64', () => {
      const a = array([-1, -2, -3, 4, 5]);
      const result = a.absolute();

      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('performs absolute on int32', () => {
      const a = array([-1, -2, -3, 4, 5], 'int32');
      const result = a.absolute();

      expect(result.dtype).toBe('int32');
      expect(result.toArray()).toEqual([1, 2, 3, 4, 5]);
    });

    it('performs sqrt with preserveDtype=false (promotes to float64)', () => {
      const a = array([4, 9, 16], 'int32');
      const result = a.sqrt();

      // sqrt should promote integer types to float64
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([2, 3, 4]);
    });
  });

  describe('Unary operations with BigInt input', () => {
    it('converts int64 input for unary operations', () => {
      const a = array([1, 4, 9], 'int64');
      const result = a.sqrt();

      // sqrt promotes to float64 for precision
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([1, 2, 3]);
    });

    it('performs absolute on int64', () => {
      const a = array([-1, -2, 3], 'int64');
      const result = a.absolute();

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(BigInt(1));
      expect(result.data[1]).toBe(BigInt(2));
      expect(result.data[2]).toBe(BigInt(3));
    });

    it('performs unary operation on uint64', () => {
      const a = array([1, 4, 9], 'uint64');
      const result = a.sqrt();

      // sqrt promotes to float64
      expect(result.dtype).toBe('float64');
      expect(result.toArray()).toEqual([1, 2, 3]);
    });
  });

  describe('Comparison operations', () => {
    it('compares int64 arrays', () => {
      const a = array([1, 2, 3], 'int64');
      const b = array([2, 2, 1], 'int64');
      const result = a.greater(b);

      expect(result.dtype).toBe('bool');
      expect(result.toArray()).toEqual([0, 0, 1]);
    });

    it('compares uint64 arrays', () => {
      const a = array([5, 10, 15], 'uint64');
      const b = array([10, 10, 10], 'uint64');
      const result = a.less(b);

      expect(result.dtype).toBe('bool');
      expect(result.toArray()).toEqual([1, 0, 0]);
    });

    it('compares mixed BigInt and regular dtypes', () => {
      const a = array([1, 2, 3], 'int64');
      const b = array([1, 2, 3], 'int32');
      const result = a.equal(b);

      expect(result.dtype).toBe('bool');
      expect(result.toArray()).toEqual([1, 1, 1]);
    });
  });

  describe('Broadcasting with BigInt dtypes', () => {
    it('broadcasts scalar to int64 array', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'int64'
      );
      const b = array([10], 'int64');
      const result = a.add(b);

      expect(result.dtype).toBe('int64');
      expect(result.data[0]).toBe(BigInt(11));
      expect(result.data[1]).toBe(BigInt(12));
      expect(result.data[2]).toBe(BigInt(13));
      expect(result.data[3]).toBe(BigInt(14));
    });

    it('broadcasts with mixed dtypes involving BigInt', () => {
      const a = array(
        [
          [1, 2],
          [3, 4],
        ],
        'int64'
      );
      const b = array([10, 20], 'int32');
      const result = a.add(b);

      expect(result.dtype).toBe('int64');
      expect(result.shape).toEqual([2, 2]);
    });
  });

  describe('Edge cases', () => {
    it('handles operations on empty arrays', () => {
      const a = zeros([0], 'float64');
      const b = zeros([0], 'float64');
      const result = a.add(b);

      expect(result.shape).toEqual([0]);
      expect(result.size).toBe(0);
    });

    it('handles operations on scalar arrays', () => {
      const a = array([5]);
      const b = array([10]);
      const result = a.add(b);

      expect(result.toArray()).toEqual([15]);
    });

    it('throws error for incompatible broadcast shapes', () => {
      const a = zeros([2, 3], 'float64');
      const b = zeros([2, 4], 'float64');

      expect(() => a.add(b)).toThrow(/could not be broadcast/);
    });
  });
});

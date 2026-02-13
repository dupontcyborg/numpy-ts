/**
 * Unit tests for array indexing operations (get and set)
 */

import { describe, it, expect } from 'vitest';
import { array, zeros, iindex, bindex } from '../../src';

describe('Array Indexing', () => {
  describe('get()', () => {
    describe('1D arrays', () => {
      it('gets element at positive index', () => {
        const arr = array([10, 20, 30, 40, 50]);
        expect(arr.get([0])).toBe(10);
        expect(arr.get([2])).toBe(30);
        expect(arr.get([4])).toBe(50);
      });

      it('gets element at negative index', () => {
        const arr = array([10, 20, 30, 40, 50]);
        expect(arr.get([-1])).toBe(50);
        expect(arr.get([-2])).toBe(40);
        expect(arr.get([-5])).toBe(10);
      });

      it('throws on out of bounds positive index', () => {
        const arr = array([10, 20, 30]);
        expect(() => arr.get([3])).toThrow(/out of bounds/);
        expect(() => arr.get([100])).toThrow(/out of bounds/);
      });

      it('throws on out of bounds negative index', () => {
        const arr = array([10, 20, 30]);
        expect(() => arr.get([-4])).toThrow(/out of bounds/);
        expect(() => arr.get([-100])).toThrow(/out of bounds/);
      });
    });

    describe('2D arrays', () => {
      it('gets element at positive indices', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        expect(arr.get([0, 0])).toBe(1);
        expect(arr.get([0, 2])).toBe(3);
        expect(arr.get([1, 1])).toBe(5);
        expect(arr.get([2, 2])).toBe(9);
      });

      it('gets element at negative indices', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        expect(arr.get([-1, -1])).toBe(9);
        expect(arr.get([-1, 0])).toBe(7);
        expect(arr.get([0, -1])).toBe(3);
        expect(arr.get([-2, -2])).toBe(5);
      });

      it('gets element with mixed positive and negative indices', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(arr.get([0, -1])).toBe(3);
        expect(arr.get([-1, 0])).toBe(4);
        expect(arr.get([-1, -2])).toBe(5);
      });

      it('throws on out of bounds', () => {
        const arr = array([
          [1, 2],
          [3, 4],
        ]);
        expect(() => arr.get([2, 0])).toThrow(/out of bounds/);
        expect(() => arr.get([0, 2])).toThrow(/out of bounds/);
        expect(() => arr.get([-3, 0])).toThrow(/out of bounds/);
      });
    });

    describe('3D arrays', () => {
      it('gets element at positive indices', () => {
        const arr = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]);
        expect(arr.get([0, 0, 0])).toBe(1);
        expect(arr.get([0, 1, 1])).toBe(4);
        expect(arr.get([1, 0, 1])).toBe(6);
        expect(arr.get([1, 1, 1])).toBe(8);
      });

      it('gets element at negative indices', () => {
        const arr = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]);
        expect(arr.get([-1, -1, -1])).toBe(8);
        expect(arr.get([-2, -1, -2])).toBe(3);
        expect(arr.get([-1, 0, -1])).toBe(6);
      });
    });

    describe('dtype handling', () => {
      it('returns number for float32', () => {
        const arr = zeros([3], 'float32');
        arr.data[0] = 1.5;
        arr.data[1] = 2.7;
        arr.data[2] = 3.9;
        expect(arr.get([0])).toBe(1.5);
        expect(arr.get([1])).toBeCloseTo(2.7);
        expect(arr.get([2])).toBeCloseTo(3.9);
        expect(typeof arr.get([0])).toBe('number');
      });

      it('returns number for float64', () => {
        const arr = zeros([3], 'float64');
        arr.data[0] = 1.5;
        arr.data[1] = 2.7;
        arr.data[2] = 3.9;
        expect(arr.get([0])).toBe(1.5);
        expect(arr.get([1])).toBe(2.7);
        expect(arr.get([2])).toBe(3.9);
        expect(typeof arr.get([0])).toBe('number');
      });

      it('returns number for int32', () => {
        const arr = zeros([3], 'int32');
        arr.data[0] = 10;
        arr.data[1] = -20;
        arr.data[2] = 30;
        expect(arr.get([0])).toBe(10);
        expect(arr.get([1])).toBe(-20);
        expect(arr.get([2])).toBe(30);
        expect(typeof arr.get([0])).toBe('number');
      });

      it('returns number for uint8', () => {
        const arr = zeros([3], 'uint8');
        arr.data[0] = 10;
        arr.data[1] = 200;
        arr.data[2] = 255;
        expect(arr.get([0])).toBe(10);
        expect(arr.get([1])).toBe(200);
        expect(arr.get([2])).toBe(255);
        expect(typeof arr.get([0])).toBe('number');
      });

      it('returns BigInt for int64', () => {
        const arr = zeros([3], 'int64');
        arr.data[0] = 1000000000000n;
        arr.data[1] = -2000000000000n;
        arr.data[2] = 3000000000000n;
        expect(arr.get([0])).toBe(1000000000000n);
        expect(arr.get([1])).toBe(-2000000000000n);
        expect(arr.get([2])).toBe(3000000000000n);
        expect(typeof arr.get([0])).toBe('bigint');
      });

      it('returns BigInt for uint64', () => {
        const arr = zeros([3], 'uint64');
        arr.data[0] = 1000000000000n;
        arr.data[1] = 2000000000000n;
        arr.data[2] = 18446744073709551615n; // Max uint64
        expect(arr.get([0])).toBe(1000000000000n);
        expect(arr.get([1])).toBe(2000000000000n);
        expect(arr.get([2])).toBe(18446744073709551615n);
        expect(typeof arr.get([0])).toBe('bigint');
      });

      it('returns number for bool (0 or 1)', () => {
        const arr = zeros([3], 'bool');
        arr.data[0] = 1; // true
        arr.data[1] = 0; // false
        arr.data[2] = 1; // true
        expect(arr.get([0])).toBe(1);
        expect(arr.get([1])).toBe(0);
        expect(arr.get([2])).toBe(1);
        expect(typeof arr.get([0])).toBe('number');
      });
    });

    describe('error validation', () => {
      it('throws when too few indices provided', () => {
        const arr = array([
          [1, 2],
          [3, 4],
        ]);
        expect(() => arr.get([0])).toThrow(/Index has 1 dimensions, but array has 2 dimensions/);
      });

      it('throws when too many indices provided', () => {
        const arr = array([1, 2, 3]);
        expect(() => arr.get([0, 0])).toThrow(/Index has 2 dimensions, but array has 1 dimensions/);
      });

      it('provides clear error message with axis information', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        expect(() => arr.get([0, 5])).toThrow(/Index 5 is out of bounds for axis 1 with size 3/);
        expect(() => arr.get([3, 0])).toThrow(/Index 3 is out of bounds for axis 0 with size 2/);
      });
    });

    describe('edge cases', () => {
      it.skip('gets single element from 0D array', () => {
        // 0D arrays (scalars) are not supported by stdlib's array() constructor
        const arr = array(42);
        expect(arr.get([])).toBe(42);
      });

      it('handles large arrays', () => {
        const arr = zeros([1000], 'float64');
        arr.data[500] = 3.14;
        expect(arr.get([500])).toBe(3.14);
        expect(arr.get([-500])).toBe(3.14);
      });

      it('handles arrays with size 1 dimensions', () => {
        const arr = array([[1], [2], [3]]);
        expect(arr.shape).toEqual([3, 1]);
        expect(arr.get([0, 0])).toBe(1);
        expect(arr.get([1, 0])).toBe(2);
        expect(arr.get([-1, -1])).toBe(3);
      });
    });
  });

  describe('set()', () => {
    describe('1D arrays', () => {
      it('sets element at positive index', () => {
        const arr = zeros([5], 'float64');
        arr.set([0], 10);
        arr.set([2], 30);
        arr.set([4], 50);
        expect(arr.get([0])).toBe(10);
        expect(arr.get([1])).toBe(0);
        expect(arr.get([2])).toBe(30);
        expect(arr.get([4])).toBe(50);
      });

      it('sets element at negative index', () => {
        const arr = zeros([5], 'float64');
        arr.set([-1], 50);
        arr.set([-2], 40);
        arr.set([-5], 10);
        expect(arr.get([0])).toBe(10);
        expect(arr.get([3])).toBe(40);
        expect(arr.get([4])).toBe(50);
      });

      it('throws on out of bounds positive index', () => {
        const arr = zeros([3], 'float64');
        expect(() => arr.set([3], 1)).toThrow(/out of bounds/);
        expect(() => arr.set([100], 1)).toThrow(/out of bounds/);
      });

      it('throws on out of bounds negative index', () => {
        const arr = zeros([3], 'float64');
        expect(() => arr.set([-4], 1)).toThrow(/out of bounds/);
        expect(() => arr.set([-100], 1)).toThrow(/out of bounds/);
      });
    });

    describe('2D arrays', () => {
      it('sets element at positive indices', () => {
        const arr = zeros([3, 3], 'float64');
        arr.set([0, 0], 1);
        arr.set([0, 2], 3);
        arr.set([1, 1], 5);
        arr.set([2, 2], 9);

        expect(arr.get([0, 0])).toBe(1);
        expect(arr.get([0, 2])).toBe(3);
        expect(arr.get([1, 1])).toBe(5);
        expect(arr.get([2, 2])).toBe(9);
        expect(arr.get([0, 1])).toBe(0); // untouched
      });

      it('sets element at negative indices', () => {
        const arr = zeros([3, 3], 'float64');
        arr.set([-1, -1], 9);
        arr.set([-1, 0], 7);
        arr.set([0, -1], 3);
        arr.set([-2, -2], 5);

        expect(arr.get([2, 2])).toBe(9);
        expect(arr.get([2, 0])).toBe(7);
        expect(arr.get([0, 2])).toBe(3);
        expect(arr.get([1, 1])).toBe(5);
      });

      it('sets element with mixed positive and negative indices', () => {
        const arr = zeros([2, 3], 'float64');
        arr.set([0, -1], 3);
        arr.set([-1, 0], 4);
        arr.set([-1, -2], 5);

        expect(arr.get([0, 2])).toBe(3);
        expect(arr.get([1, 0])).toBe(4);
        expect(arr.get([1, 1])).toBe(5);
      });

      it('throws on out of bounds', () => {
        const arr = zeros([2, 2], 'float64');
        expect(() => arr.set([2, 0], 1)).toThrow(/out of bounds/);
        expect(() => arr.set([0, 2], 1)).toThrow(/out of bounds/);
        expect(() => arr.set([-3, 0], 1)).toThrow(/out of bounds/);
      });
    });

    describe('3D arrays', () => {
      it('sets element at positive indices', () => {
        const arr = zeros([2, 2, 2], 'float64');
        arr.set([0, 0, 0], 1);
        arr.set([0, 1, 1], 4);
        arr.set([1, 0, 1], 6);
        arr.set([1, 1, 1], 8);

        expect(arr.get([0, 0, 0])).toBe(1);
        expect(arr.get([0, 1, 1])).toBe(4);
        expect(arr.get([1, 0, 1])).toBe(6);
        expect(arr.get([1, 1, 1])).toBe(8);
      });

      it('sets element at negative indices', () => {
        const arr = zeros([2, 2, 2], 'float64');
        arr.set([-1, -1, -1], 8);
        arr.set([-2, -1, -2], 3);
        arr.set([-1, 0, -1], 6);

        expect(arr.get([1, 1, 1])).toBe(8);
        expect(arr.get([0, 1, 0])).toBe(3);
        expect(arr.get([1, 0, 1])).toBe(6);
      });
    });

    describe('dtype handling', () => {
      it('sets float32 values', () => {
        const arr = zeros([3], 'float32');
        arr.set([0], 1.5);
        arr.set([1], 2.7);
        arr.set([2], 3.9);

        expect(arr.get([0])).toBeCloseTo(1.5);
        expect(arr.get([1])).toBeCloseTo(2.7);
        expect(arr.get([2])).toBeCloseTo(3.9);
      });

      it('sets float64 values', () => {
        const arr = zeros([3], 'float64');
        arr.set([0], 1.5);
        arr.set([1], 2.7);
        arr.set([2], 3.9);

        expect(arr.get([0])).toBe(1.5);
        expect(arr.get([1])).toBe(2.7);
        expect(arr.get([2])).toBe(3.9);
      });

      it('sets int32 values', () => {
        const arr = zeros([3], 'int32');
        arr.set([0], 10);
        arr.set([1], -20);
        arr.set([2], 30);

        expect(arr.get([0])).toBe(10);
        expect(arr.get([1])).toBe(-20);
        expect(arr.get([2])).toBe(30);
      });

      it('sets uint8 values', () => {
        const arr = zeros([3], 'uint8');
        arr.set([0], 10);
        arr.set([1], 200);
        arr.set([2], 255);

        expect(arr.get([0])).toBe(10);
        expect(arr.get([1])).toBe(200);
        expect(arr.get([2])).toBe(255);
      });

      it('sets int64 values with BigInt', () => {
        const arr = zeros([3], 'int64');
        arr.set([0], 1000000000000n);
        arr.set([1], -2000000000000n);
        arr.set([2], 3000000000000n);

        expect(arr.get([0])).toBe(1000000000000n);
        expect(arr.get([1])).toBe(-2000000000000n);
        expect(arr.get([2])).toBe(3000000000000n);
      });

      it('sets int64 values with number (auto-converts to BigInt)', () => {
        const arr = zeros([3], 'int64');
        arr.set([0], 1000);
        arr.set([1], -2000);
        arr.set([2], 3000);

        expect(arr.get([0])).toBe(1000n);
        expect(arr.get([1])).toBe(-2000n);
        expect(arr.get([2])).toBe(3000n);
      });

      it('sets uint64 values with BigInt', () => {
        const arr = zeros([3], 'uint64');
        arr.set([0], 1000000000000n);
        arr.set([1], 2000000000000n);
        arr.set([2], 18446744073709551615n); // Max uint64

        expect(arr.get([0])).toBe(1000000000000n);
        expect(arr.get([1])).toBe(2000000000000n);
        expect(arr.get([2])).toBe(18446744073709551615n);
      });

      it('sets bool values (truthy/falsy conversion)', () => {
        const arr = zeros([4], 'bool');
        arr.set([0], 1); // truthy
        arr.set([1], 0); // falsy
        arr.set([2], 42); // truthy
        arr.set([3], -1); // truthy

        expect(arr.get([0])).toBe(1);
        expect(arr.get([1])).toBe(0);
        expect(arr.get([2])).toBe(1);
        expect(arr.get([3])).toBe(1);
      });

      it('converts float to int (rounds)', () => {
        const arr = zeros([3], 'int32');
        arr.set([0], 1.4);
        arr.set([1], 2.6);
        arr.set([2], -3.5);

        // JavaScript Number() converts, then int32 storage truncates
        expect(arr.get([0])).toBe(1);
        expect(arr.get([1])).toBe(2);
        expect(arr.get([2])).toBe(-3);
      });
    });

    describe('error validation', () => {
      it('throws when too few indices provided', () => {
        const arr = zeros([2, 2], 'float64');
        expect(() => arr.set([0], 1)).toThrow(/Index has 1 dimensions, but array has 2 dimensions/);
      });

      it('throws when too many indices provided', () => {
        const arr = zeros([3], 'float64');
        expect(() => arr.set([0, 0], 1)).toThrow(
          /Index has 2 dimensions, but array has 1 dimensions/
        );
      });

      it('provides clear error message with axis information', () => {
        const arr = zeros([2, 3], 'float64');
        expect(() => arr.set([0, 5], 1)).toThrow(/Index 5 is out of bounds for axis 1 with size 3/);
        expect(() => arr.set([3, 0], 1)).toThrow(/Index 3 is out of bounds for axis 0 with size 2/);
      });
    });

    describe('integration tests', () => {
      it('can get and set multiple elements', () => {
        const arr = zeros([3, 3], 'float64');

        // Set a diagonal
        arr.set([0, 0], 1);
        arr.set([1, 1], 2);
        arr.set([2, 2], 3);

        // Verify diagonal
        expect(arr.get([0, 0])).toBe(1);
        expect(arr.get([1, 1])).toBe(2);
        expect(arr.get([2, 2])).toBe(3);

        // Verify zeros elsewhere
        expect(arr.get([0, 1])).toBe(0);
        expect(arr.get([1, 0])).toBe(0);
      });

      it('can overwrite existing values', () => {
        const arr = array([1, 2, 3, 4, 5]);
        arr.set([0], 10);
        arr.set([2], 30);
        arr.set([0], 100); // Overwrite again

        expect(arr.get([0])).toBe(100);
        expect(arr.get([1])).toBe(2);
        expect(arr.get([2])).toBe(30);
      });

      it('handles large arrays', () => {
        const arr = zeros([1000], 'float64');
        arr.set([500], 3.14);
        arr.set([-400], 2.71); // -400 = index 600

        expect(arr.get([500])).toBe(3.14);
        expect(arr.get([600])).toBe(2.71); // -400 is same as 600 for size 1000
        expect(arr.get([-400])).toBe(2.71);
        expect(arr.get([499])).toBe(0);
      });

      it('handles arrays with size 1 dimensions', () => {
        const arr = zeros([3, 1], 'float64');
        arr.set([0, 0], 1);
        arr.set([1, 0], 2);
        arr.set([-1, -1], 3);

        expect(arr.get([0, 0])).toBe(1);
        expect(arr.get([1, 0])).toBe(2);
        expect(arr.get([2, 0])).toBe(3);
      });
    });
  });

  describe('Fancy Indexing', () => {
    describe('iindex() - integer array indexing', () => {
      it('selects rows by integer indices from 2D array', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const result = arr.iindex([0, 2]);
        expect(result.shape).toEqual([2, 3]);
        // Row 0: [1, 2, 3], Row 2: [7, 8, 9]
        expect(Array.from(result.data)).toEqual([1, 2, 3, 7, 8, 9]);
      });

      it('selects columns by integer indices with axis=1', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const result = arr.iindex([0, 2], 1);
        expect(result.shape).toEqual([3, 2]);
        // Each row gets columns 0 and 2: [1, 3], [4, 6], [7, 9]
        expect(Array.from(result.data)).toEqual([1, 3, 4, 6, 7, 9]);
      });

      it('works with NDArray indices', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const indices = array([0, 2]);
        const result = arr.iindex(indices);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.data)).toEqual([1, 2, 3, 7, 8, 9]);
      });

      it('selects elements from 1D array', () => {
        const arr = array([10, 20, 30, 40, 50]);
        const result = arr.iindex([0, 2, 4]);
        expect(result.shape).toEqual([3]);
        expect(Array.from(result.data)).toEqual([10, 30, 50]);
      });

      it('allows duplicate indices', () => {
        const arr = array([10, 20, 30]);
        const result = arr.iindex([0, 0, 1, 1, 2, 2]);
        expect(Array.from(result.data)).toEqual([10, 10, 20, 20, 30, 30]);
      });

      it('supports negative indices', () => {
        const arr = array([10, 20, 30, 40, 50]);
        const result = arr.iindex([-1, -3]);
        expect(Array.from(result.data)).toEqual([50, 30]);
      });

      it('works with top-level iindex function', () => {
        const arr = array([
          [1, 2],
          [3, 4],
          [5, 6],
        ]);
        const result = iindex(arr, [0, 2]);
        expect(result.shape).toEqual([2, 2]);
        expect(Array.from(result.data)).toEqual([1, 2, 5, 6]);
      });

      it('throws on out of bounds indices', () => {
        const arr = array([10, 20, 30]);
        expect(() => arr.iindex([5])).toThrow(/out of bounds/);
      });
    });

    describe('bindex() - boolean array indexing', () => {
      it('selects elements where mask is true (1D)', () => {
        const arr = array([10, 20, 30, 40, 50]);
        const mask = array([true, false, true, false, true], 'bool');
        const result = arr.bindex(mask);
        expect(result.shape).toEqual([3]);
        expect(Array.from(result.data)).toEqual([10, 30, 50]);
      });

      it('selects elements where mask is true (2D flattened)', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const mask = array(
          [
            [true, false, true],
            [false, true, false],
          ],
          'bool'
        );
        const result = arr.bindex(mask);
        expect(result.shape).toEqual([3]);
        expect(Array.from(result.data)).toEqual([1, 3, 5]);
      });

      it('selects rows where mask is true (axis=0)', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);
        const mask = array([true, false, true], 'bool');
        const result = arr.bindex(mask, 0);
        expect(result.shape).toEqual([2, 3]);
        expect(Array.from(result.data)).toEqual([1, 2, 3, 7, 8, 9]);
      });

      it('selects columns where mask is true (axis=1)', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const mask = array([true, false, true], 'bool');
        const result = arr.bindex(mask, 1);
        expect(result.shape).toEqual([2, 2]);
        expect(Array.from(result.data)).toEqual([1, 3, 4, 6]);
      });

      it('works with comparison results', () => {
        const arr = array([1, 5, 3, 8, 2, 9, 4]);
        const mask = arr.greater(4);
        const result = arr.bindex(mask);
        expect(Array.from(result.data)).toEqual([5, 8, 9]);
      });

      it('works with top-level bindex function', () => {
        const arr = array([10, 20, 30, 40]);
        const mask = array([false, true, true, false], 'bool');
        const result = bindex(arr, mask);
        expect(Array.from(result.data)).toEqual([20, 30]);
      });

      it('returns empty array when no elements match', () => {
        const arr = array([1, 2, 3]);
        const mask = array([false, false, false], 'bool');
        const result = arr.bindex(mask);
        expect(result.size).toBe(0);
      });

      it('returns all elements when all match', () => {
        const arr = array([1, 2, 3]);
        const mask = array([true, true, true], 'bool');
        const result = arr.bindex(mask);
        expect(Array.from(result.data)).toEqual([1, 2, 3]);
      });
    });

    describe('fancy indexing integration', () => {
      it('combines with other operations', () => {
        const arr = array([
          [1, 2, 3],
          [4, 5, 6],
          [7, 8, 9],
        ]);

        // Select rows 0 and 2, then sum along columns
        const selected = arr.iindex([0, 2]);
        const colSums = selected.sum(0);
        // sum(0) on [[1,2,3],[7,8,9]] = [8, 10, 12]
        expect(Array.from(colSums.data)).toEqual([8, 10, 12]);
      });

      it('preserves dtype through fancy indexing', () => {
        const arr = array([1, 2, 3, 4], 'int32');
        const result = arr.iindex([0, 2]);
        expect(result.dtype).toBe('int32');
      });

      it('works with bigint dtypes', () => {
        const arr = array([1n, 2n, 3n, 4n, 5n], 'int64');
        const result = arr.iindex([0, 2, 4]);
        expect(Array.from(result.data)).toEqual([1n, 3n, 5n]);
        expect(result.dtype).toBe('int64');
      });
    });
  });
});

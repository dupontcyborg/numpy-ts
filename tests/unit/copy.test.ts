/**
 * Unit tests for array copy() method
 */

import { describe, it, expect } from 'vitest';
import { array, zeros } from '../../src';

describe('Array Copy', () => {
  describe('copy()', () => {
    describe('basic functionality', () => {
      it('creates independent copy of 1D array', () => {
        const original = array([1, 2, 3, 4, 5]);
        const copied = original.copy();

        // Modify copy
        copied.set([0], 99);
        copied.set([2], 88);

        // Original should be unchanged
        expect(original.get([0])).toBe(1);
        expect(original.get([2])).toBe(3);

        // Copy should be modified
        expect(copied.get([0])).toBe(99);
        expect(copied.get([2])).toBe(88);
      });

      it('creates independent copy of 2D array', () => {
        const original = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const copied = original.copy();

        // Modify copy
        copied.set([0, 0], 99);
        copied.set([1, 2], 88);

        // Original should be unchanged
        expect(original.get([0, 0])).toBe(1);
        expect(original.get([1, 2])).toBe(6);

        // Copy should be modified
        expect(copied.get([0, 0])).toBe(99);
        expect(copied.get([1, 2])).toBe(88);
      });

      it('creates independent copy of 3D array', () => {
        const original = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]);
        const copied = original.copy();

        // Modify copy
        copied.set([0, 0, 0], 99);
        copied.set([1, 1, 1], 88);

        // Original should be unchanged
        expect(original.get([0, 0, 0])).toBe(1);
        expect(original.get([1, 1, 1])).toBe(8);

        // Copy should be modified
        expect(copied.get([0, 0, 0])).toBe(99);
        expect(copied.get([1, 1, 1])).toBe(88);
      });
    });

    describe('dtype preservation', () => {
      it('preserves float64 dtype', () => {
        const original = array([1.5, 2.7, 3.9], 'float64');
        const copied = original.copy();

        expect(copied.dtype).toBe('float64');
        expect(copied.get([0])).toBe(1.5);
        expect(copied.get([1])).toBe(2.7);
        expect(copied.get([2])).toBe(3.9);
      });

      it('preserves float32 dtype', () => {
        const original = zeros([3], 'float32');
        original.data[0] = 1.5;
        original.data[1] = 2.7;
        original.data[2] = 3.9;

        const copied = original.copy();

        expect(copied.dtype).toBe('float32');
        expect(copied.get([0])).toBeCloseTo(1.5);
        expect(copied.get([1])).toBeCloseTo(2.7);
        expect(copied.get([2])).toBeCloseTo(3.9);
      });

      it('preserves int32 dtype', () => {
        const original = array([10, -20, 30], 'int32');
        const copied = original.copy();

        expect(copied.dtype).toBe('int32');
        expect(copied.get([0])).toBe(10);
        expect(copied.get([1])).toBe(-20);
        expect(copied.get([2])).toBe(30);
      });

      it('preserves int8 dtype', () => {
        const original = array([10, -20, 30], 'int8');
        const copied = original.copy();

        expect(copied.dtype).toBe('int8');
        expect(copied.get([0])).toBe(10);
        expect(copied.get([1])).toBe(-20);
        expect(copied.get([2])).toBe(30);
      });

      it('preserves uint8 dtype', () => {
        const original = array([10, 200, 255], 'uint8');
        const copied = original.copy();

        expect(copied.dtype).toBe('uint8');
        expect(copied.get([0])).toBe(10);
        expect(copied.get([1])).toBe(200);
        expect(copied.get([2])).toBe(255);
      });

      it('preserves int64 dtype', () => {
        const original = zeros([3], 'int64');
        original.data[0] = 1000000000000n;
        original.data[1] = -2000000000000n;
        original.data[2] = 3000000000000n;

        const copied = original.copy();

        expect(copied.dtype).toBe('int64');
        expect(copied.get([0])).toBe(1000000000000n);
        expect(copied.get([1])).toBe(-2000000000000n);
        expect(copied.get([2])).toBe(3000000000000n);
      });

      it('preserves uint64 dtype', () => {
        const original = zeros([3], 'uint64');
        original.data[0] = 1000000000000n;
        original.data[1] = 2000000000000n;
        original.data[2] = 18446744073709551615n; // Max uint64

        const copied = original.copy();

        expect(copied.dtype).toBe('uint64');
        expect(copied.get([0])).toBe(1000000000000n);
        expect(copied.get([1])).toBe(2000000000000n);
        expect(copied.get([2])).toBe(18446744073709551615n);
      });

      it('preserves bool dtype', () => {
        const original = zeros([3], 'bool');
        original.data[0] = 1;
        original.data[1] = 0;
        original.data[2] = 1;

        const copied = original.copy();

        expect(copied.dtype).toBe('bool');
        expect(copied.get([0])).toBe(1);
        expect(copied.get([1])).toBe(0);
        expect(copied.get([2])).toBe(1);
      });
    });

    describe('shape preservation', () => {
      it('preserves shape of 1D array', () => {
        const original = array([1, 2, 3, 4, 5]);
        const copied = original.copy();

        expect(copied.shape).toEqual([5]);
        expect(copied.ndim).toBe(1);
        expect(copied.size).toBe(5);
      });

      it('preserves shape of 2D array', () => {
        const original = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const copied = original.copy();

        expect(copied.shape).toEqual([2, 3]);
        expect(copied.ndim).toBe(2);
        expect(copied.size).toBe(6);
      });

      it('preserves shape of 3D array', () => {
        const original = array([
          [
            [1, 2],
            [3, 4],
          ],
          [
            [5, 6],
            [7, 8],
          ],
        ]);
        const copied = original.copy();

        expect(copied.shape).toEqual([2, 2, 2]);
        expect(copied.ndim).toBe(3);
        expect(copied.size).toBe(8);
      });

      it('preserves shape with size 1 dimensions', () => {
        const original = array([[1], [2], [3]]);
        const copied = original.copy();

        expect(copied.shape).toEqual([3, 1]);
        expect(copied.ndim).toBe(2);
        expect(copied.size).toBe(3);
      });
    });

    describe('data independence', () => {
      it('ensures copied data is independent (not shared)', () => {
        const original = array([1, 2, 3, 4, 5]);
        const copied = original.copy();

        // Get pointers to data
        const originalData = original.data;
        const copiedData = copied.data;

        // Data buffers should be different objects
        expect(originalData).not.toBe(copiedData);

        // Modify original data directly
        originalData[0] = 99;

        // Copied data should be unchanged
        expect(copiedData[0]).toBe(1);
      });

      it('creates deep copy for BigInt arrays', () => {
        const original = zeros([3], 'int64');
        original.data[0] = 100n;
        original.data[1] = 200n;
        original.data[2] = 300n;

        const copied = original.copy();

        // Modify original
        original.data[0] = 999n;

        // Copy should be unchanged
        expect(copied.get([0])).toBe(100n);
        expect(original.get([0])).toBe(999n);
      });
    });

    describe('edge cases', () => {
      it('copies empty array', () => {
        const original = array([]);
        const copied = original.copy();

        expect(copied.shape).toEqual([0]);
        expect(copied.size).toBe(0);
      });

      it('copies single element array', () => {
        const original = array([42]);
        const copied = original.copy();

        copied.set([0], 99);

        expect(original.get([0])).toBe(42);
        expect(copied.get([0])).toBe(99);
      });

      it('copies large array', () => {
        const original = zeros([1000], 'float64');
        original.data[500] = 3.14;
        original.data[999] = 2.71;

        const copied = original.copy();

        // Modify copy
        copied.set([500], 99.99);

        // Original should be unchanged
        expect(original.get([500])).toBe(3.14);
        expect(copied.get([500])).toBe(99.99);
        expect(copied.get([999])).toBe(2.71);
      });

      it('handles multiple copies correctly', () => {
        const original = array([1, 2, 3]);
        const copy1 = original.copy();
        const copy2 = original.copy();
        const copy3 = copy1.copy(); // Copy of a copy

        copy1.set([0], 10);
        copy2.set([0], 20);
        copy3.set([0], 30);

        expect(original.get([0])).toBe(1);
        expect(copy1.get([0])).toBe(10);
        expect(copy2.get([0])).toBe(20);
        expect(copy3.get([0])).toBe(30);
      });
    });

    describe('operations on copies', () => {
      it('allows arithmetic operations on copy', () => {
        const original = array([1, 2, 3]);
        const copied = original.copy();

        const result = copied.add(10);

        // Original should be unchanged
        expect(original.toArray()).toEqual([1, 2, 3]);
        // Result should be modified
        expect(result.toArray()).toEqual([11, 12, 13]);
      });

      it('allows reshaping copy without affecting original', () => {
        const original = array([1, 2, 3, 4, 5, 6]);
        const copied = original.copy();

        const reshaped = copied.reshape(2, 3);

        // Original should keep its shape
        expect(original.shape).toEqual([6]);
        // Reshaped should have new shape
        expect(reshaped.shape).toEqual([2, 3]);
      });

      it('allows transposing copy without affecting original', () => {
        const original = array([
          [1, 2, 3],
          [4, 5, 6],
        ]);
        const copied = original.copy();

        const transposed = copied.transpose();

        // Original shape should be unchanged
        expect(original.shape).toEqual([2, 3]);
        // Transposed should have flipped shape
        expect(transposed.shape).toEqual([3, 2]);
      });
    });
  });
});

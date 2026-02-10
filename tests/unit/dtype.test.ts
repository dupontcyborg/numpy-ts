/**
 * Unit tests for dtype support
 */

import { describe, it, expect } from 'vitest';
import { zeros, ones, array, arange, linspace, eye } from '../../src';

describe('DType Support', () => {
  describe('Array creation with dtype', () => {
    it('creates zeros with different dtypes', () => {
      const float64Arr = zeros([2, 3], 'float64');
      expect(float64Arr.dtype).toBe('float64');
      expect(float64Arr.shape).toEqual([2, 3]);

      const float32Arr = zeros([2, 3], 'float32');
      expect(float32Arr.dtype).toBe('float32');

      const int32Arr = zeros([2, 3], 'int32');
      expect(int32Arr.dtype).toBe('int32');

      const uint8Arr = zeros([2, 3], 'uint8');
      expect(uint8Arr.dtype).toBe('uint8');
    });

    it('creates ones with different dtypes', () => {
      const float64Arr = ones([2, 3], 'float64');
      expect(float64Arr.dtype).toBe('float64');
      expect(float64Arr.toArray()).toEqual([
        [1, 1, 1],
        [1, 1, 1],
      ]);

      const int32Arr = ones([2, 3], 'int32');
      expect(int32Arr.dtype).toBe('int32');
      expect(int32Arr.toArray()).toEqual([
        [1, 1, 1],
        [1, 1, 1],
      ]);
    });

    it('creates array with explicit dtype', () => {
      const arr = array([1, 2, 3], 'float32');
      expect(arr.dtype).toBe('float32');
      expect(arr.toArray()).toEqual([1, 2, 3]);
    });

    it('creates arange with dtype', () => {
      const arr = arange(0, 5, 1, 'int32');
      expect(arr.dtype).toBe('int32');
      expect(arr.toArray()).toEqual([0, 1, 2, 3, 4]);
    });

    it('creates linspace with dtype', () => {
      const arr = linspace(0, 1, 5, 'float32');
      expect(arr.dtype).toBe('float32');
      expect(arr.shape).toEqual([5]);
    });

    it('creates eye with dtype', () => {
      const arr = eye(3, undefined, 0, 'int32');
      expect(arr.dtype).toBe('int32');
      expect(arr.toArray()).toEqual([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ]);
    });
  });

  describe('astype() method', () => {
    it('converts float64 to float32', () => {
      const arr = array([1.5, 2.5, 3.5]);
      expect(arr.dtype).toBe('float64');

      const converted = arr.astype('float32');
      expect(converted.dtype).toBe('float32');
      expect(converted.toArray()).toEqual([1.5, 2.5, 3.5]);
    });

    it('converts float to int (truncates)', () => {
      const arr = array([1.7, 2.3, 3.9]);
      const converted = arr.astype('int32');
      expect(converted.dtype).toBe('int32');
      // Values are truncated (not rounded) when converting to int
      expect(converted.toArray()).toEqual([1, 2, 3]);
    });

    it('converts to bool dtype', () => {
      const arr = array([0, 1, 2, 0, -1]);
      const converted = arr.astype('bool');
      expect(converted.dtype).toBe('bool');
      expect(converted.toArray()).toEqual([0, 1, 1, 0, 1]);
    });

    it('converts bool to int', () => {
      const arr = array([0, 1, 1, 0], 'bool');
      const converted = arr.astype('int32');
      expect(converted.dtype).toBe('int32');
      expect(converted.toArray()).toEqual([0, 1, 1, 0]);
    });

    it('returns same array with copy=false when dtype matches', () => {
      const arr = array([1, 2, 3]);
      const same = arr.astype('float64', false);
      expect(same).toBe(arr); // Same object
    });

    it('creates copy with copy=true when dtype matches', () => {
      const arr = array([1, 2, 3]);
      const copy = arr.astype('float64', true);
      expect(copy).not.toBe(arr); // Different object
      expect(copy.toArray()).toEqual([1, 2, 3]);
    });
  });

  describe('BigInt dtypes (int64/uint64)', () => {
    it('creates int64 array', () => {
      const arr = zeros([2, 3], 'int64');
      expect(arr.dtype).toBe('int64');
      expect(arr.shape).toEqual([2, 3]);
    });

    it('creates ones with int64', () => {
      const arr = ones([2, 2], 'int64');
      expect(arr.dtype).toBe('int64');
      // BigInt arrays contain bigint values
      expect(arr.data[0]).toBe(BigInt(1));
    });

    it('creates uint64 array', () => {
      const arr = zeros([2, 3], 'uint64');
      expect(arr.dtype).toBe('uint64');
    });

    it('creates arange with int64', () => {
      const arr = arange(0, 5, 1, 'int64');
      expect(arr.dtype).toBe('int64');
      // BigInt values in data buffer
      expect(arr.data[0]).toBe(BigInt(0));
      expect(arr.data[1]).toBe(BigInt(1));
    });

    it('converts float to int64', () => {
      const arr = array([1.5, 2.7, 3.2]);
      const converted = arr.astype('int64');
      expect(converted.dtype).toBe('int64');
      // Values are rounded when converting to BigInt
      expect(converted.data[0]).toBe(BigInt(2));
      expect(converted.data[1]).toBe(BigInt(3));
      expect(converted.data[2]).toBe(BigInt(3));
    });

    it('converts int64 to float64', () => {
      const arr = ones([2, 2], 'int64');
      const converted = arr.astype('float64');
      expect(converted.dtype).toBe('float64');
      expect(converted.toArray()).toEqual([
        [1, 1],
        [1, 1],
      ]);
    });

    it('converts between int64 and uint64', () => {
      const arr = ones([2, 2], 'int64');
      const converted = arr.astype('uint64');
      expect(converted.dtype).toBe('uint64');
      expect(converted.data[0]).toBe(BigInt(1));
    });
  });

  // Complex dtypes are no longer supported (removed for simplicity)

  describe('dtype property', () => {
    it('returns correct dtype for default arrays', () => {
      const arr = array([1, 2, 3]);
      expect(arr.dtype).toBe('float64');
    });

    it('returns correct dtype for explicitly typed arrays', () => {
      expect(zeros([2, 3], 'float32').dtype).toBe('float32');
      expect(zeros([2, 3], 'int32').dtype).toBe('int32');
      expect(zeros([2, 3], 'uint8').dtype).toBe('uint8');
      expect(zeros([2, 3], 'bool').dtype).toBe('bool');
    });
  });

  describe('Edge cases', () => {
    it('handles empty arrays with dtype', () => {
      const arr = zeros([0], 'float32');
      expect(arr.dtype).toBe('float32');
      expect(arr.size).toBe(0);
    });

    it('handles 1D arrays with dtype', () => {
      const arr = ones([5], 'int32');
      expect(arr.dtype).toBe('int32');
      expect(arr.shape).toEqual([5]);
    });

    it('handles multidimensional arrays with dtype', () => {
      const arr = zeros([2, 3, 4], 'float32');
      expect(arr.dtype).toBe('float32');
      expect(arr.shape).toEqual([2, 3, 4]);
    });
  });
});

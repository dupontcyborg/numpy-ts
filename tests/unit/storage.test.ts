/**
 * Unit tests for ArrayStorage internals
 * Tests the low-level storage abstraction directly
 */

import { describe, it, expect } from 'vitest';
import { ArrayStorage } from '../../src/common/storage';

describe('ArrayStorage', () => {
  describe('Basic properties', () => {
    it('has correct shape, size, and ndim', () => {
      const storage = ArrayStorage.zeros([2, 3, 4], 'float64');
      expect(storage.shape).toEqual([2, 3, 4]);
      expect(storage.size).toBe(24);
      expect(storage.ndim).toBe(3);
      expect(storage.dtype).toBe('float64');
    });

    it('returns data buffer', () => {
      const storage = ArrayStorage.zeros([2, 3], 'float32');
      expect(storage.data).toBeInstanceOf(Float32Array);
      expect(storage.data.length).toBe(6);
    });

    it('returns strides for row-major layout', () => {
      const storage = ArrayStorage.zeros([2, 3, 4], 'float64');
      expect(storage.strides).toEqual([12, 4, 1]);
    });
  });

  describe('isCContiguous', () => {
    it('returns true for 1-d arrays with stride 1', () => {
      const storage = ArrayStorage.zeros([5], 'float64');
      expect(storage.isCContiguous).toBe(true);
    });

    it('returns true for row-major 2-d arrays', () => {
      const storage = ArrayStorage.zeros([3, 4], 'float64');
      expect(storage.isCContiguous).toBe(true);
    });

    it('returns true for row-major 3-d arrays', () => {
      const storage = ArrayStorage.zeros([2, 3, 4], 'float64');
      expect(storage.isCContiguous).toBe(true);
    });
  });

  describe('isFContiguous', () => {
    it('returns true for 1-d arrays with stride 1', () => {
      const storage = ArrayStorage.zeros([5], 'float64');
      expect(storage.isFContiguous).toBe(true);
    });

    it('returns false for row-major 2-d arrays (not F-contiguous)', () => {
      const storage = ArrayStorage.zeros([3, 4], 'float64');
      expect(storage.isFContiguous).toBe(false);
    });
  });

  describe('Static factory methods', () => {
    describe('zeros()', () => {
      it('creates zeros with default dtype', () => {
        const storage = ArrayStorage.zeros([2, 3]);
        expect(storage.dtype).toBe('float64');
        expect(storage.size).toBe(6);
        expect(storage.data[0]).toBe(0);
        expect(storage.data[5]).toBe(0);
      });

      it('creates zeros with float32 dtype', () => {
        const storage = ArrayStorage.zeros([2, 3], 'float32');
        expect(storage.dtype).toBe('float32');
        expect(storage.data).toBeInstanceOf(Float32Array);
      });

      it('creates zeros with int32 dtype', () => {
        const storage = ArrayStorage.zeros([2, 3], 'int32');
        expect(storage.dtype).toBe('int32');
        expect(storage.data).toBeInstanceOf(Int32Array);
      });

      it('creates zeros with int64 dtype', () => {
        const storage = ArrayStorage.zeros([2, 3], 'int64');
        expect(storage.dtype).toBe('int64');
        expect(storage.data).toBeInstanceOf(BigInt64Array);
        expect(storage.data[0]).toBe(BigInt(0));
      });

      it('creates zeros with uint64 dtype', () => {
        const storage = ArrayStorage.zeros([2, 3], 'uint64');
        expect(storage.dtype).toBe('uint64');
        expect(storage.data).toBeInstanceOf(BigUint64Array);
      });

      it('creates zeros with bool dtype', () => {
        const storage = ArrayStorage.zeros([2, 3], 'bool');
        expect(storage.dtype).toBe('bool');
        expect(storage.data).toBeInstanceOf(Uint8Array);
      });
    });

    describe('ones()', () => {
      it('creates ones with default dtype', () => {
        const storage = ArrayStorage.ones([2, 3]);
        expect(storage.dtype).toBe('float64');
        expect(storage.size).toBe(6);
        expect(storage.data[0]).toBe(1);
        expect(storage.data[5]).toBe(1);
      });

      it('creates ones with float32 dtype', () => {
        const storage = ArrayStorage.ones([2, 3], 'float32');
        expect(storage.dtype).toBe('float32');
        expect(storage.data[0]).toBe(1);
        expect(storage.data[5]).toBe(1);
      });

      it('creates ones with int32 dtype', () => {
        const storage = ArrayStorage.ones([2, 3], 'int32');
        expect(storage.dtype).toBe('int32');
        expect(storage.data[0]).toBe(1);
        expect(storage.data[5]).toBe(1);
      });

      it('creates ones with int64 dtype (BigInt)', () => {
        const storage = ArrayStorage.ones([2, 3], 'int64');
        expect(storage.dtype).toBe('int64');
        expect(storage.data).toBeInstanceOf(BigInt64Array);
        expect(storage.data[0]).toBe(BigInt(1));
        expect(storage.data[5]).toBe(BigInt(1));
      });

      it('creates ones with uint64 dtype (BigInt)', () => {
        const storage = ArrayStorage.ones([2, 3], 'uint64');
        expect(storage.dtype).toBe('uint64');
        expect(storage.data).toBeInstanceOf(BigUint64Array);
        expect(storage.data[0]).toBe(BigInt(1));
        expect(storage.data[5]).toBe(BigInt(1));
      });

      it('creates ones with bool dtype', () => {
        const storage = ArrayStorage.ones([2, 3], 'bool');
        expect(storage.dtype).toBe('bool');
        expect(storage.data[0]).toBe(1);
        expect(storage.data[5]).toBe(1);
      });
    });

    describe('fromData()', () => {
      it('creates storage from Float64Array', () => {
        const data = new Float64Array([1, 2, 3, 4, 5, 6]);
        const storage = ArrayStorage.fromData(data, [2, 3], 'float64');
        expect(storage.shape).toEqual([2, 3]);
        expect(storage.dtype).toBe('float64');
        expect(storage.data).toBe(data);
      });

      it('creates storage from Int32Array', () => {
        const data = new Int32Array([1, 2, 3, 4]);
        const storage = ArrayStorage.fromData(data, [2, 2], 'int32');
        expect(storage.shape).toEqual([2, 2]);
        expect(storage.dtype).toBe('int32');
      });

      it('creates storage from BigInt64Array', () => {
        const data = new BigInt64Array([BigInt(1), BigInt(2), BigInt(3), BigInt(4)]);
        const storage = ArrayStorage.fromData(data, [2, 2], 'int64');
        expect(storage.shape).toEqual([2, 2]);
        expect(storage.dtype).toBe('int64');
        expect(storage.data[0]).toBe(BigInt(1));
      });

      it('creates storage from Uint8Array (bool)', () => {
        const data = new Uint8Array([1, 0, 1, 0]);
        const storage = ArrayStorage.fromData(data, [2, 2], 'bool');
        expect(storage.shape).toEqual([2, 2]);
        expect(storage.dtype).toBe('bool');
      });
    });
  });

  describe('copy()', () => {
    it('creates deep copy of float64 storage', () => {
      const original = ArrayStorage.ones([2, 3], 'float64');
      const copy = original.copy();

      expect(copy.shape).toEqual(original.shape);
      expect(copy.dtype).toBe(original.dtype);
      expect(copy.data).not.toBe(original.data); // Different buffer

      // Modify original
      original.data[0] = 99;
      expect(copy.data[0]).toBe(1); // Copy unchanged
    });

    it('creates deep copy of float32 storage', () => {
      const original = ArrayStorage.ones([2, 3], 'float32');
      const copy = original.copy();

      expect(copy.dtype).toBe('float32');
      expect(copy.data).not.toBe(original.data);
      expect(copy.data[0]).toBe(1);
    });

    it('creates deep copy of int32 storage', () => {
      const original = ArrayStorage.ones([2, 3], 'int32');
      const copy = original.copy();

      expect(copy.dtype).toBe('int32');
      expect(copy.data).not.toBe(original.data);
      expect(copy.data[0]).toBe(1);
    });

    it('creates deep copy of int64 storage (BigInt)', () => {
      const original = ArrayStorage.ones([2, 3], 'int64');
      const copy = original.copy();

      expect(copy.dtype).toBe('int64');
      expect(copy.data).not.toBe(original.data);
      expect(copy.data[0]).toBe(BigInt(1));

      // Modify original
      (original.data as BigInt64Array)[0] = BigInt(99);
      expect(copy.data[0]).toBe(BigInt(1)); // Copy unchanged
    });

    it('creates deep copy of uint64 storage (BigInt)', () => {
      const original = ArrayStorage.ones([2, 3], 'uint64');
      const copy = original.copy();

      expect(copy.dtype).toBe('uint64');
      expect(copy.data).not.toBe(original.data);
      expect(copy.data[0]).toBe(BigInt(1));

      // Modify original
      (original.data as BigUint64Array)[0] = BigInt(99);
      expect(copy.data[0]).toBe(BigInt(1)); // Copy unchanged
    });

    it('creates deep copy of bool storage', () => {
      const original = ArrayStorage.ones([2, 3], 'bool');
      const copy = original.copy();

      expect(copy.dtype).toBe('bool');
      expect(copy.data).not.toBe(original.data);
      expect(copy.data[0]).toBe(1);
    });

    it('copies arrays with different shapes', () => {
      const original = ArrayStorage.ones([3, 4, 5], 'float64');
      const copy = original.copy();

      expect(copy.shape).toEqual([3, 4, 5]);
      expect(copy.size).toBe(60);
    });
  });

  describe('Edge cases', () => {
    it('handles 1-d arrays', () => {
      const storage = ArrayStorage.zeros([5], 'float64');
      expect(storage.shape).toEqual([5]);
      expect(storage.ndim).toBe(1);
      expect(storage.size).toBe(5);
    });

    it('handles 2-d arrays', () => {
      const storage = ArrayStorage.zeros([2, 3], 'float64');
      expect(storage.shape).toEqual([2, 3]);
      expect(storage.ndim).toBe(2);
      expect(storage.size).toBe(6);
    });

    it('handles 3-d arrays', () => {
      const storage = ArrayStorage.zeros([2, 3, 4], 'float64');
      expect(storage.shape).toEqual([2, 3, 4]);
      expect(storage.ndim).toBe(3);
      expect(storage.size).toBe(24);
    });
  });

  describe('dtype variations', () => {
    it('supports all integer dtypes', () => {
      const dtypes: Array<
        'int8' | 'int16' | 'int32' | 'int64' | 'uint8' | 'uint16' | 'uint32' | 'uint64'
      > = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'];

      dtypes.forEach((dtype) => {
        const storage = ArrayStorage.zeros([2, 2], dtype);
        expect(storage.dtype).toBe(dtype);
        expect(storage.size).toBe(4);
      });
    });

    it('supports all float dtypes', () => {
      const dtypes: Array<'float32' | 'float64'> = ['float32', 'float64'];

      dtypes.forEach((dtype) => {
        const storage = ArrayStorage.zeros([2, 2], dtype);
        expect(storage.dtype).toBe(dtype);
        expect(storage.size).toBe(4);
      });
    });
  });
});

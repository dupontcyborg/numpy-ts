import { describe, it, expect } from 'vitest';
import { array, zeros } from '../../src';
import { NDArrayCore } from '../../src/common/ndarray-core';

/**
 * Tests for uncovered properties/methods on src/common/ndarray-core.ts
 */

describe('NDArrayCore Coverage', () => {
  // ========================================
  // Properties
  // ========================================
  describe('Properties', () => {
    it('flags.C_CONTIGUOUS is true for standard array', () => {
      const a = array([1, 2, 3]);
      expect(a.flags.C_CONTIGUOUS).toBe(true);
    });

    it('flags.F_CONTIGUOUS', () => {
      const a = array([1, 2, 3]);
      // 1D array is both C and F contiguous
      expect(a.flags.F_CONTIGUOUS).toBeDefined();
    });

    it('flags.OWNDATA is true for owned array', () => {
      const a = array([1, 2, 3]);
      expect(a.flags.OWNDATA).toBe(true);
    });

    it('flags.OWNDATA is false for view', () => {
      const a = array([1, 2, 3, 4, 5]);
      const v = a.slice('1:4');
      expect(v.flags.OWNDATA).toBe(false);
    });

    it('base is null for owned array', () => {
      const a = array([1, 2, 3]);
      expect(a.base).toBeNull();
    });

    it('base is set for view', () => {
      const a = array([1, 2, 3, 4]);
      const v = a.slice('1:3');
      expect(v.base).not.toBeNull();
    });

    it('itemsize for different dtypes', () => {
      expect(array([1], 'float64').itemsize).toBe(8);
      expect(array([1], 'float32').itemsize).toBe(4);
      expect(array([1], 'int32').itemsize).toBe(4);
      expect(array([1], 'int16').itemsize).toBe(2);
      expect(array([1], 'uint8').itemsize).toBe(1);
    });

    it('nbytes', () => {
      const a = array([1, 2, 3, 4], 'float32');
      expect(a.nbytes).toBe(16); // 4 * 4 bytes
    });

    it('strides', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(a.strides).toBeDefined();
      expect(a.strides.length).toBe(2);
    });

    it('storage', () => {
      const a = array([1, 2, 3]);
      expect(a.storage).toBeDefined();
    });
  });

  // ========================================
  // fill()
  // ========================================
  describe('fill()', () => {
    it('fills with number', () => {
      const a = array([0, 0, 0]);
      a.fill(5);
      expect(a.toArray()).toEqual([5, 5, 5]);
    });

    it('fills bigint array', () => {
      const a = array([0n, 0n, 0n], 'int64');
      a.fill(7n);
      expect(a.toArray()).toEqual([7n, 7n, 7n]);
    });

    it('fills bool array', () => {
      const a = zeros([3], 'bool');
      a.fill(1);
      expect(a.toArray()).toEqual([1, 1, 1]);
    });

    it('fills with number converting to bigint', () => {
      const a = array([0n, 0n], 'int64');
      a.fill(5 as unknown as bigint);
      // Should convert number to bigint
      expect(a.toArray()).toEqual([5n, 5n]);
    });
  });

  // ========================================
  // [Symbol.iterator]()
  // ========================================
  describe('Iterator', () => {
    it('iterates 1D array yields scalars', () => {
      const a = array([10, 20, 30]);
      const values = [...a];
      expect(values).toEqual([10, 20, 30]);
    });

    it('iterates 2D array yields rows', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const rows = [...a];
      expect(rows).toHaveLength(2);
      // Each row is an NDArrayCore
      expect((rows[0] as NDArrayCore).toArray()).toEqual([1, 2]);
      expect((rows[1] as NDArrayCore).toArray()).toEqual([3, 4]);
    });
  });

  // ========================================
  // tobytes()
  // ========================================
  describe('tobytes()', () => {
    it('returns ArrayBuffer for contiguous array', () => {
      const a = array([1, 2, 3], 'int32');
      const bytes = a.tobytes();
      expect(bytes).toBeInstanceOf(ArrayBuffer);
      expect(bytes.byteLength).toBe(12); // 3 * 4 bytes
    });

    it('returns ArrayBuffer for non-contiguous (sliced) array', () => {
      const a = array([1, 2, 3, 4, 5, 6], 'int32');
      const sliced = a.slice('::2'); // [1, 3, 5] - non-contiguous
      const bytes = sliced.tobytes();
      expect(bytes).toBeInstanceOf(ArrayBuffer);
    });
  });

  // ========================================
  // toString()
  // ========================================
  describe('toString()', () => {
    it('includes shape and dtype', () => {
      const a = array([1, 2, 3]);
      const str = a.toString();
      expect(str).toContain('NDArray');
      expect(str).toContain('[3]');
    });
  });

  // ========================================
  // toArray()
  // ========================================
  describe('toArray()', () => {
    it('converts 1D array', () => {
      const a = array([1, 2, 3]);
      expect(a.toArray()).toEqual([1, 2, 3]);
    });

    it('converts 2D array', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(a.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('converts 3D array', () => {
      const a = array([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
      expect(a.toArray()).toEqual([
        [
          [1, 2],
          [3, 4],
        ],
        [
          [5, 6],
          [7, 8],
        ],
      ]);
    });
  });

  // ========================================
  // tolist()
  // ========================================
  describe('tolist()', () => {
    it('same as toArray', () => {
      const a = array([1, 2, 3]);
      expect(a.tolist()).toEqual(a.toArray());
    });
  });

  // ========================================
  // item()
  // ========================================
  describe('item()', () => {
    it('returns scalar from size-1 array', () => {
      const a = array([42]);
      expect(a.item()).toBe(42);
    });

    it('throws for multi-element array without index', () => {
      const a = array([1, 2, 3]);
      expect(() => a.item()).toThrow('can only convert');
    });

    it('returns element by flat index', () => {
      const a = array([10, 20, 30]);
      expect(a.item(1)).toBe(20);
    });

    it('throws for out-of-bounds flat index', () => {
      const a = array([1, 2, 3]);
      expect(() => a.item(5)).toThrow('out of bounds');
    });

    it('returns element by multi-index', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(a.item(1, 0)).toBe(3);
    });
  });

  // ========================================
  // copy()
  // ========================================
  describe('copy()', () => {
    it('returns deep copy', () => {
      const a = array([1, 2, 3]);
      const b = a.copy();
      b.iset(0, 99);
      expect(a.iget(0)).toBe(1);
      expect(b.iget(0)).toBe(99);
    });
  });

  // ========================================
  // astype() edge cases
  // ========================================
  describe('astype()', () => {
    it('same dtype, no copy returns self', () => {
      const a = array([1, 2, 3], 'float64');
      const b = a.astype('float64', false);
      // Should be the same instance
      expect(b).toBe(a);
    });

    it('same dtype, copy returns new array', () => {
      const a = array([1, 2, 3], 'float64');
      const b = a.astype('float64', true);
      expect(b).not.toBe(a);
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('bool to float', () => {
      const a = array([0, 1, 1, 0], 'bool');
      const b = a.astype('float64');
      expect(b.dtype).toBe('float64');
      expect(b.toArray()).toEqual([0, 1, 1, 0]);
    });

    it('float to bool', () => {
      const a = array([0.0, 1.5, -2.3, 0.0], 'float64');
      const b = a.astype('bool');
      expect(b.dtype).toBe('bool');
      expect(b.toArray()).toEqual([0, 1, 1, 0]);
    });

    it('bigint to float', () => {
      const a = array([1n, 2n, 3n], 'int64');
      const b = a.astype('float64');
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('float to bigint', () => {
      const a = array([1.5, 2.5, 3.5], 'float64');
      const b = a.astype('int64');
      expect(b.dtype).toBe('int64');
      expect(b.toArray()).toEqual([2n, 3n, 4n]);
    });

    it('bigint to bool', () => {
      const a = array([0n, 1n, 5n], 'int64');
      const b = a.astype('bool');
      expect(b.toArray()).toEqual([0, 1, 1]);
    });

    it('int to int (non-bigint)', () => {
      const a = array([1, 2, 3], 'int32');
      const b = a.astype('int16');
      expect(b.dtype).toBe('int16');
    });

    it('bigint to bigint', () => {
      const a = array([1n, 2n], 'int64');
      const b = a.astype('uint64');
      expect(b.dtype).toBe('uint64');
    });
  });

  // ========================================
  // slice() edge cases
  // ========================================
  describe('slice()', () => {
    it('empty slice returns self', () => {
      const a = array([1, 2, 3]);
      const s = a.slice();
      expect(s).toBe(a);
    });

    it('throws for too many indices', () => {
      const a = array([1, 2, 3]);
      expect(() => a.slice('0', '1')).toThrow('Too many indices');
    });

    it('scalar indexing removes dimension', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const row = a.slice('0');
      expect(row.shape).toEqual([3]);
      expect(row.toArray()).toEqual([1, 2, 3]);
    });

    it('range slicing preserves dimension', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const sub = a.slice('0:1', '1:3');
      expect(sub.shape).toEqual([1, 2]);
    });

    it('step slicing', () => {
      const a = array([1, 2, 3, 4, 5, 6]);
      const s = a.slice('::2');
      expect(s.toArray()).toEqual([1, 3, 5]);
    });

    it('base is set for sliced view', () => {
      const a = array([1, 2, 3, 4]);
      const s = a.slice('1:3');
      expect(s.base).not.toBeNull();
    });
  });

  // ========================================
  // get/set edge cases
  // ========================================
  describe('get/set edge cases', () => {
    it('get() throws for wrong number of indices', () => {
      const a = array([1, 2, 3]);
      expect(() => a.get([0, 0])).toThrow('dimensions');
    });

    it('get() throws for out of bounds', () => {
      const a = array([1, 2, 3]);
      expect(() => a.get([10])).toThrow('out of bounds');
    });

    it('set() throws for wrong number of indices', () => {
      const a = array([1, 2, 3]);
      expect(() => a.set([0, 0], 5)).toThrow('dimensions');
    });

    it('set() throws for out of bounds', () => {
      const a = array([1, 2, 3]);
      expect(() => a.set([10], 5)).toThrow('out of bounds');
    });

    it('negative index wraps around', () => {
      const a = array([10, 20, 30]);
      expect(a.get([-1])).toBe(30);
      expect(a.get([-2])).toBe(20);
    });

    it('set() with bigint dtype', () => {
      const a = array([0n, 0n], 'int64');
      a.set([0], 42n);
      expect(a.get([0])).toBe(42n);
    });

    it('set() converts number to bigint for bigint dtype', () => {
      const a = array([0n, 0n], 'int64');
      a.set([0], 5 as unknown as bigint);
      expect(a.get([0])).toBe(5n);
    });

    it('set() with bool dtype', () => {
      const a = zeros([3], 'bool');
      a.set([1], 1);
      expect(a.get([1])).toBe(1);
    });
  });

  // ========================================
  // _fromStorage static method
  // ========================================
  describe('_fromStorage()', () => {
    it('creates NDArrayCore from storage', () => {
      const a = array([1, 2, 3]);
      const core = NDArrayCore._fromStorage(a.storage);
      expect(core.toArray()).toEqual([1, 2, 3]);
    });

    it('creates NDArrayCore with base', () => {
      const a = array([1, 2, 3]);
      const core = NDArrayCore._fromStorage(a.storage, a);
      expect(core.base).toBe(a);
    });
  });
});

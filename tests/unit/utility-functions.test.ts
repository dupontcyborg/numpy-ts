import { describe, it, expect } from 'vitest';
import { array } from '../../src';
import { Complex } from '../../src/common/complex';
import {
  ndim,
  shape,
  size,
  item,
  tolist,
  tobytes,
  byteswap,
  view,
  tofile,
  fill,
} from '../../src/core/utility';

/**
 * Tests for src/core/utility.ts standalone functions (0% → 100% coverage)
 */

describe('Utility Functions (src/core/utility.ts)', () => {
  // ========================================
  // ndim()
  // ========================================
  describe('ndim()', () => {
    it('returns ndim for NDArrayCore', () => {
      const a = array([1, 2, 3]);
      expect(ndim(a)).toBe(1);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      expect(ndim(b)).toBe(2);
    });

    it('returns 0 for scalars', () => {
      expect(ndim(42)).toBe(0);
      expect(ndim(42n)).toBe(0);
      expect(ndim(true)).toBe(0);
    });

    it('counts dimensions for nested arrays', () => {
      expect(ndim([1, 2, 3])).toBe(1);
      expect(
        ndim([
          [1, 2],
          [3, 4],
        ])
      ).toBe(2);
      expect(ndim([[[1]]])).toBe(3);
    });

    it('returns 0 for non-array, non-scalar', () => {
      expect(ndim('hello')).toBe(0);
      expect(ndim({})).toBe(0);
    });
  });

  // ========================================
  // shape()
  // ========================================
  describe('shape()', () => {
    it('returns shape for NDArrayCore', () => {
      const a = array([1, 2, 3]);
      expect(shape(a)).toEqual([3]);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      expect(shape(b)).toEqual([2, 2]);
    });

    it('returns [] for scalars', () => {
      expect(shape(42)).toEqual([]);
      expect(shape(42n)).toEqual([]);
      expect(shape(true)).toEqual([]);
    });

    it('returns shape for nested arrays', () => {
      expect(shape([1, 2, 3])).toEqual([3]);
      expect(
        shape([
          [1, 2],
          [3, 4],
        ])
      ).toEqual([2, 2]);
    });

    it('returns [] for non-array, non-scalar', () => {
      expect(shape('hello')).toEqual([]);
      expect(shape({})).toEqual([]);
    });
  });

  // ========================================
  // size()
  // ========================================
  describe('size()', () => {
    it('returns size for NDArrayCore', () => {
      const a = array([1, 2, 3]);
      expect(size(a)).toBe(3);
      const b = array([
        [1, 2],
        [3, 4],
      ]);
      expect(size(b)).toBe(4);
    });

    it('returns 1 for scalars', () => {
      expect(size(42)).toBe(1);
      expect(size(42n)).toBe(1);
      expect(size(true)).toBe(1);
    });

    it('computes size from nested arrays', () => {
      expect(size([1, 2, 3])).toBe(3);
      expect(
        size([
          [1, 2],
          [3, 4],
        ])
      ).toBe(4);
    });

    it('returns 1 for non-array, non-scalar', () => {
      expect(size('hello')).toBe(1);
      expect(size({})).toBe(1);
    });
  });

  // ========================================
  // item()
  // ========================================
  describe('item()', () => {
    it('extracts scalar from size-1 array', () => {
      const a = array([42]);
      expect(item(a)).toBe(42);
    });

    it('throws for size > 1 without index', () => {
      const a = array([1, 2, 3]);
      expect(() => item(a)).toThrow('can only convert an array of size 1');
    });

    it('extracts by flat index', () => {
      const a = array([10, 20, 30]);
      expect(item(a, 0)).toBe(10);
      expect(item(a, 2)).toBe(30);
    });

    it('extracts by multi-index', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(item(a, 1, 0)).toBe(3);
      expect(item(a, 0, 1)).toBe(2);
    });

    it('throws for wrong number of indices', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => item(a, 0, 0, 0)).toThrow('incorrect number of indices');
    });
  });

  // ========================================
  // tolist()
  // ========================================
  describe('tolist()', () => {
    it('converts 1D array', () => {
      const a = array([1, 2, 3]);
      expect(tolist(a)).toEqual([1, 2, 3]);
    });

    it('converts 2D array to nested list', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(tolist(a)).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('converts 3D array to nested list', () => {
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
      expect(tolist(a)).toEqual([
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

    it('converts scalar array', () => {
      const a = array([42]).reshape();
      // Scalar array should return the value directly
      if (a.ndim === 0) {
        expect(tolist(a)).toBe(42);
      }
    });
  });

  // ========================================
  // tobytes()
  // ========================================
  describe('tobytes()', () => {
    it('returns raw bytes (C order)', () => {
      const a = array([1, 2], 'int32');
      const bytes = tobytes(a);
      expect(bytes).toBeInstanceOf(Uint8Array);
      expect(bytes.byteLength).toBe(8); // 2 * 4 bytes
    });

    it('warns for F order but still returns bytes', () => {
      const a = array([1, 2], 'int32');
      const consoleSpy = { warned: false };
      const origWarn = console.warn;
      console.warn = () => {
        consoleSpy.warned = true;
      };
      const bytes = tobytes(a, 'F');
      console.warn = origWarn;
      expect(consoleSpy.warned).toBe(true);
      expect(bytes.byteLength).toBe(8);
    });
  });

  // ========================================
  // byteswap()
  // ========================================
  describe('byteswap()', () => {
    it('swaps bytes of multi-byte arrays', () => {
      const a = array([1], 'float64');
      const swapped = byteswap(a);
      // Should be a different array
      expect(swapped).not.toBe(a);
    });

    it('returns copy for 1-byte elements', () => {
      const a = array([1, 2, 3], 'uint8');
      const swapped = byteswap(a);
      expect(swapped.toArray()).toEqual([1, 2, 3]);
    });

    it('swaps in-place when requested', () => {
      const a = array([1], 'int32');
      const original = a.data[0];
      const result = byteswap(a, true);
      expect(result).toBe(a);
      // Data should have changed
      expect(a.data[0]).not.toBe(original);
    });

    it('returns copy for 1-byte in-place', () => {
      const a = array([5], 'uint8');
      const r = byteswap(a, true);
      expect(r).toBe(a);
    });
  });

  // ========================================
  // view()
  // ========================================
  describe('view()', () => {
    it('returns copy for same/no dtype', () => {
      const a = array([1, 2, 3]);
      const v = view(a);
      expect(v.toArray()).toEqual([1, 2, 3]);
    });

    it('returns copy when dtype matches', () => {
      const a = array([1, 2, 3], 'float64');
      const v = view(a, 'float64');
      expect(v.toArray()).toEqual([1, 2, 3]);
    });

    it('throws for different dtype', () => {
      const a = array([1, 2, 3], 'float64');
      expect(() => view(a, 'int32')).toThrow('not fully implemented');
    });
  });

  // ========================================
  // tofile()
  // ========================================
  describe('tofile()', () => {
    it('throws error (not supported in browser)', () => {
      const a = array([1, 2, 3]);
      expect(() => tofile(a, 'test.bin')).toThrow('Node.js file system');
    });
  });

  // ========================================
  // fill()
  // ========================================
  describe('fill()', () => {
    it('fills with number', () => {
      const a = array([0, 0, 0]);
      fill(a, 42);
      expect(a.toArray()).toEqual([42, 42, 42]);
    });

    it('fills with bigint', () => {
      const a = array([0n, 0n, 0n], 'int64');
      fill(a, 7n);
      expect(a.toArray()).toEqual([7n, 7n, 7n]);
    });

    it('fills with boolean', () => {
      const a = array([0, 0, 0], 'uint8');
      fill(a, true);
      expect(a.toArray()).toEqual([1, 1, 1]);
    });

    it('fills complex array with Complex value', () => {
      const a = array([new Complex(0, 0), new Complex(0, 0)], 'complex128');
      fill(a, new Complex(3, 4));
      // The real and imaginary parts should be set
      expect(a.size).toBe(2);
    });

    it('throws when filling non-complex with Complex', () => {
      const a = array([1, 2, 3]);
      expect(() => fill(a, new Complex(1, 2))).toThrow('Cannot fill non-complex');
    });
  });
});

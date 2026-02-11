import { describe, it, expect } from 'vitest';
import { NDArrayCore } from '../../src/common/ndarray-core';
import { Complex } from '../../src/common/complex';
import { ArrayStorage } from '../../src/common/storage';
import type { DType } from '../../src/common/dtype';

/**
 * Tests for src/common/ndarray-core.ts
 *
 * IMPORTANT: NDArray (full) overrides ALL NDArrayCore methods, so calling
 * array() or zeros() from src/ creates NDArray instances whose methods hit
 * src/full/ndarray.ts, NOT src/common/ndarray-core.ts.
 *
 * To actually cover ndarray-core.ts, we must create NDArrayCore instances
 * directly via ArrayStorage + NDArrayCore.fromStorage().
 */

// Helper: create an NDArrayCore from a number array with the right typed array
function core(data: number[], shape: number[], dtype: DType = 'float64'): NDArrayCore {
  let ta;
  switch (dtype) {
    case 'float64':
      ta = new Float64Array(data);
      break;
    case 'float32':
      ta = new Float32Array(data);
      break;
    case 'int32':
      ta = new Int32Array(data);
      break;
    case 'int16':
      ta = new Int16Array(data);
      break;
    case 'int8':
      ta = new Int8Array(data);
      break;
    case 'uint32':
      ta = new Uint32Array(data);
      break;
    case 'uint16':
      ta = new Uint16Array(data);
      break;
    case 'uint8':
      ta = new Uint8Array(data);
      break;
    default:
      ta = new Float64Array(data);
      break;
  }
  const storage = ArrayStorage.fromData(ta, shape, dtype);
  return NDArrayCore.fromStorage(storage);
}

function coreBigInt(data: bigint[], shape: number[], dtype: DType = 'int64'): NDArrayCore {
  const ta = dtype === 'uint64' ? new BigUint64Array(data) : new BigInt64Array(data);
  const storage = ArrayStorage.fromData(ta, shape, dtype);
  return NDArrayCore.fromStorage(storage);
}

function coreBool(data: number[], shape: number[]): NDArrayCore {
  const ta = new Uint8Array(data);
  const storage = ArrayStorage.fromData(ta, shape, 'bool');
  return NDArrayCore.fromStorage(storage);
}

function coreComplex(reals: number[], imags: number[], shape: number[]): NDArrayCore {
  const interleaved = new Float64Array(reals.length * 2);
  for (let i = 0; i < reals.length; i++) {
    interleaved[i * 2] = reals[i]!;
    interleaved[i * 2 + 1] = imags[i]!;
  }
  const storage = ArrayStorage.fromData(interleaved, shape, 'complex128');
  return NDArrayCore.fromStorage(storage);
}

function core0d(value: number): NDArrayCore {
  const data = new Float64Array([value]);
  const storage = ArrayStorage.fromData(data, [], 'float64');
  return NDArrayCore.fromStorage(storage);
}

describe('NDArrayCore Coverage', () => {
  // ========================================
  // Properties
  // ========================================
  describe('Properties', () => {
    it('shape, ndim, size, dtype, data', () => {
      const a = core([1, 2, 3, 4, 5, 6], [2, 3]);
      expect(a.shape).toEqual([2, 3]);
      expect(a.ndim).toBe(2);
      expect(a.size).toBe(6);
      expect(a.dtype).toBe('float64');
      expect(a.data).toBeInstanceOf(Float64Array);
    });

    it('strides', () => {
      const a = core([1, 2, 3, 4, 5, 6], [2, 3]);
      expect(a.strides).toBeDefined();
      expect(a.strides.length).toBe(2);
    });

    it('storage', () => {
      const a = core([1, 2, 3], [3]);
      expect(a.storage).toBeInstanceOf(ArrayStorage);
    });

    it('flags.C_CONTIGUOUS is true for standard array', () => {
      const a = core([1, 2, 3], [3]);
      expect(a.flags.C_CONTIGUOUS).toBe(true);
    });

    it('flags.F_CONTIGUOUS for 1D', () => {
      const a = core([1, 2, 3], [3]);
      expect(a.flags.F_CONTIGUOUS).toBeDefined();
    });

    it('flags.OWNDATA is true for owned array', () => {
      const a = core([1, 2, 3], [3]);
      expect(a.flags.OWNDATA).toBe(true);
    });

    it('flags.OWNDATA is false for view', () => {
      const a = core([1, 2, 3, 4, 5], [5]);
      const v = a.slice('1:4');
      expect(v.flags.OWNDATA).toBe(false);
    });

    it('base is null for owned array', () => {
      const a = core([1, 2, 3], [3]);
      expect(a.base).toBeNull();
    });

    it('base is set for view', () => {
      const a = core([1, 2, 3, 4], [4]);
      const v = a.slice('1:3');
      expect(v.base).not.toBeNull();
    });

    it('itemsize for different dtypes', () => {
      expect(core([1], [1], 'float64').itemsize).toBe(8);
      expect(core([1], [1], 'float32').itemsize).toBe(4);
      expect(core([1], [1], 'int32').itemsize).toBe(4);
      expect(core([1], [1], 'int16').itemsize).toBe(2);
      expect(core([1], [1], 'uint8').itemsize).toBe(1);
    });

    it('nbytes', () => {
      const a = core([1, 2, 3, 4], [4], 'float32');
      expect(a.nbytes).toBe(16); // 4 * 4 bytes
    });
  });

  // ========================================
  // fill()
  // ========================================
  describe('fill()', () => {
    it('fills with number', () => {
      const a = core([0, 0, 0], [3]);
      a.fill(5);
      expect(a.toArray()).toEqual([5, 5, 5]);
    });

    it('fills bigint array with bigint', () => {
      const a = coreBigInt([0n, 0n, 0n], [3]);
      a.fill(7n);
      expect(a.toArray()).toEqual([7n, 7n, 7n]);
    });

    it('fills bigint array with number (converts to bigint)', () => {
      const a = coreBigInt([0n, 0n], [2]);
      a.fill(5 as unknown as bigint);
      expect(a.toArray()).toEqual([5n, 5n]);
    });

    it('fills bool array with truthy value', () => {
      const a = coreBool([0, 0, 0], [3]);
      a.fill(1);
      expect(a.toArray()).toEqual([1, 1, 1]);
    });

    it('fills bool array with falsy value', () => {
      const a = coreBool([1, 1, 1], [3]);
      a.fill(0);
      expect(a.toArray()).toEqual([0, 0, 0]);
    });
  });

  // ========================================
  // [Symbol.iterator]()
  // ========================================
  describe('Iterator', () => {
    it('iterates 0-d array yields single scalar', () => {
      const a = core0d(42);
      const values = [...a];
      expect(values).toEqual([42]);
    });

    it('iterates 1D array yields scalars', () => {
      const a = core([10, 20, 30], [3]);
      const values = [...a];
      expect(values).toEqual([10, 20, 30]);
    });

    it('iterates 2D array yields sub-arrays (NDArrayCore instances)', () => {
      const a = core([1, 2, 3, 4], [2, 2]);
      const rows = [...a];
      expect(rows).toHaveLength(2);
      // NDArrayCore.slice('0') keeps dim as size-1, so each "row" is shape [1, 2]
      expect((rows[0] as NDArrayCore).toArray()).toEqual([[1, 2]]);
      expect((rows[1] as NDArrayCore).toArray()).toEqual([[3, 4]]);
    });
  });

  // ========================================
  // get()
  // ========================================
  describe('get()', () => {
    it('gets element from 1D array', () => {
      const a = core([10, 20, 30], [3]);
      expect(a.get([1])).toBe(20);
    });

    it('gets element from 2D array', () => {
      const a = core([1, 2, 3, 4], [2, 2]);
      expect(a.get([1, 0])).toBe(3);
    });

    it('negative index wraps around', () => {
      const a = core([10, 20, 30], [3]);
      expect(a.get([-1])).toBe(30);
      expect(a.get([-2])).toBe(20);
    });

    it('throws for wrong number of indices', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.get([0, 0])).toThrow('dimensions');
    });

    it('throws for out of bounds', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.get([10])).toThrow('out of bounds');
    });

    it('throws for negative index still out of bounds after wrapping', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.get([-4])).toThrow('out of bounds');
    });
  });

  // ========================================
  // set() - all dtype branches
  // ========================================
  describe('set()', () => {
    it('sets number on float64 array', () => {
      const a = core([0, 0], [2]);
      a.set([0], 42);
      expect(a.get([0])).toBe(42);
    });

    it('sets on complex128 array', () => {
      const a = coreComplex([0, 0], [0, 0], [2]);
      a.set([0], new Complex(3, 4));
      const val = a.get([0]);
      expect(val).toBeInstanceOf(Complex);
      expect((val as Complex).re).toBe(3);
      expect((val as Complex).im).toBe(4);
    });

    it('sets plain object on complex128 array', () => {
      const a = coreComplex([0, 0], [0, 0], [2]);
      a.set([0], { re: 1, im: 2 });
      const val = a.get([0]);
      expect(val).toBeInstanceOf(Complex);
      expect((val as Complex).re).toBe(1);
      expect((val as Complex).im).toBe(2);
    });

    it('sets bigint on int64 array', () => {
      const a = coreBigInt([0n, 0n], [2]);
      a.set([0], 42n);
      expect(a.get([0])).toBe(42n);
    });

    it('sets number on int64 array (converts to bigint)', () => {
      const a = coreBigInt([0n, 0n], [2]);
      a.set([0], 5 as unknown as bigint);
      expect(a.get([0])).toBe(5n);
    });

    it('sets Complex value on int64 array (extracts re)', () => {
      const a = coreBigInt([0n, 0n], [2]);
      a.set([0], new Complex(7, 0));
      expect(a.get([0])).toBe(7n);
    });

    it('sets value on bool array (truthy)', () => {
      const a = coreBool([0, 0], [2]);
      a.set([0], 5);
      expect(a.get([0])).toBe(1);
    });

    it('sets value on bool array (falsy)', () => {
      const a = coreBool([1, 1], [2]);
      a.set([0], 0);
      expect(a.get([0])).toBe(0);
    });

    it('sets Complex value on bool array (truthy re)', () => {
      const a = coreBool([0, 0], [2]);
      a.set([0], new Complex(5, 3));
      expect(a.get([0])).toBe(1);
    });

    it('sets Complex value on bool array (zero re → false)', () => {
      const a = coreBool([1, 1], [2]);
      a.set([0], new Complex(0, 3));
      expect(a.get([0])).toBe(0);
    });

    it('sets Complex value on float64 array (extracts re)', () => {
      const a = core([0, 0], [2]);
      a.set([0], new Complex(3.14, 2.7));
      expect(a.get([0])).toBeCloseTo(3.14);
    });

    it('negative index wraps correctly', () => {
      const a = core([10, 20, 30], [3]);
      a.set([-1], 99);
      expect(a.get([2])).toBe(99);
    });

    it('throws for wrong number of indices', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.set([0, 0], 5)).toThrow('dimensions');
    });

    it('throws for out of bounds', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.set([10], 5)).toThrow('out of bounds');
    });

    it('throws for negative index still out of bounds after wrapping', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.set([-4], 99)).toThrow('out of bounds');
    });
  });

  // ========================================
  // iget/iset
  // ========================================
  describe('iget/iset', () => {
    it('iget returns element by flat index', () => {
      const a = core([10, 20, 30], [3]);
      expect(a.iget(0)).toBe(10);
      expect(a.iget(2)).toBe(30);
    });

    it('iset sets element by flat index', () => {
      const a = core([0, 0, 0], [3]);
      a.iset(1, 42);
      expect(a.iget(1)).toBe(42);
    });
  });

  // ========================================
  // copy()
  // ========================================
  describe('copy()', () => {
    it('returns deep copy', () => {
      const a = core([1, 2, 3], [3]);
      const b = a.copy();
      b.iset(0, 99);
      expect(a.iget(0)).toBe(1);
      expect(b.iget(0)).toBe(99);
    });
  });

  // ========================================
  // astype() - all conversion paths
  // ========================================
  describe('astype()', () => {
    it('same dtype, no copy returns self', () => {
      const a = core([1, 2, 3], [3], 'float64');
      const b = a.astype('float64', false);
      expect(b).toBe(a);
    });

    it('same dtype, copy returns new array', () => {
      const a = core([1, 2, 3], [3], 'float64');
      const b = a.astype('float64', true);
      expect(b).not.toBe(a);
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('bigint to float (BigInt→non-BigInt, non-bool)', () => {
      const a = coreBigInt([1n, 2n, 3n], [3]);
      const b = a.astype('float64');
      expect(b.dtype).toBe('float64');
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('bigint to bool (BigInt→non-BigInt, bool)', () => {
      const a = coreBigInt([0n, 1n, 5n], [3]);
      const b = a.astype('bool');
      expect(b.dtype).toBe('bool');
      expect(b.toArray()).toEqual([0, 1, 1]);
    });

    it('bigint to int32 (BigInt→non-BigInt, non-bool)', () => {
      const a = coreBigInt([10n, 20n, 30n], [3]);
      const b = a.astype('int32');
      expect(b.dtype).toBe('int32');
      expect(b.toArray()).toEqual([10, 20, 30]);
    });

    it('float to bigint (non-BigInt→BigInt)', () => {
      const a = core([1.5, 2.5, 3.5], [3], 'float64');
      const b = a.astype('int64');
      expect(b.dtype).toBe('int64');
      expect(b.toArray()).toEqual([2n, 3n, 4n]);
    });

    it('bool to bigint (non-BigInt→BigInt)', () => {
      const a = coreBool([0, 1, 1, 0], [4]);
      const b = a.astype('int64');
      expect(b.dtype).toBe('int64');
      expect(b.toArray()).toEqual([0n, 1n, 1n, 0n]);
    });

    it('float to bool (non-BigInt, non-bool → bool)', () => {
      const a = core([0, 1.5, -2.3, 0], [4], 'float64');
      const b = a.astype('bool');
      expect(b.dtype).toBe('bool');
      expect(b.toArray()).toEqual([0, 1, 1, 0]);
    });

    it('bool to float (bool → non-BigInt)', () => {
      const a = coreBool([0, 1, 1, 0], [4]);
      const b = a.astype('float64');
      expect(b.dtype).toBe('float64');
      expect(b.toArray()).toEqual([0, 1, 1, 0]);
    });

    it('int32 to float32 (non-BigInt → non-BigInt)', () => {
      const a = core([1, 2, 3], [3], 'int32');
      const b = a.astype('float32');
      expect(b.dtype).toBe('float32');
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('int32 to int16 (non-BigInt → non-BigInt)', () => {
      const a = core([1, 2, 3], [3], 'int32');
      const b = a.astype('int16');
      expect(b.dtype).toBe('int16');
    });

    it('int64 to uint64 (BigInt → BigInt)', () => {
      const a = coreBigInt([1n, 2n], [2], 'int64');
      const b = a.astype('uint64');
      expect(b.dtype).toBe('uint64');
      expect(b.toArray()).toEqual([1n, 2n]);
    });

    it('uint64 to int64 (BigInt → BigInt)', () => {
      const a = coreBigInt([1n, 2n, 3n], [3], 'uint64');
      const b = a.astype('int64');
      expect(b.dtype).toBe('int64');
      expect(b.toArray()).toEqual([1n, 2n, 3n]);
    });
  });

  // ========================================
  // slice()
  // ========================================
  describe('slice()', () => {
    it('empty slice returns self', () => {
      const a = core([1, 2, 3], [3]);
      const s = a.slice();
      expect(s).toBe(a);
    });

    it('throws for too many indices', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.slice('0', '1')).toThrow('Too many indices');
    });

    it('scalar indexing produces size-1 dimension (NDArrayCore does not check isIndex)', () => {
      const a = core([1, 2, 3, 4, 5, 6], [2, 3]);
      const row = a.slice('0');
      // NDArrayCore.slice uses step===0 check, but normalizeSlice returns step=1 for isIndex
      // So scalar indexing gives size-1 dim, not dimension removal (NDArray override handles that)
      expect(row.shape).toEqual([1, 3]);
      expect(row.toArray()).toEqual([[1, 2, 3]]);
    });

    it('range slicing preserves dimension', () => {
      const a = core([1, 2, 3, 4, 5, 6], [2, 3]);
      const sub = a.slice('0:1', '1:3');
      expect(sub.shape).toEqual([1, 2]);
    });

    it('step slicing', () => {
      const a = core([1, 2, 3, 4, 5, 6], [6]);
      const s = a.slice('::2');
      expect(s.toArray()).toEqual([1, 3, 5]);
    });

    it('base is set for sliced view', () => {
      const a = core([1, 2, 3, 4], [4]);
      const s = a.slice('1:3');
      expect(s.base).not.toBeNull();
    });

    it('pads missing dimensions with full slices', () => {
      const a = core([1, 2, 3, 4, 5, 6], [2, 3]);
      const s = a.slice('0:1');
      expect(s.shape).toEqual([1, 3]);
      expect(s.toArray()).toEqual([[1, 2, 3]]);
    });

    it('3D array with only first dim sliced (scalar index)', () => {
      const a = core([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
      const s = a.slice('1');
      // NDArrayCore keeps the dimension as size-1 (NDArray override removes it)
      expect(s.shape).toEqual([1, 2, 2]);
      expect(s.toArray()).toEqual([
        [
          [5, 6],
          [7, 8],
        ],
      ]);
    });

    it('slicing a view sets base to the original array', () => {
      const a = core([1, 2, 3, 4, 5, 6], [6]);
      const v1 = a.slice('1:5');
      const v2 = v1.slice('0:2');
      expect(v2.base).toBe(a);
      expect(v2.toArray()).toEqual([2, 3]);
    });
  });

  // ========================================
  // toString()
  // ========================================
  describe('toString()', () => {
    it('includes shape and dtype', () => {
      const a = core([1, 2, 3], [3]);
      const str = a.toString();
      expect(str).toContain('NDArray');
      expect(str).toContain('[3]');
      expect(str).toContain('float64');
    });
  });

  // ========================================
  // toArray()
  // ========================================
  describe('toArray()', () => {
    it('converts 0-d array to scalar', () => {
      const a = core0d(99);
      expect(a.toArray()).toBe(99);
    });

    it('converts 1D array', () => {
      const a = core([1, 2, 3], [3]);
      expect(a.toArray()).toEqual([1, 2, 3]);
    });

    it('converts 2D array', () => {
      const a = core([1, 2, 3, 4], [2, 2]);
      expect(a.toArray()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('converts 3D array', () => {
      const a = core([1, 2, 3, 4, 5, 6, 7, 8], [2, 2, 2]);
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
      const a = core([1, 2, 3], [3]);
      expect(a.tolist()).toEqual(a.toArray());
    });
  });

  // ========================================
  // tobytes()
  // ========================================
  describe('tobytes()', () => {
    it('returns ArrayBuffer for contiguous array', () => {
      const a = core([1, 2, 3], [3], 'int32');
      const bytes = a.tobytes();
      expect(bytes).toBeInstanceOf(ArrayBuffer);
      expect(bytes.byteLength).toBe(12);
      const view = new Int32Array(bytes);
      expect(Array.from(view)).toEqual([1, 2, 3]);
    });

    it('returns ArrayBuffer for non-contiguous (step-sliced) array', () => {
      const a = core([1, 2, 3, 4, 5, 6], [6], 'int32');
      const sliced = a.slice('::2');
      const bytes = sliced.tobytes();
      expect(bytes).toBeInstanceOf(ArrayBuffer);
      const view = new Int32Array(bytes);
      expect(Array.from(view)).toEqual([1, 3, 5]);
    });
  });

  // ========================================
  // item()
  // ========================================
  describe('item()', () => {
    it('returns scalar from size-1 array', () => {
      const a = core([42], [1]);
      expect(a.item()).toBe(42);
    });

    it('returns scalar from 0-d array', () => {
      const a = core0d(7);
      expect(a.item()).toBe(7);
    });

    it('throws for multi-element array without index', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.item()).toThrow('can only convert');
    });

    it('returns element by flat index', () => {
      const a = core([10, 20, 30], [3]);
      expect(a.item(1)).toBe(20);
    });

    it('throws for out-of-bounds flat index', () => {
      const a = core([1, 2, 3], [3]);
      expect(() => a.item(5)).toThrow('out of bounds');
    });

    it('returns element by multi-index', () => {
      const a = core([1, 2, 3, 4], [2, 2]);
      expect(a.item(1, 0)).toBe(3);
    });
  });

  // ========================================
  // fromStorage static method
  // ========================================
  describe('fromStorage()', () => {
    it('creates NDArrayCore from storage', () => {
      const storage = ArrayStorage.fromData(new Float64Array([1, 2, 3]), [3], 'float64');
      const a = NDArrayCore.fromStorage(storage);
      expect(a.toArray()).toEqual([1, 2, 3]);
    });

    it('creates NDArrayCore with base', () => {
      const storage = ArrayStorage.fromData(new Float64Array([1, 2, 3]), [3], 'float64');
      const base = NDArrayCore.fromStorage(storage);
      const view = NDArrayCore.fromStorage(storage, base);
      expect(view.base).toBe(base);
    });
  });
});

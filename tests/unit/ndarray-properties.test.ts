import { describe, it, expect } from 'vitest';
import { zeros, array, copyto } from '../../src';

describe('NDArray Properties', () => {
  describe('.T (transpose shorthand)', () => {
    it('transposes 2D array', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.shape).toEqual([2, 3]);
      const transposed = arr.T;
      expect(transposed.shape).toEqual([3, 2]);
      expect(transposed.toArray()).toEqual([
        [1, 4],
        [2, 5],
        [3, 6],
      ]);
    });

    it('transposes 1D array (no-op)', () => {
      const arr = array([1, 2, 3]);
      expect(arr.T.shape).toEqual([3]);
      expect(arr.T.toArray()).toEqual([1, 2, 3]);
    });

    it('transposes 3D array (reverses axes)', () => {
      const arr = zeros([2, 3, 4]);
      const transposed = arr.T;
      expect(transposed.shape).toEqual([4, 3, 2]);
    });

    it('returns a view (shares data)', () => {
      const arr = array([
        [1, 2],
        [3, 4],
      ]);
      const transposed = arr.T;
      expect(transposed.base).toBe(arr);
    });

    it('is same as .transpose()', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      expect(arr.T.toArray()).toEqual(arr.transpose().toArray());
    });
  });

  describe('.itemsize', () => {
    it('returns 8 for float64', () => {
      const arr = zeros([3], 'float64');
      expect(arr.itemsize).toBe(8);
    });

    it('returns 4 for float32', () => {
      const arr = zeros([3], 'float32');
      expect(arr.itemsize).toBe(4);
    });

    it('returns 8 for int64', () => {
      const arr = zeros([3], 'int64');
      expect(arr.itemsize).toBe(8);
    });

    it('returns 4 for int32', () => {
      const arr = zeros([3], 'int32');
      expect(arr.itemsize).toBe(4);
    });

    it('returns 2 for int16', () => {
      const arr = zeros([3], 'int16');
      expect(arr.itemsize).toBe(2);
    });

    it('returns 1 for int8', () => {
      const arr = zeros([3], 'int8');
      expect(arr.itemsize).toBe(1);
    });

    it('returns 1 for uint8', () => {
      const arr = zeros([3], 'uint8');
      expect(arr.itemsize).toBe(1);
    });

    it('returns 1 for bool', () => {
      const arr = zeros([3], 'bool');
      expect(arr.itemsize).toBe(1);
    });
  });

  describe('.nbytes', () => {
    it('returns size * itemsize for float64', () => {
      const arr = zeros([10], 'float64');
      expect(arr.nbytes).toBe(10 * 8);
    });

    it('returns size * itemsize for int32', () => {
      const arr = zeros([5, 4], 'int32');
      expect(arr.nbytes).toBe(20 * 4);
    });

    it('returns size * itemsize for uint8', () => {
      const arr = zeros([100], 'uint8');
      expect(arr.nbytes).toBe(100 * 1);
    });

    it('returns 0 for empty array', () => {
      const arr = zeros([0], 'float64');
      expect(arr.nbytes).toBe(0);
    });
  });

  describe('.fill()', () => {
    it('fills float64 array with number', () => {
      const arr = zeros([5]);
      arr.fill(7);
      expect(arr.toArray()).toEqual([7, 7, 7, 7, 7]);
    });

    it('fills int32 array with number', () => {
      const arr = zeros([3], 'int32');
      arr.fill(42);
      expect(arr.toArray()).toEqual([42, 42, 42]);
    });

    it('fills int64 array with bigint', () => {
      const arr = zeros([3], 'int64');
      arr.fill(BigInt(999));
      expect(arr.toArray()).toEqual([BigInt(999), BigInt(999), BigInt(999)]);
    });

    it('fills int64 array with number (converts)', () => {
      const arr = zeros([3], 'int64');
      arr.fill(42);
      expect(arr.toArray()).toEqual([BigInt(42), BigInt(42), BigInt(42)]);
    });

    it('fills bool array', () => {
      const arr = zeros([4], 'bool');
      arr.fill(1);
      expect(arr.toArray()).toEqual([1, 1, 1, 1]);
    });

    it('fills 2D array', () => {
      const arr = zeros([2, 3]);
      arr.fill(5);
      expect(arr.toArray()).toEqual([
        [5, 5, 5],
        [5, 5, 5],
      ]);
    });

    it('modifies in place', () => {
      const arr = array([1, 2, 3]);
      const original = arr;
      arr.fill(0);
      expect(arr).toBe(original);
      expect(arr.toArray()).toEqual([0, 0, 0]);
    });

    it('fills non-contiguous view', () => {
      const arr = array([
        [1, 2, 3],
        [4, 5, 6],
      ]);
      const view = arr.slice(':', '::2'); // columns 0 and 2
      view.fill(99);
      expect(arr.toArray()).toEqual([
        [99, 2, 99],
        [99, 5, 99],
      ]);
    });
  });

  describe('Iterator protocol', () => {
    it('iterates over 1D array elements', () => {
      const arr = array([1, 2, 3, 4]);
      const values: number[] = [];
      for (const val of arr) {
        values.push(val as number);
      }
      expect(values).toEqual([1, 2, 3, 4]);
    });

    it('iterates over 2D array rows', () => {
      const arr = array([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
      const rows: number[][] = [];
      for (const row of arr) {
        rows.push((row as ReturnType<typeof array>).toArray());
      }
      expect(rows).toEqual([
        [1, 2],
        [3, 4],
        [5, 6],
      ]);
    });

    it('iterates over 3D array 2D slices', () => {
      const arr = zeros([2, 3, 4]);
      let count = 0;
      for (const slice of arr) {
        expect((slice as ReturnType<typeof zeros>).shape).toEqual([3, 4]);
        count++;
      }
      expect(count).toBe(2);
    });

    it('works with spread operator', () => {
      const arr = array([10, 20, 30]);
      const values = [...arr];
      expect(values).toEqual([10, 20, 30]);
    });

    it('works with Array.from', () => {
      const arr = array([5, 10, 15]);
      const values = Array.from(arr);
      expect(values).toEqual([5, 10, 15]);
    });

    it('handles int64 array', () => {
      const arr = array([BigInt(1), BigInt(2), BigInt(3)], 'int64');
      const values: bigint[] = [];
      for (const val of arr) {
        values.push(val as bigint);
      }
      expect(values).toEqual([BigInt(1), BigInt(2), BigInt(3)]);
    });

    it('iterates over 0D array (yields single element)', () => {
      // Create a 0D array via reshape
      const arr = array([42]).reshape();
      const values: number[] = [];
      for (const val of arr) {
        values.push(val as number);
      }
      expect(values).toEqual([42]);
    });
  });
});

describe('copyto()', () => {
  it('copies scalar to array', () => {
    const dst = zeros([3]);
    copyto(dst, 5);
    expect(dst.toArray()).toEqual([5, 5, 5]);
  });

  it('copies same-shape array', () => {
    const dst = zeros([3]);
    const src = array([1, 2, 3]);
    copyto(dst, src);
    expect(dst.toArray()).toEqual([1, 2, 3]);
  });

  it('broadcasts 1D to 2D', () => {
    const dst = zeros([3, 4]);
    const src = array([1, 2, 3, 4]);
    copyto(dst, src);
    expect(dst.toArray()).toEqual([
      [1, 2, 3, 4],
      [1, 2, 3, 4],
      [1, 2, 3, 4],
    ]);
  });

  it('broadcasts column vector', () => {
    const dst = zeros([3, 4]);
    const src = array([[1], [2], [3]]);
    copyto(dst, src);
    expect(dst.toArray()).toEqual([
      [1, 1, 1, 1],
      [2, 2, 2, 2],
      [3, 3, 3, 3],
    ]);
  });

  it('throws for incompatible shapes', () => {
    const dst = zeros([3, 4]);
    const src = array([1, 2, 3]); // shape [3] can't broadcast to [3, 4]
    expect(() => copyto(dst, src)).toThrow(/could not broadcast/);
  });

  it('handles different dtypes', () => {
    const dst = zeros([3], 'int32');
    const src = array([1.5, 2.7, 3.9], 'float64');
    copyto(dst, src);
    // float64 values are copied to int32, truncated
    expect(dst.toArray()).toEqual([1, 2, 3]);
  });

  it('copies to int64 array', () => {
    const dst = zeros([3], 'int64');
    const src = array([10, 20, 30]);
    copyto(dst, src);
    expect(dst.toArray()).toEqual([BigInt(10), BigInt(20), BigInt(30)]);
  });

  it('copies to bool array', () => {
    const dst = zeros([3], 'bool');
    const src = array([0, 5, -1]);
    copyto(dst, src);
    expect(dst.toArray()).toEqual([0, 1, 1]);
  });

  it('modifies dst in place', () => {
    const dst = array([1, 2, 3]);
    const original = dst;
    copyto(dst, array([9, 9, 9]));
    expect(dst).toBe(original);
  });
});

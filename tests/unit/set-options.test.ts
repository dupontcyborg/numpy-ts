/**
 * Unit tests for the options of the set functions, isclose, allclose, put_along_axis and the
 * dtype promotion of the joining functions: edge cases and error paths.
 */

import { describe, expect, it } from 'vitest';
import {
  allclose,
  append,
  array,
  block,
  column_stack,
  concatenate,
  dstack,
  hstack,
  in1d,
  intersect1d,
  isclose,
  isin,
  put_along_axis,
  setdiff1d,
  setxor1d,
  stack,
  union1d,
  vstack,
  zeros,
} from '../../src';
import { Complex } from '../../src/common/complex';
import { castStorage } from '../../src/common/ops/shape';
import { ArrayStorage } from '../../src/common/storage';

describe('isin and in1d options', () => {
  it('inverts for bool, complex and empty inputs', () => {
    const flags = array([1, 0, 1], 'float64').astype('bool');
    expect(isin(flags, array([1], 'float64').astype('bool'), false, true).toArray()).toEqual([
      0, 1, 0,
    ]);
    const z = array(
      [new Complex(1, 1), new Complex(2, 0), new Complex(Number.NaN, 0)],
      'complex128',
    );
    expect(
      isin(z, array([new Complex(1, 1), new Complex(Number.NaN, 0)], 'complex128')).toArray(),
    ).toEqual([1, 0, 0]);
    expect(isin(z, array([new Complex(1, 1)], 'complex128'), false, true).toArray()).toEqual([
      0, 1, 1,
    ]);
    expect(isin(array([1, 2]), zeros([0]), false, true).toArray()).toEqual([1, 1]);
    expect(isin(zeros([0]), array([1]), false, true).toArray()).toEqual([]);
  });

  it('matches a real array against a complex one and 64-bit integers exactly', () => {
    const z = array([new Complex(2, 0), new Complex(3, 1)], 'complex128');
    expect(isin(array([2, 3]), z).toArray()).toEqual([1, 0]);
    const big = array([2n ** 60n, 2n ** 60n + 1n], 'int64');
    expect(isin(big, array([2n ** 60n], 'int64')).toArray()).toEqual([1, 0]);
  });

  it('flattens the first argument for in1d regardless of its shape', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(in1d(a, array([4, 1]), true).toArray()).toEqual([1, 0, 0, 1]);
  });

  it('ignores the sort method when the second array is small', () => {
    expect(isin(array([1, 1, 2]), array([1]), true).toArray()).toEqual([1, 1, 0]);
  });
});

describe('set function options', () => {
  it('keeps NaN in setdiff1d and both NaNs in setxor1d like NumPy', () => {
    const nan = Number.NaN;
    expect(setdiff1d(array([nan, 1, nan]), array([nan])).toArray()).toEqual([1, nan]);
    expect(setxor1d(array([nan, 1]), array([nan, 2])).toArray()).toEqual([1, 2, nan, nan]);
    expect(intersect1d(array([nan, 1]), array([nan, 1])).toArray()).toEqual([1]);
  });

  it('promotes the result dtype of intersect1d, setxor1d and union1d', () => {
    const i = array([1, 2, 3], 'float64').astype('int32');
    const f = array([2, 3, 4]);
    expect(intersect1d(i, f).dtype).toBe('float64');
    expect(setxor1d(i, f).dtype).toBe('float64');
    expect(union1d(i, f).dtype).toBe('float64');
    expect(setdiff1d(i, f).dtype).toBe('int32');
  });

  it('handles empty and complex inputs', () => {
    expect(setxor1d(zeros([0]), zeros([0])).toArray()).toEqual([]);
    expect(intersect1d(zeros([0]), array([1])).toArray()).toEqual([]);
    const z = array([new Complex(1, 2), new Complex(0, 1)], 'complex128');
    const w = array([new Complex(1, 2), new Complex(5, 0)], 'complex128');
    expect(intersect1d(z, w).toArray()).toEqual([new Complex(1, 2)]);
    expect(setxor1d(z, w).toArray()).toEqual([new Complex(0, 1), new Complex(5, 0)]);
    expect(setdiff1d(z, w, true).toArray()).toEqual([new Complex(0, 1)]);
    const [common, i1, i2] = intersect1d(z, w, false, true);
    expect(common.toArray()).toEqual([new Complex(1, 2)]);
    expect(i1.toArray()).toEqual([0]);
    expect(i2.toArray()).toEqual([0]);
  });

  it('handles 64-bit integers and non-contiguous input', () => {
    const a = array([5n, 2n ** 60n, 5n], 'int64');
    const b = array([2n ** 60n, 7n], 'int64');
    expect(intersect1d(a, b).toArray()).toEqual([2n ** 60n]);
    expect(setxor1d(a, b).toArray()).toEqual([5n, 7n]);
    const m = array([
      [1, 5],
      [2, 6],
    ]);
    expect(setdiff1d(m.T, array([5])).toArray()).toEqual([1, 2, 6]);
    expect(isin(m.T, array([5, 2])).toArray()).toEqual([
      [0, 1],
      [1, 0],
    ]);
  });

  it('reports positions inside ar1 for repeats when assume_unique is wrong', () => {
    const [common, i1, i2] = intersect1d(array([1, 1, 2]), array([2]), true, true);
    expect(common.toArray()).toEqual([1, 2]);
    expect(i1.toArray()).toEqual([0, 2]);
    expect(i2.toArray()).toEqual([-2, 0]);
  });
});

describe('isclose and allclose options', () => {
  it('rejects tolerances that cannot broadcast', () => {
    expect(() => isclose(array([1, 2, 3]), array([1, 2, 3]), 1e-5, array([1, 2]))).toThrow(
      /broadcast/,
    );
  });

  it('broadcasts every operand to a common shape', () => {
    const r = isclose(array([[1], [2]]), array([1, 2]), array([0, 0.5]), 0);
    expect(r.toArray()).toEqual([
      [1, 1],
      [0, 1],
    ]);
  });

  it('compares complex values with equal_nan and infinities', () => {
    const z = array(
      [new Complex(Number.NaN, 0), new Complex(Infinity, 0), new Complex(1, 1)],
      'complex128',
    );
    const w = array(
      [new Complex(Number.NaN, 0), new Complex(Infinity, 0), new Complex(1, 1.0000001)],
      'complex128',
    );
    expect(isclose(z, w).toArray()).toEqual([0, 1, 1]);
    expect(isclose(z, w, 1e-5, 1e-8, true).toArray()).toEqual([1, 1, 1]);
    expect(isclose(z, 1).toArray()).toEqual([0, 0, 0]);
  });

  it('rounds float32 and float16 arithmetic to the dtype precision', () => {
    const f32 = array([1], 'float64').astype('float32');
    expect(isclose(f32, 1.00001).toArray()).toEqual([0]);
    expect(isclose(f32, array([1.00001], 'float64').astype('float32'), 1e-5, 0).toArray()).toEqual([
      0,
    ]);
    const f16 = array([1], 'float64').astype('float16');
    expect(isclose(f16, 1.0009, 1e-3, 0).toArray()).toEqual([1]);
  });

  it('treats integers as floats and handles a scalar nan', () => {
    const ints = array([1, 2, 3], 'float64').astype('int32');
    expect(isclose(ints, 2.0000001, 0, 1e-3).toArray()).toEqual([0, 1, 0]);
    expect(isclose(array([Number.NaN]), Number.NaN, 1e-5, 1e-8, true).toArray()).toEqual([1]);
    expect(allclose(ints, ints)).toBe(true);
    expect(allclose(ints, array([1, 2, 4]))).toBe(false);
    expect(allclose(array([1, 2]), array([1.5, 2.5]), array([0.5, 0.5]), array([0, 0.5]))).toBe(
      true,
    );
    expect(allclose(array([1, 2]), array([1.5, 2.5]), array([0.1, 0.1]), 0)).toBe(false);
    expect(allclose(array([1, 2]), 1.5, 1, 0)).toBe(true);
    expect(allclose(array([1, 2]), array([1.5, 2.5]), 0, array([0.5, 0.5]))).toBe(true);
    expect(allclose(array([Number.NaN]), array([Number.NaN]), 1e-5, 1e-8, true)).toBe(true);
  });
});

describe('put_along_axis', () => {
  it('writes a bigint into int64 and a Complex into complex128', () => {
    const big = zeros([1, 2], 'int64');
    put_along_axis(big, array([[0, 1]]), 5n, 1);
    expect(big.toArray()).toEqual([[5n, 5n]]);
    const z = zeros([1, 2], 'complex128');
    put_along_axis(z, array([[1, 0]]), new Complex(1, -1), 1);
    expect(z.toArray()).toEqual([[new Complex(1, -1), new Complex(1, -1)]]);
  });

  it('converts between bigint and number values', () => {
    const big = zeros([1, 2], 'int64');
    put_along_axis(big, array([[0, 1]]), array([[3, 4]]), 1);
    expect(big.toArray()).toEqual([[3n, 4n]]);
    const f = zeros([1, 2]);
    put_along_axis(f, array([[0, 1]]), array([[3n, 4n]], 'int64'), 1);
    expect(f.toArray()).toEqual([[3, 4]]);
  });

  it('writes complex values into a complex array and bool values into bool', () => {
    const z = zeros([1, 2], 'complex64');
    put_along_axis(
      z,
      array([[0, 1]]),
      array([[new Complex(1, 2), new Complex(3, 4)]], 'complex128'),
      1,
    );
    expect(z.toArray()).toEqual([[new Complex(1, 2), new Complex(3, 4)]]);
    const bools = zeros([1, 2], 'bool');
    put_along_axis(
      bools,
      array([[0, 1]]),
      array([[new Complex(0, 1), new Complex(0, 0)]], 'complex128'),
      1,
    );
    expect(bools.toArray()).toEqual([[1, 0]]);
    const real = zeros([1, 2]);
    put_along_axis(
      real,
      array([[0, 1]]),
      array([[new Complex(1, 2), new Complex(3, 4)]], 'complex128'),
      1,
    );
    expect(real.toArray()).toEqual([[1, 3]]);
  });

  it('rejects a plain array for values', () => {
    expect(() => put_along_axis(zeros([1, 2]), array([[0]]), [1] as never, 1)).toThrow(TypeError);
  });
});

describe('joining functions promote with result_type', () => {
  const i8 = array([1, 2], 'float64').astype('int8');
  const u16 = array([3, 4], 'float64').astype('uint16');
  const f32 = array([0.5, 1.5], 'float64').astype('float32');

  it('gives float32 for int8, uint16, float32 in any order', () => {
    for (const mix of [
      [i8, u16, f32],
      [f32, u16, i8],
      [u16, f32, i8],
    ]) {
      expect(concatenate(mix).dtype).toBe('float32');
      expect(stack(mix).dtype).toBe('float32');
      expect(vstack(mix).dtype).toBe('float32');
      expect(hstack(mix).dtype).toBe('float32');
      expect(dstack(mix).dtype).toBe('float32');
      expect(column_stack(mix).dtype).toBe('float32');
      expect(block(mix).dtype).toBe('float32');
    }
    expect(append(i8, u16).dtype).toBe('int32');
  });

  it('preserves values across the cast, including bool, bigint, complex and views', () => {
    const flags = array([1, 0], 'float64').astype('bool');
    const big = array([2n ** 60n], 'int64');
    expect(concatenate([flags, big]).toArray()).toEqual([1n, 0n, 2n ** 60n]);
    const z = array([new Complex(1, 2)], 'complex64');
    const out = concatenate([f32, z, array([new Complex(7, 8)], 'complex128')]);
    expect(out.dtype).toBe('complex128');
    expect(out.toArray()).toEqual([
      new Complex(0.5, 0),
      new Complex(1.5, 0),
      new Complex(1, 2),
      new Complex(7, 8),
    ]);
    const m = array([
      [1, 2],
      [3, 4],
    ]).astype('int32');
    expect(concatenate([m.T, array([[9, 9]])], 0).toArray()).toEqual([
      [1, 3],
      [2, 4],
      [9, 9],
    ]);
    expect(concatenate([array([1n], 'int64'), array([2.5])]).toArray()).toEqual([1, 2.5]);
  });

  it('pads the leaves of a ragged block to the list depth', () => {
    expect(block([[i8, u16], [concatenate([f32, f32])]]).toArray()).toEqual([
      [1, 2, 3, 4],
      [0.5, 1.5, 0.5, 1.5],
    ]);
  });

  it('refuses to narrow complex in castStorage', () => {
    const z = ArrayStorage.zeros([2], 'complex128');
    expect(() => castStorage(z, 'float64')).toThrow(/narrow/);
    expect(castStorage(z, 'complex128')).toBe(z);
  });
});

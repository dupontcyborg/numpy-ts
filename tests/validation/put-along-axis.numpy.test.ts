/**
 * NumPy validation: put_along_axis with scalar and array values, broadcasting of indices and
 * values, negative and repeated indices, views, and every array and value dtype.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import { array, put_along_axis, zeros } from '../../src';
import { Complex } from '../../src/common/complex';
import { DTYPES, expectSame, lcg, sample, toJs, toPy } from './_dtype-grid';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

interface Layout {
  arr: number[];
  idx: number[];
  axis: number;
  values: number[];
}

// Array shape, indices shape, axis and the value shapes that broadcast against the result.
const LAYOUTS: Layout[] = [
  { arr: [3, 4], idx: [3, 2], axis: 1, values: [3, 2] },
  { arr: [3, 4], idx: [3, 2], axis: 1, values: [1, 2] },
  { arr: [3, 4], idx: [3, 2], axis: 1, values: [2] },
  { arr: [3, 4], idx: [3, 2], axis: 1, values: [3, 1] },
  { arr: [3, 4], idx: [3, 2], axis: 1, values: [] },
  { arr: [3, 4], idx: [3, 2], axis: -1, values: [1] },
  { arr: [3, 4], idx: [2, 4], axis: 0, values: [2, 4] },
  { arr: [3, 4], idx: [1, 4], axis: 0, values: [4] },
  { arr: [3, 4], idx: [3, 1], axis: 0, values: [3, 4] },
  { arr: [3, 4], idx: [1, 1], axis: 0, values: [] },
  { arr: [2, 3, 4], idx: [2, 3, 2], axis: 2, values: [2, 3, 2] },
  { arr: [2, 3, 4], idx: [2, 1, 4], axis: 1, values: [2, 1, 4] },
  { arr: [2, 3, 4], idx: [2, 3, 5], axis: 2, values: [5] },
  { arr: [5], idx: [3], axis: 0, values: [3] },
  { arr: [5], idx: [3], axis: 0, values: [] },
  { arr: [1, 3], idx: [2, 3], axis: 1, values: [2, 3] },
];

const prod = (s: number[]) => s.reduce((a, b) => a * b, 1);

function indexSample(rng: () => number, shape: number[], axisSize: number) {
  const re = Array.from(
    { length: prod(shape) },
    () => Math.floor(rng() * (2 * axisSize)) - axisSize,
  );
  return { re, shape };
}

describe('NumPy Validation: put_along_axis', { timeout: 120_000 }, () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('matches NumPy for array values of every dtype pair and layout', () => {
    const rng = lcg(5);
    for (const da of DTYPES) {
      for (const dv of DTYPES) {
        // Out-of-range float to unsigned casts are undefined in NumPy, so keep both sides in range.
        const unsignedTarget = da.startsWith('uint');
        if (unsignedTarget && !(dv.startsWith('uint') || dv === 'bool')) continue;
        const layout = LAYOUTS[Math.floor(rng() * LAYOUTS.length)]!;
        const axisSize =
          layout.arr[layout.axis < 0 ? layout.arr.length + layout.axis : layout.axis]!;
        const sa = sample(rng, da, layout.arr, { finite: true });
        const sv = sample(rng, dv, layout.values.length ? layout.values : [1], { finite: true });
        const si = indexSample(rng, layout.idx, axisSize);
        const arr = toJs(sa, da);
        const idx = toJs(si, 'int64');
        const values = layout.values.length ? toJs(sv, dv) : toJs(sv, dv).reshape([]);
        const py = `a = ${toPy(sa, da)}
idx = ${toPy(si, 'int64')}
v = ${toPy(sv, dv)}${layout.values.length ? '' : '.reshape(())'}
np.put_along_axis(a, idx, v, ${layout.axis})
result = a`;
        const label = `${da} <- ${dv} ${JSON.stringify(layout)} idx=${si.re}`;
        put_along_axis(arr, idx, values, layout.axis);
        expectSame(arr, runNumPy(py), label);
      }
    }
  });

  it('accepts a scalar for every array dtype', () => {
    const rng = lcg(6);
    for (const da of DTYPES) {
      const layout = LAYOUTS[0]!;
      const sa = sample(rng, da, layout.arr, { finite: true });
      const si = indexSample(rng, layout.idx, 4);
      const scalars: [number | bigint | Complex, string][] = [
        [7, '7'],
        [1, '1'],
        [0, '0'],
      ];
      if (da === 'int64' || da === 'uint64') scalars.push([5n, '5']);
      if (da.startsWith('complex')) scalars.push([new Complex(2, -3), 'complex(2, -3)']);
      if (da.startsWith('float') || da.startsWith('complex')) scalars.push([0.5, '0.5']);
      for (const [value, pyValue] of scalars) {
        if (da === 'bool' && value === 7) continue;
        const arr = toJs(sa, da);
        put_along_axis(arr, toJs(si, 'int64'), value, layout.axis);
        expectSame(
          arr,
          runNumPy(`a = ${toPy(sa, da)}
idx = ${toPy(si, 'int64')}
np.put_along_axis(a, idx, ${pyValue}, ${layout.axis})
result = a`),
          `${da} <- ${String(value)}`,
        );
      }
    }
  });

  it('keeps the last value for a repeated index and honours negative indices', () => {
    const arr = zeros([2, 3]);
    put_along_axis(
      arr,
      array([
        [1, -2, 1],
        [0, 0, -1],
      ]),
      array([
        [1, 2, 3],
        [4, 5, 6],
      ]),
      1,
    );
    expectSame(
      arr,
      runNumPy(`a = np.zeros((2, 3))
np.put_along_axis(a, np.array([[1, -2, 1], [0, 0, -1]]), np.array([[1, 2, 3], [4, 5, 6]]), 1)
result = a`),
      'repeated and negative',
    );
  });

  it('writes through a transposed view into the original', () => {
    const base = zeros([3, 2]);
    put_along_axis(
      base.T,
      array([
        [0, 2],
        [1, 1],
      ]),
      9,
      1,
    );
    expectSame(
      base,
      runNumPy(`a = np.zeros((3, 2))
np.put_along_axis(a.T, np.array([[0, 2], [1, 1]]), 9, 1)
result = a`),
      'transposed',
    );
  });

  it('rejects mismatched ndim, bad axis, out of range index and unbroadcastable values', () => {
    const arr = zeros([2, 3]);
    const idx = array([[0], [1]]);
    expect(() => put_along_axis(arr, array([0, 1]), 1, 1)).toThrow(/same number of dimensions/);
    expect(() => put_along_axis(arr, idx, 1, 2)).toThrow(/out of bounds/);
    expect(() => put_along_axis(arr, array([[0], [3]]), 1, 1)).toThrow(
      /out of bounds for axis 1 with size 3/,
    );
    expect(() => put_along_axis(arr, array([[0], [-4]]), 1, 1)).toThrow(
      /index -4 is out of bounds/,
    );
    expect(() => put_along_axis(arr, idx, array([1, 2, 3]), 1)).toThrow(/could not be broadcast/);
    expect(() => put_along_axis(arr, idx, zeros([1, 1, 2]), 1)).toThrow(/could not be broadcast/);
    expect(() =>
      put_along_axis(
        arr,
        array([
          [0, 1, 2],
          [0, 1, 2],
          [0, 1, 2],
        ]),
        1,
        1,
      ),
    ).toThrow(/shape mismatch/);
  });

  it('is a no-op for empty indices', () => {
    const arr = zeros([2, 3]);
    put_along_axis(arr, zeros([2, 0]), 5, 1);
    expect(arr.toArray()).toEqual([
      [0, 0, 0],
      [0, 0, 0],
    ]);
  });

  it('truncates floats and drops the imaginary part like an assignment', () => {
    const ints = zeros([1, 3], 'int32');
    put_along_axis(ints, array([[0, 1, 2]]), array([[2.9, -2.9, 1.5]]), 1);
    expect(ints.toArray()).toEqual([[2, -2, 1]]);
    const reals = zeros([1, 2]);
    put_along_axis(reals, array([[0, 1]]), new Complex(4, 5), 1);
    expect(reals.toArray()).toEqual([[4, 4]]);
    const bools = zeros([1, 2], 'bool');
    put_along_axis(bools, array([[0, 1]]), Number.NaN, 1);
    expect(bools.toArray()).toEqual([[1, 1]]);
  });
});

/**
 * NumPy validation tests for unwrap: period and discont handling, N-D arrays along any axis,
 * dtype rules, and non-finite input.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

// Deterministic pseudo-random steps wide enough that several wraps happen in one lane.
function walk(n: number, seed: number, scale: number, round = false): number[] {
  let state = seed;
  let acc = 0;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    state = (state * 1103515245 + 12345) % 2147483648;
    const step = (state / 2147483648 - 0.5) * scale;
    acc += round ? Math.round(step) : step;
    out.push(acc);
  }
  return out;
}

function pyLit(v: number): string {
  if (Number.isNaN(v)) return 'np.nan';
  if (v === Infinity) return 'np.inf';
  if (v === -Infinity) return '-np.inf';
  return String(v);
}

function pyNum(v: number, forceInt: boolean): string {
  return forceInt ? String(v) : Number.isInteger(v) ? `${v}.0` : String(v);
}

interface Opts {
  discont?: number;
  axis?: number;
  period?: number;
  // Emit the period as a Python int, which keeps integer inputs integer.
  intPeriod?: boolean;
}

function check(vals: number[], shape: number[], dtype: string, opts: Opts = {}): void {
  const a = np.array(vals, dtype as never).reshape(shape);
  const args = [
    opts.discont === undefined ? 'None' : pyNum(opts.discont, false),
    String(opts.axis ?? -1),
  ];
  const pyPeriod =
    opts.period === undefined ? '2 * np.pi' : pyNum(opts.period, opts.intPeriod ?? false);
  const expected = runNumPy(`
a = np.array([${vals.map(pyLit).join(', ')}], dtype='${dtype}').reshape(${JSON.stringify(shape)})
result = np.unwrap(a, ${args[0]}, ${args[1]}, period=${pyPeriod})
`);
  const r = np.unwrap(a, opts.discont, opts.axis ?? -1, opts.period ?? 2 * Math.PI);
  expect(Array.from(r.shape)).toEqual(expected.shape);
  expect(r.dtype).toBe(expected.dtype);
  const rtol = dtype === 'float32' ? 1e-5 : dtype === 'float16' ? 5e-3 : 1e-9;
  expect(arraysClose(r.toArray(), expected.value, rtol, rtol)).toBe(true);
}

describe('NumPy Validation: unwrap', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  it('corrects the documented jump', () => {
    check([0, 1, 2, 3, -3, -2, -1, 0], [8], 'float64');
  });

  describe('float64 1-D', () => {
    for (const seed of [1, 2, 3, 4, 5]) {
      it(`default arguments, seed ${seed}`, () => {
        check(walk(40, seed, 14), [40], 'float64');
      });
    }
    for (const period of [1, 2, 4, 6.5, 360, 0.25]) {
      it(`period ${period}`, () => {
        check(walk(40, 7, period * 3), [40], 'float64', { period });
      });
    }
    for (const discont of [0, 0.5, 1, 3, Math.PI, 5, 100, -1]) {
      it(`discont ${discont}`, () => {
        check(walk(40, 11, 14), [40], 'float64', { discont });
      });
    }
    it('discont with a custom period', () => {
      check(walk(40, 12, 40), [40], 'float64', { period: 10, discont: 2 });
    });
    it('default discont follows a custom period', () => {
      check(walk(40, 13, 40), [40], 'float64', { period: 10 });
    });
    it('negative period', () => {
      check(walk(20, 14, 20), [20], 'float64', { period: -6 });
    });
    it('negative period with jumps that are exact multiples', () => {
      check([0, 3, 6, 9, 12], [5], 'float64', { period: -6 });
    });
    it('zero period gives NaN', () => {
      check([0, 1, 7, 2], [4], 'float64', { period: 0 });
    });
    it('jumps that land exactly on half a period', () => {
      check([0, Math.PI, 0, -Math.PI, 0], [5], 'float64');
      check([0, 3, 0, -3, 0, 3, 6, 3], [8], 'float64', { period: 6 });
    });
    it('empty, single element and two element inputs', () => {
      check([], [0], 'float64');
      check([4], [1], 'float64');
      check([0, 6], [2], 'float64');
    });
    it('NaN and infinity propagate', () => {
      check([0, 1, Number.NaN, 3, 9], [5], 'float64');
      check([0, 1, Infinity, 3, 9], [5], 'float64');
      check([0, -Infinity, 1], [3], 'float64');
    });
  });

  describe('N-D along each axis', () => {
    const shapes: number[][] = [
      [3, 8],
      [8, 3],
      [2, 3, 7],
      [4, 1, 6],
      [1, 5],
      [2, 2, 2, 5],
    ];
    for (const shape of shapes) {
      const n = shape.reduce((x, y) => x * y, 1);
      for (let axis = -shape.length; axis < shape.length; axis++) {
        it(`shape ${JSON.stringify(shape)} axis ${axis}`, () => {
          check(walk(n, 21 + axis, 12), shape, 'float64', { axis });
        });
      }
    }
    it('N-D with period and discont', () => {
      check(walk(60, 31, 50), [3, 4, 5], 'float64', { axis: 1, period: 8, discont: 3 });
    });
    it('zero-length axis in an N-D array', () => {
      check([], [2, 0], 'float64');
      check([], [0, 3], 'float64', { axis: 0 });
    });
  });

  describe('dtypes', () => {
    it('float32 keeps its dtype and rounds like NumPy', () => {
      check(walk(40, 41, 14), [40], 'float32');
      check(walk(48, 42, 14), [4, 12], 'float32', { axis: 0 });
      check(walk(40, 43, 40), [40], 'float32', { period: 10, discont: 2 });
    });
    it('float16 keeps its dtype', () => {
      check(walk(30, 44, 14), [30], 'float16');
    });
    for (const dtype of ['int8', 'int16', 'int32', 'uint8', 'uint16', 'uint32', 'bool']) {
      it(`${dtype} with the default period gives float64`, () => {
        const vals =
          dtype === 'bool'
            ? [0, 1, 1, 0, 1, 0, 0, 1]
            : walk(20, 45, 20, true).map((v) => Math.abs(Math.round(v)) % 100);
        check(vals, [vals.length], dtype);
      });
    }
    it('integer input with a float period gives float64', () => {
      const vals = walk(30, 46, 30, true).map(Math.round);
      check(vals, [30], 'int32', { period: 6.5 });
    });
    for (const dtype of ['int8', 'int16', 'int32', 'int64']) {
      for (const period of dtype === 'int8' ? [6, 5, 20, 1] : [6, 5, 64, 1]) {
        it(`${dtype} with integer period ${period} stays integer`, () => {
          const vals = walk(30, 47 + period, period * 3, true).map(Math.round);
          check(vals, [30], dtype, { period, intPeriod: true });
        });
      }
    }
    it('integer period over N-D input along axis 0', () => {
      const vals = walk(24, 48, 30, true).map(Math.round);
      check(vals, [6, 4], 'int32', { period: 8, intPeriod: true, axis: 0 });
    });
    it('integer period with an explicit discont', () => {
      const vals = walk(30, 49, 30, true).map(Math.round);
      check(vals, [30], 'int32', { period: 8, intPeriod: true, discont: 3 });
    });
    it('unsigned input with an integer period is rejected like NumPy', () => {
      const a = np.array([1, 2, 3], 'uint8');
      expect(() => np.unwrap(a, undefined, -1, 6)).toThrow(/out of bounds for uint8/);
      expect(() =>
        runNumPy("result = np.unwrap(np.array([1,2,3], dtype='uint8'), period=6)"),
      ).toThrow();
    });
  });

  describe('errors', () => {
    it('rejects an out-of-range axis', () => {
      const a = np.array([
        [1, 2],
        [3, 4],
      ]);
      expect(() => np.unwrap(a, undefined, 2)).toThrow(/out of bounds/);
      expect(() => np.unwrap(a, undefined, -3)).toThrow(/out of bounds/);
    });
    it('rejects 0-d input', () => {
      expect(() => np.unwrap(np.array(3))).toThrow(/at least one dimensional/);
    });
    it('rejects complex input', () => {
      expect(() => np.unwrap(np.array([1, 2]).astype('complex128'))).toThrow(/complex/);
    });
  });

  it('works on a non-contiguous view', () => {
    const base = np.array(walk(24, 51, 14)).reshape([4, 6]);
    const view = base.T;
    const r = np.unwrap(view, undefined, 0);
    const expected = runNumPy(`
b = np.array([${walk(24, 51, 14).map(pyLit).join(', ')}]).reshape(4, 6)
result = np.unwrap(b.T, axis=0)
`);
    expect(arraysClose(r.toArray(), expected.value, 1e-9, 1e-9)).toBe(true);
  });
});

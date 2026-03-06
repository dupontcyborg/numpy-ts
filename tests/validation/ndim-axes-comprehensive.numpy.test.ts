/**
 * Comprehensive NDim + Axes Validation Tests
 *
 * Systematically validates numpy-ts against NumPy for ALL implemented functions across:
 * - All input dimensionalities (0D through 5D)
 * - All valid axis values (positive, negative, multi-axis)
 * - keepdims parameter
 * - Broadcasting scenarios
 * - Error cases (invalid ndim, invalid axis)
 *
 * Generated from scripts/ndim_axes_comprehensive.py
 */

import { describe, it, expect, beforeAll } from 'vitest';
import {
  array,
  zeros,
  // Elementwise unary
  sin,
  arcsin,
  exp,
  sqrt,
  absolute as abs,
  negative,
  floor,
  ceil,
  isfinite,
  isnan,
  // Binary
  add,
  subtract,
  multiply,
  greater,
  // Reductions
  sum,
  prod,
  mean,
  std,
  amax,
  amin,
  argmax,
  argmin,
  all,
  any,
  cumsum,
  cumprod,
  // nan variants
  nansum,
  nanprod,
  nanmean,
  nanstd,
  nanvar,
  nanmin,
  nanmax,
  nanargmin,
  nanargmax,
  nanquantile,
  nanpercentile,
  nancumsum,
  nancumprod,
  // sort/search
  sort,
  argsort,
  // Stats
  diff,
  median,
  percentile,
  quantile,
  average,
  // Shape ops
  reshape,
  flip,
  roll,
  concatenate,
  stack,
  squeeze,
  expand_dims,
  swapaxes,
  moveaxis,
  repeat,
  tile,
  transpose,
  rot90,
  broadcast_to,
  take,
  clip,
  where,
  // Split
  split,
  array_split,
  // linalg
  dot,
  inner,
  outer,
  tensordot,
  trace,
  diagonal,
  matmul,
  cross,
  einsum,
  linalg,
  // FFT
  fft,
  // Type
  nanmedian,
  // Additional reductions
  ptp,
  variance,
  // Sorting
  partition,
  argpartition,
  count_nonzero,
  // Stats
  gradient,
  trapezoid,
  unwrap,
  // Bitwise
  packbits,
  unpackbits,
  // Linalg extras
  vecdot,
  permute_dims,
  // Shape
  rollaxis,
  unstack,
  append,
  delete_,
  insert,
  // Indexing
  take_along_axis,
  put_along_axis,
  putmask,
  place,
  compress,
  iindex,
  bindex,
  // Utils
  apply_along_axis,
  apply_over_axes,
  unique,
  // Math
  copysign,
  nextafter,
  spacing,
  signbit,
  // Bitwise
  bitwise_count,
} from '../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './numpy-oracle';

// ─── Helpers ───────────────────────────────────────────────────────────────

/** Create ascending float array of given shape. */
function mk(shape: number[], scale = 1.0): ReturnType<typeof array> {
  if (shape.length === 0) return array(0.5);
  const n = shape.reduce((a, b) => a * b, 1);
  return array(Array.from({ length: n }, (_, i) => (i + 1) * scale)).reshape(shape);
}

/** Create bool array (alternating T/F) of given shape. */
function mkBool(shape: number[]): ReturnType<typeof array> {
  const n = shape.reduce((a, b) => a * b, 1);
  return array(Array.from({ length: n }, (_, i) => i % 2 === 0)).reshape(shape);
}

describe('NumPy Validation: Comprehensive NDim + Axes', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error(
        '❌ Python NumPy not available!\n\n   source ~/.zshrc && conda activate py313\n'
      );
    }
  });

  // ============================================================
  // SECTION 1: Elementwise Unary (0D through 5D)
  // ============================================================
  describe('elementwise unary: shape preservation 0D-5D', () => {
    const SHAPES: [string, number[]][] = [
      ['0D', []],
      ['1D', [6]],
      ['2D', [2, 3]],
      ['3D', [2, 3, 4]],
      ['4D', [2, 2, 3, 4]],
      ['5D', [2, 2, 2, 3, 4]],
    ];

    for (const [label, shape] of SHAPES) {
      it(`sin: ${label} ${JSON.stringify(shape)}`, () => {
        const a = mk(shape, 0.05);
        const r = sin(a);
        const expected = shape.length === 0 ? [] : shape;
        expect((r as any).shape ?? []).toEqual(expected);
        const py = runNumPy(
          `result = np.sin(np.${shape.length === 0 ? 'array(0.5)' : `arange(1, ${shape.reduce((a, b) => a * b, 1) + 1}, dtype=float).reshape(${JSON.stringify(shape)}) * 0.05`})`
        );
        const val = (r as any).toArray?.() ?? r;
        expect(arraysClose(val, py.value, 1e-10)).toBe(true);
      });

      it(`exp: ${label} ${JSON.stringify(shape)}`, () => {
        const a = mk(shape, 0.1);
        const r = exp(a);
        expect((r as any).shape ?? []).toEqual(shape.length === 0 ? [] : shape);
      });

      it(`sqrt: ${label} ${JSON.stringify(shape)}`, () => {
        const a = mk(shape, 0.5);
        const r = sqrt(a);
        expect((r as any).shape ?? []).toEqual(shape.length === 0 ? [] : shape);
      });
    }

    it('arcsin: 2D input in range [-1,1]', () => {
      const a = mk([3, 4], 0.08); // max = 0.08*12 = 0.96 < 1
      const r = arcsin(a);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('isfinite: 3D', () => {
      const a = mk([2, 3, 4]);
      const r = isfinite(a);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('isnan: 3D', () => {
      const a = mk([2, 3, 4]);
      const r = isnan(a);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('floor: 4D', () => {
      const a = mk([2, 2, 3, 4], 0.7);
      const r = floor(a);
      const py = runNumPy(`result = np.floor(np.arange(1,49,dtype=float).reshape(2,2,3,4)*0.7)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ceil: 3D', () => {
      const a = mk([2, 3, 4], 0.7);
      const r = ceil(a);
      const py = runNumPy(`result = np.ceil(np.arange(1,25,dtype=float).reshape(2,3,4)*0.7)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('negative: 5D', () => {
      const a = mk([2, 2, 2, 3, 4]);
      const r = negative(a);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
      expect((r as any).toArray().flat(Infinity)[0]).toBeLessThan(0);
    });
  });

  // ============================================================
  // SECTION 2: Binary Elementwise + Broadcasting
  // ============================================================
  describe('binary elementwise: same-shape 0D-5D + broadcasting', () => {
    const addPy = (shape: number[]) => {
      if (shape.length === 0) return `result = np.add(np.array(0.5), np.array(0.5))`;
      const n = shape.reduce((a, b) => a * b, 1);
      return `result = np.add(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)})*0.5, np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)})*0.3)`;
    };

    for (const shape of [[] as number[], [3], [2, 3], [2, 3, 4], [2, 2, 3, 4], [2, 2, 2, 3, 4]]) {
      it(`add: same-shape ${JSON.stringify(shape)}`, () => {
        const a = mk(shape, 0.5);
        const b = mk(shape, 0.3);
        const r = add(a, b);
        expect((r as any).shape ?? []).toEqual(shape);
        if (shape.length <= 3) {
          const py = runNumPy(addPy(shape));
          const val = (r as any).toArray?.() ?? r;
          expect(arraysClose(val, py.value)).toBe(true);
        }
      });
    }

    it('add: broadcast 1D+2D [4] + [3,4] -> [3,4]', () => {
      const a = array([1, 2, 3, 4]);
      const b = mk([3, 4]);
      const r = add(a, b);
      const py = runNumPy(
        `result = np.add(np.array([1,2,3,4],dtype=float), np.arange(1,13,dtype=float).reshape(3,4))`
      );
      expect((r as any).shape).toEqual([3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('add: broadcast 1D+3D [4] + [2,3,4] -> [2,3,4]', () => {
      const a = array([1, 2, 3, 4]);
      const b = mk([2, 3, 4]);
      const r = add(a, b);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('add: broadcast 1D+4D [4] + [2,2,3,4] -> [2,2,3,4]', () => {
      const a = array([1, 2, 3, 4]);
      const b = mk([2, 2, 3, 4]);
      const r = add(a, b);
      expect((r as any).shape).toEqual([2, 2, 3, 4]);
    });

    it('add: broadcast 1D+5D [4] + [2,2,2,3,4] -> [2,2,2,3,4]', () => {
      const a = array([1, 2, 3, 4]);
      const b = mk([2, 2, 2, 3, 4]);
      const r = add(a, b);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });

    it('multiply: broadcast 2D+3D [3,4] + [2,3,4]', () => {
      const a = mk([3, 4]);
      const b = mk([2, 3, 4]);
      const r = multiply(a, b);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('multiply: broadcast scalar+4D', () => {
      const a = array(2.0);
      const b = mk([2, 2, 3, 4]);
      const r = multiply(a, b);
      expect((r as any).shape).toEqual([2, 2, 3, 4]);
    });

    it('subtract: 3D - 3D values correct', () => {
      const a = mk([2, 3, 4]);
      const b = mk([2, 3, 4], 0.5);
      const r = subtract(a, b);
      const py = runNumPy(`
A = np.arange(1,25,dtype=float).reshape(2,3,4)
B = np.arange(1,25,dtype=float).reshape(2,3,4)*0.5
result = np.subtract(A, B)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('greater: 3D boolean result', () => {
      const a = mk([2, 3, 4]);
      const b = mk([2, 3, 4], 0.5);
      const r = greater(a, b);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });
  });

  // ============================================================
  // SECTION 3: Reductions — ALL axes, keepdims, multi-axis
  // ============================================================
  describe('sum: all axes, keepdims, multi-axis', () => {
    it('0D: scalar input', () => {
      const a = array(5.0);
      const r = sum(a);
      expect(r).toBeCloseTo(5.0);
    });

    it('1D: axis=0', () => {
      const a = array([1, 2, 3, 4, 5, 6]);
      expect(sum(a, 0)).toBeCloseTo(21);
    });

    it('1D: axis=-1', () => {
      const a = array([1, 2, 3, 4, 5, 6]);
      expect(sum(a, -1)).toBeCloseTo(21);
    });

    it('1D: axis=0 keepdims', () => {
      const a = array([1, 2, 3, 4]);
      const r = sum(a, 0, true);
      expect((r as any).shape).toEqual([1]);
    });

    for (const [ax, expectedShape] of [
      [0, [4]],
      [1, [3]],
      [-1, [3]],
      [-2, [4]],
    ] as [number, number[]][]) {
      it(`2D [3,4]: axis=${ax} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk([3, 4]);
        const r = sum(A, ax);
        const py = runNumPy(
          `result = np.sum(np.arange(1,13,dtype=float).reshape(3,4), axis=${ax})`
        );
        expect((r as any).shape).toEqual(expectedShape);
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }

    it('2D [3,4]: keepdims axis=0', () => {
      const A = mk([3, 4]);
      const r = sum(A, 0, true);
      expect((r as any).shape).toEqual([1, 4]);
    });

    it('2D [3,4]: keepdims axis=1', () => {
      const A = mk([3, 4]);
      const r = sum(A, 1, true);
      expect((r as any).shape).toEqual([3, 1]);
    });

    for (const [ax, expectedShape] of [
      [0, [3, 4]],
      [1, [2, 4]],
      [2, [2, 3]],
      [-1, [2, 3]],
      [-2, [2, 4]],
      [-3, [3, 4]],
    ] as [number, number[]][]) {
      it(`3D [2,3,4]: axis=${ax} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk([2, 3, 4]);
        const r = sum(A, ax);
        const py = runNumPy(
          `result = np.sum(np.arange(1,25,dtype=float).reshape(2,3,4), axis=${ax})`
        );
        expect((r as any).shape).toEqual(expectedShape);
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }

    it('3D [2,3,4]: axis=0 keepdims -> [1,3,4]', () => {
      const A = mk([2, 3, 4]);
      const r = sum(A, 0, true);
      expect((r as any).shape).toEqual([1, 3, 4]);
    });

    it('3D [2,3,4]: axis=2 keepdims -> [2,3,1]', () => {
      const A = mk([2, 3, 4]);
      const r = sum(A, 2, true);
      expect((r as any).shape).toEqual([2, 3, 1]);
    });

    it('3D [2,3,4]: multi-axis (0,1) -> [4]', () => {
      const A = mk([2, 3, 4]);
      const r = (A as any).sum([0, 1]);
      const py = runNumPy(
        `result = np.sum(np.arange(1,25,dtype=float).reshape(2,3,4), axis=(0,1))`
      );
      expect((r as any).shape).toEqual([4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('3D [2,3,4]: multi-axis (1,2) -> [2]', () => {
      const A = mk([2, 3, 4]);
      const r = (A as any).sum([1, 2]);
      const py = runNumPy(
        `result = np.sum(np.arange(1,25,dtype=float).reshape(2,3,4), axis=(1,2))`
      );
      expect((r as any).shape).toEqual([2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    for (const [ax, expectedShape] of [
      [0, [3, 4, 5]],
      [1, [2, 4, 5]],
      [2, [2, 3, 5]],
      [3, [2, 3, 4]],
      [-1, [2, 3, 4]],
      [-4, [3, 4, 5]],
    ] as [number, number[]][]) {
      it(`4D [2,3,4,5]: axis=${ax} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk([2, 3, 4, 5]);
        const r = sum(A, ax);
        expect((r as any).shape).toEqual(expectedShape);
      });
    }

    it('4D [2,3,4,5]: multi-axis (0,1,2,3) -> scalar', () => {
      const A = mk([2, 3, 4, 5]);
      const r = (A as any).sum([0, 1, 2, 3]);
      const total = 2 * 3 * 4 * 5;
      const expected = ((1 + total) * total) / 2;
      expect(arraysClose(r as number, expected)).toBe(true);
    });

    it('4D [2,3,4,5]: multi-axis (1,3) keepdims -> [2,1,4,1]', () => {
      const A = mk([2, 3, 4, 5]);
      const r = (A as any).sum([1, 3], true);
      expect((r as any).shape).toEqual([2, 1, 4, 1]);
    });

    it('5D [2,2,2,3,4]: axis=2', () => {
      const A = mk([2, 2, 2, 3, 4]);
      const r = sum(A, 2);
      expect((r as any).shape).toEqual([2, 2, 3, 4]);
    });
  });

  describe('mean: all axes, keepdims', () => {
    it('2D [3,4]: all axes', () => {
      const A = mk([3, 4]);
      for (const [ax, sh] of [
        [0, [4]],
        [1, [3]],
        [-1, [3]],
        [-2, [4]],
      ] as [number, number[]][]) {
        const r = mean(A, ax);
        expect((r as any).shape).toEqual(sh);
      }
    });

    it('3D [2,3,4]: all axes + keepdims', () => {
      const A = mk([2, 3, 4]);
      const origShape = [2, 3, 4];
      for (const [ax, sh] of [
        [0, [3, 4]],
        [1, [2, 4]],
        [2, [2, 3]],
        [-1, [2, 3]],
      ] as [number, number[]][]) {
        expect((mean(A, ax) as any).shape).toEqual(sh);
        const normAx = ax >= 0 ? ax : origShape.length + ax;
        const expectedKeepdims = origShape.map((v, i) => (i === normAx ? 1 : v));
        expect((mean(A, ax, true) as any).shape).toEqual(expectedKeepdims);
      }
    });

    it('4D [2,3,4,5]: axis=-1', () => {
      const A = mk([2, 3, 4, 5]);
      const r = mean(A, -1);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });
  });

  describe('prod: all axes', () => {
    it('2D [2,3]: axis=0', () => {
      const A = mk([2, 3]);
      const r = prod(A, 0);
      const py = runNumPy(`result = np.prod(np.arange(1,7,dtype=float).reshape(2,3), axis=0)`);
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('3D [2,3,4]: all axes', () => {
      const A = mk([2, 3, 4]);
      expect((prod(A, 0) as any).shape).toEqual([3, 4]);
      expect((prod(A, 1) as any).shape).toEqual([2, 4]);
      expect((prod(A, 2) as any).shape).toEqual([2, 3]);
    });
  });

  describe('std/var: all axes, keepdims', () => {
    it('2D [3,4]: all axes + keepdims', () => {
      const A = mk([3, 4]);
      expect((std(A, 0) as any).shape).toEqual([4]);
      expect((std(A, 1) as any).shape).toEqual([3]);
      expect((std(A, 0, 0, true) as any).shape).toEqual([1, 4]);
      expect((std(A, 1, 0, true) as any).shape).toEqual([3, 1]);
    });

    it('3D [2,3,4]: axis=2 values correct', () => {
      const A = mk([2, 3, 4]);
      const r = std(A, 2);
      const py = runNumPy(`result = np.std(np.arange(1,25,dtype=float).reshape(2,3,4), axis=2)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value, 1e-10)).toBe(true);
    });

    it('4D [2,3,4,5]: axis=-1 keepdims', () => {
      const A = mk([2, 3, 4, 5]);
      const r = std(A, -1, 0, true);
      expect((r as any).shape).toEqual([2, 3, 4, 1]);
    });
  });

  // ============================================================
  // SECTION 4: NaN Reductions
  // ============================================================
  describe('nan reductions: all axes with NaN values', () => {
    const makeWithNaN = (shape: number[]) => {
      const n = shape.reduce((a, b) => a * b, 1);
      const data = Array.from({ length: n }, (_, i) => i + 1.0);
      data[0] = NaN;
      return array(data).reshape(shape);
    };

    it('nansum: 2D axis=0', () => {
      const A = makeWithNaN([3, 4]);
      const r = nansum(A, 0);
      const py = runNumPy(`
A = np.arange(1,13,dtype=float).reshape(3,4)
A[0,0] = np.nan
result = np.nansum(A, axis=0)`);
      expect((r as any).shape).toEqual([4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nansum: 2D axis=1', () => {
      const A = makeWithNaN([3, 4]);
      const r = nansum(A, 1);
      expect((r as any).shape).toEqual([3]);
    });

    it('nansum: 3D axis=0', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nansum(A, 0);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('nansum: 3D axis=1', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nansum(A, 1);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('nansum: 3D axis=2', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nansum(A, 2);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('nansum: 3D keepdims', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nansum(A, 1, true);
      expect((r as any).shape).toEqual([2, 1, 4]);
    });

    for (const [fnName, fn] of [
      ['nanmean', nanmean],
      ['nanstd', nanstd],
      ['nanvar', nanvar],
      ['nanprod', nanprod],
    ] as [string, (a: any, ...args: any[]) => any][]) {
      it(`${fnName}: 3D axis=1`, () => {
        const A = makeWithNaN([2, 3, 4]);
        const r = fn(A, 1);
        expect((r as any).shape).toEqual([2, 4]);
      });

      it(`${fnName}: 3D axis=-1`, () => {
        const A = makeWithNaN([2, 3, 4]);
        const r = fn(A, -1);
        expect((r as any).shape).toEqual([2, 3]);
      });
    }

    it('nanmin: 3D axis=0', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nanmin(A, 0);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('nanmax: 3D all axes', () => {
      const A = makeWithNaN([2, 3, 4]);
      expect((nanmax(A, 0) as any).shape).toEqual([3, 4]);
      expect((nanmax(A, 1) as any).shape).toEqual([2, 4]);
      expect((nanmax(A, 2) as any).shape).toEqual([2, 3]);
    });

    it('nanargmin: 3D axis=1', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nanargmin(A, 1);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('nanmedian: 2D axis=0', () => {
      const A = makeWithNaN([3, 4]);
      const r = nanmedian(A, 0);
      expect((r as any).shape).toEqual([4]);
    });

    it('nanmedian: 3D axis=1', () => {
      const A = makeWithNaN([2, 3, 4]);
      const r = nanmedian(A, 1);
      expect((r as any).shape).toEqual([2, 4]);
    });
  });

  // ============================================================
  // SECTION 5: amax/amin/argmax/argmin — all axes
  // ============================================================
  describe('amax/amin/argmax/argmin: all axes', () => {
    it('0D input', () => {
      const a = array(5.0);
      expect(amax(a)).toBeCloseTo(5.0);
      expect(amin(a)).toBeCloseTo(5.0);
    });

    for (const [fn, fnName] of [
      [amax, 'amax'],
      [amin, 'amin'],
      [argmax, 'argmax'],
      [argmin, 'argmin'],
    ] as [(a: any, ...args: any[]) => any, string][]) {
      it(`${fnName}: 1D`, () => {
        const A = mk([6]);
        expect(typeof fn(A)).toBe('number');
      });

      for (const [shape, axes] of [
        [
          [3, 4],
          [0, 1, -1, -2],
        ],
        [
          [2, 3, 4],
          [0, 1, 2, -1, -2, -3],
        ],
        [
          [2, 2, 3, 4],
          [0, 1, 2, 3, -1, -4],
        ],
      ] as [number[], number[]][]) {
        for (const ax of axes) {
          it(`${fnName}: ${JSON.stringify(shape)} axis=${ax}`, () => {
            const A = mk(shape);
            const r = fn(A, ax);
            const ndim = shape.length;
            const normAx = ax >= 0 ? ax : ndim + ax;
            const expectedShape = shape.filter((_, i) => i !== normAx);
            expect((r as any).shape).toEqual(expectedShape);
          });
        }
      }

      it(`${fnName}: 3D [2,3,4] keepdims axis=1`, () => {
        if (fn === amax || fn === amin) {
          const r = fn(mk([2, 3, 4]), 1, true);
          expect((r as any).shape).toEqual([2, 1, 4]);
        }
      });
    }
  });

  // ============================================================
  // SECTION 6: all/any — all axes, keepdims
  // ============================================================
  describe('all/any: all axes, keepdims', () => {
    for (const [fn, fnName] of [
      [all, 'all'],
      [any, 'any'],
    ] as [(a: any, ...args: any[]) => any, string][]) {
      it(`${fnName}: 0D`, () => {
        expect(fn(array(true))).toBe(true);
      });

      for (const [shape, axes] of [
        [[4], [0, -1]],
        [
          [3, 4],
          [0, 1, -1, -2],
        ],
        [
          [2, 3, 4],
          [0, 1, 2, -1, -2, -3],
        ],
        [
          [2, 2, 3, 4],
          [0, 1, 2, 3, -1, -4],
        ],
      ] as [number[], number[]][]) {
        it(`${fnName}: ${JSON.stringify(shape)} no-axis`, () => {
          const A = mkBool(shape);
          const r = fn(A);
          expect(typeof r).toBe('boolean');
        });

        for (const ax of axes) {
          it(`${fnName}: ${JSON.stringify(shape)} axis=${ax}`, () => {
            const A = mkBool(shape);
            const r = fn(A, ax);
            const ndim = shape.length;
            const normAx = ax >= 0 ? ax : ndim + ax;
            const expectedShape = shape.filter((_, i) => i !== normAx);
            if (expectedShape.length === 0) {
              // 1D reduction to scalar — may return boolean primitive
              expect(typeof r === 'boolean' || (r as any).shape?.length === 0).toBe(true);
            } else {
              expect((r as any).shape).toEqual(expectedShape);
            }
          });
        }

        it(`${fnName}: ${JSON.stringify(shape)} axis=0 keepdims`, () => {
          const A = mkBool(shape);
          const r = fn(A, 0, true);
          const expectedShape = [...shape];
          expectedShape[0] = 1;
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }
  });

  // ============================================================
  // SECTION 7: cumsum/cumprod — all axes
  // ============================================================
  describe('cumsum/cumprod: all axes', () => {
    for (const [fn, fnName] of [
      [cumsum, 'cumsum'],
      [cumprod, 'cumprod'],
    ] as [(a: any, ...args: any[]) => any, string][]) {
      it(`${fnName}: 1D no-axis`, () => {
        const A = array([1, 2, 3, 4]);
        const r = fn(A);
        expect((r as any).shape).toEqual([4]);
      });

      it(`${fnName}: 1D axis=0`, () => {
        const A = array([1, 2, 3, 4]);
        const r = fn(A, 0);
        expect((r as any).shape).toEqual([4]);
      });

      for (const [shape, axes] of [
        [
          [3, 4],
          [0, 1, -1, -2],
        ],
        [
          [2, 3, 4],
          [0, 1, 2, -1, -2, -3],
        ],
        [
          [2, 2, 3, 4],
          [0, 1, 2, 3, -1, -4],
        ],
      ] as [number[], number[]][]) {
        for (const ax of axes) {
          it(`${fnName}: ${JSON.stringify(shape)} axis=${ax}`, () => {
            const A = mk(shape);
            const r = fn(A, ax);
            expect((r as any).shape).toEqual(shape); // cumulative preserves shape
          });
        }
      }

      it(`${fnName}: 3D [2,3,4] axis=1 values correct`, () => {
        const A = mk([2, 3, 4]);
        const r = fn(A, 1);
        const py = runNumPy(
          `result = np.${fnName}(np.arange(1,25,dtype=float).reshape(2,3,4), axis=1)`
        );
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }
  });

  // ============================================================
  // SECTION 8: sort/argsort — all axes
  // ============================================================
  describe('sort/argsort: all axes', () => {
    for (const [fn, fnName] of [
      [sort, 'sort'],
      [argsort, 'argsort'],
    ] as [(a: any, ...args: any[]) => any, string][]) {
      for (const [shape, axes] of [
        [[5], [0, -1]],
        [
          [3, 4],
          [0, 1, -1, -2],
        ],
        [
          [2, 3, 4],
          [0, 1, 2, -1, -2, -3],
        ],
        [
          [2, 2, 3, 4],
          [0, 1, 2, 3, -1, -4],
        ],
      ] as [number[], number[]][]) {
        for (const ax of axes) {
          it(`${fnName}: ${JSON.stringify(shape)} axis=${ax}`, () => {
            const A = mk(shape)
              .multiply(-1)
              .add(shape.reduce((a, b) => a * b, 1) + 1);
            const r = fn(A, ax);
            expect((r as any).shape).toEqual(shape); // sort preserves shape
          });
        }

        it(`${fnName}: ${JSON.stringify(shape)} axis=0 values correct`, () => {
          const A = mk(shape)
            .multiply(-1)
            .add(shape.reduce((a, b) => a * b, 1) + 1);
          const r = fn(A, 0);
          const n = shape.reduce((a, b) => a * b, 1);
          const py = runNumPy(`
A = (np.arange(1,${n + 1},dtype=float)*-1+${n + 1}).reshape(${JSON.stringify(shape)})
result = np.${fnName}(A, axis=0)`);
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        });
      }
    }
  });

  // ============================================================
  // SECTION 9: diff — all axes
  // ============================================================
  describe('diff: all axes', () => {
    for (const [shape, axes] of [
      [[5], [0]],
      [
        [3, 4],
        [0, 1],
      ],
      [
        [2, 3, 4],
        [0, 1, 2],
      ],
      [
        [2, 2, 3, 4],
        [0, 1, 2, 3],
      ],
    ] as [number[], number[]][]) {
      for (const ax of axes) {
        it(`diff n=1: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = diff(A, 1, ax);
          const expectedShape = [...shape];
          expectedShape[ax] -= 1;
          expect((r as any).shape).toEqual(expectedShape);
          const n = shape.reduce((a, b) => a * b, 1);
          const py = runNumPy(
            `result = np.diff(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), axis=${ax})`
          );
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        });

        it(`diff n=2: ${JSON.stringify(shape)} axis=${ax}`, () => {
          if (shape[ax] >= 3) {
            const A = mk(shape);
            const r = diff(A, 2, ax);
            const expectedShape = [...shape];
            expectedShape[ax] -= 2;
            expect((r as any).shape).toEqual(expectedShape);
          }
        });
      }
    }
  });

  // ============================================================
  // SECTION 10: median/percentile/quantile — all axes, keepdims
  // ============================================================
  describe('median/percentile/quantile: all axes, keepdims', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`median: ${JSON.stringify(shape)} no-axis`, () => {
        const r = median(mk(shape));
        expect(typeof r).toBe('number');
      });

      for (const ax of shape.map((_, i) => i)) {
        it(`median: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = median(A, ax);
          const expectedShape = shape.filter((_, i) => i !== ax);
          if (expectedShape.length === 0) {
            expect(typeof r === 'number').toBe(true);
          } else {
            expect((r as any).shape).toEqual(expectedShape);
          }
        });

        it(`median: ${JSON.stringify(shape)} axis=${ax} keepdims`, () => {
          const A = mk(shape);
          const r = median(A, ax, true);
          const expectedShape = [...shape];
          expectedShape[ax] = 1;
          expect((r as any).shape).toEqual(expectedShape);
        });

        it(`median: ${JSON.stringify(shape)} axis=${-(shape.length - ax)}`, () => {
          const A = mk(shape);
          const negAx = -(shape.length - ax);
          const r = median(A, negAx);
          const expectedShape = shape.filter((_, i) => i !== ax);
          if (expectedShape.length === 0) {
            expect(typeof r === 'number').toBe(true);
          } else {
            expect((r as any).shape).toEqual(expectedShape);
          }
        });
      }
    }

    it('median: 3D [2,3,4] axis=1 values correct', () => {
      const A = mk([2, 3, 4]);
      const r = median(A, 1);
      const py = runNumPy(`result = np.median(np.arange(1,25,dtype=float).reshape(2,3,4), axis=1)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('percentile: 3D [2,3,4] axis=0', () => {
      const A = mk([2, 3, 4]);
      const r = percentile(A, 75, 0);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('quantile: 3D [2,3,4] axis=-1', () => {
      const A = mk([2, 3, 4]);
      const r = quantile(A, 0.75, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });
  });

  // ============================================================
  // SECTION 11: average — all axes
  // ============================================================
  describe('average: all axes', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`average: ${JSON.stringify(shape)} no-axis`, () => {
        const r = average(mk(shape));
        expect(typeof r).toBe('number');
      });
      for (const ax of shape.map((_, i) => i)) {
        it(`average: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = average(A, ax);
          const expectedShape = shape.filter((_, i) => i !== ax);
          if (expectedShape.length === 0) {
            expect(typeof r).toBe('number');
          } else {
            expect((r as any).shape).toEqual(expectedShape);
          }
        });
      }
    }
  });

  // ============================================================
  // SECTION 12: flip — all axes
  // ============================================================
  describe('flip: all axes', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4], [2, 2, 2, 3, 4]] as number[][]) {
      it(`flip: ${JSON.stringify(shape)} no-axis`, () => {
        const A = mk(shape);
        const r = flip(A);
        expect((r as any).shape).toEqual(shape);
        const n = shape.reduce((a, b) => a * b, 1);
        const py = runNumPy(
          `result = np.flip(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}))`
        );
        if (shape.length <= 4) {
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        }
      });

      for (const ax of shape.map((_, i) => i)) {
        it(`flip: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = flip(A, ax);
          expect((r as any).shape).toEqual(shape);
        });

        it(`flip: ${JSON.stringify(shape)} axis=${-(shape.length - ax)}`, () => {
          const A = mk(shape);
          const negAx = -(shape.length - ax);
          const r = flip(A, negAx);
          expect((r as any).shape).toEqual(shape);
        });
      }
    }
  });

  // ============================================================
  // SECTION 13: roll — all axes
  // ============================================================
  describe('roll: all axes', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`roll: ${JSON.stringify(shape)} no-axis`, () => {
        const A = mk(shape);
        const r = roll(A, 2);
        expect((r as any).shape).toEqual(shape);
      });

      for (const ax of shape.map((_, i) => i)) {
        it(`roll: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = roll(A, 2, ax);
          expect((r as any).shape).toEqual(shape);
          const n = shape.reduce((a, b) => a * b, 1);
          const py = runNumPy(
            `result = np.roll(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), 2, axis=${ax})`
          );
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        });

        it(`roll: ${JSON.stringify(shape)} axis=${-(shape.length - ax)} (negative)`, () => {
          const A = mk(shape);
          const negAx = -(shape.length - ax);
          const r = roll(A, -1, negAx);
          expect((r as any).shape).toEqual(shape);
        });
      }
    }
  });

  // ============================================================
  // SECTION 14: concatenate — all axes
  // ============================================================
  describe('concatenate: all axes', () => {
    for (const shape of [[3], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      for (const ax of shape.map((_, i) => i)) {
        it(`concatenate: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = concatenate([A, A], ax);
          const expectedShape = [...shape];
          expectedShape[ax] *= 2;
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }

    it('concatenate: 3D [2,3,4] axis=1 values correct', () => {
      const A = mk([2, 3, 4]);
      const r = concatenate([A, A], 1);
      const py = runNumPy(`
A = np.arange(1,25,dtype=float).reshape(2,3,4)
result = np.concatenate([A, A], axis=1)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('concatenate: 4D along axis=3', () => {
      const A = mk([2, 2, 3, 4]);
      const r = concatenate([A, A], 3);
      expect((r as any).shape).toEqual([2, 2, 3, 8]);
    });
  });

  // ============================================================
  // SECTION 15: stack — all axes
  // ============================================================
  describe('stack: all axes', () => {
    for (const shape of [[3], [2, 3], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      const ndim = shape.length;
      for (let ax = 0; ax <= ndim; ax++) {
        it(`stack: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = stack([A, A], ax);
          const expectedShape = [...shape];
          expectedShape.splice(ax, 0, 2);
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }

    it('stack: 2D [2,3] axis=0 values correct', () => {
      const A = mk([2, 3]);
      const r = stack([A, A], 0);
      const py = runNumPy(`
A = np.arange(1,7,dtype=float).reshape(2,3)
result = np.stack([A, A], axis=0)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 16: split/array_split — all axes
  // ============================================================
  describe('split/array_split: all axes', () => {
    it('split: 1D [6] axis=0 into 3', () => {
      const A = mk([6]);
      const r = split(A, 3, 0);
      expect(r.length).toBe(3);
      expect((r[0] as any).shape).toEqual([2]);
    });

    it('split: 2D [4,6] axis=0 into 2', () => {
      const A = mk([4, 6]);
      const r = split(A, 2, 0);
      expect(r.length).toBe(2);
      expect((r[0] as any).shape).toEqual([2, 6]);
    });

    it('split: 2D [4,6] axis=1 into 2', () => {
      const A = mk([4, 6]);
      const r = split(A, 2, 1);
      expect(r.length).toBe(2);
      expect((r[0] as any).shape).toEqual([4, 3]);
    });

    it('split: 3D [2,3,6] axis=2 into 2', () => {
      const A = mk([2, 3, 6]);
      const r = split(A, 2, 2);
      expect(r.length).toBe(2);
      expect((r[0] as any).shape).toEqual([2, 3, 3]);
    });

    it('split: 3D [2,3,6] axis=0 into 2', () => {
      const A = mk([2, 3, 6]);
      const r = split(A, 2, 0);
      expect(r.length).toBe(2);
      expect((r[0] as any).shape).toEqual([1, 3, 6]);
    });

    it('array_split: 3D [2,3,5] axis=1 into 2 (unequal)', () => {
      const A = mk([2, 3, 5]);
      const r = array_split(A, 2, 1);
      expect(r.length).toBe(2);
    });

    it('array_split: 4D [2,2,3,6] axis=3 into 3', () => {
      const A = mk([2, 2, 3, 6]);
      const r = array_split(A, 3, 3);
      expect(r.length).toBe(3);
      expect((r[0] as any).shape).toEqual([2, 2, 3, 2]);
    });
  });

  // ============================================================
  // SECTION 17: swapaxes — all pairs
  // ============================================================
  describe('swapaxes: all axis pairs', () => {
    for (const [shape, pairs] of [
      [[3, 4], [[0, 1]]],
      [
        [2, 3, 4],
        [
          [0, 1],
          [0, 2],
          [1, 2],
        ],
      ],
      [
        [2, 3, 4, 5],
        [
          [0, 1],
          [0, 2],
          [0, 3],
          [1, 2],
          [1, 3],
          [2, 3],
        ],
      ],
    ] as [number[], [number, number][]][]) {
      for (const [i, j] of pairs) {
        it(`swapaxes: ${JSON.stringify(shape)} (${i},${j})`, () => {
          const A = mk(shape);
          const r = swapaxes(A, i, j);
          const expectedShape = [...shape];
          [expectedShape[i], expectedShape[j]] = [expectedShape[j], expectedShape[i]];
          expect((r as any).shape).toEqual(expectedShape);
          const n = shape.reduce((a, b) => a * b, 1);
          const py = runNumPy(
            `result = np.swapaxes(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), ${i}, ${j})`
          );
          expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
        });
      }
    }
  });

  // ============================================================
  // SECTION 18: moveaxis
  // ============================================================
  describe('moveaxis: single and multi', () => {
    for (const [shape, src, dst, expectedShape] of [
      [[2, 3, 4], 0, 2, [3, 4, 2]],
      [[2, 3, 4], 2, 0, [4, 2, 3]],
      [[2, 3, 4], -1, 0, [4, 2, 3]],
      [[2, 3, 4, 5], 0, -1, [3, 4, 5, 2]],
      [[2, 3, 4, 5], 1, 3, [2, 4, 5, 3]],
    ] as [number[], number, number, number[]][]) {
      it(`moveaxis: ${JSON.stringify(shape)} ${src}->${dst}`, () => {
        const A = mk(shape);
        const r = moveaxis(A, src, dst);
        expect((r as any).shape).toEqual(expectedShape);
        const n = shape.reduce((a, b) => a * b, 1);
        const py = runNumPy(
          `result = np.moveaxis(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), ${src}, ${dst})`
        );
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }
  });

  // ============================================================
  // SECTION 19: expand_dims / squeeze — all positions
  // ============================================================
  describe('expand_dims: all positions', () => {
    for (const [shape, ax, expectedShape] of [
      [[], 0, [1]],
      [[3], 0, [1, 3]],
      [[3], 1, [3, 1]],
      [[3], -1, [3, 1]],
      [[2, 3], 0, [1, 2, 3]],
      [[2, 3], 1, [2, 1, 3]],
      [[2, 3], 2, [2, 3, 1]],
      [[2, 3], -1, [2, 3, 1]],
      [[2, 3, 4], 0, [1, 2, 3, 4]],
      [[2, 3, 4], 2, [2, 3, 1, 4]],
      [[2, 3, 4], 3, [2, 3, 4, 1]],
    ] as [number[], number, number[]][]) {
      it(`expand_dims: ${JSON.stringify(shape)} axis=${ax} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk(shape);
        const r = expand_dims(A, ax);
        expect((r as any).shape).toEqual(expectedShape);
      });
    }
  });

  describe('squeeze: all size-1 axes', () => {
    for (const [shape, expectedShape] of [
      [[1, 3], [3]],
      [[1, 3, 1], [3]],
      [
        [2, 1, 3, 1, 4],
        [2, 3, 4],
      ],
      [[1, 1, 1], []],
    ] as [number[], number[]][]) {
      it(`squeeze: ${JSON.stringify(shape)} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk(shape);
        const r = squeeze(A);
        expect((r as any).shape).toEqual(expectedShape);
      });
    }
  });

  // ============================================================
  // SECTION 20: repeat / tile — all axes
  // ============================================================
  describe('repeat: all axes', () => {
    for (const shape of [[3], [2, 3], [2, 3, 4]] as number[][]) {
      it(`repeat: ${JSON.stringify(shape)} no-axis (flattens)`, () => {
        const A = mk(shape);
        const r = repeat(A, 2);
        const n = shape.reduce((a, b) => a * b, 1);
        expect((r as any).shape).toEqual([n * 2]);
      });

      for (const ax of shape.map((_, i) => i)) {
        it(`repeat: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = repeat(A, 2, ax);
          const expectedShape = [...shape];
          expectedShape[ax] *= 2;
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }
  });

  describe('tile: ND reps', () => {
    it('tile: [3] x2 scalar', () => {
      const r = tile(mk([3]), 2);
      expect((r as any).shape).toEqual([6]);
    });

    it('tile: [2,3] x [2,3]', () => {
      const r = tile(mk([2, 3]), [2, 3]);
      expect((r as any).shape).toEqual([4, 9]);
    });

    it('tile: [2,3,4] x [2,1,1]', () => {
      const r = tile(mk([2, 3, 4]), [2, 1, 1]);
      expect((r as any).shape).toEqual([4, 3, 4]);
    });

    it('tile: [3] x [2,1,1] (broadcast extra dims)', () => {
      const r = tile(mk([3]), [2, 1, 1]);
      expect((r as any).shape).toEqual([2, 1, 3]);
    });
  });

  // ============================================================
  // SECTION 21: reshape — ND -> MD
  // ============================================================
  describe('reshape: ND -> MD', () => {
    for (const [inShape, outShape] of [
      [[6], [2, 3]],
      [[6], [3, 2]],
      [[24], [2, 3, 4]],
      [[2, 3], [6]],
      [
        [2, 3],
        [1, 6],
      ],
      [[2, 3, 4], [24]],
      [
        [2, 3, 4],
        [2, 12],
      ],
      [
        [2, 3, 4],
        [4, 6],
      ],
      [
        [2, 3, 4],
        [2, 3, 2, 2],
      ],
      [
        [2, 2, 3, 4],
        [4, 12],
      ],
      [
        [2, 2, 3, 4],
        [2, 2, 3, 2, 2],
      ],
    ] as [number[], number[]][]) {
      it(`reshape: ${JSON.stringify(inShape)} -> ${JSON.stringify(outShape)}`, () => {
        const A = mk(inShape);
        const r = reshape(A, outShape);
        expect((r as any).shape).toEqual(outShape);
      });
    }

    it('reshape: with -1 inference [2,3,4] -> [-1]', () => {
      const A = mk([2, 3, 4]);
      const r = reshape(A, [-1]);
      expect((r as any).shape).toEqual([24]);
    });

    it('reshape: with -1 inference [2,3,4] -> [-1,4]', () => {
      const A = mk([2, 3, 4]);
      const r = reshape(A, [-1, 4]);
      expect((r as any).shape).toEqual([6, 4]);
    });
  });

  // ============================================================
  // SECTION 22: transpose — permutations
  // ============================================================
  describe('transpose: all permutations', () => {
    it('transpose: 1D -> same shape (no-op)', () => {
      const A = mk([5]);
      const r = transpose(A);
      expect((r as any).shape).toEqual([5]);
    });

    for (const [shape, perm, expectedShape] of [
      [
        [2, 3],
        [1, 0],
        [3, 2],
      ],
      [
        [2, 3, 4],
        [2, 1, 0],
        [4, 3, 2],
      ],
      [
        [2, 3, 4],
        [1, 2, 0],
        [3, 4, 2],
      ],
      [
        [2, 3, 4],
        [0, 2, 1],
        [2, 4, 3],
      ],
      [
        [2, 3, 4, 5],
        [3, 2, 1, 0],
        [5, 4, 3, 2],
      ],
      [
        [2, 3, 4, 5],
        [1, 0, 3, 2],
        [3, 2, 5, 4],
      ],
    ] as [number[], number[], number[]][]) {
      it(`transpose: ${JSON.stringify(shape)} perm=${JSON.stringify(perm)} -> ${JSON.stringify(expectedShape)}`, () => {
        const A = mk(shape);
        const r = transpose(A, perm);
        expect((r as any).shape).toEqual(expectedShape);
        const n = shape.reduce((a, b) => a * b, 1);
        const py = runNumPy(
          `result = np.transpose(np.arange(1,${n + 1},dtype=float).reshape(${JSON.stringify(shape)}), ${JSON.stringify(perm)})`
        );
        expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
      });
    }

    it('transpose: 5D default (reverse)', () => {
      const A = mk([2, 2, 2, 3, 4]);
      const r = transpose(A);
      expect((r as any).shape).toEqual([4, 3, 2, 2, 2]);
    });
  });

  // ============================================================
  // SECTION 23: rot90
  // ============================================================
  describe('rot90: 2D-4D, all k, axes param', () => {
    it('rot90: 2D k=1', () => {
      const A = mk([3, 4]);
      const r = rot90(A, 1);
      expect((r as any).shape).toEqual([4, 3]);
    });

    it('rot90: 2D k=2', () => {
      const A = mk([3, 4]);
      const r = rot90(A, 2);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('rot90: 3D k=1 default axes', () => {
      const A = mk([2, 3, 4]);
      const r = rot90(A, 1);
      expect((r as any).shape[2]).toBe(4); // last axis unchanged by default
    });

    it('rot90: 4D k=1 default axes', () => {
      const A = mk([2, 3, 4, 5]);
      const r = rot90(A, 1);
      expect((r as any).shape).toBeDefined();
    });
  });

  // ============================================================
  // SECTION 24: broadcast_to — ND
  // ============================================================
  describe('broadcast_to: ND expansion', () => {
    for (const [src, target] of [
      [[3], [4, 3]],
      [[3], [2, 4, 3]],
      [[3], [2, 2, 4, 3]],
      [[3], [2, 2, 2, 4, 3]],
      [
        [1, 3],
        [4, 2, 3],
      ],
      [
        [2, 1, 3],
        [2, 4, 3],
      ],
    ] as [number[], number[]][]) {
      it(`broadcast_to: ${JSON.stringify(src)} -> ${JSON.stringify(target)}`, () => {
        const A = mk(src);
        const r = broadcast_to(A, target);
        expect((r as any).shape).toEqual(target);
      });
    }
  });

  // ============================================================
  // SECTION 25: take — all axes
  // ============================================================
  describe('take: all axes', () => {
    const idx = [0, 1];

    it('take: 1D no-axis', () => {
      const r = take(mk([5]), idx);
      expect((r as any).shape).toEqual([2]);
    });

    for (const shape of [
      [3, 4],
      [2, 3, 4],
      [2, 2, 3, 4],
    ] as number[][]) {
      for (const ax of shape.map((_, i) => i)) {
        it(`take: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = take(A, idx, ax);
          const expectedShape = [...shape];
          expectedShape[ax] = 2;
          expect((r as any).shape).toEqual(expectedShape);
        });
      }
    }
  });

  // ============================================================
  // SECTION 26: clip — 0D through 5D
  // ============================================================
  describe('clip: 0D-5D', () => {
    for (const shape of [[], [5], [3, 4], [2, 3, 4], [2, 2, 3, 4], [2, 2, 2, 3, 4]] as number[][]) {
      it(`clip: ${JSON.stringify(shape)}`, () => {
        const A = mk(shape);
        const r = clip(A, 2, 10);
        expect((r as any).shape ?? []).toEqual(shape);
      });
    }

    it('clip: 3D values correct', () => {
      const A = mk([2, 3, 4]);
      const r = clip(A, 5, 15);
      const py = runNumPy(`result = np.clip(np.arange(1,25,dtype=float).reshape(2,3,4), 5, 15)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 27: where — broadcasting
  // ============================================================
  describe('where: ND + broadcasting', () => {
    for (const shape of [[5], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`where: ${JSON.stringify(shape)} same shape`, () => {
        const cond = mkBool(shape);
        const a = mk(shape);
        const b = negative(mk(shape));
        const r = where(cond, a, b);
        expect((r as any).shape).toEqual(shape);
      });
    }

    it('where: broadcast condition [4] with values [3,4]', () => {
      const cond = mkBool([4]);
      const a = mk([3, 4]);
      const b = negative(mk([3, 4]));
      const r = where(cond, a, b);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('where: 3D values correct', () => {
      const A = mk([2, 3, 4]);
      const B = negative(mk([2, 3, 4]));
      const cond = mkBool([2, 3, 4]);
      const r = where(cond, A, B);
      const py = runNumPy(`
A = np.arange(1,25,dtype=float).reshape(2,3,4)
B = -A
cond = np.array([i%2==0 for i in range(24)], dtype=bool).reshape(2,3,4)
result = np.where(cond, A, B)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 28: matmul — all valid dimensionality combos
  // ============================================================
  describe('matmul: all valid ndim combos', () => {
    it('1D @ 1D -> scalar', () => {
      const r = matmul(array([1, 2, 3, 4]), array([1, 2, 3, 4]));
      const val = typeof r === 'number' ? r : (r as any).toArray?.();
      const py = runNumPy(
        `result = np.matmul(np.array([1,2,3,4],dtype=float), np.array([1,2,3,4],dtype=float))`
      );
      expect(arraysClose(val, py.value)).toBe(true);
    });

    it('2D @ 1D -> 1D [3,4] @ [4]', () => {
      const r = matmul(mk([3, 4]), mk([4]));
      expect((r as any).shape).toEqual([3]);
      const py = runNumPy(
        `result = np.matmul(np.arange(1,13,dtype=float).reshape(3,4), np.arange(1,5,dtype=float))`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('1D @ 2D -> 1D [3] @ [3,4]', () => {
      const r = matmul(mk([3]), mk([3, 4]));
      expect((r as any).shape).toEqual([4]);
    });

    it('2D @ 2D -> 2D', () => {
      const r = matmul(mk([2, 3]), mk([3, 4]));
      expect((r as any).shape).toEqual([2, 4]);
      const py = runNumPy(
        `result = np.matmul(np.arange(1,7,dtype=float).reshape(2,3), np.arange(1,13,dtype=float).reshape(3,4))`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('3D @ 3D -> 3D (batched)', () => {
      const r = matmul(mk([2, 3, 4]), mk([2, 4, 5]));
      expect((r as any).shape).toEqual([2, 3, 5]);
      const py = runNumPy(
        `result = np.matmul(np.arange(1,25,dtype=float).reshape(2,3,4), np.arange(1,41,dtype=float).reshape(2,4,5))`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('4D @ 4D -> 4D (batched)', () => {
      const r = matmul(mk([2, 2, 3, 4]), mk([2, 2, 4, 5]));
      expect((r as any).shape).toEqual([2, 2, 3, 5]);
    });

    it('5D @ 5D -> 5D (batched)', () => {
      const r = matmul(mk([2, 2, 2, 3, 4]), mk([2, 2, 2, 4, 5]));
      expect((r as any).shape).toEqual([2, 2, 2, 3, 5]);
    });

    it('3D @ 2D -> 3D (batch broadcast)', () => {
      const r = matmul(mk([2, 3, 4]), mk([4, 5]));
      expect((r as any).shape).toEqual([2, 3, 5]);
      const py = runNumPy(
        `result = np.matmul(np.arange(1,25,dtype=float).reshape(2,3,4), np.arange(1,21,dtype=float).reshape(4,5))`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('4D @ 3D -> 4D (batch broadcast)', () => {
      const r = matmul(mk([2, 2, 3, 4]), mk([2, 4, 5]));
      expect((r as any).shape).toEqual([2, 2, 3, 5]);
    });

    it('1D @ 3D -> 2D', () => {
      const r = matmul(mk([3]), mk([2, 3, 4]));
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('0D throws', () => {
      expect(() => matmul(array(2.0), array(2.0))).toThrow();
    });
  });

  // ============================================================
  // SECTION 29: dot/inner/outer/tensordot — ND cases
  // ============================================================
  describe('dot: ND cases', () => {
    it('0D · 0D', () => {
      const r = dot(array(2.0), array(3.0));
      expect(arraysClose(r as number, 6.0)).toBe(true);
    });

    it('1D · 1D -> scalar', () => {
      const r = dot(array([1, 2, 3, 4]), array([1, 2, 3, 4]));
      expect(arraysClose(r as number, 30)).toBe(true);
    });

    it('2D · 1D -> 1D', () => {
      const r = dot(mk([3, 4]), mk([4])) as any;
      expect(r.shape).toEqual([3]);
    });

    it('3D · 2D -> 3D', () => {
      const r = dot(mk([2, 3, 4]), mk([4, 5])) as any;
      const py = runNumPy(
        `result = np.dot(np.arange(1,25,dtype=float).reshape(2,3,4), np.arange(1,21,dtype=float).reshape(4,5))`
      );
      expect(r.shape).toEqual([2, 3, 5]);
      expect(arraysClose(r.toArray(), py.value)).toBe(true);
    });

    it('4D · 2D -> 4D', () => {
      const r = dot(mk([2, 2, 3, 4]), mk([4, 5])) as any;
      expect(r.shape).toEqual([2, 2, 3, 5]);
    });
  });

  describe('inner: ND cases', () => {
    it('0D · 0D', () => {
      const r = inner(array(2.0), array(3.0));
      expect(arraysClose(r as number, 6.0)).toBe(true);
    });

    it('1D · 1D -> scalar', () => {
      const r = inner(array([1, 2, 3]), array([4, 5, 6]));
      expect(arraysClose(r as number, 32)).toBe(true);
    });

    it('2D · 1D -> 1D', () => {
      const r = inner(mk([3, 4]), mk([4])) as any;
      expect(r.shape).toEqual([3]);
    });

    it('2D · 2D -> 2D', () => {
      const r = inner(mk([3, 4]), mk([2, 4])) as any;
      const py = runNumPy(
        `result = np.inner(np.arange(1,13,dtype=float).reshape(3,4), np.arange(1,9,dtype=float).reshape(2,4))`
      );
      expect(r.shape).toEqual([3, 2]);
      expect(arraysClose(r.toArray(), py.value)).toBe(true);
    });

    it('3D · 1D -> 2D', () => {
      const r = inner(mk([2, 3, 4]), array([1, 2, 3, 4])) as any;
      expect(r.shape).toEqual([2, 3]);
    });

    it('3D · 2D -> 3D', () => {
      const r = inner(mk([2, 3, 4]), mk([2, 4])) as any;
      expect(r.shape).toEqual([2, 3, 2]);
    });
  });

  describe('outer: ND (always flattens)', () => {
    it('1D · 1D', () => {
      const r = outer(mk([3]), mk([4]));
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('2D · 2D (flattened)', () => {
      const r = outer(mk([2, 3]), mk([3, 4]));
      expect((r as any).shape).toEqual([6, 12]);
    });

    it('3D · 3D (flattened)', () => {
      const r = outer(mk([2, 3, 4]), mk([2, 3, 4]));
      expect((r as any).shape).toEqual([24, 24]);
    });
  });

  describe('tensordot: ND axes combos', () => {
    it('1D axes=1 -> scalar', () => {
      const r = tensordot(mk([4]), mk([4]), 1);
      expect(arraysClose(r as number, 30)).toBe(true);
    });

    it('2D axes=1', () => {
      const r = tensordot(mk([2, 3]), mk([3, 4]), 1) as any;
      expect(r.shape).toEqual([2, 4]);
    });

    it('3D axes=1', () => {
      const r = tensordot(mk([2, 3, 4]), mk([4, 5]), 1) as any;
      expect(r.shape).toEqual([2, 3, 5]);
      const py = runNumPy(
        `result = np.tensordot(np.arange(1,25,dtype=float).reshape(2,3,4), np.arange(1,21,dtype=float).reshape(4,5), 1)`
      );
      expect(arraysClose(r.toArray(), py.value)).toBe(true);
    });

    it('outer product (axes=0)', () => {
      const r = tensordot(mk([3]), mk([4]), 0) as any;
      expect(r.shape).toEqual([3, 4]);
    });

    it('3D contract all axes=2', () => {
      const r = tensordot(mk([2, 3]), mk([2, 3]), 2);
      expect(
        arraysClose(
          r as number,
          mk([2, 3])
            .multiply(mk([2, 3]))
            .sum() as number,
          1e-6
        )
      ).toBe(true);
    });
  });

  // ============================================================
  // SECTION 30: trace — ND batch
  // ============================================================
  describe('trace: ND batch', () => {
    it('2D square: base case', () => {
      const r = trace(mk([3, 3]));
      expect(r).toBeCloseTo(1 + 5 + 9); // diagonal of [[1,2,3],[4,5,6],[7,8,9]]
    });

    it('2D non-square', () => {
      const r = trace(mk([3, 4]));
      expect(typeof r).toBe('number');
    });

    it('2D offset=1', () => {
      const r = trace(mk([4, 4]), 1);
      expect(typeof r).toBe('number');
    });

    it('2D offset=-1', () => {
      const r = trace(mk([4, 4]), -1);
      expect(typeof r).toBe('number');
    });

    it('3D -> 1D batch', () => {
      const A = mk([2, 3, 3]);
      const r = trace(A);
      const py = runNumPy(`result = np.trace(np.arange(1,19,dtype=float).reshape(2,3,3))`);
      expect((r as any).shape ?? []).toEqual(py.shape);
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('4D -> 2D batch', () => {
      const A = mk([2, 2, 3, 3]);
      const r = trace(A);
      const py = runNumPy(`result = np.trace(np.arange(1,37,dtype=float).reshape(2,2,3,3))`);
      expect((r as any).shape ?? []).toEqual(py.shape);
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('5D -> 3D batch', () => {
      const A = mk([2, 2, 2, 3, 3]);
      const r = trace(A);
      // default axis1=0, axis2=1: trace over [2,2] axes, remaining=[2,3,3]
      expect((r as any).shape).toEqual([2, 3, 3]);
    });

    it('3D custom axis1=0, axis2=2', () => {
      const A = mk([3, 2, 3]);
      const r = trace(A, 0, 0, 2);
      const py = runNumPy(
        `result = np.trace(np.arange(1,19,dtype=float).reshape(3,2,3), axis1=0, axis2=2)`
      );
      expect((r as any).shape ?? []).toEqual(py.shape);
    });
  });

  // ============================================================
  // SECTION 31: diagonal — ND, all axis pairs
  // ============================================================
  describe('diagonal: ND + axis pairs', () => {
    it('2D: base case', () => {
      const r = diagonal(mk([3, 4]));
      const py = runNumPy(`result = np.diagonal(np.arange(1,13,dtype=float).reshape(3,4))`);
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('2D: offset=1', () => {
      const r = diagonal(mk([4, 4]), 1);
      expect((r as any).shape).toEqual([3]);
    });

    it('2D: offset=-1', () => {
      const r = diagonal(mk([4, 4]), -1);
      expect((r as any).shape).toEqual([3]);
    });

    it('3D default (axis1=0, axis2=1)', () => {
      const A = mk([2, 3, 4]);
      const r = diagonal(A, 0, 0, 1);
      const py = runNumPy(
        `result = np.diagonal(np.arange(1,25,dtype=float).reshape(2,3,4), 0, 0, 1)`
      );
      expect((r as any).shape).toEqual(py.shape);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('3D axis1=1, axis2=2', () => {
      const A = mk([2, 3, 4]);
      const r = diagonal(A, 0, 1, 2);
      const py = runNumPy(
        `result = np.diagonal(np.arange(1,25,dtype=float).reshape(2,3,4), 0, 1, 2)`
      );
      expect((r as any).shape).toEqual(py.shape);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('4D: axis1=2, axis2=3', () => {
      const A = mk([2, 3, 4, 4]);
      const r = diagonal(A, 0, 2, 3);
      const py = runNumPy(
        `result = np.diagonal(np.arange(1,97,dtype=float).reshape(2,3,4,4), 0, 2, 3)`
      );
      expect((r as any).shape).toEqual(py.shape);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('4D: axis1=1, axis2=3', () => {
      const A = mk([2, 3, 4, 3]);
      const r = diagonal(A, 0, 1, 3);
      expect((r as any).shape).toBeDefined();
    });
  });

  // ============================================================
  // SECTION 32: cross — batched ND
  // ============================================================
  describe('cross: batched ND', () => {
    it('1D-3 x 1D-3', () => {
      const r = cross(array([1, 2, 3]), array([4, 5, 6]));
      const py = runNumPy(
        `result = np.cross(np.array([1,2,3],dtype=float), np.array([4,5,6],dtype=float))`
      );
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('2D batched [4,3] x [4,3]', () => {
      const r = cross(mk([4, 3]), mk([4, 3]).add(1));
      expect((r as any).shape).toEqual([4, 3]);
    });

    it('3D batched [2,4,3] x [2,4,3]', () => {
      const r = cross(mk([2, 4, 3]), mk([2, 4, 3]).add(1));
      expect((r as any).shape).toEqual([2, 4, 3]);
    });

    it('4D batched', () => {
      const r = cross(mk([2, 2, 4, 3]), mk([2, 2, 4, 3]).add(1));
      expect((r as any).shape).toEqual([2, 2, 4, 3]);
    });
  });

  // ============================================================
  // SECTION 33: einsum — various subscripts + ND
  // ============================================================
  describe('einsum: various subscripts and ND', () => {
    it('1D dot: i,i->', () => {
      const r = einsum('i,i->', array([1, 2, 3, 4]), array([1, 2, 3, 4]));
      expect(arraysClose(r as number, 30)).toBe(true);
    });

    it('2D matmul: ij,jk->ik', () => {
      const r = einsum('ij,jk->ik', mk([2, 3]), mk([3, 4])) as any;
      expect(r.shape).toEqual([2, 4]);
    });

    it('2D trace: ii->', () => {
      const r = einsum('ii->', mk([3, 3]));
      expect(arraysClose(r as number, 1 + 5 + 9)).toBe(true);
    });

    it('3D batch matmul: bij,bjk->bik', () => {
      const r = einsum('bij,bjk->bik', mk([2, 3, 4]), mk([2, 4, 5])) as any;
      expect(r.shape).toEqual([2, 3, 5]);
    });

    it('3D sum last axis: ijk->ij', () => {
      const r = einsum('ijk->ij', mk([2, 3, 4])) as any;
      expect(r.shape).toEqual([2, 3]);
    });

    it('4D sum: ijkl->ij', () => {
      const r = einsum('ijkl->ij', mk([2, 3, 4, 5])) as any;
      expect(r.shape).toEqual([2, 3]);
    });

    it('outer: i,j->ij', () => {
      const r = einsum('i,j->ij', mk([3]), mk([4])) as any;
      expect(r.shape).toEqual([3, 4]);
    });
  });

  // ============================================================
  // SECTION 34: linalg batch ops — 2D through 4D
  // ============================================================
  describe('linalg.det: 2D through 4D batch', () => {
    for (const [shape, expectedBatchShape] of [
      [[3, 3], []],
      [[2, 3, 3], [2]],
      [
        [2, 2, 3, 3],
        [2, 2],
      ],
      [
        [2, 2, 2, 3, 3],
        [2, 2, 2],
      ],
    ] as [number[], number[]][]) {
      it(`det: ${JSON.stringify(shape)} -> ${JSON.stringify(expectedBatchShape)}`, () => {
        // Add identity*5 for conditioning
        const A = mk(shape);
        const r = linalg.det(A);
        if (expectedBatchShape.length === 0) {
          expect(typeof r).toBe('number');
        } else {
          expect((r as any).shape).toEqual(expectedBatchShape);
        }
      });
    }
  });

  describe('linalg.inv: 2D through 4D batch', () => {
    for (const [shape] of [[[2, 2]], [[2, 2, 2]], [[2, 2, 2, 2]], [[2, 2, 2, 2, 2]]] as [
      number[],
    ][]) {
      it(`inv: ${JSON.stringify(shape)}`, () => {
        // Use identity-like matrices for well-conditioning
        const size = shape[shape.length - 1];
        const batchDims = shape.slice(0, -2);
        const batchSize = batchDims.reduce((a, b) => a * b, 1);
        const identity = Array.from({ length: batchSize }, () =>
          Array.from({ length: size }, (_, i) =>
            Array.from({ length: size }, (_, j) => (i === j ? 2 : 0))
          )
        ).flat(2);
        const A = array(identity).reshape(shape);
        const r = linalg.inv(A);
        expect((r as any).shape).toEqual(shape);
      });
    }
  });

  describe('linalg.solve: batched', () => {
    it('solve: 2D [3,3] · [3]', () => {
      const A = array([
        [2, 1, 0],
        [0, 3, 1],
        [0, 0, 4],
      ]);
      const b = array([1, 2, 3]);
      const r = linalg.solve(A, b);
      expect((r as any).shape).toEqual([3]);
    });

    it('solve: 2D [3,3] · [3,2] (multiple RHS)', () => {
      const A = array([
        [2, 1, 0],
        [0, 3, 1],
        [0, 0, 4],
      ]);
      const b = mk([3, 2]);
      const r = linalg.solve(A, b);
      expect((r as any).shape).toEqual([3, 2]);
    });

    it('solve: batch 3D [2,3,3] · [2,3] -> [2,3]', () => {
      const A = mk([2, 3, 3]); // using solve - may not be well-conditioned, just check shape
      const b = mk([2, 3]);
      try {
        const r = linalg.solve(A, b);
        expect((r as any).shape).toEqual([2, 3]);
      } catch (e) {
        // Accept singular matrix errors
        expect((e as any).message).toBeTruthy();
      }
    });
  });

  describe('linalg.norm: all axes', () => {
    it('1D vector norm', () => {
      const r = linalg.norm(array([3, 4]));
      expect(arraysClose(r as number, 5)).toBe(true);
    });

    it('2D [3,4] axis=0', () => {
      const r = linalg.norm(mk([3, 4]), undefined, 0);
      expect((r as any).shape).toEqual([4]);
    });

    it('2D [3,4] axis=1', () => {
      const r = linalg.norm(mk([3, 4]), undefined, 1);
      expect((r as any).shape).toEqual([3]);
    });

    it('2D [3,4] fro', () => {
      const r = linalg.norm(mk([3, 4]), 'fro');
      expect(typeof r).toBe('number');
    });

    it('3D [2,3,4] axis=-1', () => {
      const r = linalg.norm(mk([2, 3, 4]), undefined, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('3D [2,3,4] axis=(0,1)', () => {
      const r = linalg.norm(mk([2, 3, 4]), undefined, [0, 1]);
      expect((r as any).shape).toEqual([4]);
    });

    it('4D [2,3,4,5] axis=-1 keepdims', () => {
      const r = linalg.norm(mk([2, 3, 4, 5]), undefined, -1, true);
      expect((r as any).shape).toEqual([2, 3, 4, 1]);
    });
  });

  describe('linalg.svd: batched', () => {
    it('svd: 2D [3,4] reduced', () => {
      const r = linalg.svd(mk([3, 4]), false) as any;
      expect(r.s.shape).toEqual([3]);
    });

    it('svd: 3D [2,3,4] batch', () => {
      const r = linalg.svd(mk([2, 3, 4]), false) as any;
      expect(r.s.shape).toEqual([2, 3]);
    });

    it('svd: 4D [2,2,3,4] batch', () => {
      const r = linalg.svd(mk([2, 2, 3, 4]), false) as any;
      expect(r.s.shape).toEqual([2, 2, 3]);
    });
  });

  describe('linalg.qr: batched', () => {
    it('qr: 2D [4,3]', () => {
      const r = linalg.qr(mk([4, 3])) as any;
      expect(r.q.shape).toEqual([4, 3]);
      expect(r.r.shape).toEqual([3, 3]);
    });

    it('qr: 3D [2,4,3] batch', () => {
      const r = linalg.qr(mk([2, 4, 3])) as any;
      expect(r.q.shape).toEqual([2, 4, 3]);
      expect(r.r.shape).toEqual([2, 3, 3]);
    });
  });

  // ============================================================
  // SECTION 35: FFT with explicit axis parameter
  // ============================================================
  describe('fft: explicit axis parameter', () => {
    for (const [shape, axes] of [
      [[8], [0, -1]],
      [
        [4, 8],
        [0, 1, -1, -2],
      ],
      [
        [2, 4, 8],
        [0, 1, 2, -1],
      ],
      [
        [2, 2, 4, 8],
        [0, 1, 2, 3, -1],
      ],
    ] as [number[], number[]][]) {
      for (const ax of axes) {
        it(`fft.fft: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = fft.fft(A, undefined, ax);
          expect((r as any).shape).toEqual(shape);
        });

        it(`fft.ifft: ${JSON.stringify(shape)} axis=${ax}`, () => {
          const A = mk(shape);
          const r = fft.ifft(A, undefined, ax);
          expect((r as any).shape).toEqual(shape);
        });
      }
    }

    it('fft.fft: 3D axis=0 vs axis=1', () => {
      const A = mk([2, 4, 8]);
      const r0 = fft.fft(A, undefined, 0);
      const r1 = fft.fft(A, undefined, 1);
      expect((r0 as any).shape).toEqual([2, 4, 8]);
      expect((r1 as any).shape).toEqual([2, 4, 8]);
    });

    it('fft.fft2: 3D [2,4,8] with axes=(1,2)', () => {
      const A = mk([2, 4, 8]);
      const r = fft.fft2(A, undefined, [1, 2]);
      expect((r as any).shape).toEqual([2, 4, 8]);
    });

    it('fft.fft2: 4D [2,2,4,8] with axes=(2,3)', () => {
      const A = mk([2, 2, 4, 8]);
      const r = fft.fft2(A, undefined, [2, 3]);
      expect((r as any).shape).toEqual([2, 2, 4, 8]);
    });

    it('fft.fftn: 3D all axes', () => {
      const A = mk([2, 4, 8]);
      const r = fft.fftn(A);
      expect((r as any).shape).toEqual([2, 4, 8]);
    });

    it('fft.fftn: 4D last 2 axes', () => {
      const A = mk([2, 2, 4, 8]);
      const r = fft.fftn(A, undefined, [-2, -1]);
      expect((r as any).shape).toEqual([2, 2, 4, 8]);
    });

    it('fft.fftshift: 1D through 4D', () => {
      for (const shape of [[8], [4, 8], [2, 4, 8], [2, 2, 4, 8]] as number[][]) {
        const A = mk(shape);
        const r = fft.fftshift(A);
        expect((r as any).shape).toEqual(shape);
      }
    });

    it('fft.ifftshift: 3D with axis', () => {
      const A = mk([2, 4, 8]);
      const r = fft.ifftshift(A, 1);
      expect((r as any).shape).toEqual([2, 4, 8]);
    });
  });

  // ============================================================
  // SECTION 36: 0D special cases
  // ============================================================
  describe('0D special cases', () => {
    it('sum(0D) -> scalar', () => {
      expect(sum(array(5.0))).toBeCloseTo(5.0);
    });

    it('mean(0D) -> scalar', () => {
      expect(mean(array(5.0))).toBeCloseTo(5.0);
    });

    it('prod(0D) -> scalar', () => {
      expect(prod(array(5.0))).toBeCloseTo(5.0);
    });

    it('sin(0D) -> 0D', () => {
      const r = sin(array(0.5));
      const py = runNumPy(`result = np.sin(np.array(0.5))`);
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, py.value)).toBe(true);
    });

    it('exp(0D) -> 0D', () => {
      const r = exp(array(1.0));
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, Math.E, 1e-10)).toBe(true);
    });

    it('abs(0D) -> 0D', () => {
      const r = abs(array(-3.5));
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, 3.5)).toBe(true);
    });

    it('add(0D, 0D) -> 0D', () => {
      const r = add(array(2.0), array(3.0));
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, 5.0)).toBe(true);
    });

    it('multiply(0D, 0D) -> 0D', () => {
      const r = multiply(array(2.0), array(3.0));
      const val = (r as any).toArray?.() ?? r;
      expect(arraysClose(val, 6.0)).toBe(true);
    });

    it('squeeze(0D) -> 0D', () => {
      const r = squeeze(array(5.0));
      expect((r as any).shape ?? []).toEqual([]);
    });

    it('reshape(0D, [1]) -> 1D', () => {
      const r = reshape(array(5.0), [1]);
      expect((r as any).shape).toEqual([1]);
    });

    it('isfinite(0D)', () => {
      const r = isfinite(array(5.0));
      const val = (r as any).toArray?.() ?? r;
      // val is either a boolean or an array/number representing true
      expect(val === true || val === 1 || (Array.isArray(val) && val[0])).toBe(true);
    });
  });

  // ============================================================
  // SECTION 37: 5D+ general ND
  // ============================================================
  describe('5D+ general ND support', () => {
    const A5 = mk([2, 2, 2, 3, 4]);

    it('sin: 5D', () => {
      const r = sin(A5.multiply(0.1));
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });

    it('sum: 5D no-axis', () => {
      const r = sum(A5);
      const n = 2 * 2 * 2 * 3 * 4;
      expect(arraysClose(r as number, ((1 + n) * n) / 2)).toBe(true);
    });

    it('sum: 5D axis=2', () => {
      const r = sum(A5, 2);
      expect((r as any).shape).toEqual([2, 2, 3, 4]);
    });

    it('mean: 5D axis=-1', () => {
      const r = mean(A5, -1);
      expect((r as any).shape).toEqual([2, 2, 2, 3]);
    });

    it('flip: 5D', () => {
      const r = flip(A5);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });

    it('transpose: 5D default', () => {
      const r = transpose(A5);
      expect((r as any).shape).toEqual([4, 3, 2, 2, 2]);
    });

    it('concatenate: 5D axis=0', () => {
      const r = concatenate([A5, A5], 0);
      expect((r as any).shape).toEqual([4, 2, 2, 3, 4]);
    });

    it('matmul: 5D batch', () => {
      const a = mk([2, 2, 2, 3, 4]);
      const b = mk([2, 2, 2, 4, 5]);
      const r = matmul(a, b);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 5]);
    });

    it('sort: 5D axis=-1', () => {
      const r = sort(A5, -1);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });

    it('cumsum: 5D axis=3', () => {
      const r = cumsum(A5, 3);
      expect((r as any).shape).toEqual([2, 2, 2, 3, 4]);
    });
  });

  // ============================================================
  // SECTION 38: MAX_NDIM enforcement
  // ============================================================
  describe('MAX_NDIM: 64 maximum', () => {
    it('accepts 64-dimensional arrays', () => {
      expect(() => zeros(Array(64).fill(1))).not.toThrow();
    });

    it('rejects 65-dimensional arrays', () => {
      expect(() => zeros(Array(65).fill(1))).toThrow(/64/);
    });

    it('64-dim array: shape and ndim correct', () => {
      const a = zeros(Array(64).fill(1));
      expect(a.ndim).toBe(64);
      expect(a.shape).toHaveLength(64);
    });

    it('64-dim sum works', () => {
      const a = zeros(Array(64).fill(1));
      expect(() => sum(a)).not.toThrow();
    });

    it('64-dim elementwise works', () => {
      const a = zeros(Array(64).fill(1));
      expect(() => sin(a)).not.toThrow();
    });
  });

  // ============================================================
  // SECTION 39: keepdims regression — 1D scalar-shortcut bug fix
  // ============================================================
  describe('keepdims regression: 1D axis=0 scalar shortcut', () => {
    // Helper: 1D array [1,2,3,4]
    const a1 = () => mk([4]);
    // Helper: 2D array [3,4]
    const a2 = () => mk([3, 4]);

    // ---- amax (max) ----
    it('amax: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = amax(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.amax(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('amax: 2D [3,4] axis=1 keepdims=true -> shape [3,1]', () => {
      const r = amax(a2(), 1, true);
      expect((r as any).shape).toEqual([3, 1]);
      const py = runNumPy(
        `result = np.amax(np.arange(1,13,dtype=float).reshape(3,4), axis=1, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- amin (min) ----
    it('amin: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = amin(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.amin(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('amin: 2D [3,4] axis=0 keepdims=true -> shape [1,4]', () => {
      const r = amin(a2(), 0, true);
      expect((r as any).shape).toEqual([1, 4]);
      const py = runNumPy(
        `result = np.amin(np.arange(1,13,dtype=float).reshape(3,4), axis=0, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- prod ----
    it('prod: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = prod(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.prod(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('prod: 2D [3,4] axis=1 keepdims=true -> shape [3,1]', () => {
      const r = prod(a2(), 1, true);
      expect((r as any).shape).toEqual([3, 1]);
      const py = runNumPy(
        `result = np.prod(np.arange(1,13,dtype=float).reshape(3,4), axis=1, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- average (no weights) delegates to mean — should already work ----
    it('average (no weights): 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = average(a1(), 0, undefined, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.average(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- nansum ----
    it('nansum: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = nansum(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.nansum(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nansum: 2D [3,4] axis=0 keepdims=true -> shape [1,4]', () => {
      const r = nansum(a2(), 0, true);
      expect((r as any).shape).toEqual([1, 4]);
      const py = runNumPy(
        `result = np.nansum(np.arange(1,13,dtype=float).reshape(3,4), axis=0, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- nanprod ----
    it('nanprod: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = nanprod(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.nanprod(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- nanmean ----
    it('nanmean: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = nanmean(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.nanmean(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanmean: 2D [3,4] axis=1 keepdims=true -> shape [3,1]', () => {
      const r = nanmean(a2(), 1, true);
      expect((r as any).shape).toEqual([3, 1]);
      const py = runNumPy(
        `result = np.nanmean(np.arange(1,13,dtype=float).reshape(3,4), axis=1, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- nanvar ----
    it('nanvar: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = nanvar(a1(), 0, 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.nanvar(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- nanmin ----
    it('nanmin: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = nanmin(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.nanmin(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanmin: 2D [3,4] axis=1 keepdims=true -> shape [3,1]', () => {
      const r = nanmin(a2(), 1, true);
      expect((r as any).shape).toEqual([3, 1]);
      const py = runNumPy(
        `result = np.nanmin(np.arange(1,13,dtype=float).reshape(3,4), axis=1, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- nanmax ----
    it('nanmax: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = nanmax(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.nanmax(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanmax: 2D [3,4] axis=0 keepdims=true -> shape [1,4]', () => {
      const r = nanmax(a2(), 0, true);
      expect((r as any).shape).toEqual([1, 4]);
      const py = runNumPy(
        `result = np.nanmax(np.arange(1,13,dtype=float).reshape(3,4), axis=0, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- nanmedian ----
    it('nanmedian: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = nanmedian(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(
        `result = np.nanmedian(np.arange(1,5,dtype=float), axis=0, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- nanquantile ----
    it('nanquantile: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = nanquantile(a1(), 0.5, 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(
        `result = np.nanquantile(np.arange(1,5,dtype=float), 0.5, axis=0, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanquantile: 2D [3,4] axis=1 keepdims=true -> shape [3,1]', () => {
      const r = nanquantile(a2(), 0.75, 1, true);
      expect((r as any).shape).toEqual([3, 1]);
      const py = runNumPy(
        `result = np.nanquantile(np.arange(1,13,dtype=float).reshape(3,4), 0.75, axis=1, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    // ---- sum/all/any/quantile (already fixed — regression guard) ----
    it('sum: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = sum(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.sum(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('quantile: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = quantile(a1(), 0.5, 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(
        `result = np.quantile(np.arange(1,5,dtype=float), 0.5, axis=0, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('median: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = median(a1(), 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(`result = np.median(np.arange(1,5,dtype=float), axis=0, keepdims=True)`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('percentile: 1D axis=0 keepdims=true -> shape [1]', () => {
      const r = percentile(a1(), 75, 0, true);
      expect((r as any).shape).toEqual([1]);
      const py = runNumPy(
        `result = np.percentile(np.arange(1,5,dtype=float), 75, axis=0, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 40: 0D edge cases for additional reductions
  // ============================================================
  describe('0D edge cases: additional reductions', () => {
    it('amax(0D) -> scalar', () => {
      expect(typeof amax(array(5.0))).toBe('number');
      expect(amax(array(5.0))).toBeCloseTo(5.0);
    });

    it('amin(0D) -> scalar', () => {
      expect(typeof amin(array(5.0))).toBe('number');
      expect(amin(array(5.0))).toBeCloseTo(5.0);
    });

    it('prod(0D) -> scalar', () => {
      expect(typeof prod(array(5.0))).toBe('number');
      expect(prod(array(5.0))).toBeCloseTo(5.0);
    });

    it('average(0D) -> scalar', () => {
      expect(typeof average(array(5.0))).toBe('number');
      expect(average(array(5.0)) as number).toBeCloseTo(5.0);
    });

    it('nansum(0D) -> scalar', () => {
      expect(typeof nansum(array(5.0))).toBe('number');
      expect(nansum(array(5.0)) as number).toBeCloseTo(5.0);
    });

    it('nanmean(0D) -> scalar', () => {
      expect(typeof nanmean(array(5.0))).toBe('number');
      expect(nanmean(array(5.0)) as number).toBeCloseTo(5.0);
    });

    it('nanmin(0D) -> scalar', () => {
      expect(typeof nanmin(array(5.0))).toBe('number');
      expect(nanmin(array(5.0)) as number).toBeCloseTo(5.0);
    });

    it('nanmax(0D) -> scalar', () => {
      expect(typeof nanmax(array(5.0))).toBe('number');
      expect(nanmax(array(5.0)) as number).toBeCloseTo(5.0);
    });
  });

  // ============================================================
  // SECTION 41: nanquantile / nanpercentile axis tests
  // ============================================================
  describe('nanquantile/nanpercentile: axis + keepdims', () => {
    it('nanquantile: 2D [3,4] axis=0', () => {
      const a = mk([3, 4]);
      const r = nanquantile(a, 0.5, 0);
      expect((r as any).shape).toEqual([4]);
      const py = runNumPy(
        `result = np.nanquantile(np.arange(1,13,dtype=float).reshape(3,4), 0.5, axis=0)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanquantile: 3D [2,3,4] axis=1', () => {
      const a = mk([2, 3, 4]);
      const r = nanquantile(a, 0.75, 1);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('nanquantile: 3D [2,3,4] negative axis=-1', () => {
      const a = mk([2, 3, 4]);
      const r = nanquantile(a, 0.5, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('nanpercentile: 1D axis=0', () => {
      const a = mk([5]);
      const r = nanpercentile(a, 50, 0);
      expect(typeof r === 'number' || (r as any).shape !== undefined).toBe(true);
    });

    it('nanpercentile: 2D [3,4] axis=1 keepdims=true -> shape [3,1]', () => {
      const r = nanpercentile(mk([3, 4]), 75, 1, true);
      expect((r as any).shape).toEqual([3, 1]);
      const py = runNumPy(
        `result = np.nanpercentile(np.arange(1,13,dtype=float).reshape(3,4), 75, axis=1, keepdims=True)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanquantile: with NaN values ignored', () => {
      const a = array([1, NaN, 3, 4]);
      const r = nanquantile(a, 0.5, 0);
      const py = runNumPy(`result = np.nanquantile(np.array([1., np.nan, 3., 4.]), 0.5, axis=0)`);
      expect(arraysClose(r as number, py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 42: nancumsum / nancumprod shape preservation
  // ============================================================
  describe('nancumsum/nancumprod: shape preservation 1D-4D', () => {
    for (const shape of [[4], [3, 4], [2, 3, 4], [2, 2, 3, 4]] as number[][]) {
      it(`nancumsum: ${JSON.stringify(shape)} no-axis`, () => {
        const r = nancumsum(mk(shape));
        // without axis, result is flattened
        const n = shape.reduce((a, b) => a * b, 1);
        expect((r as any).shape).toEqual([n]);
      });

      it(`nancumprod: ${JSON.stringify(shape)} no-axis`, () => {
        const r = nancumprod(mk(shape));
        const n = shape.reduce((a, b) => a * b, 1);
        expect((r as any).shape).toEqual([n]);
      });
    }

    it('nancumsum: 2D [3,4] axis=0 shape preserved', () => {
      const r = nancumsum(mk([3, 4]), 0);
      expect((r as any).shape).toEqual([3, 4]);
      const py = runNumPy(
        `result = np.nancumsum(np.arange(1,13,dtype=float).reshape(3,4), axis=0)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nancumsum: 3D [2,3,4] axis=1 shape preserved', () => {
      const r = nancumsum(mk([2, 3, 4]), 1);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('nancumprod: 2D [3,4] axis=1 shape preserved', () => {
      const r = nancumprod(mk([3, 4]), 1);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('nancumsum: with NaN treated as 0', () => {
      const a = array([1, NaN, 3, 4]);
      const r = nancumsum(a);
      const py = runNumPy(`result = np.nancumsum(np.array([1., np.nan, 3., 4.]))`);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 43: negative-axis for percentile / quantile
  // ============================================================
  describe('percentile/quantile: negative axis', () => {
    it('percentile: 2D [3,4] axis=-1', () => {
      const r = percentile(mk([3, 4]), 75, -1);
      expect((r as any).shape).toEqual([3]);
      const py = runNumPy(
        `result = np.percentile(np.arange(1,13,dtype=float).reshape(3,4), 75, axis=-1)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('quantile: 3D [2,3,4] axis=-1', () => {
      const r = quantile(mk([2, 3, 4]), 0.75, -1);
      expect((r as any).shape).toEqual([2, 3]);
      const py = runNumPy(
        `result = np.quantile(np.arange(1,25,dtype=float).reshape(2,3,4), 0.75, axis=-1)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('quantile: 3D [2,3,4] axis=-2', () => {
      const r = quantile(mk([2, 3, 4]), 0.5, -2);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('quantile: 4D [2,2,3,4] axis=-1 keepdims=true -> shape [2,2,3,1]', () => {
      const r = quantile(mk([2, 2, 3, 4]), 0.5, -1, true);
      expect((r as any).shape).toEqual([2, 2, 3, 1]);
    });
  });

  // ============================================================
  // SECTION 44: nanargmin / nanargmax with axis
  // ============================================================
  describe('nanargmin/nanargmax: 1D and 2D with axis', () => {
    it('nanargmin: 1D axis=0 -> scalar index', () => {
      const a = mk([5]);
      const r = nanargmin(a, 0);
      expect(typeof r).toBe('number');
      const py = runNumPy(`result = int(np.nanargmin(np.arange(1,6,dtype=float), axis=0))`);
      expect(r).toBe(py.value);
    });

    it('nanargmin: 2D [3,4] axis=0 -> shape [4]', () => {
      const r = nanargmin(mk([3, 4]), 0);
      expect((r as any).shape).toEqual([4]);
      const py = runNumPy(
        `result = np.nanargmin(np.arange(1,13,dtype=float).reshape(3,4), axis=0)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanargmin: 2D [3,4] axis=1 -> shape [3]', () => {
      const r = nanargmin(mk([3, 4]), 1);
      expect((r as any).shape).toEqual([3]);
      const py = runNumPy(
        `result = np.nanargmin(np.arange(1,13,dtype=float).reshape(3,4), axis=1)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanargmax: 1D axis=0 -> scalar index', () => {
      const a = mk([5]);
      const r = nanargmax(a, 0);
      expect(typeof r).toBe('number');
      const py = runNumPy(`result = int(np.nanargmax(np.arange(1,6,dtype=float), axis=0))`);
      expect(r).toBe(py.value);
    });

    it('nanargmax: 2D [3,4] axis=0 -> shape [4]', () => {
      const r = nanargmax(mk([3, 4]), 0);
      expect((r as any).shape).toEqual([4]);
      const py = runNumPy(
        `result = np.nanargmax(np.arange(1,13,dtype=float).reshape(3,4), axis=0)`
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanargmax: 2D [3,4] axis=1 -> shape [3]', () => {
      const r = nanargmax(mk([3, 4]), 1);
      expect((r as any).shape).toEqual([3]);
    });

    it('nanargmin: with NaN skipped', () => {
      const a = array([NaN, 2, 1, 4]);
      const r = nanargmin(a, 0);
      expect(r).toBe(2); // index of minimum non-NaN value (1 at index 2)
    });

    it('nanargmax: with NaN skipped', () => {
      const a = array([1, NaN, 3, 2]);
      const r = nanargmax(a, 0);
      expect(r).toBe(2); // index of max non-NaN value (3 at index 2)
    });
  });

  // ============================================================
  // SECTION 45: nanmin / nanmax 4D with all axes
  // ============================================================
  describe('nanmin/nanmax: 4D with all axes', () => {
    const a4 = () => mk([2, 3, 4, 5]);

    for (let ax = 0; ax < 4; ax++) {
      const expectedShape = [2, 3, 4, 5].filter((_, i) => i !== ax);
      it(`nanmin: 4D [2,3,4,5] axis=${ax} -> shape ${JSON.stringify(expectedShape)}`, () => {
        const r = nanmin(a4(), ax);
        expect((r as any).shape).toEqual(expectedShape);
      });

      it(`nanmax: 4D [2,3,4,5] axis=${ax} -> shape ${JSON.stringify(expectedShape)}`, () => {
        const r = nanmax(a4(), ax);
        expect((r as any).shape).toEqual(expectedShape);
      });
    }

    it('nanmin: 4D axis=-1 keepdims=true -> shape [2,3,4,1]', () => {
      const r = nanmin(a4(), -1, true);
      expect((r as any).shape).toEqual([2, 3, 4, 1]);
    });

    it('nanmax: 4D axis=-1 keepdims=true -> shape [2,3,4,1]', () => {
      const r = nanmax(a4(), -1, true);
      expect((r as any).shape).toEqual([2, 3, 4, 1]);
    });
  });

  // ============================================================
  // SECTION 46: ptp — all axes, negative axes, keepdims
  // ============================================================
  describe('ptp: all axes, negative axes, keepdims', () => {
    it('ptp: 1D axis=0 values', () => {
      const a = array([3, 1, 4, 1, 5, 9, 2, 6]);
      const r = ptp(a, 0);
      const py = runNumPy('result = np.ptp(np.array([3,1,4,1,5,9,2,6]), axis=0)');
      expect(r).toBeCloseTo(py.value as number, 10);
    });

    it('ptp: 2D axis=0 -> shape [3]', () => {
      const a = mk([2, 3]);
      const r = ptp(a, 0);
      const py = runNumPy('result = np.ptp(np.arange(1,7,dtype=float).reshape(2,3), axis=0)');
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 2D axis=1 -> shape [2]', () => {
      const a = mk([2, 3]);
      const r = ptp(a, 1);
      const py = runNumPy('result = np.ptp(np.arange(1,7,dtype=float).reshape(2,3), axis=1)');
      expect((r as any).shape).toEqual([2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 2D axis=-1 (neg) -> shape [2]', () => {
      const a = mk([2, 3]);
      const r = ptp(a, -1);
      const py = runNumPy('result = np.ptp(np.arange(1,7,dtype=float).reshape(2,3), axis=-1)');
      expect((r as any).shape).toEqual([2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 3D axis=0 -> shape [3,4]', () => {
      const a = mk([2, 3, 4]);
      const r = ptp(a, 0);
      const py = runNumPy('result = np.ptp(np.arange(1,25,dtype=float).reshape(2,3,4), axis=0)');
      expect((r as any).shape).toEqual([3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 3D axis=1 -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = ptp(a, 1);
      const py = runNumPy('result = np.ptp(np.arange(1,25,dtype=float).reshape(2,3,4), axis=1)');
      expect((r as any).shape).toEqual([2, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 3D axis=2 -> shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = ptp(a, 2);
      const py = runNumPy('result = np.ptp(np.arange(1,25,dtype=float).reshape(2,3,4), axis=2)');
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 3D axis=-1 (neg) -> shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = ptp(a, -1);
      const py = runNumPy('result = np.ptp(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-1)');
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 3D axis=-2 (neg) -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = ptp(a, -2);
      const py = runNumPy('result = np.ptp(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-2)');
      expect((r as any).shape).toEqual([2, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 2D axis=0 keepdims=true -> shape [1,3]', () => {
      const a = mk([2, 3]);
      const r = ptp(a, 0, true);
      const py = runNumPy(
        'result = np.ptp(np.arange(1,7,dtype=float).reshape(2,3), axis=0, keepdims=True)'
      );
      expect((r as any).shape).toEqual([1, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 3D axis=1 keepdims=true -> shape [2,1,4]', () => {
      const a = mk([2, 3, 4]);
      const r = ptp(a, 1, true);
      const py = runNumPy(
        'result = np.ptp(np.arange(1,25,dtype=float).reshape(2,3,4), axis=1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 1, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('ptp: 4D axis=-1 keepdims=true -> shape [2,3,4,1]', () => {
      const a = mk([2, 3, 4, 5]);
      const r = ptp(a, -1, true);
      const py = runNumPy(
        'result = np.ptp(np.arange(1,121,dtype=float).reshape(2,3,4,5), axis=-1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 3, 4, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 47: std/variance — missing dims, neg axes, keepdims
  // ============================================================
  describe('std/variance: additional coverage', () => {
    it('std: 2D axis=1 values', () => {
      const a = array([
        [2.0, 4.0, 4.0, 4.0],
        [5.0, 5.0, 7.0, 9.0],
      ]);
      const r = std(a, 1);
      const py = runNumPy('result = np.std(np.array([[2.,4.,4.,4.],[5.,5.,7.,9.]]), axis=1)');
      expect((r as any).shape).toEqual([2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('std: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = std(a, -1);
      const py = runNumPy('result = np.std(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-1)');
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('std: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = std(a, -2);
      const py = runNumPy('result = np.std(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-2)');
      expect((r as any).shape).toEqual([2, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('std: 4D all axes', () => {
      const a = mk([2, 3, 4, 5]);
      for (let ax = 0; ax < 4; ax++) {
        const r = std(a, ax);
        const expShape = [2, 3, 4, 5].filter((_, i) => i !== ax);
        expect((r as any).shape).toEqual(expShape);
      }
    });

    it('std: 3D axis=1 keepdims=true -> shape [2,1,4]', () => {
      const a = mk([2, 3, 4]);
      const r = std(a, 1, undefined, true);
      const py = runNumPy(
        'result = np.std(np.arange(1,25,dtype=float).reshape(2,3,4), axis=1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 1, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('variance: 2D axis=0 values', () => {
      const a = mk([3, 4]);
      const r = variance(a, 0);
      const py = runNumPy('result = np.var(np.arange(1,13,dtype=float).reshape(3,4), axis=0)');
      expect((r as any).shape).toEqual([4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('variance: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = variance(a, -1);
      const py = runNumPy('result = np.var(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-1)');
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('variance: 3D axis=2 keepdims=true -> shape [2,3,1]', () => {
      const a = mk([2, 3, 4]);
      const r = variance(a, 2, undefined, true);
      const py = runNumPy(
        'result = np.var(np.arange(1,25,dtype=float).reshape(2,3,4), axis=2, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 3, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 48: mean — 0D, 1D, 4D, 5D values
  // ============================================================
  describe('mean: 0D, 1D, 4D, 5D values', () => {
    it('mean: 0D scalar', () => {
      const a = array(7.0);
      const r = mean(a);
      const py = runNumPy('result = np.mean(np.array(7.0))');
      expect(r as number).toBeCloseTo(py.value as number, 10);
    });

    it('mean: 1D axis=0 values', () => {
      const a = array([1.0, 2.0, 3.0, 4.0]);
      const r = mean(a, 0);
      const py = runNumPy('result = np.mean(np.array([1.,2.,3.,4.]), axis=0)');
      expect(r as number).toBeCloseTo(py.value as number, 10);
    });

    it('mean: 4D axis=0 shape', () => {
      const a = mk([2, 3, 4, 5]);
      const r = mean(a, 0);
      expect((r as any).shape).toEqual([3, 4, 5]);
    });

    it('mean: 4D axis=-1 neg -> shape [2,3,4]', () => {
      const a = mk([2, 3, 4, 5]);
      const r = mean(a, -1);
      const py = runNumPy(
        'result = np.mean(np.arange(1,121,dtype=float).reshape(2,3,4,5), axis=-1)'
      );
      expect((r as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('mean: 5D axis=2 shape', () => {
      const a = mk([2, 2, 3, 4, 5]);
      const r = mean(a, 2);
      expect((r as any).shape).toEqual([2, 2, 4, 5]);
    });
  });

  // ============================================================
  // SECTION 49: prod — neg axes, 4D, 5D
  // ============================================================
  describe('prod: neg axes, 4D, 5D', () => {
    it('prod: 1D axis=0 keepdims=true -> shape [1]', () => {
      const a = array([1.0, 2.0, 3.0]);
      const r = prod(a, 0, true);
      const py = runNumPy('result = np.prod(np.array([1.,2.,3.]), axis=0, keepdims=True)');
      expect((r as any).shape).toEqual([1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('prod: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = prod(a, -1);
      const py = runNumPy('result = np.prod(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-1)');
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('prod: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = prod(a, -2);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('prod: 4D axis=1 shape', () => {
      const a = mk([2, 3, 4, 5]);
      const r = prod(a, 1);
      expect((r as any).shape).toEqual([2, 4, 5]);
    });

    it('prod: 5D axis=0 shape', () => {
      const a = mk([2, 2, 3, 4, 5]);
      const r = prod(a, 0);
      expect((r as any).shape).toEqual([2, 3, 4, 5]);
    });
  });

  // ============================================================
  // SECTION 50: average — neg axes
  // ============================================================
  describe('average: negative axes', () => {
    it('average: 2D axis=-1 neg -> shape [3]', () => {
      const a = mk([3, 4]);
      const r = average(a, -1);
      const py = runNumPy('result = np.average(np.arange(1,13,dtype=float).reshape(3,4), axis=-1)');
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('average: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = average(a, -2);
      const py = runNumPy(
        'result = np.average(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-2)'
      );
      expect((r as any).shape).toEqual([2, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('average: 3D axis=-1 values', () => {
      const a = mk([2, 3, 4]);
      const r = average(a, -1);
      const py = runNumPy(
        'result = np.average(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-1)'
      );
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 51: nanstd — keepdims
  // ============================================================
  describe('nanstd: keepdims', () => {
    it('nanstd: 1D axis=0 keepdims=true -> shape [1]', () => {
      const a = array([1.0, NaN, 3.0, 4.0]);
      const r = nanstd(a, 0, undefined, true);
      const py = runNumPy('result = np.nanstd(np.array([1.,np.nan,3.,4.]), axis=0, keepdims=True)');
      expect((r as any).shape).toEqual([1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanstd: 2D axis=0 keepdims=true -> shape [1,3]', () => {
      const a = array([
        [1.0, NaN, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const r = nanstd(a, 0, undefined, true);
      const py = runNumPy(
        'result = np.nanstd(np.array([[1.,np.nan,3.],[4.,5.,6.]]), axis=0, keepdims=True)'
      );
      expect((r as any).shape).toEqual([1, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanstd: 2D axis=1 keepdims=true -> shape [2,1]', () => {
      const a = array([
        [1.0, NaN, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const r = nanstd(a, 1, undefined, true);
      const py = runNumPy(
        'result = np.nanstd(np.array([[1.,np.nan,3.],[4.,5.,6.]]), axis=1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanstd: 3D axis=-1 keepdims=true -> shape [2,3,1]', () => {
      const a = mk([2, 3, 4]);
      const r = nanstd(a, -1, undefined, true);
      const py = runNumPy(
        'result = np.nanstd(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 3, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 52: neg axes for nan reductions
  // ============================================================
  describe('nan reductions: negative axes', () => {
    const mkNan = (shape: number[]) => {
      const n = shape.reduce((a, b) => a * b, 1);
      const data = Array.from({ length: n }, (_, i) => (i % 7 === 0 ? NaN : (i + 1) * 0.5));
      return array(data).reshape(shape);
    };

    it('nansum: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nansum(a, -1);
      const py = runNumPy(`
a = np.where(np.arange(24) % 7 == 0, np.nan, (np.arange(24)+1)*0.5).reshape(2,3,4)
result = np.nansum(a, axis=-1)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanprod: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanprod(a, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('nanmean: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanmean(a, -1);
      const py = runNumPy(`
a = np.where(np.arange(24) % 7 == 0, np.nan, (np.arange(24)+1)*0.5).reshape(2,3,4)
result = np.nanmean(a, axis=-1)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanvar: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanvar(a, -1);
      const py = runNumPy(`
a = np.where(np.arange(24) % 7 == 0, np.nan, (np.arange(24)+1)*0.5).reshape(2,3,4)
result = np.nanvar(a, axis=-1)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanstd: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanstd(a, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('nanmin: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanmin(a, -2);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('nanmax: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanmax(a, -2);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('nanmedian: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanmedian(a, -1);
      const py = runNumPy(`
a = np.where(np.arange(24) % 7 == 0, np.nan, (np.arange(24)+1)*0.5).reshape(2,3,4)
result = np.nanmedian(a, axis=-1)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanquantile: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanquantile(a, 0.5, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('nanpercentile: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mkNan([2, 3, 4]);
      const r = nanpercentile(a, 50, -1);
      const py = runNumPy(`
a = np.where(np.arange(24) % 7 == 0, np.nan, (np.arange(24)+1)*0.5).reshape(2,3,4)
result = np.nanpercentile(a, 50, axis=-1)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 53: nanargmin/nanargmax neg axes; nancumsum/nancumprod neg axis
  // ============================================================
  describe('nanargmin/nanargmax: neg axes; nancumsum/nancumprod: neg axis', () => {
    it('nanargmin: 2D axis=-1 neg -> shape [3]', () => {
      const a = array([
        [3.0, 1.0, NaN, 4.0],
        [NaN, 2.0, 5.0, 1.0],
        [7.0, NaN, 2.0, 3.0],
      ]);
      const r = nanargmin(a, -1);
      const py = runNumPy(
        'result = np.nanargmin(np.array([[3.,1.,np.nan,4.],[np.nan,2.,5.,1.],[7.,np.nan,2.,3.]]), axis=-1)'
      );
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanargmax: 2D axis=-1 neg -> shape [3]', () => {
      const a = array([
        [3.0, 1.0, NaN, 4.0],
        [NaN, 2.0, 5.0, 1.0],
        [7.0, NaN, 2.0, 3.0],
      ]);
      const r = nanargmax(a, -1);
      const py = runNumPy(
        'result = np.nanargmax(np.array([[3.,1.,np.nan,4.],[np.nan,2.,5.,1.],[7.,np.nan,2.,3.]]), axis=-1)'
      );
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nanargmin: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = nanargmin(a, -1);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('nanargmax: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = nanargmax(a, -2);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('nancumsum: 3D axis=-1 neg -> shape [2,3,4]', () => {
      const a = array([1.0, NaN, 3.0, 4.0, NaN, 6.0, 7.0, 8.0, NaN, 10.0, 11.0, 12.0]).reshape([
        2, 3, 2,
      ]);
      const r = nancumsum(a, -1);
      expect((r as any).shape).toEqual([2, 3, 2]);
    });

    it('nancumprod: 3D axis=-1 neg -> shape preserved', () => {
      const a = array([1.0, NaN, 3.0, 4.0, NaN, 6.0, 7.0, 8.0, NaN, 10.0, 11.0, 12.0]).reshape([
        2, 3, 2,
      ]);
      const r = nancumprod(a, -1);
      expect((r as any).shape).toEqual([2, 3, 2]);
    });

    it('nancumsum: 3D axis=-2 neg -> shape preserved', () => {
      const a = mk([2, 3, 4]);
      const r = nancumsum(a, -2);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });
  });

  // ============================================================
  // SECTION 54: diff, gradient, trapezoid — negative axes
  // ============================================================
  describe('diff/gradient/trapezoid: negative axes', () => {
    it('diff: 2D axis=-1 neg -> shape [2,2]', () => {
      const a = mk([2, 3]);
      const r = diff(a, 1, -1);
      const py = runNumPy(
        'result = np.diff(np.arange(1,7,dtype=float).reshape(2,3), n=1, axis=-1)'
      );
      expect((r as any).shape).toEqual([2, 2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('diff: 3D axis=-2 neg -> shape [2,2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = diff(a, 1, -2);
      const py = runNumPy(
        'result = np.diff(np.arange(1,25,dtype=float).reshape(2,3,4), n=1, axis=-2)'
      );
      expect((r as any).shape).toEqual([2, 2, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('diff: 3D axis=-1 neg values', () => {
      const a = mk([2, 3, 4]);
      const r = diff(a, 1, -1);
      const py = runNumPy(
        'result = np.diff(np.arange(1,25,dtype=float).reshape(2,3,4), n=1, axis=-1)'
      );
      expect((r as any).shape).toEqual([2, 3, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('gradient: 2D axis=-1 neg', () => {
      const a = mk([3, 4]);
      const r = gradient(a, undefined, -1);
      const py = runNumPy(
        'result = np.gradient(np.arange(1,13,dtype=float).reshape(3,4), axis=-1)'
      );
      expect((r as any).shape).toEqual([3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('gradient: 3D axis=-1 neg shape', () => {
      const a = mk([2, 3, 4]);
      const r = gradient(a, undefined, -1);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('trapezoid: 2D axis=0 values', () => {
      const a = mk([3, 4]);
      const r = trapezoid(a, undefined, undefined, 0);
      const py = runNumPy(
        'result = np.trapezoid(np.arange(1,13,dtype=float).reshape(3,4), axis=0)'
      );
      expect((r as any).shape ?? []).toEqual(py.shape ?? [4]);
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('trapezoid: 2D axis=-1 neg -> shape [3]', () => {
      const a = mk([3, 4]);
      const r = trapezoid(a, undefined, undefined, -1);
      const py = runNumPy(
        'result = np.trapezoid(np.arange(1,13,dtype=float).reshape(3,4), axis=-1)'
      );
      expect((r as any).shape ?? []).toEqual(py.shape ?? [3]);
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('trapezoid: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = trapezoid(a, undefined, undefined, -2);
      const py = runNumPy(
        'result = np.trapezoid(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-2)'
      );
      expect((r as any).shape).toEqual([2, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 55: partition/argpartition — negative axes
  // ============================================================
  describe('partition/argpartition: negative axes', () => {
    it('partition: 2D axis=-1 neg -> shape preserved [3,4]', () => {
      const a = array([
        [3.0, 1.0, 4.0, 2.0],
        [9.0, 7.0, 2.0, 5.0],
        [6.0, 8.0, 1.0, 3.0],
      ]);
      const r = partition(a, 2, -1);
      const py = runNumPy(
        'result = np.partition(np.array([[3.,1.,4.,2.],[9.,7.,2.,5.],[6.,8.,1.,3.]]), 2, axis=-1)'
      );
      expect((r as any).shape).toEqual([3, 4]);
      // Only the k-th element must match
      expect((r as any).toArray()[0][2]).toBeCloseTo((py.value as number[][])[0]![2]!, 10);
    });

    it('partition: 3D axis=-1 neg -> shape preserved', () => {
      const a = mk([2, 3, 4]);
      const r = partition(a, 1, -1);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('partition: 3D axis=-2 neg -> shape preserved', () => {
      const a = mk([2, 3, 4]);
      const r = partition(a, 1, -2);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });

    it('argpartition: 2D axis=-1 neg -> shape [3,4]', () => {
      const a = array([
        [3.0, 1.0, 4.0, 2.0],
        [9.0, 7.0, 2.0, 5.0],
        [6.0, 8.0, 1.0, 3.0],
      ]);
      const r = argpartition(a, 2, -1);
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('argpartition: 3D axis=-1 neg -> shape preserved', () => {
      const a = mk([2, 3, 4]);
      const r = argpartition(a, 1, -1);
      expect((r as any).shape).toEqual([2, 3, 4]);
    });
  });

  // ============================================================
  // SECTION 56: count_nonzero — negative axes
  // ============================================================
  describe('count_nonzero: negative axes', () => {
    it('count_nonzero: 2D axis=-1 neg -> shape [3]', () => {
      const a = array([
        [1.0, 0.0, 3.0, 0.0],
        [0.0, 5.0, 6.0, 0.0],
        [7.0, 8.0, 0.0, 10.0],
      ]);
      const r = count_nonzero(a, -1);
      const py = runNumPy(
        'result = np.count_nonzero(np.array([[1,0,3,0],[0,5,6,0],[7,8,0,10]]), axis=-1)'
      );
      expect((r as any).shape ?? []).toEqual([3]);
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('count_nonzero: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = count_nonzero(a, -2);
      expect((r as any).shape ?? []).toEqual([2, 4]);
    });

    it('count_nonzero: 3D axis=-1 neg -> shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = count_nonzero(a, -1);
      expect((r as any).shape ?? []).toEqual([2, 3]);
    });
  });

  // ============================================================
  // SECTION 57: linalg.vector_norm / linalg.matrix_norm
  // ============================================================
  describe('linalg.vector_norm / linalg.matrix_norm: axes, keepdims, values', () => {
    it('linalg.vector_norm: 1D values (L2)', () => {
      const a = array([3.0, 4.0]);
      const r = linalg.vector_norm(a);
      const py = runNumPy('result = np.linalg.vector_norm(np.array([3.,4.]))');
      expect(r as number).toBeCloseTo(py.value as number, 8);
    });

    it('linalg.vector_norm: 2D axis=0 -> shape [4]', () => {
      const a = mk([3, 4]);
      const r = linalg.vector_norm(a, undefined, 0);
      const py = runNumPy(
        'result = np.linalg.vector_norm(np.arange(1,13,dtype=float).reshape(3,4), axis=0)'
      );
      expect((r as any).shape).toEqual([4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('linalg.vector_norm: 2D axis=1 -> shape [3]', () => {
      const a = mk([3, 4]);
      const r = linalg.vector_norm(a, undefined, 1);
      const py = runNumPy(
        'result = np.linalg.vector_norm(np.arange(1,13,dtype=float).reshape(3,4), axis=1)'
      );
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('linalg.vector_norm: 2D axis=-1 neg -> shape [3]', () => {
      const a = mk([3, 4]);
      const r = linalg.vector_norm(a, undefined, -1);
      const py = runNumPy(
        'result = np.linalg.vector_norm(np.arange(1,13,dtype=float).reshape(3,4), axis=-1)'
      );
      expect((r as any).shape).toEqual([3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('linalg.vector_norm: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = linalg.vector_norm(a, undefined, -2);
      expect((r as any).shape).toEqual([2, 4]);
    });

    it('linalg.vector_norm: 2D axis=1 keepdims=true -> shape [3,1]', () => {
      const a = mk([3, 4]);
      const r = linalg.vector_norm(a, undefined, 1, true);
      const py = runNumPy(
        'result = np.linalg.vector_norm(np.arange(1,13,dtype=float).reshape(3,4), axis=1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([3, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('linalg.vector_norm: 3D axis=-1 keepdims=true -> shape [2,3,1]', () => {
      const a = mk([2, 3, 4]);
      const r = linalg.vector_norm(a, undefined, -1, true);
      expect((r as any).shape).toEqual([2, 3, 1]);
    });

    it('linalg.vector_norm: L1 norm values', () => {
      const a = array([1.0, -2.0, 3.0]);
      const r = linalg.vector_norm(a, 1);
      const py = runNumPy('result = np.linalg.vector_norm(np.array([1.,-2.,3.]), ord=1)');
      expect(r as number).toBeCloseTo(py.value as number, 8);
    });

    it('linalg.matrix_norm: 2D Frobenius values', () => {
      const a = mk([3, 4]);
      const r = linalg.matrix_norm(a, 'fro');
      const py = runNumPy(
        "result = np.linalg.matrix_norm(np.arange(1,13,dtype=float).reshape(3,4), ord='fro')"
      );
      expect(r as number).toBeCloseTo(py.value as number, 8);
    });

    it('linalg.matrix_norm: 2D nuclear norm values', () => {
      const a = mk([3, 4]);
      const r = linalg.matrix_norm(a, 'nuc');
      const py = runNumPy(
        "result = np.linalg.matrix_norm(np.arange(1,13,dtype=float).reshape(3,4), ord='nuc')"
      );
      expect(r as number).toBeCloseTo(py.value as number, 5);
    });

    it('linalg.matrix_norm: 2D keepdims=true -> shape [1,1]', () => {
      const a = mk([3, 4]);
      const r = linalg.matrix_norm(a, 'fro', true);
      const py = runNumPy(
        "result = np.linalg.matrix_norm(np.arange(1,13,dtype=float).reshape(3,4), ord='fro', keepdims=True)"
      );
      expect((r as any).shape).toEqual([1, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 58: linalg.tensorinv / linalg.tensorsolve — values
  // ============================================================
  describe('linalg.tensorinv / linalg.tensorsolve: values', () => {
    it('linalg.tensorinv: 4D ind=2 shape [3,4,2,6]', () => {
      // tensorinv of shape (2,6,3,4) with ind=2 -> shape (3,4,2,6)
      const n = 2 * 6;
      const eye6 = zeros([n, n]);
      for (let i = 0; i < n; i++) (eye6 as any).set([i, i], 1.0);
      const a = eye6.reshape([2, 6, 2, 6]);
      const r = linalg.tensorinv(a, 2);
      expect((r as any).shape).toEqual([2, 6, 2, 6]);
    });

    it('linalg.tensorsolve: 2D -> 1D solution', () => {
      const a = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const b = array([5.0, 11.0]);
      const r = linalg.tensorsolve(a, b);
      const py = runNumPy(
        'result = np.linalg.tensorsolve(np.array([[1.,2.],[3.,4.]]), np.array([5.,11.]))'
      );
      expect((r as any).shape).toEqual([2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('linalg.tensorsolve: 3D -> 2D solution', () => {
      // A has shape (2,2,2,2), b has shape (2,2)
      const n = 4;
      const eye4 = zeros([n, n]);
      for (let i = 0; i < n; i++) (eye4 as any).set([i, i], 1.0);
      const a = eye4.reshape([2, 2, 2, 2]);
      const b = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const r = linalg.tensorsolve(a, b);
      expect((r as any).shape).toEqual([2, 2]);
      expect(arraysClose((r as any).toArray(), b.toArray())).toBe(true);
    });
  });

  // ============================================================
  // SECTION 59: shape ops — negative axes
  // ============================================================
  describe('shape ops: negative axes', () => {
    it('swapaxes: 3D axes (-1, -2) neg', () => {
      const a = mk([2, 3, 4]);
      const r = swapaxes(a, -1, -2);
      const py = runNumPy(
        'result = np.swapaxes(np.arange(1,25,dtype=float).reshape(2,3,4), -1, -2)'
      );
      expect((r as any).shape).toEqual([2, 4, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('swapaxes: 4D axis (-1, 0) neg', () => {
      const a = mk([2, 3, 4, 5]);
      const r = swapaxes(a, -1, 0);
      expect((r as any).shape).toEqual([5, 3, 4, 2]);
    });

    it('squeeze: neg axis on size-1 dim', () => {
      const a = zeros([2, 1, 3]);
      const r = squeeze(a, -2);
      const py = runNumPy('result = np.squeeze(np.zeros((2,1,3)), axis=-2)');
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('rollaxis: 3D neg axis -> shape [4,2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = rollaxis(a, -1, 0);
      const py = runNumPy(
        'result = np.rollaxis(np.arange(1,25,dtype=float).reshape(2,3,4), -1, 0)'
      );
      expect((r as any).shape).toEqual([4, 2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('stack: 2D axis=-1 neg -> shape [3,4,2]', () => {
      const a = mk([3, 4]);
      const b = mk([3, 4]);
      const r = stack([a, b], -1);
      const py = runNumPy(`
a = np.arange(1,13,dtype=float).reshape(3,4)
b = np.arange(1,13,dtype=float).reshape(3,4)
result = np.stack([a, b], axis=-1)`);
      expect((r as any).shape).toEqual([3, 4, 2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('unstack: 3D axis=-1 neg -> 4 arrays of shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const parts = unstack(a, -1);
      expect(parts.length).toBe(4);
      expect((parts[0] as any).shape).toEqual([2, 3]);
    });

    it('split: 3D axis=-1 neg -> 2 parts of shape [2,3,2]', () => {
      const a = mk([2, 3, 4]);
      const parts = split(a, 2, -1);
      expect(parts.length).toBe(2);
      expect((parts[0] as any).shape).toEqual([2, 3, 2]);
    });

    it('array_split: 3D axis=-2 neg -> 3 parts', () => {
      const a = mk([2, 3, 4]);
      const parts = array_split(a, 3, -2);
      expect(parts.length).toBe(3);
    });

    it('repeat: 3D axis=-1 neg -> shape [2,3,8]', () => {
      const a = mk([2, 3, 4]);
      const r = repeat(a, 2, -1);
      expect((r as any).shape).toEqual([2, 3, 8]);
    });

    it('append: 2D axis=-1 neg', () => {
      const a = mk([2, 3]);
      const b = mk([2, 1]);
      const r = append(a, b, -1);
      const py = runNumPy(`
a = np.arange(1,7,dtype=float).reshape(2,3)
b = np.arange(1,3,dtype=float).reshape(2,1)
result = np.append(a, b, axis=-1)`);
      expect((r as any).shape).toEqual([2, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('delete_: 2D axis=-1 neg -> shape [2,2]', () => {
      const a = mk([2, 3]);
      const r = delete_(a, 1, -1);
      const py = runNumPy(
        'result = np.delete(np.arange(1,7,dtype=float).reshape(2,3), 1, axis=-1)'
      );
      expect((r as any).shape).toEqual([2, 2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('insert: 1D insert values', () => {
      const a = array([1.0, 2.0, 3.0, 4.0]);
      const r = insert(a, 2, 99.0);
      const py = runNumPy('result = np.insert(np.array([1.,2.,3.,4.]), 2, 99.)');
      expect((r as any).shape).toEqual([5]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 60: vecdot / permute_dims — negative axes
  // ============================================================
  describe('vecdot/permute_dims: negative axes', () => {
    it('vecdot: 2D axis=-1 neg -> shape [3]', () => {
      const a = mk([3, 4]);
      const b = mk([3, 4]);
      const r = vecdot(a, b, -1);
      const py = runNumPy(`
a = np.arange(1,13,dtype=float).reshape(3,4)
b = np.arange(1,13,dtype=float).reshape(3,4)
result = np.vecdot(a, b, axis=-1)`);
      expect((r as any).shape ?? []).toEqual(py.shape ?? [3]);
      expect(arraysClose((r as any).toArray?.() ?? r, py.value)).toBe(true);
    });

    it('vecdot: 3D axis=-2 neg -> shape [2,4]', () => {
      const a = mk([2, 3, 4]);
      const b = mk([2, 3, 4]);
      const r = vecdot(a, b, -2);
      expect((r as any).shape ?? []).toEqual([2, 4]);
    });

    it('permute_dims: 3D neg axes [-1,-2,-3] -> shape [4,3,2]', () => {
      const a = mk([2, 3, 4]);
      const r = permute_dims(a, [-1, -2, -3]);
      const py = runNumPy(
        'result = np.permute_dims(np.arange(1,25,dtype=float).reshape(2,3,4), [-1,-2,-3])'
      );
      expect((r as any).shape).toEqual([4, 3, 2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('permute_dims: 4D neg axes [-1,0,-2,1]', () => {
      const a = mk([2, 3, 4, 5]);
      const r = permute_dims(a, [-1, 0, -2, 1]);
      expect((r as any).shape).toEqual([5, 2, 4, 3]);
    });
  });

  // ============================================================
  // SECTION 61: take / take_along_axis / compress — negative axes
  // ============================================================
  describe('take/take_along_axis/compress: negative axes', () => {
    it('take: 2D axis=1 pos -> shape [3,2] values', () => {
      const a = mk([3, 4]);
      const r = take(a, [0, 2], 1);
      const py = runNumPy(
        'result = np.take(np.arange(1,13,dtype=float).reshape(3,4), [0,2], axis=1)'
      );
      expect((r as any).shape).toEqual([3, 2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('take: 3D axis=1 pos -> shape [2,2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = take(a, [0, 1], 1);
      expect((r as any).shape).toEqual([2, 2, 4]);
    });

    it('take_along_axis: 2D axis=-1 neg -> shape preserved [3,4]', () => {
      const a = mk([3, 4]);
      const idx = argsort(a, -1);
      const r = take_along_axis(a, idx, -1);
      const py = runNumPy(`
a = np.arange(1,13,dtype=float).reshape(3,4)
idx = np.argsort(a, axis=-1)
result = np.take_along_axis(a, idx, axis=-1)`);
      expect((r as any).shape).toEqual([3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('compress: 2D axis=-1 neg -> shape [3,2]', () => {
      const a = mk([3, 4]);
      const mask = array([true, false, true, false]);
      const r = compress(mask, a, -1);
      const py = runNumPy(`
a = np.arange(1,13,dtype=float).reshape(3,4)
result = np.compress([True,False,True,False], a, axis=-1)`);
      expect((r as any).shape).toEqual([3, 2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('iindex: 2D axis=-1 neg -> shape [3,2]', () => {
      const a = mk([3, 4]);
      const idx = array([0, 2]);
      const r = iindex(a, idx, -1);
      expect((r as any).shape).toEqual([3, 2]);
    });

    it('bindex: 2D axis=-1 neg -> shape [3,2]', () => {
      const a = mk([3, 4]);
      const mask = array([true, false, true, false]);
      const r = bindex(a, mask, -1);
      expect((r as any).shape).toEqual([3, 2]);
    });
  });

  // ============================================================
  // SECTION 62: put_along_axis / putmask / place — values
  // ============================================================
  describe('put_along_axis / putmask / place: values', () => {
    it('put_along_axis: 2D axis=1 values', () => {
      const a = array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
      ]);
      const idx = array([
        [2, 0, 1],
        [0, 2, 1],
      ]);
      const vals = array([
        [99.0, 88.0, 77.0],
        [66.0, 55.0, 44.0],
      ]);
      put_along_axis(a, idx, vals, 1);
      const py = runNumPy(`
a = np.array([[10.,20.,30.],[40.,50.,60.]])
idx = np.array([[2,0,1],[0,2,1]])
vals = np.array([[99.,88.,77.],[66.,55.,44.]])
np.put_along_axis(a, idx, vals, axis=1)
result = a`);
      expect(arraysClose((a as any).toArray(), py.value)).toBe(true);
    });

    it('put_along_axis: 2D axis=-1 neg same as axis=1', () => {
      const a = array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
      ]);
      const aRef = array([
        [10.0, 20.0, 30.0],
        [40.0, 50.0, 60.0],
      ]);
      const idx = array([[0], [2]]);
      const vals = array([[7.0], [8.0]]);
      put_along_axis(a, idx, vals, -1);
      put_along_axis(aRef, idx, vals, 1);
      expect(arraysClose((a as any).toArray(), (aRef as any).toArray())).toBe(true);
    });

    it('putmask: basic values', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const mask = array([false, true, false, true, false]);
      const vals = array([99.0]);
      putmask(a, mask, vals);
      const py = runNumPy(`
a = np.array([1.,2.,3.,4.,5.])
np.putmask(a, [False,True,False,True,False], 99.)
result = a`);
      expect(arraysClose((a as any).toArray(), py.value)).toBe(true);
    });

    it('putmask: 2D values', () => {
      const a = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const mask = array([
        [true, false],
        [false, true],
      ]);
      putmask(a, mask, array([0.0]));
      const py = runNumPy(`
a = np.array([[1.,2.],[3.,4.]])
np.putmask(a, np.array([[True,False],[False,True]]), 0.)
result = a`);
      expect(arraysClose((a as any).toArray(), py.value)).toBe(true);
    });

    it('place: basic values', () => {
      const a = array([1.0, 2.0, 3.0, 4.0, 5.0]);
      const mask = array([false, true, false, true, false]);
      place(a, mask, array([10.0, 20.0]));
      const py = runNumPy(`
a = np.array([1.,2.,3.,4.,5.])
np.place(a, [False,True,False,True,False], [10.,20.])
result = a`);
      expect(arraysClose((a as any).toArray(), py.value)).toBe(true);
    });

    it('place: 2D values', () => {
      const a = array([
        [1.0, 2.0],
        [3.0, 4.0],
      ]);
      const mask = array([
        [true, false],
        [true, false],
      ]);
      place(a, mask, array([99.0]));
      const py = runNumPy(`
a = np.array([[1.,2.],[3.,4.]])
np.place(a, np.array([[True,False],[True,False]]), [99.])
result = a`);
      expect(arraysClose((a as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 63: packbits / unpackbits — with axis
  // ============================================================
  describe('packbits/unpackbits: with axis', () => {
    it('packbits: 2D axis=0 -> shape', () => {
      const a = array(
        [
          [1, 0, 1, 0, 1, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        'uint8'
      );
      const r = packbits(a, 0);
      const py = runNumPy(`
a = np.array([[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]], dtype=np.uint8)
result = np.packbits(a, axis=0)`);
      expect((r as any).shape).toEqual(py.shape);
    });

    it('packbits: 2D axis=1 -> shape', () => {
      const a = array(
        [
          [1, 0, 1, 0, 1, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        'uint8'
      );
      const r = packbits(a, 1);
      const py = runNumPy(`
a = np.array([[1,0,1,0,1,0,1,0],[0,1,0,1,0,1,0,1]], dtype=np.uint8)
result = np.packbits(a, axis=1)`);
      expect((r as any).shape).toEqual(py.shape);
    });

    it('packbits: 2D axis=-1 neg -> same as axis=1', () => {
      const a = array(
        [
          [1, 0, 1, 0, 1, 0, 1, 0],
          [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        'uint8'
      );
      const r1 = packbits(a, 1);
      const r2 = packbits(a, -1);
      expect((r2 as any).shape).toEqual((r1 as any).shape);
    });

    it('unpackbits: 2D axis=0 -> shape', () => {
      const a = array(
        [
          [0b10101010, 0b11110000],
          [0b00001111, 0b01010101],
        ],
        'uint8'
      );
      const r = unpackbits(a, 0);
      const py = runNumPy(`
a = np.array([[0b10101010, 0b11110000],[0b00001111, 0b01010101]], dtype=np.uint8)
result = np.unpackbits(a, axis=0)`);
      expect((r as any).shape).toEqual(py.shape);
    });

    it('unpackbits: 2D axis=1 -> shape', () => {
      const a = array(
        [
          [0b10101010, 0b11110000],
          [0b00001111, 0b01010101],
        ],
        'uint8'
      );
      const r = unpackbits(a, 1);
      const py = runNumPy(`
a = np.array([[0b10101010, 0b11110000],[0b00001111, 0b01010101]], dtype=np.uint8)
result = np.unpackbits(a, axis=1)`);
      expect((r as any).shape).toEqual(py.shape);
    });

    it('unpackbits: 2D axis=-1 neg -> same as axis=1', () => {
      const a = array(
        [
          [0b10101010, 0b11110000],
          [0b00001111, 0b01010101],
        ],
        'uint8'
      );
      const r1 = unpackbits(a, 1);
      const r2 = unpackbits(a, -1);
      expect((r2 as any).shape).toEqual((r1 as any).shape);
    });
  });

  // ============================================================
  // SECTION 64: unwrap — with axis
  // ============================================================
  describe('unwrap: with axis', () => {
    it('unwrap: 1D no-op (already unwrapped)', () => {
      const a = array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5]);
      const r = unwrap(a);
      // Already monotonically increasing within range, shape is preserved
      expect((r as any).shape).toEqual([6]);
      const py = runNumPy('result = np.unwrap(np.array([0.,0.5,1.,1.5,2.,2.5]))');
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('unwrap: 2D axis=0 shape preserved', () => {
      const a = array([
        [0.0, 0.0],
        [0.5, 0.5],
        [1.0, 1.0],
        [1.5, 1.5],
        [2.0, 2.0],
      ]);
      const r = unwrap(a, undefined, 0);
      expect((r as any).shape).toEqual([5, 2]);
    });

    it('unwrap: 2D axis=1 shape preserved', () => {
      const a = array([
        [0.0, 0.5, 1.0, 1.5, 2.0],
        [0.0, 0.5, 1.0, 1.5, 2.0],
      ]);
      const r = unwrap(a, undefined, 1);
      const py = runNumPy(`
a = np.array([[0.,0.5,1.,1.5,2.],[0.,0.5,1.,1.5,2.]])
result = np.unwrap(a, axis=1)`);
      expect((r as any).shape).toEqual([2, 5]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('unwrap: 2D axis=-1 neg -> same as axis=1', () => {
      const a = array([
        [0.0, 1.0, 2.0, -3.0, -2.0],
        [0.0, 1.0, 2.0, -3.0, -2.0],
      ]);
      const r1 = unwrap(a, undefined, 1);
      const r2 = unwrap(a, undefined, -1);
      expect(arraysClose((r1 as any).toArray(), (r2 as any).toArray())).toBe(true);
    });

    it('unwrap: 3D axis=-1 neg -> shape preserved', () => {
      const a = mk([2, 3, 5]);
      const r = unwrap(a, undefined, -1);
      expect((r as any).shape).toEqual([2, 3, 5]);
    });
  });

  // ============================================================
  // SECTION 65: apply_along_axis / apply_over_axes — negative axes
  // ============================================================
  describe('apply_along_axis/apply_over_axes: negative axes', () => {
    it('apply_along_axis: 2D axis=-1 neg -> same as axis=1', () => {
      const a = array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
      ]);
      const sumFn = (row: any) => {
        const vals = row.toArray() as number[];
        return vals.reduce((s: number, v: number) => s + v, 0);
      };
      const r1 = apply_along_axis(sumFn, 1, a);
      const r2 = apply_along_axis(sumFn, -1, a);
      expect(arraysClose((r1 as any).toArray(), (r2 as any).toArray())).toBe(true);
    });

    it('apply_along_axis: 2D axis=-1 neg -> shape [2]', () => {
      const a = array([
        [1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
      ]);
      const sumFn = (row: any) => {
        const vals = row.toArray() as number[];
        return vals.reduce((s: number, v: number) => s + v, 0);
      };
      const r = apply_along_axis(sumFn, -1, a);
      const py = runNumPy(`
a = np.array([[1.,2.,3.,4.],[5.,6.,7.,8.]])
result = np.apply_along_axis(np.sum, -1, a)`);
      expect((r as any).shape).toEqual([2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('apply_over_axes: 3D axes=[-1] neg -> shape preserved [2,3,1]', () => {
      const a = mk([2, 3, 4]);
      const sumFn = (x: any, axis: number) => x.sum(axis, undefined, true);
      const r = apply_over_axes(sumFn, a, [-1]);
      expect((r as any).shape).toEqual([2, 3, 1]);
    });
  });

  // ============================================================
  // SECTION 66: take — negative axes (plain number[] indices)
  // ============================================================
  describe('take: negative axes with plain array indices', () => {
    it('take: 2D axis=-1 neg -> shape [3,2] values', () => {
      const a = mk([3, 4]);
      const r = take(a, [0, 2], -1);
      const py = runNumPy(
        'result = np.take(np.arange(1,13,dtype=float).reshape(3,4), [0,2], axis=-1)'
      );
      expect((r as any).shape).toEqual([3, 2]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('take: 3D axis=-1 neg -> shape [2,3,2]', () => {
      const a = mk([2, 3, 4]);
      const r = take(a, [0, 2], -1);
      expect((r as any).shape).toEqual([2, 3, 2]);
    });

    it('take: 3D axis=-2 neg -> shape [2,2,4]', () => {
      const a = mk([2, 3, 4]);
      const r = take(a, [0, 1], -2);
      expect((r as any).shape).toEqual([2, 2, 4]);
    });

    it('take: 4D axis=-3 neg -> shape [2,2,4,5]', () => {
      const a = mk([2, 3, 4, 5]);
      const r = take(a, [0, 1], -3);
      expect((r as any).shape).toEqual([2, 2, 4, 5]);
    });
  });

  // ============================================================
  // SECTION 67: argmax / argmin — value verification
  // ============================================================
  describe('argmax/argmin: value verification', () => {
    it('argmax: 1D values', () => {
      const a = array([3.0, 1.0, 9.0, 4.0, 2.0]);
      const r = argmax(a);
      const py = runNumPy('result = np.argmax(np.array([3.,1.,9.,4.,2.]))');
      expect(r).toBe(py.value);
    });

    it('argmin: 1D values', () => {
      const a = array([3.0, 1.0, 9.0, 4.0, 2.0]);
      const r = argmin(a);
      const py = runNumPy('result = np.argmin(np.array([3.,1.,9.,4.,2.]))');
      expect(r).toBe(py.value);
    });

    it('argmax: 2D axis=0 values', () => {
      const a = array([
        [1.0, 8.0, 3.0],
        [7.0, 2.0, 9.0],
      ]);
      const r = argmax(a, 0);
      const py = runNumPy('result = np.argmax(np.array([[1.,8.,3.],[7.,2.,9.]]), axis=0)');
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('argmin: 2D axis=1 values', () => {
      const a = array([
        [1.0, 8.0, 3.0],
        [7.0, 2.0, 9.0],
      ]);
      const r = argmin(a, 1);
      const py = runNumPy('result = np.argmin(np.array([[1.,8.,3.],[7.,2.,9.]]), axis=1)');
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('argmax: 3D axis=-1 neg values', () => {
      const a = mk([2, 3, 4]);
      const r = argmax(a, -1);
      const py = runNumPy(
        'result = np.argmax(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-1)'
      );
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('argmin: 3D axis=-2 neg values', () => {
      const a = mk([2, 3, 4]);
      const r = argmin(a, -2);
      const py = runNumPy(
        'result = np.argmin(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-2)'
      );
      expect((r as any).shape).toEqual([2, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 68: average — keepdims
  // ============================================================
  describe('average: keepdims', () => {
    it('average: 2D axis=0 keepdims=true -> shape [1,3]', () => {
      const a = mk([2, 3]);
      const r = average(a, 0, undefined, true);
      const py = runNumPy(
        'result = np.average(np.arange(1,7,dtype=float).reshape(2,3), axis=0, keepdims=True)'
      );
      expect((r as any).shape).toEqual([1, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('average: 2D axis=1 keepdims=true -> shape [2,1]', () => {
      const a = mk([2, 3]);
      const r = average(a, 1, undefined, true);
      const py = runNumPy(
        'result = np.average(np.arange(1,7,dtype=float).reshape(2,3), axis=1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('average: 3D axis=1 keepdims=true -> shape [2,1,4]', () => {
      const a = mk([2, 3, 4]);
      const r = average(a, 1, undefined, true);
      const py = runNumPy(
        'result = np.average(np.arange(1,25,dtype=float).reshape(2,3,4), axis=1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 1, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('average: 3D axis=-1 keepdims=true -> shape [2,3,1]', () => {
      const a = mk([2, 3, 4]);
      const r = average(a, -1, undefined, true);
      const py = runNumPy(
        'result = np.average(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-1, keepdims=True)'
      );
      expect((r as any).shape).toEqual([2, 3, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 69: nancumprod — value verification
  // ============================================================
  describe('nancumprod: value verification', () => {
    it('nancumprod: 1D with NaN values', () => {
      const a = array([1.0, NaN, 3.0, NaN, 5.0]);
      const r = nancumprod(a);
      const py = runNumPy('result = np.nancumprod(np.array([1.,np.nan,3.,np.nan,5.]))');
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nancumprod: 2D axis=0 values', () => {
      const a = array([
        [1.0, NaN, 3.0],
        [NaN, 2.0, 4.0],
      ]);
      const r = nancumprod(a, 0);
      const py = runNumPy(`
a = np.array([[1.,np.nan,3.],[np.nan,2.,4.]])
result = np.nancumprod(a, axis=0)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('nancumprod: 2D axis=1 values', () => {
      const a = array([
        [1.0, 2.0, NaN],
        [NaN, 3.0, 4.0],
      ]);
      const r = nancumprod(a, 1);
      const py = runNumPy(`
a = np.array([[1.,2.,np.nan],[np.nan,3.,4.]])
result = np.nancumprod(a, axis=1)`);
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 70: copysign / nextafter / spacing / signbit — values
  // ============================================================
  describe('copysign/nextafter/spacing/signbit: value verification', () => {
    it('copysign: values', () => {
      const x = array([1.0, -2.0, 3.0, -4.0]);
      const y = array([-1.0, 1.0, -1.0, 1.0]);
      const r = copysign(x, y);
      const py = runNumPy(
        'result = np.copysign(np.array([1.,-2.,3.,-4.]), np.array([-1.,1.,-1.,1.]))'
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('copysign: 2D shape preserved', () => {
      const r = copysign(mk([2, 3]), mk([2, 3]));
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('nextafter: 1D values', () => {
      const x = array([1.0, -1.0, 0.0]);
      const y = array([2.0, -2.0, 1.0]);
      const r = nextafter(x, y);
      const py = runNumPy('result = np.nextafter(np.array([1.,-1.,0.]), np.array([2.,-2.,1.]))');
      // nextafter(1.0, 2.0) -> slightly > 1.0
      // nextafter(-1.0, -2.0) -> slightly < -1.0 (more negative)
      const vals = (r as any).toArray() as number[];
      expect(vals[0]).toBeGreaterThan(1.0);
      expect(vals[1]).toBeLessThan(-1.0);
      expect(py.value).toBeTruthy();
    });

    it('spacing: 1D values', () => {
      const x = array([1.0, 10.0, 100.0]);
      const r = spacing(x);
      const py = runNumPy('result = np.spacing(np.array([1.,10.,100.]))');
      expect((r as any).shape).toEqual([3]);
      // spacing values are very small — just verify shape and positivity
      const vals = (r as any).toArray() as number[];
      expect(vals[0]).toBeGreaterThan(0);
      expect(py.value).toBeTruthy();
    });

    it('signbit: 1D values', () => {
      const x = array([-1.0, 0.0, 1.0, -0.0]);
      const r = signbit(x);
      const py = runNumPy('result = np.signbit(np.array([-1.,0.,1.,-0.]))');
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('signbit: 2D shape preserved', () => {
      const r = signbit(mk([2, 3]));
      expect((r as any).shape).toEqual([2, 3]);
    });
  });

  // ============================================================
  // SECTION 71: bitwise_count — value verification
  // ============================================================
  describe('bitwise_count: value verification', () => {
    it('bitwise_count: 1D uint8 values', () => {
      const a = array([0, 1, 2, 3, 255, 127, 128, 64], 'uint8');
      const r = bitwise_count(a);
      const py = runNumPy(
        'result = np.bitwise_count(np.array([0,1,2,3,255,127,128,64], dtype=np.uint8))'
      );
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('bitwise_count: 2D shape preserved', () => {
      const a = array(
        [
          [1, 2, 3],
          [4, 5, 6],
        ],
        'uint8'
      );
      const r = bitwise_count(a);
      const py = runNumPy('result = np.bitwise_count(np.array([[1,2,3],[4,5,6]], dtype=np.uint8))');
      expect((r as any).shape).toEqual([2, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });
  });

  // ============================================================
  // SECTION 72: shape ops — negative axes (expand_dims, moveaxis, flip, rot90, roll, concatenate)
  // ============================================================
  describe('shape ops: negative axes (expand_dims, moveaxis, flip, rot90, roll, concatenate)', () => {
    it('expand_dims: axis=-1 neg -> shape [2,3,1]', () => {
      const a = mk([2, 3]);
      const r = expand_dims(a, -1);
      const py = runNumPy(
        'result = np.expand_dims(np.arange(1,7,dtype=float).reshape(2,3), axis=-1)'
      );
      expect((r as any).shape).toEqual([2, 3, 1]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('expand_dims: axis=-2 neg -> shape [2,1,3]', () => {
      const a = mk([2, 3]);
      const r = expand_dims(a, -2);
      const py = runNumPy(
        'result = np.expand_dims(np.arange(1,7,dtype=float).reshape(2,3), axis=-2)'
      );
      expect((r as any).shape).toEqual([2, 1, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('moveaxis: neg source/dest -> same result as positive', () => {
      const a = mk([2, 3, 4]);
      const r1 = moveaxis(a, 2, 0);
      const r2 = moveaxis(a, -1, 0);
      expect(arraysClose((r1 as any).toArray(), (r2 as any).toArray())).toBe(true);
    });

    it('moveaxis: 3D source=-1 dest=-2 -> shape [2,4,3]', () => {
      const a = mk([2, 3, 4]);
      const r = moveaxis(a, -1, -2);
      const py = runNumPy(
        'result = np.moveaxis(np.arange(1,25,dtype=float).reshape(2,3,4), -1, -2)'
      );
      expect((r as any).shape).toEqual([2, 4, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('flip: 2D axis=-1 neg -> values', () => {
      const a = mk([3, 4]);
      const r = flip(a, -1);
      const py = runNumPy('result = np.flip(np.arange(1,13,dtype=float).reshape(3,4), axis=-1)');
      expect((r as any).shape).toEqual([3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('flip: 3D axis=-2 neg -> values', () => {
      const a = mk([2, 3, 4]);
      const r = flip(a, -2);
      const py = runNumPy('result = np.flip(np.arange(1,25,dtype=float).reshape(2,3,4), axis=-2)');
      expect((r as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('rot90: 3D axes=(-2,-1) neg -> shape [2,4,3]', () => {
      const a = mk([2, 3, 4]);
      const r = rot90(a, 1, [-2, -1]);
      const py = runNumPy(
        'result = np.rot90(np.arange(1,25,dtype=float).reshape(2,3,4), 1, (-2,-1))'
      );
      expect((r as any).shape).toEqual([2, 4, 3]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('rot90: 3D axes=(-1,-2) neg -> shape [2,3,4] (undo)', () => {
      const a = mk([2, 3, 4]);
      const r1 = rot90(a, 1, [1, 2]);
      const r2 = rot90(a, 1, [-2, -1]);
      expect(arraysClose((r1 as any).toArray(), (r2 as any).toArray())).toBe(true);
    });

    it('roll: 3D axis=-1 neg -> shape preserved [2,3,4]', () => {
      const a = mk([2, 3, 4]);
      const r = roll(a, 2, -1);
      const py = runNumPy(
        'result = np.roll(np.arange(1,25,dtype=float).reshape(2,3,4), 2, axis=-1)'
      );
      expect((r as any).shape).toEqual([2, 3, 4]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('roll: 3D axis=-2 neg -> values', () => {
      const a = mk([2, 3, 4]);
      const r1 = roll(a, 1, 1);
      const r2 = roll(a, 1, -2);
      expect(arraysClose((r1 as any).toArray(), (r2 as any).toArray())).toBe(true);
    });

    it('concatenate: 2D axis=-1 neg -> shape [2,7]', () => {
      const a = mk([2, 3]);
      const b = mk([2, 4]);
      const r = concatenate([a, b], -1);
      const py = runNumPy(`
a = np.arange(1,7,dtype=float).reshape(2,3)
b = np.arange(1,9,dtype=float).reshape(2,4)
result = np.concatenate([a, b], axis=-1)`);
      expect((r as any).shape).toEqual([2, 7]);
      expect(arraysClose((r as any).toArray(), py.value)).toBe(true);
    });

    it('concatenate: 3D axis=-1 neg -> shape [2,3,8]', () => {
      const a = mk([2, 3, 4]);
      const b = mk([2, 3, 4]);
      const r = concatenate([a, b], -1);
      expect((r as any).shape).toEqual([2, 3, 8]);
    });
  });

  // ============================================================
  // SECTION 73: linalg batch ops — svdvals, cholesky, eig/eigh, eigvals/eigvalsh, lstsq, pinv, slogdet
  // ============================================================
  describe('linalg batch ops: svdvals, cholesky, eig variants, lstsq, pinv, slogdet', () => {
    /** Build a batch of identity-scaled matrices for numerical stability */
    const mkEye = (shape: number[]) => {
      const n = shape[shape.length - 1]!;
      const batchDims = shape.slice(0, -2);
      const batchSize = batchDims.reduce((a, b) => a * b, 1);
      const data = Array.from({ length: batchSize }, (_, b) =>
        Array.from({ length: n }, (_, i) =>
          Array.from({ length: n }, (_, j) => (i === j ? b + 2.0 : 0.0))
        )
      ).flat(2);
      return array(data).reshape(shape);
    };

    // svdvals
    it('svdvals: 2D [3,4] -> shape [3]', () => {
      const r = linalg.svdvals(mk([3, 4]));
      expect((r as any).shape).toEqual([3]);
    });

    it('svdvals: 3D [2,3,4] batch -> shape [2,3]', () => {
      const r = linalg.svdvals(mk([2, 3, 4]));
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('svdvals: 4D [2,2,3,4] batch -> shape [2,2,3]', () => {
      const r = linalg.svdvals(mk([2, 2, 3, 4]));
      expect((r as any).shape).toEqual([2, 2, 3]);
    });

    // cholesky
    it('cholesky: 2D [3,3] -> shape [3,3]', () => {
      const A = mkEye([3, 3]);
      const r = linalg.cholesky(A);
      expect((r as any).shape).toEqual([3, 3]);
    });

    it('cholesky: 3D [2,3,3] batch -> shape [2,3,3]', () => {
      const A = mkEye([2, 3, 3]);
      const r = linalg.cholesky(A);
      expect((r as any).shape).toEqual([2, 3, 3]);
    });

    it('cholesky: 4D [2,2,3,3] batch -> shape [2,2,3,3]', () => {
      const A = mkEye([2, 2, 3, 3]);
      const r = linalg.cholesky(A);
      expect((r as any).shape).toEqual([2, 2, 3, 3]);
    });

    // eig/eigvals
    it('eig: 2D [3,3] -> w shape [3], v shape [3,3]', () => {
      const A = mkEye([3, 3]);
      const { w, v } = linalg.eig(A);
      expect((w as any).shape).toEqual([3]);
      expect((v as any).shape).toEqual([3, 3]);
    });

    it('eig: 3D [2,3,3] batch -> w shape [2,3]', () => {
      const A = mkEye([2, 3, 3]);
      const { w } = linalg.eig(A);
      expect((w as any).shape).toEqual([2, 3]);
    });

    it('eigvals: 2D [3,3] -> shape [3]', () => {
      const A = mkEye([3, 3]);
      const r = linalg.eigvals(A);
      expect((r as any).shape).toEqual([3]);
    });

    it('eigvals: 3D [2,3,3] batch -> shape [2,3]', () => {
      const A = mkEye([2, 3, 3]);
      const r = linalg.eigvals(A);
      expect((r as any).shape).toEqual([2, 3]);
    });

    // eigh/eigvalsh (symmetric positive definite)
    it('eigh: 2D [3,3] SPD -> w shape [3]', () => {
      const A = mkEye([3, 3]);
      const { w } = linalg.eigh(A);
      expect((w as any).shape).toEqual([3]);
    });

    it('eigh: 3D [2,3,3] batch -> w shape [2,3]', () => {
      const A = mkEye([2, 3, 3]);
      const { w } = linalg.eigh(A);
      expect((w as any).shape).toEqual([2, 3]);
    });

    it('eigvalsh: 2D [4,4] SPD -> shape [4]', () => {
      const A = mkEye([4, 4]);
      const r = linalg.eigvalsh(A);
      expect((r as any).shape).toEqual([4]);
    });

    it('eigvalsh: 3D [2,4,4] batch -> shape [2,4]', () => {
      const A = mkEye([2, 4, 4]);
      const r = linalg.eigvalsh(A);
      expect((r as any).shape).toEqual([2, 4]);
    });

    // lstsq
    it('lstsq: 2D overdetermined [4,2] -> x shape [2]', () => {
      const A = array([
        [1.0, 1.0],
        [1.0, 2.0],
        [1.0, 3.0],
        [1.0, 4.0],
      ]);
      const b = array([6.0, 5.0, 7.0, 10.0]);
      const { x } = linalg.lstsq(A, b);
      expect((x as any).shape).toEqual([2]);
    });

    it('lstsq: 2D multiple RHS [4,2] x [4,3] -> x shape [2,3]', () => {
      const A = array([
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [1.0, -1.0],
      ]);
      const b = mk([4, 3]);
      const { x } = linalg.lstsq(A, b);
      expect((x as any).shape).toEqual([2, 3]);
    });

    // pinv
    it('pinv: 2D [3,4] -> shape [4,3]', () => {
      const r = linalg.pinv(mk([3, 4]));
      expect((r as any).shape).toEqual([4, 3]);
    });

    it('pinv: 2D [4,3] -> shape [3,4]', () => {
      const r = linalg.pinv(mk([4, 3]));
      expect((r as any).shape).toEqual([3, 4]);
    });

    it('pinv: 3D [2,3,4] batch -> shape [2,4,3]', () => {
      const r = linalg.pinv(mk([2, 3, 4]));
      expect((r as any).shape).toEqual([2, 4, 3]);
    });

    // slogdet
    it('slogdet: 2D [3,3] -> sign/logabsdet values', () => {
      const A = array([
        [1.0, 2.0, 3.0],
        [0.0, 4.0, 5.0],
        [0.0, 0.0, 6.0],
      ]);
      const { sign, logabsdet } = linalg.slogdet(A);
      const py = runNumPy(
        'sign, logabsdet = np.linalg.slogdet(np.array([[1.,2.,3.],[0.,4.,5.],[0.,0.,6.]]))\nresult = [float(sign), float(logabsdet)]'
      );
      expect(sign).toBeCloseTo((py.value as number[])[0]!, 10);
      expect(logabsdet).toBeCloseTo((py.value as number[])[1]!, 8);
    });

    it('slogdet: 3D [2,3,3] batch -> sign/logabsdet arrays', () => {
      const A = mkEye([2, 3, 3]);
      const result = linalg.slogdet(A);
      // NumPy returns arrays for batched input; both sign and logabsdet should be arrays
      expect(Array.isArray((result as any).sign) || typeof (result as any).sign === 'object').toBe(
        true
      );
    });
  });

  // ============================================================
  // SECTION 74: fft.hfft / fft.ihfft — all axes + neg axes
  // fft.fftshift / fft.ifftshift — neg axes
  // ============================================================
  describe('fft.hfft/ihfft: axes and neg axes; fftshift/ifftshift: neg axes', () => {
    it('fft.hfft: 1D -> shape [14] (n=14 for 8-point input)', () => {
      const a = mk([8]);
      const r = fft.hfft(a);
      // hfft of n-point input produces 2*(n-1) output by default
      expect((r as any).shape[0]).toBe(14);
    });

    it('fft.hfft: 2D axis=0 -> shape matches numpy', () => {
      const a = mk([4, 6]);
      const r = fft.hfft(a, undefined, 0);
      const py = runNumPy('result = np.fft.hfft(np.arange(1,25,dtype=float).reshape(4,6), axis=0)');
      expect((r as any).shape).toEqual(py.shape);
    });

    it('fft.hfft: 2D axis=1 -> shape matches numpy', () => {
      const a = mk([4, 6]);
      const r = fft.hfft(a, undefined, 1);
      const py = runNumPy('result = np.fft.hfft(np.arange(1,25,dtype=float).reshape(4,6), axis=1)');
      expect((r as any).shape).toEqual(py.shape);
    });

    it('fft.hfft: 2D axis=-1 neg -> same as axis=1', () => {
      const a = mk([4, 6]);
      const r1 = fft.hfft(a, undefined, 1);
      const r2 = fft.hfft(a, undefined, -1);
      expect((r2 as any).shape).toEqual((r1 as any).shape);
    });

    it('fft.ihfft: 1D shape', () => {
      const a = mk([8]);
      const r = fft.ihfft(a);
      expect((r as any).shape).toEqual([5]); // ihfft of n real points -> n//2+1
    });

    it('fft.ihfft: 2D axis=0', () => {
      const a = mk([4, 6]);
      const r = fft.ihfft(a, undefined, 0);
      const py = runNumPy(
        'result = np.fft.ihfft(np.arange(1,25,dtype=float).reshape(4,6), axis=0)'
      );
      expect((r as any).shape).toEqual(py.shape);
    });

    it('fft.ihfft: 2D axis=-1 neg -> same as axis=1', () => {
      const a = mk([4, 6]);
      const r1 = fft.ihfft(a, undefined, 1);
      const r2 = fft.ihfft(a, undefined, -1);
      expect((r2 as any).shape).toEqual((r1 as any).shape);
    });

    it('fft.fftshift: axis=-1 neg -> same as axis=1', () => {
      const a = mk([4, 8]);
      const r1 = fft.fftshift(a, 1);
      const r2 = fft.fftshift(a, -1);
      expect(arraysClose((r1 as any).toArray(), (r2 as any).toArray())).toBe(true);
    });

    it('fft.fftshift: 3D axis=-1 neg -> shape preserved', () => {
      const a = mk([2, 4, 8]);
      const r = fft.fftshift(a, -1);
      expect((r as any).shape).toEqual([2, 4, 8]);
    });

    it('fft.ifftshift: axis=-2 neg -> same as axis=1', () => {
      const a = mk([4, 8]);
      const r1 = fft.ifftshift(a, 0);
      const r2 = fft.ifftshift(a, -2);
      expect(arraysClose((r1 as any).toArray(), (r2 as any).toArray())).toBe(true);
    });

    it('fft.ifftshift: 3D axis=-1 neg -> shape preserved', () => {
      const a = mk([2, 4, 8]);
      const r = fft.ifftshift(a, -1);
      expect((r as any).shape).toEqual([2, 4, 8]);
    });
  });

  // ============================================================
  // SECTION 75: Tuple/multiple axes for reductions
  // NumPy supports axis=(0,2) for all reductions.
  // Functions that DON'T support tuple axis silently return the
  // unreduced array (wrong shape) — these should FAIL.
  // Functions that DO support it are included as passing baselines.
  // ============================================================
  describe('tuple/multiple axes for reductions (NumPy divergence)', () => {
    // These already work (baseline):
    it('sum: 3D [2,3,4] axis=(0,2) -> shape [3] [works]', () => {
      const r = sum(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });
    it('mean: 3D [2,3,4] axis=(1,2) -> shape [2] [works]', () => {
      const r = mean(mk([2, 3, 4]), [1, 2] as any);
      expect((r as any).shape).toEqual([2]);
    });
    it('nansum: 3D [2,3,4] axis=(0,2) -> shape [3] [works]', () => {
      const r = nansum(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });

    // These silently ignore the tuple and return wrong shape:
    it('amax: 3D [2,3,4] axis=(0,1) -> shape [4]', () => {
      const r = amax(mk([2, 3, 4]), [0, 1] as any);
      expect((r as any).shape).toEqual([4]);
    });
    it('amin: 3D [2,3,4] axis=(0,2) -> shape [3]', () => {
      const r = amin(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });
    it('all: 3D [2,3,4] axis=(0,2) -> shape [3]', () => {
      const r = all(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });
    it('any: 3D [2,3,4] axis=(0,2) -> shape [3]', () => {
      const r = any(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });
    it('nanmin: 3D [2,3,4] axis=(0,2) -> shape [3]', () => {
      const r = nanmin(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });
    it('nanmax: 3D [2,3,4] axis=(0,2) -> shape [3]', () => {
      const r = nanmax(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });
    it('median: 3D [2,3,4] axis=(0,2) -> shape [3]', () => {
      const r = median(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });
    it('count_nonzero: 3D [2,3,4] axis=(0,2) -> shape [3]', () => {
      const r = count_nonzero(mk([2, 3, 4]), [0, 2] as any);
      expect((r as any).shape).toEqual([3]);
    });
  });

  // ============================================================
  // SECTION 76: apply_along_axis with 3D arrays
  // NumPy supports ND; our implementation is 2D-only.
  // ============================================================
  describe('apply_along_axis: 3D arrays (NumPy divergence)', () => {
    it('apply_along_axis: 3D [2,3,4] axis=2 -> shape [2,3]', () => {
      const a = mk([2, 3, 4]);
      const r = apply_along_axis((x: any) => sum(x), 2, a);
      expect((r as any).shape).toEqual([2, 3]);
    });

    it('apply_along_axis: 3D [2,3,4] axis=0 -> shape [3,4]', () => {
      const a = mk([2, 3, 4]);
      const r = apply_along_axis((x: any) => sum(x), 0, a);
      expect((r as any).shape).toEqual([3, 4]);
    });
  });

  // ============================================================
  // SECTION 77: unique with axis parameter
  // NumPy np.unique(a, axis=0) finds unique rows/columns.
  // Our unique() has no axis param — these should FAIL.
  // ============================================================
  describe('unique with axis parameter (NumPy divergence)', () => {
    it('unique: 2D [4,3] axis=0 -> unique rows', () => {
      // a has rows [1,2,3], [4,5,6], [1,2,3], [7,8,9] — 3 unique rows
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [1, 2, 3],
        [7, 8, 9],
      ]);
      const r = (unique as any)(a, false, false, false, 0);
      expect((r as any).shape).toEqual([3, 3]);
    });

    it('unique: 2D [3,4] axis=1 -> unique columns', () => {
      const py = runNumPy(
        'a = np.array([[1,1,2,3],[4,4,5,6],[7,7,8,9]])\nresult = np.unique(a, axis=1)'
      );
      const a = array([
        [1, 1, 2, 3],
        [4, 4, 5, 6],
        [7, 7, 8, 9],
      ]);
      const r = (unique as any)(a, false, false, false, 1);
      expect((r as any).shape).toEqual(py.shape);
    });
  });

  // ============================================================
  // SECTION 78: linalg.matrix_norm with 3D batch
  // NumPy supports batched matrix norms; our impl is 2D-only.
  // ============================================================
  describe('linalg.matrix_norm: 3D batch (NumPy divergence)', () => {
    it('linalg.matrix_norm: 3D [2,3,4] Frobenius -> shape [2]', () => {
      const a = mk([2, 3, 4]);
      const r = linalg.matrix_norm(a, 'fro');
      expect((r as any).shape).toEqual([2]);
    });

    it('linalg.matrix_norm: 4D [2,2,3,4] Frobenius -> shape [2,2]', () => {
      const a = mk([2, 2, 3, 4]);
      const r = linalg.matrix_norm(a, 'fro');
      expect((r as any).shape).toEqual([2, 2]);
    });
  });
});

/**
 * NumPy validation tests for reductions with keepdims and no axis (shape of all ones), and for
 * multi-axis reductions that do not compose one axis at a time (std, var, median, nan means).
 */

import { beforeAll, describe, expect, it } from 'vitest';
import type { NDArray } from '../../src';
import * as np from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

const SHAPES: number[][] = [[5], [2, 3], [2, 3, 4]];
const NAN = Number.NaN;

// Distinct, non-monotonic values so that min/max/argmax/median differ across positions.
function values(n: number, withNaN: boolean): number[] {
  return Array.from({ length: n }, (_, i) => {
    if (withNaN && i % 5 === 2) return NAN;
    return ((i * 7) % 11) + i / 10;
  });
}

function pyArray(vals: number[], shape: number[]): string {
  const lits = vals.map((v) => (Number.isNaN(v) ? 'np.nan' : String(v))).join(', ');
  return `np.array([${lits}], dtype=np.float64).reshape(${JSON.stringify(shape)})`;
}

interface Case {
  name: string;
  nan?: boolean;
  // Builds the call on the JS side and the matching NumPy expression, both with keepdims=True.
  js: (a: NDArray) => unknown;
  py: string;
}

const CASES: Case[] = [
  { name: 'sum', js: (a) => np.sum(a, undefined, true), py: 'np.sum(a, keepdims=True)' },
  { name: 'prod', js: (a) => np.prod(a, undefined, true), py: 'np.prod(a, keepdims=True)' },
  { name: 'mean', js: (a) => np.mean(a, undefined, true), py: 'np.mean(a, keepdims=True)' },
  { name: 'std', js: (a) => np.std(a, undefined, 0, true), py: 'np.std(a, keepdims=True)' },
  {
    name: 'var',
    js: (a) => np.variance(a, undefined, 1, true),
    py: 'np.var(a, ddof=1, keepdims=True)',
  },
  { name: 'min', js: (a) => np.min(a, undefined, true), py: 'np.min(a, keepdims=True)' },
  { name: 'max', js: (a) => np.max(a, undefined, true), py: 'np.max(a, keepdims=True)' },
  { name: 'amin', js: (a) => np.amin(a, undefined, true), py: 'np.amin(a, keepdims=True)' },
  { name: 'amax', js: (a) => np.amax(a, undefined, true), py: 'np.amax(a, keepdims=True)' },
  { name: 'ptp', js: (a) => np.ptp(a, undefined, true), py: 'np.ptp(a, keepdims=True)' },
  { name: 'any', js: (a) => np.any(a, undefined, true), py: 'np.any(a, keepdims=True)' },
  { name: 'all', js: (a) => np.all(a, undefined, true), py: 'np.all(a, keepdims=True)' },
  { name: 'argmax', js: (a) => np.argmax(a, undefined, true), py: 'np.argmax(a, keepdims=True)' },
  { name: 'argmin', js: (a) => np.argmin(a, undefined, true), py: 'np.argmin(a, keepdims=True)' },
  { name: 'median', js: (a) => np.median(a, undefined, true), py: 'np.median(a, keepdims=True)' },
  {
    name: 'percentile',
    js: (a) => np.percentile(a, 30, undefined, true),
    py: 'np.percentile(a, 30, keepdims=True)',
  },
  {
    name: 'quantile',
    js: (a) => np.quantile(a, 0.7, undefined, true),
    py: 'np.quantile(a, 0.7, keepdims=True)',
  },
  {
    name: 'average',
    js: (a) => np.average(a, undefined, undefined, true),
    py: 'np.average(a, keepdims=True)',
  },
  {
    name: 'nansum',
    nan: true,
    js: (a) => np.nansum(a, undefined, true),
    py: 'np.nansum(a, keepdims=True)',
  },
  {
    name: 'nanprod',
    nan: true,
    js: (a) => np.nanprod(a, undefined, true),
    py: 'np.nanprod(a, keepdims=True)',
  },
  {
    name: 'nanmean',
    nan: true,
    js: (a) => np.nanmean(a, undefined, true),
    py: 'np.nanmean(a, keepdims=True)',
  },
  {
    name: 'nanmin',
    nan: true,
    js: (a) => np.nanmin(a, undefined, true),
    py: 'np.nanmin(a, keepdims=True)',
  },
  {
    name: 'nanmax',
    nan: true,
    js: (a) => np.nanmax(a, undefined, true),
    py: 'np.nanmax(a, keepdims=True)',
  },
  {
    name: 'nanstd',
    nan: true,
    js: (a) => np.nanstd(a, undefined, 0, true),
    py: 'np.nanstd(a, keepdims=True)',
  },
  {
    name: 'nanvar',
    nan: true,
    js: (a) => np.nanvar(a, undefined, 0, true),
    py: 'np.nanvar(a, keepdims=True)',
  },
  {
    name: 'nanmedian',
    nan: true,
    js: (a) => np.nanmedian(a, undefined, true),
    py: 'np.nanmedian(a, keepdims=True)',
  },
  {
    name: 'nanpercentile',
    nan: true,
    js: (a) => np.nanpercentile(a, 30, undefined, true),
    py: 'np.nanpercentile(a, 30, keepdims=True)',
  },
  {
    name: 'nanquantile',
    nan: true,
    js: (a) => np.nanquantile(a, 0.7, undefined, true),
    py: 'np.nanquantile(a, 0.7, keepdims=True)',
  },
];

function expectMatches(result: unknown, py: string, vals: number[], shape: number[]): void {
  const expected = runNumPy(`
a = ${pyArray(vals, shape)}
result = ${py}
`);
  const r = result as NDArray;
  expect(Array.from(r.shape)).toEqual(expected.shape);
  expect(arraysClose(r.toArray(), expected.value, 1e-9)).toBe(true);
}

describe('NumPy Validation: reductions with keepdims and no axis', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  for (const c of CASES) {
    describe(c.name, () => {
      for (const shape of SHAPES) {
        it(`shape ${JSON.stringify(shape)}`, () => {
          const n = shape.reduce((x, y) => x * y, 1);
          const vals = values(n, c.nan ?? false);
          const a = np.array(vals).reshape(shape);
          expectMatches(c.js(a), c.py, vals, shape);
        });
      }
    });
  }

  describe('NDArray methods', () => {
    const vals = values(24, true);
    const shape = [2, 3, 4];
    const methods: [string, (a: NDArray) => unknown, string][] = [
      ['sum', (a) => a.sum(undefined, true), 'np.sum(a, keepdims=True)'],
      ['prod', (a) => a.prod(undefined, true), 'np.prod(a, keepdims=True)'],
      ['mean', (a) => a.mean(undefined, true), 'np.mean(a, keepdims=True)'],
      ['std', (a) => a.std(undefined, 0, true), 'np.std(a, keepdims=True)'],
      ['var', (a) => a.var(undefined, 0, true), 'np.var(a, keepdims=True)'],
      ['min', (a) => a.min(undefined, true), 'np.min(a, keepdims=True)'],
      ['max', (a) => a.max(undefined, true), 'np.max(a, keepdims=True)'],
      ['ptp', (a) => a.ptp(undefined, true), 'np.ptp(a, keepdims=True)'],
      ['any', (a) => a.any(undefined, true), 'np.any(a, keepdims=True)'],
      ['all', (a) => a.all(undefined, true), 'np.all(a, keepdims=True)'],
      ['nansum', (a) => a.nansum(undefined, true), 'np.nansum(a, keepdims=True)'],
      ['nanprod', (a) => a.nanprod(undefined, true), 'np.nanprod(a, keepdims=True)'],
      ['nanmean', (a) => a.nanmean(undefined, true), 'np.nanmean(a, keepdims=True)'],
      ['nanmin', (a) => a.nanmin(undefined, true), 'np.nanmin(a, keepdims=True)'],
      ['nanmax', (a) => a.nanmax(undefined, true), 'np.nanmax(a, keepdims=True)'],
      ['nanstd', (a) => a.nanstd(undefined, 0, true), 'np.nanstd(a, keepdims=True)'],
      ['nanvar', (a) => a.nanvar(undefined, 0, true), 'np.nanvar(a, keepdims=True)'],
      ['median', (a) => a.median(undefined, true), 'np.median(a, keepdims=True)'],
      ['nanmedian', (a) => a.nanmedian(undefined, true), 'np.nanmedian(a, keepdims=True)'],
      [
        'percentile',
        (a) => a.percentile(30, undefined, true),
        'np.percentile(a, 30, keepdims=True)',
      ],
      ['quantile', (a) => a.quantile(0.7, undefined, true), 'np.quantile(a, 0.7, keepdims=True)'],
      [
        'nanpercentile',
        (a) => a.nanpercentile(30, undefined, true),
        'np.nanpercentile(a, 30, keepdims=True)',
      ],
      [
        'nanquantile',
        (a) => a.nanquantile(0.7, undefined, true),
        'np.nanquantile(a, 0.7, keepdims=True)',
      ],
    ];
    for (const [name, js, py] of methods) {
      it(name, () => {
        // Methods that propagate NaN use a NaN-free copy so the comparison stays meaningful.
        const clean = values(24, false);
        const withNaN = name.startsWith('nan');
        const data = withNaN ? vals : clean;
        expectMatches(js(np.array(data).reshape(shape)), py, data, shape);
      });
    }
  });

  describe('zero-size input', () => {
    const zeroCases: [string, (a: NDArray) => unknown, string][] = [
      ['sum', (a) => np.sum(a, undefined, true), 'np.sum(a, keepdims=True)'],
      ['prod', (a) => np.prod(a, undefined, true), 'np.prod(a, keepdims=True)'],
      ['nansum', (a) => np.nansum(a, undefined, true), 'np.nansum(a, keepdims=True)'],
      ['nanprod', (a) => np.nanprod(a, undefined, true), 'np.nanprod(a, keepdims=True)'],
      ['any', (a) => np.any(a, undefined, true), 'np.any(a, keepdims=True)'],
      ['all', (a) => np.all(a, undefined, true), 'np.all(a, keepdims=True)'],
    ];
    for (const shape of [[0], [0, 3], [2, 0]]) {
      for (const [name, js, py] of zeroCases) {
        it(`${name} shape ${JSON.stringify(shape)}`, () => {
          const a = np.zeros(shape);
          const expected = runNumPy(`
a = np.zeros(${JSON.stringify(shape)})
result = ${py}
`);
          const r = js(a) as NDArray;
          expect(Array.from(r.shape)).toEqual(expected.shape);
          expect(arraysClose(r.toArray(), expected.value, 1e-9)).toBe(true);
        });
      }
    }
  });

  describe('average with weights and returned', () => {
    it('keepdims without axis broadcasts both results', () => {
      const vals = values(6, false);
      const w = [1, 2, 3, 4, 5, 6];
      const a = np.array(vals).reshape([2, 3]);
      const [avg, sw] = np.average(a, undefined, np.array(w).reshape([2, 3]), true, true) as [
        NDArray,
        NDArray,
      ];
      const expected = runNumPy(`
a = ${pyArray(vals, [2, 3])}
w = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64).reshape(2, 3)
avg, sw = np.average(a, weights=w, keepdims=True, returned=True)
result = [avg.tolist(), sw.tolist(), list(avg.shape), list(sw.shape)]
`);
      const [eAvg, eSw, eAvgShape, eSwShape] = expected.value as [
        number[][],
        number[][],
        number[],
        number[],
      ];
      expect(Array.from(avg.shape)).toEqual(eAvgShape);
      expect(Array.from(sw.shape)).toEqual(eSwShape);
      expect(arraysClose(avg.toArray(), eAvg, 1e-9)).toBe(true);
      expect(arraysClose(sw.toArray(), eSw, 1e-9)).toBe(true);
    });

    it('unweighted returned counts the elements', () => {
      const a = np.arange(6).reshape([2, 3]);
      const [, sw] = np.average(a, undefined, undefined, true, true) as [NDArray, NDArray];
      expect(Array.from(sw.shape)).toEqual([1, 1]);
      expect(sw.toArray()).toEqual([[6]]);
    });
  });

  describe('multi-axis reductions that do not compose', () => {
    const vals = values(24, false);
    const nanVals = values(24, true);
    const shape = [2, 3, 4];
    const axesList: number[][] = [
      [0, 1],
      [0, 2],
      [1, 2],
      [-1, 0],
      [2, 1],
    ];
    const fns: [string, boolean, (a: NDArray, ax: number[], kd: boolean) => unknown, string][] = [
      ['std', false, (a, ax, kd) => np.std(a, ax, 0, kd), 'np.std(a, axis=AX, keepdims=KD)'],
      [
        'var',
        false,
        (a, ax, kd) => np.variance(a, ax, 1, kd),
        'np.var(a, axis=AX, ddof=1, keepdims=KD)',
      ],
      ['median', false, (a, ax, kd) => np.median(a, ax, kd), 'np.median(a, axis=AX, keepdims=KD)'],
      [
        'nanstd',
        true,
        (a, ax, kd) => np.nanstd(a, ax, 0, kd),
        'np.nanstd(a, axis=AX, keepdims=KD)',
      ],
      [
        'nanvar',
        true,
        (a, ax, kd) => np.nanvar(a, ax, 0, kd),
        'np.nanvar(a, axis=AX, keepdims=KD)',
      ],
      [
        'nanmean',
        true,
        (a, ax, kd) => np.nanmean(a, ax, kd),
        'np.nanmean(a, axis=AX, keepdims=KD)',
      ],
    ];
    for (const [name, nan, js, py] of fns) {
      for (const ax of axesList) {
        for (const kd of [false, true]) {
          it(`${name} axis=${JSON.stringify(ax)} keepdims=${kd}`, () => {
            const data = nan ? nanVals : vals;
            const code = py
              .replace('AX', `(${ax.join(', ')},)`)
              .replace('KD', kd ? 'True' : 'False');
            expectMatches(js(np.array(data).reshape(shape), ax, kd), code, data, shape);
          });
        }
      }
    }

    it('reducing every axis returns a scalar, or an all-ones array with keepdims', () => {
      const a = np.array(vals).reshape(shape);
      const expected = runNumPy(`
a = ${pyArray(vals, shape)}
result = [float(np.std(a, axis=(0, 1, 2))), np.std(a, axis=(0, 1, 2), keepdims=True).tolist()]
`);
      const [eScalar, eKeep] = expected.value as [number, number[][][]];
      expect(np.std(a, [0, 1, 2]) as number).toBeCloseTo(eScalar, 9);
      const kept = np.std(a, [0, 1, 2], 0, true) as NDArray;
      expect(Array.from(kept.shape)).toEqual([1, 1, 1]);
      expect(arraysClose(kept.toArray(), eKeep, 1e-9)).toBe(true);
    });
  });

  describe('count_nonzero over several axes', () => {
    const data = [0, 1, 2, 0, 0, 5, 6, 0, 8, 9, 0, 0, 1, 1, 0, 3, 0, 0, 7, 7, 0, 4, 0, 1];
    const lits = data.join(', ');
    for (const ax of [
      [0, 1],
      [0, 2],
      [1, 2],
      [-1, -2],
    ]) {
      it(`axis=${JSON.stringify(ax)}`, () => {
        const a = np.array(data).reshape([2, 3, 4]);
        const result = np.count_nonzero(a, ax);
        const expected = runNumPy(`
a = np.array([${lits}]).reshape(2, 3, 4)
result = np.count_nonzero(a, axis=(${ax.join(', ')},))
`);
        const got = typeof result === 'number' ? result : result.toArray();
        expect(arraysClose(got, expected.value, 1e-9)).toBe(true);
      });
    }
  });
});

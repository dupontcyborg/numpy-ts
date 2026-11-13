/**
 * NumPy validation tests for histogram_bin_edges and the edges and counts that histogram,
 * histogram2d and histogramdd derive from it.
 */

import { beforeAll, describe, expect, it } from 'vitest';
import type { NDArray } from '../../src';
import * as np from '../../src';
import { arraysClose, checkNumPyAvailable, runNumPy } from './numpy-oracle';

const STRATEGIES = ['auto', 'fd', 'doane', 'scott', 'stone', 'rice', 'sturges', 'sqrt'] as const;

function lcg(n: number, seed: number, scale: number, round = false): number[] {
  let state = seed;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    state = (state * 1103515245 + 12345) % 2147483648;
    const v = (state / 2147483648 - 0.3) * scale;
    out.push(round ? Math.round(v) : v);
  }
  return out;
}

// Skewed data so the strategies disagree with each other.
const DATASETS: Record<string, number[]> = {
  uniform: lcg(50, 1, 10),
  skewed: lcg(80, 2, 10).map((v) => v ** 3),
  small: [1, 2, 4, 7],
  ints: lcg(60, 3, 40, true),
  dup: [1, 1, 1, 2, 2, 5, 5, 5, 5, 9],
  pair: [3, 8],
};

const lit = (v: number): string => (Number.isNaN(v) ? 'np.nan' : String(v));
const pyArr = (vals: number[], dtype: string): string =>
  `np.array([${vals.map(lit).join(', ')}], dtype='${dtype}')`;

function expectEdges(actual: NDArray, py: string, vals: number[], dtype: string, tol = 1e-9): void {
  const expected = runNumPy(`
a = ${pyArr(vals, dtype)}
result = ${py}
`);
  expect(Array.from(actual.shape)).toEqual(expected.shape);
  expect(actual.dtype).toBe(expected.dtype);
  expect(arraysClose(actual.toArray(), expected.value, tol, tol)).toBe(true);
}

describe('NumPy Validation: histogram_bin_edges', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  describe('strategies', () => {
    for (const [name, vals] of Object.entries(DATASETS)) {
      const dtype = name === 'ints' ? 'int32' : 'float64';
      for (const s of STRATEGIES) {
        it(`${s} on ${name}`, () => {
          const a = np.array(vals, dtype as never);
          expectEdges(
            np.histogram_bin_edges(a, s),
            `np.histogram_bin_edges(a, bins='${s}')`,
            vals,
            dtype,
          );
        });
      }
    }
    for (const s of STRATEGIES) {
      it(`${s} with a range`, () => {
        const vals = DATASETS.skewed!;
        const a = np.array(vals);
        expectEdges(
          np.histogram_bin_edges(a, s, [-20, 20]),
          `np.histogram_bin_edges(a, bins='${s}', range=(-20, 20))`,
          vals,
          'float64',
        );
        expectEdges(
          np.histogram_bin_edges(a, s, [-1, 1]),
          `np.histogram_bin_edges(a, bins='${s}', range=(-1, 1))`,
          vals,
          'float64',
        );
      });
      it(`${s} with a range containing no data`, () => {
        const a = np.array(DATASETS.uniform!);
        expectEdges(
          np.histogram_bin_edges(a, s, [100, 200]),
          `np.histogram_bin_edges(a, bins='${s}', range=(100, 200))`,
          DATASETS.uniform!,
          'float64',
        );
      });
      it(`${s} on float32`, () => {
        const vals = DATASETS.uniform!;
        const a = np.array(vals, 'float32');
        expectEdges(
          np.histogram_bin_edges(a, s),
          `np.histogram_bin_edges(a, bins='${s}')`,
          vals,
          'float32',
          1e-5,
        );
      });
      it(`${s} on constant, single and empty input`, () => {
        for (const vals of [[3, 3, 3, 3], [5], []]) {
          const a = np.array(vals);
          expectEdges(
            np.histogram_bin_edges(a, s),
            `np.histogram_bin_edges(a, bins='${s}')`,
            vals,
            'float64',
          );
        }
      });
      it(`${s} on a non-contiguous view`, () => {
        const vals = lcg(40, 9, 10);
        const a = np.array(vals).reshape([8, 5]).T;
        const expected = runNumPy(`
b = ${pyArr(vals, 'float64')}.reshape(8, 5).T
result = np.histogram_bin_edges(b, bins='${s}')
`);
        const r = np.histogram_bin_edges(a, s);
        expect(arraysClose(r.toArray(), expected.value, 1e-9, 1e-9)).toBe(true);
      });
    }
    it('integer data never uses a width below one', () => {
      const vals = [0, 1, 1, 2, 2, 2, 3, 3, 4];
      for (const s of STRATEGIES) {
        expectEdges(
          np.histogram_bin_edges(np.array(vals, 'int64'), s),
          `np.histogram_bin_edges(a, bins='${s}')`,
          vals,
          'int64',
        );
      }
    });
  });

  describe('integer bins', () => {
    for (const bins of [1, 2, 5, 10, 33]) {
      for (const [name, vals] of Object.entries(DATASETS)) {
        it(`${bins} bins on ${name}`, () => {
          const dtype = name === 'ints' ? 'int32' : 'float64';
          expectEdges(
            np.histogram_bin_edges(np.array(vals, dtype as never), bins),
            `np.histogram_bin_edges(a, bins=${bins})`,
            vals,
            dtype,
          );
        });
      }
    }
    it('too many bins for the data range', () => {
      expect(() => np.histogram_bin_edges(np.array([0, 1]), 10, [0, 5e-324])).toThrow(
        /Too many bins for data range/,
      );
      expect(() =>
        runNumPy('result = np.histogram_bin_edges(np.array([0., 1]), bins=10, range=(0, 5e-324))'),
      ).toThrow(/Too many bins/);
    });
    it('stone places values that land on a computed edge like NumPy', () => {
      const vals = Array.from({ length: 11 }, (_, i) => i / 10);
      expectEdges(
        np.histogram_bin_edges(np.array(vals), 'stone'),
        "np.histogram_bin_edges(a, bins='stone')",
        vals,
        'float64',
      );
      const dense = Array.from({ length: 300 }, (_, i) => (i % 37) / 37 + (i % 7) / 70);
      expectEdges(
        np.histogram_bin_edges(np.array(dense), 'stone'),
        "np.histogram_bin_edges(a, bins='stone')",
        dense,
        'float64',
      );
    });
    it('documented example', () => {
      expectEdges(
        np.histogram_bin_edges(np.array([1, 2, 3, 4, 5]), 5),
        'np.histogram_bin_edges(a, bins=5)',
        [1, 2, 3, 4, 5],
        'float64',
      );
    });
    it('default bins, range and constant arrays', () => {
      const vals = DATASETS.uniform!;
      expectEdges(
        np.histogram_bin_edges(np.array(vals)),
        'np.histogram_bin_edges(a)',
        vals,
        'float64',
      );
      expectEdges(
        np.histogram_bin_edges(np.array(vals), 7, [-3, 11]),
        'np.histogram_bin_edges(a, bins=7, range=(-3, 11))',
        vals,
        'float64',
      );
      for (const v of [[4, 4, 4], [], [0], [-2.5]]) {
        expectEdges(
          np.histogram_bin_edges(np.array(v), 4),
          'np.histogram_bin_edges(a, bins=4)',
          v,
          'float64',
        );
      }
    });
    it('float32 and float16', () => {
      const vals = lcg(30, 5, 7);
      expectEdges(
        np.histogram_bin_edges(np.array(vals, 'float32'), 9),
        'np.histogram_bin_edges(a, bins=9)',
        vals,
        'float32',
        1e-6,
      );
      expectEdges(
        np.histogram_bin_edges(np.array(vals, 'float32'), 9, [0, 1]),
        'np.histogram_bin_edges(a, bins=9, range=(0, 1))',
        vals,
        'float32',
        1e-6,
      );
      expectEdges(
        np.histogram_bin_edges(np.array(vals, 'float16'), 4),
        'np.histogram_bin_edges(a, bins=4)',
        vals,
        'float16',
        5e-3,
      );
    });
  });

  describe('explicit edges', () => {
    it('returns the edges as given', () => {
      const vals = DATASETS.uniform!;
      expectEdges(
        np.histogram_bin_edges(np.array(vals), np.array([0, 1, 4, 9])),
        'np.histogram_bin_edges(a, bins=np.array([0., 1, 4, 9]))',
        vals,
        'float64',
      );
      expectEdges(
        np.histogram_bin_edges(np.array(vals), np.array([0, 1, 4, 9], 'int32')),
        "np.histogram_bin_edges(a, bins=np.array([0, 1, 4, 9], dtype='int32'))",
        vals,
        'float64',
      );
    });
    it('does not alias the caller array', () => {
      const bins = np.array([0, 1, 2]);
      const e = np.histogram_bin_edges(np.array([1, 2]), bins);
      e.set([0], 99);
      expect(bins.get([0])).toBe(0);
    });
  });

  describe('weights', () => {
    it('are accepted for integer and explicit bins and ignored', () => {
      const vals = DATASETS.small!;
      const w = np.array([1, 2, 3, 4]);
      expectEdges(
        np.histogram_bin_edges(np.array(vals), 3, undefined, w),
        'np.histogram_bin_edges(a, bins=3, weights=np.array([1., 2, 3, 4]))',
        vals,
        'float64',
      );
      expectEdges(
        np.histogram_bin_edges(np.array(vals), np.array([0, 5, 9]), undefined, w),
        'np.histogram_bin_edges(a, bins=np.array([0., 5, 9]), weights=np.array([1., 2, 3, 4]))',
        vals,
        'float64',
      );
    });
    it('are rejected for strategies, as in NumPy', () => {
      const a = np.array([1, 2, 3, 4]);
      const w = np.array([1, 1, 1, 1]);
      for (const s of STRATEGIES) {
        expect(() => np.histogram_bin_edges(a, s, undefined, w)).toThrow(TypeError);
        expect(() =>
          runNumPy(
            `result = np.histogram_bin_edges(np.array([1., 2, 3, 4]), bins='${s}', weights=np.ones(4))`,
          ),
        ).toThrow(/not supported for weighted data/);
      }
    });
    it('must match the data shape', () => {
      const a = np.array([1, 2, 3, 4]);
      expect(() => np.histogram_bin_edges(a, 3, undefined, np.array([1, 2]))).toThrow(/same shape/);
    });
  });

  describe('errors', () => {
    const a = np.array([1, 2, 3]);
    it('non-positive and non-integer bin counts', () => {
      expect(() => np.histogram_bin_edges(a, 0)).toThrow(/positive/);
      expect(() => np.histogram_bin_edges(a, -2)).toThrow(/positive/);
      expect(() => np.histogram_bin_edges(a, 2.5)).toThrow(TypeError);
      expect(() => runNumPy('result = np.histogram_bin_edges(np.arange(3.), bins=0)')).toThrow(
        /positive/,
      );
    });
    it('unknown strategy', () => {
      expect(() => np.histogram_bin_edges(a, 'nope' as never)).toThrow(/not a valid estimator/);
    });
    it('bad ranges', () => {
      expect(() => np.histogram_bin_edges(a, 3, [2, 1])).toThrow(/max must be larger/);
      expect(() => np.histogram_bin_edges(a, 3, [0, Infinity])).toThrow(/not finite/);
      expect(() => np.histogram_bin_edges(a, 3, [Number.NaN, 1])).toThrow(/not finite/);
    });
    it('NaN or infinite data without a range', () => {
      expect(() => np.histogram_bin_edges(np.array([1, Number.NaN]), 3)).toThrow(/not finite/);
      expect(() => np.histogram_bin_edges(np.array([1, Infinity]), 3)).toThrow(/not finite/);
      expect(() => np.histogram_bin_edges(np.array([1, Number.NaN]), 'auto')).toThrow(/not finite/);
    });
    it('NaN data with a range is dropped', () => {
      const vals = [1, Number.NaN, 2, 3, 8];
      expectEdges(
        np.histogram_bin_edges(np.array(vals), 'sturges', [0, 10]),
        "np.histogram_bin_edges(a, bins='sturges', range=(0, 10))",
        vals,
        'float64',
      );
    });
    it('bad explicit edges', () => {
      expect(() => np.histogram_bin_edges(a, np.array([3, 1, 2]))).toThrow(/monotonically/);
      expect(() =>
        np.histogram_bin_edges(
          a,
          np.array([
            [1, 2],
            [3, 4],
          ]),
        ),
      ).toThrow(/1d/);
    });
    it('complex input', () => {
      expect(() => np.histogram_bin_edges(np.array([1, 2]).astype('complex128'))).toThrow(/real/);
    });
  });
});

describe('NumPy Validation: histogram edges and counts', () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  function expectHistogram(
    vals: number[],
    bins: number | string | number[],
    opts: { range?: [number, number]; density?: boolean; weights?: number[]; dtype?: string } = {},
  ): void {
    const dtype = opts.dtype ?? 'float64';
    const a = np.array(vals, dtype as never);
    const jsBins = Array.isArray(bins) ? np.array(bins) : bins;
    const [hist, edges] = np.histogram(
      a,
      jsBins as never,
      opts.range,
      opts.density,
      opts.weights ? np.array(opts.weights) : undefined,
    );
    const pyBins = Array.isArray(bins)
      ? `np.array([${bins.join(', ')}])`
      : typeof bins === 'string'
        ? `'${bins}'`
        : String(bins);
    const expected = runNumPy(`
a = ${pyArr(vals, dtype)}
h, e = np.histogram(a, bins=${pyBins}, range=${opts.range ? `(${opts.range.join(', ')})` : 'None'}, density=${opts.density ? 'True' : 'False'}, weights=${opts.weights ? `np.array([${opts.weights.join(', ')}])` : 'None'})
result = np.concatenate([h.astype(float), e.astype(float)])
`);
    const got = [...(hist.toArray() as number[]), ...(edges.toArray() as number[])];
    const tol = dtype === 'float32' ? 1e-5 : 1e-9;
    expect(arraysClose(got, expected.value, tol, tol)).toBe(true);
  }

  for (const s of STRATEGIES) {
    for (const [name, vals] of Object.entries(DATASETS)) {
      it(`histogram ${s} on ${name}`, () => {
        expectHistogram(vals, s, { dtype: name === 'ints' ? 'int32' : 'float64' });
      });
    }
    it(`histogram ${s} with range, density and float32`, () => {
      expectHistogram(DATASETS.skewed!, s, { range: [-30, 30], density: true });
      expectHistogram(DATASETS.uniform!, s, { dtype: 'float32' });
    });
  }

  it('histogram with integer bins, range, weights and density', () => {
    const vals = DATASETS.uniform!;
    expectHistogram(vals, 7);
    expectHistogram(vals, 7, { range: [0, 5] });
    expectHistogram(vals, 7, { weights: vals.map((_, i) => i / 3) });
    expectHistogram(vals, 7, { density: true, weights: vals.map((_, i) => 1 + i) });
    expectHistogram([3, 3, 3], 4);
    expectHistogram([], 4);
  });

  it('histogram with explicit edges, including a value on the last edge', () => {
    expectHistogram([0, 1, 2, 3, 4, 5, 9, 10, 11, -1], [0, 2, 4, 10]);
    expectHistogram([0.5, 1.5, 2.5], [0, 1, 1, 3]);
  });

  it('histogram on a non-contiguous view', () => {
    const vals = lcg(40, 4, 10);
    const [hist, edges] = np.histogram(np.array(vals).reshape([8, 5]).T, 6);
    const expected = runNumPy(`
b = ${pyArr(vals, 'float64')}.reshape(8, 5).T
h, e = np.histogram(b, bins=6)
result = np.concatenate([h.astype(float), e])
`);
    expect(
      arraysClose(
        [...(hist.toArray() as number[]), ...(edges.toArray() as number[])],
        expected.value,
        1e-9,
        1e-9,
      ),
    ).toBe(true);
  });

  it('histogram rejects what NumPy rejects', () => {
    expect(() => np.histogram(np.array([1, Number.NaN]), 3)).toThrow(/not finite/);
    expect(() =>
      np.histogram(np.array([1, 2]), 'sqrt', undefined, false, np.array([1, 1])),
    ).toThrow(TypeError);
    expect(() => np.histogram(np.array([1, 2]), 0)).toThrow(/positive/);
  });

  describe('histogram2d and histogramdd', () => {
    const xs = lcg(60, 6, 10);
    const ys = lcg(60, 7, 4).map((v) => v * v);

    function check2d(
      bins: string,
      opts: { range?: string; dtype?: string; density?: boolean } = {},
    ) {
      const dtype = opts.dtype ?? 'float64';
      const jsBins = JSON.parse(bins) as number | [number, number];
      const jsRange = opts.range
        ? (JSON.parse(opts.range) as [[number, number], [number, number]])
        : undefined;
      const [h, xe, ye] = np.histogram2d(
        np.array(xs, dtype as never),
        np.array(ys, dtype as never),
        jsBins,
        jsRange,
        opts.density,
      );
      const expected = runNumPy(`
x = ${pyArr(xs, dtype)}
y = ${pyArr(ys, dtype)}
h, xe, ye = np.histogram2d(x, y, bins=${bins}, range=${opts.range ?? 'None'}, density=${opts.density ? 'True' : 'False'})
result = np.concatenate([h.ravel().astype(float), xe.astype(float), ye.astype(float)])
`);
      const tol = dtype === 'float32' ? 1e-5 : 1e-9;
      expect(
        arraysClose(
          [
            ...(h.toArray() as number[][]).flat(),
            ...(xe.toArray() as number[]),
            ...(ye.toArray() as number[]),
          ],
          expected.value,
          tol,
          tol,
        ),
      ).toBe(true);
      const exp = runNumPy(`
x = ${pyArr(xs, dtype)}
y = ${pyArr(ys, dtype)}
result = np.histogram2d(x, y, bins=${bins}, range=${opts.range ?? 'None'})[1]
`);
      expect(xe.dtype).toBe(exp.dtype);
    }

    it('integer bins', () => {
      check2d('4');
      check2d('[3, 5]');
      check2d('[3, 5]', { density: true });
    });
    it('with range', () => {
      check2d('4', { range: '[[-5, 5], [0, 20]]' });
    });
    it('float32 data', () => {
      check2d('5', { dtype: 'float32' });
    });
    it('constant data and NaN', () => {
      const [h, xe] = np.histogram2d(np.array([2, 2, 2]), np.array([1, 1, 1]), 2);
      const expected = runNumPy(
        'h, xe, ye = np.histogram2d(np.array([2., 2, 2]), np.array([1., 1, 1]), bins=2)\nresult = np.concatenate([h.ravel(), xe])',
      );
      expect(
        arraysClose(
          [...(h.toArray() as number[][]).flat(), ...(xe.toArray() as number[])],
          expected.value,
        ),
      ).toBe(true);
      expect(() => np.histogram2d(np.array([1, Number.NaN]), np.array([1, 2]), 2)).toThrow(
        /not finite/,
      );
    });
    it('explicit edges', () => {
      const [h, xe, ye] = np.histogram2d(np.array(xs), np.array(ys), [
        np.array([-10, -2, 0, 3, 10]),
        np.array([0, 1, 5, 40]),
      ]);
      const expected = runNumPy(`
x = ${pyArr(xs, 'float64')}
y = ${pyArr(ys, 'float64')}
h, xe, ye = np.histogram2d(x, y, bins=[np.array([-10., -2, 0, 3, 10]), np.array([0., 1, 5, 40])])
result = np.concatenate([h.ravel(), xe, ye])
`);
      expect(
        arraysClose(
          [
            ...(h.toArray() as number[][]).flat(),
            ...(xe.toArray() as number[]),
            ...(ye.toArray() as number[]),
          ],
          expected.value,
        ),
      ).toBe(true);
    });
    it('non-increasing edges are rejected', () => {
      expect(() =>
        np.histogram2d(np.array([1, 2]), np.array([1, 2]), [np.array([2, 1]), np.array([0, 3])]),
      ).toThrow(/monotonically/);
    });
    it('views and weights', () => {
      const base = np.array(xs).reshape([6, 10]);
      const x = base.T.reshape([60]);
      const w = np.array(xs.map((_, i) => i + 1));
      const [h, xe] = np.histogram2d(x, np.array(ys), 3, undefined, false, w);
      const expected = runNumPy(`
x = ${pyArr(xs, 'float64')}.reshape(6, 10).T.reshape(60)
y = ${pyArr(ys, 'float64')}
h, xe, ye = np.histogram2d(x, y, bins=3, weights=np.arange(1., 61))
result = np.concatenate([h.ravel(), xe])
`);
      expect(
        arraysClose(
          [...(h.toArray() as number[][]).flat(), ...(xe.toArray() as number[])],
          expected.value,
        ),
      ).toBe(true);
    });

    it('histogramdd edges, counts and density', () => {
      const sample = xs.slice(0, 30).flatMap((v, i) => [v, ys[i]!, xs[i + 30]!]);
      for (const [bins, range, density] of [
        [3, undefined, false],
        [[2, 3, 4], undefined, false],
        [
          3,
          [
            [-5, 5],
            [0, 20],
            [-5, 5],
          ],
          false,
        ],
        [[2, 2, 2], undefined, true],
      ] as [number | number[], [number, number][] | undefined, boolean][]) {
        const [h, edges] = np.histogramdd(np.array(sample).reshape([30, 3]), bins, range, density);
        const expected = runNumPy(`
s = ${pyArr(sample, 'float64')}.reshape(30, 3)
h, e = np.histogramdd(s, bins=${JSON.stringify(bins)}, range=${range ? JSON.stringify(range) : 'None'}, density=${density ? 'True' : 'False'})
result = np.concatenate([h.ravel()] + [x for x in e])
`);
        const got = [
          ...(h.toArray() as number[][][]).flat(2),
          ...edges.flatMap((e) => e.toArray() as number[]),
        ];
        expect(arraysClose(got, expected.value, 1e-9, 1e-9)).toBe(true);
      }
    });
    it('histogramdd edges when the step underflows', () => {
      const [, edges] = np.histogramdd(np.array([0, 1]), 10, [[0, 5e-324]]);
      const expected = runNumPy(
        'h, e = np.histogramdd(np.array([0., 1]), bins=10, range=[(0, 5e-324)])\nresult = e[0]',
      );
      expect(arraysClose(edges[0]!.toArray(), expected.value, 0, 1e-300)).toBe(true);
    });
    it('histogramdd float32, 1-D sample, constant column', () => {
      const [h, edges] = np.histogramdd(np.array(xs.slice(0, 20), 'float32'), 4);
      const expected = runNumPy(`
s = ${pyArr(xs.slice(0, 20), 'float32')}
h, e = np.histogramdd(s, bins=4)
result = np.concatenate([h.ravel(), e[0].astype(float)])
_result_orig = e[0]
`);
      expect(
        arraysClose(
          [...(h.toArray() as number[]), ...(edges[0]!.toArray() as number[])],
          expected.value,
          1e-5,
          1e-5,
        ),
      ).toBe(true);
      expect(edges[0]!.dtype).toBe('float32');
      const [h2, e2] = np.histogramdd(
        np.array([
          [1, 5],
          [1, 6],
          [1, 7],
        ]),
        2,
      );
      const exp2 = runNumPy(
        'h, e = np.histogramdd(np.array([[1., 5], [1, 6], [1, 7]]), bins=2)\nresult = np.concatenate([h.ravel()] + list(e))',
      );
      expect(
        arraysClose(
          [...(h2.toArray() as number[][]).flat(), ...e2.flatMap((e) => e.toArray() as number[])],
          exp2.value,
        ),
      ).toBe(true);
    });
  });
});

/**
 * Unit tests for the axis argument of reductions: invalid values throw a TypeError instead of being
 * read as an axis, null means no axis, and valid calls are unchanged.
 */

import { describe, expect, it } from 'vitest';
import type { NDArray } from '../../src';
import * as np from '../../src';
import { wasmConfig } from '../../src/common/wasm/config';

const m = () =>
  np.array([
    [1, 2, 3],
    [4, 5, 6],
  ]);

// Every reduction that accepts an axis, as [name, call]. The call receives the axis argument.
const REDUCTIONS: [string, (a: NDArray, axis: unknown) => unknown, boolean][] = [
  ['sum', (a, ax) => np.sum(a, ax as number), true],
  ['prod', (a, ax) => np.prod(a, ax as number), true],
  ['mean', (a, ax) => np.mean(a, ax as number), true],
  ['std', (a, ax) => np.std(a, ax as number), true],
  ['variance', (a, ax) => np.variance(a, ax as number), true],
  ['min', (a, ax) => np.min(a, ax as number), true],
  ['max', (a, ax) => np.max(a, ax as number), true],
  ['ptp', (a, ax) => np.ptp(a, ax as number), true],
  ['any', (a, ax) => np.any(a, ax as number), true],
  ['all', (a, ax) => np.all(a, ax as number), true],
  ['median', (a, ax) => np.median(a, ax as number), true],
  ['count_nonzero', (a, ax) => np.count_nonzero(a, ax as number), true],
  ['nansum', (a, ax) => np.nansum(a, ax as number), true],
  ['nanprod', (a, ax) => np.nanprod(a, ax as number), true],
  ['nanmean', (a, ax) => np.nanmean(a, ax as number), true],
  ['nanstd', (a, ax) => np.nanstd(a, ax as number), true],
  ['nanvar', (a, ax) => np.nanvar(a, ax as number), true],
  ['nanmin', (a, ax) => np.nanmin(a, ax as number), true],
  ['nanmax', (a, ax) => np.nanmax(a, ax as number), true],
  ['argmax', (a, ax) => np.argmax(a, ax as number), false],
  ['argmin', (a, ax) => np.argmin(a, ax as number), false],
  ['nanargmax', (a, ax) => np.nanargmax(a, ax as number), false],
  ['nanargmin', (a, ax) => np.nanargmin(a, ax as number), false],
  ['nanmedian', (a, ax) => np.nanmedian(a, ax as number), false],
  ['percentile', (a, ax) => np.percentile(a, 50, ax as number), false],
  ['quantile', (a, ax) => np.quantile(a, 0.5, ax as number), false],
  ['nanpercentile', (a, ax) => np.nanpercentile(a, 50, ax as number), false],
  ['nanquantile', (a, ax) => np.nanquantile(a, 0.5, ax as number), false],
  ['average', (a, ax) => np.average(a, ax as number), false],
  ['cumsum', (a, ax) => np.cumsum(a, ax as number), false],
  ['cumprod', (a, ax) => np.cumprod(a, ax as number), false],
  ['nancumsum', (a, ax) => np.nancumsum(a, ax as number), false],
  ['nancumprod', (a, ax) => np.nancumprod(a, ax as number), false],
];

describe('reduction axis argument', () => {
  describe('rejects an argument that is not an axis', () => {
    for (const [name, call, tuple] of REDUCTIONS) {
      it(`${name} rejects an options object`, () => {
        expect(() => call(m(), { axis: 0 })).toThrow(TypeError);
        expect(() => call(m(), { axis: 0 })).toThrow(/axis must be an integer.*got an object/);
      });

      it(`${name} rejects a string, a boolean and a fractional axis`, () => {
        expect(() => call(m(), '0')).toThrow(/got a string/);
        expect(() => call(m(), true)).toThrow(/got a boolean/);
        expect(() => call(m(), 0.5)).toThrow(/non-integer number 0.5/);
        expect(() => call(m(), Number.NaN)).toThrow(TypeError);
      });

      it(`${name} ${tuple ? 'rejects an array with a non-integer' : 'rejects an array'}`, () => {
        if (tuple) {
          expect(() => call(m(), [0, 1.5])).toThrow(/array of integers, got an array containing/);
          expect(() => call(m(), ['0'])).toThrow(TypeError);
        } else {
          expect(() => call(m(), [0])).toThrow(/axis must be an integer, got an array/);
        }
      });
    }

    it('the message names both accepted forms where a tuple is allowed', () => {
      expect(() => np.sum(m(), { axis: 0 } as unknown as number)).toThrow(
        'axis must be an integer or an array of integers, got an object',
      );
    });

    it('the same guard applies to the NDArray methods', () => {
      const bad = { axis: 0 } as unknown as number;
      expect(() => m().sum(bad)).toThrow(TypeError);
      expect(() => m().mean(bad)).toThrow(TypeError);
      expect(() => m().std(bad)).toThrow(TypeError);
      expect(() => m().argmax(bad)).toThrow(TypeError);
      expect(() => m().percentile(50, bad)).toThrow(TypeError);
      expect(() => m().cumsum(bad)).toThrow(TypeError);
    });
  });

  describe('null is no axis', () => {
    for (const [name, call] of REDUCTIONS) {
      it(`${name}(null) equals ${name}(undefined)`, () => {
        const viaNull = call(m(), null);
        const viaUndefined = call(m(), undefined);
        const plain = (r: unknown) => (r instanceof np.NDArray ? r.toArray() : r);
        expect(plain(viaNull)).toEqual(plain(viaUndefined));
      });
    }

    it('null axis with keepdims keeps every dimension', () => {
      const r = np.sum(m(), null as unknown as undefined, true) as NDArray;
      expect(Array.from(r.shape)).toEqual([1, 1]);
      expect(r.toArray()).toEqual([[21]]);
    });
  });

  describe('valid calls are unchanged', () => {
    it('integer axes, negative axes and tuples', () => {
      expect((np.sum(m(), 0) as NDArray).toArray()).toEqual([5, 7, 9]);
      expect((np.sum(m(), 1) as NDArray).toArray()).toEqual([6, 15]);
      expect((np.sum(m(), -1) as NDArray).toArray()).toEqual([6, 15]);
      expect(np.sum(m(), [0, 1])).toBe(21);
      expect((np.mean(m(), 1) as NDArray).toArray()).toEqual([2, 5]);
      expect((np.max(m(), [1]) as NDArray).toArray()).toEqual([3, 6]);
      expect(np.argmax(m(), 1) as NDArray).toBeInstanceOf(np.NDArray);
      expect(np.count_nonzero(m(), [0, 1])).toBe(6);
    });

    it('an empty tuple reduces nothing', () => {
      expect((np.sum(m(), []) as NDArray).toArray()).toEqual(m().toArray());
    });

    it('the NDArray methods accept the same forms', () => {
      expect((m().sum(0) as NDArray).toArray()).toEqual([5, 7, 9]);
      expect((m().mean([0, 1], true) as NDArray).toArray()).toEqual([[3.5]]);
    });
  });

  describe('keepdims without an axis', () => {
    it('returns an array of ones-shape instead of a scalar', () => {
      const a3 = np.arange(24).reshape(2, 3, 4);
      const r = np.sum(a3, undefined, true) as NDArray;
      expect(Array.from(r.shape)).toEqual([1, 1, 1]);
      expect(r.toArray()).toEqual([[[276]]]);
    });

    it('works on a non-contiguous view', () => {
      const t = m().T;
      const r = np.max(t, undefined, true) as NDArray;
      expect(Array.from(r.shape)).toEqual([1, 1]);
      expect(r.toArray()).toEqual([[6]]);
    });

    it('keeps the reduced dtype', () => {
      const i8 = np.array([1, 2, 3], 'int8');
      expect((np.sum(i8, undefined, true) as NDArray).dtype).toBe('int64');
      expect((np.max(i8, undefined, true) as NDArray).dtype).toBe('int8');
      expect((np.mean(i8, undefined, true) as NDArray).dtype).toBe('float64');
      expect((np.argmax(i8, undefined, true) as NDArray).dtype).toBe('int32');
    });

    it('a 0-d input has nothing to keep and returns the scalar', () => {
      const zero = np.array([5]).reshape([]);
      expect(np.sum(zero, undefined, true)).toBe(5);
    });

    it('multi-axis keepdims over every axis keeps the full rank', () => {
      const r = np.std(m(), [0, 1], 0, true) as NDArray;
      expect(Array.from(r.shape)).toEqual([1, 1]);
    });
  });

  describe('joint multi-axis reductions', () => {
    it('rejects duplicate and out-of-range axes', () => {
      expect(() => np.std(m(), [0, 0])).toThrow(/duplicate value in 'axis'/);
      expect(() => np.std(m(), [0, 2])).toThrow(/axis 2 is out of bounds for array of dimension 2/);
      expect(() => np.std(m(), [-3])).toThrow(/out of bounds/);
    });
  });

  describe('keepdims with an explicit axis on a 1-D input', () => {
    it('a narrow-int sum or prod keeps the 64-bit accumulator dtype', () => {
      const i8 = np.array([1, 2, 3], 'int8');
      const s = np.sum(i8, 0, true) as NDArray;
      expect(s.dtype).toBe('int64');
      expect(Array.from(s.shape)).toEqual([1]);
      expect(s.toArray()).toEqual([6n]);
      expect((np.prod(i8, 0, true) as NDArray).toArray()).toEqual([6n]);
    });

    it('nanmin and nanmax keep a float32 dtype and values', () => {
      const f32 = np.array([3, 1, 2], 'float32');
      const mn = np.nanmin(f32, 0, true) as NDArray;
      const mx = np.nanmax(f32, 0, true) as NDArray;
      expect(mn.dtype).toBe('float32');
      expect(mx.dtype).toBe('float32');
      expect(mn.toArray()).toEqual([1]);
      expect(mx.toArray()).toEqual([3]);
      const m2 = np.nanmin(
        np.array(
          [
            [3, 1],
            [2, 5],
          ],
          'float32',
        ),
        0,
        true,
      ) as NDArray;
      expect(m2.dtype).toBe('float32');
      expect(m2.toArray()).toEqual([[2, 1]]);
    });

    it('a complex median returns an array', () => {
      const c = np.array([1, 2, 3], 'complex64');
      const r = np.median(c, 0, true) as NDArray;
      expect(Array.from(r.shape)).toEqual([1]);
      expect(np.median(c, undefined, true)).toBeInstanceOf(np.NDArray);
      expect(np.nanmedian(c, undefined, true)).toBeInstanceOf(np.NDArray);
    });

    it('argmax with keepdims and no axis uses the int32 index dtype', () => {
      const r = np.argmax(np.arange(6).reshape(2, 3), undefined, true) as NDArray;
      expect(r.dtype).toBe('int32');
      expect(r.toArray()).toEqual([[5]]);
    });
  });

  describe('ptp over several axes', () => {
    it('returns a number over every axis', () => {
      expect(np.ptp(m(), [0, 1])).toBe(5);
    });

    it('handles bigint results', () => {
      const a = np.array(
        [
          [1, 5],
          [9, 2],
        ],
        'int64',
      );
      expect(np.ptp(a, [0, 1])).toBe(8n);
    });

    it('keeps an array result when an axis remains', () => {
      const a = np.arange(24).reshape(2, 3, 4);
      expect((np.ptp(a, [0, 1]) as NDArray).toArray()).toEqual([20, 20, 20, 20]);
    });

    it('complex input over every axis matches the no-axis result', () => {
      const c = np.array(
        [
          [1, 2],
          [3, 4],
        ],
        'complex128',
      );
      expect(np.ptp(c, [0, 1])).toEqual(np.ptp(c));
      expect(np.ptp(c, [0, 1])).toEqual(new np.Complex(3, 0));
    });

    it('wraps narrow integers over every axis as the no-axis path does', () => {
      const a = np.array([[-100, 100]], 'int8');
      expect(np.ptp(a, [0, 1])).toBe(-56);
    });
  });

  describe('a tuple naming every axis', () => {
    const a3 = () => np.arange(24).astype('int32').reshape(2, 3, 4);

    it('returns a number for narrow input, as the no-axis call does', () => {
      expect(np.sum(a3(), [0, 1, 2])).toBe(276);
      expect(np.sum(a3(), [2, 0, -2])).toBe(276);
      expect(np.prod(np.array([[1, 2, 3]], 'int8'), [0, 1])).toBe(6);
      expect(np.nansum(np.array([[1, 2, 3]], 'uint8'), [0, 1])).toBe(6);
      expect(np.nanprod(np.array([[1, 2, 3]], 'uint8'), [0, 1])).toBe(6);
      expect(
        np.sum(
          np.array([
            [true, false],
            [true, true],
          ]),
          [0, 1],
        ),
      ).toBe(3);
      expect(np.max(a3(), [0, 1, 2])).toBe(23);
      expect(np.mean(a3(), [0, 1, 2])).toBe(11.5);
    });

    it('returns a bigint only for 64-bit integer input', () => {
      expect(np.sum(a3().astype('int64'), [0, 1, 2])).toBe(276n);
      expect(np.sum(a3().astype('uint64'), [0, 1, 2])).toBe(276n);
    });

    it('keepdims gives the same dtype and shape as the no-axis call', () => {
      const t = np.sum(a3(), [0, 1, 2], true) as NDArray;
      const n = np.sum(a3(), undefined, true) as NDArray;
      expect(t.dtype).toBe(n.dtype);
      expect(Array.from(t.shape)).toEqual([1, 1, 1]);
      expect(t.toArray()).toEqual(n.toArray());
    });

    it('a partial tuple keeps the single-axis result dtype', () => {
      const t = np.sum(a3(), [0, 1]) as NDArray;
      expect(t.dtype).toBe((np.sum(a3(), 0) as NDArray).dtype);
      expect(t.toArray()).toEqual([60n, 66n, 72n, 78n]);
      expect((np.max(a3(), [0, 2]) as NDArray).dtype).toBe('int32');
      expect((np.mean(a3(), [0, 1]) as NDArray).dtype).toBe('float64');
    });
  });

  describe('duplicate axes in a tuple', () => {
    const a3 = () => np.arange(24).reshape(2, 3, 4);
    const TUPLE_REDUCTIONS: [string, (a: NDArray, axis: number[]) => unknown][] = [
      ['sum', (a, ax) => np.sum(a, ax)],
      ['prod', (a, ax) => np.prod(a, ax)],
      ['mean', (a, ax) => np.mean(a, ax)],
      ['max', (a, ax) => np.max(a, ax)],
      ['min', (a, ax) => np.min(a, ax)],
      ['ptp', (a, ax) => np.ptp(a, ax)],
      ['all', (a, ax) => np.all(a, ax)],
      ['any', (a, ax) => np.any(a, ax)],
      ['nansum', (a, ax) => np.nansum(a, ax)],
      ['nanprod', (a, ax) => np.nanprod(a, ax)],
      ['nanmin', (a, ax) => np.nanmin(a, ax)],
      ['nanmax', (a, ax) => np.nanmax(a, ax)],
      ['count_nonzero', (a, ax) => np.count_nonzero(a, ax)],
      ['std', (a, ax) => np.std(a, ax)],
      ['nanmean', (a, ax) => np.nanmean(a, ax)],
    ];
    for (const [name, call] of TUPLE_REDUCTIONS) {
      it(`${name} rejects a repeated axis, including one given as a negative`, () => {
        expect(() => call(a3(), [0, 0])).toThrow("duplicate value in 'axis'");
        expect(() => call(a3(), [1, 1])).toThrow("duplicate value in 'axis'");
        expect(() => call(a3(), [0, -3])).toThrow("duplicate value in 'axis'");
        expect(() => call(a3(), [0, 1, 2, 2])).toThrow("duplicate value in 'axis'");
      });

      it(`${name} reports an out-of-bounds axis ahead of a duplicate`, () => {
        expect(() => call(a3(), [0, 0, 5])).toThrow(
          /axis 5 is out of bounds for array of dimension 3/,
        );
      });
    }
  });

  describe('average with an axis and returned', () => {
    it('a negative axis counts the elements of that axis', () => {
      const [avg, sw] = np.average(m(), -1, undefined, false, true) as [NDArray, NDArray];
      expect(avg.toArray()).toEqual([2, 5]);
      expect(sw.toArray()).toEqual([3, 3]);
    });

    it('keepdims keeps the averaged axis as size 1', () => {
      const [avg, sw] = np.average(m(), 1, undefined, true, true) as [NDArray, NDArray];
      expect(Array.from(avg.shape)).toEqual([2, 1]);
      expect(Array.from(sw.shape)).toEqual([2, 1]);
    });
  });

  describe('all and any with initial', () => {
    it('a false initial turns all into false', () => {
      expect(np.all(m(), undefined, false, { initial: false })).toBe(false);
      expect(np.all(m(), undefined, false, { initial: true })).toBe(true);
    });

    it('a true initial turns any into true', () => {
      const z = np.zeros([2, 2]);
      expect(np.any(z, undefined, false, { initial: true })).toBe(true);
      expect(np.any(z, undefined, false, { initial: false })).toBe(false);
    });
  });

  describe('quantile with an array of q', () => {
    type QFn = (a: NDArray, q: NDArray | number, axis?: number, keepdims?: boolean) => unknown;
    const FNS: [string, QFn, number][] = [
      ['quantile', np.quantile as QFn, 1],
      ['percentile', np.percentile as QFn, 100],
      ['nanquantile', np.nanquantile as QFn, 1],
      ['nanpercentile', np.nanpercentile as QFn, 100],
    ];
    const Q = [0, 0.13, 1 / 3, 0.5, 0.77, 0.999, 1];

    // Values spread over a wide range, with NaNs and one all-NaN lane along the last axis.
    const data = (dtype: string, shape: number[]) => {
      const size = shape.reduce((x, y) => x * y, 1);
      const vals = Array.from({ length: size }, (_, i) => ((i * 7919) % 101) - 50 + (i % 7) / 9);
      if (dtype.startsWith('float')) {
        vals[1] = NaN;
        vals[size - 2] = NaN;
        for (let i = 0; i < shape[shape.length - 1]! && shape.length > 1; i++) vals[i] = NaN;
      }
      return np
        .array(vals)
        .astype(dtype as 'float64')
        .reshape(shape);
    };

    // The array-q result must equal, bit for bit, one scalar-q reduction per q stacked in front.
    const expectMatchesPerQ = (
      f: QFn,
      a: NDArray,
      q: NDArray,
      axis: number | undefined,
      keepdims: boolean,
    ) => {
      const r = f(a, q, axis, keepdims) as NDArray;
      const widened =
        r.dtype === 'float64' && a.dtype.startsWith('float') ? a.astype('float64') : a;
      const flat = r.flatten().toArray() as number[];
      const qs = q.flatten().toArray() as number[];
      const inner = flat.length / qs.length;
      qs.forEach((qv, i) => {
        const one = f(widened, qv, axis, keepdims);
        const vals =
          typeof one === 'number' ? [one] : ((one as NDArray).flatten().toArray() as number[]);
        const stored = np.array(vals).astype(r.dtype).toArray() as number[];
        expect(flat.slice(i * inner, (i + 1) * inner)).toEqual(stored);
      });
      return r;
    };

    for (const multiplier of [1, 0, Infinity]) {
      describe(`with the WASM threshold multiplier at ${multiplier}`, () => {
        for (const [name, f, scale] of FNS) {
          it(`${name} matches one reduction per q`, () => {
            const saved = wasmConfig.thresholdMultiplier;
            wasmConfig.thresholdMultiplier = multiplier;
            try {
              for (const dtype of ['float64', 'float32', 'float16', 'int32', 'int64']) {
                for (const shape of [[9], [5, 7], [4, 6, 5]]) {
                  const base = data(dtype, shape);
                  for (const a of shape.length > 1 ? [base, base.transpose()] : [base]) {
                    for (const qDtype of ['float64', 'float32'] as const) {
                      const q = np.array(Q.map((x) => x * scale)).astype(qDtype);
                      for (const axis of [undefined, 0, -1]) {
                        for (const keepdims of [false, true]) {
                          expectMatchesPerQ(f, a, q, axis, keepdims);
                        }
                      }
                    }
                  }
                }
              }
            } finally {
              wasmConfig.thresholdMultiplier = saved;
            }
          });
        }
      });
    }

    it('sorts the data once however many q values there are', () => {
      const a = np.arange(2000).reshape(40, 50).astype('float64');
      const q = np.linspace(0, 100, 51);
      for (const call of [() => np.percentile(a, q, 1), () => np.nanpercentile(a, q)]) {
        wasmConfig.wasmCallCount = 0;
        call();
        expect(wasmConfig.wasmCallCount).toBeLessThanOrEqual(1);
      }
    });

    it('places the q shape in front of the reduced shape', () => {
      const a = data('float64', [4, 6, 5]);
      const q = np.array([
        [0.1, 0.9],
        [0.5, 0.25],
      ]);
      const r = expectMatchesPerQ(np.nanquantile as QFn, a, q, 1, true);
      expect(Array.from(r.shape)).toEqual([2, 2, 4, 1, 5]);
      expect(Array.from((np.quantile(a, q) as NDArray).shape)).toEqual([2, 2]);
      expect(Array.from((np.quantile(a, np.array([]), 2) as NDArray).shape)).toEqual([0, 4, 6]);
    });

    it('propagates NaN lanes unless the nan variant skips them', () => {
      const a = np.array([
        [1, NaN, 3],
        [NaN, NaN, NaN],
        [4, 5, 6],
      ]);
      const q = np.array([0, 0.5, 1]);
      expect((np.quantile(a, q, 1) as NDArray).toArray()).toEqual([
        [NaN, NaN, 4],
        [NaN, NaN, 5],
        [NaN, NaN, 6],
      ]);
      expect((np.nanquantile(a, q, 1) as NDArray).toArray()).toEqual([
        [1, NaN, 4],
        [2, NaN, 5],
        [3, NaN, 6],
      ]);
    });

    it('raises the same errors as a single q', () => {
      const a = data('float64', [3, 4]);
      expect(() => np.quantile(a, np.array([0.5, 2]))).toThrow(
        'Quantiles must be in the range [0, 1]',
      );
      expect(() => np.percentile(a, np.array([50, -1]), 0)).toThrow(
        'Percentiles must be in the range [0, 100]',
      );
      expect(() => np.quantile(a, np.array([0.5]), 2)).toThrow(
        'axis 2 is out of bounds for array of dimension 2',
      );
      // An out-of-bounds axis is reported before a bad q that is not the first.
      expect(() => np.quantile(a, np.array([0.5, 2]), 2)).toThrow(/out of bounds/);
      expect(() => np.quantile(np.array([true, false]), np.array([0.5]))).toThrow(TypeError);
      expect(() => np.nanquantile(np.array([1, 2], 'complex128'), np.array([0.5]))).toThrow(
        /not orderable/,
      );
    });
  });
});

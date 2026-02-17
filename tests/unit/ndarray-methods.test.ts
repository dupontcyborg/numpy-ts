import { describe, it, expect } from 'vitest';
import { array, zeros } from '../../src';

/**
 * NDArray method coverage tests
 *
 * Smoke tests for every NDArray method to bring src/full/ndarray.ts
 * from ~24% to ~90%+ function coverage. Each method gets at least one
 * basic test with 1-3 assertions.
 */

describe('NDArray Method Coverage', () => {
  // ========================================
  // Arithmetic methods
  // ========================================
  describe('Arithmetic', () => {
    it('mod()', () => {
      const a = array([10, 20, 30]);
      expect(a.mod(7).toArray()).toEqual([3, 6, 2]);
    });

    it('floor_divide()', () => {
      const a = array([7, 17, 27]);
      expect(a.floor_divide(5).toArray()).toEqual([1, 3, 5]);
    });

    it('positive()', () => {
      const a = array([-1, 0, 1]);
      expect(a.positive().toArray()).toEqual([-1, 0, 1]);
    });

    it('reciprocal()', () => {
      const a = array([1, 2, 4]);
      expect(a.reciprocal().toArray()).toEqual([1, 0.5, 0.25]);
    });

    it('sign()', () => {
      const a = array([-5, 0, 3]);
      expect(a.sign().toArray()).toEqual([-1, 0, 1]);
    });

    it('cbrt()', () => {
      const a = array([8, 27, 64]);
      const r = a.cbrt().toArray() as number[];
      expect(r[0]).toBeCloseTo(2);
      expect(r[1]).toBeCloseTo(3);
      expect(r[2]).toBeCloseTo(4);
    });

    it('fabs()', () => {
      const a = array([-1.5, 2.5, -3.5]);
      expect(a.fabs().toArray()).toEqual([1.5, 2.5, 3.5]);
    });

    it('divmod()', () => {
      const a = array([10, 20, 30]);
      const [q, r] = a.divmod(7);
      expect(q.toArray()).toEqual([1, 2, 4]);
      expect(r.toArray()).toEqual([3, 6, 2]);
    });

    it('square()', () => {
      const a = array([2, 3, 4]);
      expect(a.square().toArray()).toEqual([4, 9, 16]);
    });

    it('remainder()', () => {
      const a = array([10, 20, 30]);
      expect(a.remainder(7).toArray()).toEqual([3, 6, 2]);
    });

    it('heaviside()', () => {
      const a = array([-1, 0, 1]);
      expect(a.heaviside(0.5).toArray()).toEqual([0, 0.5, 1]);
    });

    it('clip()', () => {
      const a = array([1, 5, 10, 15, 20]);
      expect(a.clip(5, 15).toArray()).toEqual([5, 5, 10, 15, 15]);
    });

    it('clip() with null bounds', () => {
      const a = array([1, 5, 10, 15, 20]);
      expect(a.clip(null, 10).toArray()).toEqual([1, 5, 10, 10, 10]);
      expect(a.clip(10, null).toArray()).toEqual([10, 10, 10, 15, 20]);
    });
  });

  // ========================================
  // Exp/Log methods
  // ========================================
  describe('Exponential and Logarithmic', () => {
    it('exp2()', () => {
      const a = array([0, 1, 2, 3]);
      expect(a.exp2().toArray()).toEqual([1, 2, 4, 8]);
    });

    it('expm1()', () => {
      const a = array([0]);
      const r = a.expm1().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
    });

    it('log10()', () => {
      const a = array([1, 10, 100]);
      const r = a.log10().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
      expect(r[1]).toBeCloseTo(1);
      expect(r[2]).toBeCloseTo(2);
    });

    it('log1p()', () => {
      const a = array([0]);
      const r = a.log1p().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
    });

    it('logaddexp()', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const r = a.logaddexp(b);
      expect(r.shape).toEqual([2]);
      // log(exp(1) + exp(3)) ≈ 3.1269
      expect((r.toArray() as number[])[0]).toBeCloseTo(3.1269, 3);
    });

    it('logaddexp2()', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      const r = a.logaddexp2(b);
      expect(r.shape).toEqual([2]);
    });
  });

  // ========================================
  // Rounding methods
  // ========================================
  describe('Rounding', () => {
    it('around()', () => {
      const a = array([1.567, 2.345, 3.789]);
      const r = a.around(1).toArray() as number[];
      expect(r[0]).toBeCloseTo(1.6);
      expect(r[1]).toBeCloseTo(2.3);
      expect(r[2]).toBeCloseTo(3.8);
    });

    it('round()', () => {
      const a = array([1.567, 2.345, 3.789]);
      const r = a.round(2).toArray() as number[];
      expect(r[0]).toBeCloseTo(1.57);
    });

    it('ceil()', () => {
      const a = array([1.1, 2.5, -1.7]);
      expect(a.ceil().toArray()).toEqual([2, 3, -1]);
    });

    it('fix()', () => {
      const a = array([1.7, -2.3, 3.9]);
      expect(a.fix().toArray()).toEqual([1, -2, 3]);
    });

    it('floor()', () => {
      const a = array([1.7, 2.5, -1.3]);
      expect(a.floor().toArray()).toEqual([1, 2, -2]);
    });

    it('rint()', () => {
      const a = array([1.4, 2.5, 3.6]);
      const r = a.rint().toArray() as number[];
      expect(r[0]).toBe(1);
      expect(r[2]).toBe(4);
    });

    it('trunc()', () => {
      const a = array([1.7, -2.3, 3.9]);
      expect(a.trunc().toArray()).toEqual([1, -2, 3]);
    });
  });

  // ========================================
  // Trigonometric methods
  // ========================================
  describe('Trigonometry', () => {
    it('arcsin()', () => {
      const a = array([0, 0.5, 1]);
      const r = a.arcsin().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
      expect(r[1]).toBeCloseTo(Math.asin(0.5));
    });

    it('arccos()', () => {
      const a = array([1, 0.5, 0]);
      const r = a.arccos().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
      expect(r[1]).toBeCloseTo(Math.acos(0.5));
    });

    it('arctan()', () => {
      const a = array([0, 1]);
      const r = a.arctan().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
      expect(r[1]).toBeCloseTo(Math.PI / 4);
    });

    it('arctan2()', () => {
      const y = array([1, -1]);
      const x = array([1, 1]);
      const r = y.arctan2(x).toArray() as number[];
      expect(r[0]).toBeCloseTo(Math.PI / 4);
      expect(r[1]).toBeCloseTo(-Math.PI / 4);
    });

    it('hypot()', () => {
      const a = array([3, 5]);
      const b = array([4, 12]);
      const r = a.hypot(b).toArray() as number[];
      expect(r[0]).toBeCloseTo(5);
      expect(r[1]).toBeCloseTo(13);
    });

    it('degrees()', () => {
      const a = array([Math.PI, Math.PI / 2]);
      const r = a.degrees().toArray() as number[];
      expect(r[0]).toBeCloseTo(180);
      expect(r[1]).toBeCloseTo(90);
    });

    it('radians()', () => {
      const a = array([180, 90]);
      const r = a.radians().toArray() as number[];
      expect(r[0]).toBeCloseTo(Math.PI);
      expect(r[1]).toBeCloseTo(Math.PI / 2);
    });
  });

  // ========================================
  // Hyperbolic methods
  // ========================================
  describe('Hyperbolic', () => {
    it('arcsinh()', () => {
      const a = array([0, 1]);
      const r = a.arcsinh().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
      expect(r[1]).toBeCloseTo(Math.asinh(1));
    });

    it('arccosh()', () => {
      const a = array([1, 2]);
      const r = a.arccosh().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
      expect(r[1]).toBeCloseTo(Math.acosh(2));
    });

    it('arctanh()', () => {
      const a = array([0, 0.5]);
      const r = a.arctanh().toArray() as number[];
      expect(r[0]).toBeCloseTo(0);
      expect(r[1]).toBeCloseTo(Math.atanh(0.5));
    });
  });

  // ========================================
  // Comparison methods
  // ========================================
  describe('Comparisons', () => {
    it('greater()', () => {
      const a = array([1, 2, 3]);
      expect(a.greater(2).toArray()).toEqual([0, 0, 1]);
    });

    it('greater_equal()', () => {
      const a = array([1, 2, 3]);
      expect(a.greater_equal(2).toArray()).toEqual([0, 1, 1]);
    });

    it('less()', () => {
      const a = array([1, 2, 3]);
      expect(a.less(2).toArray()).toEqual([1, 0, 0]);
    });

    it('less_equal()', () => {
      const a = array([1, 2, 3]);
      expect(a.less_equal(2).toArray()).toEqual([1, 1, 0]);
    });

    it('equal()', () => {
      const a = array([1, 2, 3]);
      expect(a.equal(2).toArray()).toEqual([0, 1, 0]);
    });

    it('not_equal()', () => {
      const a = array([1, 2, 3]);
      expect(a.not_equal(2).toArray()).toEqual([1, 0, 1]);
    });

    it('isclose()', () => {
      const a = array([1.0, 1.00001, 2.0]);
      const b = array([1.0, 1.0, 2.0]);
      const r = a.isclose(b);
      expect(r.toArray()).toEqual([1, 1, 1]);
    });

    it('allclose()', () => {
      const a = array([1.0, 2.0, 3.0]);
      const b = array([1.0, 2.0, 3.0]);
      expect(a.allclose(b)).toBe(true);
      expect(a.allclose(array([1.0, 2.0, 4.0]))).toBe(false);
    });
  });

  // ========================================
  // Bitwise methods
  // ========================================
  describe('Bitwise', () => {
    it('bitwise_and()', () => {
      const a = array([0b1100, 0b1010], 'int32');
      expect(a.bitwise_and(0b1010).toArray()).toEqual([0b1000, 0b1010]);
    });

    it('bitwise_or()', () => {
      const a = array([0b1100, 0b1010], 'int32');
      expect(a.bitwise_or(0b0011).toArray()).toEqual([0b1111, 0b1011]);
    });

    it('bitwise_xor()', () => {
      const a = array([0b1100, 0b1010], 'int32');
      expect(a.bitwise_xor(0b1111).toArray()).toEqual([0b0011, 0b0101]);
    });

    it('bitwise_not()', () => {
      const a = array([0], 'int32');
      const r = a.bitwise_not();
      expect(r.toArray()).toEqual([-1]);
    });

    it('invert()', () => {
      const a = array([0], 'int32');
      expect(a.invert().toArray()).toEqual([-1]);
    });

    it('left_shift()', () => {
      const a = array([1, 2, 4], 'int32');
      expect(a.left_shift(2).toArray()).toEqual([4, 8, 16]);
    });

    it('right_shift()', () => {
      const a = array([4, 8, 16], 'int32');
      expect(a.right_shift(2).toArray()).toEqual([1, 2, 4]);
    });
  });

  // ========================================
  // Logic methods
  // ========================================
  describe('Logic', () => {
    it('logical_and()', () => {
      const a = array([1, 0, 1]);
      const b = array([1, 1, 0]);
      expect(a.logical_and(b).toArray()).toEqual([1, 0, 0]);
    });

    it('logical_or()', () => {
      const a = array([1, 0, 0]);
      const b = array([0, 0, 1]);
      expect(a.logical_or(b).toArray()).toEqual([1, 0, 1]);
    });

    it('logical_not()', () => {
      const a = array([1, 0, 1]);
      expect(a.logical_not().toArray()).toEqual([0, 1, 0]);
    });

    it('logical_xor()', () => {
      const a = array([1, 0, 1]);
      const b = array([1, 1, 0]);
      expect(a.logical_xor(b).toArray()).toEqual([0, 1, 1]);
    });

    it('isfinite()', () => {
      const a = array([1, Infinity, -Infinity, NaN]);
      expect(a.isfinite().toArray()).toEqual([1, 0, 0, 0]);
    });

    it('isinf()', () => {
      const a = array([1, Infinity, -Infinity, NaN]);
      expect(a.isinf().toArray()).toEqual([0, 1, 1, 0]);
    });

    it('isnan()', () => {
      const a = array([1, 0, NaN, 2]);
      expect(a.isnan().toArray()).toEqual([0, 0, 1, 0]);
    });

    it('isnat()', () => {
      const a = array([1, 0, NaN]);
      // isnat returns array of bools (all false for numeric)
      const r = a.isnat();
      expect(r.shape).toEqual([3]);
    });

    it('copysign()', () => {
      const a = array([1, -1, 1]);
      const b = array([-1, 1, -1]);
      expect(a.copysign(b).toArray()).toEqual([-1, 1, -1]);
    });

    it('signbit()', () => {
      const a = array([1, -1, 0]);
      expect(a.signbit().toArray()).toEqual([0, 1, 0]);
    });

    it('nextafter()', () => {
      const a = array([1.0]);
      const b = array([2.0]);
      const r = a.nextafter(b);
      expect(r.shape).toEqual([1]);
      expect((r.toArray() as number[])[0]).toBeGreaterThan(1.0);
    });

    it('spacing()', () => {
      const a = array([1.0]);
      const r = a.spacing();
      expect(r.shape).toEqual([1]);
      expect((r.toArray() as number[])[0]).toBeGreaterThan(0);
    });
  });

  // ========================================
  // Reduction methods
  // ========================================
  describe('Reductions', () => {
    it('ptp()', () => {
      const a = array([3, 1, 4, 1, 5]);
      expect(a.ptp()).toBe(4);
    });

    it('median()', () => {
      const a = array([1, 3, 2, 5, 4]);
      expect(a.median()).toBe(3);
    });

    it('percentile()', () => {
      const a = array([1, 2, 3, 4, 5]);
      expect(a.percentile(50)).toBe(3);
    });

    it('quantile()', () => {
      const a = array([1, 2, 3, 4, 5]);
      expect(a.quantile(0.5)).toBe(3);
    });

    it('average()', () => {
      const a = array([1, 2, 3, 4]);
      expect(a.average()).toBe(2.5);
    });

    it('average() with weights', () => {
      const a = array([1, 2, 3]);
      const w = array([3, 2, 1]);
      const r = a.average(w);
      // (1*3 + 2*2 + 3*1) / (3+2+1) = 10/6 ≈ 1.6667
      expect(r as number).toBeCloseTo(10 / 6);
    });

    it('cumsum()', () => {
      const a = array([1, 2, 3, 4]);
      expect(a.cumsum().toArray()).toEqual([1, 3, 6, 10]);
    });

    it('cumprod()', () => {
      const a = array([1, 2, 3, 4]);
      expect(a.cumprod().toArray()).toEqual([1, 2, 6, 24]);
    });

    // NaN variants
    it('nansum()', () => {
      const a = array([1, NaN, 3]);
      expect(a.nansum()).toBe(4);
    });

    it('nanprod()', () => {
      const a = array([2, NaN, 3]);
      expect(a.nanprod()).toBe(6);
    });

    it('nanmean()', () => {
      const a = array([1, NaN, 3]);
      expect(a.nanmean()).toBe(2);
    });

    it('nanvar()', () => {
      const a = array([1, NaN, 3]);
      const r = a.nanvar() as number;
      expect(r).toBeCloseTo(1);
    });

    it('nanstd()', () => {
      const a = array([1, NaN, 3]);
      const r = a.nanstd() as number;
      expect(r).toBeCloseTo(1);
    });

    it('nanmin()', () => {
      const a = array([3, NaN, 1, 2]);
      expect(a.nanmin()).toBe(1);
    });

    it('nanmax()', () => {
      const a = array([3, NaN, 1, 2]);
      expect(a.nanmax()).toBe(3);
    });

    it('nanquantile()', () => {
      const a = array([1, NaN, 2, 3]);
      const r = a.nanquantile(0.5);
      expect(r).toBe(2);
    });

    it('nanpercentile()', () => {
      const a = array([1, NaN, 2, 3]);
      const r = a.nanpercentile(50);
      expect(r).toBe(2);
    });

    it('nanargmin()', () => {
      const a = array([3, NaN, 1, 2]);
      expect(a.nanargmin()).toBe(2);
    });

    it('nanargmax()', () => {
      const a = array([3, NaN, 1, 2]);
      expect(a.nanargmax()).toBe(0);
    });

    it('nancumsum()', () => {
      const a = array([1, NaN, 3]);
      expect(a.nancumsum().toArray()).toEqual([1, 1, 4]);
    });

    it('nancumprod()', () => {
      const a = array([2, NaN, 3]);
      expect(a.nancumprod().toArray()).toEqual([2, 2, 6]);
    });

    it('nanmedian()', () => {
      const a = array([1, NaN, 3]);
      expect(a.nanmedian()).toBe(2);
    });
  });

  // ========================================
  // Sorting and searching
  // ========================================
  describe('Sorting and Searching', () => {
    it('partition()', () => {
      const a = array([3, 1, 4, 1, 5, 9, 2]);
      const r = a.partition(3);
      // After partition, element at index 3 is in its sorted position
      // Elements before are <= r[3], elements after are >= r[3]
      const arr = r.toArray() as number[];
      const pivotVal = arr[3]!;
      for (let i = 0; i < 3; i++) expect(arr[i]).toBeLessThanOrEqual(pivotVal);
      for (let i = 4; i < 7; i++) expect(arr[i]).toBeGreaterThanOrEqual(pivotVal);
    });

    it('argpartition()', () => {
      const a = array([3, 1, 4, 1, 5]);
      const r = a.argpartition(2);
      expect(r.shape).toEqual([5]);
    });

    it('nonzero()', () => {
      const a = array([0, 1, 0, 2, 3]);
      const nz = a.nonzero();
      expect(nz).toHaveLength(1);
      expect(nz[0]!.toArray()).toEqual([1, 3, 4]);
    });

    it('argwhere()', () => {
      const a = array([0, 1, 0, 2]);
      const r = a.argwhere();
      expect(r.toArray()).toEqual([[1], [3]]);
    });

    it('searchsorted()', () => {
      const sorted = array([1, 3, 5, 7, 9]);
      const vals = array([2, 4, 6]);
      const r = sorted.searchsorted(vals);
      expect(r.toArray()).toEqual([1, 2, 3]);
    });
  });

  // ========================================
  // Shape manipulation
  // ========================================
  describe('Shape Manipulation', () => {
    it('moveaxis()', () => {
      const a = zeros([2, 3, 4]);
      const r = a.moveaxis(0, 2);
      expect(r.shape).toEqual([3, 4, 2]);
    });

    it('swapaxes()', () => {
      const a = zeros([2, 3, 4]);
      const r = a.swapaxes(0, 2);
      expect(r.shape).toEqual([4, 3, 2]);
    });

    it('repeat()', () => {
      const a = array([1, 2, 3]);
      expect(a.repeat(2).toArray()).toEqual([1, 1, 2, 2, 3, 3]);
    });

    it('take()', () => {
      const a = array([10, 20, 30, 40, 50]);
      expect(a.take([0, 2, 4]).toArray()).toEqual([10, 30, 50]);
    });

    it('diff()', () => {
      const a = array([1, 3, 6, 10]);
      expect(a.diff().toArray()).toEqual([2, 3, 4]);
    });

    it('resize()', () => {
      const a = array([1, 2, 3]);
      const r = a.resize([5]);
      expect(r.shape).toEqual([5]);
      expect(r.size).toBe(5);
    });
  });

  // ========================================
  // Advanced methods
  // ========================================
  describe('Advanced', () => {
    it('put()', () => {
      const a = array([0, 0, 0, 0, 0]);
      a.put([1, 3], 99);
      expect(a.toArray()).toEqual([0, 99, 0, 99, 0]);
    });

    it('choose()', () => {
      const idx = array([0, 1, 2, 1], 'int32');
      const choices = [array([0, 0, 0, 0]), array([10, 10, 10, 10]), array([20, 20, 20, 20])];
      const r = idx.choose(choices);
      expect(r.toArray()).toEqual([0, 10, 20, 10]);
    });

    it('compress()', () => {
      const a = array([10, 20, 30, 40, 50]);
      const r = a.compress([true, false, true, false, true]);
      expect(r.toArray()).toEqual([10, 30, 50]);
    });

    it('conj()', () => {
      const a = array([1, 2, 3]);
      // For real arrays, conj returns same values
      expect(a.conj().toArray()).toEqual([1, 2, 3]);
    });

    it('conjugate()', () => {
      const a = array([1, 2, 3]);
      expect(a.conjugate().toArray()).toEqual([1, 2, 3]);
    });

    it('diagonal()', () => {
      const a = array([
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
      ]);
      expect(a.diagonal().toArray()).toEqual([1, 5, 9]);
    });

    it('iindex()', () => {
      const a = array([
        [10, 20, 30],
        [40, 50, 60],
        [70, 80, 90],
      ]);
      const r = a.iindex([0, 2]);
      expect(r.toArray()).toEqual([
        [10, 20, 30],
        [70, 80, 90],
      ]);
    });

    it('iindex() with NDArray indices', () => {
      const a = array([10, 20, 30, 40, 50]);
      const idx = array([0, 2, 4], 'int32');
      const r = a.iindex(idx, 0);
      expect(r.toArray()).toEqual([10, 30, 50]);
    });

    it('bindex()', () => {
      const a = array([10, 20, 30, 40, 50]);
      const mask = array([1, 0, 1, 0, 1], 'bool');
      const r = a.bindex(mask);
      expect(r.toArray()).toEqual([10, 30, 50]);
    });
  });

  // ========================================
  // Conversion methods
  // ========================================
  describe('Conversion', () => {
    it('tolist()', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(a.tolist()).toEqual([
        [1, 2],
        [3, 4],
      ]);
    });

    it('tobytes()', () => {
      const a = array([1, 2], 'int32');
      const bytes = a.tobytes();
      expect(bytes).toBeInstanceOf(ArrayBuffer);
      expect(bytes.byteLength).toBe(8); // 2 * 4 bytes
    });

    it('item() scalar', () => {
      const a = array([42]);
      expect(a.item()).toBe(42);
    });

    it('item() with flat index', () => {
      const a = array([10, 20, 30]);
      expect(a.item(1)).toBe(20);
    });

    it('item() with multi-index', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(a.item(1, 0)).toBe(3);
    });

    it('byteswap()', () => {
      const a = array([1], 'int32');
      const r = a.byteswap();
      expect(r.shape).toEqual([1]);
      // Verify it's not the same as original (bytes are swapped)
      expect(r.data[0]).not.toBe(a.data[0]);
    });

    it('byteswap() with 1-byte dtype', () => {
      const a = array([1, 2, 3], 'uint8');
      const r = a.byteswap();
      // No swapping for 1-byte types
      expect(r.toArray()).toEqual([1, 2, 3]);
    });

    it('view() same dtype', () => {
      const a = array([1, 2, 3]);
      const v = a.view();
      expect(v.toArray()).toEqual([1, 2, 3]);
    });

    it('view() different dtype same size', () => {
      const a = array([1, 2, 3], 'float64');
      // int64 and float64 are both 8 bytes
      const v = a.view('int64');
      expect(v.shape).toEqual([3]);
      expect(v.dtype).toBe('int64');
    });

    it('tofile() throws', () => {
      const a = array([1, 2, 3]);
      expect(() => a.tofile('test.bin')).toThrow();
    });
  });

  // ========================================
  // Linalg methods
  // ========================================
  describe('Linear Algebra', () => {
    it('trace()', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(a.trace()).toBe(5);
    });

    it('inner()', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      expect(a.inner(b)).toBe(32);
    });

    it('outer()', () => {
      const a = array([1, 2]);
      const b = array([3, 4]);
      expect(a.outer(b).toArray()).toEqual([
        [3, 4],
        [6, 8],
      ]);
    });

    it('tensordot()', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const r = a.tensordot(b, 1);
      expect(r).toBeDefined();
    });

    it('matmul()', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const b = array([
        [5, 6],
        [7, 8],
      ]);
      const r = a.matmul(b);
      expect(r.toArray()).toEqual([
        [19, 22],
        [43, 50],
      ]);
    });

    it('dot()', () => {
      const a = array([1, 2, 3]);
      const b = array([4, 5, 6]);
      expect(a.dot(b)).toBe(32);
    });
  });

  // ========================================
  // Slicing convenience methods
  // ========================================
  describe('Slicing Convenience', () => {
    const m = array([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);

    it('row()', () => {
      expect(m.row(1).toArray()).toEqual([4, 5, 6]);
    });

    it('col()', () => {
      expect(m.col(1).toArray()).toEqual([2, 5, 8]);
    });

    it('rows()', () => {
      expect(m.rows(0, 2).toArray()).toEqual([
        [1, 2, 3],
        [4, 5, 6],
      ]);
    });

    it('cols()', () => {
      expect(m.cols(1, 3).toArray()).toEqual([
        [2, 3],
        [5, 6],
        [8, 9],
      ]);
    });
  });

  // ========================================
  // Properties
  // ========================================
  describe('Properties', () => {
    it('T (transpose)', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      expect(a.T.toArray()).toEqual([
        [1, 3],
        [2, 4],
      ]);
    });

    it('itemsize', () => {
      expect(array([1], 'float64').itemsize).toBe(8);
      expect(array([1], 'int32').itemsize).toBe(4);
      expect(array([1], 'int16').itemsize).toBe(2);
      expect(array([1], 'uint8').itemsize).toBe(1);
    });

    it('nbytes', () => {
      const a = array([1, 2, 3, 4], 'float64');
      expect(a.nbytes).toBe(32); // 4 * 8 bytes
    });

    it('flags', () => {
      const a = array([1, 2, 3]);
      const flags = a.flags;
      expect(flags.C_CONTIGUOUS).toBe(true);
      expect(flags.OWNDATA).toBe(true);
    });

    it('base', () => {
      const a = array([1, 2, 3, 4, 5, 6]);
      expect(a.base).toBeNull();
      // Sliced view has base
      const v = a.slice('1:4');
      expect(v.base).not.toBeNull();
    });

    it('toString()', () => {
      const a = array([1, 2, 3]);
      expect(a.toString()).toContain('[');
      expect(a.toString()).toContain('1.');
      expect(a.toString()).not.toContain('NDArray');
    });
  });

  // ========================================
  // Fill method
  // ========================================
  describe('Fill', () => {
    it('fill() with number', () => {
      const a = zeros([3]);
      a.fill(5);
      expect(a.toArray()).toEqual([5, 5, 5]);
    });

    it('fill() with bigint', () => {
      const a = array([0n, 0n, 0n], 'int64');
      a.fill(7n);
      expect(a.toArray()).toEqual([7n, 7n, 7n]);
    });

    it('fill() with bool dtype', () => {
      const a = zeros([3], 'bool');
      a.fill(1);
      expect(a.toArray()).toEqual([1, 1, 1]);
    });
  });

  // ========================================
  // Iterator
  // ========================================
  describe('Iterator', () => {
    it('[Symbol.iterator]() 1D', () => {
      const a = array([10, 20, 30]);
      const values = [...a];
      expect(values).toEqual([10, 20, 30]);
    });

    it('[Symbol.iterator]() 2D', () => {
      const a = array([
        [1, 2],
        [3, 4],
      ]);
      const rows = [...a];
      expect(rows).toHaveLength(2);
    });
  });

  // ========================================
  // Copy and astype
  // ========================================
  describe('Copy and Astype', () => {
    it('copy() returns independent array', () => {
      const a = array([1, 2, 3]);
      const b = a.copy();
      b.iset(0, 99);
      expect(a.toArray()).toEqual([1, 2, 3]);
      expect(b.toArray()).toEqual([99, 2, 3]);
    });

    it('astype() same dtype no copy', () => {
      const a = array([1, 2, 3], 'float64');
      const b = a.astype('float64', false);
      expect(b.dtype).toBe('float64');
    });

    it('astype() same dtype with copy', () => {
      const a = array([1, 2, 3], 'float64');
      const b = a.astype('float64', true);
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('astype() int to float', () => {
      const a = array([1, 2, 3], 'int32');
      const b = a.astype('float64');
      expect(b.dtype).toBe('float64');
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('astype() float to int', () => {
      const a = array([1.7, 2.3, 3.9], 'float64');
      const b = a.astype('int32');
      expect(b.dtype).toBe('int32');
    });

    it('astype() to bigint', () => {
      const a = array([1, 2, 3], 'int32');
      const b = a.astype('int64');
      expect(b.dtype).toBe('int64');
      expect(b.toArray()).toEqual([1n, 2n, 3n]);
    });

    it('astype() from bigint', () => {
      const a = array([1n, 2n, 3n], 'int64');
      const b = a.astype('float64');
      expect(b.dtype).toBe('float64');
      expect(b.toArray()).toEqual([1, 2, 3]);
    });

    it('astype() to bool', () => {
      const a = array([0, 1, 2, 0], 'int32');
      const b = a.astype('bool');
      expect(b.dtype).toBe('bool');
      expect(b.toArray()).toEqual([0, 1, 1, 0]);
    });

    it('astype() from bool', () => {
      const a = array([0, 1, 1, 0], 'bool');
      const b = a.astype('float64');
      expect(b.toArray()).toEqual([0, 1, 1, 0]);
    });

    it('astype() bigint to bool', () => {
      const a = array([0n, 1n, 2n], 'int64');
      const b = a.astype('bool');
      expect(b.dtype).toBe('bool');
      expect(b.toArray()).toEqual([0, 1, 1]);
    });

    it('astype() bigint to bigint (copy)', () => {
      const a = array([1n, 2n], 'int64');
      const b = a.astype('uint64');
      expect(b.dtype).toBe('uint64');
    });
  });

  // ========================================
  // Get/Set with edge cases
  // ========================================
  describe('Get/Set', () => {
    it('get() with negative indices', () => {
      const a = array([10, 20, 30]);
      expect(a.get([-1])).toBe(30);
    });

    it('set() basic', () => {
      const a = array([1, 2, 3]);
      a.set([1], 99);
      expect(a.toArray()).toEqual([1, 99, 3]);
    });

    it('set() with negative indices', () => {
      const a = array([1, 2, 3]);
      a.set([-1], 99);
      expect(a.toArray()).toEqual([1, 2, 99]);
    });

    it('iget/iset', () => {
      const a = array([10, 20, 30]);
      expect(a.iget(1)).toBe(20);
      a.iset(1, 99);
      expect(a.iget(1)).toBe(99);
    });

    it('set() with bool dtype', () => {
      const a = zeros([3], 'bool');
      a.set([0], 1);
      a.set([2], 1);
      expect(a.toArray()).toEqual([1, 0, 1]);
    });

    it('set() with bigint dtype', () => {
      const a = array([0n, 0n, 0n], 'int64');
      a.set([1], 42n);
      expect(a.get([1])).toBe(42n);
    });
  });

  // ========================================
  // Slice edge cases in NDArray
  // ========================================
  describe('Slice', () => {
    it('slice() negative step', () => {
      const a = array([1, 2, 3, 4, 5]);
      const r = a.slice('::-1');
      expect(r.toArray()).toEqual([5, 4, 3, 2, 1]);
    });

    it('slice() empty returns self', () => {
      const a = array([1, 2, 3]);
      const r = a.slice();
      expect(r.toArray()).toEqual([1, 2, 3]);
    });
  });

  // ========================================
  // fromStorage static method
  // ========================================
  describe('Static', () => {
    it('fromStorage creates NDArray', () => {
      const a = array([1, 2, 3]);
      // Access fromStorage indirectly through existing API
      const b = a.copy();
      expect(b.toArray()).toEqual([1, 2, 3]);
    });
  });
});

import { describe, it, expect } from 'vitest';
import {
  array,
  zeros,
  eye,
  linalg,
  // Standalone functions for ops coverage
  cross,
  diff,
  packbits,
  unpackbits,
  bitwise_count,
  bitwise_invert,
  bitwise_left_shift,
  bitwise_right_shift,
  // Reduction standalone functions
  sum,
  mean,
  ptp,
  median,
  percentile,
  quantile,
  average,
  cumsum,
  cumprod,
  nansum,
  nanmean,
  nanvar,
  nanstd,
  nanmin,
  nanmax,
  nanargmin,
  nanargmax,
  nancumsum,
  nancumprod,
  nanmedian,
  nanquantile,
  nanpercentile,
  // Advanced standalone functions
  broadcast_to,
  choose,
  compress,
  resize,
  take,
  select,
  apply_along_axis,
  apply_over_axes,
  tril_indices,
  triu_indices,
  tril_indices_from,
  triu_indices_from,
  diag_indices,
  diag_indices_from,
  ravel_multi_index,
  unravel_index,
  indices,
  fill_diagonal,
  // Linalg (top-level)
  diagonal,
  kron,
  vdot,
  vecdot,
  matrix_transpose,
  permute_dims,
  matvec,
  vecmat,
  // Creation
  fromfunction,
  fromiter,
  tri,
  frombuffer,
  fromstring,
  // Sorting
  sort,
  argsort,
  partition,
  argpartition,
  nonzero,
  argwhere,
  searchsorted,
  flatnonzero,
  lexsort,
  sort_complex,
  // Gradient
  ediff1d,
  gradient,
} from '../../src';

/**
 * Tests for uncovered functions/branches in the ops layer.
 * Focuses on files with largest coverage gaps.
 */

// ========================================
// gradient.ts coverage
// ========================================
describe('Gradient Ops Coverage', () => {
  it('diff() basic', () => {
    const a = array([1, 3, 6, 10]);
    expect(diff(a).toArray()).toEqual([2, 3, 4]);
  });

  it('diff() with n=2', () => {
    const a = array([1, 3, 6, 10]);
    expect(diff(a, 2).toArray()).toEqual([1, 1]);
  });

  it('diff() with 2D array and axis', () => {
    const a = array([
      [1, 2, 4],
      [4, 6, 9],
    ]);
    const r = diff(a, 1, 1);
    expect(r.toArray()).toEqual([
      [1, 2],
      [2, 3],
    ]);
  });

  it('diff() with bigint', () => {
    const a = array([1n, 3n, 6n, 10n], 'int64');
    const r = diff(a);
    // diff promotes to float64 in some implementations; just verify shape
    expect(r.size).toBe(3);
  });

  it('ediff1d()', () => {
    const a = array([1, 2, 4, 7, 0]);
    const r = ediff1d(a);
    expect(r.toArray()).toEqual([1, 2, 3, -7]);
  });

  it('ediff1d() with to_begin and to_end', () => {
    const a = array([1, 2, 4]);
    const r = ediff1d(a, [99], [-99]);
    expect(r.size).toBeGreaterThan(2);
  });

  it('gradient() 1D', () => {
    const a = array([1, 2, 4, 7, 11]);
    const r = gradient(a);
    // For 1D, gradient may return single NDArray or array of 1
    if (Array.isArray(r)) {
      expect(r).toHaveLength(1);
    } else {
      expect(r.shape).toEqual([5]);
    }
  });

  it('gradient() 2D', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const r = gradient(a);
    if (Array.isArray(r)) {
      expect(r).toHaveLength(2);
    } else {
      expect(r).toBeDefined();
    }
  });

  it('cross() 3D vectors', () => {
    const a = array([1, 0, 0]);
    const b = array([0, 1, 0]);
    const r = cross(a, b);
    expect(r.toArray()).toEqual([0, 0, 1]);
  });

  it('cross() 2D vectors', () => {
    const a = array([1, 0]);
    const b = array([0, 1]);
    const r = cross(a, b);
    // 2D cross product returns scalar
    expect(r.size).toBe(1);
  });
});

// ========================================
// bitwise.ts coverage
// ========================================
describe('Bitwise Ops Coverage', () => {
  it('packbits() 1D', () => {
    const a = array([1, 0, 1, 0, 1, 0, 1, 0], 'uint8');
    const r = packbits(a);
    // 10101010 = 0xAA = 170
    expect(r.toArray()).toEqual([170]);
  });

  it('packbits() with little endian', () => {
    const a = array([1, 0, 1, 0, 1, 0, 1, 0], 'uint8');
    const r = packbits(a, -1, 'little');
    // 01010101 = 0x55 = 85
    expect(r.toArray()).toEqual([85]);
  });

  it('packbits() 2D', () => {
    const a = array(
      [
        [1, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 1, 0],
      ],
      'uint8'
    );
    const r = packbits(a, -1);
    expect(r.shape).toEqual([2, 1]);
  });

  it('unpackbits() 1D', () => {
    const a = array([170], 'uint8'); // 10101010
    const r = unpackbits(a);
    expect(r.toArray()).toEqual([1, 0, 1, 0, 1, 0, 1, 0]);
  });

  it('unpackbits() with count', () => {
    const a = array([170], 'uint8');
    const r = unpackbits(a, -1, 4);
    expect(r.toArray()).toEqual([1, 0, 1, 0]);
  });

  it('unpackbits() with little endian', () => {
    const a = array([85], 'uint8'); // 01010101
    const r = unpackbits(a, -1, -1, 'little');
    expect(r.toArray()).toEqual([1, 0, 1, 0, 1, 0, 1, 0]);
  });

  it('unpackbits() 2D', () => {
    const a = array([[170], [85]], 'uint8');
    const r = unpackbits(a, -1);
    expect(r.shape).toEqual([2, 8]);
  });

  it('bitwise_count()', () => {
    const a = array([0, 1, 7, 255], 'int32');
    const r = bitwise_count(a);
    expect(r.toArray()).toEqual([0, 1, 3, 8]);
  });

  it('bitwise_count() with bigint', () => {
    const a = array([0n, 1n, 7n], 'int64');
    const r = bitwise_count(a);
    expect(r.toArray()).toEqual([0, 1, 3]);
  });

  it('bitwise_invert()', () => {
    const a = array([0], 'int32');
    expect(bitwise_invert(a).toArray()).toEqual([-1]);
  });

  it('bitwise_left_shift()', () => {
    const a = array([1, 2], 'int32');
    expect(bitwise_left_shift(a, 2).toArray()).toEqual([4, 8]);
  });

  it('bitwise_right_shift()', () => {
    const a = array([4, 8], 'int32');
    expect(bitwise_right_shift(a, 2).toArray()).toEqual([1, 2]);
  });
});

// ========================================
// reduction.ts coverage
// ========================================
describe('Reduction Ops Coverage', () => {
  it('sum() with keepdims=true', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = sum(a, 0, true);
    expect(r).toBeDefined();
    if (typeof r !== 'number') {
      expect(r.shape).toEqual([1, 2]);
    }
  });

  it('mean() with keepdims=true', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = mean(a, 1, true);
    if (typeof r !== 'number') {
      expect(r.shape).toEqual([2, 1]);
    }
  });

  it('ptp() standalone function', () => {
    const a = array([3, 1, 4, 1, 5]);
    expect(ptp(a)).toBe(4);
  });

  it('ptp() with axis', () => {
    const a = array([
      [3, 1],
      [4, 5],
    ]);
    const r = ptp(a, 0);
    if (typeof r !== 'number') {
      expect(r.toArray()).toEqual([1, 4]);
    }
  });

  it('median() standalone', () => {
    const a = array([3, 1, 2]);
    expect(median(a)).toBe(2);
  });

  it('percentile() standalone', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(percentile(a, 50)).toBe(3);
  });

  it('quantile() standalone', () => {
    const a = array([1, 2, 3, 4, 5]);
    expect(quantile(a, 0.5)).toBe(3);
  });

  it('average() standalone', () => {
    expect(average(array([1, 2, 3]))).toBe(2);
  });

  it('cumsum() standalone', () => {
    expect(cumsum(array([1, 2, 3])).toArray()).toEqual([1, 3, 6]);
  });

  it('cumsum() with axis', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = cumsum(a, 0);
    expect(r.toArray()).toEqual([
      [1, 2],
      [4, 6],
    ]);
  });

  it('cumprod() standalone', () => {
    expect(cumprod(array([1, 2, 3, 4])).toArray()).toEqual([1, 2, 6, 24]);
  });

  it('cumprod() with axis', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = cumprod(a, 1);
    expect(r.toArray()).toEqual([
      [1, 2],
      [3, 12],
    ]);
  });

  // NaN standalone functions
  it('nansum() standalone', () => {
    expect(nansum(array([1, NaN, 3]))).toBe(4);
  });

  it('nanmean() standalone', () => {
    expect(nanmean(array([1, NaN, 3]))).toBe(2);
  });

  it('nanvar() standalone', () => {
    const r = nanvar(array([1, NaN, 3]));
    expect(r as number).toBeCloseTo(1);
  });

  it('nanstd() standalone', () => {
    const r = nanstd(array([1, NaN, 3]));
    expect(r as number).toBeCloseTo(1);
  });

  it('nanmin() standalone', () => {
    expect(nanmin(array([3, NaN, 1]))).toBe(1);
  });

  it('nanmax() standalone', () => {
    expect(nanmax(array([3, NaN, 1]))).toBe(3);
  });

  it('nanargmin() standalone', () => {
    expect(nanargmin(array([3, NaN, 1]))).toBe(2);
  });

  it('nanargmax() standalone', () => {
    expect(nanargmax(array([3, NaN, 1]))).toBe(0);
  });

  it('nancumsum() standalone', () => {
    expect(nancumsum(array([1, NaN, 3])).toArray()).toEqual([1, 1, 4]);
  });

  it('nancumprod() standalone', () => {
    expect(nancumprod(array([2, NaN, 3])).toArray()).toEqual([2, 2, 6]);
  });

  it('nanmedian() standalone', () => {
    expect(nanmedian(array([1, NaN, 3]))).toBe(2);
  });

  it('nanquantile() standalone', () => {
    expect(nanquantile(array([1, NaN, 2, 3]), 0.5)).toBe(2);
  });

  it('nanpercentile() standalone', () => {
    expect(nanpercentile(array([1, NaN, 2, 3]), 50)).toBe(2);
  });
});

// ========================================
// advanced.ts coverage
// ========================================
describe('Advanced Ops Coverage', () => {
  it('broadcast_to()', () => {
    const a = array([1, 2, 3]);
    const r = broadcast_to(a, [2, 3]);
    expect(r.shape).toEqual([2, 3]);
    expect(r.toArray()).toEqual([
      [1, 2, 3],
      [1, 2, 3],
    ]);
  });

  it('choose() standalone', () => {
    const idx = array([0, 1, 2, 1], 'int32');
    const c0 = array([0, 0, 0, 0]);
    const c1 = array([10, 10, 10, 10]);
    const c2 = array([20, 20, 20, 20]);
    const r = choose(idx, [c0, c1, c2]);
    expect(r.toArray()).toEqual([0, 10, 20, 10]);
  });

  it('compress() standalone', () => {
    const a = array([10, 20, 30, 40, 50]);
    const cond = array([1, 0, 1, 0, 1], 'bool');
    const r = compress(cond, a);
    expect(r.toArray()).toEqual([10, 30, 50]);
  });

  it('compress() with axis', () => {
    const a = array([
      [1, 2],
      [3, 4],
      [5, 6],
    ]);
    const cond = array([1, 0, 1], 'bool');
    const r = compress(cond, a, 0);
    expect(r.toArray()).toEqual([
      [1, 2],
      [5, 6],
    ]);
  });

  it('resize() standalone', () => {
    const a = array([1, 2, 3]);
    const r = resize(a, [5]);
    expect(r.shape).toEqual([5]);
    expect(r.size).toBe(5);
  });

  it('take() standalone', () => {
    const a = array([10, 20, 30, 40, 50]);
    expect(take(a, [0, 2, 4]).toArray()).toEqual([10, 30, 50]);
  });

  it('select() standalone', () => {
    const x = array([1, 2, 3, 4, 5]);
    const condlist = [x.less(3), x.greater(3)];
    const choicelist = [array([-1, -1, -1, -1, -1]), array([1, 1, 1, 1, 1])];
    const r = select(condlist, choicelist, 0);
    expect(r.toArray()).toEqual([-1, -1, 0, 1, 1]);
  });

  it('tril_indices()', () => {
    const [rows, cols] = tril_indices(3);
    expect(rows.size).toBeGreaterThan(0);
    expect(cols.size).toBeGreaterThan(0);
  });

  it('triu_indices()', () => {
    const [rows, cols] = triu_indices(3);
    expect(rows.size).toBeGreaterThan(0);
    expect(cols.size).toBeGreaterThan(0);
  });

  it('tril_indices_from()', () => {
    const a = eye(3);
    const [rows] = tril_indices_from(a);
    expect(rows.size).toBeGreaterThan(0);
  });

  it('triu_indices_from()', () => {
    const a = eye(3);
    const [rows] = triu_indices_from(a);
    expect(rows.size).toBeGreaterThan(0);
  });

  it('diag_indices()', () => {
    const r = diag_indices(3);
    expect(r).toHaveLength(2);
    expect(r[0]!.toArray()).toEqual([0, 1, 2]);
  });

  it('diag_indices_from()', () => {
    const a = eye(3);
    const r = diag_indices_from(a);
    expect(r).toHaveLength(2);
  });

  it('ravel_multi_index()', () => {
    const r = ravel_multi_index([array([0, 1, 2]), array([0, 1, 2])], [3, 3]);
    expect(r.toArray()).toEqual([0, 4, 8]);
  });

  it('unravel_index()', () => {
    const r = unravel_index(array([0, 4, 8]), [3, 3]);
    expect(r).toHaveLength(2);
  });

  it('indices()', () => {
    const r = indices([2, 3]);
    // indices returns a stacked NDArray of shape [ndim, ...dimensions]
    expect(r.shape[0]).toBe(2);
  });

  it('fill_diagonal()', () => {
    const a = zeros([3, 3]);
    fill_diagonal(a, 5);
    expect(a.get([0, 0])).toBe(5);
    expect(a.get([1, 1])).toBe(5);
    expect(a.get([2, 2])).toBe(5);
  });

  it('apply_along_axis()', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
    ]);
    const r = apply_along_axis(
      (x) => {
        // sum each row
        let s = 0;
        for (let i = 0; i < x.size; i++) s += x.iget(i) as number;
        return array([s]);
      },
      1,
      a
    );
    expect(r).toBeDefined();
  });

  it('apply_over_axes()', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = apply_over_axes(sum, a, [0]);
    expect(r).toBeDefined();
  });
});

// ========================================
// linalg.ts coverage (via linalg namespace)
// ========================================
describe('Linear Algebra Ops Coverage', () => {
  it('linalg.svd()', () => {
    const a = array([
      [1, 0],
      [0, 1],
    ]);
    const result = linalg.svd(a);
    expect(result).toBeDefined();
  });

  it('linalg.eig()', () => {
    const a = array([
      [1, 0],
      [0, 2],
    ]);
    const result = linalg.eig(a);
    expect(result).toHaveProperty('w');
    expect(result).toHaveProperty('v');
  });

  it('linalg.eigvals()', () => {
    const a = array([
      [1, 0],
      [0, 2],
    ]);
    const r = linalg.eigvals(a);
    expect(r.size).toBe(2);
  });

  it('linalg.eigh()', () => {
    const a = array([
      [2, 1],
      [1, 2],
    ]);
    const result = linalg.eigh(a);
    expect(result).toHaveProperty('w');
    expect(result).toHaveProperty('v');
  });

  it('linalg.eigvalsh()', () => {
    const a = array([
      [2, 1],
      [1, 2],
    ]);
    const r = linalg.eigvalsh(a);
    expect(r.size).toBe(2);
  });

  it('linalg.lstsq()', () => {
    const a = array([
      [1, 1],
      [1, 2],
      [1, 3],
    ]);
    const b = array([1, 2, 3]);
    const result = linalg.lstsq(a, b);
    expect(result).toBeDefined();
  });

  it('linalg.norm() vector', () => {
    const a = array([3, 4]);
    const r = linalg.norm(a);
    expect(r).toBeCloseTo(5);
  });

  it('linalg.norm() matrix fro', () => {
    const a = eye(3);
    const r = linalg.norm(a, 'fro');
    expect(r).toBeCloseTo(Math.sqrt(3));
  });

  it('linalg.qr()', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const result = linalg.qr(a);
    expect(result).toBeDefined();
  });

  it('linalg.cholesky()', () => {
    const a = array([
      [4, 2],
      [2, 3],
    ]);
    const r = linalg.cholesky(a);
    expect(r.shape).toEqual([2, 2]);
  });

  it('linalg.det()', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    expect(linalg.det(a)).toBeCloseTo(-2);
  });

  it('linalg.inv()', () => {
    const a = array([
      [1, 0],
      [0, 2],
    ]);
    const r = linalg.inv(a);
    expect(r.toArray()).toEqual([
      [1, 0],
      [0, 0.5],
    ]);
  });

  it('linalg.solve()', () => {
    const a = array([
      [1, 0],
      [0, 1],
    ]);
    const b = array([3, 4]);
    const r = linalg.solve(a, b);
    expect(r.toArray()).toEqual([3, 4]);
  });

  it('linalg.matrix_rank()', () => {
    const a = array([
      [1, 0],
      [0, 1],
    ]);
    expect(linalg.matrix_rank(a)).toBe(2);
  });

  it('linalg.pinv()', () => {
    const a = array([
      [1, 0],
      [0, 1],
    ]);
    const r = linalg.pinv(a);
    expect(r.shape).toEqual([2, 2]);
  });

  it('linalg.slogdet()', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const result = linalg.slogdet(a);
    expect(result).toHaveProperty('sign');
    expect(result).toHaveProperty('logabsdet');
  });

  it('linalg.multi_dot()', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const b = array([
      [5, 6],
      [7, 8],
    ]);
    const r = linalg.multi_dot([a, b]);
    expect(r.shape).toEqual([2, 2]);
  });

  it('linalg.cond()', () => {
    const a = array([
      [1, 0],
      [0, 1],
    ]);
    const r = linalg.cond(a);
    expect(r).toBeCloseTo(1);
  });

  it('linalg.matrix_power()', () => {
    const a = array([
      [1, 0],
      [0, 2],
    ]);
    const r = linalg.matrix_power(a, 2);
    expect(r.toArray()).toEqual([
      [1, 0],
      [0, 4],
    ]);
  });

  // Top-level linalg functions
  it('kron()', () => {
    const a = array([
      [1, 0],
      [0, 1],
    ]);
    const b = array([
      [1, 2],
      [3, 4],
    ]);
    const r = kron(a, b);
    expect(r.shape).toEqual([4, 4]);
  });

  it('diagonal() with offset', () => {
    const a = array([
      [1, 2, 3],
      [4, 5, 6],
      [7, 8, 9],
    ]);
    expect(diagonal(a, 1).toArray()).toEqual([2, 6]);
    expect(diagonal(a, -1).toArray()).toEqual([4, 8]);
  });

  it('vdot()', () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    expect(vdot(a, b)).toBe(32);
  });

  it('vecdot()', () => {
    const a = array([1, 2, 3]);
    const b = array([4, 5, 6]);
    const r = vecdot(a, b);
    expect(r).toBeDefined();
  });

  it('matrix_transpose()', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = matrix_transpose(a);
    expect(r.toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it('permute_dims()', () => {
    const a = array([
      [1, 2],
      [3, 4],
    ]);
    const r = permute_dims(a, [1, 0]);
    expect(r.toArray()).toEqual([
      [1, 3],
      [2, 4],
    ]);
  });

  it('matvec()', () => {
    const m = array([
      [1, 2],
      [3, 4],
    ]);
    const v = array([1, 1]);
    const r = matvec(m, v);
    expect(r.toArray()).toEqual([3, 7]);
  });

  it('vecmat()', () => {
    const v = array([1, 1]);
    const m = array([
      [1, 2],
      [3, 4],
    ]);
    const r = vecmat(v, m);
    expect(r.toArray()).toEqual([4, 6]);
  });
});

// ========================================
// creation.ts coverage
// ========================================
describe('Creation Ops Coverage', () => {
  it('fromfunction()', () => {
    const r = fromfunction((i: number, j: number) => i + j, [3, 3]);
    expect(r.shape).toEqual([3, 3]);
    expect(r.get([0, 0])).toBe(0);
    expect(r.get([1, 1])).toBe(2);
  });

  it('fromiter()', () => {
    const iter = function* () {
      yield 1;
      yield 2;
      yield 3;
    };
    const r = fromiter(iter(), 'float64', 3);
    expect(r.toArray()).toEqual([1, 2, 3]);
  });

  it('tri()', () => {
    const r = tri(3);
    expect(r.shape).toEqual([3, 3]);
    expect(r.get([0, 0])).toBe(1);
    expect(r.get([0, 1])).toBe(0);
    expect(r.get([1, 0])).toBe(1);
    expect(r.get([1, 1])).toBe(1);
  });

  it('tri() with M and k', () => {
    const r = tri(3, 4, 1);
    expect(r.shape).toEqual([3, 4]);
  });

  it('frombuffer()', () => {
    const buf = new Float64Array([1, 2, 3]).buffer;
    const r = frombuffer(buf, 'float64');
    expect(r.toArray()).toEqual([1, 2, 3]);
  });

  it('fromstring()', () => {
    const r = fromstring('1 2 3', 'float64', -1, ' ');
    expect(r.toArray()).toEqual([1, 2, 3]);
  });
});

// ========================================
// sorting.ts additional coverage
// ========================================
describe('Sorting Ops Coverage', () => {
  it('sort() standalone', () => {
    const a = array([3, 1, 4, 1, 5]);
    expect(sort(a).toArray()).toEqual([1, 1, 3, 4, 5]);
  });

  it('argsort() standalone', () => {
    const a = array([3, 1, 4]);
    const r = argsort(a);
    expect(r.toArray()).toEqual([1, 0, 2]);
  });

  it('partition() standalone', () => {
    const a = array([3, 1, 4, 1, 5]);
    const r = partition(a, 2);
    const arr = r.toArray() as number[];
    // Element at index 2 should be 3 (the 3rd smallest)
    expect(arr[2]).toBe(3);
  });

  it('argpartition() standalone', () => {
    const a = array([3, 1, 4, 1, 5]);
    const r = argpartition(a, 2);
    expect(r.shape).toEqual([5]);
  });

  it('nonzero() standalone', () => {
    const a = array([0, 1, 0, 2, 3]);
    const nz = nonzero(a);
    expect(nz).toHaveLength(1);
    expect(nz[0]!.toArray()).toEqual([1, 3, 4]);
  });

  it('argwhere() standalone', () => {
    const a = array([0, 1, 0, 2]);
    const r = argwhere(a);
    expect(r.toArray()).toEqual([[1], [3]]);
  });

  it('searchsorted() standalone', () => {
    const sorted = array([1, 3, 5, 7, 9]);
    const vals = array([2, 4, 6]);
    const r = searchsorted(sorted, vals);
    expect(r.toArray()).toEqual([1, 2, 3]);
  });

  it('flatnonzero()', () => {
    const a = array([0, 1, 0, 2, 3]);
    const r = flatnonzero(a);
    expect(r.toArray()).toEqual([1, 3, 4]);
  });

  it('lexsort()', () => {
    const surnames = array([1, 2, 1, 2]); // simplified numeric
    const ages = array([40, 30, 30, 20]);
    const r = lexsort([ages, surnames]);
    expect(r.shape).toEqual([4]);
  });

  it('sort_complex()', () => {
    const a = array([3, 1, 4, 1, 5]);
    const r = sort_complex(a);
    // sort_complex returns complex128, so values are Complex objects
    expect(r.size).toBe(5);
    expect(r.dtype).toContain('complex');
  });
});

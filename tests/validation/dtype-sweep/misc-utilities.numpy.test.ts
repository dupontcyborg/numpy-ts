/**
 * DType Sweep: Miscellaneous utility functions.
 * Tests shape/transform ops, comparison, linalg misc, unique variants,
 * histogram, apply functions, in-place mutation, memory introspection,
 * and misc utilities across ALL dtypes, validated against NumPy.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectBothRejectPre,
  expectComplexFixture,
  expectMatchPre,
  isComplex,
  isFloat,
  npDtype,
  pyArrayCast,
  pyScalarCast,
  runNumPyBatch,
  scalarClose,
} from './_helpers';

const { array } = np;

// One definition per fixture, shared by the oracle snippet and the test body:
// the two sides must build the same input, and a separate declaration on each
// side drifts silently. Complex dtypes carry an imaginary part, so an op that
// only moves or reduces the real half shows up as a mismatch — vdot in
// particular conjugates its first argument, which a real fixture cannot catch.
const fx = {
  d2d: (d: string) =>
    asDtypeData(
      d === 'bool'
        ? [
            [1, 0],
            [1, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ],
      d,
    ),
  d1d: (d: string) => asDtypeData(d === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4], d),
  dUniq: (d: string) => asDtypeData(d === 'bool' ? [1, 0, 1, 1, 0, 1] : [3, 1, 4, 1, 5, 3], d),
  dBlock: (d: string) => asDtypeData(d === 'bool' ? [1, 0] : [1, 2], d),
  placeVals: (d: string) => asDtypeData(d === 'bool' ? [1] : [99], d),
  putmaskVals: (d: string) => asDtypeData(d === 'bool' ? [1] : [88], d),
};

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const sc = pyScalarCast(dtype);
    const d2d = fx.d2d(dtype).py;
    const d1d = fx.d1d(dtype).py;
    const dUniq = fx.dUniq(dtype).py;

    // matrix_transpose
    snippets[`matrix_transpose_${dtype}`] = `
a = np.array(${d2d}, dtype=${npDtype(dtype)})
_result_orig = np.transpose(a)
result = _result_orig.astype(${ac})`;

    // permute_dims
    snippets[`permute_dims_${dtype}`] = `
a = np.array(${d2d}, dtype=${npDtype(dtype)})
_result_orig = np.transpose(a, (1, 0))
result = _result_orig.astype(${ac})`;

    // rollaxis
    snippets[`rollaxis_${dtype}`] = `
a = np.array(${d2d}, dtype=${npDtype(dtype)})
_result_orig = np.rollaxis(a, 1, 0)
result = _result_orig.astype(${ac})`;

    // block (flat list concat)
    snippets[`block_${dtype}`] = `
a = np.array(${fx.dBlock(dtype).py}, dtype=${npDtype(dtype)})
_result_orig = np.block([a, a])
result = _result_orig.astype(${ac})`;

    // array_equal
    snippets[`array_equal_${dtype}`] = `
a = np.array(${d1d}, dtype=${npDtype(dtype)})
result = bool(np.array_equal(a, a))`;

    // vdot
    snippets[`vdot_${dtype}`] = `
a = np.array(${d1d}, dtype=${npDtype(dtype)})
b = np.array(${d1d}, dtype=${npDtype(dtype)})
result = ${sc}(np.vdot(a, b))`;

    // unique_values
    snippets[`unique_values_${dtype}`] = `
a = np.array(${dUniq}, dtype=${npDtype(dtype)})
_result_orig = np.unique(a)
result = _result_orig.astype(${ac})`;

    // unique_counts — values
    snippets[`unique_counts_values_${dtype}`] = `
a = np.array(${dUniq}, dtype=${npDtype(dtype)})
vals, counts = np.unique(a, return_counts=True)
_result_orig = vals
result = _result_orig.astype(${ac})`;

    // unique_inverse — values
    snippets[`unique_inverse_values_${dtype}`] = `
a = np.array(${dUniq}, dtype=${npDtype(dtype)})
vals, inv = np.unique(a, return_inverse=True)
_result_orig = vals
result = _result_orig.astype(${ac})`;

    // unique_all — values
    snippets[`unique_all_values_${dtype}`] = `
a = np.array(${dUniq}, dtype=${npDtype(dtype)})
vals, idx, inv, counts = np.unique(a, return_index=True, return_inverse=True, return_counts=True)
_result_orig = vals
result = _result_orig.astype(${ac})`;

    // apply_along_axis
    snippets[`apply_along_axis_${dtype}`] = `
a = np.array(${d2d}, dtype=${npDtype(dtype)})
_result_orig = np.apply_along_axis(np.sum, 0, a)
result = _result_orig.astype(${ac})`;

    // apply_over_axes
    snippets[`apply_over_axes_${dtype}`] = `
a = np.array(${d2d}, dtype=${npDtype(dtype)})
_result_orig = np.apply_over_axes(np.sum, a, [0])
result = _result_orig.astype(${ac})`;

    // place
    snippets[`place_${dtype}`] = `
a = np.array(${d1d}, dtype=${npDtype(dtype)})
mask = np.array([True, False, True, False])
np.place(a, mask, np.array(${fx.placeVals(dtype).py}, dtype=${npDtype(dtype)}))
_result_orig = a
result = _result_orig.astype(${ac})`;

    // putmask
    snippets[`putmask_${dtype}`] = `
a = np.array(${d1d}, dtype=${npDtype(dtype)})
mask = np.array([True, False, True, False])
np.putmask(a, mask, np.array(${fx.putmaskVals(dtype).py}, dtype=${npDtype(dtype)}))
_result_orig = a
result = _result_orig.astype(${ac})`;

    // copyto
    snippets[`copyto_${dtype}`] = `
dst = np.zeros(4, dtype=${npDtype(dtype)})
src = np.array(${d1d}, dtype=${npDtype(dtype)})
np.copyto(dst, src)
_result_orig = dst
result = _result_orig.astype(${ac})`;

    // einsum trace
    snippets[`einsum_${dtype}`] = `
a = np.array(${d2d}, dtype=${npDtype(dtype)})
result = ${sc}(np.einsum('ii', a))`;
  }

  // Float-only: histogram_bin_edges
  for (const dtype of ALL_DTYPES) {
    if (isFloat(dtype)) {
      snippets[`histogram_bin_edges_${dtype}`] = `
a = np.array([1, 2, 3, 4, 5], dtype=${npDtype(dtype)})
_result_orig = np.histogram_bin_edges(a)
result = _result_orig.astype(np.float64)`;
    }
  }

  // histogram2d / histogramdd — real dtypes only
  for (const dtype of ALL_DTYPES) {
    if (!isComplex(dtype)) {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
      snippets[`histogram2d_${dtype}`] = `
x = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
y = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
H, xedges, yedges = np.histogram2d(x, y, bins=3)
result = H.astype(np.float64)`;

      snippets[`histogramdd_${dtype}`] = `
x = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
sample = np.column_stack([x, x])
H, edges = np.histogramdd(sample, bins=3)
result = H.astype(np.float64)`;
    }
  }

  // einsum_path
  for (const dtype of ALL_DTYPES) {
    snippets[`einsum_path_${dtype}`] = `
a = np.array(${fx.d2d(dtype).py}, dtype=${npDtype(dtype)})
path, info = np.einsum_path('ij,jk->ik', a, a)
result = [str(p) for p in path]`;
  }

  // Non-dtype-dependent
  snippets['broadcast_shapes'] = `
result = list(np.broadcast_shapes((2, 3), (3,)))`;

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Shape/transform ops', () => {
  for (const dtype of ALL_DTYPES) {
    it(`matrix_transpose ${dtype}`, () => {
      const a = array(fx.d2d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `matrix_transpose ${dtype}`);
      expectMatchPre(np.matrix_transpose(a), oracle.get(`matrix_transpose_${dtype}`)!);
    });

    it(`permute_dims ${dtype}`, () => {
      const a = array(fx.d2d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `permute_dims ${dtype}`);
      expectMatchPre(np.permute_dims(a, [1, 0]), oracle.get(`permute_dims_${dtype}`)!);
    });

    it(`rollaxis ${dtype}`, () => {
      const a = array(fx.d2d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `rollaxis ${dtype}`);
      expectMatchPre(np.rollaxis(a, 1, 0), oracle.get(`rollaxis_${dtype}`)!);
    });

    it(`block ${dtype}`, () => {
      const a = array(fx.dBlock(dtype).js, dtype);
      expectComplexFixture(a, dtype, `block ${dtype}`);
      expectMatchPre(np.block([a, a]), oracle.get(`block_${dtype}`)!);
    });
  }
});

describe('DType Sweep: Comparison/equality', () => {
  for (const dtype of ALL_DTYPES) {
    it(`array_equal ${dtype}`, () => {
      const a = array(fx.d1d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `array_equal ${dtype}`);
      const jsResult = np.array_equal(a, a);
      const pyResult = oracle.get(`array_equal_${dtype}`)!;
      if (pyResult.error) throw new Error(`NumPy error: ${pyResult.error}`);
      expect(jsResult).toBe(true);
      expect(pyResult.value).toBe(true);
    });
  }
});

describe('DType Sweep: vdot', () => {
  for (const dtype of ALL_DTYPES) {
    it(`vdot ${dtype}`, () => {
      const a = array(fx.d1d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `vdot ${dtype}`);
      const jsResult = np.vdot(a, a);
      const pyResult = oracle.get(`vdot_${dtype}`)!;
      if (pyResult.error) throw new Error(`NumPy error: ${pyResult.error}`);
      scalarClose(jsResult, pyResult.value);
    });
  }
});

describe('DType Sweep: Unique variants', () => {
  for (const dtype of ALL_DTYPES) {
    it(`unique_values ${dtype}`, () => {
      const a = array(fx.dUniq(dtype).js, dtype);
      expectComplexFixture(a, dtype, `unique_values ${dtype}`);
      expectMatchPre(np.unique_values(a), oracle.get(`unique_values_${dtype}`)!);
    });

    it(`unique_counts ${dtype}`, () => {
      const a = array(fx.dUniq(dtype).js, dtype);
      expectComplexFixture(a, dtype, `unique_counts ${dtype}`);
      expectMatchPre(np.unique_counts(a).values, oracle.get(`unique_counts_values_${dtype}`)!);
    });

    it(`unique_inverse ${dtype}`, () => {
      const a = array(fx.dUniq(dtype).js, dtype);
      expectComplexFixture(a, dtype, `unique_inverse ${dtype}`);
      expectMatchPre(np.unique_inverse(a).values, oracle.get(`unique_inverse_values_${dtype}`)!);
    });

    it(`unique_all ${dtype}`, () => {
      const a = array(fx.dUniq(dtype).js, dtype);
      expectComplexFixture(a, dtype, `unique_all ${dtype}`);
      expectMatchPre(np.unique_all(a).values, oracle.get(`unique_all_values_${dtype}`)!);
    });
  }
});

describe('DType Sweep: Histogram', () => {
  for (const dtype of ALL_DTYPES) {
    if (!isFloat(dtype)) continue;
    it(`histogram_bin_edges ${dtype}`, () => {
      const a = array([1, 2, 3, 4, 5], dtype);
      const jsResult = np.histogram_bin_edges(a);
      expectMatchPre(jsResult, oracle.get(`histogram_bin_edges_${dtype}`)!, {
        rtol: 1e-3,
        atol: 1e-5,
      });
    });
  }
});

describe('DType Sweep: Apply functions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`apply_along_axis ${dtype}`, () => {
      const a = array(fx.d2d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `apply_along_axis ${dtype}`);
      const jsResult = np.apply_along_axis((arr: any): any => np.sum(arr), 0, a);
      expectMatchPre(jsResult, oracle.get(`apply_along_axis_${dtype}`)!);
    });

    it(`apply_over_axes ${dtype}`, () => {
      const a = array(fx.d2d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `apply_over_axes ${dtype}`);
      const jsResult = np.apply_over_axes(
        (arr: any, ax: number): any => np.sum(arr, ax, true),
        a,
        [0],
      );
      expectMatchPre(jsResult, oracle.get(`apply_over_axes_${dtype}`)!);
    });
  }
});

describe('DType Sweep: In-place mutation', () => {
  for (const dtype of ALL_DTYPES) {
    it(`place ${dtype}`, () => {
      const a = array(fx.d1d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `place ${dtype}`);
      const mask = array([1, 0, 1, 0], 'bool');
      np.place(a, mask, array(fx.placeVals(dtype).js, dtype));
      expectMatchPre(a, oracle.get(`place_${dtype}`)!);
    });

    it(`putmask ${dtype}`, () => {
      const a = array(fx.d1d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `putmask ${dtype}`);
      const mask = array([1, 0, 1, 0], 'bool');
      np.putmask(a, mask, array(fx.putmaskVals(dtype).js, dtype));
      expectMatchPre(a, oracle.get(`putmask_${dtype}`)!);
    });

    it(`copyto ${dtype}`, () => {
      const dst = np.zeros([4], dtype);
      const src = array(fx.d1d(dtype).js, dtype);
      expectComplexFixture(src, dtype, `copyto ${dtype}`);
      np.copyto(dst, src);
      expectMatchPre(dst, oracle.get(`copyto_${dtype}`)!);
    });
  }
});

describe('DType Sweep: einsum', () => {
  for (const dtype of ALL_DTYPES) {
    it(`einsum trace ${dtype}`, () => {
      const a = array(fx.d2d(dtype).js, dtype);
      expectComplexFixture(a, dtype, `einsum trace ${dtype}`);
      const jsResult = np.einsum('ii', a);
      const pyResult = oracle.get(`einsum_${dtype}`)!;
      if (pyResult.error) throw new Error(`NumPy error: ${pyResult.error}`);
      scalarClose(jsResult, pyResult.value);
    });
  }
});

describe('DType Sweep: Memory introspection', () => {
  it('may_share_memory — same array', () => {
    const a = array([1, 2, 3], 'float64');
    expect(np.may_share_memory(a, a)).toBe(true);
  });

  it('may_share_memory — different arrays', () => {
    const a = array([1, 2, 3], 'float64');
    const b = array([4, 5, 6], 'float64');
    // may_share_memory can return true or false depending on allocator;
    // just verify it returns a boolean
    expect(typeof np.may_share_memory(a, b)).toBe('boolean');
  });

  it('shares_memory — same array', () => {
    const a = array([1, 2, 3], 'float64');
    expect(np.shares_memory(a, a)).toBe(true);
  });

  it('shares_memory — different arrays', () => {
    const a = array([1, 2, 3], 'float64');
    const b = array([4, 5, 6], 'float64');
    expect(np.shares_memory(a, b)).toBe(false);
  });
});

describe('Misc: broadcast_shapes', () => {
  it('broadcast_shapes([2,3], [3])', () => {
    const result = np.broadcast_shapes([2, 3], [3]);
    const pyResult = oracle.get('broadcast_shapes')!;
    if (pyResult.error) throw new Error(`NumPy error: ${pyResult.error}`);
    expect(result).toEqual(pyResult.value);
  });
});

describe('DType Sweep: Histograms', () => {
  describe('histogram2d', () => {
    for (const dtype of ALL_DTYPES) {
      if (isComplex(dtype)) continue; // complex not supported for histograms
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
        const x = array(data, dtype);
        const y = array(data, dtype);
        const pyResult = oracle.get(`histogram2d_${dtype}`)!;
        const r = expectBothRejectPre(
          `histogram2d may not support ${dtype}`,
          () => np.histogram2d(x, y, 3),
          pyResult,
        );
        if (r === 'both-reject') return;
        const [H] = np.histogram2d(x, y, 3);
        expectMatchPre(H, pyResult, { rtol: 1e-3 });
      });
    }
  });

  describe('histogramdd', () => {
    for (const dtype of ALL_DTYPES) {
      if (isComplex(dtype)) continue;
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
        const x = array(data, dtype);
        const sample = np.stack([x, x], -1);
        const pyResult = oracle.get(`histogramdd_${dtype}`)!;
        const r = expectBothRejectPre(
          `histogramdd may not support ${dtype}`,
          () => np.histogramdd(sample, 3),
          pyResult,
        );
        if (r === 'both-reject') return;
        const [H] = np.histogramdd(sample, 3);
        expectMatchPre(H, pyResult, { rtol: 1e-3 });
      });
    }
  });
});

describe('DType Sweep: Einsum', () => {
  describe('einsum_path', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const a = array(fx.d2d(dtype).js, dtype);
        expectComplexFixture(a, dtype, `einsum_path ${dtype}`);
        const pyResult = oracle.get(`einsum_path_${dtype}`)!;
        const r = expectBothRejectPre(
          `einsum_path may not support ${dtype}`,
          () => np.einsum_path('ij,jk->ik', a, a),
          pyResult,
        );
        if (r === 'both-reject') return;
        const [path, info] = np.einsum_path('ij,jk->ik', a, a);
        expect(Array.isArray(path)).toBe(true);
        expect(typeof info).toBe('string');
      });
    }
  });
});

/**
 * DType Sweep: Miscellaneous utility functions.
 * Tests shape/transform ops, comparison, linalg misc, unique variants,
 * histogram, apply functions, in-place mutation, memory introspection,
 * and misc utilities across ALL dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  isFloat,
  runNumPyBatch,
  expectMatchPre,
  expectBothRejectPre,
  pyArrayCast,
  pyScalarCast,
  scalarClose,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const sc = pyScalarCast(dtype);
    const d2d =
      dtype === 'bool'
        ? [
            [1, 0],
            [1, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ];
    const d1d = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
    const dUniq = dtype === 'bool' ? [1, 0, 1, 1, 0, 1] : [3, 1, 4, 1, 5, 3];

    // matrix_transpose
    snippets[`matrix_transpose_${dtype}`] = `
a = np.array(${JSON.stringify(d2d)}, dtype=${npDtype(dtype)})
_result_orig = np.transpose(a)
result = _result_orig.astype(${ac})`;

    // permute_dims
    snippets[`permute_dims_${dtype}`] = `
a = np.array(${JSON.stringify(d2d)}, dtype=${npDtype(dtype)})
_result_orig = np.transpose(a, (1, 0))
result = _result_orig.astype(${ac})`;

    // rollaxis
    snippets[`rollaxis_${dtype}`] = `
a = np.array(${JSON.stringify(d2d)}, dtype=${npDtype(dtype)})
_result_orig = np.rollaxis(a, 1, 0)
result = _result_orig.astype(${ac})`;

    // block (flat list concat)
    const dBlock = dtype === 'bool' ? [1, 0] : [1, 2];
    snippets[`block_${dtype}`] = `
a = np.array(${JSON.stringify(dBlock)}, dtype=${npDtype(dtype)})
_result_orig = np.block([a, a])
result = _result_orig.astype(${ac})`;

    // array_equal
    snippets[`array_equal_${dtype}`] = `
a = np.array(${JSON.stringify(d1d)}, dtype=${npDtype(dtype)})
result = bool(np.array_equal(a, a))`;

    // vdot
    snippets[`vdot_${dtype}`] = `
a = np.array(${JSON.stringify(d1d)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(d1d)}, dtype=${npDtype(dtype)})
result = ${sc}(np.vdot(a, b))`;

    // unique_values
    snippets[`unique_values_${dtype}`] = `
a = np.array(${JSON.stringify(dUniq)}, dtype=${npDtype(dtype)})
_result_orig = np.unique(a)
result = _result_orig.astype(${ac})`;

    // unique_counts — values
    snippets[`unique_counts_values_${dtype}`] = `
a = np.array(${JSON.stringify(dUniq)}, dtype=${npDtype(dtype)})
vals, counts = np.unique(a, return_counts=True)
_result_orig = vals
result = _result_orig.astype(${ac})`;

    // unique_inverse — values
    snippets[`unique_inverse_values_${dtype}`] = `
a = np.array(${JSON.stringify(dUniq)}, dtype=${npDtype(dtype)})
vals, inv = np.unique(a, return_inverse=True)
_result_orig = vals
result = _result_orig.astype(${ac})`;

    // unique_all — values
    snippets[`unique_all_values_${dtype}`] = `
a = np.array(${JSON.stringify(dUniq)}, dtype=${npDtype(dtype)})
vals, idx, inv, counts = np.unique(a, return_index=True, return_inverse=True, return_counts=True)
_result_orig = vals
result = _result_orig.astype(${ac})`;

    // apply_along_axis
    snippets[`apply_along_axis_${dtype}`] = `
a = np.array(${JSON.stringify(d2d)}, dtype=${npDtype(dtype)})
_result_orig = np.apply_along_axis(np.sum, 0, a)
result = _result_orig.astype(${ac})`;

    // apply_over_axes
    snippets[`apply_over_axes_${dtype}`] = `
a = np.array(${JSON.stringify(d2d)}, dtype=${npDtype(dtype)})
_result_orig = np.apply_over_axes(np.sum, a, [0])
result = _result_orig.astype(${ac})`;

    // place
    snippets[`place_${dtype}`] = `
a = np.array(${JSON.stringify(d1d)}, dtype=${npDtype(dtype)})
mask = np.array([True, False, True, False])
np.place(a, mask, np.array([99], dtype=${npDtype(dtype)}))
_result_orig = a
result = _result_orig.astype(${ac})`;

    // putmask
    snippets[`putmask_${dtype}`] = `
a = np.array(${JSON.stringify(d1d)}, dtype=${npDtype(dtype)})
mask = np.array([True, False, True, False])
np.putmask(a, mask, np.array([88], dtype=${npDtype(dtype)}))
_result_orig = a
result = _result_orig.astype(${ac})`;

    // copyto
    snippets[`copyto_${dtype}`] = `
dst = np.zeros(4, dtype=${npDtype(dtype)})
src = np.array(${JSON.stringify(d1d)}, dtype=${npDtype(dtype)})
np.copyto(dst, src)
_result_orig = dst
result = _result_orig.astype(${ac})`;

    // einsum trace
    snippets[`einsum_${dtype}`] = `
a = np.array(${JSON.stringify(d2d)}, dtype=${npDtype(dtype)})
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
    const d2d =
      dtype === 'bool'
        ? [
            [1, 0],
            [1, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ];
    snippets[`einsum_path_${dtype}`] = `
a = np.array(${JSON.stringify(d2d)}, dtype=${npDtype(dtype)})
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
    const d2d =
      dtype === 'bool'
        ? [
            [1, 0],
            [1, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ];

    it(`matrix_transpose ${dtype}`, () => {
      const jsResult = np.matrix_transpose(array(d2d, dtype));
      expectMatchPre(jsResult, oracle.get(`matrix_transpose_${dtype}`)!);
    });

    it(`permute_dims ${dtype}`, () => {
      const jsResult = np.permute_dims(array(d2d, dtype), [1, 0]);
      expectMatchPre(jsResult, oracle.get(`permute_dims_${dtype}`)!);
    });

    it(`rollaxis ${dtype}`, () => {
      const jsResult = np.rollaxis(array(d2d, dtype), 1, 0);
      expectMatchPre(jsResult, oracle.get(`rollaxis_${dtype}`)!);
    });

    it(`block ${dtype}`, () => {
      const dBlock = dtype === 'bool' ? [1, 0] : [1, 2];
      const a = array(dBlock, dtype);
      const jsResult = np.block([a, a]);
      expectMatchPre(jsResult, oracle.get(`block_${dtype}`)!);
    });
  }
});

describe('DType Sweep: Comparison/equality', () => {
  for (const dtype of ALL_DTYPES) {
    it(`array_equal ${dtype}`, () => {
      const d = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
      const a = array(d, dtype);
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
      const d = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
      const a = array(d, dtype);
      const jsResult = np.vdot(a, a);
      const pyResult = oracle.get(`vdot_${dtype}`)!;
      if (pyResult.error) throw new Error(`NumPy error: ${pyResult.error}`);
      scalarClose(jsResult, pyResult.value);
    });
  }
});

describe('DType Sweep: Unique variants', () => {
  for (const dtype of ALL_DTYPES) {
    const dUniq = dtype === 'bool' ? [1, 0, 1, 1, 0, 1] : [3, 1, 4, 1, 5, 3];

    it(`unique_values ${dtype}`, () => {
      const jsResult = np.unique_values(array(dUniq, dtype));
      expectMatchPre(jsResult, oracle.get(`unique_values_${dtype}`)!);
    });

    it(`unique_counts ${dtype}`, () => {
      const result = np.unique_counts(array(dUniq, dtype));
      expectMatchPre(result.values, oracle.get(`unique_counts_values_${dtype}`)!);
    });

    it(`unique_inverse ${dtype}`, () => {
      const result = np.unique_inverse(array(dUniq, dtype));
      expectMatchPre(result.values, oracle.get(`unique_inverse_values_${dtype}`)!);
    });

    it(`unique_all ${dtype}`, () => {
      const result = np.unique_all(array(dUniq, dtype));
      expectMatchPre(result.values, oracle.get(`unique_all_values_${dtype}`)!);
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
    const d2d =
      dtype === 'bool'
        ? [
            [1, 0],
            [1, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ];

    it(`apply_along_axis ${dtype}`, () => {
      const jsResult = np.apply_along_axis((arr: any): any => np.sum(arr), 0, array(d2d, dtype));
      expectMatchPre(jsResult, oracle.get(`apply_along_axis_${dtype}`)!);
    });

    it(`apply_over_axes ${dtype}`, () => {
      const jsResult = np.apply_over_axes(
        (arr: any, ax: number): any => np.sum(arr, ax, true),
        array(d2d, dtype),
        [0]
      );
      expectMatchPre(jsResult, oracle.get(`apply_over_axes_${dtype}`)!);
    });
  }
});

describe('DType Sweep: In-place mutation', () => {
  for (const dtype of ALL_DTYPES) {
    const d1d = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];

    it(`place ${dtype}`, () => {
      const a = array(d1d, dtype);
      const mask = array([1, 0, 1, 0], 'bool');
      const vals = array([dtype === 'bool' ? 1 : 99], dtype);
      np.place(a, mask, vals);
      expectMatchPre(a, oracle.get(`place_${dtype}`)!);
    });

    it(`putmask ${dtype}`, () => {
      const a = array(d1d, dtype);
      const mask = array([1, 0, 1, 0], 'bool');
      const vals = array([dtype === 'bool' ? 1 : 88], dtype);
      np.putmask(a, mask, vals);
      expectMatchPre(a, oracle.get(`putmask_${dtype}`)!);
    });

    it(`copyto ${dtype}`, () => {
      const dst = np.zeros([4], dtype);
      const src = array(d1d, dtype);
      np.copyto(dst, src);
      expectMatchPre(dst, oracle.get(`copyto_${dtype}`)!);
    });
  }
});

describe('DType Sweep: einsum', () => {
  for (const dtype of ALL_DTYPES) {
    it(`einsum trace ${dtype}`, () => {
      const d2d =
        dtype === 'bool'
          ? [
              [1, 0],
              [1, 1],
            ]
          : [
              [1, 2],
              [3, 4],
            ];
      const jsResult = np.einsum('ii', array(d2d, dtype));
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
          pyResult
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
          pyResult
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
        const d2d =
          dtype === 'bool'
            ? [
                [1, 0],
                [1, 1],
              ]
            : [
                [1, 2],
                [3, 4],
              ];
        const a = array(d2d, dtype);
        const pyResult = oracle.get(`einsum_path_${dtype}`)!;
        const r = expectBothRejectPre(
          `einsum_path may not support ${dtype}`,
          () => np.einsum_path('ij,jk->ik', a, a),
          pyResult
        );
        if (r === 'both-reject') return;
        const [path, info] = np.einsum_path('ij,jk->ik', a, a);
        expect(Array.isArray(path)).toBe(true);
        expect(typeof info).toBe('string');
      });
    }
  });
});

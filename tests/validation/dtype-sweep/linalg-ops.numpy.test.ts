/**
 * DType Sweep: linalg utility & product functions (np.linalg.*).
 * Lighter ops: matrix_rank, matrix_power, pinv, cond, slogdet, svdvals, multi_dot,
 * norms, tensorinv, tensorsolve, cross, vecdot, dot, inner, outer, matmul, tensordot,
 * trace, diagonal, transpose, matrix_transpose, permute_dims.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectBothReject,
  expectBothRejectPre,
  expectComplexFixture,
  expectMatchPre,
  npDtype,
  pyArrayCast,
  pyScalarCast,
  runNumPyBatch,
  scalarClose,
} from './_helpers';

const { array } = np;

/**
 * Single definition of every fixture, read by both the oracle snippets and the
 * test bodies. Declaring the same logical input twice lets the two sides drift
 * apart and compare different arrays. Complex dtypes carry a real imaginary
 * part here, so the imaginary half of each op is actually exercised.
 */
function fixtures(dtype: string) {
  const b = dtype === 'bool';
  return {
    mat: asDtypeData(
      b
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ],
      dtype,
    ),
    vec: asDtypeData(b ? [1, 0] : [3, 4], dtype),
    id: asDtypeData(
      [
        [1, 0],
        [0, 1],
      ],
      dtype,
    ),
    tsB: asDtypeData(b ? [1, 0] : [1, 2], dtype),
    crossA: asDtypeData(b ? [1, 0, 1] : [1, 2, 3], dtype),
    crossB: asDtypeData(b ? [0, 1, 0] : [4, 5, 6], dtype),
    dotA: asDtypeData(b ? [1, 0] : [1, 2], dtype),
    dotB: asDtypeData(b ? [0, 1] : [3, 4], dtype),
  };
}

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const f = fixtures(dtype);
    const sc = pyScalarCast(dtype);
    const ac = pyArrayCast(dtype);
    const dt = npDtype(dtype);

    snippets[`linalg_matrix_rank_${dtype}`] =
      `result = int(np.linalg.matrix_rank(np.array(${f.mat.py}, dtype=${dt})))`;

    snippets[`linalg_matrix_power_${dtype}`] = `
a = np.array(${f.mat.py}, dtype=${dt})
_result_orig = np.linalg.matrix_power(a, 2)
result = _result_orig.astype(${ac})`;

    snippets[`linalg_pinv_${dtype}`] = `
_result_orig = np.linalg.pinv(np.array(${f.mat.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_cond_${dtype}`] =
      `result = ${sc}(np.linalg.cond(np.array(${f.mat.py}, dtype=${dt})))`;

    snippets[`linalg_slogdet_${dtype}`] = `
sign, logabsdet = np.linalg.slogdet(np.array(${f.mat.py}, dtype=${dt}))
result = np.array([${sc}(sign), ${sc}(logabsdet)])`;

    snippets[`linalg_svdvals_${dtype}`] = `
_result_orig = np.linalg.svdvals(np.array(${f.mat.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_multi_dot_${dtype}`] = `
a = np.array(${f.mat.py}, dtype=${dt})
_result_orig = np.linalg.multi_dot([a, a])
result = _result_orig.astype(${ac})`;

    snippets[`linalg_vector_norm_${dtype}`] =
      `result = ${sc}(np.linalg.vector_norm(np.array(${f.vec.py}, dtype=${dt})))`;

    snippets[`linalg_matrix_norm_${dtype}`] =
      `result = ${sc}(np.linalg.matrix_norm(np.array(${f.mat.py}, dtype=${dt})))`;

    snippets[`linalg_tensorinv_${dtype}`] = `
_result_orig = np.linalg.tensorinv(np.array(${f.id.py}, dtype=${dt}), ind=1)
result = _result_orig.astype(${ac})`;

    snippets[`linalg_tensorsolve_${dtype}`] = `
_result_orig = np.linalg.tensorsolve(np.array(${f.id.py}, dtype=${dt}), np.array(${f.tsB.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_cross_${dtype}`] = `
a = np.array(${f.crossA.py}, dtype=${dt})
b = np.array(${f.crossB.py}, dtype=${dt})
_result_orig = np.cross(a, b)
result = _result_orig.astype(${ac})`;

    snippets[`linalg_vecdot_${dtype}`] =
      `result = ${sc}(np.vecdot(np.array(${f.crossA.py}, dtype=${dt}), np.array(${f.crossB.py}, dtype=${dt})))`;

    snippets[`linalg_dot_${dtype}`] =
      `result = ${sc}(np.dot(np.array(${f.dotA.py}, dtype=${dt}), np.array(${f.dotB.py}, dtype=${dt})))`;

    snippets[`linalg_inner_${dtype}`] =
      `result = ${sc}(np.inner(np.array(${f.dotA.py}, dtype=${dt}), np.array(${f.dotB.py}, dtype=${dt})))`;

    snippets[`linalg_outer_${dtype}`] = `
_result_orig = np.outer(np.array(${f.dotA.py}, dtype=${dt}), np.array(${f.dotB.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_matmul_${dtype}`] = `
a = np.array(${f.mat.py}, dtype=${dt})
_result_orig = np.matmul(a, a)
result = _result_orig.astype(${ac})`;

    snippets[`linalg_tensordot_${dtype}`] = `
a = np.array(${f.mat.py}, dtype=${dt})
result = ${sc}(np.tensordot(a, a))`;

    snippets[`linalg_trace_${dtype}`] =
      `result = ${sc}(np.trace(np.array(${f.mat.py}, dtype=${dt})))`;

    snippets[`linalg_diagonal_${dtype}`] = `
_result_orig = np.diagonal(np.array(${f.mat.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_transpose_${dtype}`] = `
_result_orig = np.transpose(np.array(${f.mat.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_matrix_transpose_${dtype}`] = `
_result_orig = np.array(${f.mat.py}, dtype=${dt}).T
result = _result_orig.astype(${ac})`;

    snippets[`linalg_permute_dims_${dtype}`] = `
_result_orig = np.transpose(np.array(${f.mat.py}, dtype=${dt}), [1, 0])
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: linalg ops', () => {
  for (const dtype of ALL_DTYPES) {
    const f = fixtures(dtype);

    it(`linalg.matrix_rank ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `matrix_rank ${dtype}`);
      const py = oracle.get(`linalg_matrix_rank_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.matrix_rank(a),
        py,
      );
      if (r === 'both-reject') return;
      expect(Number(np.linalg.matrix_rank(a))).toBe(Number(py.value));
    });

    it(`linalg.matrix_power ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `matrix_power ${dtype}`);
      const py = oracle.get(`linalg_matrix_power_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.matrix_power(a, 2),
        py,
      );
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.matrix_power(a, 2), py, { rtol: 1e-4 });
    });

    it(`linalg.pinv ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `pinv ${dtype}`);
      const py = oracle.get(`linalg_pinv_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.pinv(a), py);
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.pinv(a), py, { rtol: 1e-4, atol: 1e-6 });
    });

    it(`linalg.cond ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `cond ${dtype}`);
      const py = oracle.get(`linalg_cond_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.cond(a), py);
      if (r === 'both-reject') return;
      scalarClose(np.linalg.cond(a), py.value);
    });

    it(`linalg.slogdet ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `slogdet ${dtype}`);
      const py = oracle.get(`linalg_slogdet_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.slogdet(a),
        py,
      );
      if (r === 'both-reject') return;
      const { sign, logabsdet } = np.linalg.slogdet(a) as any;
      scalarClose(sign, py.value[0]);
      scalarClose(logabsdet, py.value[1]);
    });

    it(`linalg.svdvals ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `svdvals ${dtype}`);
      const py = oracle.get(`linalg_svdvals_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.svdvals(a),
        py,
      );
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.svdvals(a), py, { rtol: 1e-4 });
    });

    it(`linalg.multi_dot ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `multi_dot ${dtype}`);
      expectMatchPre(np.linalg.multi_dot([a, a]), oracle.get(`linalg_multi_dot_${dtype}`)!, {
        rtol: 1e-4,
      });
    });

    it(`linalg.vector_norm ${dtype}`, () => {
      const a = array(f.vec.js, dtype);
      expectComplexFixture(a, dtype, `vector_norm ${dtype}`);
      scalarClose(np.linalg.vector_norm(a), oracle.get(`linalg_vector_norm_${dtype}`)!.value);
    });

    it(`linalg.matrix_norm ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `matrix_norm ${dtype}`);
      const py = oracle.get(`linalg_matrix_norm_${dtype}`)!;
      scalarClose(np.linalg.matrix_norm(a), py.value, dtype === 'float16' ? 2 : 4);
    });

    it(`linalg.tensorinv ${dtype}`, () => {
      const a = array(f.id.js, dtype);
      expectComplexFixture(a, dtype, `tensorinv ${dtype}`);
      const py = oracle.get(`linalg_tensorinv_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.tensorinv(a, 1),
        py,
      );
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.tensorinv(a, 1), py, { rtol: 1e-4 });
    });

    it(`linalg.tensorsolve ${dtype}`, () => {
      const a = array(f.id.js, dtype);
      const b = array(f.tsB.js, dtype);
      expectComplexFixture(a, dtype, `tensorsolve ${dtype}`);
      const py = oracle.get(`linalg_tensorsolve_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.tensorsolve(a, b),
        py,
      );
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.tensorsolve(a, b), py, { rtol: 1e-4 });
    });

    it(`linalg.cross ${dtype}`, () => {
      const a = array(f.crossA.js, dtype);
      const b = array(f.crossB.js, dtype);
      expectComplexFixture(a, dtype, `cross ${dtype}`);
      if (dtype === 'bool') {
        const pyCode = `
a = np.array(${f.crossA.py}, dtype=${npDtype(dtype)})
b = np.array(${f.crossB.py}, dtype=${npDtype(dtype)})
result = np.cross(a, b)`;
        const _r = expectBothReject(
          'cross uses subtract internally, not supported for bool',
          () => np.linalg.cross(a, b),
          pyCode,
        );
        if (_r === 'both-reject') return;
      }
      expectMatchPre(np.linalg.cross(a, b), oracle.get(`linalg_cross_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`linalg.vecdot ${dtype}`, () => {
      const a = array(f.crossA.js, dtype);
      const b = array(f.crossB.js, dtype);
      expectComplexFixture(a, dtype, `vecdot ${dtype}`);
      scalarClose(np.linalg.vecdot(a, b), oracle.get(`linalg_vecdot_${dtype}`)!.value);
    });

    it(`linalg.dot ${dtype}`, () => {
      const a = array(f.dotA.js, dtype);
      const b = array(f.dotB.js, dtype);
      expectComplexFixture(a, dtype, `dot ${dtype}`);
      scalarClose(np.linalg.dot(a, b), oracle.get(`linalg_dot_${dtype}`)!.value);
    });

    it(`linalg.inner ${dtype}`, () => {
      const a = array(f.dotA.js, dtype);
      const b = array(f.dotB.js, dtype);
      expectComplexFixture(a, dtype, `inner ${dtype}`);
      scalarClose(np.linalg.inner(a, b), oracle.get(`linalg_inner_${dtype}`)!.value);
    });

    it(`linalg.outer ${dtype}`, () => {
      const a = array(f.dotA.js, dtype);
      const b = array(f.dotB.js, dtype);
      expectComplexFixture(a, dtype, `outer ${dtype}`);
      expectMatchPre(np.linalg.outer(a, b), oracle.get(`linalg_outer_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`linalg.matmul ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `matmul ${dtype}`);
      expectMatchPre(np.linalg.matmul(a, a), oracle.get(`linalg_matmul_${dtype}`)!, {
        rtol: 1e-4,
      });
    });

    it(`linalg.tensordot ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `tensordot ${dtype}`);
      scalarClose(np.linalg.tensordot(a, a), oracle.get(`linalg_tensordot_${dtype}`)!.value);
    });

    it(`linalg.trace ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `trace ${dtype}`);
      scalarClose(np.linalg.trace(a), oracle.get(`linalg_trace_${dtype}`)!.value);
    });

    it(`linalg.diagonal ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `diagonal ${dtype}`);
      expectMatchPre(np.linalg.diagonal(a), oracle.get(`linalg_diagonal_${dtype}`)!);
    });

    it(`linalg.transpose ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `transpose ${dtype}`);
      expectMatchPre(np.linalg.transpose(a), oracle.get(`linalg_transpose_${dtype}`)!);
    });

    it(`linalg.matrix_transpose ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `matrix_transpose ${dtype}`);
      expectMatchPre(
        np.linalg.matrix_transpose(a),
        oracle.get(`linalg_matrix_transpose_${dtype}`)!,
      );
    });

    it(`linalg.permute_dims ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `permute_dims ${dtype}`);
      expectMatchPre(
        np.linalg.permute_dims(a, [1, 0]),
        oracle.get(`linalg_permute_dims_${dtype}`)!,
      );
    });
  }
});

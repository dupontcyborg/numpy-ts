/**
 * DType Sweep: linalg utility & product functions (np.linalg.*).
 * Lighter ops: matrix_rank, matrix_power, pinv, cond, slogdet, svdvals, multi_dot,
 * norms, tensorinv, tensorsolve, cross, vecdot, dot, inner, outer, matmul, tensordot,
 * trace, diagonal, transpose, matrix_transpose, permute_dims.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  arraysClose,
  checkNumPyAvailable,
  npDtype,
  pyScalarCast,
  pyArrayCast,
  toComparable,
  scalarClose,
  expectBothReject,
  expectBothRejectPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const mat =
      dtype === 'bool'
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ];
    const vec = dtype === 'bool' ? [1, 0] : [3, 4];
    const sc = pyScalarCast(dtype);
    const ac = pyArrayCast(dtype);

    snippets[`linalg_matrix_rank_${dtype}`] =
      `result = int(np.linalg.matrix_rank(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_matrix_power_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.linalg.matrix_power(a, 2).astype(${ac})`;

    snippets[`linalg_pinv_${dtype}`] = `
result = np.linalg.pinv(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(${ac})`;

    snippets[`linalg_cond_${dtype}`] =
      `result = ${sc}(np.linalg.cond(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_slogdet_${dtype}`] = `
sign, logabsdet = np.linalg.slogdet(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.array([${sc}(sign), ${sc}(logabsdet)])`;

    snippets[`linalg_svdvals_${dtype}`] = `
result = np.linalg.svdvals(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(${ac})`;

    snippets[`linalg_multi_dot_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.linalg.multi_dot([a, a]).astype(${ac})`;

    snippets[`linalg_vector_norm_${dtype}`] =
      `result = ${sc}(np.linalg.vector_norm(np.array(${JSON.stringify(vec)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_matrix_norm_${dtype}`] =
      `result = ${sc}(np.linalg.matrix_norm(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    const id = [
      [1, 0],
      [0, 1],
    ];
    snippets[`linalg_tensorinv_${dtype}`] = `
result = np.linalg.tensorinv(np.array(${JSON.stringify(id)}, dtype=${npDtype(dtype)}), ind=1).astype(${ac})`;

    const tsB = dtype === 'bool' ? [1, 0] : [1, 2];
    snippets[`linalg_tensorsolve_${dtype}`] = `
result = np.linalg.tensorsolve(np.array(${JSON.stringify(id)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(tsB)}, dtype=${npDtype(dtype)})).astype(${ac})`;

    const crossA = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    const crossB = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
    snippets[`linalg_cross_${dtype}`] = `
a = np.array(${JSON.stringify(crossA)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(crossB)}, dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(${ac})`;

    snippets[`linalg_vecdot_${dtype}`] = `
result = ${sc}(np.vecdot(np.array(${JSON.stringify(crossA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(crossB)}, dtype=${npDtype(dtype)})))`;

    const dotA = dtype === 'bool' ? [1, 0] : [1, 2];
    const dotB = dtype === 'bool' ? [0, 1] : [3, 4];
    snippets[`linalg_dot_${dtype}`] = `
result = ${sc}(np.dot(np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_inner_${dtype}`] = `
result = ${sc}(np.inner(np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_outer_${dtype}`] = `
result = np.outer(np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})).astype(${ac})`;

    snippets[`linalg_matmul_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.matmul(a, a).astype(${ac})`;

    snippets[`linalg_tensordot_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = ${sc}(np.tensordot(a, a))`;

    snippets[`linalg_trace_${dtype}`] =
      `result = ${sc}(np.trace(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_diagonal_${dtype}`] =
      `result = np.diagonal(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(${ac})`;

    snippets[`linalg_transpose_${dtype}`] =
      `result = np.transpose(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(${ac})`;

    snippets[`linalg_matrix_transpose_${dtype}`] =
      `result = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}).T.astype(${ac})`;

    snippets[`linalg_permute_dims_${dtype}`] =
      `result = np.transpose(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}), [1, 0]).astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: linalg ops', () => {
  for (const dtype of ALL_DTYPES) {
    const mat =
      dtype === 'bool'
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ];
    const vec = dtype === 'bool' ? [1, 0] : [3, 4];

    it(`linalg.matrix_rank ${dtype}`, () => {
      const py = oracle.get(`linalg_matrix_rank_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.matrix_rank(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      expect(Number(np.linalg.matrix_rank(array(mat, dtype)))).toBe(Number(py.value));
    });

    it(`linalg.matrix_power ${dtype}`, () => {
      const py = oracle.get(`linalg_matrix_power_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.matrix_power(array(mat, dtype), 2),
        py
      );
      if (r === 'both-reject') return;
      expect(
        arraysClose(toComparable(np.linalg.matrix_power(array(mat, dtype), 2)), py.value, 1e-4)
      ).toBe(true);
    });

    it(`linalg.pinv ${dtype}`, () => {
      const py = oracle.get(`linalg_pinv_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.pinv(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      expect(arraysClose(toComparable(np.linalg.pinv(array(mat, dtype))), py.value, 1e-4)).toBe(
        true
      );
    });

    it(`linalg.cond ${dtype}`, () => {
      const py = oracle.get(`linalg_cond_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.cond(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      scalarClose(np.linalg.cond(array(mat, dtype)), py.value);
    });

    it(`linalg.slogdet ${dtype}`, () => {
      const py = oracle.get(`linalg_slogdet_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.slogdet(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      const { sign, logabsdet } = np.linalg.slogdet(array(mat, dtype)) as any;
      scalarClose(sign, py.value[0]);
      scalarClose(logabsdet, py.value[1]);
    });

    it(`linalg.svdvals ${dtype}`, () => {
      const py = oracle.get(`linalg_svdvals_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.svdvals(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      expect(arraysClose(toComparable(np.linalg.svdvals(array(mat, dtype))), py.value, 1e-4)).toBe(
        true
      );
    });

    it(`linalg.multi_dot ${dtype}`, () => {
      const a = array(mat, dtype);
      const jsResult = np.linalg.multi_dot([a, a]);
      const py = oracle.get(`linalg_multi_dot_${dtype}`)!;
      expect(arraysClose(toComparable(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.vector_norm ${dtype}`, () => {
      const jsResult = np.linalg.vector_norm(array(vec, dtype));
      const py = oracle.get(`linalg_vector_norm_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`linalg.matrix_norm ${dtype}`, () => {
      const jsResult = np.linalg.matrix_norm(array(mat, dtype));
      const py = oracle.get(`linalg_matrix_norm_${dtype}`)!;
      scalarClose(jsResult, py.value, dtype === 'float16' ? 2 : 4);
    });

    it(`linalg.tensorinv ${dtype}`, () => {
      const id = [
        [1, 0],
        [0, 1],
      ];
      const py = oracle.get(`linalg_tensorinv_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.tensorinv(array(id, dtype), 1),
        py
      );
      if (r === 'both-reject') return;
      expect(
        arraysClose(toComparable(np.linalg.tensorinv(array(id, dtype), 1)), py.value, 1e-4)
      ).toBe(true);
    });

    it(`linalg.tensorsolve ${dtype}`, () => {
      const id = [
        [1, 0],
        [0, 1],
      ];
      const b = dtype === 'bool' ? [1, 0] : [1, 2];
      const py = oracle.get(`linalg_tensorsolve_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.tensorsolve(array(id, dtype), array(b, dtype)),
        py
      );
      if (r === 'both-reject') return;
      expect(
        arraysClose(
          toComparable(np.linalg.tensorsolve(array(id, dtype), array(b, dtype))),
          py.value,
          1e-4
        )
      ).toBe(true);
    });

    it(`linalg.cross ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      if (dtype === 'bool') {
        const pyCode = `
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = np.cross(a, b)`;
        const _r = expectBothReject(
          'cross uses subtract internally, not supported for bool',
          () => np.linalg.cross(array(a, dtype), array(b, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.linalg.cross(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_cross_${dtype}`)!;
      expect(arraysClose(toComparable(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.vecdot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.linalg.vecdot(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_vecdot_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`linalg.dot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.dot(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_dot_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`linalg.inner ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.inner(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_inner_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`linalg.outer ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.outer(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_outer_${dtype}`)!;
      expect(arraysClose(toComparable(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.matmul ${dtype}`, () => {
      const jsResult = np.linalg.matmul(array(mat, dtype), array(mat, dtype));
      const py = oracle.get(`linalg_matmul_${dtype}`)!;
      expect(arraysClose(toComparable(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.tensordot ${dtype}`, () => {
      const jsResult = np.linalg.tensordot(array(mat, dtype), array(mat, dtype));
      const py = oracle.get(`linalg_tensordot_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`linalg.trace ${dtype}`, () => {
      const jsResult = np.linalg.trace(array(mat, dtype));
      const py = oracle.get(`linalg_trace_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`linalg.diagonal ${dtype}`, () => {
      const jsResult = np.linalg.diagonal(array(mat, dtype));
      const py = oracle.get(`linalg_diagonal_${dtype}`)!;
      expect(arraysClose(toComparable(jsResult), py.value)).toBe(true);
    });

    it(`linalg.transpose ${dtype}`, () => {
      const jsResult = np.linalg.transpose(array(mat, dtype));
      const py = oracle.get(`linalg_transpose_${dtype}`)!;
      expect(arraysClose(toComparable(jsResult), py.value)).toBe(true);
    });

    it(`linalg.matrix_transpose ${dtype}`, () => {
      const jsResult = np.linalg.matrix_transpose(array(mat, dtype));
      const py = oracle.get(`linalg_matrix_transpose_${dtype}`)!;
      expect(arraysClose(toComparable(jsResult), py.value)).toBe(true);
    });

    it(`linalg.permute_dims ${dtype}`, () => {
      const jsResult = np.linalg.permute_dims(array(mat, dtype), [1, 0]);
      const py = oracle.get(`linalg_permute_dims_${dtype}`)!;
      expect(arraysClose(toComparable(jsResult), py.value)).toBe(true);
    });
  }
});

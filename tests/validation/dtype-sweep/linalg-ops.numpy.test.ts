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
  expectBothReject,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

function toNumbers(arr: any): any {
  const flat = arr.toArray();
  return JSON.parse(
    JSON.stringify(flat, (_k: string, v: any) => (typeof v === 'bigint' ? Number(v) : v))
  );
}

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

    snippets[`linalg_matrix_rank_${dtype}`] =
      `result = int(np.linalg.matrix_rank(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_matrix_power_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.linalg.matrix_power(a, 2).astype(np.float64)`;

    snippets[`linalg_pinv_${dtype}`] = `
result = np.linalg.pinv(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    snippets[`linalg_cond_${dtype}`] =
      `result = float(np.linalg.cond(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_slogdet_${dtype}`] = `
sign, logabsdet = np.linalg.slogdet(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.array([float(sign), float(logabsdet)])`;

    snippets[`linalg_svdvals_${dtype}`] = `
result = np.linalg.svdvals(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    snippets[`linalg_multi_dot_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.linalg.multi_dot([a, a]).astype(np.float64)`;

    snippets[`linalg_vector_norm_${dtype}`] =
      `result = float(np.linalg.vector_norm(np.array(${JSON.stringify(vec)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_matrix_norm_${dtype}`] =
      `result = float(np.linalg.matrix_norm(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    const id = [
      [1, 0],
      [0, 1],
    ];
    snippets[`linalg_tensorinv_${dtype}`] = `
result = np.linalg.tensorinv(np.array(${JSON.stringify(id)}, dtype=${npDtype(dtype)}), ind=1).astype(np.float64)`;

    const tsB = dtype === 'bool' ? [1, 0] : [1, 2];
    snippets[`linalg_tensorsolve_${dtype}`] = `
result = np.linalg.tensorsolve(np.array(${JSON.stringify(id)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(tsB)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    const crossA = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    const crossB = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
    snippets[`linalg_cross_${dtype}`] = `
a = np.array(${JSON.stringify(crossA)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(crossB)}, dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(np.float64)`;

    snippets[`linalg_vecdot_${dtype}`] = `
result = float(np.vecdot(np.array(${JSON.stringify(crossA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(crossB)}, dtype=${npDtype(dtype)})))`;

    const dotA = dtype === 'bool' ? [1, 0] : [1, 2];
    const dotB = dtype === 'bool' ? [0, 1] : [3, 4];
    snippets[`linalg_dot_${dtype}`] = `
result = float(np.dot(np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_inner_${dtype}`] = `
result = float(np.inner(np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_outer_${dtype}`] = `
result = np.outer(np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    snippets[`linalg_matmul_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.matmul(a, a).astype(np.float64)`;

    snippets[`linalg_tensordot_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = float(np.tensordot(a, a))`;

    snippets[`linalg_trace_${dtype}`] =
      `result = float(np.trace(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_diagonal_${dtype}`] =
      `result = np.diagonal(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    snippets[`linalg_transpose_${dtype}`] =
      `result = np.transpose(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    snippets[`linalg_matrix_transpose_${dtype}`] =
      `result = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}).T.astype(np.float64)`;

    snippets[`linalg_permute_dims_${dtype}`] =
      `result = np.transpose(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}), [1, 0]).astype(np.float64)`;
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
      const jsResult = np.linalg.matrix_rank(array(mat, dtype));
      const py = oracle.get(`linalg_matrix_rank_${dtype}`)!;
      expect(Number(jsResult)).toBe(Number(py.value));
    });

    it(`linalg.matrix_power ${dtype}`, () => {
      const jsResult = np.linalg.matrix_power(array(mat, dtype), 2);
      const py = oracle.get(`linalg_matrix_power_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.pinv ${dtype}`, () => {
      const jsResult = np.linalg.pinv(array(mat, dtype));
      const py = oracle.get(`linalg_pinv_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.cond ${dtype}`, () => {
      const jsResult = np.linalg.cond(array(mat, dtype));
      const py = oracle.get(`linalg_cond_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 2);
    });

    it(`linalg.slogdet ${dtype}`, () => {
      const { sign, logabsdet } = np.linalg.slogdet(array(mat, dtype)) as any;
      const py = oracle.get(`linalg_slogdet_${dtype}`)!;
      expect(Number(sign)).toBeCloseTo(Number(py.value[0]), 4);
      expect(Number(logabsdet)).toBeCloseTo(Number(py.value[1]), 4);
    });

    it(`linalg.svdvals ${dtype}`, () => {
      const jsResult = np.linalg.svdvals(array(mat, dtype));
      const py = oracle.get(`linalg_svdvals_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.multi_dot ${dtype}`, () => {
      const a = array(mat, dtype);
      const jsResult = np.linalg.multi_dot([a, a]);
      const py = oracle.get(`linalg_multi_dot_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.vector_norm ${dtype}`, () => {
      const jsResult = np.linalg.vector_norm(array(vec, dtype));
      const py = oracle.get(`linalg_vector_norm_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.matrix_norm ${dtype}`, () => {
      const jsResult = np.linalg.matrix_norm(array(mat, dtype));
      const py = oracle.get(`linalg_matrix_norm_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.tensorinv ${dtype}`, () => {
      const id = [
        [1, 0],
        [0, 1],
      ];
      const jsResult = np.linalg.tensorinv(array(id, dtype), 1);
      const py = oracle.get(`linalg_tensorinv_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.tensorsolve ${dtype}`, () => {
      const id = [
        [1, 0],
        [0, 1],
      ];
      const b = dtype === 'bool' ? [1, 0] : [1, 2];
      const jsResult = np.linalg.tensorsolve(array(id, dtype), array(b, dtype));
      const py = oracle.get(`linalg_tensorsolve_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.cross ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      if (dtype === 'bool') {
        const pyCode = `
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(np.float64)`;
        const _r = expectBothReject(
          'cross uses subtract internally, not supported for bool',
          () => np.linalg.cross(array(a, dtype), array(b, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.linalg.cross(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_cross_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.vecdot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.linalg.vecdot(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_vecdot_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.dot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.dot(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_dot_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.inner ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.inner(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_inner_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.outer ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.outer(array(a, dtype), array(b, dtype));
      const py = oracle.get(`linalg_outer_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.matmul ${dtype}`, () => {
      const jsResult = np.linalg.matmul(array(mat, dtype), array(mat, dtype));
      const py = oracle.get(`linalg_matmul_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.tensordot ${dtype}`, () => {
      const jsResult = np.linalg.tensordot(array(mat, dtype), array(mat, dtype));
      const py = oracle.get(`linalg_tensordot_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.trace ${dtype}`, () => {
      const jsResult = np.linalg.trace(array(mat, dtype));
      const py = oracle.get(`linalg_trace_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.diagonal ${dtype}`, () => {
      const jsResult = np.linalg.diagonal(array(mat, dtype));
      const py = oracle.get(`linalg_diagonal_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value)).toBe(true);
    });

    it(`linalg.transpose ${dtype}`, () => {
      const jsResult = np.linalg.transpose(array(mat, dtype));
      const py = oracle.get(`linalg_transpose_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value)).toBe(true);
    });

    it(`linalg.matrix_transpose ${dtype}`, () => {
      const jsResult = np.linalg.matrix_transpose(array(mat, dtype));
      const py = oracle.get(`linalg_matrix_transpose_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value)).toBe(true);
    });

    it(`linalg.permute_dims ${dtype}`, () => {
      const jsResult = np.linalg.permute_dims(array(mat, dtype), [1, 0]);
      const py = oracle.get(`linalg_permute_dims_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value)).toBe(true);
    });
  }
});

/**
 * DType Sweep: Top-level linear algebra functions (np.dot, np.cross, etc.).
 * Tests across ALL dtypes, validated against NumPy.
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

// Helper: convert bigint array values to numbers for comparison
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
    const dotA = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    const dotB = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];

    snippets[`dot_${dtype}`] = `
a = np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})
result = float(np.dot(a, b))`;

    snippets[`inner_${dtype}`] = `
a = np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})
result = float(np.inner(a, b))`;

    const outerA = dtype === 'bool' ? [1, 0] : [1, 2, 3];
    snippets[`outer_${dtype}`] = `
a = np.array(${JSON.stringify(outerA)}, dtype=${npDtype(dtype)})
result = np.outer(a, a).astype(np.float64)`;

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

    snippets[`matmul_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.matmul(a, a).astype(np.float64)`;

    snippets[`cross_${dtype}`] = `
a = np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(np.float64)`;

    const kronA = dtype === 'bool' ? [1, 0] : [1, 2];
    const kronB = dtype === 'bool' ? [0, 1] : [3, 4];
    snippets[`kron_${dtype}`] = `
a = np.array(${JSON.stringify(kronA)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(kronB)}, dtype=${npDtype(dtype)})
result = np.kron(a, b).astype(np.float64)`;

    snippets[`trace_${dtype}`] =
      `result = float(np.trace(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`tensordot_${dtype}`] = `
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = float(np.tensordot(a, a))`;

    snippets[`vecdot_${dtype}`] = `
a = np.array(${JSON.stringify(dotA)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(dotB)}, dtype=${npDtype(dtype)})
result = float(np.vecdot(a, b))`;

    const mvM = mat;
    const mvV = dtype === 'bool' ? [1, 0] : [5, 6];
    snippets[`matvec_${dtype}`] = `
m = np.array(${JSON.stringify(mvM)}, dtype=${npDtype(dtype)})
v = np.array(${JSON.stringify(mvV)}, dtype=${npDtype(dtype)})
result = np.matvec(m, v).astype(np.float64)`;

    snippets[`vecmat_${dtype}`] = `
m = np.array(${JSON.stringify(mvM)}, dtype=${npDtype(dtype)})
v = np.array(${JSON.stringify(mvV)}, dtype=${npDtype(dtype)})
result = np.vecmat(v, m).astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Top-level linalg', () => {
  for (const dtype of ALL_DTYPES) {
    it(`dot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.dot(array(a, dtype), array(b, dtype));
      const py = oracle.get(`dot_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`inner ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.inner(array(a, dtype), array(b, dtype));
      const py = oracle.get(`inner_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`outer ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2, 3], dtype);
      const r = np.outer(a, a);
      const py = oracle.get(`outer_${dtype}`)!;
      expect(arraysClose(toNumbers(r), py.value, 1e-4)).toBe(true);
    });

    it(`matmul ${dtype}`, () => {
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
      const jsResult = np.matmul(array(mat, dtype), array(mat, dtype));
      const py = oracle.get(`matmul_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`cross ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      // cross(bool): NumPy rejects — "boolean subtract not supported"
      if (dtype === 'bool') {
        const pyCode = `
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(np.float64)`;
        const _r = expectBothReject(
          'cross uses subtract internally, not supported for bool',
          () => np.cross(array(a, dtype), array(b, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const r = np.cross(array(a, dtype), array(b, dtype));
      const py = oracle.get(`cross_${dtype}`)!;
      expect(arraysClose(toNumbers(r), py.value, 1e-4)).toBe(true);
    });

    it(`kron ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const r = np.kron(array(a, dtype), array(b, dtype));
      const py = oracle.get(`kron_${dtype}`)!;
      expect(arraysClose(toNumbers(r), py.value, 1e-4)).toBe(true);
    });

    it(`trace ${dtype}`, () => {
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
      const jsResult = np.trace(array(mat, dtype));
      const py = oracle.get(`trace_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`tensordot ${dtype}`, () => {
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
      const r = np.tensordot(array(mat, dtype), array(mat, dtype));
      const py = oracle.get(`tensordot_${dtype}`)!;
      expect(Number(r)).toBeCloseTo(Number(py.value), 4);
    });

    it(`vecdot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.vecdot(array(a, dtype), array(b, dtype));
      const py = oracle.get(`vecdot_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`matvec ${dtype}`, () => {
      const m =
        dtype === 'bool'
          ? [
              [1, 0],
              [0, 1],
            ]
          : [
              [1, 2],
              [3, 4],
            ];
      const v = dtype === 'bool' ? [1, 0] : [5, 6];
      const jsResult = np.matvec(array(m, dtype), array(v, dtype));
      const py = oracle.get(`matvec_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`vecmat ${dtype}`, () => {
      const m =
        dtype === 'bool'
          ? [
              [1, 0],
              [0, 1],
            ]
          : [
              [1, 2],
              [3, 4],
            ];
      const v = dtype === 'bool' ? [1, 0] : [5, 6];
      const jsResult = np.vecmat(array(v, dtype), array(m, dtype));
      const py = oracle.get(`vecmat_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });
  }
});

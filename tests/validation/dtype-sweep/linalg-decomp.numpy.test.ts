/**
 * DType Sweep: linalg decomposition & solve functions (np.linalg.*).
 * Heavy ops: norm, det, inv, cholesky, qr, svd, eig, eigh, eigvals, eigvalsh, solve, lstsq.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPyBatch, arraysClose, checkNumPyAvailable, npDtype } from './_helpers';
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

    snippets[`linalg_norm_${dtype}`] =
      `result = float(np.linalg.norm(np.array(${JSON.stringify(vec)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_det_${dtype}`] =
      `result = float(np.linalg.det(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_inv_${dtype}`] =
      `result = np.linalg.inv(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    const pdMat =
      dtype === 'bool'
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [4, 2],
            [2, 5],
          ];
    snippets[`linalg_cholesky_${dtype}`] =
      `result = np.linalg.cholesky(np.array(${JSON.stringify(pdMat)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    snippets[`linalg_qr_${dtype}`] =
      `result = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}).astype(np.float64)`;

    snippets[`linalg_svd_${dtype}`] = `
u, s, vh = np.linalg.svd(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = s.astype(np.float64)`;

    snippets[`linalg_eig_${dtype}`] = `
w, v = np.linalg.eig(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.sort(np.abs(w)).astype(np.float64)`;

    const symMat =
      dtype === 'bool'
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [2, 1],
            [1, 3],
          ];
    snippets[`linalg_eigh_${dtype}`] = `
w, v = np.linalg.eigh(np.array(${JSON.stringify(symMat)}, dtype=${npDtype(dtype)}))
result = w.astype(np.float64)`;

    snippets[`linalg_eigvals_${dtype}`] = `
w = np.linalg.eigvals(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.sort(np.abs(w)).astype(np.float64)`;

    snippets[`linalg_eigvalsh_${dtype}`] = `
result = np.linalg.eigvalsh(np.array(${JSON.stringify(symMat)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    const A =
      dtype === 'bool'
        ? [
            [1, 0],
            [0, 1],
          ]
        : [
            [3, 1],
            [1, 2],
          ];
    const b = dtype === 'bool' ? [1, 0] : [9, 8];
    snippets[`linalg_solve_${dtype}`] =
      `result = np.linalg.solve(np.array(${JSON.stringify(A)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    const lstsqA =
      dtype === 'bool'
        ? [
            [1, 0],
            [0, 1],
            [1, 1],
          ]
        : [
            [1, 1],
            [1, 2],
            [1, 3],
          ];
    const lstsqB = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    snippets[`linalg_lstsq_${dtype}`] = `
x, res, rank, sv = np.linalg.lstsq(np.array(${JSON.stringify(lstsqA)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(lstsqB)}, dtype=${npDtype(dtype)}), rcond=None)
result = x.astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: linalg decompositions', () => {
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

    it(`linalg.norm ${dtype}`, () => {
      const jsResult = np.linalg.norm(array(vec, dtype));
      const py = oracle.get(`linalg_norm_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.det ${dtype}`, () => {
      const jsResult = np.linalg.det(array(mat, dtype));
      const py = oracle.get(`linalg_det_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`linalg.inv ${dtype}`, () => {
      const jsResult = np.linalg.inv(array(mat, dtype));
      const py = oracle.get(`linalg_inv_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.cholesky ${dtype}`, () => {
      const pdMat =
        dtype === 'bool'
          ? [
              [1, 0],
              [0, 1],
            ]
          : [
              [4, 2],
              [2, 5],
            ];
      const jsResult = np.linalg.cholesky(array(pdMat, dtype));
      const py = oracle.get(`linalg_cholesky_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.qr ${dtype}`, () => {
      const { q, r } = np.linalg.qr(array(mat, dtype)) as any;
      const reconstructed = np.matmul(q, r);
      const py = oracle.get(`linalg_qr_${dtype}`)!;
      expect(arraysClose(toNumbers(reconstructed), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.svd ${dtype}`, () => {
      const { s } = np.linalg.svd(array(mat, dtype)) as any;
      const py = oracle.get(`linalg_svd_${dtype}`)!;
      expect(arraysClose(toNumbers(s), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.eig ${dtype}`, { timeout: 30000 }, () => {
      const { w } = np.linalg.eig(array(mat, dtype)) as any;
      const py = oracle.get(`linalg_eig_${dtype}`)!;
      const jsAbs = np.sort(np.absolute(w)).toArray();
      expect(arraysClose(toNumbers({ toArray: () => jsAbs }), py.value, 1e-3)).toBe(true);
    });

    it(`linalg.eigh ${dtype}`, { timeout: 30000 }, () => {
      const symMat =
        dtype === 'bool'
          ? [
              [1, 0],
              [0, 1],
            ]
          : [
              [2, 1],
              [1, 3],
            ];
      const { w } = np.linalg.eigh(array(symMat, dtype)) as any;
      const py = oracle.get(`linalg_eigh_${dtype}`)!;
      expect(arraysClose(toNumbers(w), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.eigvals ${dtype}`, { timeout: 30000 }, () => {
      const jsResult = np.linalg.eigvals(array(mat, dtype));
      const py = oracle.get(`linalg_eigvals_${dtype}`)!;
      const jsAbs = np.sort(np.absolute(jsResult)).toArray();
      expect(arraysClose(toNumbers({ toArray: () => jsAbs }), py.value, 1e-3)).toBe(true);
    });

    it(`linalg.eigvalsh ${dtype}`, { timeout: 30000 }, () => {
      const symMat =
        dtype === 'bool'
          ? [
              [1, 0],
              [0, 1],
            ]
          : [
              [2, 1],
              [1, 3],
            ];
      const jsResult = np.linalg.eigvalsh(array(symMat, dtype));
      const py = oracle.get(`linalg_eigvalsh_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.solve ${dtype}`, () => {
      const A =
        dtype === 'bool'
          ? [
              [1, 0],
              [0, 1],
            ]
          : [
              [3, 1],
              [1, 2],
            ];
      const b = dtype === 'bool' ? [1, 0] : [9, 8];
      const jsResult = np.linalg.solve(array(A, dtype), array(b, dtype));
      const py = oracle.get(`linalg_solve_${dtype}`)!;
      expect(arraysClose(toNumbers(jsResult), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.lstsq ${dtype}`, () => {
      const A =
        dtype === 'bool'
          ? [
              [1, 0],
              [0, 1],
              [1, 1],
            ]
          : [
              [1, 1],
              [1, 2],
              [1, 3],
            ];
      const b = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const { x } = np.linalg.lstsq(array(A, dtype), array(b, dtype)) as any;
      const py = oracle.get(`linalg_lstsq_${dtype}`)!;
      expect(arraysClose(toNumbers(x), py.value, 1e-3)).toBe(true);
    });
  }
});

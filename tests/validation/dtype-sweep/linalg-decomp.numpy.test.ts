/**
 * DType Sweep: linalg decomposition & solve functions (np.linalg.*).
 * Heavy ops: norm, det, inv, cholesky, qr, svd, eig, eigh, eigvals, eigvalsh, solve, lstsq.
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

    snippets[`linalg_norm_${dtype}`] =
      `result = ${sc}(np.linalg.norm(np.array(${JSON.stringify(vec)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_det_${dtype}`] =
      `result = ${sc}(np.linalg.det(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`;

    snippets[`linalg_inv_${dtype}`] =
      `result = np.linalg.inv(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(${ac})`;

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
      `result = np.linalg.cholesky(np.array(${JSON.stringify(pdMat)}, dtype=${npDtype(dtype)})).astype(${ac})`;

    snippets[`linalg_qr_${dtype}`] =
      `result = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}).astype(${ac})`;

    snippets[`linalg_svd_${dtype}`] = `
u, s, vh = np.linalg.svd(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = s.astype(${ac})`;

    snippets[`linalg_eig_${dtype}`] = `
w, v = np.linalg.eig(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.sort(np.abs(w)).astype(${ac})`;

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
result = w.astype(${ac})`;

    snippets[`linalg_eigvals_${dtype}`] = `
w = np.linalg.eigvals(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.sort(np.abs(w)).astype(${ac})`;

    snippets[`linalg_eigvalsh_${dtype}`] = `
result = np.linalg.eigvalsh(np.array(${JSON.stringify(symMat)}, dtype=${npDtype(dtype)})).astype(${ac})`;

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
      `result = np.linalg.solve(np.array(${JSON.stringify(A)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})).astype(${ac})`;

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
result = x.astype(${ac})`;
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
      scalarClose(jsResult, py.value);
    });

    it(`linalg.det ${dtype}`, () => {
      const py = oracle.get(`linalg_det_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.det(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      scalarClose(np.linalg.det(array(mat, dtype)), py.value);
    });

    it(`linalg.inv ${dtype}`, () => {
      const py = oracle.get(`linalg_inv_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.inv(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      expect(arraysClose(toComparable(np.linalg.inv(array(mat, dtype))), py.value, 1e-4)).toBe(
        true
      );
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
      const py = oracle.get(`linalg_cholesky_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.cholesky(array(pdMat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      expect(
        arraysClose(toComparable(np.linalg.cholesky(array(pdMat, dtype))), py.value, 1e-4)
      ).toBe(true);
    });

    it(`linalg.qr ${dtype}`, () => {
      if (dtype === 'float16') {
        expect(() => np.linalg.qr(array(mat, dtype))).toThrow('float16 is unsupported in linalg');
        return;
      }
      const { q, r } = np.linalg.qr(array(mat, dtype)) as any;
      const py = oracle.get(`linalg_qr_${dtype}`)!;
      expect(arraysClose(toComparable(np.matmul(q, r)), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.svd ${dtype}`, () => {
      const py = oracle.get(`linalg_svd_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.svd(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      const { s } = np.linalg.svd(array(mat, dtype)) as any;
      expect(arraysClose(toComparable(s), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.eig ${dtype}`, { timeout: 30000 }, () => {
      const py = oracle.get(`linalg_eig_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.eig(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      const { w } = np.linalg.eig(array(mat, dtype)) as any;
      const jsAbs = np.sort(np.absolute(w)).toArray();
      expect(arraysClose(toComparable({ toArray: () => jsAbs }), py.value, 1e-3)).toBe(true);
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
      const py = oracle.get(`linalg_eigh_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.eigh(array(symMat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      const { w } = np.linalg.eigh(array(symMat, dtype)) as any;
      expect(arraysClose(toComparable(w), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.eigvals ${dtype}`, { timeout: 30000 }, () => {
      const py = oracle.get(`linalg_eigvals_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.eigvals(array(mat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      const jsResult = np.linalg.eigvals(array(mat, dtype));
      const jsAbs = np.sort(np.absolute(jsResult)).toArray();
      expect(arraysClose(toComparable({ toArray: () => jsAbs }), py.value, 1e-3)).toBe(true);
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
      const py = oracle.get(`linalg_eigvalsh_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.eigvalsh(array(symMat, dtype)),
        py
      );
      if (r === 'both-reject') return;
      expect(
        arraysClose(toComparable(np.linalg.eigvalsh(array(symMat, dtype))), py.value, 1e-4)
      ).toBe(true);
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
      const py = oracle.get(`linalg_solve_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.solve(array(A, dtype), array(b, dtype)),
        py
      );
      if (r === 'both-reject') return;
      expect(
        arraysClose(toComparable(np.linalg.solve(array(A, dtype), array(b, dtype))), py.value, 1e-4)
      ).toBe(true);
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
      const py = oracle.get(`linalg_lstsq_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.lstsq(array(A, dtype), array(b, dtype)),
        py
      );
      if (r === 'both-reject') return;
      const { x } = np.linalg.lstsq(array(A, dtype), array(b, dtype)) as any;
      expect(arraysClose(toComparable(x), py.value, 1e-3)).toBe(true);
    });
  }
});

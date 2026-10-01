/**
 * DType Sweep: linalg decomposition & solve functions (np.linalg.*).
 * Heavy ops: norm, det, inv, cholesky, qr, svd, eig, eigh, eigvals, eigvalsh, solve, lstsq.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  arraysClose,
  asDtypeData,
  checkNumPyAvailable,
  expectBothRejectPre,
  expectComplexFixture,
  expectMatchPre,
  npDtype,
  pyArrayCast,
  pyScalarCast,
  runNumPyBatch,
  scalarClose,
  toComparable,
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
  const id = [
    [1, 0],
    [0, 1],
  ];
  return {
    mat: asDtypeData(
      b
        ? id
        : [
            [1, 2],
            [3, 4],
          ],
      dtype,
    ),
    vec: asDtypeData(b ? [1, 0] : [3, 4], dtype),
    pdMat: asDtypeData(
      b
        ? id
        : [
            [4, 2],
            [2, 5],
          ],
      dtype,
    ),
    symMat: asDtypeData(
      b
        ? id
        : [
            [2, 1],
            [1, 3],
          ],
      dtype,
    ),
    solveA: asDtypeData(
      b
        ? id
        : [
            [3, 1],
            [1, 2],
          ],
      dtype,
    ),
    solveB: asDtypeData(b ? [1, 0] : [9, 8], dtype),
    lstsqA: asDtypeData(
      b
        ? [
            [1, 0],
            [0, 1],
            [1, 1],
          ]
        : [
            [1, 1],
            [1, 2],
            [1, 3],
          ],
      dtype,
    ),
    lstsqB: asDtypeData(b ? [1, 0, 1] : [1, 2, 3], dtype),
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

    snippets[`linalg_norm_${dtype}`] =
      `result = ${sc}(np.linalg.norm(np.array(${f.vec.py}, dtype=${dt})))`;

    snippets[`linalg_det_${dtype}`] =
      `result = ${sc}(np.linalg.det(np.array(${f.mat.py}, dtype=${dt})))`;

    snippets[`linalg_inv_${dtype}`] = `
_result_orig = np.linalg.inv(np.array(${f.mat.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_cholesky_${dtype}`] = `
_result_orig = np.linalg.cholesky(np.array(${f.pdMat.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    // QR is sign-ambiguous per column, so the test checks the q @ r
    // reconstruction against the input rather than q and r themselves.
    snippets[`linalg_qr_${dtype}`] = `result = np.array(${f.mat.py}, dtype=${dt}).astype(${ac})`;

    snippets[`linalg_svd_${dtype}`] = `
u, s, vh = np.linalg.svd(np.array(${f.mat.py}, dtype=${dt}))
_result_orig = s
result = s.astype(${ac})`;

    snippets[`linalg_eig_${dtype}`] = `
w, v = np.linalg.eig(np.array(${f.mat.py}, dtype=${dt}))
result = np.sort(np.abs(w)).astype(${ac})`;

    snippets[`linalg_eigh_${dtype}`] = `
w, v = np.linalg.eigh(np.array(${f.symMat.py}, dtype=${dt}))
_result_orig = w
result = w.astype(${ac})`;

    snippets[`linalg_eigvals_${dtype}`] = `
w = np.linalg.eigvals(np.array(${f.mat.py}, dtype=${dt}))
result = np.sort(np.abs(w)).astype(${ac})`;

    snippets[`linalg_eigvalsh_${dtype}`] = `
_result_orig = np.linalg.eigvalsh(np.array(${f.symMat.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_solve_${dtype}`] = `
_result_orig = np.linalg.solve(np.array(${f.solveA.py}, dtype=${dt}), np.array(${f.solveB.py}, dtype=${dt}))
result = _result_orig.astype(${ac})`;

    snippets[`linalg_lstsq_${dtype}`] = `
x, res, rank, sv = np.linalg.lstsq(np.array(${f.lstsqA.py}, dtype=${dt}), np.array(${f.lstsqB.py}, dtype=${dt}), rcond=None)
_result_orig = x
result = x.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: linalg decompositions', () => {
  for (const dtype of ALL_DTYPES) {
    const f = fixtures(dtype);

    it(`linalg.norm ${dtype}`, () => {
      const a = array(f.vec.js, dtype);
      expectComplexFixture(a, dtype, `norm ${dtype}`);
      scalarClose(np.linalg.norm(a), oracle.get(`linalg_norm_${dtype}`)!.value);
    });

    it(`linalg.det ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `det ${dtype}`);
      const py = oracle.get(`linalg_det_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.det(a), py);
      if (r === 'both-reject') return;
      scalarClose(np.linalg.det(a), py.value);
    });

    it(`linalg.inv ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `inv ${dtype}`);
      const py = oracle.get(`linalg_inv_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.inv(a), py);
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.inv(a), py, { rtol: 1e-4, atol: 1e-6 });
    });

    it(`linalg.cholesky ${dtype}`, () => {
      const a = array(f.pdMat.js, dtype);
      expectComplexFixture(a, dtype, `cholesky ${dtype}`);
      const py = oracle.get(`linalg_cholesky_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.cholesky(a),
        py,
      );
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.cholesky(a), py, { rtol: 1e-4, atol: 1e-6 });
    });

    it(`linalg.qr ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `qr ${dtype}`);
      if (dtype === 'float16') {
        expect(() => np.linalg.qr(a)).toThrow('float16 is unsupported in linalg');
        return;
      }
      // Column signs of q (and the matching rows of r) are arbitrary, so only
      // the reconstruction is well defined.
      const { q, r } = np.linalg.qr(a) as any;
      const py = oracle.get(`linalg_qr_${dtype}`)!;
      expect(arraysClose(toComparable(np.matmul(q, r)), py.value, 1e-4)).toBe(true);
    });

    it(`linalg.svd ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `svd ${dtype}`);
      const py = oracle.get(`linalg_svd_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.svd(a), py);
      if (r === 'both-reject') return;
      const { s } = np.linalg.svd(a) as any;
      expectMatchPre(s, py, { rtol: 1e-4 });
    });

    it(`linalg.eig ${dtype}`, { timeout: 30000 }, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `eig ${dtype}`);
      const py = oracle.get(`linalg_eig_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.eig(a), py);
      if (r === 'both-reject') return;
      // Eigenvector scale and eigenvalue order are arbitrary, so compare the
      // sorted magnitudes of the eigenvalues.
      const { w } = np.linalg.eig(a) as any;
      const jsAbs = np.sort(np.absolute(w)).toArray();
      expect(arraysClose(jsAbs, py.value, 1e-3)).toBe(true);
    });

    it(`linalg.eigh ${dtype}`, { timeout: 30000 }, () => {
      const a = array(f.symMat.js, dtype);
      expectComplexFixture(a, dtype, `eigh ${dtype}`);
      const py = oracle.get(`linalg_eigh_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.eigh(a), py);
      if (r === 'both-reject') return;
      const { w } = np.linalg.eigh(a) as any;
      expectMatchPre(w, py, { rtol: 1e-4 });
    });

    it(`linalg.eigvals ${dtype}`, { timeout: 30000 }, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `eigvals ${dtype}`);
      const py = oracle.get(`linalg_eigvals_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.eigvals(a),
        py,
      );
      if (r === 'both-reject') return;
      const jsAbs = np.sort(np.absolute(np.linalg.eigvals(a))).toArray();
      expect(arraysClose(jsAbs, py.value, 1e-3)).toBe(true);
    });

    it(`linalg.eigvalsh ${dtype}`, { timeout: 30000 }, () => {
      const a = array(f.symMat.js, dtype);
      expectComplexFixture(a, dtype, `eigvalsh ${dtype}`);
      const py = oracle.get(`linalg_eigvalsh_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.eigvalsh(a),
        py,
      );
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.eigvalsh(a), py, { rtol: 1e-4 });
    });

    it(`linalg.solve ${dtype}`, () => {
      const a = array(f.solveA.js, dtype);
      const b = array(f.solveB.js, dtype);
      expectComplexFixture(a, dtype, `solve ${dtype}`);
      const py = oracle.get(`linalg_solve_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.solve(a, b),
        py,
      );
      if (r === 'both-reject') return;
      expectMatchPre(np.linalg.solve(a, b), py, { rtol: 1e-4, atol: 1e-6 });
    });

    it(`linalg.lstsq ${dtype}`, () => {
      const a = array(f.lstsqA.js, dtype);
      const b = array(f.lstsqB.js, dtype);
      expectComplexFixture(a, dtype, `lstsq ${dtype}`);
      const py = oracle.get(`linalg_lstsq_${dtype}`)!;
      const r = expectBothRejectPre(
        'float16 unsupported in linalg',
        () => np.linalg.lstsq(a, b),
        py,
      );
      if (r === 'both-reject') return;
      const { x } = np.linalg.lstsq(a, b) as any;
      expectMatchPre(x, py, { rtol: 1e-3, atol: 1e-3 });
    });
  }
});

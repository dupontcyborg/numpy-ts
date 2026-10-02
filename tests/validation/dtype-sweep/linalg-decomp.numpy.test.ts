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
  asHermitianData,
  checkNumPyAvailable,
  expectBothRejectPre,
  expectComplexFixture,
  expectMatchPre,
  isComplex,
  npDtype,
  pyArrayCast,
  pyScalarCast,
  runNumPyBatch,
  scalarClose,
} from './_helpers';
import {
  conjT,
  expectDType,
  expectEigenpairs,
  expectReconstructs,
  expectUnitaryColumns,
  expectUpperTriangular,
  linalgDType,
  linalgRealDType,
  mat,
  matmulC,
  tolFor,
  vec,
} from './_invariants';

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
    pdMat: asHermitianData(
      b
        ? id
        : [
            [4, 2],
            [2, 5],
          ],
      dtype,
    ),
    symMat: asHermitianData(
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
      const l = np.linalg.cholesky(a) as any;
      expectMatchPre(l, py, { rtol: 1e-4, atol: 1e-6 });
      const tol = tolFor(dtype);
      expectDType(l, linalgDType(dtype), `cholesky ${dtype}`);
      const lm = mat(l.toArray());
      expectUpperTriangular(conjT(lm), `cholesky ${dtype} L transposed`, tol);
      expectReconstructs(matmulC(lm, conjT(lm)), mat(a.toArray()), `cholesky ${dtype}`, tol);
    });

    it(`linalg.qr ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `qr ${dtype}`);
      if (dtype === 'float16') {
        expect(() => np.linalg.qr(a)).toThrow('float16 is unsupported in linalg');
        return;
      }
      // Column signs of q (and the matching rows of r) are arbitrary, so the
      // factors are pinned by their algebra rather than compared elementwise.
      const { q, r } = np.linalg.qr(a) as any;
      const tol = tolFor(dtype);
      const qm = mat(q.toArray());
      const rm = mat(r.toArray());
      expectDType(q, linalgDType(dtype), `qr ${dtype} q`);
      expectDType(r, linalgDType(dtype), `qr ${dtype} r`);
      expectReconstructs(matmulC(qm, rm), mat(a.toArray()), `qr ${dtype}`, tol);
      expectUnitaryColumns(qm, `qr ${dtype} q`, tol);
      expectUpperTriangular(rm, `qr ${dtype} r`, tol);
    });

    it(`linalg.svd ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `svd ${dtype}`);
      const py = oracle.get(`linalg_svd_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.svd(a), py);
      if (r === 'both-reject') return;
      const { u, s, vt } = np.linalg.svd(a) as any;
      expectMatchPre(s, py, { rtol: 1e-4 });
      const tol = tolFor(dtype);
      expectDType(u, linalgDType(dtype), `svd ${dtype} u`);
      expectDType(vt, linalgDType(dtype), `svd ${dtype} vt`);
      expectDType(s, linalgRealDType(dtype), `svd ${dtype} s`);
      const um = mat(u.toArray());
      const vtm = mat(vt.toArray());
      const sv = s.toArray().map(Number);
      // U diag(s) V^H, taking the first k columns of U and rows of V^H.
      const scaled = um.map((row) =>
        row.slice(0, sv.length).map((c, j) => ({
          re: c.re * sv[j]!,
          im: c.im * sv[j]!,
        })),
      );
      expectReconstructs(
        matmulC(scaled, vtm.slice(0, sv.length)),
        mat(a.toArray()),
        `svd ${dtype}`,
        tol,
      );
      expectUnitaryColumns(um, `svd ${dtype} u`, tol);
      expectUnitaryColumns(vtm, `svd ${dtype} vt`, tol);
    });

    it(`linalg.eig ${dtype}`, { timeout: 30000 }, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `eig ${dtype}`);
      const py = oracle.get(`linalg_eig_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.eig(a), py);
      if (r === 'both-reject') return;
      // Eigenvector scale and eigenvalue order are arbitrary, so the values are
      // compared as sorted magnitudes and the vectors are pinned by A v = lambda v.
      // Issue #161 was exactly a case the magnitude comparison alone let through.
      const { w, v } = np.linalg.eig(a) as any;
      const jsAbs = np.sort(np.absolute(w)).toArray();
      expect(arraysClose(jsAbs, py.value, 1e-3)).toBe(true);
      const eigDType = isComplex(dtype) ? linalgDType(dtype) : 'float64';
      expectDType(w, eigDType, `eig ${dtype} w`);
      expectDType(v, eigDType, `eig ${dtype} v`);
      expectEigenpairs(
        mat(a.toArray()),
        vec(w.toArray()),
        mat(v.toArray()),
        `eig ${dtype}`,
        tolFor(dtype),
      );
    });

    it(`linalg.eigh ${dtype}`, { timeout: 30000 }, () => {
      const a = array(f.symMat.js, dtype);
      expectComplexFixture(a, dtype, `eigh ${dtype}`);
      const py = oracle.get(`linalg_eigh_${dtype}`)!;
      const r = expectBothRejectPre('float16 unsupported in linalg', () => np.linalg.eigh(a), py);
      if (r === 'both-reject') return;
      const { w, v } = np.linalg.eigh(a) as any;
      expectMatchPre(w, py, { rtol: 1e-4 });
      const tol = tolFor(dtype);
      // A Hermitian matrix has real eigenvalues, so w narrows even for complex input.
      expectDType(w, linalgRealDType(dtype), `eigh ${dtype} w`);
      expectDType(v, linalgDType(dtype), `eigh ${dtype} v`);
      const vm = mat(v.toArray());
      expectUnitaryColumns(vm, `eigh ${dtype} v`, tol);
      expectEigenpairs(mat(a.toArray()), vec(w.toArray()), vm, `eigh ${dtype}`, tol);
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
      const w = np.linalg.eigvals(a);
      const jsAbs = np.sort(np.absolute(w)).toArray();
      expect(arraysClose(jsAbs, py.value, 1e-3)).toBe(true);
      expectDType(w, isComplex(dtype) ? linalgDType(dtype) : 'float64', `eigvals ${dtype}`);
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
      const w = np.linalg.eigvalsh(a);
      expectMatchPre(w, py, { rtol: 1e-4 });
      expectDType(w, linalgRealDType(dtype), `eigvalsh ${dtype}`);
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

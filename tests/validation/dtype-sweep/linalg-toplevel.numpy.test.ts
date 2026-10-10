/**
 * DType Sweep: Top-level linear algebra functions (np.dot, np.cross, etc.).
 * Tests across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectBothReject,
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
    dotA: asDtypeData(b ? [1, 0, 1] : [1, 2, 3], dtype),
    dotB: asDtypeData(b ? [0, 1, 0] : [4, 5, 6], dtype),
    outerA: asDtypeData(b ? [1, 0] : [1, 2, 3], dtype),
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
    kronA: asDtypeData(b ? [1, 0] : [1, 2], dtype),
    kronB: asDtypeData(b ? [0, 1] : [3, 4], dtype),
    vec: asDtypeData(b ? [1, 0] : [5, 6], dtype),
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

    snippets[`dot_${dtype}`] = `
a = np.array(${f.dotA.py}, dtype=${dt})
b = np.array(${f.dotB.py}, dtype=${dt})
result = ${sc}(np.dot(a, b))`;

    snippets[`inner_${dtype}`] = `
a = np.array(${f.dotA.py}, dtype=${dt})
b = np.array(${f.dotB.py}, dtype=${dt})
result = ${sc}(np.inner(a, b))`;

    snippets[`outer_${dtype}`] = `
a = np.array(${f.outerA.py}, dtype=${dt})
_result_orig = np.outer(a, a)
result = _result_orig.astype(${ac})`;

    snippets[`matmul_${dtype}`] = `
a = np.array(${f.mat.py}, dtype=${dt})
_result_orig = np.matmul(a, a)
result = _result_orig.astype(${ac})`;

    snippets[`cross_${dtype}`] = `
a = np.array(${f.dotA.py}, dtype=${dt})
b = np.array(${f.dotB.py}, dtype=${dt})
_result_orig = np.cross(a, b)
result = _result_orig.astype(${ac})`;

    snippets[`kron_${dtype}`] = `
a = np.array(${f.kronA.py}, dtype=${dt})
b = np.array(${f.kronB.py}, dtype=${dt})
_result_orig = np.kron(a, b)
result = _result_orig.astype(${ac})`;

    snippets[`trace_${dtype}`] = `result = ${sc}(np.trace(np.array(${f.mat.py}, dtype=${dt})))`;

    snippets[`tensordot_${dtype}`] = `
a = np.array(${f.mat.py}, dtype=${dt})
result = ${sc}(np.tensordot(a, a))`;

    snippets[`vecdot_${dtype}`] = `
a = np.array(${f.dotA.py}, dtype=${dt})
b = np.array(${f.dotB.py}, dtype=${dt})
result = ${sc}(np.vecdot(a, b))`;

    snippets[`matvec_${dtype}`] = `
m = np.array(${f.mat.py}, dtype=${dt})
v = np.array(${f.vec.py}, dtype=${dt})
_result_orig = np.matvec(m, v)
result = _result_orig.astype(${ac})`;

    snippets[`vecmat_${dtype}`] = `
m = np.array(${f.mat.py}, dtype=${dt})
v = np.array(${f.vec.py}, dtype=${dt})
_result_orig = np.vecmat(v, m)
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Top-level linalg', () => {
  for (const dtype of ALL_DTYPES) {
    const f = fixtures(dtype);

    it(`dot ${dtype}`, () => {
      const a = array(f.dotA.js, dtype);
      const b = array(f.dotB.js, dtype);
      expectComplexFixture(a, dtype, `dot ${dtype}`);
      const jsResult = np.dot(a, b);
      const py = oracle.get(`dot_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`inner ${dtype}`, () => {
      const a = array(f.dotA.js, dtype);
      const b = array(f.dotB.js, dtype);
      expectComplexFixture(a, dtype, `inner ${dtype}`);
      const jsResult = np.inner(a, b);
      const py = oracle.get(`inner_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`outer ${dtype}`, () => {
      const a = array(f.outerA.js, dtype);
      expectComplexFixture(a, dtype, `outer ${dtype}`);
      expectMatchPre(np.outer(a, a), oracle.get(`outer_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`matmul ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `matmul ${dtype}`);
      expectMatchPre(np.matmul(a, a), oracle.get(`matmul_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`cross ${dtype}`, () => {
      const a = array(f.dotA.js, dtype);
      const b = array(f.dotB.js, dtype);
      expectComplexFixture(a, dtype, `cross ${dtype}`);
      // cross(bool): NumPy rejects — "boolean subtract not supported"
      if (dtype === 'bool') {
        const pyCode = `
a = np.array(${f.dotA.py}, dtype=${npDtype(dtype)})
b = np.array(${f.dotB.py}, dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(np.float64)`;
        const _r = expectBothReject(
          'cross uses subtract internally, not supported for bool',
          () => np.cross(a, b),
          pyCode,
        );
        if (_r === 'both-reject') return;
      }
      expectMatchPre(np.cross(a, b), oracle.get(`cross_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`kron ${dtype}`, () => {
      const a = array(f.kronA.js, dtype);
      const b = array(f.kronB.js, dtype);
      expectComplexFixture(a, dtype, `kron ${dtype}`);
      expectMatchPre(np.kron(a, b), oracle.get(`kron_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`trace ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `trace ${dtype}`);
      const jsResult = np.trace(a);
      const py = oracle.get(`trace_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`tensordot ${dtype}`, () => {
      const a = array(f.mat.js, dtype);
      expectComplexFixture(a, dtype, `tensordot ${dtype}`);
      const r = np.tensordot(a, a);
      const py = oracle.get(`tensordot_${dtype}`)!;
      scalarClose(r, py.value);
    });

    it(`vecdot ${dtype}`, () => {
      const a = array(f.dotA.js, dtype);
      const b = array(f.dotB.js, dtype);
      expectComplexFixture(a, dtype, `vecdot ${dtype}`);
      const jsResult = np.vecdot(a, b);
      const py = oracle.get(`vecdot_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`matvec ${dtype}`, () => {
      const m = array(f.mat.js, dtype);
      const v = array(f.vec.js, dtype);
      expectComplexFixture(m, dtype, `matvec ${dtype}`);
      expectMatchPre(np.matvec(m, v), oracle.get(`matvec_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`vecmat ${dtype}`, () => {
      const m = array(f.mat.js, dtype);
      const v = array(f.vec.js, dtype);
      expectComplexFixture(m, dtype, `vecmat ${dtype}`);
      expectMatchPre(np.vecmat(v, m), oracle.get(`vecmat_${dtype}`)!, { rtol: 1e-4 });
    });
  }
});

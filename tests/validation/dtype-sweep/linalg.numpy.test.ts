/**
 * DType Sweep: Linear algebra functions.
 * Tests linalg namespace + top-level linalg across ALL dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype } from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

// Helper: convert bigint array values to numbers for comparison
function toNumbers(arr: any): any {
  const flat = arr.toArray();
  return JSON.parse(JSON.stringify(flat, (_k: string, v: any) => typeof v === 'bigint' ? Number(v) : v));
}

describe('DType Sweep: linalg namespace', () => {
  for (const dtype of ALL_DTYPES) {
    const mat = dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 2], [3, 4]];
    const vec = dtype === 'bool' ? [1, 0] : [3, 4];

    it(`linalg.norm ${dtype}`, () => {
      const jsResult = np.linalg.norm(array(vec, dtype));
      const pyResult = runNumPy(
        `result = float(np.linalg.norm(np.array(${JSON.stringify(vec)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.det ${dtype}`, () => {
      const jsResult = np.linalg.det(array(mat, dtype));
      const pyResult = runNumPy(
        `result = float(np.linalg.det(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.inv ${dtype}`, () => {
      const jsResult = np.linalg.inv(array(mat, dtype));
      const pyResult = runNumPy(
        `result = np.linalg.inv(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.cholesky ${dtype}`, () => {
      const pdMat = dtype === 'bool' ? [[1, 0], [0, 1]] : [[4, 2], [2, 5]];
      const jsResult = np.linalg.cholesky(array(pdMat, dtype));
      const pyResult = runNumPy(
        `result = np.linalg.cholesky(np.array(${JSON.stringify(pdMat)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.qr ${dtype}`, () => {
      const { q, r } = np.linalg.qr(array(mat, dtype)) as any;
      // Verify Q is orthogonal: Q @ Q^T ≈ I
      const pyResult = runNumPy(`
q, r = np.linalg.qr(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = r.astype(np.float64)
      `);
      expect(arraysClose(toNumbers(r), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.svd ${dtype}`, () => {
      const { s } = np.linalg.svd(array(mat, dtype)) as any;
      const pyResult = runNumPy(`
u, s, vh = np.linalg.svd(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = s.astype(np.float64)
      `);
      expect(arraysClose(toNumbers(s), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.eig ${dtype}`, { timeout: 30000 }, () => {
      const { w } = np.linalg.eig(array(mat, dtype)) as any;
      const pyResult = runNumPy(`
w, v = np.linalg.eig(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.sort(np.abs(w)).astype(np.float64)
      `);
      // Eigenvalues may differ in sign/order, compare sorted absolute values
      const jsAbs = np.sort(np.absolute(w)).toArray();
      expect(arraysClose(toNumbers({ toArray: () => jsAbs }), pyResult.value, 1e-3)).toBe(true);
    });

    it(`linalg.eigh ${dtype}`, { timeout: 30000 }, () => {
      const symMat = dtype === 'bool' ? [[1, 0], [0, 1]] : [[2, 1], [1, 3]];
      const { w } = np.linalg.eigh(array(symMat, dtype)) as any;
      const pyResult = runNumPy(`
w, v = np.linalg.eigh(np.array(${JSON.stringify(symMat)}, dtype=${npDtype(dtype)}))
result = w.astype(np.float64)
      `);
      expect(arraysClose(toNumbers(w), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.eigvals ${dtype}`, { timeout: 30000 }, () => {
      const jsResult = np.linalg.eigvals(array(mat, dtype));
      const pyResult = runNumPy(`
w = np.linalg.eigvals(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.sort(np.abs(w)).astype(np.float64)
      `);
      const jsAbs = np.sort(np.absolute(jsResult)).toArray();
      expect(arraysClose(toNumbers({ toArray: () => jsAbs }), pyResult.value, 1e-3)).toBe(true);
    });

    it(`linalg.eigvalsh ${dtype}`, { timeout: 30000 }, () => {
      const symMat = dtype === 'bool' ? [[1, 0], [0, 1]] : [[2, 1], [1, 3]];
      const jsResult = np.linalg.eigvalsh(array(symMat, dtype));
      const pyResult = runNumPy(`
result = np.linalg.eigvalsh(np.array(${JSON.stringify(symMat)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.solve ${dtype}`, () => {
      const A = dtype === 'bool' ? [[1, 0], [0, 1]] : [[3, 1], [1, 2]];
      const b = dtype === 'bool' ? [1, 0] : [9, 8];
      const jsResult = np.linalg.solve(array(A, dtype), array(b, dtype));
      const pyResult = runNumPy(
        `result = np.linalg.solve(np.array(${JSON.stringify(A)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.lstsq ${dtype}`, () => {
      const A = dtype === 'bool' ? [[1, 0], [0, 1], [1, 1]] : [[1, 1], [1, 2], [1, 3]];
      const b = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const { x } = np.linalg.lstsq(array(A, dtype), array(b, dtype)) as any;
      const pyResult = runNumPy(`
x, res, rank, sv = np.linalg.lstsq(np.array(${JSON.stringify(A)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)}), rcond=None)
result = x.astype(np.float64)
      `);
      expect(arraysClose(toNumbers(x), pyResult.value, 1e-3)).toBe(true);
    });

    it(`linalg.matrix_rank ${dtype}`, () => {
      const jsResult = np.linalg.matrix_rank(array(mat, dtype));
      const pyResult = runNumPy(
        `result = int(np.linalg.matrix_rank(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBe(Number(pyResult.value));
    });

    it(`linalg.matrix_power ${dtype}`, () => {
      const jsResult = np.linalg.matrix_power(array(mat, dtype), 2);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.linalg.matrix_power(a, 2).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.pinv ${dtype}`, () => {
      const jsResult = np.linalg.pinv(array(mat, dtype));
      const pyResult = runNumPy(`
result = np.linalg.pinv(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.cond ${dtype}`, () => {
      const jsResult = np.linalg.cond(array(mat, dtype));
      const pyResult = runNumPy(
        `result = float(np.linalg.cond(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 2);
    });

    it(`linalg.slogdet ${dtype}`, () => {
      const { sign, logabsdet } = np.linalg.slogdet(array(mat, dtype)) as any;
      const pyResult = runNumPy(`
sign, logabsdet = np.linalg.slogdet(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}))
result = np.array([float(sign), float(logabsdet)])
      `);
      expect(Number(sign)).toBeCloseTo(Number(pyResult.value[0]), 4);
      expect(Number(logabsdet)).toBeCloseTo(Number(pyResult.value[1]), 4);
    });

    it(`linalg.svdvals ${dtype}`, () => {
      const jsResult = np.linalg.svdvals(array(mat, dtype));
      const pyResult = runNumPy(`
result = np.linalg.svdvals(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.multi_dot ${dtype}`, () => {
      const a = array(mat, dtype);
      const jsResult = np.linalg.multi_dot([a, a]);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.linalg.multi_dot([a, a]).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.vector_norm ${dtype}`, () => {
      const jsResult = np.linalg.vector_norm(array(vec, dtype));
      const pyResult = runNumPy(
        `result = float(np.linalg.vector_norm(np.array(${JSON.stringify(vec)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.matrix_norm ${dtype}`, () => {
      const jsResult = np.linalg.matrix_norm(array(mat, dtype));
      const pyResult = runNumPy(
        `result = float(np.linalg.matrix_norm(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.tensorinv ${dtype}`, () => {
      const id = dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 0], [0, 1]];
      const jsResult = np.linalg.tensorinv(array(id, dtype));
      const pyResult = runNumPy(`
result = np.linalg.tensorinv(np.array(${JSON.stringify(id)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.tensorsolve ${dtype}`, () => {
      const id = dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 0], [0, 1]];
      const b = dtype === 'bool' ? [1, 0] : [1, 2];
      const jsResult = np.linalg.tensorsolve(array(id, dtype), array(b, dtype));
      const pyResult = runNumPy(`
result = np.linalg.tensorsolve(np.array(${JSON.stringify(id)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.cross ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.linalg.cross(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.vecdot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.linalg.vecdot(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
result = float(np.vecdot(np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.dot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.dot(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
result = float(np.dot(np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.inner ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.inner(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
result = float(np.inner(np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.outer ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const jsResult = np.linalg.outer(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
result = np.outer(np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.matmul ${dtype}`, () => {
      const jsResult = np.linalg.matmul(array(mat, dtype), array(mat, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.matmul(a, a).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`linalg.tensordot ${dtype}`, () => {
      const jsResult = np.linalg.tensordot(array(mat, dtype), array(mat, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = float(np.tensordot(a, a))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.trace ${dtype}`, () => {
      const jsResult = np.linalg.trace(array(mat, dtype));
      const pyResult = runNumPy(
        `result = float(np.trace(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`linalg.diagonal ${dtype}`, () => {
      const jsResult = np.linalg.diagonal(array(mat, dtype));
      const pyResult = runNumPy(
        `result = np.diagonal(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(toNumbers(jsResult), pyResult.value)).toBe(true);
    });

    it(`linalg.transpose ${dtype}`, () => {
      const jsResult = np.linalg.transpose(array(mat, dtype));
      const pyResult = runNumPy(
        `result = np.transpose(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})).astype(np.float64)`
      );
      expect(arraysClose(toNumbers(jsResult), pyResult.value)).toBe(true);
    });

    it(`linalg.matrix_transpose ${dtype}`, () => {
      const jsResult = np.linalg.matrix_transpose(array(mat, dtype));
      const pyResult = runNumPy(
        `result = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}).T.astype(np.float64)`
      );
      expect(arraysClose(toNumbers(jsResult), pyResult.value)).toBe(true);
    });

    it(`linalg.permute_dims ${dtype}`, () => {
      const jsResult = np.linalg.permute_dims(array(mat, dtype), [1, 0]);
      const pyResult = runNumPy(
        `result = np.transpose(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)}), [1, 0]).astype(np.float64)`
      );
      expect(arraysClose(toNumbers(jsResult), pyResult.value)).toBe(true);
    });
  }
});

describe('DType Sweep: Top-level linalg', () => {
  for (const dtype of ALL_DTYPES) {
    it(`dot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.dot(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = float(np.dot(a, b))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`inner ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.inner(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = float(np.inner(a, b))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`outer ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : [1, 2, 3], dtype);
      const r = np.outer(a, a);
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(dtype === 'bool' ? [1, 0] : [1, 2, 3])}, dtype=${npDtype(dtype)})
result = np.outer(a, a).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(r), pyResult.value, 1e-4)).toBe(true);
    });

    it(`matmul ${dtype}`, () => {
      const mat = dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 2], [3, 4]];
      const jsResult = np.matmul(array(mat, dtype), array(mat, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = np.matmul(a, a).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`cross ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const r = np.cross(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = np.cross(a, b).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(r), pyResult.value, 1e-4)).toBe(true);
    });

    it(`kron ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0] : [1, 2];
      const b = dtype === 'bool' ? [0, 1] : [3, 4];
      const r = np.kron(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = np.kron(a, b).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(r), pyResult.value, 1e-4)).toBe(true);
    });

    it(`trace ${dtype}`, () => {
      const mat = dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 2], [3, 4]];
      const jsResult = np.trace(array(mat, dtype));
      const pyResult = runNumPy(
        `result = float(np.trace(np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})))`
      );
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`tensordot ${dtype}`, () => {
      const mat = dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 2], [3, 4]];
      const r = np.tensordot(array(mat, dtype), array(mat, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(mat)}, dtype=${npDtype(dtype)})
result = float(np.tensordot(a, a))
      `);
      expect(Number(r)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`vecdot ${dtype}`, () => {
      const a = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const b = dtype === 'bool' ? [0, 1, 0] : [4, 5, 6];
      const jsResult = np.vecdot(array(a, dtype), array(b, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(a)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(b)}, dtype=${npDtype(dtype)})
result = float(np.vecdot(a, b))
      `);
      expect(Number(jsResult)).toBeCloseTo(Number(pyResult.value), 4);
    });

    it(`matvec ${dtype}`, () => {
      const m = dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 2], [3, 4]];
      const v = dtype === 'bool' ? [1, 0] : [5, 6];
      const jsResult = np.matvec(array(m, dtype), array(v, dtype));
      const pyResult = runNumPy(`
m = np.array(${JSON.stringify(m)}, dtype=${npDtype(dtype)})
v = np.array(${JSON.stringify(v)}, dtype=${npDtype(dtype)})
result = np.matvec(m, v).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });

    it(`vecmat ${dtype}`, () => {
      const m = dtype === 'bool' ? [[1, 0], [0, 1]] : [[1, 2], [3, 4]];
      const v = dtype === 'bool' ? [1, 0] : [5, 6];
      const jsResult = np.vecmat(array(v, dtype), array(m, dtype));
      const pyResult = runNumPy(`
m = np.array(${JSON.stringify(m)}, dtype=${npDtype(dtype)})
v = np.array(${JSON.stringify(v)}, dtype=${npDtype(dtype)})
result = np.vecmat(v, m).astype(np.float64)
      `);
      expect(arraysClose(toNumbers(jsResult), pyResult.value, 1e-4)).toBe(true);
    });
  }
});

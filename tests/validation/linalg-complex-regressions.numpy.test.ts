/**
 * Regressions for complex and mixed-dtype np.linalg paths: reduced and
 * rank-deficient complex SVD, batched complex decompositions, mixed real and
 * complex solve and lstsq, eig dtype narrowing and non-finite rejection, and
 * slogdet of an empty complex matrix. Each case is checked against NumPy for
 * dtype and, where the result is unique, for values; otherwise through the
 * algebra the factors must satisfy.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../src';
import { Complex } from '../../src/common/complex';
import { arraysClose, BACKEND, checkNumPyAvailable, runNumPy } from './dtype-sweep/_helpers';
import {
  type C,
  conjT,
  expectDType,
  expectReconstructs,
  expectUnitaryColumns,
  expectUpperTriangular,
  mat,
  matmulC,
  tolFor,
} from './dtype-sweep/_invariants';

const { array } = np;

type Num = number | Complex;

/** Deterministic pseudo-random values in [-0.5, 0.5). */
function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 16807) % 2147483647;
    return s / 2147483647 - 0.5;
  };
}

function complexMatrix(m: number, n: number, seed: number): Complex[][] {
  const r = lcg(seed);
  return Array.from({ length: m }, () => Array.from({ length: n }, () => new Complex(r(), r())));
}

function realMatrix(m: number, n: number, seed: number): number[][] {
  const r = lcg(seed);
  return Array.from({ length: m }, () => Array.from({ length: n }, () => r()));
}

/** A Hermitian positive-definite matrix, B^H B + n I. */
function hpd(n: number, seed: number): Complex[][] {
  const b = mat(complexMatrix(n, n, seed));
  const g = matmulC(conjT(b), b);
  return g.map((row, i) => row.map((c, j) => new Complex(c.re + (i === j ? n : 0), c.im)));
}

function pyNum(v: Num): string {
  return v instanceof Complex ? `complex(${v.re}, ${v.im})` : `${v}`;
}

/** Python literal for a nested array of numbers or Complex values. */
function pyLit(x: unknown): string {
  if (Array.isArray(x)) return `[${x.map(pyLit).join(', ')}]`;
  return pyNum(x as Num);
}

function pyArr(x: unknown, dtype: string): string {
  return `np.array(${pyLit(x)}, dtype=np.${dtype})`;
}

function diagC(s: number[]): C[][] {
  return s.map((v, i) => s.map((_, j) => ({ re: i === j ? v : 0, im: 0 })));
}

/** U diag(s) Vh for possibly rectangular factors from a reduced or full SVD. */
function svdProduct(u: C[][], s: number[], vt: C[][]): C[][] {
  const k = s.length;
  const uk = u.map((row) => row.slice(0, k));
  const vk = vt.slice(0, k);
  return matmulC(matmulC(uk, diagC(s)), vk);
}

function sliceAt(x: any, b: number): any {
  return x.toArray()[b];
}

describe(`linalg complex regressions [${BACKEND}]`, () => {
  beforeAll(() => {
    if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
  });

  describe('svd full_matrices=false', () => {
    for (const [dtype, m, n] of [
      ['complex128', 4, 3],
      ['complex64', 3, 5],
      ['float32', 5, 3],
    ] as const) {
      it(`${dtype} ${m}x${n}`, () => {
        const data = dtype === 'float32' ? realMatrix(m, n, 7) : complexMatrix(m, n, 7);
        const a = array(data as any, dtype);
        const { u, s, vt } = np.linalg.svd(a, false) as any;
        const pyU = runNumPy(
          `result = np.linalg.svd(${pyArr(data, dtype)}, full_matrices=False)[0]`,
        );
        const pyS = runNumPy(
          `result = np.linalg.svd(${pyArr(data, dtype)}, full_matrices=False)[1]`,
        );
        expectDType(u, pyU.dtype, 'svd reduced u');
        expectDType(vt, pyU.dtype, 'svd reduced vt');
        expectDType(s, pyS.dtype, 'svd reduced s');
        expect(u.shape).toEqual(pyU.shape);
        expect(arraysClose(s.toArray(), pyS.value, 1e-4)).toBe(true);
        const tol = tolFor(dtype);
        const um = mat(u.toArray());
        const vm = mat(vt.toArray());
        expectUnitaryColumns(um, 'svd reduced u', tol);
        expectUnitaryColumns(conjT(vm), 'svd reduced vt', tol);
        expectReconstructs(svdProduct(um, s.toArray(), vm), mat(data as any), 'svd reduced', tol);
      });
    }
  });

  describe('svd unitarity edge cases', () => {
    // Below the WASM size threshold, so only FORCE_BACKEND=wasm reaches the kernel.
    it('rank-deficient complex input keeps U unitary', () => {
      const x = complexMatrix(3, 1, 11);
      const y = complexMatrix(1, 3, 12);
      const data = mat(x).map((xi) =>
        mat(y)[0]!.map(
          (yj) =>
            new Complex(
              xi[0]!.re * yj.re - xi[0]!.im * yj.im,
              xi[0]!.re * yj.im + xi[0]!.im * yj.re,
            ),
        ),
      );
      const a = array(data as any, 'complex128');
      const { u, s, vt } = np.linalg.svd(a) as any;
      const pyS = runNumPy(
        `result = np.linalg.svd(${pyArr(data, 'complex128')}, compute_uv=False)`,
      );
      expect(arraysClose(s.toArray(), pyS.value, 1e-9)).toBe(true);
      const um = mat(u.toArray());
      const vm = mat(vt.toArray());
      expectUnitaryColumns(um, 'rank-deficient u', 1e-9);
      expectUnitaryColumns(conjT(vm), 'rank-deficient vt', 1e-9);
      expectReconstructs(svdProduct(um, s.toArray(), vm), mat(data), 'rank-deficient svd', 1e-9);
    });

    it('wide 6x20 complex input keeps Vh unitary', () => {
      const data = complexMatrix(6, 20, 3);
      const a = array(data as any, 'complex128');
      const { u, s, vt } = np.linalg.svd(a) as any;
      const pyS = runNumPy(
        `result = np.linalg.svd(${pyArr(data, 'complex128')}, compute_uv=False)`,
      );
      expect(arraysClose(s.toArray(), pyS.value, 1e-9)).toBe(true);
      const um = mat(u.toArray());
      const vm = mat(vt.toArray());
      expectUnitaryColumns(um, 'wide u', 1e-9);
      expectUnitaryColumns(conjT(vm), 'wide vt', 1e-9);
      expectReconstructs(svdProduct(um, s.toArray(), vm), mat(data), 'wide svd', 1e-9);
    });
  });

  describe('batched input', () => {
    for (const dtype of ['complex128', 'complex64', 'float32'] as const) {
      const tol = tolFor(dtype);
      const gen = (m: number, n: number, seed: number) =>
        dtype === 'float32' ? realMatrix(m, n, seed) : complexMatrix(m, n, seed);
      const stack = [gen(4, 3, 21), gen(4, 3, 22)];

      it(`svd ${dtype}`, () => {
        const a = array(stack as any, dtype);
        for (const full of [true, false]) {
          const { u, s, vt } = np.linalg.svd(a, full) as any;
          const py = runNumPy(
            `_r = np.linalg.svd(${pyArr(stack, dtype)}, full_matrices=${full ? 'True' : 'False'})\nresult = _r[1]\n_result_orig = _r[0]`,
          );
          expectDType(u, py.orig_dtype!, `batched svd ${dtype} u`);
          expectDType(vt, py.orig_dtype!, `batched svd ${dtype} vt`);
          expectDType(s, py.dtype, `batched svd ${dtype} s`);
          expect(u.shape).toEqual(py.orig_shape);
          expect(arraysClose(s.toArray(), py.value, 1e-4)).toBe(true);
          for (let b = 0; b < stack.length; b++) {
            const um = mat(sliceAt(u, b));
            const vm = mat(sliceAt(vt, b));
            expectUnitaryColumns(um, `batched svd ${dtype} u[${b}]`, tol);
            expectUnitaryColumns(conjT(vm), `batched svd ${dtype} vt[${b}]`, tol);
            expectReconstructs(
              svdProduct(um, sliceAt(s, b), vm),
              mat(stack[b] as any),
              `batched svd ${dtype}[${b}]`,
              tol,
            );
          }
        }
        const sOnly = np.linalg.svd(a, true, false) as any;
        const pyS = runNumPy(`result = np.linalg.svd(${pyArr(stack, dtype)}, compute_uv=False)`);
        expectDType(sOnly, pyS.dtype, `batched svdvals ${dtype}`);
        expect(arraysClose(sOnly.toArray(), pyS.value, 1e-4)).toBe(true);
      });

      it(`qr ${dtype}`, () => {
        const a = array(stack as any, dtype);
        const { q, r } = np.linalg.qr(a) as any;
        const py = runNumPy(
          `_r = np.linalg.qr(${pyArr(stack, dtype)})\nresult = _r[1]\n_result_orig = _r[0]`,
        );
        expectDType(q, py.orig_dtype!, `batched qr ${dtype} q`);
        expectDType(r, py.dtype, `batched qr ${dtype} r`);
        expect(q.shape).toEqual(py.orig_shape);
        expect(r.shape).toEqual(py.shape);
        for (let b = 0; b < stack.length; b++) {
          const qm = mat(sliceAt(q, b));
          const rm = mat(sliceAt(r, b));
          expectUnitaryColumns(qm, `batched qr ${dtype} q[${b}]`, tol);
          expectUpperTriangular(rm, `batched qr ${dtype} r[${b}]`, tol);
          expectReconstructs(
            matmulC(qm, rm),
            mat(stack[b] as any),
            `batched qr ${dtype}[${b}]`,
            tol,
          );
        }
        const rOnly = np.linalg.qr(a, 'r') as any;
        expect(rOnly.shape).toEqual(py.shape);
        expectDType(rOnly, py.dtype, `batched qr r-mode ${dtype}`);
      });

      it(`pinv ${dtype}`, () => {
        const a = array(stack as any, dtype);
        const p = np.linalg.pinv(a) as any;
        const py = runNumPy(`result = np.linalg.pinv(${pyArr(stack, dtype)})`);
        expectDType(p, py.dtype, `batched pinv ${dtype}`);
        expect(p.shape).toEqual(py.shape);
        expect(arraysClose(p.toArray(), py.value, 1e-4)).toBe(true);
      });

      // The real part of a Hermitian positive-definite matrix is symmetric
      // positive definite, which gives the float32 case its input.
      const sq =
        dtype === 'float32'
          ? [hpd(3, 31), hpd(3, 32)].map((m) => m.map((row) => row.map((c) => c.re)))
          : [hpd(3, 31), hpd(3, 32)];

      it(`cholesky ${dtype}`, () => {
        const a = array(sq as any, dtype);
        const l = np.linalg.cholesky(a) as any;
        const py = runNumPy(`result = np.linalg.cholesky(${pyArr(sq, dtype)})`);
        expectDType(l, py.dtype, `batched cholesky ${dtype}`);
        expect(arraysClose(l.toArray(), py.value, 1e-4)).toBe(true);
      });

      it(`eigh ${dtype}`, () => {
        const a = array(sq as any, dtype);
        const { w, v } = np.linalg.eigh(a) as any;
        const py = runNumPy(
          `_r = np.linalg.eigh(${pyArr(sq, dtype)})\nresult = _r[0]\n_result_orig = _r[1]`,
        );
        expectDType(w, py.dtype, `batched eigh ${dtype} w`);
        expectDType(v, py.orig_dtype!, `batched eigh ${dtype} v`);
        expect(arraysClose(w.toArray(), py.value, 1e-4)).toBe(true);
        for (let b = 0; b < sq.length; b++) {
          const vm = mat(sliceAt(v, b));
          const wd = diagC(sliceAt(w, b));
          expectUnitaryColumns(vm, `batched eigh ${dtype} v[${b}]`, tol);
          expectReconstructs(
            matmulC(matmulC(vm, wd), conjT(vm)),
            mat(sq[b] as any),
            `batched eigh ${dtype}[${b}]`,
            1e3 * tol,
          );
        }
      });
    }
  });

  describe('mixed real and complex operands', () => {
    const aData = [
      [4, 1, 0.5],
      [1, 3, 0.25],
      [0.5, 0.25, 2],
    ];
    const bVec = [new Complex(1, 2), new Complex(-1, 0.5), new Complex(0, -3)];
    const bMat = [
      [new Complex(1, 2), new Complex(0, 1)],
      [new Complex(-1, 0.5), new Complex(2, 0)],
      [new Complex(0, -3), new Complex(1, 1)],
    ];

    for (const [aDtype, bDtype] of [
      ['float64', 'complex128'],
      ['float32', 'complex64'],
      ['float64', 'complex64'],
      ['int32', 'complex128'],
    ] as const) {
      it(`solve ${aDtype} A with ${bDtype} b`, () => {
        const a = array(aData, aDtype);
        for (const bData of [bVec, bMat]) {
          const b = array(bData as any, bDtype);
          const x = np.linalg.solve(a, b);
          const py = runNumPy(
            `result = np.linalg.solve(${pyArr(aData, aDtype)}, ${pyArr(bData, bDtype)})`,
          );
          expectDType(x, py.dtype, `solve ${aDtype}/${bDtype}`);
          expect(arraysClose(x.toArray() as any, py.value, 1e-4)).toBe(true);
        }
      });
    }

    it('solve complex128 A with real b', () => {
      const cData = complexMatrix(3, 3, 41);
      const a = array(cData as any, 'complex128');
      const b = array([1, 2, 3], 'float64');
      const x = np.linalg.solve(a, b);
      const py = runNumPy(
        `result = np.linalg.solve(${pyArr(cData, 'complex128')}, np.array([1.0, 2.0, 3.0]))`,
      );
      expectDType(x, py.dtype, 'solve complex/real');
      expect(arraysClose(x.toArray() as any, py.value, 1e-9)).toBe(true);
    });

    for (const [aDtype, bDtype] of [
      ['float64', 'complex128'],
      ['float32', 'complex64'],
    ] as const) {
      it(`lstsq ${aDtype} A with ${bDtype} b`, () => {
        const tall = realMatrix(5, 3, 51);
        const bData = complexMatrix(5, 1, 52).map((row) => row[0]!);
        const a = array(tall, aDtype);
        const b = array(bData as any, bDtype);
        const { x, residuals, rank, s } = np.linalg.lstsq(a, b) as any;
        const call = `np.linalg.lstsq(${pyArr(tall, aDtype)}, ${pyArr(bData, bDtype)})`;
        const pyX = runNumPy(`result = ${call}[0]`);
        const pyR = runNumPy(`result = ${call}[1]`);
        const pyS = runNumPy(`result = ${call}[3]`);
        const pyRank = runNumPy(`result = int(${call}[2])`);
        const rtol = aDtype === 'float32' ? 1e-3 : 1e-8;
        expectDType(x, pyX.dtype, 'lstsq x');
        expectDType(residuals, pyR.dtype, 'lstsq residuals');
        expectDType(s, pyS.dtype, 'lstsq s');
        expect(rank).toBe(pyRank.value);
        expect(arraysClose(x.toArray(), pyX.value, rtol)).toBe(true);
        expect(arraysClose(residuals.toArray(), pyR.value, rtol)).toBe(true);
        expect(arraysClose(s.toArray(), pyS.value, rtol)).toBe(true);
      });
    }
  });

  describe('eig', () => {
    for (const [label, data] of [
      [
        'real spectrum',
        [
          [2, 1],
          [1, 3],
        ],
      ],
      [
        'real spectrum, non-symmetric',
        [
          [1, 2],
          [3, 4],
        ],
      ],
      [
        'complex spectrum',
        [
          [0, -1],
          [1, 0],
        ],
      ],
    ] as const) {
      it(`float32 ${label} keeps single precision`, () => {
        const a = array(data as any, 'float32');
        const { w, v } = np.linalg.eig(a) as any;
        const py = runNumPy(
          `_r = np.linalg.eig(${pyArr(data, 'float32')})\nresult = _r[0]\n_result_orig = _r[1]`,
        );
        expectDType(w, py.dtype, `eig float32 ${label} w`);
        expectDType(v, py.orig_dtype!, `eig float32 ${label} v`);
        const vals = np.linalg.eigvals(a);
        expectDType(vals, py.dtype, `eigvals float32 ${label}`);
        const jsAbs = np.sort(np.absolute(w)).toArray();
        const pyAbs = runNumPy(
          `result = np.sort(np.abs(np.linalg.eigvals(${pyArr(data, 'float32')})))`,
        );
        expect(arraysClose(jsAbs, pyAbs.value, 1e-5)).toBe(true);
      });
    }

    for (const bad of ['NaN', 'Infinity']) {
      for (const dtype of ['float64', 'complex128'] as const) {
        it(`${dtype} input containing ${bad} is rejected, as NumPy does`, () => {
          const v = bad === 'NaN' ? Number.NaN : Number.POSITIVE_INFINITY;
          const data = [
            [1, v],
            [0, 1],
          ];
          const pyV = bad === 'NaN' ? 'np.nan' : 'np.inf';
          expect(() =>
            runNumPy(
              `result = np.linalg.eig(np.array([[1, ${pyV}], [0, 1]], dtype=np.${dtype}))[0]`,
            ),
          ).toThrow(/infs or NaNs/);
          const a = array(data, dtype);
          expect(() => np.linalg.eig(a)).toThrow(/infs or NaNs/);
          expect(() => np.linalg.eigvals(a)).toThrow(/infs or NaNs/);
        });
      }
    }
  });

  describe('slogdet', () => {
    it('empty complex matrix has complex sign 1', () => {
      const a = np.zeros([0, 0], 'complex128');
      const { sign, logabsdet } = np.linalg.slogdet(a) as any;
      const py = runNumPy('result = np.linalg.slogdet(np.zeros((0, 0), dtype=np.complex128))[0]');
      expect(py.dtype).toBe('complex128');
      expect(sign).toBeInstanceOf(Complex);
      expect(sign.re).toBe(1);
      expect(sign.im).toBe(0);
      expect(logabsdet).toBe(0);
    });

    it('batched complex64 keeps float32 logabsdet', () => {
      const stack = [complexMatrix(3, 3, 61), complexMatrix(3, 3, 62)];
      const a = array(stack as any, 'complex64');
      const { sign, logabsdet } = np.linalg.slogdet(a) as any;
      const py = runNumPy(
        `_r = np.linalg.slogdet(${pyArr(stack, 'complex64')})\nresult = _r[1]\n_result_orig = _r[0]`,
      );
      expectDType(logabsdet, py.dtype, 'slogdet complex64 logabsdet');
      expectDType(sign, py.orig_dtype!, 'slogdet complex64 sign');
      expect(arraysClose(logabsdet.toArray(), py.value, 1e-4)).toBe(true);
    });
  });
});

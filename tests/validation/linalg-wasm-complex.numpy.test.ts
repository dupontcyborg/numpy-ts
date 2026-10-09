/**
 * NumPy validation for the complex WASM linalg kernels on inputs that the
 * generic sweeps miss: offset views and wide SVD.
 */

import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { array, Complex, linalg, type NDArray, wasmConfig } from '../../src';
import { wasmCholeskyComplex } from '../../src/common/wasm/cholesky';
import { wasmSvdComplex } from '../../src/common/wasm/svd';
import { checkNumPyAvailable, runNumPy } from './numpy-oracle';

// FORCE_BACKEND=js routes everything through JS; otherwise the default
// thresholds already send these shapes to WASM, and each test checks that.
const forced = process.env.FORCE_BACKEND as 'js' | 'wasm' | undefined;
const expectWasm = forced !== 'js';

type CDtype = 'complex128' | 'complex64';
type Pairs = [number, number][][];

function lcg(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s * 1103515245 + 12345) % 2147483648;
    return s / 2147483648 - 0.5;
  };
}

function randomPairs(rows: number, cols: number, rnd: () => number): Pairs {
  return Array.from({ length: rows }, () =>
    Array.from({ length: cols }, () => [rnd(), rnd()] as [number, number]),
  );
}

/** B·B^H + n·I, which is Hermitian positive definite. */
function hermitianPD(n: number, rnd: () => number): Pairs {
  const b = randomPairs(n, n, rnd);
  const h: Pairs = [];
  for (let i = 0; i < n; i++) {
    const row: [number, number][] = [];
    for (let j = 0; j < n; j++) {
      let re = i === j ? n : 0;
      let im = 0;
      for (let k = 0; k < n; k++) {
        const [ar, ai] = b[i]![k]!;
        const [br, bi] = b[j]![k]!;
        re += ar * br + ai * bi;
        im += ai * br - ar * bi;
      }
      row.push([re, im]);
    }
    h.push(row);
  }
  return h;
}

function toComplexNested(p: Pairs | Pairs[]): unknown {
  return (p as unknown[]).map((x) =>
    typeof (x as unknown[])[0] === 'number'
      ? new Complex((x as number[])[0]!, (x as number[])[1]!)
      : toComplexNested(x as Pairs),
  );
}

function toPairs(a: NDArray): unknown {
  const walk = (x: unknown): unknown =>
    Array.isArray(x) ? x.map(walk) : [(x as Complex).re, (x as Complex).im];
  return walk(a.toArray());
}

const PY_PAIRS =
  'def c(j):\n    x = np.array(j, dtype=np.float64)\n    return x[..., 0] + 1j * x[..., 1]\n';

describe('NumPy Validation: complex WASM linalg kernels', () => {
  let savedMultiplier: number;

  beforeAll(() => {
    if (!checkNumPyAvailable()) {
      throw new Error('Python NumPy not available');
    }
    savedMultiplier = wasmConfig.thresholdMultiplier;
    if (forced === 'js') wasmConfig.thresholdMultiplier = Infinity;
    else if (forced === 'wasm') wasmConfig.thresholdMultiplier = 0;
  });

  afterAll(() => {
    wasmConfig.thresholdMultiplier = savedMultiplier;
  });

  describe('cholesky on an offset view', () => {
    for (const dtype of ['complex128', 'complex64'] as CDtype[]) {
      it(`matches NumPy for stack[1] of a (2, 6, 6) ${dtype} stack`, () => {
        const n = 6;
        const rnd = lcg(dtype === 'complex128' ? 7 : 11);
        const stackPairs = [hermitianPD(n, rnd), hermitianPD(n, rnd)];
        const stack = array(toComplexNested(stackPairs) as Complex[][][], dtype);
        const view = stack.slice('1') as NDArray;
        expect(view.storage.offset).toBe(n * n);
        expect(view.storage.isCContiguous).toBe(true);
        if (expectWasm) expect(wasmCholeskyComplex(view.storage)).not.toBeNull();

        const L = linalg.cholesky(view) as NDArray;
        expect(L.dtype).toBe(dtype);

        const py = runNumPy(`
${PY_PAIRS}
stack = c(${JSON.stringify(stackPairs)}).astype(np.${dtype})
L = c(${JSON.stringify(toPairs(L))})
want = np.linalg.cholesky(stack[1]).astype(np.complex128)
result = float(np.abs(L - want).max() / np.abs(want).max())
`);
        expect(py.value).toBeLessThan(dtype === 'complex128' ? 1e-12 : 1e-5);
      });
    }
  });

  describe('wide SVD', () => {
    const shapes: [number, number][] = [
      [4, 7],
      [4, 9],
      [6, 20],
    ];
    for (const dtype of ['complex128', 'complex64'] as CDtype[]) {
      for (const [m, n] of shapes) {
        it(`${dtype} ${m}x${n}: Vh is unitary and U diag(s) Vh reconstructs A`, () => {
          const aPairs = randomPairs(m, n, lcg(m * 100 + n));
          const a = array(toComplexNested(aPairs) as Complex[][], dtype);
          if (expectWasm) expect(wasmSvdComplex(a.storage)).not.toBeNull();

          const { u, s, vt } = linalg.svd(a) as { u: NDArray; s: NDArray; vt: NDArray };
          expect(vt.shape).toEqual([n, n]);

          const py = runNumPy(`
${PY_PAIRS}
A = c(${JSON.stringify(aPairs)}).astype(np.${dtype}).astype(np.complex128)
U = c(${JSON.stringify(toPairs(u))})
Vh = c(${JSON.stringify(toPairs(vt))})
s = np.array(${JSON.stringify(s.toArray())}, dtype=np.float64)
k = ${m}
I = np.eye(${n})
result = [
    float(np.abs(Vh.conj().T @ Vh - I).max()),
    float(np.abs(Vh @ Vh.conj().T - I).max()),
    float(np.abs((U[:, :k] * s) @ Vh[:k] - A).max()),
    float(np.abs(s - np.linalg.svd(A, compute_uv=False)).max()),
]
`);
          const tol = dtype === 'complex128' ? 1e-10 : 1e-4;
          const [vhVh, vhVhT, recon, sigma] = py.value as number[];
          expect(vhVh).toBeLessThan(tol);
          expect(vhVhT).toBeLessThan(tol);
          expect(recon).toBeLessThan(tol);
          expect(sigma).toBeLessThan(tol);
        });
      }
    }
  });
});

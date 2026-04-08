/**
 * DType Sweep: Complex number operations.
 * Tests complex128 and complex64 across math/reduction/linalg, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { checkNumPyAvailable, runNumPyBatch, expectMatchPre } from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array, Complex } = np;
const COMPLEX = ['complex128', 'complex64'] as const;

function npDtype(d: string) {
  return `np.${d}`;
}

// Helper: compare a JS Complex scalar against a NumPy result
function expectComplexClose(js: any, py: NumPyResult & { error?: string }, tol = 1e-4) {
  if (py.error) throw new Error(`NumPy error: ${py.error}`);
  const jsC = js instanceof Complex ? js : new Complex(Number(js), 0);
  const pyVal = py.value;
  const pyRe = pyVal.re ?? pyVal[0] ?? pyVal;
  const pyIm = pyVal.im ?? pyVal[1] ?? 0;
  expect(jsC.re).toBeCloseTo(Number(pyRe), tol > 0.01 ? 2 : 4);
  expect(jsC.im).toBeCloseTo(Number(pyIm), tol > 0.01 ? 2 : 4);
}

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};
  const data = '[1+2j, 3+4j, 5+6j]';

  for (const dtype of COMPLEX) {
    const npCast = dtype === 'complex64' ? '.astype(np.complex128)' : '';

    // Unary ops
    snippets[`absolute_${dtype}`] = `
_result_orig = np.absolute(np.array(${data}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`conj_${dtype}`] = `
_result_orig = np.conj(np.array(${data}, dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    snippets[`real_${dtype}`] = `
_result_orig = np.real(np.array(${data}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`imag_${dtype}`] = `
_result_orig = np.imag(np.array(${data}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`angle_${dtype}`] = `
_result_orig = np.angle(np.array(${data}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`negative_${dtype}`] = `
_result_orig = np.negative(np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    snippets[`sqrt_${dtype}`] = `
_result_orig = np.sqrt(np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    snippets[`exp_${dtype}`] = `
_result_orig = np.exp(np.array([0+0j, 1+0j], dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    snippets[`sin_${dtype}`] = `
_result_orig = np.sin(np.array([1+0j, 0+1j], dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    snippets[`cos_${dtype}`] = `
_result_orig = np.cos(np.array([1+0j, 0+1j], dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    // Binary ops
    snippets[`add_${dtype}`] = `
_result_orig = np.add(np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}), np.array([5+6j, 7+8j], dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    snippets[`multiply_${dtype}`] = `
_result_orig = np.multiply(np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}), np.array([5+6j, 7+8j], dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    snippets[`subtract_${dtype}`] = `
_result_orig = np.subtract(np.array([5+6j, 7+8j], dtype=${npDtype(dtype)}), np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}))
result = _result_orig${npCast}`;

    // Reductions (scalar complex results)
    snippets[`sum_${dtype}`] =
      `result = complex(np.sum(np.array(${data}, dtype=${npDtype(dtype)})))`;

    snippets[`mean_${dtype}`] =
      `result = complex(np.mean(np.array(${data}, dtype=${npDtype(dtype)})))`;

    // Linalg
    snippets[`dot_${dtype}`] = `
a = np.array(${data}, dtype=${npDtype(dtype)})
result = complex(np.dot(a, a))`;

    snippets[`inner_${dtype}`] = `
a = np.array(${data}, dtype=${npDtype(dtype)})
result = complex(np.inner(a, a))`;

    snippets[`matmul_${dtype}`] = `
_result_orig = np.array([[1+0j, 0+1j], [1+1j, 1-1j]], dtype=${npDtype(dtype)}) @ np.array([[1+0j, 0+1j], [1+1j, 1-1j]], dtype=${npDtype(dtype)})
result = _result_orig${npCast}`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Complex', () => {
  for (const dtype of COMPLEX) {
    const tol = dtype === 'complex64' ? 1e-2 : 1e-6;

    describe(`unary ${dtype}`, () => {
      it(`absolute`, () => {
        const jsResult = np.absolute(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        expectMatchPre(jsResult, oracle.get(`absolute_${dtype}`)!, { rtol: tol });
      });

      it(`conj`, () => {
        const jsResult = np.conj(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        expectMatchPre(jsResult, oracle.get(`conj_${dtype}`)!, { rtol: tol });
      });

      it(`real`, () => {
        const jsResult = np.real(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        expectMatchPre(jsResult, oracle.get(`real_${dtype}`)!, { rtol: tol });
      });

      it(`imag`, () => {
        const jsResult = np.imag(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        expectMatchPre(jsResult, oracle.get(`imag_${dtype}`)!, { rtol: tol });
      });

      it(`angle`, () => {
        const jsResult = np.angle(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        expectMatchPre(jsResult, oracle.get(`angle_${dtype}`)!, { rtol: tol });
      });

      it(`negative`, () => {
        const jsResult = np.negative(array([new Complex(1, 2), new Complex(3, 4)], dtype));
        expectMatchPre(jsResult, oracle.get(`negative_${dtype}`)!, { rtol: tol });
      });

      it(`sqrt`, () => {
        const jsResult = np.sqrt(array([new Complex(1, 2), new Complex(3, 4)], dtype));
        expectMatchPre(jsResult, oracle.get(`sqrt_${dtype}`)!, { rtol: tol });
      });

      it(`exp`, () => {
        const jsResult = np.exp(array([new Complex(0, 0), new Complex(1, 0)], dtype));
        expectMatchPre(jsResult, oracle.get(`exp_${dtype}`)!, { rtol: tol });
      });

      it(`sin`, () => {
        const jsResult = np.sin(array([new Complex(1, 0), new Complex(0, 1)], dtype));
        expectMatchPre(jsResult, oracle.get(`sin_${dtype}`)!, { rtol: tol });
      });

      it(`cos`, () => {
        const jsResult = np.cos(array([new Complex(1, 0), new Complex(0, 1)], dtype));
        expectMatchPre(jsResult, oracle.get(`cos_${dtype}`)!, { rtol: tol });
      });
    });

    describe(`binary ${dtype}`, () => {
      it(`add`, () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)], dtype);
        const b = array([new Complex(5, 6), new Complex(7, 8)], dtype);
        const jsResult = np.add(a, b);
        expectMatchPre(jsResult, oracle.get(`add_${dtype}`)!, { rtol: tol });
      });

      it(`multiply`, () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)], dtype);
        const b = array([new Complex(5, 6), new Complex(7, 8)], dtype);
        const jsResult = np.multiply(a, b);
        expectMatchPre(jsResult, oracle.get(`multiply_${dtype}`)!, { rtol: tol });
      });

      it(`subtract`, () => {
        const a = array([new Complex(5, 6), new Complex(7, 8)], dtype);
        const b = array([new Complex(1, 2), new Complex(3, 4)], dtype);
        const jsResult = np.subtract(a, b);
        expectMatchPre(jsResult, oracle.get(`subtract_${dtype}`)!, { rtol: tol });
      });
    });

    describe(`reductions ${dtype}`, () => {
      it(`sum`, () => {
        const jsResult = np.sum(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        expectComplexClose(jsResult, oracle.get(`sum_${dtype}`)!, tol);
      });

      it(`mean`, () => {
        const jsResult = np.mean(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        expectComplexClose(jsResult, oracle.get(`mean_${dtype}`)!, tol);
      });
    });

    describe(`linalg ${dtype}`, () => {
      it(`dot`, () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype);
        const jsResult = np.dot(a, a);
        expectComplexClose(jsResult, oracle.get(`dot_${dtype}`)!, tol);
      });

      it(`inner`, () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype);
        const jsResult = np.inner(a, a);
        expectComplexClose(jsResult, oracle.get(`inner_${dtype}`)!, tol);
      });

      it(`matmul`, () => {
        const a = array(
          [
            [new Complex(1, 0), new Complex(0, 1)],
            [new Complex(1, 1), new Complex(1, -1)],
          ],
          dtype
        );
        const jsResult = np.matmul(a, a);
        expectMatchPre(jsResult, oracle.get(`matmul_${dtype}`)!, { rtol: tol });
      });
    });

    it(`sort ${dtype}`, () => {
      const a = array([new Complex(3, 1), new Complex(1, 2), new Complex(2, 3)], dtype);
      const jsResult = np.sort(a);
      // Just verify it doesn't crash and shape is preserved
      expect(jsResult.shape).toEqual([3]);
    });
  }
});

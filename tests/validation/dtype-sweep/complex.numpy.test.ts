/**
 * DType Sweep: Complex number operations.
 * Tests complex128 and complex64 across math/reduction/linalg, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { runNumPy, arraysClose, checkNumPyAvailable } from './_helpers';

const { array, Complex } = np;
const COMPLEX = ['complex128', 'complex64'] as const;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

function npDtype(d: string) {
  return `np.${d}`;
}

// Helper: compare a JS Complex scalar against a NumPy result
function expectComplexClose(js: any, py: any, tol = 1e-4) {
  const jsC = js instanceof Complex ? js : new Complex(Number(js), 0);
  const pyRe = py.re ?? py[0] ?? py;
  const pyIm = py.im ?? py[1] ?? 0;
  expect(jsC.re).toBeCloseTo(Number(pyRe), tol > 0.01 ? 2 : 4);
  expect(jsC.im).toBeCloseTo(Number(pyIm), tol > 0.01 ? 2 : 4);
}

describe('DType Sweep: Complex', () => {
  for (const dtype of COMPLEX) {
    const tol = dtype === 'complex64' ? 1e-2 : 1e-6;
    const npCast = dtype === 'complex64' ? '.astype(np.complex128)' : '';
    const data = '[1+2j, 3+4j, 5+6j]';

    describe(`unary ${dtype}`, () => {
      it(`absolute`, () => {
        const jsResult = np.absolute(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        const pyResult = runNumPy(
          `result = np.absolute(np.array(${data}, dtype=${npDtype(dtype)})).astype(np.float64)`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`conj`, () => {
        const jsResult = np.conj(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        const pyResult = runNumPy(
          `result = np.conj(np.array(${data}, dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`real`, () => {
        const jsResult = np.real(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        const pyResult = runNumPy(
          `result = np.real(np.array(${data}, dtype=${npDtype(dtype)})).astype(np.float64)`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`imag`, () => {
        const jsResult = np.imag(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        const pyResult = runNumPy(
          `result = np.imag(np.array(${data}, dtype=${npDtype(dtype)})).astype(np.float64)`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`angle`, () => {
        const jsResult = np.angle(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        const pyResult = runNumPy(
          `result = np.angle(np.array(${data}, dtype=${npDtype(dtype)})).astype(np.float64)`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`negative`, () => {
        const jsResult = np.negative(array([new Complex(1, 2), new Complex(3, 4)], dtype));
        const pyResult = runNumPy(
          `result = np.negative(np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`sqrt`, () => {
        const jsResult = np.sqrt(array([new Complex(1, 2), new Complex(3, 4)], dtype));
        const pyResult = runNumPy(
          `result = np.sqrt(np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`exp`, () => {
        const jsResult = np.exp(array([new Complex(0, 0), new Complex(1, 0)], dtype));
        const pyResult = runNumPy(
          `result = np.exp(np.array([0+0j, 1+0j], dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`sin`, () => {
        const jsResult = np.sin(array([new Complex(1, 0), new Complex(0, 1)], dtype));
        const pyResult = runNumPy(
          `result = np.sin(np.array([1+0j, 0+1j], dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`cos`, () => {
        const jsResult = np.cos(array([new Complex(1, 0), new Complex(0, 1)], dtype));
        const pyResult = runNumPy(
          `result = np.cos(np.array([1+0j, 0+1j], dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });
    });

    describe(`binary ${dtype}`, () => {
      it(`add`, () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)], dtype);
        const b = array([new Complex(5, 6), new Complex(7, 8)], dtype);
        const jsResult = np.add(a, b);
        const pyResult = runNumPy(
          `result = np.add(np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}), np.array([5+6j, 7+8j], dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`multiply`, () => {
        const a = array([new Complex(1, 2), new Complex(3, 4)], dtype);
        const b = array([new Complex(5, 6), new Complex(7, 8)], dtype);
        const jsResult = np.multiply(a, b);
        const pyResult = runNumPy(
          `result = np.multiply(np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}), np.array([5+6j, 7+8j], dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });

      it(`subtract`, () => {
        const a = array([new Complex(5, 6), new Complex(7, 8)], dtype);
        const b = array([new Complex(1, 2), new Complex(3, 4)], dtype);
        const jsResult = np.subtract(a, b);
        const pyResult = runNumPy(
          `result = np.subtract(np.array([5+6j, 7+8j], dtype=${npDtype(dtype)}), np.array([1+2j, 3+4j], dtype=${npDtype(dtype)}))${npCast}`
        );
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
      });
    });

    describe(`reductions ${dtype}`, () => {
      it(`sum`, () => {
        const jsResult = np.sum(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        const pyResult = runNumPy(
          `result = complex(np.sum(np.array(${data}, dtype=${npDtype(dtype)})))`
        );
        expectComplexClose(jsResult, pyResult.value, tol);
      });

      it(`mean`, () => {
        const jsResult = np.mean(
          array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype)
        );
        const pyResult = runNumPy(
          `result = complex(np.mean(np.array(${data}, dtype=${npDtype(dtype)})))`
        );
        expectComplexClose(jsResult, pyResult.value, tol);
      });
    });

    describe(`linalg ${dtype}`, () => {
      it(`dot`, () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype);
        const jsResult = np.dot(a, a);
        const pyResult = runNumPy(`
a = np.array(${data}, dtype=${npDtype(dtype)})
result = complex(np.dot(a, a))
        `);
        expectComplexClose(jsResult, pyResult.value, tol);
      });

      it(`inner`, () => {
        const a = array([new Complex(1, 2), new Complex(3, 4), new Complex(5, 6)], dtype);
        const jsResult = np.inner(a, a);
        const pyResult = runNumPy(`
a = np.array(${data}, dtype=${npDtype(dtype)})
result = complex(np.inner(a, a))
        `);
        expectComplexClose(jsResult, pyResult.value, tol);
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
        const pyResult = runNumPy(`
a = np.array([[1+0j, 0+1j], [1+1j, 1-1j]], dtype=${npDtype(dtype)})
result = (a @ a)${npCast}
        `);
        expect(arraysClose(jsResult.toArray(), pyResult.value, tol)).toBe(true);
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

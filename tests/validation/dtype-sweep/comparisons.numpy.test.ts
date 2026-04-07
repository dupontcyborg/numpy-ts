/**
 * DType Sweep: Comparison functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isComplex } from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Comparisons', () => {
  const compOps = ['greater', 'greater_equal', 'less', 'less_equal', 'equal', 'not_equal'];

  for (const name of compOps) {
    describe(name, () => {
      for (const dtype of ALL_DTYPES) {
        it(`${dtype}`, () => {
          const data1 = dtype === 'bool' ? [1, 0, 1, 0] : isComplex(dtype) ? [1, 2, 3, 4] : [1, 2, 3, 4];
          const data2 = dtype === 'bool' ? [0, 1, 1, 0] : isComplex(dtype) ? [4, 3, 2, 1] : [4, 3, 2, 1];
          const a = array(data1, dtype);
          const b = array(data2, dtype);
          const jsResult = (np as any)[name](a, b);
          const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.${name}(a, b).astype(np.float64)
          `);
          expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
        });
      }
    });
  }
});

describe('DType Sweep: Close comparisons', () => {
  for (const dtype of ALL_DTYPES) {
    it(`isclose ${dtype}`, () => {
      const data1 = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
      const data2 = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
      const jsResult = np.isclose(array(data1, dtype), array(data2, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data1)}, dtype=${npDtype(dtype)})
b = np.array(${JSON.stringify(data2)}, dtype=${npDtype(dtype)})
result = np.isclose(a, b).astype(np.float64)
      `);
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`allclose ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1.0, 2.0, 3.0];
      const jsResult = np.allclose(array(data, dtype), array(data, dtype));
      const pyResult = runNumPy(`
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = bool(np.allclose(a, a))
      `);
      expect(Boolean(jsResult)).toBe(Boolean(pyResult.value));
    });
  }
});

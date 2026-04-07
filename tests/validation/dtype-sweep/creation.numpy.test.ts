/**
 * DType Sweep: Creation functions.
 * Validates that arrays are created with correct dtype and shape, across ALL dtypes.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import { ALL_DTYPES, runNumPy, arraysClose, checkNumPyAvailable, npDtype, isComplex } from './_helpers';

const { array } = np;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');
});

describe('DType Sweep: Creation', () => {
  for (const dtype of ALL_DTYPES) {
    it(`array ${dtype}`, () => {
      const a = array(
        isComplex(dtype) ? [1, 2, 3] : dtype === 'bool' ? [1, 0, 1] : [1, 2, 3],
        dtype
      );
      expect(a.dtype).toBe(dtype);
      expect(a.shape).toEqual([3]);
    });

    it(`zeros ${dtype}`, () => {
      expect(np.zeros([3], dtype).dtype).toBe(dtype);
    });

    it(`ones ${dtype}`, () => {
      expect(np.ones([3], dtype).dtype).toBe(dtype);
    });

    it(`full ${dtype}`, () => {
      expect(np.full([3], dtype === 'bool' ? 1 : 5, dtype).dtype).toBe(dtype);
    });

    it(`empty ${dtype}`, () => {
      expect(np.empty([3], dtype).dtype).toBe(dtype);
    });

    it(`eye ${dtype}`, () => {
      expect(np.eye(3, undefined, undefined, dtype).dtype).toBe(dtype);
    });

    it(`identity ${dtype}`, () => {
      expect(np.identity(3, dtype).dtype).toBe(dtype);
    });

    it(`asarray ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : isComplex(dtype) ? [1, 2] : [1, 2], dtype);
      expect(np.asarray(a).dtype).toBe(dtype);
    });

    it(`ascontiguousarray ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : isComplex(dtype) ? [1, 2] : [1, 2], dtype);
      expect(np.ascontiguousarray(a).dtype).toBe(dtype);
    });

    it(`asfortranarray ${dtype}`, () => {
      const a = array(dtype === 'bool' ? [1, 0] : isComplex(dtype) ? [1, 2] : [1, 2], dtype);
      expect(np.asfortranarray(a).dtype).toBe(dtype);
    });

    it(`arange ${dtype}`, () => {
      const jsResult = np.arange(0, 5, 1, dtype);
      const pyResult = runNumPy(
        `result = np.arange(0, 5, 1, dtype=${npDtype(dtype)}).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value)).toBe(true);
    });

    it(`linspace ${dtype}`, () => {
      const jsResult = np.linspace(0, 1, 5, dtype);
      const pyResult = runNumPy(
        `result = np.linspace(0, 1, 5, dtype=${npDtype(dtype)}).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-4)).toBe(true);
    });

    it(`logspace ${dtype}`, () => {
      const jsResult = np.logspace(0, 2, 5, undefined, dtype);
      const pyResult = runNumPy(
        `result = np.logspace(0, 2, 5, dtype=${npDtype(dtype)}).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-3)).toBe(true);
    });

    it(`geomspace ${dtype}`, () => {
      const jsResult = np.geomspace(1, 100, 5, dtype);
      const pyResult = runNumPy(
        `result = np.geomspace(1, 100, 5, dtype=${npDtype(dtype)}).astype(np.float64)`
      );
      expect(arraysClose(jsResult.toArray(), pyResult.value, 1e-3)).toBe(true);
    });
  }
});

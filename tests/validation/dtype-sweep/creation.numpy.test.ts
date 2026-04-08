/**
 * DType Sweep: Creation functions.
 * Validates that arrays are created with correct dtype and shape, across ALL dtypes.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  runNumPyBatch,
  expectBothReject,
  expectMatchPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    snippets[`arange_${dtype}`] = `
_result_orig = np.arange(0, 5, 1, dtype=${npDtype(dtype)})
result = _result_orig.astype(np.float64)`;

    snippets[`linspace_${dtype}`] = `
_result_orig = np.linspace(0, 1, 5, dtype=${npDtype(dtype)})
result = _result_orig.astype(np.float64)`;

    snippets[`logspace_${dtype}`] = `
_result_orig = np.logspace(0, 2, 5, dtype=${npDtype(dtype)})
result = _result_orig.astype(np.float64)`;

    snippets[`geomspace_${dtype}`] = `
_result_orig = np.geomspace(1, 100, 5, dtype=${npDtype(dtype)})
result = _result_orig.astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
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
      // arange(bool) with length > 2: NumPy rejects
      if (dtype === 'bool') {
        const pyCode = `
_result_orig = np.arange(0, 5, 1, dtype=${npDtype(dtype)})
result = _result_orig.astype(np.float64)`;
        const _r = expectBothReject(
          'arange(bool) with length > 2 not supported by NumPy',
          () => np.arange(0, 5, 1, dtype),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.arange(0, 5, 1, dtype);
      expectMatchPre(jsResult, oracle.get(`arange_${dtype}`)!);
    });

    it(`linspace ${dtype}`, () => {
      const jsResult = np.linspace(0, 1, 5, dtype);
      expectMatchPre(jsResult, oracle.get(`linspace_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`logspace ${dtype}`, () => {
      const jsResult = np.logspace(0, 2, 5, undefined, dtype);
      expectMatchPre(jsResult, oracle.get(`logspace_${dtype}`)!, { rtol: 1e-3 });
    });

    it(`geomspace ${dtype}`, () => {
      const jsResult = np.geomspace(1, 100, 5, dtype);
      expectMatchPre(jsResult, oracle.get(`geomspace_${dtype}`)!, { rtol: 1e-3 });
    });
  }
});

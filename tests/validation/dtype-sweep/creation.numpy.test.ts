/**
 * DType Sweep: Creation functions.
 * Validates that arrays are created with correct dtype and shape, across ALL dtypes.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, expect, it } from 'vitest';
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
  runNumPyBatch,
} from './_helpers';

const { array } = np;

// One definition per fixture, shared by the oracle snippets and the test bodies.
const data3 = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
const data2 = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0] : [1, 2], dtype);

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);

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

    snippets[`eye_${dtype}`] = `
_result_orig = np.eye(3, dtype=${npDtype(dtype)})
result = _result_orig.astype(${ac})`;

    snippets[`identity_${dtype}`] = `
_result_orig = np.identity(3, dtype=${npDtype(dtype)})
result = _result_orig.astype(${ac})`;

    snippets[`array_${dtype}`] = `
_result_orig = np.array(${data3(dtype).py}, dtype=${npDtype(dtype)})
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Creation', () => {
  for (const dtype of ALL_DTYPES) {
    it(`array ${dtype}`, () => {
      const a = array(data3(dtype).js, dtype);
      expectComplexFixture(a, dtype, `array ${dtype}`);
      expect(a.dtype).toBe(dtype);
      expect(a.shape).toEqual([3]);
      expectMatchPre(a, oracle.get(`array_${dtype}`)!);
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
      const jsResult = np.eye(3, undefined, undefined, dtype);
      expect(jsResult.dtype).toBe(dtype);
      expectMatchPre(jsResult, oracle.get(`eye_${dtype}`)!);
    });

    it(`identity ${dtype}`, () => {
      const jsResult = np.identity(3, dtype);
      expect(jsResult.dtype).toBe(dtype);
      expectMatchPre(jsResult, oracle.get(`identity_${dtype}`)!);
    });

    it(`asarray ${dtype}`, () => {
      const a = array(data2(dtype).js, dtype);
      expectComplexFixture(a, dtype, `asarray ${dtype}`);
      const r = np.asarray(a);
      expect(r.dtype).toBe(dtype);
      expectComplexFixture(r, dtype, `asarray ${dtype} result`);
    });

    it(`ascontiguousarray ${dtype}`, () => {
      const a = array(data2(dtype).js, dtype);
      expectComplexFixture(a, dtype, `ascontiguousarray ${dtype}`);
      const r = np.ascontiguousarray(a);
      expect(r.dtype).toBe(dtype);
      expectComplexFixture(r, dtype, `ascontiguousarray ${dtype} result`);
    });

    it(`asfortranarray ${dtype}`, () => {
      const a = array(data2(dtype).js, dtype);
      expectComplexFixture(a, dtype, `asfortranarray ${dtype}`);
      const r = np.asfortranarray(a);
      expect(r.dtype).toBe(dtype);
      expectComplexFixture(r, dtype, `asfortranarray ${dtype} result`);
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
          pyCode,
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

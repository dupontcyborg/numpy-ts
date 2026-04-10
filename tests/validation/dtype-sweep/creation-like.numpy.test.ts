/**
 * DType Sweep: Creation-like and utility functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  pyArrayCast,
  expectMatchPre,
  expectBothRejectPre,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
    const data2d =
      dtype === 'bool'
        ? [
            [1, 0],
            [1, 1],
          ]
        : [
            [1, 2],
            [3, 4],
          ];
    const fillVal = dtype === 'bool' ? 1 : 42;

    // _like functions
    snippets[`zeros_like_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.zeros_like(a)
result = _result_orig.astype(${ac})`;

    snippets[`ones_like_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.ones_like(a)
result = _result_orig.astype(${ac})`;

    snippets[`full_like_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.full_like(a, ${fillVal})
result = _result_orig.astype(${ac})`;

    snippets[`copy_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.copy(a)
result = _result_orig.astype(${ac})`;

    // Matrix creation
    snippets[`tri_${dtype}`] = `
_result_orig = np.tri(3, dtype=${npDtype(dtype)})
result = _result_orig.astype(${ac})`;

    snippets[`tril_${dtype}`] = `
a = np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})
_result_orig = np.tril(a)
result = _result_orig.astype(${ac})`;

    snippets[`triu_${dtype}`] = `
a = np.array(${JSON.stringify(data2d)}, dtype=${npDtype(dtype)})
_result_orig = np.triu(a)
result = _result_orig.astype(${ac})`;

    snippets[`meshgrid_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.meshgrid(a, a)[0]
result = _result_orig.astype(${ac})`;

    snippets[`vander_${dtype}`] = `
a = np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
_result_orig = np.vander(a, 3)
result = _result_orig.astype(${ac})`;

    // Conversion
    snippets[`asanyarray_${dtype}`] = `
_result_orig = np.asanyarray(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = _result_orig.astype(${ac})`;

    snippets[`asarray_chkfinite_${dtype}`] = `
_result_orig = np.asarray_chkfinite(${JSON.stringify(data)}, dtype=${npDtype(dtype)})
result = _result_orig.astype(${ac})`;

    // fromfunction
    snippets[`fromfunction_${dtype}`] = `
_result_orig = np.fromfunction(lambda i: i, (4,), dtype=${npDtype(dtype)})
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Creation-like', () => {
  describe('zeros_like', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const a = array(data, dtype);
        const jsResult = np.zeros_like(a);
        expectMatchPre(jsResult, oracle.get(`zeros_like_${dtype}`)!);
      });
    }
  });

  describe('ones_like', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const a = array(data, dtype);
        const jsResult = np.ones_like(a);
        expectMatchPre(jsResult, oracle.get(`ones_like_${dtype}`)!);
      });
    }
  });

  describe('full_like', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const fillVal = dtype === 'bool' ? 1 : 42;
        const a = array(data, dtype);
        const jsResult = np.full_like(a, fillVal);
        expectMatchPre(jsResult, oracle.get(`full_like_${dtype}`)!);
      });
    }
  });

  describe('copy', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const a = array(data, dtype);
        const jsResult = np.copy(a);
        expectMatchPre(jsResult, oracle.get(`copy_${dtype}`)!);
      });
    }
  });
});

describe('DType Sweep: Matrix creation', () => {
  describe('tri', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const pyResult = oracle.get(`tri_${dtype}`)!;
        const r = expectBothRejectPre(
          `tri may not support ${dtype}`,
          () => np.tri(3, undefined, 0, dtype),
          pyResult
        );
        if (r === 'both-reject') return;
        const jsResult = np.tri(3, undefined, 0, dtype);
        const isVeryLowPrecision = dtype === 'float16';
        const isLowPrecision = dtype === 'float32' || dtype === 'complex64';
        const rtol = isVeryLowPrecision ? 5e-2 : isLowPrecision ? 1e-2 : 1e-5;
        const atol = isVeryLowPrecision ? 1e-2 : isLowPrecision ? 1e-5 : 1e-8;
        expectMatchPre(jsResult, pyResult, { rtol, atol });
      });
    }
  });

  describe('tril', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data2d =
          dtype === 'bool'
            ? [
                [1, 0],
                [1, 1],
              ]
            : [
                [1, 2],
                [3, 4],
              ];
        const a = array(data2d, dtype);
        const jsResult = np.tril(a);
        expectMatchPre(jsResult, oracle.get(`tril_${dtype}`)!);
      });
    }
  });

  describe('triu', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data2d =
          dtype === 'bool'
            ? [
                [1, 0],
                [1, 1],
              ]
            : [
                [1, 2],
                [3, 4],
              ];
        const a = array(data2d, dtype);
        const jsResult = np.triu(a);
        expectMatchPre(jsResult, oracle.get(`triu_${dtype}`)!);
      });
    }
  });

  describe('meshgrid', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const a = array(data, dtype);
        const pyResult = oracle.get(`meshgrid_${dtype}`)!;
        const r = expectBothRejectPre(
          `meshgrid may not support ${dtype}`,
          () => np.meshgrid(a, a)[0],
          pyResult
        );
        if (r === 'both-reject') return;
        const jsResult = np.meshgrid(a, a)[0];
        expectMatchPre(jsResult, pyResult);
      });
    }
  });

  describe('vander', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const a = array(data, dtype);
        const pyResult = oracle.get(`vander_${dtype}`)!;
        const r = expectBothRejectPre(
          `vander may not support ${dtype}`,
          () => np.vander(a, 3),
          pyResult
        );
        if (r === 'both-reject') return;
        const jsResult = np.vander(a, 3);
        const isVeryLowPrecision = dtype === 'float16';
        const isLowPrecision = dtype === 'float32' || dtype === 'complex64';
        const rtol = isVeryLowPrecision ? 5e-2 : isLowPrecision ? 1e-2 : 1e-5;
        const atol = isVeryLowPrecision ? 1e-2 : isLowPrecision ? 1e-5 : 1e-8;
        expectMatchPre(jsResult, pyResult, { rtol, atol });
      });
    }
  });
});

describe('DType Sweep: Conversion', () => {
  describe('asanyarray', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const jsResult = np.asanyarray(data, dtype);
        expectMatchPre(jsResult, oracle.get(`asanyarray_${dtype}`)!);
      });
    }
  });

  describe('asarray_chkfinite', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = dtype === 'bool' ? [1, 0, 1, 1] : [1, 2, 3, 4];
        const jsResult = np.asarray_chkfinite(data, dtype);
        expectMatchPre(jsResult, oracle.get(`asarray_chkfinite_${dtype}`)!);
      });
    }
  });
});

describe('DType Sweep: fromfunction', () => {
  for (const dtype of ALL_DTYPES) {
    it(dtype, () => {
      const pyResult = oracle.get(`fromfunction_${dtype}`)!;
      const r = expectBothRejectPre(
        `fromfunction may not support ${dtype}`,
        () => np.fromfunction((i: number) => i, [4], dtype),
        pyResult
      );
      if (r === 'both-reject') return;
      const jsResult = np.fromfunction((i: number) => i, [4], dtype);
      const isVeryLowPrecision = dtype === 'float16';
      const isLowPrecision = dtype === 'float32' || dtype === 'complex64';
      const rtol = isVeryLowPrecision ? 5e-2 : isLowPrecision ? 1e-2 : 1e-5;
      const atol = isVeryLowPrecision ? 1e-2 : isLowPrecision ? 1e-5 : 1e-8;
      expectMatchPre(jsResult, pyResult, { rtol, atol });
    });
  }
});

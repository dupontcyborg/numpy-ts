/**
 * DType Sweep: Polynomial and misc math functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle -- all Python computations run in a single subprocess.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  pyArrayCast,
  isComplex,
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
    const nd = npDtype(dtype);

    // --- Coefficient data ---
    const isUnsigned = dtype.startsWith('uint');
    const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
    const coeffs2 = dtype === 'bool' ? [1, 1] : [1, 1];
    const rootsData = dtype === 'bool' ? [1, 0] : [1, 2];
    const polyvalX = [0, 1, 2, 3];
    const polyfitX = [0, 1, 2, 3];
    const polyfitY = [0, 1, 4, 9];
    const modfData =
      dtype === 'bool' ? [1, 0, 1, 0] : isUnsigned ? [1.5, 2.7, 0.3, 4.0] : [1.5, 2.7, -0.3, 4.0];
    const unwrapData = [0, 1, 2, 3, 4, 5];

    // poly(roots)
    snippets[`poly_${dtype}`] = `
a = np.array(${JSON.stringify(rootsData)}, dtype=${nd})
_result_orig = np.poly(a)
result = _result_orig.astype(${ac})`;

    // polyadd
    snippets[`polyadd_${dtype}`] = `
p1 = np.array(${JSON.stringify(coeffs)}, dtype=${nd})
p2 = np.array(${JSON.stringify(coeffs2)}, dtype=${nd})
_result_orig = np.polyadd(p1, p2)
result = _result_orig.astype(${ac})`;

    // polysub
    snippets[`polysub_${dtype}`] = `
p1 = np.array(${JSON.stringify(coeffs)}, dtype=${nd})
p2 = np.array(${JSON.stringify(coeffs2)}, dtype=${nd})
_result_orig = np.polysub(p1, p2)
result = _result_orig.astype(${ac})`;

    // polymul
    snippets[`polymul_${dtype}`] = `
p1 = np.array(${JSON.stringify(coeffs)}, dtype=${nd})
p2 = np.array(${JSON.stringify(coeffs2)}, dtype=${nd})
_result_orig = np.polymul(p1, p2)
result = _result_orig.astype(${ac})`;

    // polydiv — quotient only
    snippets[`polydiv_${dtype}`] = `
p1 = np.array(${JSON.stringify(coeffs)}, dtype=${nd})
p2 = np.array(${JSON.stringify(coeffs2)}, dtype=${nd})
_result_orig = np.polydiv(p1, p2)[0]
result = _result_orig.astype(${ac})`;

    // polyder
    snippets[`polyder_${dtype}`] = `
p = np.array(${JSON.stringify(coeffs)}, dtype=${nd})
_result_orig = np.polyder(p)
result = _result_orig.astype(${ac})`;

    // polyint
    snippets[`polyint_${dtype}`] = `
p = np.array(${JSON.stringify(coeffs)}, dtype=${nd})
_result_orig = np.polyint(p)
result = _result_orig.astype(${ac})`;

    // polyval
    snippets[`polyval_${dtype}`] = `
p = np.array(${JSON.stringify(coeffs)}, dtype=${nd})
x = np.array(${JSON.stringify(polyvalX)}, dtype=${nd})
_result_orig = np.polyval(p, x)
result = _result_orig.astype(${ac})`;

    // polyfit — may fail for bool/integer/complex
    snippets[`polyfit_${dtype}`] = `
x = np.array(${JSON.stringify(polyfitX)}, dtype=${nd})
y = np.array(${JSON.stringify(polyfitY)}, dtype=${nd})
_result_orig = np.polyfit(x, y, 2)
result = _result_orig.astype(np.float64)`;

    // roots — sort by magnitude for stable comparison, cast to float64
    snippets[`roots_${dtype}`] = `
p = np.array(${JSON.stringify(coeffs)}, dtype=${nd})
_result_orig = np.sort(np.abs(np.roots(p)))
result = _result_orig.astype(np.float64)`;

    // modf — fractional part, real-only
    snippets[`modf_${dtype}`] = `
a = np.array(${JSON.stringify(modfData)}, dtype=${nd})
_result_orig = np.modf(a)[0]
result = _result_orig.astype(np.float64)`;

    // unwrap — real-only
    snippets[`unwrap_${dtype}`] = `
a = np.array(${JSON.stringify(unwrapData)}, dtype=${nd})
_result_orig = np.unwrap(a)
result = _result_orig.astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

function getTol(dtype: string, loose = false) {
  const isVeryLowPrecision = dtype === 'float16';
  const isLowPrecision = dtype === 'float32' || dtype === 'complex64';
  if (loose) return { rtol: 1e-2, atol: 1e-2 };
  return {
    rtol: isVeryLowPrecision ? 5e-2 : isLowPrecision ? 1e-2 : 1e-3,
    atol: isVeryLowPrecision ? 1e-2 : isLowPrecision ? 1e-5 : 1e-8,
  };
}

describe('DType Sweep: Polynomial ops', () => {
  // --- poly ---
  describe('poly', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const rootsData = dtype === 'bool' ? [1, 0] : [1, 2];
        const a = array(rootsData, dtype);
        const key = `poly_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(`poly may not support ${dtype}`, () => np.poly(a), pyResult);
        if (r === 'both-reject') return;
        expectMatchPre(np.poly(a), pyResult, getTol(dtype));
      });
    }
  });

  // --- polyadd ---
  describe('polyadd', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
        const coeffs2 = dtype === 'bool' ? [1, 1] : [1, 1];
        const p1 = array(coeffs, dtype);
        const p2 = array(coeffs2, dtype);
        const key = `polyadd_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyadd may not support ${dtype}`,
          () => np.polyadd(p1, p2),
          pyResult
        );
        if (r === 'both-reject') return;
        expectMatchPre(np.polyadd(p1, p2), pyResult, getTol(dtype));
      });
    }
  });

  // --- polysub ---
  describe('polysub', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
        const coeffs2 = dtype === 'bool' ? [1, 1] : [1, 1];
        const p1 = array(coeffs, dtype);
        const p2 = array(coeffs2, dtype);
        const key = `polysub_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polysub may not support ${dtype}`,
          () => np.polysub(p1, p2),
          pyResult
        );
        if (r === 'both-reject') return;
        expectMatchPre(np.polysub(p1, p2), pyResult, getTol(dtype));
      });
    }
  });

  // --- polymul ---
  describe('polymul', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
        const coeffs2 = dtype === 'bool' ? [1, 1] : [1, 1];
        const p1 = array(coeffs, dtype);
        const p2 = array(coeffs2, dtype);
        const key = `polymul_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polymul may not support ${dtype}`,
          () => np.polymul(p1, p2),
          pyResult
        );
        if (r === 'both-reject') return;
        expectMatchPre(np.polymul(p1, p2), pyResult, getTol(dtype));
      });
    }
  });

  // --- polydiv (quotient) ---
  describe('polydiv', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
        const coeffs2 = dtype === 'bool' ? [1, 1] : [1, 1];
        const p1 = array(coeffs, dtype);
        const p2 = array(coeffs2, dtype);
        const key = `polydiv_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polydiv may not support ${dtype}`,
          () => np.polydiv(p1, p2),
          pyResult
        );
        if (r === 'both-reject') return;
        const [q] = np.polydiv(p1, p2);
        expectMatchPre(q, pyResult, getTol(dtype));
      });
    }
  });

  // --- polyder ---
  describe('polyder', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
        const p = array(coeffs, dtype);
        const key = `polyder_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyder may not support ${dtype}`,
          () => np.polyder(p),
          pyResult
        );
        if (r === 'both-reject') return;
        expectMatchPre(np.polyder(p), pyResult, getTol(dtype));
      });
    }
  });

  // --- polyint ---
  describe('polyint', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
        const p = array(coeffs, dtype);
        const key = `polyint_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyint may not support ${dtype}`,
          () => np.polyint(p),
          pyResult
        );
        if (r === 'both-reject') return;
        expectMatchPre(np.polyint(p), pyResult, getTol(dtype));
      });
    }
  });

  // --- polyval ---
  describe('polyval', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
        const xData = [0, 1, 2, 3];
        const p = array(coeffs, dtype);
        const x = array(xData, dtype);
        const key = `polyval_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyval may not support ${dtype}`,
          () => np.polyval(p, x),
          pyResult
        );
        if (r === 'both-reject') return;
        const jsResult = np.polyval(p, x);
        expectMatchPre(jsResult as any, pyResult, getTol(dtype));
      });
    }
  });

  // --- polyfit ---
  describe('polyfit', () => {
    for (const dtype of ALL_DTYPES) {
      // Bool: x=[0,1,1,1] has duplicate values making the system singular — skip
      if (dtype === 'bool') continue;
      it(dtype, () => {
        const xData = [0, 1, 2, 3];
        const yData = [0, 1, 4, 9];
        const x = array(xData, dtype);
        const y = array(yData, dtype);
        const key = `polyfit_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyfit may not support ${dtype}`,
          () => np.polyfit(x, y, 2),
          pyResult
        );
        if (r === 'both-reject') return;
        expectMatchPre(np.polyfit(x, y, 2), pyResult, { rtol: 1e-2, atol: 1e-2 });
      });
    }
  });

  // --- roots ---
  describe('roots', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const coeffs = dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2];
        const p = array(coeffs, dtype);
        const key = `roots_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `roots may not support ${dtype}`,
          () => np.roots(p),
          pyResult
        );
        if (r === 'both-reject') return;
        // Sort by magnitude for stable comparison
        const jsRoots = np.roots(p);
        const jsSorted = np.sort(np.abs(jsRoots));
        expectMatchPre(jsSorted, pyResult, { rtol: 1e-2, atol: 1e-2 });
      });
    }
  });
});

describe('DType Sweep: Misc math', () => {
  // --- modf (fractional part) ---
  describe('modf', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const isUnsigned = dtype.startsWith('uint');
        const data =
          dtype === 'bool'
            ? [1, 0, 1, 0]
            : isUnsigned
              ? [1.5, 2.7, 0.3, 4.0]
              : [1.5, 2.7, -0.3, 4.0];
        const a = array(data, dtype);
        const key = `modf_${dtype}`;
        const pyResult = oracle.get(key)!;

        // modf rejects complex
        if (isComplex(dtype)) {
          const r = expectBothRejectPre(
            'modf is not defined for complex numbers',
            () => np.modf(a),
            pyResult
          );
          if (r === 'both-reject') return;
        }

        const r = expectBothRejectPre(`modf may not support ${dtype}`, () => np.modf(a), pyResult);
        if (r === 'both-reject') return;

        const [frac] = np.modf(a);
        expectMatchPre(frac, pyResult, getTol(dtype));
      });
    }
  });

  // --- unwrap ---
  describe('unwrap', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const data = [0, 1, 2, 3, 4, 5];
        const a = array(data, dtype);
        const key = `unwrap_${dtype}`;
        const pyResult = oracle.get(key)!;

        // unwrap rejects complex
        if (isComplex(dtype)) {
          const r = expectBothRejectPre(
            'unwrap is not defined for complex numbers',
            () => np.unwrap(a),
            pyResult
          );
          if (r === 'both-reject') return;
        }

        const r = expectBothRejectPre(
          `unwrap may not support ${dtype}`,
          () => np.unwrap(a),
          pyResult
        );
        if (r === 'both-reject') return;

        expectMatchPre(np.unwrap(a), pyResult, getTol(dtype));
      });
    }
  });
});

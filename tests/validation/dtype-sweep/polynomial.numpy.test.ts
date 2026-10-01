/**
 * DType Sweep: Polynomial and misc math functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle -- all Python computations run in a single subprocess.
 */
import { beforeAll, describe, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectBothRejectPre,
  expectComplexFixture,
  expectMatchPre,
  isComplex,
  npDtype,
  pyArrayCast,
  runNumPyBatch,
} from './_helpers';

const { array } = np;

/**
 * The one definition of every input for a dtype, shared by the oracle snippets and
 * the test bodies. Two independent copies would let the two sides drift apart and
 * compare different inputs.
 */
function polyFixtures(dtype: string) {
  const isUnsigned = dtype.startsWith('uint');
  return {
    coeffs: asDtypeData(dtype === 'bool' ? [1, 0, 1] : isUnsigned ? [1, 3, 2] : [1, -3, 2], dtype),
    coeffs2: asDtypeData([1, 1], dtype),
    rootsData: asDtypeData(dtype === 'bool' ? [1, 0] : [1, 2], dtype),
    polyvalX: asDtypeData([0, 1, 2, 3], dtype),
    polyfitX: asDtypeData([0, 1, 2, 3], dtype),
    polyfitY: asDtypeData([0, 1, 4, 9], dtype),
    modfData: asDtypeData(
      dtype === 'bool' ? [1, 0, 1, 0] : isUnsigned ? [1.5, 2.7, 0.3, 4.0] : [1.5, 2.7, -0.3, 4.0],
      dtype,
    ),
    unwrapData: asDtypeData([0, 1, 2, 3, 4, 5], dtype),
  };
}

let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const ac = pyArrayCast(dtype);
    const nd = npDtype(dtype);

    // --- Coefficient data ---
    const { coeffs, coeffs2, rootsData, polyvalX, polyfitX, polyfitY, modfData, unwrapData } =
      polyFixtures(dtype);

    // poly(roots)
    snippets[`poly_${dtype}`] = `
a = np.array(${rootsData.py}, dtype=${nd})
_result_orig = np.poly(a)
result = _result_orig.astype(${ac})`;

    // polyadd
    snippets[`polyadd_${dtype}`] = `
p1 = np.array(${coeffs.py}, dtype=${nd})
p2 = np.array(${coeffs2.py}, dtype=${nd})
_result_orig = np.polyadd(p1, p2)
result = _result_orig.astype(${ac})`;

    // polysub
    snippets[`polysub_${dtype}`] = `
p1 = np.array(${coeffs.py}, dtype=${nd})
p2 = np.array(${coeffs2.py}, dtype=${nd})
_result_orig = np.polysub(p1, p2)
result = _result_orig.astype(${ac})`;

    // polymul
    snippets[`polymul_${dtype}`] = `
p1 = np.array(${coeffs.py}, dtype=${nd})
p2 = np.array(${coeffs2.py}, dtype=${nd})
_result_orig = np.polymul(p1, p2)
result = _result_orig.astype(${ac})`;

    // polydiv — quotient only
    snippets[`polydiv_${dtype}`] = `
p1 = np.array(${coeffs.py}, dtype=${nd})
p2 = np.array(${coeffs2.py}, dtype=${nd})
_result_orig = np.polydiv(p1, p2)[0]
result = _result_orig.astype(${ac})`;

    // polyder
    snippets[`polyder_${dtype}`] = `
p = np.array(${coeffs.py}, dtype=${nd})
_result_orig = np.polyder(p)
result = _result_orig.astype(${ac})`;

    // polyint
    snippets[`polyint_${dtype}`] = `
p = np.array(${coeffs.py}, dtype=${nd})
_result_orig = np.polyint(p)
result = _result_orig.astype(${ac})`;

    // polyval
    snippets[`polyval_${dtype}`] = `
p = np.array(${coeffs.py}, dtype=${nd})
x = np.array(${polyvalX.py}, dtype=${nd})
_result_orig = np.polyval(p, x)
result = _result_orig.astype(${ac})`;

    // polyfit — may fail for bool/integer/complex
    snippets[`polyfit_${dtype}`] = `
x = np.array(${polyfitX.py}, dtype=${nd})
y = np.array(${polyfitY.py}, dtype=${nd})
_result_orig = np.polyfit(x, y, 2)
result = _result_orig.astype(np.float64)`;

    // roots — sort by magnitude for stable comparison, cast to float64
    snippets[`roots_${dtype}`] = `
p = np.array(${coeffs.py}, dtype=${nd})
_result_orig = np.sort(np.abs(np.roots(p)))
result = _result_orig.astype(np.float64)`;

    // modf — fractional part, real-only
    snippets[`modf_${dtype}`] = `
a = np.array(${modfData.py}, dtype=${nd})
_result_orig = np.modf(a)[0]
result = _result_orig.astype(np.float64)`;

    // unwrap — real-only
    snippets[`unwrap_${dtype}`] = `
a = np.array(${unwrapData.py}, dtype=${nd})
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
        const { rootsData } = polyFixtures(dtype);
        const a = array(rootsData.js as never, dtype);
        expectComplexFixture(a, dtype, `poly ${dtype}`);
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
        const { coeffs, coeffs2 } = polyFixtures(dtype);
        const p1 = array(coeffs.js as never, dtype);
        const p2 = array(coeffs2.js as never, dtype);
        expectComplexFixture(p1, dtype, `polyadd ${dtype}`);
        const key = `polyadd_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyadd may not support ${dtype}`,
          () => np.polyadd(p1, p2),
          pyResult,
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
        const { coeffs, coeffs2 } = polyFixtures(dtype);
        const p1 = array(coeffs.js as never, dtype);
        const p2 = array(coeffs2.js as never, dtype);
        expectComplexFixture(p1, dtype, `polysub ${dtype}`);
        const key = `polysub_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polysub may not support ${dtype}`,
          () => np.polysub(p1, p2),
          pyResult,
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
        const { coeffs, coeffs2 } = polyFixtures(dtype);
        const p1 = array(coeffs.js as never, dtype);
        const p2 = array(coeffs2.js as never, dtype);
        expectComplexFixture(p1, dtype, `polymul ${dtype}`);
        const key = `polymul_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polymul may not support ${dtype}`,
          () => np.polymul(p1, p2),
          pyResult,
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
        const { coeffs, coeffs2 } = polyFixtures(dtype);
        const p1 = array(coeffs.js as never, dtype);
        const p2 = array(coeffs2.js as never, dtype);
        expectComplexFixture(p1, dtype, `polydiv ${dtype}`);
        const key = `polydiv_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polydiv may not support ${dtype}`,
          () => np.polydiv(p1, p2),
          pyResult,
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
        const { coeffs } = polyFixtures(dtype);
        const p = array(coeffs.js as never, dtype);
        expectComplexFixture(p, dtype, `polyder ${dtype}`);
        const key = `polyder_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyder may not support ${dtype}`,
          () => np.polyder(p),
          pyResult,
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
        const { coeffs } = polyFixtures(dtype);
        const p = array(coeffs.js as never, dtype);
        expectComplexFixture(p, dtype, `polyint ${dtype}`);
        const key = `polyint_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyint may not support ${dtype}`,
          () => np.polyint(p),
          pyResult,
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
        const { coeffs, polyvalX } = polyFixtures(dtype);
        const p = array(coeffs.js as never, dtype);
        const x = array(polyvalX.js as never, dtype);
        expectComplexFixture(p, dtype, `polyval ${dtype}`);
        const key = `polyval_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyval may not support ${dtype}`,
          () => np.polyval(p, x),
          pyResult,
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
        const { polyfitX, polyfitY } = polyFixtures(dtype);
        const x = array(polyfitX.js as never, dtype);
        const y = array(polyfitY.js as never, dtype);
        expectComplexFixture(y, dtype, `polyfit ${dtype}`);
        const key = `polyfit_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `polyfit may not support ${dtype}`,
          () => np.polyfit(x, y, 2),
          pyResult,
        );
        if (r === 'both-reject') return;
        expectMatchPre(np.polyfit(x, y, 2), pyResult, {
          rtol: 1e-2,
          atol: 1e-2,
        });
      });
    }
  });

  // --- roots ---
  describe('roots', () => {
    for (const dtype of ALL_DTYPES) {
      it(dtype, () => {
        const { coeffs } = polyFixtures(dtype);
        const p = array(coeffs.js as never, dtype);
        expectComplexFixture(p, dtype, `roots ${dtype}`);
        const key = `roots_${dtype}`;
        const pyResult = oracle.get(key)!;
        const r = expectBothRejectPre(
          `roots may not support ${dtype}`,
          () => np.roots(p),
          pyResult,
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
        const { modfData } = polyFixtures(dtype);
        const a = array(modfData.js as never, dtype);
        expectComplexFixture(a, dtype, `modf ${dtype}`);
        const key = `modf_${dtype}`;
        const pyResult = oracle.get(key)!;

        // modf rejects complex
        if (isComplex(dtype)) {
          const r = expectBothRejectPre(
            'modf is not defined for complex numbers',
            () => np.modf(a),
            pyResult,
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
        const { unwrapData } = polyFixtures(dtype);
        const a = array(unwrapData.js as never, dtype);
        expectComplexFixture(a, dtype, `unwrap ${dtype}`);
        const key = `unwrap_${dtype}`;
        const pyResult = oracle.get(key)!;

        // unwrap rejects complex
        if (isComplex(dtype)) {
          const r = expectBothRejectPre(
            'unwrap is not defined for complex numbers',
            () => np.unwrap(a),
            pyResult,
          );
          if (r === 'both-reject') return;
        }

        const r = expectBothRejectPre(
          `unwrap may not support ${dtype}`,
          () => np.unwrap(a),
          pyResult,
        );
        if (r === 'both-reject') return;

        expectMatchPre(np.unwrap(a), pyResult, getTol(dtype));
      });
    }
  });
});

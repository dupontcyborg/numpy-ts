/**
 * DType Sweep: Polynomial and misc math functions.
 * Tests each function across ALL dtypes, validated against NumPy.
 * Uses batched oracle -- all Python computations run in a single subprocess.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  asDtypeData,
  checkNumPyAvailable,
  expectBothRejectPre,
  expectComplexFixture,
  expectMatch,
  expectMatchPre,
  isComplex,
  npDtype,
  pyArrayCast,
  runNumPy,
  runNumPyBatch,
} from './_helpers';

const { array, Complex } = np;

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

    // polyfit — may fail for bool/integer. polyfit returns complex128 for
    // complex input, so casting the oracle to float64 there would throw away
    // NumPy's own imaginary parts and compare against half an answer.
    snippets[`polyfit_${dtype}`] = `
x = np.array(${polyfitX.py}, dtype=${nd})
y = np.array(${polyfitY.py}, dtype=${nd})
_result_orig = np.polyfit(x, y, 2)
result = _result_orig.astype(${isComplex(dtype) ? 'np.complex128' : 'np.float64'})`;

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

// Views, mixed operand dtypes and conjugate-pair roots: none of these is a
// per-dtype property, so they sit outside the sweep above.
describe('Polynomial: views and dtype promotion', () => {
  const cplx = () => array([new Complex(9, 9), new Complex(1, 1), new Complex(2, 0)]);
  const tol = { rtol: 1e-9, atol: 1e-12 };

  it('polyval reads a sliced coefficient view from its offset', () => {
    expectMatch(
      np.polyval(cplx().slice('1:'), array([1])),
      `p = np.array([9+9j, 1+1j, 2])
_result_orig = result = np.polyval(p[1:], np.array([1.0]))`,
      tol,
    );
  });

  it('polyval reads a strided coefficient view through its strides', () => {
    const p = array([new Complex(1, 2), new Complex(5, 5), new Complex(3, 0), new Complex(7, 1)]);
    expectMatch(
      np.polyval(p.slice('::2'), array([2, -1])),
      `p = np.array([1+2j, 5+5j, 3, 7+1j])
_result_orig = result = np.polyval(p[::2], np.array([2.0, -1.0]))`,
      tol,
    );
  });

  it('polyval reads a strided x view', () => {
    const x = array([1, 100, 2, 100, 3]);
    expectMatch(
      np.polyval(array([1, -3, 2]), x.slice('::2')),
      `_result_orig = result = np.polyval(np.array([1.0, -3, 2]), np.array([1.0, 100, 2, 100, 3])[::2])`,
      tol,
    );
  });

  it('roots of a reversed real view are the roots of the reversed polynomial', () => {
    const r = np.roots(np.flip(array([1, -3, 2, 0])));
    expectMatch(
      np.sort(np.abs(r)),
      `_result_orig = result = np.sort(np.abs(np.roots(np.array([1.0, -3, 2, 0])[::-1])))`,
      tol,
    );
  });

  it('roots of a reversed complex view are the roots of the reversed polynomial', () => {
    const c = array([new Complex(1, 0), new Complex(-3, 1), new Complex(2, -2), new Complex(4, 0)]);
    expectMatch(
      np.sort(np.abs(np.roots(np.flip(c)))),
      `c = np.array([1, -3+1j, 2-2j, 4])
_result_orig = result = np.sort(np.abs(np.roots(c[::-1])))`,
      { rtol: 1e-7, atol: 1e-9 },
    );
  });

  it('polydiv promotes a real dividend with a complex divisor', () => {
    const [q, r] = np.polydiv(array([1, 0, 1]), array([new Complex(1, 0), new Complex(0, 1)]));
    expectMatch(
      q,
      `_result_orig = result = np.polydiv(np.array([1.0, 0, 1]), np.array([1, 1j]))[0]`,
      tol,
    );
    expectMatch(
      r,
      `_result_orig = result = np.polydiv(np.array([1.0, 0, 1]), np.array([1, 1j]))[1]`,
      tol,
    );
  });

  it('polydiv takes the wider of the two operand dtypes', () => {
    const u = array([3, 5, 2], 'float32');
    const v = array([new Complex(2, 0), new Complex(1, 0)], 'complex64');
    expectMatch(
      np.polydiv(u, v)[0],
      `_result_orig = result = np.polydiv(np.array([3, 5, 2], dtype=np.float32), np.array([2, 1], dtype=np.complex64))[0]`,
      { rtol: 1e-5, atol: 1e-6 },
    );
    expectMatch(
      np.polydiv(array([3, 5, 2], 'int32'), array([2, 1], 'float32'))[0],
      `_result_orig = result = np.polydiv(np.array([3, 5, 2], dtype=np.int32), np.array([2, 1], dtype=np.float32))[0]`,
      tol,
    );
  });

  it('polyval promotes a real polynomial at complex points', () => {
    expectMatch(
      np.polyval(array([1, 0]), array([new Complex(0, 1)])),
      `_result_orig = result = np.polyval(np.array([1.0, 0]), np.array([1j]))`,
      tol,
    );
  });

  it('polyval widens complex64 coefficients at float64 points', () => {
    const p = array([new Complex(1, 1), new Complex(2, 0)], 'complex64');
    expectMatch(
      np.polyval(p, array([1.5])),
      `_result_orig = result = np.polyval(np.array([1+1j, 2], dtype=np.complex64), np.array([1.5]))`,
      { rtol: 1e-6, atol: 1e-6 },
    );
  });

  it('polyval of integer coefficients at float points is float', () => {
    expectMatch(
      np.polyval(array([1, 2], 'int32'), array([1.5])),
      `_result_orig = result = np.polyval(np.array([1, 2], dtype=np.int32), np.array([1.5]))`,
      tol,
    );
  });

  it('polyval at a scalar point returns a scalar of the promoted kind', () => {
    const py = runNumPy(`result = np.polyval(np.array([1, 2], dtype=np.int32), 1.5)`);
    expect(py.dtype).toBe('float64');
    expect(np.polyval(array([1, 2], 'int32'), 1.5)).toBeCloseTo(Number(py.value), 12);

    const pc = runNumPy(`result = np.polyval(np.array([1+1j, 2], dtype=np.complex64), 2.0)`);
    expect(pc.dtype).toBe('complex128');
    const js = np.polyval(array([new Complex(1, 1), new Complex(2, 0)], 'complex64'), 2);
    expect(js).toBeInstanceOf(Complex);
    const v = pc.value as { re: number; im: number };
    expect((js as InstanceType<typeof Complex>).re).toBeCloseTo(v.re, 6);
    expect((js as InstanceType<typeof Complex>).im).toBeCloseTo(v.im, 6);
  });

  it('poly of conjugate-pair roots is real', () => {
    expectMatch(
      np.poly(array([new Complex(0, 1), new Complex(0, -1)])),
      `_result_orig = result = np.poly(np.array([1j, -1j]))`,
      tol,
    );
    expectMatch(
      np.poly(array([new Complex(0, 1), new Complex(0, -1)], 'complex64')),
      `_result_orig = result = np.poly(np.array([1j, -1j], dtype=np.complex64))`,
      tol,
    );
    expectMatch(
      np.poly(array([new Complex(1, 1), new Complex(2, 0), new Complex(1, -1)])),
      `_result_orig = result = np.poly(np.array([1+1j, 2, 1-1j]))`,
      tol,
    );
  });

  it('poly of roots that are not closed under conjugation stays complex', () => {
    expectMatch(
      np.poly(array([new Complex(0, 1), new Complex(1, -1)])),
      `_result_orig = result = np.poly(np.array([1j, 1-1j]))`,
      tol,
    );
  });
});

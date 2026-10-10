/**
 * DType Sweep: Statistics, misc, and complex functions.
 * All tested across ALL dtypes, value-producing tests validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { beforeAll, describe, expect, it } from 'vitest';
import * as np from '../../../src';
import type { NumPyResult } from '../numpy-oracle';
import {
  ALL_DTYPES,
  arraysClose,
  asDtypeData,
  checkNumPyAvailable,
  expectBothReject,
  expectComplexFixture,
  expectMatchPre,
  isComplex,
  npDtype,
  pyArrayCast,
  pyScalarCast,
  runNumPyBatch,
  scalarClose,
} from './_helpers';

const { array } = np;

// One definition per fixture, shared by the oracle snippets and the test bodies —
// two copies would let the JS and Python sides drift apart silently.
const histData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5], dtype);
const corrData = (dtype: string) =>
  asDtypeData(
    dtype === 'bool'
      ? [
          [1, 0, 1],
          [0, 1, 0],
        ]
      : [
          [1, 2, 3],
          [4, 5, 6],
        ],
    dtype,
  );
const bincountData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1, 0] : [0, 1, 1, 2, 3, 3, 3], dtype);
const digitizeData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [0, 1, 0, 1] : [1, 3, 5, 7], dtype);
const digitizeBins = (dtype: string) => asDtypeData(dtype === 'bool' ? [0, 1] : [2, 4, 6], dtype);
const corrD1 = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
const corrD2 = (dtype: string) => asDtypeData([0, 1], dtype);
const trapData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4], dtype);
const diffData = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 4, 7, 11], dtype);
const interpXp = (dtype: string) => asDtypeData(dtype === 'bool' ? [0, 1] : [1, 2, 3, 4], dtype);
const interpFp = (dtype: string) =>
  asDtypeData(dtype === 'bool' ? [0, 1] : [10, 20, 30, 40], dtype);
const interpX = (dtype: string) => asDtypeData(dtype === 'bool' ? [0, 1] : [1.5, 2.5, 3.5], dtype);
const clipData = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1] : [1, 5, 10], dtype);
const clipLo = (dtype: string) => (dtype === 'bool' ? 0 : 2);
const clipHi = (dtype: string) => (dtype === 'bool' ? 1 : 8);
const fmodD1 = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 1, 0] : [6, 7, 8, 9], dtype);
const fmodD2 = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 1, 1] : [4, 3, 5, 2], dtype);
const smallData = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1] : [1, 2, 3], dtype);
const frexpData = (dtype: string) => asDtypeData(dtype === 'bool' ? [1, 0, 1] : [1, 2, 4], dtype);

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

const clipPy = (dtype: string, ac: string) =>
  `_result_orig = np.clip(np.array(${clipData(dtype).py}, dtype=${npDtype(dtype)}), ${clipLo(dtype)}, ${clipHi(dtype)})
result = _result_orig.astype(${ac})`;

const fmodPy = (dtype: string, ac: string) =>
  `_result_orig = np.fmod(np.array(${fmodD1(dtype).py}, dtype=${npDtype(dtype)}), np.array(${fmodD2(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

const ldexpPy = (dtype: string, ac: string) =>
  `_result_orig = np.ldexp(np.array(${smallData(dtype).py}, dtype=${npDtype(dtype)}), np.array([1,2,3], dtype=np.int32))
result = _result_orig.astype(${ac})`;

const frexpPy = (dtype: string, ac: string) => `
m, e = np.frexp(np.array(${frexpData(dtype).py}, dtype=${npDtype(dtype)}))
_result_orig = m
result = _result_orig.astype(${ac})`;

const histPy = (dtype: string, ac: string) => `
h, e = np.histogram(np.array(${histData(dtype).py}, dtype=${npDtype(dtype)}))
_result_orig = h
result = _result_orig.astype(${ac})`;

const digitizePy = (dtype: string) =>
  `_result_orig = np.digitize(np.array(${digitizeData(dtype).py}, dtype=${npDtype(dtype)}), np.array(${digitizeBins(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig`;

const trapezoidPy = (dtype: string, sc: string) =>
  `result = ${sc}(np.trapezoid(np.array(${trapData(dtype).py}, dtype=${npDtype(dtype)})))`;

const gradientPy = (dtype: string, ac: string) =>
  `_result_orig = np.gradient(np.array(${diffData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

const interpPy = (dtype: string) => `
x = np.array(${interpX(dtype).py}, dtype=${npDtype(dtype)})
xp = np.array(${interpXp(dtype).py}, dtype=${npDtype(dtype)})
fp = np.array(${interpFp(dtype).py}, dtype=${npDtype(dtype)})
_result_orig = np.interp(x, xp, fp)
result = _result_orig`;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    const sc = pyScalarCast(dtype);
    const ac = pyArrayCast(dtype);

    // Statistics
    snippets[`stats_histogram_${dtype}`] = histPy(dtype, ac);

    snippets[`stats_corrcoef_${dtype}`] =
      `_result_orig = np.corrcoef(np.array(${corrData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    snippets[`stats_cov_${dtype}`] =
      `_result_orig = np.cov(np.array(${corrData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    snippets[`stats_bincount_${dtype}`] =
      `result = np.bincount(np.array(${bincountData(dtype).py}, dtype=${npDtype(dtype)})).astype(${ac})`;

    snippets[`stats_digitize_${dtype}`] = digitizePy(dtype);

    snippets[`stats_correlate_${dtype}`] =
      `_result_orig = np.correlate(np.array(${corrD1(dtype).py}, dtype=${npDtype(dtype)}), np.array(${corrD2(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`stats_convolve_${dtype}`] =
      `_result_orig = np.convolve(np.array(${corrD1(dtype).py}, dtype=${npDtype(dtype)}), np.array(${corrD2(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`stats_trapezoid_${dtype}`] = trapezoidPy(dtype, sc);

    snippets[`stats_diff_${dtype}`] =
      `_result_orig = np.diff(np.array(${diffData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`stats_gradient_${dtype}`] = gradientPy(dtype, ac);

    snippets[`stats_interp_${dtype}`] = interpPy(dtype);

    // Misc
    snippets[`misc_clip_${dtype}`] = clipPy(dtype, ac);

    snippets[`misc_fmod_${dtype}`] = fmodPy(dtype, ac);

    snippets[`misc_nan_to_num_${dtype}`] =
      `_result_orig = np.nan_to_num(np.array(${smallData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`misc_ldexp_${dtype}`] = ldexpPy(dtype, ac);

    snippets[`misc_frexp_${dtype}`] = frexpPy(dtype, ac);

    if (!isComplex(dtype)) {
      snippets[`misc_array_equiv_${dtype}`] =
        `result = bool(np.array_equiv(np.array(${smallData(dtype).py}, dtype=${npDtype(dtype)}), np.array(${smallData(dtype).py}, dtype=${npDtype(dtype)})))`;
    }

    // Complex functions
    snippets[`complex_real_${dtype}`] =
      `_result_orig = np.real(np.array(${smallData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`complex_imag_${dtype}`] =
      `_result_orig = np.imag(np.array(${smallData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`complex_conj_${dtype}`] =
      `_result_orig = np.conj(np.array(${smallData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`complex_angle_${dtype}`] =
      `_result_orig = np.angle(np.array(${smallData(dtype).py}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Statistics', () => {
  for (const dtype of ALL_DTYPES) {
    it(`histogram ${dtype}`, () => {
      const a = array(histData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `histogram ${dtype}`);
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'histogram requires real-valued input',
          () => np.histogram(a),
          histPy(dtype, pyArrayCast(dtype)),
        );
        if (_r === 'both-reject') return;
      }
      const [hist] = np.histogram(a) as [any, any];
      expectMatchPre(hist, oracle.get(`stats_histogram_${dtype}`)!, { indexResult: true });
    });

    it(`corrcoef ${dtype}`, () => {
      const a = array(corrData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `corrcoef ${dtype}`);
      const jsResult = np.corrcoef(a);
      expectMatchPre(jsResult, oracle.get(`stats_corrcoef_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`cov ${dtype}`, () => {
      const a = array(corrData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `cov ${dtype}`);
      const jsResult = np.cov(a);
      expectMatchPre(jsResult, oracle.get(`stats_cov_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`bincount ${dtype}`, () => {
      const a = array(bincountData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `bincount ${dtype}`);
      let pyErr = false;
      let jsErr = false;
      const py = oracle.get(`stats_bincount_${dtype}`)!;
      if (py.error) pyErr = true;
      let jsResult: any;
      try {
        jsResult = np.bincount(a);
      } catch {
        jsErr = true;
      }
      if (pyErr && jsErr) return;
      if (pyErr && !jsErr) {
        expect(jsResult.shape[0]).toBeGreaterThan(0);
        return;
      }
      if (!pyErr && jsErr) throw new Error(`JS errors but NumPy succeeds for bincount(${dtype})`);
      expect(arraysClose(jsResult.toArray(), py.value)).toBe(true);
    });

    it(`digitize ${dtype}`, () => {
      const a = array(digitizeData(dtype).js as never, dtype);
      const bins = array(digitizeBins(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `digitize ${dtype}`);
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'digitize requires real-valued input',
          () => np.digitize(a, bins),
          digitizePy(dtype),
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.digitize(a, bins);
      expectMatchPre(jsResult, oracle.get(`stats_digitize_${dtype}`)!, { indexResult: true });
    });

    it(`correlate ${dtype}`, () => {
      const a = array(corrD1(dtype).js as never, dtype);
      const v = array(corrD2(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `correlate ${dtype}`);
      const jsResult = np.correlate(a, v);
      expectMatchPre(jsResult, oracle.get(`stats_correlate_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`convolve ${dtype}`, () => {
      const a = array(corrD1(dtype).js as never, dtype);
      const v = array(corrD2(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `convolve ${dtype}`);
      const jsResult = np.convolve(a, v);
      expectMatchPre(jsResult, oracle.get(`stats_convolve_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`trapezoid ${dtype}`, () => {
      const a = array(trapData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `trapezoid ${dtype}`);
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'trapezoid is not defined for complex numbers',
          () => np.trapezoid(a),
          trapezoidPy(dtype, pyScalarCast(dtype)),
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.trapezoid(a);
      const py = oracle.get(`stats_trapezoid_${dtype}`)!;
      scalarClose(jsResult, py.value);
    });

    it(`diff ${dtype}`, () => {
      const a = array(diffData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `diff ${dtype}`);
      const jsResult = np.diff(a);
      expectMatchPre(jsResult, oracle.get(`stats_diff_${dtype}`)!);
    });

    it(`gradient ${dtype}`, () => {
      const a = array(diffData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `gradient ${dtype}`);
      if (dtype === 'bool') {
        const _r = expectBothReject(
          'gradient uses subtract internally, not supported for bool',
          () => np.gradient(a),
          gradientPy(dtype, pyArrayCast(dtype)),
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.gradient(a);
      expectMatchPre(jsResult, oracle.get(`stats_gradient_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`interp ${dtype}`, () => {
      const x = array(interpX(dtype).js as never, dtype);
      const xp = array(interpXp(dtype).js as never, dtype);
      const fp = array(interpFp(dtype).js as never, dtype);
      expectComplexFixture(x, dtype, `interp ${dtype}`);
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'interp is not defined for complex numbers',
          () => np.interp(x, xp, fp),
          interpPy(dtype),
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.interp(x, xp, fp);
      expectMatchPre(jsResult, oracle.get(`stats_interp_${dtype}`)!, { rtol: 1e-4 });
    });
  }
});

describe('DType Sweep: Misc', () => {
  for (const dtype of ALL_DTYPES) {
    it(`clip ${dtype}`, () => {
      const a = array(clipData(dtype).js as never, dtype);
      const lo = clipLo(dtype);
      const hi = clipHi(dtype);
      expectComplexFixture(a, dtype, `clip ${dtype}`);
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'clip is not defined for complex numbers (requires ordering)',
          () => np.clip(a, lo, hi),
          clipPy(dtype, pyArrayCast(dtype)),
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.clip(a, lo, hi);
      expectMatchPre(jsResult, oracle.get(`misc_clip_${dtype}`)!);
    });

    it(`fmod ${dtype}`, () => {
      const a = array(fmodD1(dtype).js as never, dtype);
      const b = array(fmodD2(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `fmod ${dtype}`);
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'fmod is not defined for complex numbers',
          () => np.fmod(a, b),
          fmodPy(dtype, pyArrayCast(dtype)),
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.fmod(a, b);
      expectMatchPre(jsResult, oracle.get(`misc_fmod_${dtype}`)!);
    });

    it(`nan_to_num ${dtype}`, () => {
      const a = array(smallData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `nan_to_num ${dtype}`);
      const jsResult = np.nan_to_num(a);
      expectMatchPre(jsResult, oracle.get(`misc_nan_to_num_${dtype}`)!);
    });

    it(`ldexp ${dtype}`, () => {
      const a = array(smallData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `ldexp ${dtype}`);
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'ldexp is only defined for real floating-point types',
          () => np.ldexp(a, array([1, 2, 3], 'int32')),
          ldexpPy(dtype, pyArrayCast(dtype)),
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.ldexp(a, array([1, 2, 3], 'int32'));
      expectMatchPre(jsResult, oracle.get(`misc_ldexp_${dtype}`)!);
    });

    it(`frexp ${dtype}`, () => {
      const a = array(frexpData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `frexp ${dtype}`);
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'frexp is only defined for real floating-point types',
          () => np.frexp(a),
          frexpPy(dtype, pyArrayCast(dtype)),
        );
        if (_r === 'both-reject') return;
      }
      const [m] = np.frexp(a) as [any, any];
      expectMatchPre(m, oracle.get(`misc_frexp_${dtype}`)!, { rtol: 1e-4 });
    });

    if (!isComplex(dtype)) {
      it(`array_equiv ${dtype}`, () => {
        const a = array(smallData(dtype).js as never, dtype);
        const b = array(smallData(dtype).js as never, dtype);
        const jsResult = np.array_equiv(a, b);
        const py = oracle.get(`misc_array_equiv_${dtype}`)!;
        expect(Boolean(jsResult)).toBe(Boolean(py.value));
      });
    }
  }
});

describe('DType Sweep: Complex functions', () => {
  for (const dtype of ALL_DTYPES) {
    it(`real ${dtype}`, () => {
      const a = array(smallData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `real ${dtype}`);
      const jsResult = np.real(a);
      expectMatchPre(jsResult, oracle.get(`complex_real_${dtype}`)!);
    });

    it(`imag ${dtype}`, () => {
      const a = array(smallData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `imag ${dtype}`);
      const jsResult = np.imag(a);
      expectMatchPre(jsResult, oracle.get(`complex_imag_${dtype}`)!);
    });

    it(`conj ${dtype}`, () => {
      const a = array(smallData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `conj ${dtype}`);
      const jsResult = np.conj(a);
      expectMatchPre(jsResult, oracle.get(`complex_conj_${dtype}`)!);
    });

    it(`angle ${dtype}`, () => {
      const a = array(smallData(dtype).js as never, dtype);
      expectComplexFixture(a, dtype, `angle ${dtype}`);
      const jsResult = np.angle(a);
      expectMatchPre(jsResult, oracle.get(`complex_angle_${dtype}`)!, { rtol: 1e-4 });
    });
  }
});

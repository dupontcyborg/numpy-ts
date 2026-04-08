/**
 * DType Sweep: Statistics, misc, and complex functions.
 * All tested across ALL dtypes, value-producing tests validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, expect, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  expectBothReject,
  expectMatchPre,
  arraysClose,
} from './_helpers';
import type { NumPyResult } from '../numpy-oracle';

const { array } = np;

// Pre-computed oracle results — filled in beforeAll
let oracle: Map<string, NumPyResult & { error?: string }>;

beforeAll(() => {
  if (!checkNumPyAvailable()) throw new Error('Python NumPy not available');

  const snippets: Record<string, string> = {};

  for (const dtype of ALL_DTYPES) {
    // Statistics
    const histData = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
    snippets[`stats_histogram_${dtype}`] = `
h, e = np.histogram(np.array(${JSON.stringify(histData)}, dtype=${npDtype(dtype)}))
_result_orig = h
result = _result_orig.astype(np.float64)`;

    const corrData =
      dtype === 'bool'
        ? [
            [1, 0, 1],
            [0, 1, 0],
          ]
        : [
            [1, 2, 3],
            [4, 5, 6],
          ];
    snippets[`stats_corrcoef_${dtype}`] =
      `_result_orig = np.corrcoef(np.array(${JSON.stringify(corrData)}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    snippets[`stats_cov_${dtype}`] =
      `_result_orig = np.cov(np.array(${JSON.stringify(corrData)}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    snippets[`stats_bincount_${dtype}`] =
      `result = np.bincount(np.array(${JSON.stringify(dtype === 'bool' ? [1, 0, 1, 0] : [0, 1, 1, 2, 3, 3, 3])}, dtype=${npDtype(dtype)})).astype(np.float64)`;

    const digitizeData = dtype === 'bool' ? [0, 1, 0, 1] : [1, 3, 5, 7];
    const digitizeBins = dtype === 'bool' ? [0, 1] : [2, 4, 6];
    snippets[`stats_digitize_${dtype}`] =
      `_result_orig = np.digitize(np.array(${JSON.stringify(digitizeData)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(digitizeBins)}, dtype=${npDtype(dtype)}))
result = _result_orig`;

    const corrD1 = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    const corrD2 = dtype === 'bool' ? [0, 1] : [0, 1];
    snippets[`stats_correlate_${dtype}`] =
      `_result_orig = np.correlate(np.array(${JSON.stringify(corrD1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(corrD2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`stats_convolve_${dtype}`] =
      `_result_orig = np.convolve(np.array(${JSON.stringify(corrD1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(corrD2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    const trapData = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
    snippets[`stats_trapezoid_${dtype}`] =
      `result = float(np.trapezoid(np.array(${JSON.stringify(trapData)}, dtype=${npDtype(dtype)})))`;

    const diffData = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 4, 7, 11];
    snippets[`stats_diff_${dtype}`] =
      `_result_orig = np.diff(np.array(${JSON.stringify(diffData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`stats_gradient_${dtype}`] =
      `_result_orig = np.gradient(np.array(${JSON.stringify(diffData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    const interpXp = dtype === 'bool' ? [0, 1] : [1, 2, 3, 4];
    const interpFp = dtype === 'bool' ? [0, 1] : [10, 20, 30, 40];
    const interpX = dtype === 'bool' ? [0, 1] : [1.5, 2.5, 3.5];
    snippets[`stats_interp_${dtype}`] = `
x = np.array(${JSON.stringify(interpX)}, dtype=${npDtype(dtype)})
xp = np.array(${JSON.stringify(interpXp)}, dtype=${npDtype(dtype)})
fp = np.array(${JSON.stringify(interpFp)}, dtype=${npDtype(dtype)})
_result_orig = np.interp(x, xp, fp)
result = _result_orig`;

    // Misc
    const clipData = dtype === 'bool' ? [1, 0, 1] : [1, 5, 10];
    const lo = dtype === 'bool' ? 0 : 2;
    const hi = dtype === 'bool' ? 1 : 8;
    snippets[`misc_clip_${dtype}`] =
      `_result_orig = np.clip(np.array(${JSON.stringify(clipData)}, dtype=${npDtype(dtype)}), ${lo}, ${hi})
result = _result_orig.astype(np.float64)`;

    const fmodD1 = dtype === 'bool' ? [1, 1, 0] : [6, 7, 8, 9];
    const fmodD2 = dtype === 'bool' ? [1, 1, 1] : [4, 3, 5, 2];
    snippets[`misc_fmod_${dtype}`] =
      `_result_orig = np.fmod(np.array(${JSON.stringify(fmodD1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(fmodD2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    const nanToNumData = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    snippets[`misc_nan_to_num_${dtype}`] =
      `_result_orig = np.nan_to_num(np.array(${JSON.stringify(nanToNumData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    const ldexpData = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    snippets[`misc_ldexp_${dtype}`] =
      `_result_orig = np.ldexp(np.array(${JSON.stringify(ldexpData)}, dtype=${npDtype(dtype)}), np.array([1,2,3], dtype=np.int32))
result = _result_orig.astype(np.float64)`;

    const frexpData = dtype === 'bool' ? [1, 0, 1] : [1, 2, 4];
    snippets[`misc_frexp_${dtype}`] = `
m, e = np.frexp(np.array(${JSON.stringify(frexpData)}, dtype=${npDtype(dtype)}))
_result_orig = m
result = _result_orig.astype(np.float64)`;

    if (!isComplex(dtype)) {
      const aeData = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      snippets[`misc_array_equiv_${dtype}`] =
        `result = bool(np.array_equiv(np.array(${JSON.stringify(aeData)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(aeData)}, dtype=${npDtype(dtype)})))`;
    }

    // Complex functions
    const complexData = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
    snippets[`complex_real_${dtype}`] =
      `_result_orig = np.real(np.array(${JSON.stringify(complexData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`complex_imag_${dtype}`] =
      `_result_orig = np.imag(np.array(${JSON.stringify(complexData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`complex_conj_${dtype}`] =
      `_result_orig = np.conj(np.array(${JSON.stringify(complexData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;

    snippets[`complex_angle_${dtype}`] =
      `_result_orig = np.angle(np.array(${JSON.stringify(complexData)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
  }

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: Statistics', () => {
  for (const dtype of ALL_DTYPES) {
    it(`histogram ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 3, 4, 5];
      const pyCode = `
h, e = np.histogram(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
_result_orig = h
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'histogram requires real-valued input',
          () => np.histogram(array(data, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const [hist] = np.histogram(array(data, dtype)) as [any, any];
      expectMatchPre(hist, oracle.get(`stats_histogram_${dtype}`)!);
    });

    it(`corrcoef ${dtype}`, () => {
      const data =
        dtype === 'bool'
          ? [
              [1, 0, 1],
              [0, 1, 0],
            ]
          : [
              [1, 2, 3],
              [4, 5, 6],
            ];
      const jsResult = np.corrcoef(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`stats_corrcoef_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`cov ${dtype}`, () => {
      const data =
        dtype === 'bool'
          ? [
              [1, 0, 1],
              [0, 1, 0],
            ]
          : [
              [1, 2, 3],
              [4, 5, 6],
            ];
      const jsResult = np.cov(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`stats_cov_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`bincount ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0] : [0, 1, 1, 2, 3, 3, 3];
      let pyErr = false;
      let jsErr = false;
      const py = oracle.get(`stats_bincount_${dtype}`)!;
      if (py.error) pyErr = true;
      let jsResult: any;
      try {
        jsResult = np.bincount(array(data, dtype));
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
      const data = dtype === 'bool' ? [0, 1, 0, 1] : [1, 3, 5, 7];
      const bins = dtype === 'bool' ? [0, 1] : [2, 4, 6];
      const pyCode = `_result_orig = np.digitize(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(bins)}, dtype=${npDtype(dtype)}))
result = _result_orig`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'digitize requires real-valued input',
          () => np.digitize(array(data, dtype), array(bins, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.digitize(array(data, dtype), array(bins, dtype));
      expectMatchPre(jsResult, oracle.get(`stats_digitize_${dtype}`)!);
    });

    it(`correlate ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const d2 = dtype === 'bool' ? [0, 1] : [0, 1];
      const jsResult = np.correlate(array(d1, dtype), array(d2, dtype));
      expectMatchPre(jsResult, oracle.get(`stats_correlate_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`convolve ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const d2 = dtype === 'bool' ? [0, 1] : [0, 1];
      const jsResult = np.convolve(array(d1, dtype), array(d2, dtype));
      expectMatchPre(jsResult, oracle.get(`stats_convolve_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`trapezoid ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
      const pyCode = `result = float(np.trapezoid(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})))`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'trapezoid is not defined for complex numbers',
          () => np.trapezoid(array(data, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.trapezoid(array(data, dtype));
      const py = oracle.get(`stats_trapezoid_${dtype}`)!;
      expect(Number(jsResult)).toBeCloseTo(Number(py.value), 4);
    });

    it(`diff ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 4, 7, 11];
      const jsResult = np.diff(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`stats_diff_${dtype}`)!);
    });

    it(`gradient ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1, 0, 1] : [1, 2, 4, 7, 11];
      const pyCode = `_result_orig = np.gradient(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
      if (dtype === 'bool') {
        const _r = expectBothReject(
          'gradient uses subtract internally, not supported for bool',
          () => np.gradient(array(data, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.gradient(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`stats_gradient_${dtype}`)!, { rtol: 1e-4 });
    });

    it(`interp ${dtype}`, () => {
      const xp = dtype === 'bool' ? [0, 1] : [1, 2, 3, 4];
      const fp = dtype === 'bool' ? [0, 1] : [10, 20, 30, 40];
      const x = dtype === 'bool' ? [0, 1] : [1.5, 2.5, 3.5];
      const pyCode = `
x = np.array(${JSON.stringify(x)}, dtype=${npDtype(dtype)})
xp = np.array(${JSON.stringify(xp)}, dtype=${npDtype(dtype)})
fp = np.array(${JSON.stringify(fp)}, dtype=${npDtype(dtype)})
_result_orig = np.interp(x, xp, fp)
result = _result_orig`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'interp is not defined for complex numbers',
          () => np.interp(array(x, dtype), array(xp, dtype), array(fp, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.interp(array(x, dtype), array(xp, dtype), array(fp, dtype));
      expectMatchPre(jsResult, oracle.get(`stats_interp_${dtype}`)!, { rtol: 1e-4 });
    });
  }
});

describe('DType Sweep: Misc', () => {
  for (const dtype of ALL_DTYPES) {
    it(`clip ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 5, 10];
      const lo = dtype === 'bool' ? 0 : 2;
      const hi = dtype === 'bool' ? 1 : 8;
      const pyCode = `_result_orig = np.clip(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}), ${lo}, ${hi})
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'clip is not defined for complex numbers (requires ordering)',
          () => np.clip(array(data, dtype), lo, hi),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.clip(array(data, dtype), lo, hi);
      expectMatchPre(jsResult, oracle.get(`misc_clip_${dtype}`)!);
    });

    it(`fmod ${dtype}`, () => {
      const d1 = dtype === 'bool' ? [1, 1, 0] : [6, 7, 8, 9];
      const d2 = dtype === 'bool' ? [1, 1, 1] : [4, 3, 5, 2];
      const pyCode = `_result_orig = np.fmod(np.array(${JSON.stringify(d1)}, dtype=${npDtype(dtype)}), np.array(${JSON.stringify(d2)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'fmod is not defined for complex numbers',
          () => np.fmod(array(d1, dtype), array(d2, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.fmod(array(d1, dtype), array(d2, dtype));
      expectMatchPre(jsResult, oracle.get(`misc_fmod_${dtype}`)!);
    });

    it(`nan_to_num ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.nan_to_num(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`misc_nan_to_num_${dtype}`)!);
    });

    it(`ldexp ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const pyCode = `_result_orig = np.ldexp(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}), np.array([1,2,3], dtype=np.int32))
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'ldexp is only defined for real floating-point types',
          () => np.ldexp(array(data, dtype), array([1, 2, 3], 'int32')),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const jsResult = np.ldexp(array(data, dtype), array([1, 2, 3], 'int32'));
      expectMatchPre(jsResult, oracle.get(`misc_ldexp_${dtype}`)!);
    });

    it(`frexp ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 4];
      const pyCode = `
m, e = np.frexp(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
_result_orig = m
result = _result_orig.astype(np.float64)`;
      if (isComplex(dtype)) {
        const _r = expectBothReject(
          'frexp is only defined for real floating-point types',
          () => np.frexp(array(data, dtype)),
          pyCode
        );
        if (_r === 'both-reject') return;
      }
      const [m] = np.frexp(array(data, dtype)) as [any, any];
      expectMatchPre(m, oracle.get(`misc_frexp_${dtype}`)!, { rtol: 1e-4 });
    });

    if (!isComplex(dtype)) {
      it(`array_equiv ${dtype}`, () => {
        const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
        const a = array(data, dtype);
        const b = array(data, dtype);
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
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.real(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`complex_real_${dtype}`)!);
    });

    it(`imag ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.imag(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`complex_imag_${dtype}`)!);
    });

    it(`conj ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.conj(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`complex_conj_${dtype}`)!);
    });

    it(`angle ${dtype}`, () => {
      const data = dtype === 'bool' ? [1, 0, 1] : [1, 2, 3];
      const jsResult = np.angle(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`complex_angle_${dtype}`)!, { rtol: 1e-4 });
    });
  }
});

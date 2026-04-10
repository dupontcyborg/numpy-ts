/**
 * DType Sweep: 1D FFT functions + utilities.
 * Tests across ALL dtypes, validated against NumPy.
 * Uses batched oracle — all Python computations run in a single subprocess.
 */
import { describe, it, beforeAll } from 'vitest';
import * as np from '../../../src';
import {
  ALL_DTYPES,
  runNumPyBatch,
  checkNumPyAvailable,
  npDtype,
  isComplex,
  pyArrayCast,
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
    const data = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
    // FFT results are complex → always cast to np.complex128 for value comparison
    // fftshift/ifftshift preserve dtype → use pyArrayCast
    const ac = pyArrayCast(dtype);

    snippets[`fft_${dtype}`] = `
_result_orig = np.fft.fft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`ifft_${dtype}`] = `
_result_orig = np.fft.ifft(np.fft.fft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})))
result = _result_orig.astype(np.complex128)`;

    snippets[`rfft_${dtype}`] = `
_result_orig = np.fft.rfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`irfft_${dtype}`] =
      `result = np.fft.irfft(np.fft.rfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})))`;

    snippets[`hfft_${dtype}`] =
      `result = np.fft.hfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))`;

    snippets[`ihfft_${dtype}`] = `
_result_orig = np.fft.ihfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;

    snippets[`fftshift_${dtype}`] = `
_result_orig = np.fft.fftshift(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;

    snippets[`ifftshift_${dtype}`] = `
_result_orig = np.fft.ifftshift(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(${ac})`;
  }

  // fftfreq/rfftfreq are dtype-independent, just need one each
  snippets['fftfreq'] = `result = np.fft.fftfreq(4)`;
  snippets['rfftfreq'] = `result = np.fft.rfftfreq(4)`;

  oracle = runNumPyBatch(snippets);
});

describe('DType Sweep: FFT 1D', () => {
  for (const dtype of ALL_DTYPES) {
    const data = dtype === 'bool' ? [1, 0, 1, 0] : [1, 2, 3, 4];
    const tol = dtype === 'float32' || dtype === 'complex64' ? 1e-2 : 1e-4;

    it(`fft.fft ${dtype}`, () => {
      const jsResult = np.fft.fft(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`fft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.ifft ${dtype}`, () => {
      const fftResult = np.fft.fft(array(data, dtype));
      const jsResult = np.fft.ifft(fftResult);
      expectMatchPre(jsResult, oracle.get(`ifft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.rfft ${dtype}`, () => {
      const pyCode = `
_result_orig = np.fft.rfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'rfft expects real-valued input, not complex',
            () => np.fft.rfft(array(data, dtype)),
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const jsResult = np.fft.rfft(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`rfft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.irfft ${dtype}`, () => {
      const pyCode = `result = np.fft.irfft(np.fft.rfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)})))`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'rfft(complex) not supported, so irfft round-trip fails',
            () => {
              const r = np.fft.rfft(array(data, dtype));
              np.fft.irfft(r);
            },
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const rfftResult = np.fft.rfft(array(data, dtype));
      const jsResult = np.fft.irfft(rfftResult);
      expectMatchPre(jsResult, oracle.get(`irfft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.hfft ${dtype}`, () => {
      const jsResult = np.fft.hfft(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`hfft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.ihfft ${dtype}`, () => {
      const pyCode = `
_result_orig = np.fft.ihfft(np.array(${JSON.stringify(data)}, dtype=${npDtype(dtype)}))
result = _result_orig.astype(np.complex128)`;
      if (isComplex(dtype)) {
        {
          const _r = expectBothReject(
            'ihfft expects real-valued input',
            () => np.fft.ihfft(array(data, dtype)),
            pyCode
          );
          if (_r === 'both-reject') return;
        }
      }
      const jsResult = np.fft.ihfft(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`ihfft_${dtype}`)!, { rtol: tol });
    });

    it(`fft.fftfreq ${dtype}`, () => {
      const jsResult = np.fft.fftfreq(4);
      expectMatchPre(jsResult, oracle.get('fftfreq')!, { rtol: 1e-10 });
    });

    it(`fft.rfftfreq ${dtype}`, () => {
      const jsResult = np.fft.rfftfreq(4);
      expectMatchPre(jsResult, oracle.get('rfftfreq')!, { rtol: 1e-10 });
    });

    it(`fft.fftshift ${dtype}`, () => {
      const jsResult = np.fft.fftshift(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`fftshift_${dtype}`)!);
    });

    it(`fft.ifftshift ${dtype}`, () => {
      const jsResult = np.fft.ifftshift(array(data, dtype));
      expectMatchPre(jsResult, oracle.get(`ifftshift_${dtype}`)!);
    });
  }
});
